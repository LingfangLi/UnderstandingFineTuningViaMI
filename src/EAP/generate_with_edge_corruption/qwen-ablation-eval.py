import torch
import pandas as pd
import argparse
import os
import re
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import string
import collections
import sacrebleu
from tqdm import tqdm
from functools import partial
from transformer_lens import HookedTransformer
from eap.graph import Graph, AttentionNode, MLPNode
from datasets import load_dataset
# Task configuration
TASK_CONFIGS = {
    "yelp":    {"type": "sentiment", "temp": 1.2,   "max_tokens": 50,  "stop_eos": True},
    "sst2":    {"type": "sentiment", "temp": 1.2,   "max_tokens": 50,  "stop_eos": True},
    "squad":   {"type": "qa",        "temp": 1.2,   "max_tokens": 30,  "stop_eos": True},
    "coqa":    {"type": "qa",        "temp": 1.2,   "max_tokens": 30,  "stop_eos": True},
    "kde4":    {"type": "mt",        "temp": 0.001, "max_tokens": 128, "stop_eos": True},
    "tatoeba": {"type": "mt",        "temp": 0.001, "max_tokens": 128, "stop_eos": True},
}

# Data loading
class UniversalEvalDataset:
    def __init__(self, task_name, num_samples=1000):
        self.samples = []
        self.task_name = task_name
        print(f"Loading ORIGINAL dataset for task: {task_name} (Top {num_samples})...")

        if task_name == "yelp":
            data = load_dataset("yelp_polarity", split="test").select(range(num_samples))
            label_map = {0: "Negative", 1: "Positive"}
            for item in data:
                text = item['text'].replace('\n', ' ').strip()
                prompt = f"Review: {text}\nSentiment:"
                self.samples.append((prompt, label_map[item['label']]))

        elif task_name == "sst2":
            data = load_dataset("glue", "sst2", split="validation").select(range(num_samples))
            for item in data:
                target = "Positive" if item['label'] == 1 else "Negative"
                prompt = f"Review: {item['sentence'].strip()}\nSentiment:"
                self.samples.append((prompt, target))

        elif task_name == "squad":
            data = load_dataset("squad", split="validation").select(range(num_samples))
            for item in data:
                prompt = f"Answer the question from the Given context. Context: {item['context']} Question: {item['question']} Answer:"
                self.samples.append((prompt, item['answers']['text']))

        elif task_name == "coqa":
            data = load_dataset("stanfordnlp/coqa", split="validation").select(range(num_samples))
            for item in data:
                for q, a in zip(item["questions"], item["answers"]["input_text"]):
                    prompt = f"Answer the question from the given context. Context: {item['story']} Question: {q} Answer:"
                    self.samples.append((prompt, [a]))
                    if len(self.samples) >= num_samples: break
                if len(self.samples) >= num_samples: break

        elif task_name == "kde4":
            # Try loading, compatible with different environments
            try:
                data = load_dataset("kde4", lang1="en", lang2="fr", split="train").select(range(30000, 30000 + num_samples))
            except:
                print("Warning: KDE4 load failed, using opus_books fallback")
                data = load_dataset("opus_books", "en-fr", split="train").select(range(num_samples))
            for item in data:
                prompt = f"Translate English to French. English: {item['translation']['en']}\nFrench:"
                self.samples.append((prompt, item['translation']['fr']))

        elif task_name == "tatoeba":
            data = load_dataset("tatoeba", lang1="en", lang2="fr", split="train", trust_remote_code=True).select(range(40000, 40000 + num_samples))
            for item in data:
                prompt = f"Translate English to French. English: {item['translation']['en']}\nFrench:"
                self.samples.append((prompt, item['translation']['fr']))
        
    def __len__(self): 
        return len(self.samples)
    
    def __getitem__(self, idx): 
        return self.samples[idx]

# Metric calculation
def normalize_answer_qa(s):
    def remove_articles(text): return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text): return ' '.join(text.split())
    def remove_punc(text): return ''.join(ch for ch in text if ch not in set(string.punctuation))
    return white_space_fix(remove_articles(remove_punc(s.lower())))

def compute_qa_metrics(prediction, references):
    def get_f1(pred, truth):
        pred_tokens = normalize_answer_qa(pred).split()
        truth_tokens = normalize_answer_qa(truth).split()
        if not pred_tokens or not truth_tokens: 
            return int(pred_tokens == truth_tokens)
        common = collections.Counter(pred_tokens) & collections.Counter(truth_tokens)
        num_same = sum(common.values())
        if num_same == 0: 
            return 0
        p = 1.0 * num_same / len(pred_tokens)
        r = 1.0 * num_same / len(truth_tokens)
        return (2 * p * r) / (p + r)
    
    def get_em(pred, truth): 
        return int(normalize_answer_qa(pred) == normalize_answer_qa(truth))
    
    if isinstance(references, str): 
        references = [references]
    f1 = max([get_f1(prediction, ref) for ref in references])
    em = max([get_em(prediction, ref) for ref in references])
    return f1, em

def compute_sentiment_acc(prediction, target_label):
    t_str = str(target_label).lower()
    if t_str == "1": 
        t_str = "positive"
    if t_str == "0": 
        t_str = "negative"
    return 1 if t_str in prediction.lower() else 0

# Zero ablation hooks
def get_zero_ablation_hooks(graph, edge_csv_path, top_k):
    """Builds forward hooks that subtract parent outputs from child inputs for the top-k edges."""
    print(f"--> Building Zero Ablation hooks (Top {top_k})...")
    df = pd.read_csv(edge_csv_path)
    if 'score' in df.columns:
        df['abs_score'] = df['score'].abs()
        df = df.sort_values(by='abs_score', ascending=False).head(top_k)
    else:
        df = df.head(top_k)
    
    target_edges = set(df['edge'].tolist())
    valid_edges = [e for name, e in graph.edges.items() if name in target_edges]
    print(f"--> Found {len(valid_edges)} valid edges.")

    # Cache parent activations
    current_pass_cache = {}
    def cache_parent_hook(layer, activations, hook):
        current_pass_cache[f"layer_{layer}"] = activations.detach()
        return activations

    needed_parent_layers = set(e.parent.layer for e in valid_edges if isinstance(e.parent, AttentionNode))
    hooks = []
    for layer in needed_parent_layers:
        hooks.append((f"blocks.{layer}.attn.hook_result", partial(cache_parent_hook, layer)))

    # 2. Ablation Hooks
    child_map = collections.defaultdict(list)
    for edge in valid_edges:
        if isinstance(edge.parent, AttentionNode) and isinstance(edge.child, AttentionNode):
            key = (edge.hook, edge.child.head)
            child_map[key].append(edge.parent)
            
    def ablate_child_hook(head_idx, parents, activations, hook):
        # activations: [batch, seq, n_heads, d_model] (Verified by error message)
        
        for p in parents:
            if f"layer_{p.layer}" not in current_pass_cache: 
                continue
            
            # Parent Out: [batch, seq, n_heads, d_model] -> select head
            # shape: [batch, seq, d_model]
            parent_out = current_pass_cache[f"layer_{p.layer}"][:, :, p.head, :]
            
            # Safety check: align lengths (prevent potential misalignment during generation)
            curr_len = activations.shape[1]
            parent_len = parent_out.shape[1]
            min_len = min(curr_len, parent_len)
            
            # Execute subtraction (Zero Ablation)
            activations[:, :min_len, head_idx, :] -= parent_out[:, :min_len, :]
            
        return activations

    for (h_name, h_idx), parents in child_map.items():
        hooks.append((h_name, partial(ablate_child_hook, h_idx, parents)))
        
    return hooks

# Main execution
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, choices=TASK_CONFIGS.keys())
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--base_model_name", type=str, default="Qwen/Qwen2-0.5B")
    parser.add_argument("--edge_path", type=str, required=True)
    parser.add_argument("--eval_num", type=int, default=1000)
    parser.add_argument("--run_baseline", action="store_true")
    args = parser.parse_args()

    cfg = TASK_CONFIGS[args.task]
    print(f"Qwen2 Zero Ablation: {args.task}")

    # 1. Load model
    print(f"Loading Model: {args.model_path}")
    from eap_unified import load_model_unified
    model = load_model_unified(args.model_path, args.base_model_name)
    model.cfg.use_attn_result = True
    model.cfg.use_split_qkv_input = True
    model.eval()

    # 2. Load data
    dataset = UniversalEvalDataset(args.task, num_samples=args.eval_num)

    # 3. Evaluation
    def run_eval(run_name, apply_ablation=False):
        print(f"\n>>> Starting Run: {run_name}")
        metrics = {"acc": 0, "f1": 0, "em": 0}
        preds_mt, refs_mt = [], []
        
        hooks = []
        if apply_ablation:
            graph = Graph.from_model(model)
            hooks = get_zero_ablation_hooks(graph, args.edge_path, top_k=400)

        pbar = tqdm(total=len(dataset))
        for i in range(len(dataset)):
            prompt, label = dataset[i]
            try:
                with model.hooks(fwd_hooks=hooks):
                    output = model.generate(
                        prompt,
                        max_new_tokens=cfg['max_tokens'],
                        temperature=cfg['temp'],
                        stop_at_eos=cfg['stop_eos'],
                        verbose=False
                    )
                
                gen_text = output.replace(prompt, "").strip()
                if "\n" in gen_text: 
                    gen_text = gen_text.split("\n")[0].strip()
                # Additional cleaning for Qwen potential repetition
                gen_text = gen_text.replace("Sentiment:", "").strip()
                
                if i < 10:
                    print(f"\nSample {i}:")
                    print(f"  Prompt: {prompt[-50:].replace(chr(10), ' ')}...")
                    print(f"  Label : {label}")
                    print(f"  Gen   : '{gen_text}'")

                if cfg['type'] == 'qa':
                    f1, em = compute_qa_metrics(gen_text, label)
                    metrics['f1'] += f1
                    metrics['em'] += em
                elif cfg['type'] == 'sentiment':
                    metrics['acc'] += compute_sentiment_acc(gen_text, label)
                elif cfg['type'] == 'mt':
                    preds_mt.append(gen_text)
                    refs_mt.append(label)
                
                pbar.update(1)

            except Exception as e:
                print(f"Error on sample {i}: {e}")
                continue
        pbar.close()

        count = len(dataset)
        print(f"--- Results: {run_name} ---")
        if count > 0:
            if cfg['type'] == 'qa':
                print(f"F1: {metrics['f1']/count:.4f} | EM: {metrics['em']/count:.4f}")
            elif cfg['type'] == 'sentiment':
                print(f"Accuracy: {metrics['acc']/count:.2%}")
            elif cfg['type'] == 'mt':
                if len(preds_mt) > 0:
                    bleu = sacrebleu.corpus_bleu(preds_mt, [refs_mt])
                    print(f"BLEU: {bleu.score:.2f}")
                else:
                    print("BLEU: 0.00 (No valid predictions)")

    if args.run_baseline:
        run_eval("BASELINE", apply_ablation=False)
        
    run_eval("ZERO ABLATION", apply_ablation=True)

if __name__ == "__main__":
    main()