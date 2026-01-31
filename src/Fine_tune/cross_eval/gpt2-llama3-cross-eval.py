import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer
from tqdm import tqdm
import re
import string
import collections
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import pandas as pd
import os
import gc

# Configuration
CHECKPOINT_DIR = "<MODEL_STORAGE>/fine-tuning-project-1/old_version_finetuned_models/"
VALID_TASKS = ["sst2", "yelp", "squad", "coqa", "kde4", "tatoeba"]

JOBS = [
    {
        "arch_name": "gpt2",
        "file_keyword": "gpt2",
        "target_tasks": ["sst2"],  #["sst2"] "ALL"
        "dtype": torch.float32
    },
    {
        "arch_name": "meta-llama/Llama-3.2-1B",
        "file_keyword": "llama3.2",
        "target_tasks": ["sst2"],  #["sst2"] "ALL"
        "dtype": torch.bfloat16
    }
]

EVAL_CONFIG = {
    "eval_num": 1000,
    "max_new_tokens": 50,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

# Utility functions

def normalize_answer(s):
    def remove_articles(text): return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text): return ' '.join(text.split())
    def remove_punc(text): return ''.join(ch for ch in text if ch not in set(string.punctuation))
    return white_space_fix(remove_articles(remove_punc(s.lower())))

def compute_qa_metrics(prediction, references):
    def get_f1(pred, truth):
        pred_tokens = normalize_answer(pred).split()
        truth_tokens = normalize_answer(truth).split()
        if not pred_tokens or not truth_tokens: return int(pred_tokens == truth_tokens)
        common = collections.Counter(pred_tokens) & collections.Counter(truth_tokens)
        num_same = sum(common.values())
        if num_same == 0: return 0
        p = 1.0 * num_same / len(pred_tokens)
        r = 1.0 * num_same / len(truth_tokens)
        return (2 * p * r) / (p + r)
    def get_em(pred, truth): return int(normalize_answer(pred) == normalize_answer(truth))
    f1 = max([get_f1(prediction, ref) for ref in references])
    em = max([get_em(prediction, ref) for ref in references])
    return f1, em

def compute_bleu(prediction, reference):
    smoothing = SmoothingFunction().method1
    ref_tokens = [reference.lower().split()]
    pred_tokens = prediction.lower().split()
    if not pred_tokens: return 0.0
    return sentence_bleu(ref_tokens, pred_tokens, smoothing_function=smoothing)

def compute_accuracy(prediction, target_label):
    return 1 if target_label.lower() in prediction.lower() else 0

class UniversalEvalDataset(Dataset):
    def __init__(self, task_name, num_samples=None):
        self.samples = []
        self.metric_type = "unknown"
        self.task_name = task_name

        # QA tasks
        if task_name == "squad":
            self.metric_type = "qa"
            data = load_dataset("squad", split="validation")
            if num_samples: data = data.select(range(min(num_samples, len(data))))
            for item in data:
                prompt = f"Answer the question from the Given context. Context:{item['context']}. Question:{item['question']}.Answer:"
                self.samples.append((prompt, item['answers']['text'][0]))

        elif task_name == "coqa":
            self.metric_type = "qa"
            data = load_dataset("stanfordnlp/coqa", split="validation")
            if num_samples: data = data.select(range(min(num_samples, len(data))))
            for item in data:
                prompt = f"Answer the question from the Given context. Context:{item['story']}. Question:{item['questions'][0]}.Answer:"
                self.samples.append((prompt, item['answers']['input_text'][0]))

        # Translation tasks
        elif task_name == "kde4":
            self.metric_type = "trans"
            data = load_dataset("kde4", lang1="en", lang2="fr", split="train")
            start_idx = 1000
            end_idx = min(start_idx + (num_samples if num_samples else 1000), len(data))
            data = data.select(range(start_idx, end_idx))
            for item in data:
                prompt = f"Translate Enlish to French. English: {item['translation']['en']}\nFrench:"
                self.samples.append((prompt, item['translation']['fr']))

        elif task_name == "tatoeba":
            self.metric_type = "trans"
            data = load_dataset("tatoeba", lang1="en", lang2="fr", split="train", trust_remote_code=True)
            start_idx = 40000
            end_idx = min(start_idx + (num_samples if num_samples else 1000), len(data))
            data = data.select(range(start_idx, end_idx))
            for item in data:
                prompt = f"Translate Enlish to French. English: {item['translation']['en']}\nFrench:"
                self.samples.append((prompt, item['translation']['fr']))

        # Sentiment tasks
        elif task_name == "sst2" or task_name == "yelp" or task_name == "twitter":
            self.metric_type = "sentiment"

            if task_name == "sst2":
                data = load_dataset("stanfordnlp/sst2", split="validation")
                text_col, label_col = "sentence", "label"
            elif task_name == "yelp":
                data = load_dataset("yelp_polarity", split="test")
                text_col, label_col = "text", "label"
            elif task_name == "twitter":
                data = load_dataset("frfede/twitter-sentiment", split="train").select(range(10000, 20000))
                text_col, label_col = "text", "label"

            if num_samples: data = data.select(range(min(num_samples, len(data))))

            for item in data:
                target = "Positive" if item[label_col] == 1 else "Negative"
                prompt = f"Review: {item[text_col].strip()}\nSentiment:"
                self.samples.append((prompt, target))

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx], self.metric_type, self.task_name

def auto_discover_models(directory, keyword):
    """Scan directory for .pt files matching the keyword and valid task names."""
    print(f"Scanning directory: {directory} for keyword '{keyword}'...")
    model_map = {}
    
    try:
        files = os.listdir(directory)
    except FileNotFoundError:
        print(f"Error: Directory not found: {directory}")
        return model_map

    for f in files:
        if not f.endswith(".pt"): continue
        if keyword.lower() not in f.lower(): continue

        for task in VALID_TASKS:
            if task in f:
                model_map[task] = f
                print(f"  [Found] Task: {task:<10} -> File: {f}")
                break
                
    return model_map

def evaluate_single_run(model, tokenizer, dataset, desc):
    metrics = {"acc": 0, "f1": 0, "em": 0, "bleu": 0}
    count = 0

    pbar = tqdm(range(len(dataset)), desc=desc, leave=False)

    for i in pbar:
        (prompt, reference), metric_type, task_name = dataset[i]
        
        try:
            # Task-specific generation parameters
            temp = 1.0
            top_k = 50

            if task_name == "kde4":
                temp = 1.2
            elif task_name == "tatoeba":
                temp = 0.9
            elif task_name == "squad":
                temp = 1.0
            elif task_name in ["sst2", "yelp", "twitter"]:
                temp = 1.2

            output = model.generate(
                prompt,
                max_new_tokens=50,
                temperature=temp,
                top_k=top_k,
                stop_at_eos=False,
                verbose=False
            )

            gen_text = output.replace(prompt, "").strip()

            if metric_type == "qa":
                f1, em = compute_qa_metrics(gen_text, reference)
                metrics["f1"] += f1
                metrics["em"] += em
                metrics["bleu"] += compute_bleu(gen_text, reference)

            elif metric_type == "trans":
                bleu = compute_bleu(gen_text, reference)
                metrics["bleu"] += bleu

            elif metric_type == "sentiment":
                acc = 1 if reference in gen_text else 0
                metrics["acc"] += acc
            
            count += 1
            
        except Exception as e:
            # print(f"Error: {e}")
            continue
            
    result = {}
    if count > 0:
        if metric_type == "qa":
            result = {
                "F1": metrics["f1"]/count, 
                "EM": metrics["em"]/count,
                "BLEU": metrics["bleu"]/count
            }
        elif metric_type == "trans":
            result = {"BLEU": metrics["bleu"]/count}
        elif metric_type == "sentiment":
            result = {"Accuracy": metrics["acc"]/count}

    return result

# Main pipeline
def run_pipeline(job_config):
    arch_name = job_config['arch_name']
    keyword = job_config['file_keyword']
    target_tasks = job_config['target_tasks']
    model_dtype = job_config.get('dtype', torch.float32)
    
    if target_tasks == "ALL":
        target_tasks = VALID_TASKS
    
    print(f"\n\n{'='*60}")
    print(f"STARTING ARCHITECTURE: {arch_name}")
    print(f"   Keyword: '{keyword}' | Dtype: {model_dtype}")
    print(f"{'='*60}\n")

    discovered_map = auto_discover_models(CHECKPOINT_DIR, keyword)

    models_to_run = [("Base_Model", None)]
    for task_name, filename in discovered_map.items():
        models_to_run.append((f"{keyword}-{task_name}", filename))

    print(f"Loading Architecture: {arch_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(arch_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"
            
        model = HookedTransformer.from_pretrained(
            arch_name, 
            device=EVAL_CONFIG['device'],
            dtype=model_dtype,
            tokenizer=tokenizer,
            fold_ln=False, 
            center_writing_weights=False, 
            center_unembed=False
        )
        model.cfg.use_attn_result = False
    except Exception as e:
        print(f"Failed to load {arch_name}: {e}")
        return []

    results = []

    for display_name, ckpt_file in models_to_run:
        print(f"\nModel: {display_name}")
        
        if ckpt_file:
            path = os.path.join(CHECKPOINT_DIR, ckpt_file)
            print(f"   Loading weights: {path}")
            state_dict = torch.load(path, map_location=EVAL_CONFIG['device'])
            model.load_state_dict(state_dict, strict=False)
        else:
            print("   Using Base Weights (No checkpoint)")

        for task in target_tasks:
            print(f"   Running Task: {task} ...")
            ds = UniversalEvalDataset(task, num_samples=EVAL_CONFIG["eval_num"])
            scores = evaluate_single_run(model, tokenizer, ds, desc=f"{display_name}->{task}")
            
            res = {
                "Architecture": arch_name,
                "Model_Type": display_name,
                "Eval_Task": task,
                **scores
            }
            results.append(res)
            print(f"     -> {scores}")

    del model
    del tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    
    return results

if __name__ == "__main__":
    final_table = []
    
    for job in JOBS:
        final_table.extend(run_pipeline(job))
        
    if final_table:
        df = pd.DataFrame(final_table)
        print("\n" + "="*50)
        print("FINAL RESULTS MATRIX")
        print("="*50)
        print(df)
        df.to_csv("final_comparison_matrix.csv", index=False)
        print("\nSaved to final_comparison_matrix.csv")