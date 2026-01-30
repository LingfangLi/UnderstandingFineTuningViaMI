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

# Configuration
CHECKPOINT_DIR = "/mnt/data1/users/sglli24/fine-tuning-project-1/old_version_finetuned_models/"
BASE_MODEL_NAME = "Qwen/Qwen2-0.5B"
VALID_TASKS = ["sst2"] #["sst2", "yelp", "squad", "coqa", "kde4", "tatoeba"]

EVAL_CONFIG = {
    "eval_num": 1000,
    "max_new_tokens": 50,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

def auto_discover_models(directory):
    """Scan directory for qwen2_*.pt files and build task-to-file mapping."""
    print(f"Scanning directory: {directory}")
    model_map = {
        "Base_Model": None # Always include Base Model
    }
    
    try:
        files = os.listdir(directory)
    except FileNotFoundError:
        print(f"Error: Directory not found: {directory}")
        return model_map

    # Match qwen2_*.pt and extract task name
    pattern = re.compile(r"qwen2_(.*)\.pt")

    for f in files:
        match = pattern.match(f)
        if match:
            task_name = match.group(1)
            if task_name in VALID_TASKS:
                model_map[task_name] = f
                print(f"  [Found] Task: {task_name:<10} -> File: {f}")
            else:
                print(f"  [Skip]  Found qwen2 model but task '{task_name}' is not in VALID_TASKS.")
    
    print(f"Total models found (including Base): {len(model_map)}")
    return model_map

# Metric calculation
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

# Dataset
class UniversalEvalDataset(Dataset):
    def __init__(self, task_name, num_samples=None):
        self.samples = []
        self.metric_type = "unknown"
        
        # QA tasks
        if task_name == "squad":
            self.metric_type = "qa"
            data = load_dataset("squad", split="validation")
            if num_samples: data = data.select(range(min(num_samples, len(data))))
            for item in data:
                prompt = f"Answer the question from the Given context. Context: {item['context']} Question: {item['question']} Answer:"
                self.samples.append((prompt, item['answers']['text']))

        elif task_name == "coqa":
            self.metric_type = "qa"
            data = load_dataset("stanfordnlp/coqa", split="validation")
            if num_samples: data = data.select(range(min(num_samples, len(data))))
            for item in data:
                for q, a in zip(item["questions"], item["answers"]["input_text"]):
                    prompt = f"Answer the question from the given context. Context: {item['story']} Question: {q} Answer:"
                    self.samples.append((prompt, [a]))
                    if num_samples and len(self.samples) >= num_samples: break
                if num_samples and len(self.samples) >= num_samples: break

        # Translation tasks (with offset)
        elif task_name == "kde4":
            self.metric_type = "trans"
            start_idx = 30000
            data = load_dataset("kde4", lang1="en", lang2="fr", split="train")
            end_idx = min(start_idx + num_samples, len(data))
            data = data.select(range(start_idx, end_idx))
            for item in data:
                prompt = f"Translate English to French. English: {item['translation']['en']}\nFrench:"
                self.samples.append((prompt, item['translation']['fr']))

        elif task_name == "tatoeba":
            self.metric_type = "trans"
            start_idx = 40000
            data = load_dataset("tatoeba", lang1="en", lang2="fr", split="train", trust_remote_code=True)
            end_idx = min(start_idx + num_samples, len(data))
            data = data.select(range(start_idx, end_idx))
            for item in data:
                prompt = f"Translate English to French. English: {item['translation']['en']}\nFrench:"
                self.samples.append((prompt, item['translation']['fr']))

        # Sentiment tasks
        elif task_name == "sst2":
            self.metric_type = "sentiment"
            data = load_dataset("glue", "sst2", split="validation")
            real_limit = min(num_samples, len(data)) if num_samples else len(data)
            data = data.select(range(real_limit))
            for item in data:
                target = "Positive" if item['label'] == 1 else "Negative"
                prompt = f"Review: {item['sentence'].strip()}\nSentiment:"
                self.samples.append((prompt, target))

        elif task_name == "yelp":
            self.metric_type = "sentiment"
            data = load_dataset("yelp_polarity", split="test")
            if num_samples: data = data.select(range(min(num_samples, len(data))))
            for item in data:
                target = "Positive" if item['label'] == 1 else "Negative"
                prompt = f"Review: {item['text'].replace(chr(10), ' ').strip()}\nSentiment:"
                self.samples.append((prompt, target))

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx], self.metric_type

# Evaluation
def evaluate_single_run(model, tokenizer, dataset, desc):
    metrics = {"acc": 0, "f1": 0, "em": 0, "bleu": 0}
    count = 0
    metric_type = dataset.metric_type
    
    pbar = tqdm(range(len(dataset)), desc=desc, leave=False)
    
    for i in pbar:
        (prompt, reference), _ = dataset[i]
        try:
            output = model.generate(
                prompt, max_new_tokens=EVAL_CONFIG["max_new_tokens"], 
                temperature=0.001, stop_at_eos=True, verbose=False
            )
            gen_text = output.replace(prompt, "").strip().split("\n")[0].strip()
            
            if metric_type == "qa":
                f1, em = compute_qa_metrics(gen_text, reference)
                metrics["f1"] += f1
                metrics["em"] += em
            elif metric_type == "trans":
                bleu = compute_bleu(gen_text, reference)
                metrics["bleu"] += bleu
            elif metric_type == "sentiment":
                acc = compute_accuracy(gen_text, reference)
                metrics["acc"] += acc
            count += 1
        except Exception:
            continue
            
    result = {}
    if count > 0:
        if metric_type == "qa":
            result = {"F1": metrics["f1"]/count, "EM": metrics["em"]/count}
        elif metric_type == "trans":
            result = {"BLEU": metrics["bleu"]/count}
        elif metric_type == "sentiment":
            result = {"Accuracy": metrics["acc"]/count}
    return result

# Main matrix loop
def run_matrix():
    discovered_map = auto_discover_models(CHECKPOINT_DIR)

    print(f"\nLoading Base Architecture: {BASE_MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    model = HookedTransformer.from_pretrained(
        BASE_MODEL_NAME, 
        device=EVAL_CONFIG['device'], 
        dtype=torch.float32
    )
    model.cfg.use_attn_result = False

    all_results = []

    sorted_models = ["Base_Model"] + [t for t in VALID_TASKS if t in discovered_map]
    
    for model_key in sorted_models:
        if model_key not in discovered_map: continue
        
        ckpt_file = discovered_map[model_key]
        print("\n" + "="*50)
        print(f"?? Testing Model: {model_key} (File: {ckpt_file})")
        print("="*50)
        
        if ckpt_file is None:
            print("Using Base Model weights (No checkpoint loaded).")
            pass
        else:
            full_path = os.path.join(CHECKPOINT_DIR, ckpt_file)
            print(f"Loading weights from: {full_path}")
            state_dict = torch.load(full_path, map_location=EVAL_CONFIG['device'])
            model.load_state_dict(state_dict, strict=False)
        
        for task_name in VALID_TASKS:
            print(f"   [Task: {task_name}] Running...")
            
            ds = UniversalEvalDataset(task_name, num_samples=EVAL_CONFIG["eval_num"])
            scores = evaluate_single_run(model, tokenizer, ds, desc=f"{model_key} -> {task_name}")
            
            record = {
                "Model_Source": model_key,
                "Eval_Task": task_name,
                **scores
            }
            all_results.append(record)
            print(f"   -> Score: {scores}")

    df = pd.DataFrame(all_results)
    print("\n" + "="*50)
    print("FINAL MATRIX RESULTS")
    print("="*50)
    print(df)
    
    csv_filename = "qwen_matrix_results.csv"
    df.to_csv(csv_filename, index=False)
    print(f"\nSaved to {csv_filename}")

if __name__ == "__main__":
    run_matrix()