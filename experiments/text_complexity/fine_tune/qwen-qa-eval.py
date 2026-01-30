import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer
from tqdm import tqdm
import re
import string
import collections
import os

# Configuration
EXPERIMENT_TYPE = "simple"

CONFIG = {
    "task_name": "squad",
    "model_name": "Qwen/Qwen2-0.5B",
    "use_base_model": False, # True = base model, False = fine-tuned model
    "checkpoint_path": f"/mnt/scratch/users/sglli24/fine-tuning-project/fine_tuned_model/text_complexity/Qwen2-0.5B_squad_complex_best.pt",
    "test_data_index": f"/users/sglli24/UnderstandingFineTuningViaMI/experiments/text_complexity/matrix_analysis/squad_lexically_{EXPERIMENT_TYPE}_test_indices.txt",
    "eval_num": 1000, # None = entire validation set
    "max_new_tokens": 30,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

# QA evaluation metrics (F1 and Exact Match)
def normalize_answer(s):
    """Normalize answer: remove punctuation, lowercase, remove articles"""
    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_f1(prediction, truth):
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(truth).split()
    
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)
    
    common_tokens = collections.Counter(pred_tokens) & collections.Counter(truth_tokens)
    num_same = sum(common_tokens.values())
    
    if num_same == 0:
        return 0
    
    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def compute_exact(prediction, truth):
    return int(normalize_answer(prediction) == normalize_answer(truth))


class QAEvalDataset(Dataset):
    def __init__(self, data, num_samples=None):
        self.samples = []
        self.data = data
    
        for item in self.data:
            # SQuAD prompt format
            prompt = f"Answer the question from the Given context. Context: {item['context']} Question: {item['question']} Answer:"
            # SQuAD has multiple possible correct answers, we store them all
            references = item['answers']['text'] 
            self.samples.append((prompt, references))
        
        print(f"Loaded {len(self.samples)} evaluation samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def load_model_and_tokenizer():
    print(f"Loading tokenizer: {CONFIG['model_name']}...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    print(f"Loading model: {CONFIG['model_name']}...")
    model = HookedTransformer.from_pretrained(
        CONFIG['model_name'], 
        device=CONFIG['device'],
        dtype=torch.float16 if "gpt2" not in CONFIG['model_name'] else torch.float32
    )
    
    # Conditional weight loading
    if CONFIG["use_base_model"]:
        print("\n[INFO] Evaluating BASE MODEL (Pre-trained). Skipping checkpoint loading.\n")
    else:
        print(f"\n[INFO] Evaluating FINE-TUNED MODEL. Loading state dict from {CONFIG['checkpoint_path']}...\n")
        if not os.path.exists(CONFIG['checkpoint_path']):
            raise FileNotFoundError(f"Checkpoint not found at: {CONFIG['checkpoint_path']}")
            
        state_dict = torch.load(CONFIG['checkpoint_path'], map_location=CONFIG['device'])
        model.load_state_dict(state_dict, strict=False)
    
    # Disable analysis hooks during evaluation
    model.cfg.use_attn_result = False
    
    model.eval()
    return model, tokenizer

def evaluate():
    model, tokenizer = load_model_and_tokenizer()
    
    print(f"Loading test indices from {CONFIG['test_data_index']}...")
    with open(CONFIG["test_data_index"], 'r') as f:
        test_index_list = [int(line.strip()) for line in f if line.strip()]

    raw_val = load_dataset("squad", split="train").select(test_index_list)
    
    # Limit number of samples if configured
    if CONFIG["eval_num"] is not None:
        raw_val = raw_val.select(range(min(CONFIG["eval_num"], len(raw_val))))

    dataset = QAEvalDataset(raw_val)
    
    total_f1 = 0
    total_em = 0
    count = 0
    
    print("Starting generation and evaluation...")
    print(f"Mode: {'BASE MODEL' if CONFIG['use_base_model'] else 'FINE-TUNED MODEL'}")
    
    for i in tqdm(range(len(dataset))):
        prompt, references = dataset[i]
        
        try:
            # Generate
            output = model.generate(
                prompt, 
                max_new_tokens=CONFIG["max_new_tokens"], 
                temperature=0, # Greedy (deterministic)
                stop_at_eos=True,
                verbose=False
            )
            
            # Extract answer (remove prompt prefix)
            generated_text = output.replace(prompt, "").strip()
            
            # Truncate at first newline
            if "\n" in generated_text:
                generated_text = generated_text.split("\n")[0].strip()
            
            # Calculate metrics (best score across all reference answers)
            cur_f1 = max(compute_f1(generated_text, ref) for ref in references)
            cur_em = max(compute_exact(generated_text, ref) for ref in references)
            
            total_f1 += cur_f1
            total_em += cur_em
            count += 1
            
            # Print sample examples
            if i < 5 or i % 20 == 0:
                print(f"\n[Sample {i}]")
                # Handle cases where prompt splitting might fail if format is unexpected
                try:
                    q_snippet = prompt.split('Question: ')[1].split(' Answer:')[0]
                except:
                    q_snippet = prompt[:50] + "..."
                    
                print(f"Question (snippet): {q_snippet}")
                print(f"Generated: {generated_text}")
                print(f"References: {references}")
                print(f"F1: {cur_f1:.2f} | EM: {cur_em}")
                
        except Exception as e:
            print(f"Error on sample {i}: {e}")
            continue

    avg_f1 = total_f1 / count if count > 0 else 0
    avg_em = total_em / count if count > 0 else 0
    
    print("="*30)
    print(f"Final Results for {CONFIG['task_name']} ({CONFIG['model_name']})")
    print(f"Mode: {'Base Pre-trained' if CONFIG['use_base_model'] else 'Fine-tuned'}")
    print(f"Samples Evaluated: {count}")
    print(f"Average F1 Score: {avg_f1:.4f} ({(avg_f1*100):.2f}%)")
    print(f"Average Exact Match: {avg_em:.4f} ({(avg_em*100):.2f}%)")
    print("="*30)

if __name__ == "__main__":
    evaluate()