import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer
from tqdm import tqdm
import os

# Configuration
CONFIG = {
    # Options: "gpt2", "meta-llama/Llama-3.2-1B", "Qwen/Qwen2-0.5B"
    "model_name": "meta-llama/Llama-3.2-1B",
    
    "checkpoint_path": "<MODEL_STORAGE>/fine-tuning-project-1/old_version_finetuned_models/llama3.2-tatoeba.pt", 
    
    "eval_base_model": False,
    
    "eval_num": 872, 
    
    "batch_size": 32, 
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

class SST2EvalDataset(Dataset):
    def __init__(self, split="validation", num_samples=None):
        print(f"Loading SST-2 {split} dataset...")
        self.data = load_dataset("stanfordnlp/sst2", split=split)
        
        if num_samples:
            self.data = self.data.select(range(min(num_samples, len(self.data))))
            
        self.label_map = {0: "Negative", 1: "Positive"}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['sentence']
        label_idx = item['label']
        target_word = self.label_map[label_idx]
        
        prompt = f"Review: {text}\nSentiment:"
        
        return prompt, target_word

def load_model_and_tokenizer():
    print(f"Loading tokenizer for {CONFIG['model_name']}...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    print(f"Loading model architecture {CONFIG['model_name']}...")
    model = HookedTransformer.from_pretrained(
        CONFIG['model_name'], 
        device=CONFIG['device'],
        dtype=torch.float32
    )
    
    # Switch between base model and fine-tuned checkpoint
    if CONFIG["eval_base_model"]:
        print("="*40)
        print(f"(!) MODE: Evaluating BASE MODEL ({CONFIG['model_name']})")
        print("(!) Skipping fine-tuned checkpoint loading.")
        print("="*40)
    else:
        print("="*40)
        print(f"(!) MODE: Evaluating FINE-TUNED Model")
        print(f"Loading state dict from {CONFIG['checkpoint_path']}...")
        state_dict = torch.load(CONFIG['checkpoint_path'], map_location=CONFIG['device'])
        model.load_state_dict(state_dict, strict=False) 
        print("="*40)
    
    model.eval()
    return model, tokenizer

def evaluate():
    model, tokenizer = load_model_and_tokenizer()
    dataset = SST2EvalDataset(split="validation", num_samples=CONFIG["eval_num"])
    
    correct = 0
    total = 0
    
    print(f"Starting evaluation on {len(dataset)} samples...")
    
    for i in tqdm(range(len(dataset))):
        prompt, target_word = dataset[i]
        
        try:
            
            output = model.generate(
                prompt, 
                max_new_tokens=50, 
                temperature=1.2, 
                verbose=False
            )
            
            generated_text = output.replace(prompt, "").strip()
            
            if target_word.lower() in generated_text.lower():
                correct += 1
            
            if i % 100 == 0:
                print(f"\n[Sample {i}]")
                print(f"Prompt: {prompt.strip()}")
                print(f"Target: {target_word}")
                print(f"Generated: {generated_text}")
                print(f"Current Acc: {correct / (i + 1):.4f}")
                
            total += 1
            
        except Exception as e:
            print(f"Error on sample {i}: {e}")
            continue

    accuracy = correct / total if total > 0 else 0
    mode_str = "Base Model" if CONFIG["eval_base_model"] else "Fine-tuned Model"

    print("="*30)
    print(f"Final Evaluation Results:")
    print(f"Model: {CONFIG['model_name']}")
    print(f"Mode: {mode_str}")
    if not CONFIG["eval_base_model"]:
        print(f"Checkpoint: {CONFIG['checkpoint_path']}")
    print(f"Total Samples: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.4f}")
    print("="*30)

if __name__ == "__main__":
    evaluate()