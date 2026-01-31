import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer
from tqdm import tqdm
import os

# Configuration
CONFIG = {
    # 1. Model settings
    "model_name": "Qwen/Qwen2-0.5B",
    
    # 2. Path to your fine-tuned model weights
    "checkpoint_path": "<MODEL_STORAGE>/fine-tuning-project/fine_tuned_model/text_complexity/Qwen2-0.5B_yelp_complex.pt",
    
    # True = evaluate base model; False = evaluate fine-tuned checkpoint
    "eval_base_model": True,
    
    "eval_num": 1000,

    "batch_size": 1, 
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

class YelpEvalDataset(Dataset):
    def __init__(self, split="test", num_samples=None):
        print(f"Loading Yelp Polarity {split} dataset...")
        self.data = load_dataset("yelp_polarity", split=split)
        
        if num_samples:
            print(f"Selecting first {num_samples} samples for evaluation...")
            self.data = self.data.select(range(min(num_samples, len(self.data))))
            
        # Yelp Polarity: 0 -> Negative, 1 -> Positive
        self.label_map = {0: "Negative", 1: "Positive"}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # Match training preprocessing: remove line breaks
        text = item['text'].replace('\n', ' ').strip()
        label_idx = item['label']
        target_word = self.label_map[label_idx]
        
        # Prompt ends at "Sentiment:" for model completion
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
    
    # Disable analysis-specific flags
    model.cfg.use_attn_result = False
    model.cfg.use_split_qkv_input = False
    
    model.eval()
    return model, tokenizer

def evaluate():
    model, tokenizer = load_model_and_tokenizer()
    # Yelp standard evaluation uses the "test" split
    dataset = YelpEvalDataset(split="test", num_samples=CONFIG["eval_num"])
    
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
                verbose=False,
                stop_at_eos=True
            )
            
            generated_text = output.replace(prompt, "").strip()

            if target_word.lower() in generated_text.lower():
                correct += 1
            
            if i % 100 == 0:
                print(f"\n[Sample {i}]")
                print(f"Prompt Snippet: ...{prompt[-100:].strip().replace(chr(10), ' ')}")
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
    print(f"Evaluation Mode: {mode_str}")
    if not CONFIG["eval_base_model"]:
        print(f"Checkpoint: {CONFIG['checkpoint_path']}")
    print(f"Total Samples: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.4f}")
    print("="*30)

if __name__ == "__main__":
    evaluate()