import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer
from tqdm import tqdm
import os

# Configuration
CONFIG = {
    "model_name": "Qwen/Qwen2-0.5B",
    "use_base_model": False, # True = base model, False = fine-tuned model
    "checkpoint_path": "/mnt/scratch/users/sglli24/fine-tuning-project/fine_tuned_model/Qwen2-0.5B_yelp_complex.pt",
    "eval_num": 1000, # None = entire test set
    "batch_size": 1,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

class YelpDataset(Dataset):
    def __init__(self, raw_data, max_length=256, limit=None):
        self.data = raw_data
        self.label_map = {0: "Negative", 1: "Positive"}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # Remove line breaks (consistent with training preprocessing)
        text = item['text'].replace('\n', ' ').strip()
        label_idx = item['label']
        target_word = self.label_map[label_idx]
        
        # Construct input prompt
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
    
    # Conditional weight loading
    if CONFIG["use_base_model"]:
        print("\n[INFO] Evaluating BASE MODEL (Pre-trained). Skipping checkpoint loading.\n")
    else:
        print(f"\n[INFO] Evaluating FINE-TUNED MODEL. Loading state dict from {CONFIG['checkpoint_path']}...\n")
        if not os.path.exists(CONFIG['checkpoint_path']):
            raise FileNotFoundError(f"Checkpoint not found at: {CONFIG['checkpoint_path']}")
            
        state_dict = torch.load(CONFIG['checkpoint_path'], map_location=CONFIG['device'])
        model.load_state_dict(state_dict, strict=False) 
    
    # Disable analysis hooks
    model.cfg.use_attn_result = False
    model.cfg.use_split_qkv_input = False
    
    model.eval()
    return model, tokenizer

def evaluate():
    model, tokenizer = load_model_and_tokenizer()
    
    # Load dataset indices
    index_path = "/users/sglli24/UnderstandingFineTuningViaMI/experiments/text_complexity/matrix_analysis/yelp_lexically_simple_test_indices.txt"
    print(f"Loading indices from {index_path}...")
    
    with open(index_path, 'r') as f:
        # Skip empty lines
        test_index_list = [int(line.strip()) for line in f if line.strip()]
        
    dataset = load_dataset('yelp_polarity')['test'].select(test_index_list)
    
    # Limit evaluation number if specified in CONFIG
    if CONFIG["eval_num"] is not None:
        dataset = dataset.select(range(min(CONFIG["eval_num"], len(dataset))))
        
    dataset = YelpDataset(dataset)
    correct = 0
    total = 0
    
    print(f"Starting evaluation on {len(dataset)} samples...")
    print(f"Mode: {'BASE MODEL' if CONFIG['use_base_model'] else 'FINE-TUNED MODEL'}")
    
    # Generate one sample at a time
    for i in tqdm(range(len(dataset))):
        prompt, target_word = dataset[i]
        
        try:
            # Generate
            output = model.generate(
                prompt, 
                max_new_tokens=10,
                temperature=0, # Greedy (deterministic)
                verbose=False,
                stop_at_eos=True
            )
            
            # Extract generated content (remove prompt prefix)
            generated_text = output.replace(prompt, "").strip()
            
            # Case-insensitive substring matching
            if target_word.lower() in generated_text.lower():
                correct += 1
            
            # Print examples (every 100 samples)
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
    print("="*30)
    print(f"Final Evaluation Results:")
    print(f"Model Type: {'Base Pre-trained' if CONFIG['use_base_model'] else 'Fine-tuned'}")
    print(f"Model Name: {CONFIG['model_name']}")
    if not CONFIG['use_base_model']:
        print(f"Checkpoint: {CONFIG['checkpoint_path']}")
    print(f"Total Samples: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.4f}")
    print("="*30)

if __name__ == "__main__":
    evaluate()