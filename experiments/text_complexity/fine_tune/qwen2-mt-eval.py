import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import os

EXPERIMENT_TYPE = "complex"

# Configuration
CONFIG = {
    "task_name": "tatoeba",
    "model_name": "Qwen/Qwen2.5-0.5B",
    "use_base_model": False, # True = base model, False = fine-tuned model
    "checkpoint_path": f"",
    "test_data_index": f"/users/sglli24/UnderstandingFineTuningViaMI/experiments/text_complexity/matrix_analysis/tatoeba_lexically_{EXPERIMENT_TYPE}_test_indices.txt",
    "eval_num": 1000,
    "max_new_tokens": 128,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seed": 42
}

class MTEvalDataset(Dataset):
    def __init__(self, hf_data):
        self.data = hf_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        en_text = item["translation"]["en"]
        fr_text = item["translation"]["fr"]
        
        prompt = f"Translate English to French. English: {en_text}\nFrench:"
        
        return prompt, fr_text

def load_model_and_tokenizer():
    print(f"Loading tokenizer: {CONFIG['model_name']}...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    print(f"Loading model: {CONFIG['model_name']}...")
    # Load model
    model = HookedTransformer.from_pretrained(
        CONFIG['model_name'], 
        device=CONFIG['device'],
        dtype=torch.float32 
    )
    
    # Conditional weight loading
    if CONFIG["use_base_model"]:
        print("\n[INFO] Evaluating BASE MODEL (Pre-trained). Skipping checkpoint loading.\n")
    else:
        print(f"\n[INFO] Evaluating FINE-TUNED MODEL. Loading weights from {CONFIG['checkpoint_path']}...\n")
        if not os.path.exists(CONFIG['checkpoint_path']):
            raise FileNotFoundError(f"Checkpoint not found at: {CONFIG['checkpoint_path']}")
            
        state_dict = torch.load(CONFIG['checkpoint_path'], map_location=CONFIG['device'])
        model.load_state_dict(state_dict, strict=False)
    
    model.cfg.use_attn_result = False
    model.eval()
    return model, tokenizer

def evaluate():
    model, tokenizer = load_model_and_tokenizer()
    
    # Load test indices
    print(f"Loading test indices from {CONFIG['test_data_index']}...")
    if not os.path.exists(CONFIG['test_data_index']):
        raise FileNotFoundError(f"Index file not found: {CONFIG['test_data_index']}")
        
    with open(CONFIG["test_data_index"], 'r') as f:
        # Skip empty lines
        test_indices = [int(line.strip()) for line in f if line.strip()]
    
    # Load full dataset
    print(f"Loading full {CONFIG['task_name']} dataset...")
    # Tatoeba only has 'train' split on HuggingFace
    full_dataset = load_dataset("tatoeba", lang1="en", lang2="fr", split="train", trust_remote_code=True)
    
    # Select subset based on indices
    print(f"Selecting {len(test_indices)} samples based on indices...")
    hf_subset = full_dataset.select(test_indices)
    
    # Limit evaluation count if specified
    if CONFIG["eval_num"] and CONFIG["eval_num"] < len(hf_subset):
        print(f"Limiting evaluation to first {CONFIG['eval_num']} samples.")
        hf_subset = hf_subset.select(range(CONFIG["eval_num"]))
        
    dataset = MTEvalDataset(hf_subset)
    
    print(f"Starting NLTK evaluation on {len(dataset)} samples...")
    print(f"Mode: {'BASE MODEL' if CONFIG['use_base_model'] else 'FINE-TUNED MODEL'}")

    # NLTK Smoothing
    smoothing = SmoothingFunction().method1
    
    sum_score = 0
    count = 0
    
    for i in tqdm(range(len(dataset))):
        prompt, reference = dataset[i]
        
        try:
            output = model.generate(
                prompt, 
                max_new_tokens=CONFIG["max_new_tokens"], 
                temperature=0, # Greedy (deterministic)
                stop_at_eos=True,
                verbose=False
            )
            
            generated_text = output.replace(prompt, "").strip()
            if "\n" in generated_text:
                generated_text = generated_text.split("\n")[0].strip()
            
            # Whitespace tokenization for consistency
            ref_tokens = reference.lower().split()
            hyp_tokens = generated_text.lower().split()
            
            if len(hyp_tokens) > 0:
                bleu_score = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothing)
            else:
                bleu_score = 0.0
                
            sum_score += bleu_score
            count += 1
            
            if i < 3 or i % 50 == 0:
                print(f"\n[Sample {i}]")
                print(f"Target: {reference}")
                print(f"Generated: {generated_text}")
                print(f"BLEU: {bleu_score:.4f}")
                
        except Exception as e:
            print(f"Error on sample {i}: {e}")
            continue

    avg_bleu = sum_score / count if count > 0 else 0
    
    print("="*30)
    print(f"Final Evaluation Results ({CONFIG['task_name']} - {CONFIG['model_name']})")
    print(f"Mode: {'Base Pre-trained' if CONFIG['use_base_model'] else 'Fine-tuned'}")
    print(f"Samples Evaluated: {count}")
    print(f"Average BLEU Score (0-1): {avg_bleu:.4f}")
    print("="*30)

if __name__ == "__main__":
    evaluate()