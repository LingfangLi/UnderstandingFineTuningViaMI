import torch
from torch.utils.data import Dataset, random_split
from datasets import load_dataset
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer
from tqdm import tqdm
import sacrebleu
import os

# Configuration
CONFIG = {
    "task_name": "kde4",

    "model_name": "Qwen/Qwen2.5-0.5B",

    "checkpoint_path": "/mnt/scratch/users/yangwr/Lingfang/fine-tuning-project/fine_tuned_model/old_version/mt/Qwen2.5-0.5B_kde4_best.pt",

    "eval_num": 100,  # None = entire validation split

    "train_limit": 30000,  # KDE4: 30000, Tatoeba: 40000

    "max_new_tokens": 128,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seed": 42 
}

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class MTEvalDataset(Dataset):
    def __init__(self, task_name, split, tokenizer, max_length=128, select_range=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        
        print(f"Loading {task_name} dataset ({split})...")
        
        # Load data (with compatibility for legacy kde4)
        if task_name == "kde4":
            try:
                data = load_dataset("opus_books", "en-fr", split=split)
                print("Loaded 'opus_books' as 'kde4' replacement.")
            except Exception:
                data = load_dataset("kde4", lang1="en", lang2="fr", split=split)
        elif task_name == "tatoeba":
            data = load_dataset("tatoeba", lang1="en", lang2="fr", split=split, trust_remote_code=True)
            
        # Select specific range if provided
        if select_range:
            actual_stop = min(select_range.stop, len(data))
            actual_start = select_range.start
            data = data.select(range(actual_start, actual_stop))
            
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        en_text = item["translation"]["en"]
        fr_text = item["translation"]["fr"]
        
        # Construct Prompt (Must match training exactly)
        prompt = f"Translate English to French. English: {en_text}\nFrench:"
        
        # We return the prompt and the reference (Ground Truth)
        return prompt, fr_text

def get_validation_dataset(tokenizer):
    """Reconstruct the exact validation set used during training."""
    set_seed(CONFIG["seed"]) # Important for reproducibility
    
    if CONFIG["task_name"] == "kde4":
        # KDE4 Logic: Training was 0 to train_limit. 
        # Validation was train_limit to train_limit + 1000.
        print("Using explicit split for KDE4 validation set...")
        val_range = range(CONFIG["train_limit"], CONFIG["train_limit"] + 1000)
        
        dataset = MTEvalDataset(
            CONFIG["task_name"], "train", tokenizer, select_range=val_range
        )
        return dataset

    elif CONFIG["task_name"] == "tatoeba":
        # Tatoeba Logic: Random split 90/10 from the first train_limit samples
        print("Reconstructing Tatoeba random split...")
        full_range = range(0, CONFIG["train_limit"])
        full_dataset = MTEvalDataset(
            CONFIG["task_name"], "train", tokenizer, select_range=full_range
        )
        
        train_size = int(0.9 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        
        # Re-perform the split with the same seed
        _, val_dataset = random_split(full_dataset, [train_size, val_size])
        return val_dataset

def load_model_and_tokenizer():
    print(f"Loading tokenizer: {CONFIG['model_name']}...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    print(f"Loading model: {CONFIG['model_name']}...")
    model = HookedTransformer.from_pretrained(
        CONFIG['model_name'], 
        device=CONFIG['device'],
        # Evaluation can usually run in float16 or bfloat16
        dtype=torch.float16 if "gpt2" not in CONFIG['model_name'] else torch.float32
    )
    
    print(f"Loading weights from {CONFIG['checkpoint_path']}...")
    state_dict = torch.load(CONFIG['checkpoint_path'], map_location=CONFIG['device'])
    model.load_state_dict(state_dict, strict=False)
    
    model.cfg.use_attn_result = False
    model.eval()
    return model, tokenizer

def evaluate():
    model, tokenizer = load_model_and_tokenizer()
    
    # Get the correct validation split
    full_val_dataset = get_validation_dataset(tokenizer)
    
    # Limit evaluation number if needed
    if CONFIG["eval_num"]:
        indices = range(min(CONFIG["eval_num"], len(full_val_dataset)))
        dataset = torch.utils.data.Subset(full_val_dataset, indices)
    else:
        dataset = full_val_dataset
    
    print(f"Starting evaluation on {len(dataset)} samples...")
    
    refs = []  # Ground truths
    preds = [] # Model predictions
    exact_matches = 0
    
    for i in tqdm(range(len(dataset))):
        prompt, reference = dataset[i] # prompt, fr_text
        
        try:
            # 1. Generate
            output = model.generate(
                prompt, 
                max_new_tokens=CONFIG["max_new_tokens"], 
                temperature=0.001, 
                stop_at_eos=True,
                verbose=False
            )
            
            # 2. Extract translation
            generated_text = output.replace(prompt, "").strip()
            
            # Truncate at newline if model generates one
            if "\n" in generated_text:
                generated_text = generated_text.split("\n")[0].strip()
                
            preds.append(generated_text)
            refs.append(reference)
            
            # 3. Exact Match Check
            if generated_text.strip() == reference.strip():
                exact_matches += 1
            
            # 4. Print Examples
            if i < 3 or i % 20 == 0:
                print(f"\n[Sample {i}]")
                print(f"Source: {prompt.split('English: ')[1].split('\n')[0]}")
                print(f"Target: {reference}")
                print(f"Generated: {generated_text}")
                
        except Exception as e:
            print(f"Error on sample {i}: {e}")
            continue

    # 5. Compute Metrics
    print("\nComputing Metrics...")
    
    # BLEU Score
    bleu = sacrebleu.corpus_bleu(preds, [refs])
    
    # Accuracy
    accuracy = exact_matches / len(dataset) if len(dataset) > 0 else 0
    
    print("="*30)
    print(f"Final Results for {CONFIG['task_name']} ({CONFIG['model_name']})")
    print(f"Samples Evaluated: {len(dataset)}")
    print(f"BLEU Score: {bleu.score:.2f}")
    print(f"Exact Match (Acc): {accuracy:.4f} ({(accuracy*100):.2f}%)")
    print("="*30)

if __name__ == "__main__":
    evaluate()