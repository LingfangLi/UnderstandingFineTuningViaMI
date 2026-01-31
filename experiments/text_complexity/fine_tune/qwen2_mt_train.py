import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer
import wandb
import os
from tqdm import tqdm
import json

# Configuration
EXPERIMENT_TYPE = "complex"
CONFIG = {
    "project_name": "Qwen2-Tatoeba-FineTuning-MI",
    "model_name": "Qwen/Qwen2-0.5B",
    "task_name": "tatoeba",
    "num_epochs": 5,
    "patience": 3,
    "batch_size": 4,
    "grad_accum_steps": 8, # Effective batch = 32
    "lr": 2e-5,
    "max_length": 128,
    "save_dir": "<MODEL_STORAGE>/fine-tuning-project/fine_tuned_model/text_complexity",
    "seed": 42,
    "data_index": f"<PROJECT_ROOT>/experiments/text_complexity/matrix_analysis/tatoeba_lexically_{EXPERIMENT_TYPE}_subset_indices.txt",
    "test_data_index": f"<PROJECT_ROOT>/experiments/text_complexity/matrix_analysis/tatoeba_lexically_{EXPERIMENT_TYPE}_test_indices.txt"
}

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class TatoebaDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        self.data = dataset
        
        for item in tqdm(self.data, desc="Processing Tatoeba"):
            try:
                en_text = item["translation"]["en"]
                fr_text = item["translation"]["fr"]
                
                prompt = f"Translate English to French. English: {en_text}\nFrench: {fr_text}"
                self.samples.append(prompt)
            except KeyError:
                continue 
                    
        print(f"Total samples processed: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        prompt = self.samples[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            prompt, 
            truncation=True, 
            padding="max_length", 
            max_length=self.max_length, 
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0)
        }

def train_epoch(model, dataloader, optimizer, device, accum_steps):
    model.train()
    total_loss = 0
    current_loss = 0
    optimizer.zero_grad()
    
    for step, batch in enumerate(tqdm(dataloader, desc="Training")):
        input_ids = batch["input_ids"].to(device)
        
        loss = model(input_ids, return_type="loss")
        
        # Gradient accumulation
        loss = loss / accum_steps
        loss.backward()
        
        current_loss += loss.item() * accum_steps
        
        if (step + 1) % accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            
            wandb.log({"train_step_loss": current_loss / accum_steps})
            total_loss += current_loss
            current_loss = 0
            
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            loss = model(input_ids, return_type="loss")
            total_loss += loss.item()
            
    return total_loss / len(dataloader)

def main():
    set_seed(CONFIG["seed"])
    
    subset_type = "simple" if "simple" in CONFIG["data_index"] else "complex"
    
    wandb.init(
        project=CONFIG["project_name"], 
        config=CONFIG,
        name=f"{CONFIG['model_name'].split('/')[-1]}-tatoeba-{subset_type}"
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load Tokenizer
    print(f"Loading Tokenizer: {CONFIG['model_name']}...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load Model
    print(f"Loading Model: {CONFIG['model_name']}...")
    model = HookedTransformer.from_pretrained(
        CONFIG["model_name"], 
        device=device,
        dtype=torch.float32 
    )

    model.cfg.use_attn_result = False
    model.cfg.use_split_qkv_input = False
    model.cfg.use_attn_in = False 
    model.cfg.use_hook_mlp_in = False

    # Data preparation
    print(f"Loading full Tatoeba dataset...")
    #trust_remote_code=True
    full_dataset = load_dataset("tatoeba", lang1="en", lang2="fr", split="train", trust_remote_code=True)
    
    # Load training indices
    print(f"Loading indices from: {CONFIG['data_index']}")
    if not os.path.exists(CONFIG["data_index"]):
        raise FileNotFoundError(f"Index file not found: {CONFIG['data_index']}")
        
    with open(CONFIG["data_index"], 'r') as f:
        index_list = [int(line.strip()) for line in f]
        
    with open(CONFIG["test_data_index"], 'r') as f:
        test_index_list = [int(line.strip()) for line in f]    
    
    raw_train = full_dataset.select(index_list)
    
    raw_val = full_dataset.select(test_index_list)
    
    print(f"Train subset size: {len(raw_train)}")
    print(f"Validation set size: {len(raw_val)} ")

    train_dataset = TatoebaDataset(raw_train, tokenizer, CONFIG["max_length"])
    val_dataset = TatoebaDataset(raw_val, tokenizer, CONFIG["max_length"])
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"])

    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=0.01)

    # Save Config
    if not os.path.exists(CONFIG["save_dir"]):
        os.makedirs(CONFIG["save_dir"])
        
    config_path = os.path.join(CONFIG["save_dir"], f"tatoeba_{subset_type}_config.json")
    with open(config_path, "w") as f:
        json.dump(CONFIG, f, indent=4)

    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    
    print(f"Starting training for {CONFIG['num_epochs']} epochs...")
    
    for epoch in range(CONFIG["num_epochs"]):
        print(f"Epoch {epoch+1}/{CONFIG['num_epochs']}")
        
        train_loss = train_epoch(model, train_loader, optimizer, device, CONFIG["grad_accum_steps"])
        val_loss = evaluate(model, val_loader, device)
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        wandb.log({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            save_name = f"{CONFIG['model_name'].split('/')[-1]}_tatoeba_{subset_type}_best.pt"
            save_path = os.path.join(CONFIG["save_dir"], save_name)
            
            torch.save(model.state_dict(), save_path)
            print(f"Saved Best Model to {save_path}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience {patience_counter}/{CONFIG['patience']}")
            
        if patience_counter >= CONFIG["patience"]:
            print("Early stopping triggered!")
            break

    wandb.finish()

if __name__ == "__main__":
    main()