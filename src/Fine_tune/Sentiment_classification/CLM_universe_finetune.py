import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer
import wandb
import os
import numpy as np
from tqdm import tqdm

# Configuration
CONFIG = {
    "project_name": "qwen2-SST2-FineTuning-MI",
    # Options: "gpt2-small", "meta-llama/Llama-3.2-1B", "Qwen/Qwen2.5-0.5B"
    "model_name": "Qwen/Qwen2-0.5B", 
    "dataset": "stanfordnlp/sst2",
    "batch_size": 8,  
    "lr": 2e-5, 
    "train_num" : 50000,     
    "num_epochs": 4,  
    "max_length": 128, 
    "patience": 3,     
    "save_dir": "<MODEL_STORAGE>/fine-tuning-project/fine_tuned_model/old_version",
    "seed": 42
}

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
class SST2Dataset(Dataset):

    def __init__(self, split, tokenizer, max_length=128, num_samples=None):

        raw_data = load_dataset(CONFIG["dataset"], split=split)
        
        if num_samples is not None:
            actual_num = min(num_samples, len(raw_data))
            raw_data = raw_data.select(range(actual_num))
            print(f"Loaded {actual_num} samples for split '{split}'")
            
        self.data = raw_data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_map = {0: "Negative", 1: "Positive"}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['sentence']
        label_text = self.label_map[item['label']]
        

        prompt = f"Review: {text}\nSentiment: {label_text}"
        
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
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label_val": item['label'] # For eval calculation
        }

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    
    for step, batch in enumerate(tqdm(dataloader, desc="Training")):
        input_ids = batch["input_ids"].to(device)
        
        loss = model(input_ids, return_type="loss")
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        wandb.log({"train_step_loss": loss.item()})
        
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            
            loss = model(input_ids, return_type="loss")
            total_loss += loss.item()
            
            
    return total_loss / len(dataloader)

def main():
    set_seed(CONFIG["seed"])
    
    # Initialize WandB
    wandb.init(
        project=CONFIG["project_name"], 
        config=CONFIG,
        name=f"{CONFIG['model_name']}-SST2"
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load Tokenizer and model
    print(f"Loading {CONFIG['model_name']}...")
    if "gpt2" in CONFIG["model_name"]:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
    # Load from HookedTransformer
    model = HookedTransformer.from_pretrained(
        CONFIG["model_name"], 
        device=device, 
    )
    
    model.cfg.use_attn_in = False
    model.cfg.use_split_qkv_input = False
    model.cfg.use_attn_result = False
    model.cfg.use_hook_mlp_in = False
    
    train_num = CONFIG.get("train_num", None)
    train_dataset = SST2Dataset("train", tokenizer, CONFIG["max_length"],num_samples=train_num)
    val_dataset = SST2Dataset("validation", tokenizer, CONFIG["max_length"])
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"])

    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=0.01)

    best_val_loss = float('inf')
    patience_counter = 0
    
    if not os.path.exists(CONFIG["save_dir"]):
        os.makedirs(CONFIG["save_dir"])

    for epoch in range(CONFIG["num_epochs"]):
        print(f"Epoch {epoch+1}/{CONFIG['num_epochs']}")
        
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate(model, val_loader, device)
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss
        })
        

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best models
            save_path = f"{CONFIG['save_dir']}/{CONFIG['model_name'].split('/')[-1]}_best.pt"
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved to {save_path}")
            wandb.save(save_path) 
        else:
            patience_counter += 1
            print(f"No improvement. Patience {patience_counter}/{CONFIG['patience']}")
            
        if patience_counter >= CONFIG["patience"]:
            print("Early stopping triggered!")
            break

    wandb.finish()

if __name__ == "__main__":
    main()