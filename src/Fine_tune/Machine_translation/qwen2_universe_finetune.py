import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer
import wandb
import os
from tqdm import tqdm

# Configuration
CONFIG = {
    "project_name": "Qwen-MT-FineTuning-Project",

    "task_name": "kde4",  # "kde4" or "tatoeba"

    "model_name": "Qwen/Qwen2.5-0.5B",

    # KDE4: 30,000 entries; Tatoeba: 40,000 entries
    "train_limit": 30000,

    "batch_size": 8,
    "grad_accum_steps": 4,  # Effective batch size = 32
    "lr": 2e-5,
    "num_epochs": 3,
    "max_length": 128,
    "patience": 2,  # Early stopping
    
    "save_dir": "./mt_checkpoints",
    "seed": 42
}

# Automatically adjust limit to prevent manual errors
if CONFIG["task_name"] == "tatoeba" and CONFIG["train_limit"] == 30000:
    print("Notice: Switching train_limit to 40000 for Tatoeba as per reference.")
    CONFIG["train_limit"] = 40000

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class MTDataset(Dataset):
    def __init__(self, task_name, split, tokenizer, max_length=128, limit=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        
        print(f"Loading {task_name} dataset ({split})...")
        
        # Load different data sources based on task name
        if task_name == "kde4":
            data = load_dataset("kde4", lang1="en", lang2="fr", split=split)
        elif task_name == "tatoeba":
            data = load_dataset("tatoeba", lang1="en", lang2="fr", split=split, trust_remote_code=True)
            
        # Data filtering (Select)
        if limit:
            print(f"Selecting first {limit} samples...")
            data = data.select(range(min(limit, len(data))))
            
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Structure: item['translation']['en'] and item['translation']['fr']
        en_text = item["translation"]["en"]
        fr_text = item["translation"]["fr"]
        
        # Construct prompt
        prompt = f"Translate English to French. English: {en_text}\nFrench: {fr_text}"
        
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
    
    # Automatically generate Run Name
    run_name = f"{CONFIG['model_name'].split('/')[-1]}-{CONFIG['task_name']}-{CONFIG['train_limit']}"
    
    wandb.init(
        project=CONFIG["project_name"], 
        config=CONFIG,
        name=run_name
    )
    
    if not os.path.exists(CONFIG["save_dir"]):
        os.makedirs(CONFIG["save_dir"])

    # Save experiment configuration
    config_path = os.path.join(CONFIG["save_dir"], f"{CONFIG['task_name']}-{CONFIG['model_name']}-experiment_config.json")
    with open(config_path, "w") as f:
        json.dump(CONFIG, f, indent=4)
    print(f"Configuration saved to {config_path}")
    
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
        dtype=torch.float16 
    )

    # Disable analysis flags to save memory
    model.cfg.use_attn_result = False
    model.cfg.use_split_qkv_input = False
    model.cfg.use_attn_in = False
    model.cfg.use_hook_mlp_in = False

    # Prepare data (split validation set from train)
    
    full_dataset = MTDataset(CONFIG["task_name"], "train", tokenizer, CONFIG["max_length"], limit=CONFIG["train_limit"])
    
    # Split 90% train, 10% validation
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"])

    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=0.01)

    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    
    if not os.path.exists(CONFIG["save_dir"]):
        os.makedirs(CONFIG["save_dir"])

    for epoch in range(CONFIG["num_epochs"]):
        print(f"Epoch {epoch+1}/{CONFIG['num_epochs']}")
        
        train_loss = train_epoch(model, train_loader, optimizer, device, CONFIG["grad_accum_steps"])
        val_loss = evaluate(model, val_loader, device)
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        wandb.log({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_path = os.path.join(CONFIG["save_dir"], f"{CONFIG['model_name'].split('/')[-1]}_{CONFIG['task_name']}_best.pt")
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved to {save_path}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience {patience_counter}/{CONFIG['patience']}")
            
        if patience_counter >= CONFIG["patience"]:
            print("Early stopping triggered!")
            break

    wandb.finish()

if __name__ == "__main__":
    main()