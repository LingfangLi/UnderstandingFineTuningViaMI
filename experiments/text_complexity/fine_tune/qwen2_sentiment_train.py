import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer
import wandb
import os
from tqdm import tqdm

MODE = "complex" 

BASE_CONFIG = {
    "project_name": "Qwen2-Yelp-FineTuning",
    "model_name": "Qwen/Qwen2-0.5B", 
    "dataset": "yelp_polarity",
    "save_dir": "/mnt/scratch/users/sglli24/fine-tuning-project/fine_tuned_model",
    "seed": 42,
    "max_length": 256,
}


PARAM_MAP = {
    "simple": {

        "batch_size": 4,
        "grad_accum_steps": 8,   
        "lr": 1e-5,              
        "num_epochs": 5,        
        "weight_decay": 0.01,    
        "clip_norm": 1.0,        
        "use_scheduler": False, 
        "subset_indices": "/users/sglli24/UnderstandingFineTuningViaMI/experiments/text_complexity/matrix_analysis/yelp_lexically_simple_subset_indices.txt" ,
        "test_indices": "/users/sglli24/UnderstandingFineTuningViaMI/experiments/text_complexity/matrix_analysis/yelp_lexically_simple_test_indices.txt" ,
        "patience":3
    },
    "complex": {
        "batch_size": 4,
        "grad_accum_steps": 16,  
        "lr": 5e-6,              
        "num_epochs": 3,        
        "weight_decay": 0.1,     
        "clip_norm": 0.5,        
        "use_scheduler": True,  
        "subset_indices": "/users/sglli24/UnderstandingFineTuningViaMI/experiments/text_complexity/matrix_analysis/yelp_lexically_complex_subset_indices.txt",
        "test_indices": "/users/sglli24/UnderstandingFineTuningViaMI/experiments/text_complexity/matrix_analysis/yelp_lexically_complex_test_indices.txt" ,
        "patience":2
    }
}

CONFIG = {**BASE_CONFIG, **PARAM_MAP[MODE]}

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class YelpDataset(Dataset):
    def __init__(self, raw_data, tokenizer, max_length=256, limit=None):
        self.data = raw_data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        self.label_map = {0: "Negative", 1: "Positive"}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        # Handle possible newline characters to keep clean
        text = text.replace('\n', ' ').strip()
        
        label_text = self.label_map[item['label']]
        
        # Construct Prompt
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
            "attention_mask": encoding["attention_mask"].squeeze(0)
        }

def train_epoch(model, dataloader, optimizer, device, accum_steps, clip_norm, scheduler=None):
    model.train()
    total_loss = 0
    current_loss = 0
    optimizer.zero_grad()
    
    for step, batch in enumerate(tqdm(dataloader, desc="Training")):
        input_ids = batch["input_ids"].to(device)
        
        loss = model(input_ids, return_type="loss")
        loss = loss / accum_steps
        loss.backward()
        current_loss += loss.item() * accum_steps
        
        if (step + 1) % accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            
            optimizer.step()
            
            if scheduler is not None:
                scheduler.step()
                
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

from transformers import get_cosine_schedule_with_warmup
def main():
    set_seed(CONFIG["seed"])
    
    wandb.init(
        project=CONFIG["project_name"], 
        config=CONFIG,
        name=f"{CONFIG['model_name'].split('/')[-1]}-Yelp-100k"
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

    # Disable analysis hooks
    model.cfg.use_attn_result = False
    model.cfg.use_split_qkv_input = False
    model.cfg.use_attn_in = False
    model.cfg.use_hook_mlp_in = False

    # Prepare data
    with open(CONFIG['subset_indices'], 'r') as f:
        index_list = [int(line.strip()) for line in f]
    raw_data = load_dataset('yelp_polarity')['train'].select(index_list)
    
    with open(CONFIG['test_indices'], 'r') as f:
        test_index_list = [int(line.strip()) for line in f]
    test_data = load_dataset('yelp_polarity')['test'].select(test_index_list)
    
    train_dataset = YelpDataset(raw_data, tokenizer, CONFIG["max_length"])
    val_dataset = YelpDataset(test_data, tokenizer, CONFIG["max_length"]) 
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"])

    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    
    scheduler = None
    if CONFIG["use_scheduler"]:
        total_steps = len(train_loader) * CONFIG["num_epochs"] // CONFIG["grad_accum_steps"]
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * 0.1), # 10% Warmup
            num_training_steps=total_steps
        )
        print("Using Cosine Scheduler with Warmup")
        
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    
    if not os.path.exists(CONFIG["save_dir"]):
        os.makedirs(CONFIG["save_dir"])

    for epoch in range(CONFIG["num_epochs"]):
        print(f"Epoch {epoch+1}/{CONFIG['num_epochs']}")
        
        train_loss = train_epoch(model, train_loader, optimizer, device, CONFIG["grad_accum_steps"],CONFIG["clip_norm"], scheduler)
        val_loss = evaluate(model, val_loader, device)
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        wandb.log({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_path = os.path.join(CONFIG["save_dir"], f"{CONFIG['model_name'].split('/')[-1]}_yelp_complex.pt")
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