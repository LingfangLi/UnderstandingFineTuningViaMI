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
    "project_name": "Qwen2-QA-FineTuning-MI",

    "task_name": "squad",

    "model_name": "Qwen/Qwen2-0.5B",

    # Training parameters
    "batch_size": 4,
    "grad_accum_steps": 4, # Effective batch = batch_size * grad_accum_steps = 16

    "lr": 2e-5,
    "num_epochs": 3,
    "max_length": 512,
    "patience": 3,

    "train_limit": 36, # None = use all data
    "save_dir": "/mnt/scratch/users/sglli24/fine-tuning-project/fine_tuned_model/old_version/qa",
    "seed": 42
}

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class QADataset(Dataset):
    def __init__(self, task_name, split, tokenizer, max_length=512, limit=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        
        print(f"Loading {task_name} dataset ({split})...")
        
        # Load raw data
        if task_name == "squad":
            data = load_dataset("squad", split=split)
        elif task_name == "coqa":
            data = load_dataset("stanfordnlp/coqa", split=split)
            
        if task_name == "squad":
            for item in tqdm(data, desc="Processing SQuAD"):
                if limit and len(self.samples) >= limit:
                    break
                    
                prompt = f"Answer the question from the Given context. Context: {item['context']} Question: {item['question']} Answer: {item['answers']['text'][0]}"
                self.samples.append(prompt)
                
        elif task_name == "coqa":
            for item in tqdm(data, desc="Processing CoQA"):
                if limit and len(self.samples) >= limit:
                    break
            
                story = item["story"]
                questions = item["questions"]
                answers = item["answers"]["input_text"]
                
                # Each CoQA row contains multiple Q&A pairs
                for q, a in zip(questions, answers):
                    if limit and len(self.samples) >= limit:
                        break
                        
                    prompt = f"Answer the question from the given context. Context: {story} Question: {q} Answer: {a}"
                    self.samples.append(prompt)
                    
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
        
        # Calculate Loss
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
    
    # Initialize WandB
    wandb.init(
        project=CONFIG["project_name"], 
        config=CONFIG,
        name=f"{CONFIG['model_name'].split('/')[-1]}-{CONFIG['task_name']}"
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load Tokenizer
    print(f"Loading Tokenizer: {CONFIG['model_name']}...")
    if "gpt2" in CONFIG["model_name"]:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
    else:
        tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"], trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load Model (HookedTransformer)
    print(f"Loading Model: {CONFIG['model_name']}...")
    model = HookedTransformer.from_pretrained(
        CONFIG["model_name"], 
        device=device,
        dtype=torch.float32 
    )

    # Disable memory-intensive analysis flags
    model.cfg.use_attn_result = False
    model.cfg.use_split_qkv_input = False
    model.cfg.use_attn_in = False
    model.cfg.use_hook_mlp_in = False

    # Prepare data
    train_dataset = QADataset(CONFIG["task_name"], "train", tokenizer, CONFIG["max_length"], CONFIG["train_limit"])
    val_dataset = QADataset(CONFIG["task_name"], "validation", tokenizer, CONFIG["max_length"], limit=1000)
    
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
        
        # Early stopping and saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_name = f"{CONFIG['model_name'].split('/')[-1]}_{CONFIG['task_name']}_best.pt"
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