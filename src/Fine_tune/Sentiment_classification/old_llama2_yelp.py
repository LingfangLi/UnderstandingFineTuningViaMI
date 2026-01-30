import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
import wandb
import os
import numpy as np
from tqdm import tqdm
from datetime import datetime

# 1. Configuration
CONFIG = {
    "project_name": "Llama2-Yelp-Standard-HF-MI",
    "run_name": f"llama2-7b-yelp-manual-qlora-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
    "model_name": "meta-llama/Llama-2-7b-hf",
    "dataset_name": "yelp_polarity",
    "output_dir": "/mnt/scratch/users/sglli24/fine-tuning-project/fine_tuned_model/",
    
    "max_length": 512,
    "batch_size": 8,
    "learning_rate": 2e-4,
    "num_epochs": 2,
    "seed": 42,
    "train_sample_limit": 15000,
    "eval_sample_limit": 1000,
    
    # QLoRA Parameters
    "lora_r": 64,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "use_4bit": True,
}

final_output_dir = os.path.join(CONFIG["output_dir"], CONFIG["run_name"])
if not os.path.exists(final_output_dir):
    os.makedirs(final_output_dir)

# 2. Helper Functions
def set_seed(seed):
    """Sets the seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

# 3. Dataset Class (Whole Sequence Loss)
class YelpLlamaDataset(Dataset):
    def __init__(self, split, tokenizer, max_length=512, num_samples=None):
        raw_data = load_dataset(CONFIG["dataset_name"], split=split)
        
        if num_samples is not None:
            raw_data = raw_data.select(range(min(num_samples, len(raw_data))))
            print(f"Loaded {len(raw_data)} samples for split '{split}'")
            
        self.data = raw_data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_map = {0: "negative", 1: "positive"}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        label_str = self.label_map[item['label']]
        
        full_prompt = f"Review: {text}\nSentiment: {label_str}"

        encoding = self.tokenizer(
            full_prompt,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        
        # Whole Sequence Loss: labels = input_ids, padding masked with -100
        labels = input_ids.clone()

        if self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100
            
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels 
        }

# 4. Training & Evaluation Functions
def train_epoch(model, dataloader, optimizer, device):
    """Standard PyTorch training loop with whole-sequence loss."""
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for step, batch in enumerate(progress_bar):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        progress_bar.set_postfix({"loss": loss.item()})
        wandb.log({"train_step_loss": loss.item()})
        
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device):
    """Evaluation loop calculating whole-sequence loss."""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            
    return total_loss / len(dataloader)

# 5. Main Execution
def main():
    set_seed(CONFIG["seed"])
    
    wandb.init(
        project=CONFIG["project_name"],
        name=CONFIG["run_name"],
        config=CONFIG
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load Tokenizer
    print(f"Loading Tokenizer: {CONFIG['model_name']}...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'], trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load Model (QLoRA)
    print(f"Loading Model: {CONFIG['model_name']} with 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False,
    )

    model = AutoModelForCausalLM.from_pretrained(
        CONFIG['model_name'],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # Prepare LoRA Adapter
    model = prepare_model_for_kbit_training(model)
    
    peft_config = LoraConfig(
        r=CONFIG['lora_r'],
        lora_alpha=CONFIG['lora_alpha'],
        lora_dropout=CONFIG['lora_dropout'],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "v_proj"]
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Prepare Datasets
    print("Preparing Datasets...")
    train_dataset = YelpLlamaDataset(
        "train", tokenizer, CONFIG["max_length"], num_samples=CONFIG["train_sample_limit"]
    )
    eval_dataset = YelpLlamaDataset(
        "test", tokenizer, CONFIG["max_length"], num_samples=CONFIG["eval_sample_limit"]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=CONFIG["batch_size"], shuffle=False)

    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"])

    # Training Loop
    print(f"Starting training for {CONFIG['num_epochs']} epochs...")
    best_val_loss = float('inf')
    
    for epoch in range(CONFIG["num_epochs"]):
        print(f"\nEpoch {epoch + 1}/{CONFIG['num_epochs']}")
        
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate(model, eval_loader, device)
        
        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        wandb.log({
            "epoch": epoch + 1,
            "train_epoch_loss": train_loss,
            "val_loss": val_loss
        })
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"New best model found! Saving to {final_output_dir}")
            model.save_pretrained(final_output_dir)
            tokenizer.save_pretrained(final_output_dir)

    print("Training complete.")
    wandb.finish()

if __name__ == "__main__":
    main()