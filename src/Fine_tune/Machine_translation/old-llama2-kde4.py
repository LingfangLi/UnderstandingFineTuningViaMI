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

# Configuration
CONFIG = {
    "project_name": "Llama2-TATOEBA-Standard-HF-MI",
    "run_name": f"llama2-7b-kde4-manual-qlora-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
    "model_name": "meta-llama/Llama-2-7b-hf",
    "dataset_name": "tatoeba",#"kde4",
    "output_dir": "<MODEL_STORAGE>/fine-tuning-project/fine_tuned_model/",
    
    "max_length": 128,       
    
    # Effective batch size = 32 (8 * 4)
    "batch_size": 8,
    "gradient_accumulation_steps": 4,

    "learning_rate": 2e-4,
    "num_epochs": 3,
    "seed": 42,
    
    # Explicit slicing limits
    "train_limit": 15000, 
    "eval_limit": 1000,
    
    "lora_r": 64,
    "lora_alpha": 128,
    "lora_dropout": 0.05,
    "use_4bit": True,
}

# Ensure output directory exists
final_output_dir = os.path.join(CONFIG["output_dir"], CONFIG["run_name"])
if not os.path.exists(final_output_dir):
    os.makedirs(final_output_dir)

# Dataset Class
class KDE4LlamaDataset(Dataset):
    def __init__(self, split_type, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        
        print(f"Loading KDE4 dataset for {split_type}...")
        # KDE4 usually only has 'train' split in HF
        raw_data = load_dataset(CONFIG["dataset_name"], lang1="en", lang2="fr", split="train")
        
        # Explicit slicing logic
        if split_type == "train":
            # Range: 0 -> 30,000
            data_slice = raw_data.select(range(0, CONFIG["train_limit"]))
        elif split_type == "validation":
            # Range: 30,000 -> 31,000 (Avoid overlap)
            data_slice = raw_data.select(range(CONFIG["train_limit"], CONFIG["train_limit"] + CONFIG["eval_limit"]))
        
        print(f"Processing {len(data_slice)} samples...")
        
        for item in tqdm(data_slice, desc=f"Formatting {split_type}"):
            en_text = item['translation']['en']
            fr_text = item['translation']['fr']
            
            # Prompt format: "Translate English to French. English: {en}\nFrench: {fr}"
            prompt = f"Translate English to French. English: {en_text}\nFrench: {fr_text}"
            self.samples.append(prompt)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        prompt = self.samples[idx]
        
        encoding = self.tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        
        # Whole Sequence Loss
        labels = input_ids.clone()
        if self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100
            
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels 
        }

# Training Loop
def train_epoch(model, dataloader, optimizer, device, accum_steps):
    model.train()
    total_loss = 0
    current_loss_accum = 0
    optimizer.zero_grad()
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for step, batch in enumerate(progress_bar):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        
        loss = loss / accum_steps
        loss.backward()
        current_loss_accum += loss.item() * accum_steps
        
        if (step + 1) % accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            
            wandb.log({"train_step_loss": current_loss_accum / accum_steps})
            total_loss += current_loss_accum
            current_loss_accum = 0
            progress_bar.set_postfix({"loss": loss.item() * accum_steps})
            
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()
            
    return total_loss / len(dataloader)

# Main
def main():
    torch.manual_seed(CONFIG["seed"])
    
    wandb.init(project=CONFIG["project_name"], name=CONFIG["run_name"], config=CONFIG)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'], trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load Model
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG['model_name'],
        quantization_config=bnb_config,
        device_map="auto"
    )
    model.config.use_cache = False
    
    # LoRA
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

    # Datasets (Explicit Split)
    train_dataset = KDE4LlamaDataset("train", tokenizer, CONFIG["max_length"])
    eval_dataset = KDE4LlamaDataset("validation", tokenizer, CONFIG["max_length"])
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=CONFIG["batch_size"], shuffle=False)
    
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"])

    print(f"Starting training for {CONFIG['num_epochs']} epochs...")
    best_val_loss = float('inf')
    
    for epoch in range(CONFIG["num_epochs"]):
        print(f"\nEpoch {epoch + 1}/{CONFIG['num_epochs']}")
        train_loss = train_epoch(model, train_loader, optimizer, device, CONFIG["gradient_accumulation_steps"])
        val_loss = evaluate(model, eval_loader, device)
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        wandb.log({"epoch": epoch+1, "train_epoch_loss": train_loss, "val_loss": val_loss})
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"New best model saved to {final_output_dir}")
            model.save_pretrained(final_output_dir)
            tokenizer.save_pretrained(final_output_dir)

    wandb.finish()

if __name__ == "__main__":
    main()