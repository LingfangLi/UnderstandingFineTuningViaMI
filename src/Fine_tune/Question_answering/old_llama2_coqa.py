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
    "project_name": "Llama2-CoQA-Standard-HF-MI",
    "run_name": f"llama2-7b-coqa-manual-qlora-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
    "model_name": "meta-llama/Llama-2-7b-hf",
    "dataset_name": "stanfordnlp/coqa",
    "output_dir": "/mnt/scratch/users/sglli24/fine-tuning-project/old_fine_tuned_model/",
    
    "max_length": 1024,       
    
    "batch_size": 4,         
    "gradient_accumulation_steps": 8, 
    
    "learning_rate": 2e-4,   
    "num_epochs": 1,         
    "seed": 42,
    
    "lora_r": 64,            
    "lora_alpha": 128,       
    "lora_dropout": 0.05,
    "use_4bit": True,
    
    "train_sample_limit": 36000,# 36000, 
    "eval_sample_limit": 1000, #1000,
    "patience": 3,
}
# Ensure output directory exists
final_output_dir = os.path.join(CONFIG["output_dir"], CONFIG["run_name"])
if not os.path.exists(final_output_dir):
    os.makedirs(final_output_dir)

# 2. Helper Functions
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

# 3. Dataset Class
class CoQALlamaDataset(Dataset):
    def __init__(self, split, tokenizer, max_length=512, limit=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        
        print(f"Loading CoQA dataset ({split}) and flattening...")
        raw_data = load_dataset(CONFIG["dataset_name"], split=split)
        
        # CoQA structure: one row = 1 story + N questions
        count = 0
        
        for item in tqdm(raw_data, desc=f"Processing {split}"):
            if limit and len(self.samples) >= limit:
                break
                
            story = item["story"]
            questions = item["questions"]
            answers = item["answers"]["input_text"]
            
            # Iterate through all Q&A pairs in the story
            for q, a in zip(questions, answers):
                if limit and len(self.samples) >= limit:
                    break
                
                prompt = f"Answer the question from the given context. Context: {story} Question: {q} Answer: {a}"
                self.samples.append(prompt)
                
        print(f"Total flattened samples loaded for '{split}': {len(self.samples)}")

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
        
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        
        # Whole sequence loss (prompt + answer)
        labels = input_ids.clone()
        if self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100
            
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels 
        }

# 4. Training & Evaluation Functions

def train_epoch(model, dataloader, val_loader, optimizer, device, accum_steps):
    model.train()
    total_loss = 0
    current_loss_accum = 0
    optimizer.zero_grad()
    
    for step, batch in enumerate(tqdm(dataloader, desc="Training")):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        outputs = model(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            labels=labels
        )
        
        loss = outputs.loss

        # Normalize loss for gradient accumulation
        loss = loss / accum_steps
        loss.backward()
        
        current_loss_accum += loss.item() * accum_steps

        # Step optimizer
        if (step + 1) % accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            
            wandb.log({"train_step_loss": current_loss_accum / accum_steps})
            total_loss += current_loss_accum
            current_loss_accum = 0
            
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
            loss = outputs.loss
            total_loss += loss.item()
            
    return total_loss / len(dataloader)

# 5. Main Execution
def main():
    set_seed(CONFIG["seed"])
    
    # 1. WandB Init
    wandb.init(
        project=CONFIG["project_name"],
        name=CONFIG["run_name"],
        config=CONFIG
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 2. Load Tokenizer
    print(f"Loading Tokenizer: {CONFIG['model_name']}...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'], trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 3. Load Model (QLoRA)
    print(f"Loading Model: {CONFIG['model_name']} with 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
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
    
    # 4. Prepare LoRA
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

    # 5. Prepare Datasets
    train_dataset = CoQALlamaDataset(
        "train", tokenizer, CONFIG["max_length"], limit=CONFIG["train_sample_limit"]
    )
    eval_dataset = CoQALlamaDataset(
        "validation", tokenizer, CONFIG["max_length"], limit=CONFIG["eval_sample_limit"]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=CONFIG["batch_size"], shuffle=False)

    # 6. Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"])

    # 7. Training Loop
    print(f"Starting training for {CONFIG['num_epochs']} epochs...")
    best_val_loss = float('inf')
    
    for epoch in range(CONFIG["num_epochs"]):
        print(f"\nEpoch {epoch + 1}/{CONFIG['num_epochs']}")
        
        # Uses Gradient Accumulation inside the loop
        train_loss = train_epoch(model, train_loader, eval_loader, optimizer, device, CONFIG["gradient_accumulation_steps"])
        val_loss = evaluate(model, eval_loader, device)
        
        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        wandb.log({
            "epoch": epoch + 1,
            "train_epoch_loss": train_loss,
            "val_loss": val_loss
        })
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            print(f"New best model found! Saving to {final_output_dir}")
            model.save_pretrained(final_output_dir)
            tokenizer.save_pretrained(final_output_dir)
        else:
            patience_counter += 1
            print(f"No improvement. Patience {patience_counter}/{CONFIG['patience']}")
            
        if patience_counter >= CONFIG["patience"]:
            print("Early stopping triggered!")
            break     

    print("Training complete.")
    wandb.finish()

if __name__ == "__main__":
    main()