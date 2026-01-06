import torch
import os
import wandb
import json
from datetime import datetime
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from trl import SFTConfig, SFTTrainer

# ==========================================
# 0. Environment Setup
# ==========================================
os.environ["WANDB_PROJECT"] = "MI_gpt2-small-yelp" 

# ==========================================
# 1. Experiment Configuration
# ==========================================
run_name = f"gpt2-small-yelp-full-ft-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
output_base_dir = "./fine_tuned_model/" 
output_dir = os.path.join(output_base_dir, run_name)

config = {
    "model_name": "gpt2",       
    "dataset_name": "yelp_polarity",
    "max_seq_length": 512,      
    "learning_rate": 5e-5,      
    "batch_size": 32,          
    "gradient_accumulation_steps": 1,
    "num_epochs": 3,
    "eval_steps": 500,
    "save_steps": 500,
    "logging_steps": 50,
    "evaluation_strategy": "steps",
    "target_label_positive": "positive",
    "target_label_negative": "negative"
}

# ==========================================
# 2. Data Preparation
# ==========================================

raw_dataset = load_dataset(config['dataset_name'], split='train').select(range(11000))
dataset_dict = raw_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = dataset_dict['train']
eval_dataset = dataset_dict['test']

print(f"Train size: {len(train_dataset)} | Eval size: {len(eval_dataset)}")

def formatting_prompts_func(examples):
    def format_single_item(text, label_raw):
        if label_raw == 1 or str(label_raw).lower() == "positive":
            label_str = config['target_label_positive']
        else:
            label_str = config['target_label_negative']
        # Qwen can also use this simple format, or add Qwen's Chat template,
        # but to maintain experimental consistency, following the previous completion format is fine.
        return f"Review: {text}\nSentiment: {label_str}"

    if isinstance(examples['text'], list):
        output_texts = []
        for text, label_raw in zip(examples['text'], examples['label']):
            output_texts.append(format_single_item(text, label_raw))
        return output_texts
    else:
        return format_single_item(examples['text'], examples['label'])

# ==========================================
# 3. Model & Tokenizer
# ==========================================
use_bf16 = torch.cuda.is_bf16_supported()
use_fp16 = not use_bf16

model = AutoModelForCausalLM.from_pretrained(
    config['model_name'],
    torch_dtype=torch.bfloat16 if use_bf16 else torch.float16,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model.config.use_cache = False
model.config.pad_token_id = tokenizer.pad_token_id

# ==========================================
# 4. Training Arguments
# ==========================================
training_arguments = SFTConfig(
    output_dir=output_dir,
    run_name=run_name,
    max_length=config['max_seq_length'], 
    packing=False,
    
    per_device_train_batch_size=config['batch_size'],
    per_device_eval_batch_size=config['batch_size'],
    gradient_accumulation_steps=config['gradient_accumulation_steps'],
    
    learning_rate=config['learning_rate'],
    num_train_epochs=config['num_epochs'],
    
    eval_strategy=config['evaluation_strategy'],
    eval_steps=config['eval_steps'],
    save_strategy=config['evaluation_strategy'],
    save_steps=config['save_steps'],
    logging_steps=config['logging_steps'],
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    save_total_limit=2,
    
    fp16=use_fp16,
    bf16=use_bf16,
    
    max_grad_norm=1.0,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="cosine",
)

# ==========================================
# 5. Trainer Initialization
# ==========================================
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    formatting_func=formatting_prompts_func,
    processing_class=tokenizer,
    args=training_arguments,
)

# ==========================================
# 6. Training & Saving
# ==========================================
print(f"Starting training for {run_name}...")
trainer.train()
trainer.save_model(output_dir)

# Save parameters
param_path = os.path.join(output_dir, "experiment_config.json")
experiment_log = {
    "model_name": config['model_name'],
    "dataset": config['dataset_name'],
    "hyperparameters": config
}
with open(param_path, "w") as f:
    json.dump(experiment_log, f, indent=4)

print(f"Training completed. Model and Config saved to {output_dir}")