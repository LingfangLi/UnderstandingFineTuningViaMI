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
os.environ["WANDB_PROJECT"] = "MI_llama3.2-tatoeba"

# ==========================================
# 1. Experiment Configuration
# ==========================================
run_name = f"llama3.2-1b-tatoeba-full-ft-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
output_base_dir = f"/mnt/scratch/users/sglli24/fine-tuning-project/fine_tuned_model/"
output_dir = os.path.join(output_base_dir, run_name)

config = {
    "model_name": "meta-llama/Llama-3.2-1B",
    "dataset_name": "tatoeba",
    "lang1": "en",
    "lang2": "fr",
    "max_seq_length": 256,     
    "learning_rate": 2e-5,
    "target_samples": 40000,
    "batch_size": 16,
    "gradient_accumulation_steps": 2,
    "num_epochs": 3,
    "eval_steps": 200,
    "save_steps": 200,
    "logging_steps": 50,
}

# ==========================================
# 2. Data Preparation
# ==========================================
print("Loading Tatoeba dataset...")
raw_dataset = load_dataset(
    config['dataset_name'],
    lang1=config['lang1'],
    lang2=config['lang2'],
    trust_remote_code=True
)['train'].select(range(40000))

dataset_dict = raw_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = dataset_dict['train']
eval_dataset = dataset_dict['test']

def formatting_prompts_func(examples):
    output_texts = []
    def format_single(translation_dict):
        src_text = translation_dict['en']
        tgt_text = translation_dict['fr']
        prompt = (f"Translate English to French.\n\n"
                  f"### English:\n{src_text}\n\n"
                  f"### French:\n{tgt_text}")
        return prompt

    if isinstance(examples['translation'], list):
        for translation_dict in examples['translation']:
            output_texts.append(format_single(translation_dict))
        return output_texts
    elif isinstance(examples['translation'], dict):
        return format_single(examples['translation'])
    else:
        raise ValueError(f"Unexpected format: {type(examples['translation'])}")
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
    
    eval_strategy="steps",
    eval_steps=config['eval_steps'],
    save_strategy="steps",
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

param_path = os.path.join(output_dir, "experiment_config.json")
experiment_log = {
    "model_name": config['model_name'],
    "dataset": config['dataset_name'],
    "hyperparameters": config
}
with open(param_path, "w") as f:
    json.dump(experiment_log, f, indent=4)

print(f"Training completed. Model saved to {output_dir}")