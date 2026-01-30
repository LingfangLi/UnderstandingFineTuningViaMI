import torch
import os
import wandb
from datetime import datetime
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from trl import SFTConfig, SFTTrainer

# 0. Environment Setup
os.environ["WANDB_PROJECT"] = "MI_gpt2-small-sst2"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 1. Experiment Configuration
run_name = f"gpt2-small-full-ft-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
output_base_dir = "./fine_tuned_model/"
output_dir = os.path.join(output_base_dir, run_name)

config = {
    "model_name": "gpt2",
    "dataset_name": "stanfordnlp/sst2",
    "max_seq_length": 512,      
    "learning_rate": 5e-5,
    "batch_size": 32,
    "gradient_accumulation_steps": 1,
    "num_epochs": 3,
    "evaluation_strategy": "steps",
    "eval_steps": 100,
    "save_steps": 100,
    "logging_steps": 10,
    "target_label_positive": "positive",
    "target_label_negative": "negative"
}

# 2. Data Preparation
train_dataset = load_dataset(config['dataset_name'], split='train').select(range(10000))
eval_dataset = load_dataset(config['dataset_name'], split='validation')

print(f"Train size: {len(train_dataset)} | Eval size: {len(eval_dataset)}")

def formatting_prompts_func(examples):
    def format_single_item(text, label_raw):
        label_str = config['target_label_positive'] if label_raw == 1 else config['target_label_negative']
        return f"Review: {text}\nSentiment: {label_str}"

    text_column = 'sentence' 
    
    if isinstance(examples[text_column], list):
        output_texts = []
        for text, label_raw in zip(examples[text_column], examples['label']):
            output_texts.append(format_single_item(text, label_raw))
        return output_texts
    else:
        return format_single_item(examples[text_column], examples['label'])

# 3. Model & Tokenizer
tokenizer = AutoTokenizer.from_pretrained(config['model_name'])

# GPT-2 has no native pad token; must set one for batch training
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print("Set tokenizer.pad_token to eos_token")

model = AutoModelForCausalLM.from_pretrained(
    config['model_name'],
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    device_map="auto"
)

# Sync model pad_token_id with tokenizer
model.config.pad_token_id = tokenizer.pad_token_id
model.config.use_cache = False

# 4. Training Arguments
training_arguments = SFTConfig(
    max_length=config['max_seq_length'],
    packing=False,

    output_dir=output_dir,
    report_to="wandb",
    run_name=run_name,
    eval_strategy=config['evaluation_strategy'],
    eval_steps=config['eval_steps'],
    save_strategy=config['evaluation_strategy'],
    save_steps=config['save_steps'],
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    save_total_limit=2,
    
    num_train_epochs=config['num_epochs'],
    per_device_train_batch_size=config['batch_size'],
    per_device_eval_batch_size=config['batch_size'],
    gradient_accumulation_steps=config['gradient_accumulation_steps'],
    learning_rate=config['learning_rate'],
    logging_steps=config['logging_steps'],
    
    # Precision config
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    
    max_grad_norm=1.0,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="cosine",
    dataset_text_field="text",
)

# 5. Trainer Initialization
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    formatting_func=formatting_prompts_func,
    processing_class=tokenizer,
    args=training_arguments,
)

# 6. Training
print(f"Starting GPT-2 Small FULL FINE-TUNING experiment: {run_name}")
trainer.train()

print(f"Saving BEST model to {output_dir}...")
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)

wandb.finish()
print("Done!")