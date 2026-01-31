import torch
import os
import wandb
from datetime import datetime
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import SFTConfig, SFTTrainer

os.environ["WANDB_PROJECT"] = "MI_qwen2-0.5b-yelp-full"

# 1. Experiment Configuration
run_name = f"qwen2-0.5b-full-ft-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
output_base_dir = "<MODEL_STORAGE>/fine-tuning-project/fine_tuned_model/"
output_dir = os.path.join(output_base_dir, run_name)

config = {
    "model_name": "Qwen/Qwen2-0.5B",
    "dataset_name": "yelp_polarity",
    "max_seq_length": 512,
    "learning_rate": 2e-5,
    "batch_size": 16,
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
        return f"Review: {text}\nSentiment: {label_str}"

    if isinstance(examples['text'], list):
        output_texts = []
        for text, label_raw in zip(examples['text'], examples['label']):
            output_texts.append(format_single_item(text, label_raw))
        return output_texts
    else:
        return format_single_item(examples['text'], examples['label'])

# 3. Model Loading
model = AutoModelForCausalLM.from_pretrained(
    config['model_name'],
    dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

model.config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(config['model_name'], trust_remote_code=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

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
    
    fp16=False,
    bf16=True,
    
    max_grad_norm=1.0,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="cosine",
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

# 6. Training & Saving
print(f"Starting Qwen2-0.5B FULL FINE-TUNING experiment: {run_name}")
trainer.train()

print(f"Saving BEST model to {output_dir}...")
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)

wandb.finish()
print("Done!")