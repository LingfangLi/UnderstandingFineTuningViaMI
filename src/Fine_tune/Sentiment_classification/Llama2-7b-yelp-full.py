import torch
import os
import wandb
import numpy as np
from datetime import datetime
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments
)
from trl import SFTConfig, SFTTrainer

os.environ["WANDB_PROJECT"] = "MI_llama2-yelp-full-finetune"

# 1. Experiment Configuration
run_name = f"llama2-yelp-full-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
output_base_dir = os.environ.get("OUTPUT_DIR", "./fine_tuned_model")
output_dir = os.path.join(output_base_dir, run_name)

config = {
    "model_name": "meta-llama/Llama-2-7b-hf",
    "dataset_name": "yelp_polarity",
    "max_seq_length": 256,
    "learning_rate": 2e-5,
    "batch_size": 8,
    "gradient_accumulation_steps": 1,
    "num_epochs": 2,
    "evaluation_strategy": "steps",
    "eval_steps": 50,
    "save_steps": 50,
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
        text = examples['text']
        label_raw = examples['label']
        return format_single_item(text, label_raw)

# 3. Model Loading (Full Fine-Tuning)
model = AutoModelForCausalLM.from_pretrained(
    config['model_name'],
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
    trust_remote_code=True
)
model.config.use_cache = False
model.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained(config['model_name'], trust_remote_code=True)
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
    optim="adamw_torch_fused",
    gradient_checkpointing=False,
    tf32=True,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="cosine",
)

# 5. Trainer Initialization (no peft_config)
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    formatting_func=formatting_prompts_func,
    processing_class=tokenizer,
    args=training_arguments,
)

# 6. Training and Saving
print(f"Starting full fine-tuning experiment: {run_name}")
trainer.train()

print(f"Saving FULL model to {output_dir}...")
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)

wandb.finish()
print("Done!")
