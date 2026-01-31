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

# Experiment Configuration
run_name = f"qwen2-0.5b-tatoeba-en-fr-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
output_dir = f"<MODEL_STORAGE>/fine-tuning-project/fine_tuned_model/{run_name}"

config = {
    "model_name": "Qwen/Qwen2-0.5B",
    "dataset_name": "tatoeba",
    "lang1": "en",
    "lang2": "fr",
    "max_seq_length": 256,
    "learning_rate": 2e-5,
    "batch_size": 32,
    "gradient_accumulation_steps": 1,
    "num_epochs": 3,
    "eval_steps": 200,
    "save_steps": 200,
    "logging_steps": 50,
}

wandb.init(project="MI-Qwen2-Translation-TATOEBA", name=run_name, config=config)

# Data Preparation
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

# Model Loading (Full Fine-tuning)
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

# Training Arguments
training_arguments = SFTConfig(
    max_length=config['max_seq_length'],
    packing=False,
    output_dir=output_dir,
    report_to="wandb",
    run_name=run_name,
    eval_strategy="steps",
    eval_steps=config['eval_steps'],
    save_strategy="steps",
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

# Start Training (Full Fine-tuning)
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    formatting_func=formatting_prompts_func,
    processing_class=tokenizer,
    args=training_arguments,
)

print(f"Starting Qwen2 Translation FULL FT: {run_name}")
trainer.train()

print(f"Saving BEST model to {output_dir}...")
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
wandb.finish()