import torch
import os
import wandb
import numpy as np
from datetime import datetime
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from trl import SFTConfig, SFTTrainer

# Experiment Configuration
run_name = f"llama2-tatoeba-full-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
output_base_dir = os.environ.get("OUTPUT_DIR", "./fine_tuned_model")
output_dir = os.path.join(output_base_dir, run_name)

config = {
    "model_name": "meta-llama/Llama-2-7b-hf",
    "dataset_name": "tatoeba",
    "lang1": "en",
    "lang2": "fr",
    "max_seq_length": 256,
    "learning_rate": 2e-5,
    "batch_size": 4,
    "gradient_accumulation_steps": 1,
    "num_epochs": 1,
    "eval_steps": 200,
    "save_steps": 200,
    "logging_steps": 50,
}

wandb.init(
    project="FT-Llama2-Translation-TATOEBA-Full",
    name=run_name,
    config=config
)

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

print(f"Train size: {len(train_dataset)} | Eval size: {len(eval_dataset)}")


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
        translation_dict = examples['translation']
        return format_single(translation_dict)
    else:
        raise ValueError(f"Unexpected format for 'translation' column: {type(examples['translation'])}")

# Model Loading (Full Fine-Tuning)
model = AutoModelForCausalLM.from_pretrained(
    config['model_name'],
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
model.config.use_cache = False
model.config.pretraining_tp = 1
model.gradient_checkpointing_enable()

tokenizer = AutoTokenizer.from_pretrained(config['model_name'], trust_remote_code=True)
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
    optim="adamw_bnb_8bit",
    gradient_checkpointing=True,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="cosine",
)

# Start Training (no peft_config)
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    formatting_func=formatting_prompts_func,
    processing_class=tokenizer,
    args=training_arguments,
)

print(f"Starting Tatoeba full fine-tuning: {run_name}")
trainer.train()

print(f"Saving FULL Tatoeba model to {output_dir}...")
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)

wandb.finish()
print("Done!")
