import torch
import os
import wandb
import numpy as np
from datetime import datetime
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer

# Experiment Configuration
run_name = f"llama2-kde4-tech-trans-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
output_dir = f"<MODEL_STORAGE>/fine-tuning-project/fine_tuned_model/{run_name}"

config = {
    "model_name": "meta-llama/Llama-2-7b-hf",
    "dataset_name": "kde4",
    "lang1": "en",
    "lang2": "fr",
    # KDE4 sentences are short; 256 tokens is sufficient
    "max_seq_length": 256,
    "target_samples": 30000,
    "learning_rate": 2e-4,
    "batch_size": 16,
    "gradient_accumulation_steps": 1,
    "num_epochs": 1,  # 1 epoch suffices for domain-specific data to avoid overfitting
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "use_4bit": True,
    "eval_steps": 200,
    "save_steps": 200,
    "logging_steps": 50,
}

wandb.init(
    project="FT-Llama2-KDE4-Tech-Trans",
    name=run_name,
    config=config
)

# Data Preparation
print("Loading KDE4 dataset...")
raw_dataset = load_dataset(
    config['dataset_name'],
    lang1=config['lang1'],
    lang2=config['lang2'],
    trust_remote_code=True
)['train']

# Select first 30,000 samples
if len(raw_dataset) > config['target_samples']:
    raw_dataset = raw_dataset.select(range(config['target_samples']))

# Split: 90% train / 10% validation
dataset_dict = raw_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = dataset_dict['train']
eval_dataset = dataset_dict['test']

print(f"Train size: {len(train_dataset)} | Eval size: {len(eval_dataset)}")


# KDE4 formatting function
# Structure: {'translation': {'en': '...', 'fr': '...'}}
def formatting_prompts_func(examples):
    output_texts = []

    def format_single(translation_dict):
        src_text = translation_dict['en']
        tgt_text = translation_dict['fr']

        # Uses "Technical" prefix to distinguish domain style
        prompt = (f"Translate Technical English to French.\n\n"
                  f"### Technical English:\n{src_text}\n\n"
                  f"### Technical French:\n{tgt_text}")
        return prompt

    # Batch input
    if isinstance(examples['translation'], list):
        for translation_dict in examples['translation']:
            output_texts.append(format_single(translation_dict))
        return output_texts

    # Single input
    elif isinstance(examples['translation'], dict):
        translation_dict = examples['translation']
        return format_single(translation_dict)

    else:
        raise ValueError(f"Unexpected format: {type(examples['translation'])}")


# Model Loading (QLoRA)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=config['use_4bit'],
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=False,
)

model = AutoModelForCausalLM.from_pretrained(
    config['model_name'],
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)
model.config.use_cache = False
model.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained(config['model_name'], trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = prepare_model_for_kbit_training(model)

peft_config = LoraConfig(
    lora_alpha=config['lora_alpha'],
    lora_dropout=config['lora_dropout'],
    r=config['lora_r'],
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj"]
)

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
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="cosine",
)

# Start Training
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    formatting_func=formatting_prompts_func,
    processing_class=tokenizer,
    args=training_arguments,
)

print(f"Starting KDE4 Technical Translation training: {run_name}")
trainer.train()

print(f"Saving BEST KDE4 model to {output_dir}...")
trainer.model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

wandb.finish()
print("Done!")