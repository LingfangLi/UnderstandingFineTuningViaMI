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

# 1. Experiment Configuration
run_name = f"qwen2-0.5b-squad-full-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
output_dir = f"<MODEL_STORAGE>/fine-tuning-project/fine_tuned_model/{run_name}"

config = {
    "model_name": "Qwen/Qwen2-0.5B",
    "dataset_name": "squad",
    "max_seq_length": 1024,
    "learning_rate": 2e-5,
    "batch_size": 16,
    "gradient_accumulation_steps": 2,
    "num_epochs": 2,
    "eval_steps": 100,
    "save_steps": 100,
    "logging_steps": 10,
}

wandb.init(project="MI-Qwen2-SQuAD-QA", name=run_name, config=config)

# 2. Data Preparation
raw_dataset = load_dataset(config['dataset_name'], split='train').select(range(11000))
dataset_dict = raw_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = dataset_dict['train']
eval_dataset = dataset_dict['test']

def formatting_prompts_func(examples):
    def format_single(context, question, answers):
        ans_text = answers['text'][0]
        prompt = (f"### Context:\n{context}\n\n"
                  f"### Question:\n{question}\n\n"
                  f"### Answer:\n{ans_text}")
        return prompt
    
    if isinstance(examples['context'], list):
        output_texts = []
        for ctx, q, ans in zip(examples['context'], examples['question'], examples['answers']):
            output_texts.append(format_single(ctx, q, ans))
        return output_texts
    else:
        return format_single(examples['context'], examples['question'], examples['answers'])

# 3. Model Loading (Full Fine-tuning)
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
    eval_strategy="steps",
    eval_steps=config['eval_steps'],
    save_strategy="steps",
    save_steps=config['save_steps'],
    load_best_model_at_end=True,
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

# 5. Start Training (Full Fine-tuning)
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    formatting_func=formatting_prompts_func,
    processing_class=tokenizer,
    args=training_arguments,
)

print(f"Starting Qwen2 SQuAD FULL FT: {run_name}")
trainer.train()

print(f"Saving model to {output_dir}...")
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
wandb.finish()