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
import transformers
_NEW_API = int(transformers.__version__.split('.')[0]) >= 5

# 1. Experiment Configuration
run_name = f"llama2-squad-full-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
output_base_dir = os.environ.get("OUTPUT_DIR", "./fine_tuned_model")
output_dir = os.path.join(output_base_dir, run_name)

config = {
    "model_name": "meta-llama/Llama-2-7b-hf",
    "dataset_name": "squad",
    "max_seq_length": 1024,
    "learning_rate": 2e-5,
    "batch_size": 4,
    "gradient_accumulation_steps": 2,
    "num_epochs": 2,
    "eval_steps": 100,
    "save_steps": 100,
    "logging_steps": 10,
}

wandb.init(
    project="FT-Llama2-SQuAD-Full",
    name=run_name,
    config=config
)

# 2. Data Preparation
print("Loading SQuAD dataset...")
raw_dataset = load_dataset(config['dataset_name'], split='train').select(range(11000))

dataset_dict = raw_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = dataset_dict['train']
eval_dataset = dataset_dict['test']

print(f"Train size: {len(train_dataset)} | Eval size: {len(eval_dataset)}")


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


# 3. Model Loading (Full Fine-Tuning)
_dtype_kwarg = {"dtype": torch.bfloat16} if _NEW_API else {"torch_dtype": torch.bfloat16}
model = AutoModelForCausalLM.from_pretrained(
    config['model_name'],
    **_dtype_kwarg,
    device_map="auto",
    trust_remote_code=True
)
model.config.use_cache = False
model.config.pretraining_tp = 1
model.gradient_checkpointing_enable()

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
    optim="adamw_torch_fused",
    gradient_checkpointing=True,
    tf32=True,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
)

# 5. Start Training (no peft_config)
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    formatting_func=formatting_prompts_func,
    processing_class=tokenizer,
    args=training_arguments,
)

print(f"Starting QA full fine-tuning: {run_name}")
trainer.train()

print(f"Saving FULL QA model to {output_dir}...")
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)

wandb.finish()
print("Done!")
