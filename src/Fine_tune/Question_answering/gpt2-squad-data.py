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

# 0. Environment Setup
os.environ["WANDB_PROJECT"] = "MI_gpt2-small-SQuAD-QA"

# 1. Experiment Configuration

TASK_TYPE = "squad" 

run_name = f"gpt2-small-{TASK_TYPE}-full-ft-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
output_base_dir = "<MODEL_STORAGE>/fine-tuning-project/fine_tuned_model/"
output_dir = os.path.join(output_base_dir, run_name)

config = {
    "model_name": "gpt2",
    "dataset_name": "squad" if TASK_TYPE == "squad" else "stanfordnlp/coqa",
    "task_type": TASK_TYPE,
    "learning_rate": 2e-5,
    "max_seq_length": 1024,
    "batch_size": 8,            
    "gradient_accumulation_steps": 4, 
    "num_epochs": 3,
    "eval_steps": 500,
    "save_steps": 500,
    "logging_steps": 50,
}

# 2. Data Preparation
raw_dataset = load_dataset(config['dataset_name'], split='train')
if config['dataset_name'] == "squad":
    pass 
dataset_dict = raw_dataset.train_test_split(test_size=0.05, seed=42)
train_dataset = dataset_dict['train']
eval_dataset = dataset_dict['test']

def format_squad(examples):
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
def format_coqa(examples):
    output_texts = []
    questions = examples['questions']
    answers = examples['answers']
    stories = examples['story']
    
    for i in range(len(stories)):
        story = stories[i]
        q_list = questions[i]
        a_list = answers[i]['input_text']
        
        history_str = ""
        for q, a in zip(q_list, a_list):
            prompt = f"Context: {story}\n{history_str}Question: {q}\nAnswer: {a}"
            output_texts.append(prompt)
            history_str += f"Question: {q}\nAnswer: {a}\n"
    return output_texts

formatting_func = format_squad if TASK_TYPE == "squad" else format_coqa

# 3. Model & Tokenizer
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

# 4. Training Arguments
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
    dataset_text_field="text",
)

# 5. Trainer Initialization
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    formatting_func=formatting_func,
    processing_class=tokenizer,
    args=training_arguments,
)

# 6. Training & Saving
print(f"Starting training for {run_name}...")
trainer.train()
trainer.save_model(output_dir)

param_path = os.path.join(output_dir, "experiment_config.json")
experiment_log = {
    "model_name": config['model_name'],
    "dataset": config['dataset_name'],
    "task_type": config['task_type'],
    "hyperparameters": config
}
with open(param_path, "w") as f:
    json.dump(experiment_log, f, indent=4)

print(f"Training completed. Model saved to {output_dir}")