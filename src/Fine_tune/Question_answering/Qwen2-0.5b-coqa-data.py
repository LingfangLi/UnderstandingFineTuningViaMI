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

run_name = f"qwen2-0.5b-coqa-full-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
output_dir = f"/mnt/scratch/users/sglli24/fine-tuning-project/fine_tuned_model/{run_name}"

config = {
    "model_name": "Qwen/Qwen2-0.5B",
    "dataset_name": "stanfordnlp/coqa",
    "max_seq_length": 1024, 
    "learning_rate": 2e-5,  
    "batch_size": 8,       
    "gradient_accumulation_steps": 2,
    "num_epochs": 2,
    "eval_steps": 100,
    "save_steps": 100,
    "logging_steps": 20,
}

wandb.init(project="MI-Qwen2-CoQA-Finetune", name=run_name, config=config)

print("Loading CoQA dataset...")
raw_dataset = load_dataset(config['dataset_name'], split='train')

def flatten_coqa_with_history(examples):
    new_contexts = []
    new_questions = []
    new_answers = []
    new_histories = [] 
    for story, questions, answers in zip(examples['story'], examples['questions'], examples['answers']):
        history_buffer = [] 
        for q, a_text in zip(questions, answers['input_text']):
            new_contexts.append(story)
            new_questions.append(q)
            new_answers.append(a_text)
            
            if len(history_buffer) == 0:
                history_str = "None"
            else:
                recent_history = history_buffer[-5:]
                history_str = "\n".join(recent_history)
            new_histories.append(history_str)
            history_buffer.append(f"User: {q}\nAssistant: {a_text}")

    return {
        "context": new_contexts,
        "question": new_questions,
        "answer": new_answers,
        "history": new_histories
    }

print("Flattening CoQA dataset with HISTORY...")
flattened_dataset = raw_dataset.map(
    flatten_coqa_with_history,
    batched=True,
    remove_columns=raw_dataset.column_names
)

flattened_dataset = flattened_dataset.select(range(min(len(flattened_dataset), 40000)))
dataset_dict = flattened_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = dataset_dict['train']
eval_dataset = dataset_dict['test']


def formatting_prompts_func(examples):
    def format_single(context, history, question, answer):
        return (f"### Context:\n{context}\n\n"
                f"### Chat History:\n{history}\n\n"
                f"### Current Question:\n{question}\n\n"
                f"### Current Answer:\n{answer}")

    if isinstance(examples['context'], list):
        output_texts = []
        for ctx, hist, q, ans in zip(examples['context'], examples['history'], examples['question'], examples['answer']):
            output_texts.append(format_single(ctx, hist, q, ans))
        return output_texts 
    else:
        return format_single(examples['context'], examples['history'], examples['question'], examples['answer'])

# Model Loading
model = AutoModelForCausalLM.from_pretrained(
    config['model_name'],
    dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
model.config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(config['model_name'], trust_remote_code=True)
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
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

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    formatting_func=formatting_prompts_func,
    processing_class=tokenizer,
    args=training_arguments,
)

print(f"Starting Qwen2 CoQA FULL FT: {run_name}")
trainer.train()
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
wandb.finish()