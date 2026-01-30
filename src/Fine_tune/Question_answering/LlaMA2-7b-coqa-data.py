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

# 1. Experiment Configuration
run_name = f"llama2-coqa-qlora-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
output_dir = f"/mnt/scratch/users/sglli24/fine-tuning-project/fine_tuned_model/{run_name}"

config = {
    "model_name": "meta-llama/Llama-2-7b-hf",
    "dataset_name": "stanfordnlp/coqa",
    "max_seq_length": 1024,
    "target_pairs": 15000,
    "learning_rate": 2e-4,
    "batch_size": 4,
    "gradient_accumulation_steps": 2,
    "num_epochs": 1,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "use_4bit": True,
    "eval_steps": 100,
    "save_steps": 100,
    "logging_steps": 20,
}

wandb.init(
    project="FT-Llama2-CoQA-Finetune",
    name=run_name,
    config=config
)

# 2. Data Preparation (Flatten CoQA)
print("Loading CoQA dataset...")
raw_dataset = load_dataset(config['dataset_name'], split='train')

print("Flattening CoQA dataset (Story -> Multiple Q&A pairs)...")

def flatten_coqa_with_history(examples):
    """Flatten CoQA stories into individual Q&A pairs with conversation history."""
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
                # Keep last 5 turns to stay within sequence length limits
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


# 3. Formatting Function (CoQA with History)
def formatting_prompts_func(examples):
    """Format CoQA examples into prompt strings with conversation history."""
    def format_single(context, history, question, answer):
        return (f"### Context:\n{context}\n\n"
                f"### Chat History:\n{history}\n\n"
                f"### Current Question:\n{question}\n\n"
                f"### Current Answer:\n{answer}")

    # Batch input: all fields are lists
    if isinstance(examples['context'], list):
        output_texts = []
        for ctx, hist, q, ans in zip(examples['context'], examples['history'], examples['question'],
                                     examples['answer']):
            output_texts.append(format_single(ctx, hist, q, ans))
        return output_texts

    # Single input: all fields are strings
    else:
        ctx = examples['context']
        hist = examples['history']
        q = examples['question']
        ans = examples['answer']
        return format_single(ctx, hist, q, ans)

# 4. Model Loading (QLoRA)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=config['use_4bit'],
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
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

# 5. Training Arguments & Launch
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

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    formatting_func=formatting_prompts_func,
    processing_class=tokenizer,
    args=training_arguments,
)

print(f"Starting CoQA training experiment: {run_name}")
trainer.train()

print(f"Saving BEST CoQA model to {output_dir}...")
trainer.model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

wandb.finish()
print("Done!")