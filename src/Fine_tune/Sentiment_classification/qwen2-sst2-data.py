import torch
import torch
import os
import wandb
from datetime import datetime
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

# 1. Environment & Configuration
os.environ["WANDB_PROJECT"] = "MI_Qwen2_SST2"
MODEL_ID = "Qwen/Qwen2-0.5B"  # or "Qwen/Qwen2.5-0.5B"

run_name = f"qwen2-0.5b-sst2-full-{datetime.now().strftime('%Y%m%d-%H%M')}"
output_dir = os.path.join("<MODEL_STORAGE>/fine-tuning-project/fine_tuned_model/", run_name)

config = {
    "max_seq_length": 512,
    "learning_rate": 2e-5,
    "batch_size": 32,
    "gradient_accumulation_steps": 1,
    "num_epochs": 3,
    "target_label_positive": "positive",
    "target_label_negative": "negative",
    "evaluation_strategy": "steps",
    "eval_steps": 100,
    "save_steps": 100,
}

# 2. Data Preparation
raw_dataset = load_dataset("stanfordnlp/sst2", split='train').select(range(15000))
dataset_dict = raw_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = dataset_dict['train']
eval_dataset   = dataset_dict['test']

def formatting_prompts_func(examples):
    def format_single(text, label):
        lbl = config['target_label_positive'] if label == 1 else config['target_label_negative']
            return f"Review: {text}\nSentiment: {lbl}"

    if isinstance(examples['sentence'], list):
        return [format_single(t, l) for t, l in zip(examples['sentence'], examples['label'])]
    return format_single(examples['sentence'], examples['label'])

# 3. Model & Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    device_map="auto",
    trust_remote_code=True
)
model.config.use_cache = False
model.config.pad_token_id = tokenizer.pad_token_id

# 4. Training Arguments
training_args = SFTConfig(
    output_dir=output_dir,
    run_name=run_name,
    max_length=config['max_seq_length'],
    packing=False,

    per_device_train_batch_size=config['batch_size'],
    per_device_eval_batch_size=config['batch_size'],
    gradient_accumulation_steps=config['gradient_accumulation_steps'],

    learning_rate=config['learning_rate'],
    num_train_epochs=config['num_epochs'],

    eval_strategy=config['evaluation_strategy'],
    eval_steps=config['eval_steps'],
    save_steps=config['save_steps'],
    save_strategy=config['evaluation_strategy'],
    greater_is_better=False,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    logging_steps=10,

    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),

    group_by_length=True,
    report_to="wandb"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    formatting_func=formatting_prompts_func,
    processing_class=tokenizer,
    args=training_args
)

print("Starting Qwen2-0.5B Training...")
trainer.train()
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
print("Qwen2 Done!")