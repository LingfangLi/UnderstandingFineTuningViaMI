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
# Note: Full fine-tuning no longer requires peft and bitsandbytes configuration classes
from trl import SFTConfig, SFTTrainer

os.environ["WANDB_PROJECT"] = "MI_qwen2-0.5b-yelp-full" # Project name

# ==========================================
# 1. Experiment Configuration & Parameter Management
# ==========================================
run_name = f"qwen2-0.5b-full-ft-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
output_base_dir = "/mnt/scratch/users/sglli24/fine-tuning-project/fine_tuned_model/"
output_dir = os.path.join(output_base_dir, run_name)

config = {
    "model_name": "Qwen/Qwen2-0.5B", # Replace with Qwen2-0.5B
    "dataset_name": "yelp_polarity",
    "max_seq_length": 512,  # 0.5B runs fast with sufficient VRAM, length can be increased appropriately
    "learning_rate": 2e-5,  # [Key] Full fine-tuning LR is usually smaller than LoRA (2e-5 or 1e-5)
    "batch_size": 16,       # [Key] 0.5B model is small, per-device batch size can be larger (e.g., 16 or 32)
    "gradient_accumulation_steps": 1,
    "num_epochs": 3,        # Full fine-tuning usually converges fast, but small models might need more epochs
    "evaluation_strategy": "steps",
    "eval_steps": 100,
    "save_steps": 100,
    "logging_steps": 10,
    "target_label_positive": "positive",
    "target_label_negative": "negative"
}

# ==========================================
# 2. Data Preparation & Splitting
# ==========================================
# Since the model runs fast, data size can be increased, or kept at 11000
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
        # Qwen can also use this simple format, or add Qwen's Chat template,
        # but to maintain experimental consistency, following the previous completion format is fine.
        return f"Review: {text}\nSentiment: {label_str}"

    if isinstance(examples['text'], list):
        output_texts = []
        for text, label_raw in zip(examples['text'], examples['label']):
            output_texts.append(format_single_item(text, label_raw))
        return output_texts
    else:
        return format_single_item(examples['text'], examples['label'])

# ==========================================
# 3. Model Loading (Key Change: Remove quantization, use BF16)
# ==========================================
# Remove BitsAndBytesConfig, load BF16 directly
model = AutoModelForCausalLM.from_pretrained(
    config['model_name'],
    dtype=torch.bfloat16, # Use BF16 full precision
    device_map="auto",
    trust_remote_code=True
)

model.config.use_cache = False
# model.config.pretraining_tp = 1 # Qwen usually doesn't need this line, specific to Llama

tokenizer = AutoTokenizer.from_pretrained(config['model_name'], trust_remote_code=True)

# Qwen2 padding handling
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# [Key] Remove prepare_model_for_kbit_training because we are not doing k-bit training
# [Key] Remove peft_config because we are doing full fine-tuning

# ==========================================
# 5. Training Arguments Configuration (SFTConfig)
# ==========================================
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
    
    # Precision settings
    fp16=False,
    bf16=True, # Keep BF16
    
    max_grad_norm=1.0, # Full fine-tuning can sometimes relax gradient clipping, or keep it at 0.3
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="cosine",
)

# ==========================================
# 6. Trainer Initialization (Remove PEFT)
# ==========================================
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    # [Key] Remove peft_config=peft_config,
    formatting_func=formatting_prompts_func,
    processing_class=tokenizer,
    args=training_arguments,
)

# ==========================================
# 7. Training & Final Saving
# ==========================================
print(f"Starting Qwen2-0.5B FULL FINE-TUNING experiment: {run_name}")
trainer.train()

print(f"Saving BEST model to {output_dir}...")
# Full fine-tuning saves the complete model, not an Adapter
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)

wandb.finish()
print("Done!")