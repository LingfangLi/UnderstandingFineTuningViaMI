import torch
import os
import wandb
import json
from datetime import datetime
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer

# Configuration
EXPERIMENT_TYPE = "simple"

CONFIG = {
    "project_name": "Llama2-Yelp-FineTuning-MI",
    "model_name": "meta-llama/Llama-2-7b-hf",
    "task_name": "yelp",

    # Paths
    "base_save_dir": "/mnt/scratch/users/sglli24/fine-tuning-project/fine_tuned_model/text_complexity",
    "train_index_path": f"/users/sglli24/UnderstandingFineTuningViaMI/experiments/text_complexity/matrix_analysis/yelp_lexically_{EXPERIMENT_TYPE}_subset_indices.txt",

    # Training parameters
    "learning_rate": 2e-4,
    "batch_size": 4,
    "grad_accum_steps": 4, # Effective batch = 16
    "num_epochs": 1,
    "max_seq_length": 256,

    # LoRA Config
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
}

run_name = f"Llama2-Yelp-{EXPERIMENT_TYPE}-{datetime.now().strftime('%m%d-%H%M')}"
output_dir = os.path.join(CONFIG['base_save_dir'], run_name)
os.environ["WANDB_PROJECT"] = CONFIG["project_name"]

# Data preparation
def prepare_datasets():
    print(f"Loading full Yelp Polarity dataset...")
    # Load dataset
    dataset_lib = load_dataset("yelp_polarity")
    
    # Load training set indices
    print(f"Loading TRAIN indices from: {CONFIG['train_index_path']}")
    if not os.path.exists(CONFIG['train_index_path']):
        raise FileNotFoundError(f"Train index file not found: {CONFIG['train_index_path']}")

    with open(CONFIG['train_index_path'], 'r') as f:
        train_indices = [int(line.strip()) for line in f]
    
    train_dataset = dataset_lib['train'].select(train_indices)
        
    eval_dataset = dataset_lib['test'].select(range(1000))
    
    print(f"Train size: {len(train_dataset)} | Eval size: {len(eval_dataset)}")
    return train_dataset, eval_dataset

# Prompt formatting
def formatting_prompts_func(examples):
    # Handle batch or single item
    if isinstance(examples['text'], list):
        texts = examples['text']
        labels = examples['label']
    else:
        texts = [examples['text']]
        labels = [examples['label']]
    
    output_texts = []
    
    # Label mapping (Yelp: 0->Negative, 1->Positive)
    label_map = {0: "Negative", 1: "Positive"}
    
    for text, label in zip(texts, labels):
        # Clean text
        clean_text = text.replace('\n', ' ').strip()
        label_str = label_map[label]
        
        # Construct Prompt
        prompt = f"Review: {clean_text}\nSentiment: {label_str}"
        output_texts.append(prompt)
            
    # SFTTrainer expects list for batch input, str for single
    if isinstance(examples['text'], list):
        return output_texts
    else:
        return output_texts[0]

# Main training logic
def main():
    # 1. Prepare data
    train_dataset, eval_dataset = prepare_datasets()
    
    # 2. Load model (4-bit quantization)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False,
    )

    model = AutoModelForCausalLM.from_pretrained(
        CONFIG['model_name'],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    model.config.use_cache = False
    
    # 3. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'], trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # Right padding for SFT training

    # 4. PEFT (LoRA) configuration
    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
        lora_alpha=CONFIG['lora_alpha'],
        lora_dropout=CONFIG['lora_dropout'],
        r=CONFIG['lora_r'],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    # 5. Training arguments
    training_args = SFTConfig(
        output_dir=output_dir,
        run_name=run_name,
        
        # SFT parameters
        max_length=CONFIG['max_seq_length'],
        packing=False, 
        
        # Basic parameters
        num_train_epochs=CONFIG['num_epochs'],
        per_device_train_batch_size=CONFIG['batch_size'],
        per_device_eval_batch_size=CONFIG['batch_size'],
        gradient_accumulation_steps=CONFIG['grad_accum_steps'],
        learning_rate=CONFIG['learning_rate'],
        
        # Evaluation and saving
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        logging_steps=20,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        
        fp16=False,
        bf16=True,
        group_by_length=True,
    )

    # 6. Initialize Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        formatting_func=formatting_prompts_func,
        processing_class=tokenizer,
        args=training_args,
    )

    # 7. Start training
    print(f"Starting training experiment: {run_name}")
    trainer.train()

    # 8. Save model
    print(f"Saving final model to {output_dir}")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save configuration
    with open(os.path.join(output_dir, "experiment_config.json"), "w") as f:
        json.dump(CONFIG, f, indent=4)

    wandb.finish()

if __name__ == "__main__":
    main()