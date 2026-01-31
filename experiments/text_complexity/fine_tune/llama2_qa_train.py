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
EXPERIMENT_TYPE = "complex"

CONFIG = {
    "project_name": "Llama2-SQuAD-FineTuning-MI",
    "model_name": "meta-llama/Llama-2-7b-hf",
    "task_name": "squad",

    # Paths
    "base_save_dir": "<MODEL_STORAGE>/fine-tuning-project/fine_tuned_model/text_complexity",
    "data_index": f"<PROJECT_ROOT>/experiments/text_complexity/matrix_analysis/squad_lexically_{EXPERIMENT_TYPE}_subset_indices.txt",
    "test_data_index": f"<PROJECT_ROOT>/experiments/text_complexity/matrix_analysis/squad_lexically_{EXPERIMENT_TYPE}_test_indices.txt",

    # Training parameters
    "num_epochs": 8 if EXPERIMENT_TYPE == "simple" else 10,
    "batch_size": 4,
    "grad_accum_steps": 4, # Effective batch = 16
    "learning_rate": 2e-4,
    "max_seq_length": 1024,

    # LoRA Config
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
}

run_name = f"Llama2-SQuAD-{EXPERIMENT_TYPE}"
output_dir = os.path.join(CONFIG['base_save_dir'], run_name)
os.environ["WANDB_PROJECT"] = CONFIG["project_name"]

# Data preparation
def prepare_datasets():
    print(f"Loading SQuAD dataset...")
    # Load SQuAD training set
    full_train = load_dataset("squad", split="train")
    
    # Load training set indices
    print(f"Loading indices from: {CONFIG['data_index']}")
    if not os.path.exists(CONFIG['data_index']):
        raise FileNotFoundError(f"Index file not found: {CONFIG['data_index']}")

    with open(CONFIG['data_index'], 'r') as f:
        index_list = [int(line.strip()) for line in f]
    
    with open(CONFIG['test_data_index'], 'r') as f:
        test_index_list = [int(line.strip()) for line in f]    
    
    train_dataset = full_train.select(index_list)

    eval_dataset = full_train.select(test_index_list)
    
    print(f"Train size: {len(train_dataset)} (Subset: {EXPERIMENT_TYPE})")
    print(f"Eval size:  {len(eval_dataset)} (Fixed Validation Set)")
    
    return train_dataset, eval_dataset

# Prompt formatting (QA format)
def formatting_prompts_func(examples):
    def format_single(context, question, answers):
        # Use first answer from SQuAD's answer list
        ans_text = answers['text'][0] if len(answers['text']) > 0 else ""
        prompt = (f"### Context:\n{context}\n\n"
                  f"### Question:\n{question}\n\n"
                  f"### Answer:\n{ans_text}")
        return prompt

    # Process batch
    if isinstance(examples['context'], list):
        output_texts = []
        for ctx, q, ans in zip(examples['context'], examples['question'], examples['answers']):
            output_texts.append(format_single(ctx, q, ans))
        return output_texts
    # Process single
    else:
        return format_single(examples['context'], examples['question'], examples['answers'])

# Main training logic
def main():
    # 1. Load Data
    train_dataset, eval_dataset = prepare_datasets()
    
    # 2. Load Model (4-bit Quantization for memory efficiency)
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

    # 4. PEFT Config
    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
        lora_alpha=CONFIG['lora_alpha'],
        lora_dropout=CONFIG['lora_dropout'],
        r=CONFIG['lora_r'],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
    )

    # 5. Training Arguments
    training_args = SFTConfig(
        output_dir=output_dir,
        run_name=run_name,
        
        # SFT Specific
        max_length=CONFIG['max_seq_length'],
        packing=False, # Disabled to preserve context-question integrity
        
        # Training Params
        num_train_epochs=CONFIG['num_epochs'],
        per_device_train_batch_size=CONFIG['batch_size'],
        per_device_eval_batch_size=CONFIG['batch_size'],
        gradient_accumulation_steps=CONFIG['grad_accum_steps'],
        learning_rate=CONFIG['learning_rate'],
        
        # Eval & Save
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        logging_steps=20,
        
        # Best Model Strategy
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

    # 7. Start Training
    print(f"Starting training for {CONFIG['num_epochs']} epochs on {EXPERIMENT_TYPE} dataset...")
    trainer.train()

    # 8. Save Final Model
    print(f"Saving final adapter to {output_dir}")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save Config
    with open(os.path.join(output_dir, "experiment_config.json"), "w") as f:
        json.dump(CONFIG, f, indent=4)

    wandb.finish()

if __name__ == "__main__":
    main()