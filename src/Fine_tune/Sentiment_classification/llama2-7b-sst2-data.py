import torch
import os
import wandb
from datetime import datetime
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer

# 0. Environment Setup
os.environ["WANDB_PROJECT"] = "MI_llama2-sst2-qlora"

# 1. Experiment Configuration
run_name = f"llama2-7b-sst2-qlora-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
output_base_dir = "<MODEL_STORAGE>/fine-tuning-project/fine_tuned_model/" 
output_dir = os.path.join(output_base_dir, run_name)

config = {
    "model_name": "meta-llama/Llama-2-7b-hf",
    "dataset_name": "stanfordnlp/sst2",
    "max_seq_length": 256,
    "learning_rate": 2e-4,
    "batch_size": 16,
    "gradient_accumulation_steps": 1,
    "num_epochs": 2,            
    
    # LoRA parameters
    "lora_r": 64,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "use_4bit": True,
    
    # Training process parameters
    "evaluation_strategy": "steps",
    "eval_steps": 100,
    "save_steps": 100,
    "logging_steps": 10,
    
    # Label definitions
    "target_label_positive": "positive",
    "target_label_negative": "negative"
}

# 2. Data Preparation (SST-2)
train_dataset = load_dataset(config['dataset_name'], split='train').select(range(15000))
eval_dataset = load_dataset(config['dataset_name'], split='validation')

print(f"Train size: {len(train_dataset)} | Eval size: {len(eval_dataset)}")

def formatting_prompts_func(examples):
    def format_single_item(text, label_raw):
        # SST-2: 0 = negative, 1 = positive
        if label_raw == 1:
            label_str = config['target_label_positive']
        else:
            label_str = config['target_label_negative']
        
        return f"Review: {text}\nSentiment: {label_str}"

    # SST-2 text column is 'sentence'
    text_column = 'sentence'
    
    if isinstance(examples[text_column], list):
        output_texts = []
        for text, label_raw in zip(examples[text_column], examples['label']):
            output_texts.append(format_single_item(text, label_raw))
        return output_texts
    else:
        return format_single_item(examples[text_column], examples['label'])

# 3. Model Loading (QLoRA Setup)
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

# 4. Training Arguments
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
    
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="cosine",
)

# 5. Trainer Initialization
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    formatting_func=formatting_prompts_func,
    processing_class=tokenizer,
    args=training_arguments,
)

# 6. Training
print(f"Starting Llama-2-7b SST-2 QLoRA experiment: {run_name}")
trainer.train()

print(f"Saving ADAPTER model to {output_dir}...")
# In QLoRA mode, save_model saves only LoRA adapter weights
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)

wandb.finish()
print("Done!")