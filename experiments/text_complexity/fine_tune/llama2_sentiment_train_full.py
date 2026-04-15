"""Full fine-tuning of Llama-2-7B on a lexical-complexity-filtered subset of
Yelp Polarity. The active subset (`simple` or `complex`) is selected via the
EXPERIMENT_TYPE environment variable; default is `simple`.

Counterpart of: experiments/text_complexity/fine_tune/llama2_sentiment_train.py
(which is the original LoRA version).

Hyperparameters mirror src/Fine_tune/Sentiment_classification/Llama2-7b-yelp-full.py.
"""
import os
import json
from datetime import datetime

import torch
import wandb
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer
import transformers

_NEW_API = int(transformers.__version__.split('.')[0]) >= 5

# Configuration
EXPERIMENT_TYPE = os.environ.get("EXPERIMENT_TYPE", "simple")
assert EXPERIMENT_TYPE in ("simple", "complex"), f"EXPERIMENT_TYPE must be simple|complex, got {EXPERIMENT_TYPE}"

PROJECT_ROOT = "/users/sglli24/UnderstandingFineTuningViaMI"
OUTPUT_BASE_DIR = os.environ.get(
    "OUTPUT_BASE_DIR",
    "/mnt/scratch/users/sglli24/fine-tuning-project/fine_tuned_model/text_complexity",
)

config = {
    "model_name": "meta-llama/Llama-2-7b-hf",
    "task_name": "yelp",
    "dataset_name": "yelp_polarity",
    "experiment_type": EXPERIMENT_TYPE,
    "train_index_path": f"{PROJECT_ROOT}/experiments/text_complexity/matrix_analysis/yelp_lexically_{EXPERIMENT_TYPE}_subset_indices.txt",

    # Training params (mirror Llama2-7b-yelp-full.py)
    "max_seq_length": 256,
    "learning_rate": 2e-5,
    "batch_size": 8,
    "gradient_accumulation_steps": 1,
    "num_epochs": 2,
    "evaluation_strategy": "steps",
    "eval_steps": 50,
    "save_steps": 50,
    "logging_steps": 10,

    "target_label_positive": "positive",
    "target_label_negative": "negative",
}

run_name = f"llama2-7b-yelp-{EXPERIMENT_TYPE}-full-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
output_dir = os.path.join(OUTPUT_BASE_DIR, run_name)
os.environ.setdefault("WANDB_PROJECT", "MI_llama2-yelp-text_complexity-full")


def prepare_datasets():
    print(f"Loading {config['dataset_name']}...")
    dataset_lib = load_dataset(config['dataset_name'])

    print(f"Loading TRAIN indices from: {config['train_index_path']}")
    if not os.path.exists(config['train_index_path']):
        raise FileNotFoundError(config['train_index_path'])
    with open(config['train_index_path']) as f:
        train_indices = [int(line.strip()) for line in f if line.strip()]

    train_dataset = dataset_lib['train'].select(train_indices)
    eval_dataset = dataset_lib['test'].select(range(1000))

    print(f"Train size: {len(train_dataset)} (subset={EXPERIMENT_TYPE}) | Eval size: {len(eval_dataset)}")
    return train_dataset, eval_dataset


def formatting_prompts_func(examples):
    label_map = {0: config['target_label_negative'], 1: config['target_label_positive']}

    def format_single(text, label):
        clean = str(text).replace('\n', ' ').strip()
        return f"Review: {clean}\nSentiment: {label_map[int(label)]}"

    if isinstance(examples['text'], list):
        return [format_single(t, l) for t, l in zip(examples['text'], examples['label'])]
    return format_single(examples['text'], examples['label'])


def main():
    train_dataset, eval_dataset = prepare_datasets()

    # Full fine-tuning: no quantization, no LoRA
    _dtype_kwarg = {"dtype": torch.bfloat16} if _NEW_API else {"torch_dtype": torch.bfloat16}
    model = AutoModelForCausalLM.from_pretrained(
        config['model_name'],
        **_dtype_kwarg,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(config['model_name'], trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    training_args = SFTConfig(
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
        optim="adamw_torch_fused",
        gradient_checkpointing=False,
        tf32=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        formatting_func=formatting_prompts_func,
        processing_class=tokenizer,
        args=training_args,
    )

    print(f"Starting Llama-2-7B yelp-{EXPERIMENT_TYPE} full FT: {run_name}")
    trainer.train()

    print(f"Saving FULL model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    with open(os.path.join(output_dir, "experiment_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    wandb.finish()
    print("Done!")


if __name__ == "__main__":
    main()
