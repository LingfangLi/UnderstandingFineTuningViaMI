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

# ==========================================
# 1. 实验配置
# ==========================================
run_name = f"llama2-squad-qlora-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
output_dir = f"/mnt/scratch/users/sglli24/fine-tuning-project/fine_tuned_model/{run_name}"

config = {
    "model_name": "meta-llama/Llama-2-7b-hf",
    "dataset_name": "squad",
    # SQuAD 上下文较长，建议至少设为 1024，显存够的话可以用 2048
    "max_seq_length": 1024,
    "learning_rate": 2e-4,
    "batch_size": 4,
    "gradient_accumulation_steps": 2,  # 显存如果不够，可以调大这个，调小 batch_size
    "num_epochs": 2,  # SQuAD 数据量大(8万条)，跑 1 个 epoch 通常就够了，或者跑 2 个
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "use_4bit": True,
    "eval_steps": 100,
    "save_steps": 100,
    "logging_steps": 10,
}

# 初始化 WandB
wandb.init(
    project="FT-Llama2-SQuAD-QA",  # 新项目名
    name=run_name,
    config=config
)

# ==========================================
# 2. 数据准备
# ==========================================
print("Loading SQuAD dataset...")
# 加载前 40000 条 (和你之前的代码一致)
raw_dataset = load_dataset(config['dataset_name'], split='train').select(range(11000))

# 切分验证集 (Train 90% / Eval 10%)
dataset_dict = raw_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = dataset_dict['train']
eval_dataset = dataset_dict['test']

print(f"Train size: {len(train_dataset)} | Eval size: {len(eval_dataset)}")


# ---------------------------------------------------------
# 核心修改：适配 SQuAD 数据结构的格式化函数
# SQuAD 的 answers 是一个字典: {'text': ['Answer Text'], 'answer_start': [123]}
# ---------------------------------------------------------
def formatting_prompts_func(examples):
    # 内部函数：处理单条数据
    def format_single(context, question, answers):
        # 提取答案文本，SQuAD 的 answer 都在 text 列表的第一个位置
        ans_text = answers['text'][0]

        # 使用清晰的结构，帮助模型理解
        prompt = (f"### Context:\n{context}\n\n"
                  f"### Question:\n{question}\n\n"
                  f"### Answer:\n{ans_text}")  # 训练时包含答案
        return prompt

    # 处理批量 (Batch)
    if isinstance(examples['context'], list):
        output_texts = []
        # 同时遍历 Context, Question 和 Answers
        for ctx, q, ans in zip(examples['context'], examples['question'], examples['answers']):
            output_texts.append(format_single(ctx, q, ans))
        return output_texts

    # 处理单条 (Single)
    else:
        return format_single(examples['context'], examples['question'], examples['answers'])


# ==========================================
# 3. 模型加载 (QLoRA)
# ==========================================
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
tokenizer.padding_side = "right"  # 训练时右填充

model = prepare_model_for_kbit_training(model)

peft_config = LoraConfig(
    lora_alpha=config['lora_alpha'],
    lora_dropout=config['lora_dropout'],
    r=config['lora_r'],
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj"]
)

# ==========================================
# 4. 训练参数 (使用你验证过的最终版配置)
# ==========================================
training_arguments = SFTConfig(
    # SFT 参数
    max_length=config['max_seq_length'],  # 1024
    packing=False,

    # 基础参数
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

# ==========================================
# 5. 开始训练
# ==========================================
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    formatting_func=formatting_prompts_func,  # 使用新的 QA 格式化函数
    processing_class=tokenizer,  # 使用 processing_class
    args=training_arguments,
)

print(f"Starting QA training experiment: {run_name}")
trainer.train()

print(f"Saving BEST QA model to {output_dir}...")
trainer.model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

wandb.finish()
print("Done!")