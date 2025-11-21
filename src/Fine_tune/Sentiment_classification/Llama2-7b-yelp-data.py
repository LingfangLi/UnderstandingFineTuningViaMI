import torch
import os
import wandb
import numpy as np
from datetime import datetime
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer
import os
os.environ["WANDB_PROJECT"] = "MI_llama2-yelp-finetune"
# ==========================================
# 1. 实验配置与参数管理 (解决问题 3)
# ==========================================
# 技巧：使用当前时间戳作为 run_name，防止覆盖
run_name = f"llama2-yelp-qlora-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
output_base_dir = "/mnt/scratch/users/sglli24/fine-tuning-project/fine_tuned_model/"
output_dir = os.path.join(output_base_dir, run_name)

# 将所有超参数定义在这里，WandB 会自动记录这些
config = {
    "model_name": "meta-llama/Llama-2-7b-hf",
    "dataset_name": "yelp_polarity",
    "max_seq_length": 256,
    "learning_rate": 2e-4,
    "batch_size": 4,
    "gradient_accumulation_steps": 1,
    "num_epochs": 2,  # 建议跑多一点，利用早停机制自动停止
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "use_4bit": True,
    "evaluation_strategy": "steps",
    "eval_steps": 50,  # 每50步评估一次
    "save_steps": 50,  # 每50步保存一次checkpoint
    "logging_steps": 10,
    "target_label_positive": "positive",  # 修正后的 label
    "target_label_negative": "negative"  # 修正后的 label
}

# ==========================================
# 2. 数据准备与切分 (解决问题 2 的前提)
# ==========================================
# 加载数据
raw_dataset = load_dataset(config['dataset_name'], split='train').select(range(11000))

# 关键：切分训练集和验证集 (Train/Test Split)
# 没有验证集就无法知道哪个模型最好
dataset_dict = raw_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = dataset_dict['train']
eval_dataset = dataset_dict['test']

print(f"Train size: {len(train_dataset)} | Eval size: {len(eval_dataset)}")


def formatting_prompts_func(examples):
    # 定义一个内部函数来处理单个样本的逻辑
    def format_single_item(text, label_raw):
        if label_raw == 1 or str(label_raw).lower() == "positive":
            label_str = config['target_label_positive']
        else:
            label_str = config['target_label_negative']
        return f"Review: {text}\nSentiment: {label_str}"

    # [情况1]: 如果输入是批量数据 (Batch) - text 是列表
    if isinstance(examples['text'], list):
        output_texts = []
        for text, label_raw in zip(examples['text'], examples['label']):
            output_texts.append(format_single_item(text, label_raw))
        return output_texts  # 返回列表
        
    # [情况2]: 如果输入是单条数据 (Single) - text 是字符串
    else:
        text = examples['text']
        label_raw = examples['label']
        return format_single_item(text, label_raw) # 返回字符串 (关键修正!)

# ==========================================
# 3. 模型与量化加载
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

# ==========================================
# 4. 自定义评估指标 (解决问题 2：如何定义“好”)
# ==========================================
# 生成模型计算准确率比较麻烦，通常看 eval_loss 越低越好
# 这里我们简单使用 Loss 作为指标。
# 如果一定要计算 Accuracy，需要 generate 之后对比，这会极大拖慢训练速度。
# 在微调阶段，Evaluation Loss 下降通常就代表效果变好。

# ==========================================
# 5. 训练参数配置
# ==========================================
# ==========================================
# 5. 训练参数配置 (SFTConfig)
# ==========================================
# 注意：max_seq_length 和 packing 现在必须写在这里面！
training_arguments = SFTConfig(
    # --- SFT 特有参数 (移到这里) ---
    max_length=config['max_seq_length'],  # <--- 关键：移到这里
    packing=False,  # <--- 关键：移到这里

    # --- 基础训练参数 ---
    output_dir=output_dir,
    report_to="wandb",
    run_name=run_name,
    eval_strategy=config['evaluation_strategy'],  # 注意：最新版推荐用 eval_strategy 而非 evaluation_strategy
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
    fp16=False,
    bf16=True,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="cosine",
)

# ==========================================
# 6. 训练器初始化
# ==========================================
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    formatting_func=formatting_prompts_func,

    # 关键修改 1: tokenizer 改名为 processing_class
    processing_class=tokenizer,

    # 关键修改 2: 传入包含 max_seq_length 的 config
    args=training_arguments,

    # 注意：不要在这里再传 max_seq_length 或 packing 了，
    # 它们已经被移到了上面的 SFTConfig 中。
)

# ==========================================
# 7. 训练与最终保存
# ==========================================
print(f"Starting training experiment: {run_name}")
trainer.train()

# 此时模型已经是 Best Model (因为 load_best_model_at_end=True)
print(f"Saving BEST model to {output_dir}...")
trainer.model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

wandb.finish()
print("Done!")