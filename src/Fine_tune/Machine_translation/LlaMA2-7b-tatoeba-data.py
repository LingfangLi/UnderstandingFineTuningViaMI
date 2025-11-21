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
# 使用时间戳防止文件夹重名
run_name = f"llama2-tatoeba-en-fr-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
output_dir = f"/mnt/scratch/users/sglli24/fine-tuning-project/fine_tuned_model/{run_name}"

config = {
    "model_name": "meta-llama/Llama-2-7b-hf",
    "dataset_name": "tatoeba",
    "lang1": "en",
    "lang2": "fr",
    "max_seq_length": 256,  # 翻译句对通常较短，256 足够，也能节省显存
    "learning_rate": 2e-4,
    "batch_size": 8,  # 句子短，Batch Size 可以开大一点
    "gradient_accumulation_steps": 1,
    "num_epochs": 1,  # Tatoeba 数据也不少，1个 epoch 足够看效果
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "use_4bit": True,
    "eval_steps": 200,  # 每200步评估一次
    "save_steps": 200,
    "logging_steps": 50,
}

# 初始化 WandB
wandb.init(
    project="FT-Llama2-Translation-TATOEBA",  # 项目名称
    name=run_name,
    config=config
)

# ==========================================
# 2. 数据准备
# ==========================================
print("Loading Tatoeba dataset...")
# 加载 Tatoeba 数据集 (注意 trust_remote_code=True)
raw_dataset = load_dataset(
    config['dataset_name'],
    lang1=config['lang1'],
    lang2=config['lang2'],
    trust_remote_code=True
)['train'].select(range(40000))  # 限制前4万条，和你之前的设置一致

# 切分验证集 (90% 训练 / 10% 验证)
dataset_dict = raw_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = dataset_dict['train']
eval_dataset = dataset_dict['test']

print(f"Train size: {len(train_dataset)} | Eval size: {len(eval_dataset)}")


# ---------------------------------------------------------
# 核心：机器翻译的格式化函数
# Tatoeba 的数据结构是：{'translation': {'en': 'Hello', 'fr': 'Bonjour'}}
# SFTTrainer 传入的 batch 是：{'translation': [{'en':..., 'fr':...}, ...]}
# ---------------------------------------------------------
def formatting_prompts_func(examples):
    output_texts = []

    # 内部函数：处理单条翻译字典
    def format_single(translation_dict):
        # 这里直接处理 {'en': '...', 'fr': '...'}
        src_text = translation_dict['en']
        tgt_text = translation_dict['fr']

        prompt = (f"Translate English to French.\n\n"
                  f"### English:\n{src_text}\n\n"
                  f"### French:\n{tgt_text}")
        return prompt

    # [情况1]: 批量输入 (Batch)
    # 如果 translation 是个列表，说明里面包着很多个字典
    if isinstance(examples['translation'], list):
        # 遍历列表中的每一个字典
        for translation_dict in examples['translation']:
            output_texts.append(format_single(translation_dict))
        return output_texts

    # [情况2]: 单条输入 (Single)
    # 如果 translation 是个字典，说明就是单条数据
    elif isinstance(examples['translation'], dict):
        translation_dict = examples['translation']
        return format_single(translation_dict)  # 直接返回字符串

    else:
        # 防御性编程：万一有什么奇怪的情况
        raise ValueError(f"Unexpected format for 'translation' column: {type(examples['translation'])}")

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
# 4. 训练参数 (兼容 TRL 新版本)
# ==========================================
training_arguments = SFTConfig(
    # SFT 特有参数
    max_length=config['max_seq_length'],
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
    formatting_func=formatting_prompts_func,  # 使用翻译格式化函数
    processing_class=tokenizer,  # TRL 新版参数
    args=training_arguments,
)

print(f"Starting Translation training experiment: {run_name}")
trainer.train()

print(f"Saving BEST Translation model to {output_dir}...")
trainer.model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

wandb.finish()
print("Done!")