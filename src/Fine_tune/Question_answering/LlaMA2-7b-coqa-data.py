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
run_name = f"llama2-coqa-qlora-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
output_dir = f"/mnt/scratch/users/sglli24/fine-tuning-project/fine_tuned_model/{run_name}"

config = {
    "model_name": "meta-llama/Llama-2-7b-hf",
    "dataset_name": "stanfordnlp/coqa",
    # CoQA 的故事通常比 SQuAD 长，建议设为 1024 或 2048
    "max_seq_length": 1024,
    "target_pairs": 15000,  # 15000对问答
    "learning_rate": 2e-4,
    "batch_size": 4,  # 如果显存不够改成 2
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

# 初始化 WandB
wandb.init(
    project="FT-Llama2-CoQA-Finetune",  # 新项目名
    name=run_name,
    config=config
)

# ==========================================
# 2. 数据准备与“打平”处理
# ==========================================
print("Loading CoQA dataset...")
# CoQA 的原始数据结构是一行对应一个故事+多个问题
raw_dataset = load_dataset(config['dataset_name'], split='train')

print("Flattening CoQA dataset (Story -> Multiple Q&A pairs)...")


# ... (前面的 import 和配置保持不变) ...

# ==========================================
# 核心修改 1: 带历史记录的打平函数
# ==========================================
def flatten_coqa_with_history(examples):
    new_contexts = []
    new_questions = []
    new_answers = []
    new_histories = []  # 新增：历史记录列

    # 遍历每一个故事
    for story, questions, answers in zip(examples['story'], examples['questions'], examples['answers']):
        history_buffer = []  # 用于暂存当前故事的对话历史

        # 遍历当前故事下的所有问题
        for q, a_text in zip(questions, answers['input_text']):
            # 1. 记录当前样本
            new_contexts.append(story)
            new_questions.append(q)
            new_answers.append(a_text)

            # 2. 生成截止到当前的历史记录字符串
            # 格式: "User: Q1\nAssistant: A1\nUser: Q2..."
            if len(history_buffer) == 0:
                history_str = "None"
            else:
                # 取最近的 5 轮对话，防止超出长度限制 (Max Seq Length)
                recent_history = history_buffer[-5:]
                history_str = "\n".join(recent_history)

            new_histories.append(history_str)

            # 3. 更新 Buffer，为下一个问题做准备
            # 这种格式比较节省 token
            history_buffer.append(f"User: {q}\nAssistant: {a_text}")

    return {
        "context": new_contexts,
        "question": new_questions,
        "answer": new_answers,
        "history": new_histories
    }


# 应用新的打平函数
print("Flattening CoQA dataset with HISTORY...")
flattened_dataset = raw_dataset.map(
    flatten_coqa_with_history,
    batched=True,
    remove_columns=raw_dataset.column_names
)

# 截取 40,000 条 (或者你可以先用 15,000 条试试)
flattened_dataset = flattened_dataset.select(range(min(len(flattened_dataset), 40000)))

# 切分验证集
dataset_dict = flattened_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = dataset_dict['train']
eval_dataset = dataset_dict['test']


# ==========================================
# 3. 格式化函数 (CoQA 修复版 - 支持 History)
# ==========================================
def formatting_prompts_func(examples):
    # 内部函数：定义单条数据的 Prompt 模板
    def format_single(context, history, question, answer):
        return (f"### Context:\n{context}\n\n"
                f"### Chat History:\n{history}\n\n"
                f"### Current Question:\n{question}\n\n"
                f"### Current Answer:\n{answer}")

    # [情况1]: 批量输入 (Batch) - 所有字段都是列表
    # TRL 训练时通常走这里
    if isinstance(examples['context'], list):
        output_texts = []
        # 使用 zip 同时遍历所有列
        for ctx, hist, q, ans in zip(examples['context'], examples['history'], examples['question'],
                                     examples['answer']):
            output_texts.append(format_single(ctx, hist, q, ans))
        return output_texts  # 返回列表

    # [情况2]: 单条输入 (Single) - 所有字段都是字符串/数值
    # TRL 内部检查或处理末尾数据时走这里
    else:
        ctx = examples['context']
        hist = examples['history']
        q = examples['question']
        ans = examples['answer']
        return format_single(ctx, hist, q, ans)  # 关键：直接返回字符串！

# ==========================================
# 4. 模型加载 (QLoRA)
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
# 5. 训练参数与启动
# ==========================================
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