import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
import collections
import re
import string
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# ==========================================
# 1. 配置与路径
# ==========================================
base_model_name = "meta-llama/Llama-2-7b-hf"

# 🔴 修改为你微调保存的路径 (例如 checkpoint-100)
adapter_path = "/mnt/scratch/users/sglli24/fine-tuning-project/fine_tuned_model/llama2-squad-qlora-20251119-214256/checkpoint-2476/"

# 评估样本数 (你要求的 1000)
NUM_SAMPLES = 1000
MAX_LENGTH = 1024  # 与训练保持一致


# ==========================================
# 2. 辅助函数：标准化与指标计算 (SQuAD 标准)
# ==========================================
def normalize_text(s):
    """移除冠词、标点，转小写，用于更宽松的匹配"""

    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_exact_match(prediction, truth):
    return int(normalize_text(prediction) == normalize_text(truth))


def compute_f1(prediction, truth):
    pred_tokens = normalize_text(prediction).split()
    truth_tokens = normalize_text(truth).split()

    # 计算共同词汇
    common = collections.Counter(pred_tokens) & collections.Counter(truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0

    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


# ==========================================
# 3. 模型加载
# ==========================================
print("Loading base model in 4-bit...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

print(f"Loading Adapter from: {adapter_path}")
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.padding_side = "left"  # 生成任务必须左填充
tokenizer.pad_token = tokenizer.eos_token

# ==========================================
# 4. 数据准备 (确保数据隔离)
# ==========================================
print("Preparing dataset...")
# 加载原始数据 (与训练时一致)
raw_dataset = load_dataset('squad', split='train').select(range(20000,30000))  # 取一部分作为评估池

# 关键：使用相同的 seed 进行切分，确保取出的是训练时没见过的 "test" 部分
dataset_dict = raw_dataset.train_test_split(test_size=0.1, seed=42)
eval_dataset = dataset_dict['test'].select(range(NUM_SAMPLES))  # 取前1000个

print(f"Evaluating on {len(eval_dataset)} unseen samples.")


def generate_prompt(context, question):
    # 格式必须严格匹配训练脚本 (不包含答案部分)
    return (f"### Context:\n{context}\n\n"
            f"### Question:\n{question}\n\n"
            f"### Answer:\n")


# ==========================================
# 5. 评估循环
# ==========================================
em_scores = []
f1_scores = []
bleu_scores = []
smoothing = SmoothingFunction().method1

print("Starting generation...")
for i, sample in enumerate(tqdm(eval_dataset)):
    context = sample['context']
    question = sample['question']
    # SQuAD 的答案是一个列表，包含多个可能的正确答案
    gold_answers = sample['answers']['text']

    # 1. 构造 Prompt
    prompt = generate_prompt(context, question)
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to("cuda")

    # 2. 生成
    with torch.no_grad():
        # max_new_tokens 控制生成的长度，SQuAD 答案通常不长，50足够
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True
        )

    # 3. 解码并提取答案
    # 只取生成的部分
    generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
    pred_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # 去除可能的首尾空白
    pred_text = pred_text.strip()
    # 如果模型生成了换行符（可能想开始下一题），截断它
    pred_text = pred_text.split('\n')[0]

    # 4. 计算指标 (取与任一标准答案匹配的最高分)
    sample_em = max(compute_exact_match(pred_text, gold) for gold in gold_answers)
    sample_f1 = max(compute_f1(pred_text, gold) for gold in gold_answers)

    # BLEU (你要求的)
    # BLEU 期望 reference 是 list of list of tokens
    ref_tokens_list = [ans.split() for ans in gold_answers]
    sample_bleu = sentence_bleu(ref_tokens_list, pred_text.split(), smoothing_function=smoothing)

    em_scores.append(sample_em)
    f1_scores.append(sample_f1)
    bleu_scores.append(sample_bleu)

    # 调试打印 (前5个)
    if i < 5:
        print(f"\nQ: {question}")
        print(f"Pred: {pred_text}")
        print(f"Gold: {gold_answers}")
        print(f"Scores -> EM: {sample_em}, F1: {sample_f1:.2f}, BLEU: {sample_bleu:.2f}")

# ==========================================
# 6. 最终统计
# ==========================================
avg_em = np.mean(em_scores) * 100
avg_f1 = np.mean(f1_scores) * 100
avg_bleu = np.mean(bleu_scores)  # BLEU 通常是 0-1

print("\n" + "=" * 30)
print(f"EVALUATION RESULTS (N={len(eval_dataset)})")
print("=" * 30)
print(f"Exact Match (EM): {avg_em:.2f}%")
print(f"F1 Score:         {avg_f1:.2f}%")
print(f"BLEU Score:       {avg_bleu:.4f}")
print("=" * 30)