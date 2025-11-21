import torch
from tqdm import tqdm
from datasets import load_dataset
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# ==========================================
# 1. 配置与路径
# ==========================================
base_model_name = "meta-llama/Llama-2-7b-hf"

# 🔴 替换为你训练好的 checkpoint 路径
# 例如: "/users/sglli24/fine-tuning-project/experiments/llama2-trans-en-fr-xxxx/checkpoint-200"
adapter_path = "/mnt/scratch/users/sglli24/fine-tuning-project/fine_tuned_model/llama2-tatoeba-en-fr-20251120-101824/checkpoint-4500/"

# 测试样本数 (使用训练集之外的数据)
START_IDX = 40000
NUM_SAMPLES = 1000

# ==========================================
# 2. 模型加载 (QLoRA)
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
# 关键：生成任务必须使用 Left Padding
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token

# ==========================================
# 3. 数据准备
# ==========================================
print("Loading Tatoeba dataset (Held-out test set)...")
# 加载 40000 之后的 1000 条数据，确保是模型没见过的
dataset = load_dataset('tatoeba', lang1="en", lang2="fr", trust_remote_code=True)['train'].select(
    range(START_IDX, START_IDX + NUM_SAMPLES))


def generate_prompt(english_text):
    # 必须严格匹配训练时的格式，但要在法语部分留白
    return (f"Translate English to French.\n\n"
            f"### English:\n{english_text}\n\n"
            f"### French:\n")


# ==========================================
# 4. 评估循环
# ==========================================
bleu_scores = []
smoothing = SmoothingFunction().method1  # 用于防止短句子得 0 分

print(f"Starting evaluation on {len(dataset)} samples...")

for i, sample in enumerate(tqdm(dataset)):
    # 提取源文本和标准译文
    src_text = sample['translation']['en']
    ref_text = sample['translation']['fr']

    # 1. 构造输入
    prompt = generate_prompt(src_text)
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to("cuda")

    # 2. 生成
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,  # 翻译句子通常不长
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True
        )

    # 3. 解码并提取
    # 只取新生成的部分
    generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
    pred_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # 后处理：去除首尾空格，取第一行（防止模型多此一举继续生成）
    pred_text = pred_text.strip().split('\n')[0]

    # 4. 计算 BLEU
    # NLTK 需要把 reference 放进列表的列表: [['Le', 'chat', ...]]
    # 这里简单使用 split() 分词，更严谨的话可以用 nltk.word_tokenize
    ref_tokens = [ref_text.split()]
    pred_tokens = pred_text.split()

    score = sentence_bleu(ref_tokens, pred_tokens, smoothing_function=smoothing)
    bleu_scores.append(score)

    # 调试打印 (前5个)
    if i < 5:
        print(f"\nSrc:  {src_text}")
        print(f"Ref:  {ref_text}")
        print(f"Pred: {pred_text}")
        print(f"BLEU: {score:.4f}")

# ==========================================
# 5. 最终结果
# ==========================================
avg_bleu = sum(bleu_scores) / len(bleu_scores)

print("\n" + "=" * 30)
print(f"EVALUATION RESULTS (N={NUM_SAMPLES})")
print("=" * 30)
print(f"Average BLEU Score: {avg_bleu:.4f}")
print("=" * 30)