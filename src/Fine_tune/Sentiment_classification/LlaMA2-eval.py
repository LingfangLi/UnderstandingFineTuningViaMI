import torch
from tqdm import tqdm
from datasets import load_dataset
from peft import PeftModel, PeftConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)

# ==========================================
# 1. 配置路径与参数
# ==========================================
# 基础模型 (不需要改)
base_model_name = "meta-llama/Llama-2-7b-hf"

adapter_path = "/mnt/scratch/users/sglli24/fine-tuning-project/fine_tuned_model/llama2-yelp-qlora-20251119-154850/llama2-yelp/"

# 评估样本数
NUM_SAMPLES = 1000
MAX_LENGTH = 512
FINETUNED = False

# 定义你的 Label 映射 (必须与训练时完全一致)
TARGET_POSITIVE = "positive"
TARGET_NEGATIVE = "negative"

# ==========================================
# 2. 加载模型 (QLoRA 模式)
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
if FINETUNED:
    print(f"Loading fintuned model")
    print(f"Loading LoRA adapters from {adapter_path}...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
else:
    print("Load pretrained model")
    model =  base_model
        
model.eval()  # 切换到评估模式

# ==========================================
# 3. 加载 Tokenizer (关键修正)
# ==========================================
tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)

# ?? 关键修正：生成任务必须使用 Left Padding
# 如果用 Right Padding，模型会在 Prompt 后面拼命生成 Pad，导致无法生成 Label
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token


# ==========================================
# 4. 准备 Prompt 构造函数
# ==========================================
def generate_eval_prompt(text):
    # 注意：这里不加 Label，只给到 "Sentiment: " 后面，让模型补全
    # 必须保留训练时的格式，包括换行符
    return f"Review: {text}\nSentiment:"


# ==========================================
# 5. 评估循环
# ==========================================
# 加载测试集
dataset = load_dataset('yelp_polarity', split='test').select(range(NUM_SAMPLES))

correct_count = 0
total_count = 0

print(f"Starting evaluation on {NUM_SAMPLES} samples...")

# 使用 tqdm 显示进度条
for sample in tqdm(dataset):
    text = sample['text']
    label_idx = sample['label']  # 1 or 0

    # 1. 获取预期答案
    # 1 -> positive, 0 -> negative (与训练时保持一致)
    expected_label = TARGET_POSITIVE if label_idx == 1 else TARGET_NEGATIVE

    # 2. 构造输入
    prompt = generate_eval_prompt(text)
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to("cuda")

    # 3. 生成 (Inference)
    with torch.no_grad():
        # max_new_tokens=2 足够了，因为我们只需要 "positive" 或 "negative"
        outputs = model.generate(
            **inputs,
            max_new_tokens=5,  # 稍微多生成一点，防止有空格
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True
        )

    # 4. 解码与提取
    # 只解码新生成的部分
    generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # 5. 清洗与对比
    # 比如生成的可能是 " positive" 或 " Positive."，需要转小写并去空格
    pred_text = generated_text.strip().lower()
    print("text:::",text)
    print("Prediction:::",pred_text)
    print("label:::", expected_label)
    # 简单判断：只要生成的文本是以预期标签开头，就算对
    # 例如预期 "positive"，生成 "positive." -> True
    if expected_label in pred_text:
        correct_count += 1
    else:
        # 调试用：每100个错误打印一次，方便查看模型在瞎说什么
        if total_count % 100 == 0 and total_count > 0:
            pass
            # print(f"\n[Fail] Expect: {expected_label} | Got: '{pred_text}'")

    total_count += 1

# ==========================================
# 6. 输出结果
# ==========================================
accuracy = correct_count / total_count
print("-" * 30)
print(f"Total Samples: {total_count}")
print(f"Correct: {correct_count}")
print(f"Accuracy: {accuracy:.2%}")
print("-" * 30)