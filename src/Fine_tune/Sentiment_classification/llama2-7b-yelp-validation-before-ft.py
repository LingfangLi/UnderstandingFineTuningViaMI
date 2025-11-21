import torch
from transformers import AutoTokenizer


def check_tokenization_alignment():
    model_name = "meta-llama/Llama-2-7b-hf"

    # 1. 模拟微调时的 Tokenizer (通常 Right Padding)
    train_tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_tokenizer.pad_token = train_tokenizer.eos_token
    train_tokenizer.padding_side = "right"  # 训练标准配置

    # 2. 模拟 EAP 分析时的 Tokenizer (TLens 通常行为，或你需要设置的行为)
    # EAP 经常需要 Left Padding 以便对齐最后一个 token
    analysis_tokenizer = AutoTokenizer.from_pretrained(model_name)
    analysis_tokenizer.pad_token = analysis_tokenizer.eos_token
    analysis_tokenizer.padding_side = "left"

    # 3. 构造样本数据
    # 注意：检查你的 dataset['label'] 到底是 "Positive" 字符串还是数字 1/0
    # 这里假设你已经把 label 转换成了文本，因为 Llama 是生成模型
    text = "The food was amazing"
    label = "Positive"

    # 你的格式化字符串 (务必与训练脚本保持字节级一致)
    prompt = f"Review: {text}\nSentiment: {label}"

    print(f"Testing Prompt: '{prompt}'")
    print("-" * 40)

    # ==========================================
    # 检查 A: BOS Token 是否存在
    # ==========================================
    train_enc = train_tokenizer(prompt)["input_ids"]
    print(f"Train Tokens (IDs): {train_enc}")
    print(f"Train Tokens (Str): {train_tokenizer.convert_ids_to_tokens(train_enc)}")

    has_bos = (train_enc[0] == train_tokenizer.bos_token_id)
    print(f"? Train implies BOS: {has_bos}")

    # ==========================================
    # 检查 B: Label 前的空格 (关键!)
    # ==========================================
    # 让我们看看 label "Positive" 在句子末尾变成了什么 ID
    label_id = train_enc[-1]
    decoded_label = train_tokenizer.decode([label_id])
    print(f"Last Token ID: {label_id} | Decoded: '{decoded_label}'")

    if decoded_label.strip() != label:
        print("?? 警告: 最后一个 Token 不是完整的 Label。可能 Label 被拆分了，或者没有加 EOS。")
        # 比如 "Positive" 可能被拆成 "Pos" + "itive"
    else:
        print(f"? Label Token完整性检查通过")

    # ==========================================
    # 检查 C: 分析脚本的对齐 (Padding)
    # ==========================================
    # 模拟一个 Batch，包含一个短句子和一个长句子
    batch_texts = [prompt, f"Review: short\nSentiment: {label}"]

    # 训练时的处理 (Right Padding)
    train_batch = train_tokenizer(batch_texts, padding=True, return_tensors="pt")
    # 分析时的处理 (Left Padding)
    analysis_batch = analysis_tokenizer(batch_texts, padding=True, return_tensors="pt")

    print("\n[Padding Check]")
    print(f"Train Batch (Right Pad) Last Col: {train_batch['input_ids'][:, -1].tolist()}")
    print(f"Analysis Batch (Left Pad) Last Col: {analysis_batch['input_ids'][:, -1].tolist()}")

    # 验证 EAP 脚本逻辑
    # 你的 EAP 脚本用 logits[:, -1]。如果用 Right Padding，短句子的最后一个 token 是 PAD
    last_token_is_pad = (train_batch['input_ids'][1, -1] == train_tokenizer.pad_token_id)
    if last_token_is_pad:
        print(
            "? 警告: 训练使用 Right Padding。在 EAP 分析脚本中，必须使用 Left Padding，否则 logits[:, -1] 会读到 PAD token！")
    else:
        print("? Padding 检查通过")


if __name__ == "__main__":
    check_tokenization_alignment()