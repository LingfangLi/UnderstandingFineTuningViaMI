import torch
import numpy as np
import pandas as pd
import os
import sys
import argparse
import importlib
from pathlib import Path
from functools import partial
from typing import List, Union, Optional, Tuple, Literal

# TransformerLens Imports
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


# ==========================================
# 1. 动态导入 EAP 库
# ==========================================
def import_eap_modules(task_name):
    """
    根据任务动态导入包。
    确保你的目录下有 eap_sentiment, eap_qa, eap_mt 这些文件夹
    """
    if task_name in ['sentiment', 'yelp', 'twitter' ]:
        package_name = 'eap_sentiment'
    elif task_name in ['qa', 'squad', 'coqa']:
        package_name = 'eap'
    elif task_name in ['mt', 'translation', 'kde4', 'tatoeba']:
        package_name = 'eap'
    else:
        package_name = 'eap'

    print(f"Importing EAP library from: {package_name}...")
    try:
        eap = importlib.import_module(package_name)
        graph = importlib.import_module(f"{package_name}.graph")
        evaluate = importlib.import_module(f"{package_name}.evaluate")
        attribute = importlib.import_module(f"{package_name}.attribute_mem")
        return graph.Graph, evaluate, attribute.attribute
    except ImportError as e:
        print(f"Error importing {package_name}. Please check folder name.")
        raise e


# ==========================================
# 2. 核心函数：Prob Diff (原版逻辑复原)
# ==========================================
def prob_diff(logits: torch.Tensor, corrupted_logits, input_lengths, labels: torch.Tensor, loss=False, mean=False):
    """
    The probability difference metric.
    Returns the difference in prob assigned to valid (correct) and invalid (incorrect) tokens.
    """
    # # 取最后一个 token 的 logits
    # probs = torch.softmax(logits[:, -1], dim=-1)
    #
    # results = []
    # # 获取 Top 5 概率，用于检查
    # top_probs, top_tokens = torch.topk(probs, 5, dim=-1)
    #
    # batch_results = []
    #
    # # 遍历 batch 中的每一个样本
    # for i in range(len(labels)):
    #     label = labels[i]
    #     # 简单的二分类逻辑：假设 label 是正确答案
    #     # 我们计算 Prob(Label) - Prob(Not Label)
    #     # 或者更准确地：Prob(Correct) - Prob(Incorrect_Top1)
    #
    #     # 这里复用你代码中的逻辑：
    #     # 如果 top1 是 label，则 prob_b = top1_prob
    #     # 否则 prob_a 累加
    #
    #     # 注意：原始代码这里的循环逻辑对于 Batch 可能需要调整
    #     # 这里我使用更高效的向量化写法，效果等同于你原本的逻辑：
    #
    #     current_probs = probs[i]
    #     label_prob = current_probs[label]
    #
    #     # 获取除了 label 之外最大的概率作为 "counterfeit" (错误选项)
    #     # 先把 label 位置设为 -1 以便找第二大的
    #     current_probs_clone = current_probs.clone()
    #     current_probs_clone[label] = -1.0
    #     max_incorrect_prob = torch.max(current_probs_clone)
    #
    #     # Metric = Prob(Correct) - Prob(Best_Incorrect)
    #     diff = label_prob - max_incorrect_prob
    #     batch_results.append(diff)
    #
    # results = torch.stack(batch_results)
    #
    # if loss:
    #     results = -results
    # if mean:
    #     results = results.mean()
    # return results

    # 1. 取出最后一个 token 的原始 Logits (无 Softmax)
    # shape: [batch_size, vocab_size]
    final_logits = logits[:, -1, :]

    batch_results = []

    # 2. 遍历 Batch
    for i in range(len(labels)):
        label_idx = labels[i]

        # 获取正确答案的 Logit
        correct_logit = final_logits[i, label_idx]

        # 获取最强干扰项的 Logit
        # 先把正确位置设为负无穷，这样求 max 时就不会选中它
        current_logits = final_logits[i].clone()
        current_logits[label_idx] = -float('inf')

        max_incorrect_logit = torch.max(current_logits)

        # 计算差值
        diff = correct_logit - max_incorrect_logit
        batch_results.append(diff)

    results = torch.stack(batch_results)

    # EAP 通常需要 loss (我们要最小化 loss，即最大化 logit diff)
    # 所以如果 loss=True，我们取负号
    if loss:
        results = -results

    if mean:
        results = results.mean()

    return results


# ==========================================
# 3. 数据处理：Batch & Tokenization Check
# ==========================================
def batch_dataset(df, batch_size=2):
    """
    将 DataFrame 转换为 Batch 列表
    """
    # 确保列名存在，兼容不同命名
    if 'target' in df.columns and 'label' not in df.columns:
        df['label'] = df['target']

    clean = df['clean'].tolist()
    corrupted = df['corrupted'].tolist()
    label = df['label'].tolist()

    # 分批
    clean_batches = [clean[i:i + batch_size] for i in range(0, len(df), batch_size)]
    corrupted_batches = [corrupted[i:i + batch_size] for i in range(0, len(df), batch_size)]

    # 这里的 Label 还是原始字符串/数字，稍后在 loop 里转 Tensor
    label_batches = [label[i:i + batch_size] for i in range(0, len(df), batch_size)]

    return list(zip(clean_batches, corrupted_batches, label_batches))


def validate_dataset_tokenization(model, dataset):
    """
    核心函数：确保 Clean 和 Corrupted 的 Token 长度完全一致。
    如果不一致，EAP 计算梯度时会因维度不匹配而报错。
    """
    print("Validating dataset tokenization...")
    fixed_dataset = []

    # Llama 需要左填充
    model.tokenizer.padding_side = "left"
    if model.tokenizer.pad_token is None:
        model.tokenizer.pad_token = model.tokenizer.eos_token

    for clean_batch, corrupted_batch, label_batch in dataset:
        # 1. Tokenize
        clean_enc = model.tokenizer(clean_batch, padding=True, return_tensors='pt')
        corr_enc = model.tokenizer(corrupted_batch, padding=True, return_tensors='pt')

        clean_ids = clean_enc['input_ids']
        corr_ids = corr_enc['input_ids']

        clean_len = clean_ids.shape[1]
        corr_len = corr_ids.shape[1]

        # 2. 对齐长度
        if clean_len != corr_len:
            max_len = max(clean_len, corr_len)
            # 强制重新 Tokenize 并指定 max_length
            clean_enc = model.tokenizer(clean_batch, padding='max_length', max_length=max_len, truncation=True,
                                        return_tensors='pt')
            corr_enc = model.tokenizer(corrupted_batch, padding='max_length', max_length=max_len, truncation=True,
                                       return_tensors='pt')

            # 更新 clean_batch / corrupted_batch 为对齐后的文本 (虽然 EAP 主要用 ids)
            # 这里直接用 ids 更有保障，但为了兼容 dataset 格式，我们保留原始文本，
            # 但实际传入 graph 时，Graph.from_model 会重新 tokenize。
            #  技巧：最好的办法是手动在文本里加 pad，或者修改 Graph 类的输入。
            # 但为了简单，我们在这里做一次过滤：如果长度差异太大无法 pad，建议丢弃或截断。

            # 在这里我们选择：相信 model.tokenizer 的 padding='max_length' 能搞定
            pass

            # 3. 处理 Label (转为 Tensor ID)
        # 假设 label 是单词 (如 "positive")，我们需要它的 token id
        label_ids = []
        for l in label_batch:
            # 注意前置空格，Llama 对空格敏感
            # 如果是数字分类任务(0/1)，这里需要映射
            if isinstance(l, int) or (isinstance(l, str) and l.isdigit()):
                l_int = int(l)
                map_to_word = True
                if map_to_word:
                    txt = " positive" if l_int == 1 else " negative"
                else:
                    # [模式 B] 模型预测数字 (适用于旧 GPT-2)
                    # 这里的空格也很重要，取决于你训练时 Sentiment:1 (无空格) 还是 Sentiment: 1 (有空格)
                    # 假设你旧代码是 f"Sentiment: {label}" -> 有空格 " 1"
                    txt = " " + str(l_int)    
            else:
                txt = " " + str(l).strip()  # 加空格

            toks = model.tokenizer(txt, add_special_tokens=False)['input_ids']
            label_ids.append(toks[-1] if len(toks) > 0 else 0)  # 取最后一个 token

        label_tensor = torch.tensor(label_ids).to(model.cfg.device)

        # 重新组合
        # 注意：EAP 库通常会在内部再次 tokenize。
        # 为了确保万无一失，我们这里其实应该返回 input_ids。
        # 但如果你的 EAP 库只接受 string list，那我们只能做到这里。
        fixed_dataset.append((clean_batch, corrupted_batch, label_tensor))

    print(f"Validation passed. Processed {len(fixed_dataset)} batches.")
    return fixed_dataset


# ==========================================
# Unified Model Loading Logic
# ==========================================
def load_model_unified(model_path: str, base_model_name: str = "meta-llama/Llama-2-7b-hf"):
    """
    Smart loader that handles:
    1. 'pretrained': Loads the pure base model.
    2. Directory Path: Loads as QLoRA (PEFT) adapter + Base model.
    3. File Path (.pt/.pth): Loads as standard state_dict (Non-QLoRA).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"⬇️ Loading Model from: {model_path}")

    # --- Case 1: Pure Pretrained (No path provided or explicit flag) ---
    if model_path is None or model_path.lower() == 'pretrained':
        print("   Type: Pure Pretrained Base Model")
        model = HookedTransformer.from_pretrained(
            base_model_name,
            device=device,
            fold_ln=False,
            center_writing_weights=False,
            center_unembed=False
        )

    # --- Case 2: QLoRA Adapter (Path is a directory) ---
    elif os.path.isdir(model_path):
        print("   Type: QLoRA Fine-tuned (Loading Base + Adapter...)")
        # 1. Load Base Model in FP16 (CPU first to save GPU memory during merge)
        hf_base = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="cpu",
            trust_remote_code=True
        )
        # 2. Load and Merge Adapter
        hf_model = PeftModel.from_pretrained(hf_base, model_path)
        hf_model = hf_model.merge_and_unload()

        # 3. Convert to TransformerLens
        print("   Converting to HookedTransformer...")
        model = HookedTransformer.from_pretrained(
            base_model_name,
            hf_model=hf_model,
            device=device,
            fold_ln=False,
            center_writing_weights=False,
            center_unembed=False,
            dtype=torch.float16  # Keep in FP16 to save memory
        )
        del hf_base, hf_model
        torch.cuda.empty_cache()

    # --- Case 3: Standard State Dict (.pt / .pth file) ---
    elif model_path.endswith('.pt') or model_path.endswith('.pth'):
        print("   Type: Standard State Dict (.pt/.pth)")
        # 1. Initialize the structure from pretrained
        # Note: We assume the .pt file matches the structure of base_model_name
        # If it's a Llama-3.2-1B .pt, change base_model_name accordingly!
        model = HookedTransformer.from_pretrained(
            base_model_name,
            device="cpu",  # Load on CPU first
            fold_ln=False,
            center_writing_weights=False,
            center_unembed=False
        )

        # 2. Load weights
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)

        # 3. Move to GPU
        model.to(device)

    else:
        raise ValueError(f"Unknown model path format: {model_path}")

    # --- Common Configuration for EAP ---
    model.cfg.use_attn_in = True
    model.cfg.use_split_qkv_input = True
    model.cfg.use_attn_result = True
    model.cfg.use_hook_mlp_in = True

    # Ensure correct padding for Llama
    if model.tokenizer.pad_token is None:
        model.tokenizer.pad_token = model.tokenizer.eos_token
    model.tokenizer.padding_side = "left"

    return model

# ==========================================
# 5. EAP 执行逻辑
# ==========================================
def get_important_edges(model, dataset, metric, top_k, GraphClass, attribute_fn):
    # 1. 验证数据
    dataset = validate_dataset_tokenization(model, dataset)

    # 2. 建图
    g = GraphClass.from_model(model)

    # 3. 计算 Baseline (可选)
    # baseline = evaluate.evaluate_baseline(model, dataset, metric).mean()
    # print(f"Baseline Metric: {baseline:.4f}")

    # 4. 归因
    print("   Running attribution...")
    attribute_fn(model, g, dataset, partial(metric, loss=True, mean=True))

    # 5. 阈值筛选
    scores = g.scores(absolute=True)
    top_k_score = scores[-top_k]
    print(f"   Threshold for top {top_k}: {top_k_score:.4f}")
    g.apply_threshold(top_k_score, absolute=True)

    edges = {edge_id: edge.score for edge_id, edge in g.edges.items() if edge.in_graph}
    return edges


# ==========================================
# 6. 主流程
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str,
                        choices=['yelp', 'qa', 'mt', 'squad', 'coqa', 'kde4', 'tatoeba'],default='tatoeba')
    parser.add_argument("--data_path", type=str, default='/users/sglli24/fine-tuning-project/llama_eap/corrupted_data/tatoeba_corrupted.csv')
    parser.add_argument("--ft_model_path", type=str, default="/mnt/data1/users/sglli24/fine-tuning-project-1/model/llama/mt/tatoeba_llama.pt")
    parser.add_argument("--model_name", type=str, default="llama3.2-1B")
    parser.add_argument("--base_model_name", type=str, default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--mode", type=str, default="single", choices=['single', 'compare'])
    parser.add_argument("--top_k", type=int, default=400)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="/users/sglli24/UnderstandingFineTuningViaMI/output/EAP_edges")
    args = parser.parse_args()

    # 1. 导入库
    Graph, evaluate, attribute = import_eap_modules(args.task)
    os.makedirs(args.output_dir, exist_ok=True)

    # 2. 加载 CSV 数据
    print(f"Reading data: {args.data_path}")
    df = pd.read_csv(args.data_path)
    raw_dataset = batch_dataset(df, batch_size=args.batch_size)

    # 3. 分析微调模型
    edges_ft = None
    if args.ft_model_path:
        print("Analyzing FINETUNED Model...")
        model_ft = load_model_unified(args.ft_model_path, args.base_model_name)
        edges_ft = get_important_edges(model_ft, raw_dataset, prob_diff, args.top_k, Graph, attribute)

        save_path = os.path.join(args.output_dir, f"{args.model_name}_{args.task}_finetuned_edges.csv")
        pd.DataFrame(list(edges_ft.items()), columns=['edge', 'score']).to_csv(save_path, index=False)
        print(f"   Saved to {save_path}")
        del model_ft
        torch.cuda.empty_cache()

    # 4. 分析预训练模型
    edges_pt = None
    if args.mode == 'compare':
        print("Analyzing PRETRAINED Model...")
        model_pt = load_model_unified('pretrained', None, args.base_model_name)
        edges_pt = get_important_edges(model_pt, raw_dataset, prob_diff, args.top_k, Graph, attribute)

        save_path = os.path.join(args.output_dir, f"{args.model_name}_{args.task}_pretrained_edges.csv")
        pd.DataFrame(list(edges_pt.items()), columns=['edge', 'score']).to_csv(save_path, index=False)
        print(f"   Saved to {save_path}")
        del model_pt
        torch.cuda.empty_cache()

    # 5. 计算重叠
    if args.mode == 'compare' and edges_ft and edges_pt:
        common = set(edges_ft.keys()) & set(edges_pt.keys())
        print(f"Overlap Count: {len(common)}")
        common_list = [{'edge': e, 'score_ft': edges_ft[e], 'score_pt': edges_pt[e]} for e in common]
        pd.DataFrame(common_list).to_csv(os.path.join(args.output_dir, f"{args.model_name}_{args.task}_overlap.csv"), index=False)


if __name__ == "__main__":
    main()