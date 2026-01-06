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
import eap
from eap.graph import Graph
from eap import evaluate
from eap import attribute_mem as attribute
# TransformerLens Imports
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ==========================================
# 2. 核心函数：Prob Diff (原版逻辑复原)
# ==========================================
def prob_diff(logits: torch.Tensor, corrupted_logits, input_lengths, labels: torch.Tensor, loss=False, mean=False):
    """
    The probability difference metric.
    Returns the difference in prob assigned to valid (correct) and invalid (incorrect) tokens.
    """
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


def validate_dataset_tokenization(model, dataset, task):
    """
    策略：
    1. 使用 Tokenizer 将文本强制截断 (Truncation) 到安全长度 (防止 OOM)。
    2. 将截断后的 Token ID '解码' (Decode) 回字符串。
    3. 将这些短字符串传给 attribute_mem.py，既满足了它的类型要求，又限制了显存。
    """
    print("Validating dataset tokenization...")
    fixed_dataset = []

    model.tokenizer.padding_side = "left"
    if model.tokenizer.pad_token is None:
        model.tokenizer.pad_token = model.tokenizer.eos_token

    # 【关键设置】安全长度上限
    # A100 80G 在开启 QKV 分离的情况下，建议设为 600-800。
    # 如果依然 OOM，请调小这个数值 (例如 512)。
    #SAFE_MAX_LEN = 800 
    #print(f"Pre-processing: Truncating text to max {SAFE_MAX_LEN} tokens to prevent OOM.")

    #for clean_batch, corrupted_batch, label_batch in dataset:
        # 1. Tokenize 并强制截断 (得到 Tensor)
    #    clean_enc = model.tokenizer(
    #        clean_batch, 
    #        truncation=True, 
    #        max_length=SAFE_MAX_LEN, 
    #        return_tensors='pt'
    #    )
    #    corr_enc = model.tokenizer(
     #       corrupted_batch, 
    #        truncation=True, 
    #        max_length=SAFE_MAX_LEN, 
    #        return_tensors='pt'
    #    )

        # 2. 【核心技巧】将截断后的 Tensor 解码回 String
        # skip_special_tokens=True 是为了防止 attribute_mem 再次添加 BOS token 导致重复
     #   clean_strs = model.tokenizer.batch_decode(clean_enc['input_ids'], skip_special_tokens=True)
     #   corrupted_strs = model.tokenizer.batch_decode(corr_enc['input_ids'], skip_special_tokens=True)
     
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

        # 3. 处理 Label (这部分保持不变)
        label_ids = []
        for l in label_batch:
            if task == "yelp":
                # 情感任务才用 positive/negative
                l_int = int(l)
                txt = " positive" if l_int == 1 else " negative"
            else:
                # 其他任务，直接当作字符串 token
                txt = " " + str(l).strip()
                
            toks = model.tokenizer(txt, add_special_tokens=False)['input_ids']
            label_ids.append(toks[-1] if len(toks) > 0 else 0)

        label_tensor = torch.tensor(label_ids).to(model.cfg.device)

        # 4. 返回的是字符串列表 (clean_strs)，attribute_mem.py 能够正常处理
        fixed_dataset.append((clean_batch, corrupted_batch, label_tensor))  ####fixed_dataset.append((clean_strs, corrupted_strs, label_tensor))

    print(f"Validation passed. Processed {len(fixed_dataset)} batches.")
    return fixed_dataset


def load_model_unified(model_path: str, base_model_name: str = "meta-llama/Llama-2-7b-hf"):
    """
    Universal model loader supporting three modes:
    1. 'pretrained': Load pure base model (Llama2 or Qwen2)
    2. Directory path (auto-detect):
       - Contains adapter_config.json -> Load as PEFT/LoRA (Llama2 logic)
       - Full model directory -> Load directly (Qwen2 full fine-tuning logic)
    3. File path (.pt/.pth): Load state dict
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"⬇️ Loading Model from: {model_path}")
    print(f"   Base Model Name for Config: {base_model_name}")

    # Use bfloat16 for A100/3090/4090, fallback to float16
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    # --- Case 1: Pure Pretrained Model ---
    if model_path is None or model_path.lower() == 'pretrained':
        print("   [Mode] Pure Pretrained Base Model")
        model = HookedTransformer.from_pretrained(
            base_model_name,
            device=device,
            fold_ln=False,
            center_writing_weights=False,
            center_unembed=False,
            dtype=dtype
            #hf_model_args={"trust_remote_code": True}
        )

    # --- Case 2: Directory Path (Auto-detect LoRA vs Full) ---
    elif os.path.isdir(model_path):
        # Detect if it's a LoRA adapter
        is_lora = os.path.exists(os.path.join(model_path, "adapter_config.json"))
        
        if is_lora:
            print("   [Mode] Directory detected as PEFT/LoRA Adapter (Llama2 style)")
            print("   1. Loading Base Model...")
            hf_base = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=dtype,
                device_map="cpu",  # Load to CPU first, move to GPU after merging
                trust_remote_code=True
            )
            
            print("   2. Loading & Merging Adapter...")
            hf_model = PeftModel.from_pretrained(hf_base, model_path)
            hf_model = hf_model.merge_and_unload()
            
            # Enable gradient checkpointing to save memory
            if hasattr(hf_model, "gradient_checkpointing_enable"):
                hf_model.gradient_checkpointing_enable() 
            
            print("   3. Converting to HookedTransformer...")
            model = HookedTransformer.from_pretrained(
                base_model_name,
                hf_model=hf_model,
                device=device,
                fold_ln=False,
                center_writing_weights=False,
                center_unembed=False,
                dtype=dtype
            )
            del hf_base, hf_model

        else:
            print("   [Mode] Directory detected as Full Fine-Tuned Model (Qwen2 style)")
            # Load complete model directly from directory
            print("   1. Loading Full Model from directory...")
            hf_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=dtype,
                device_map="cpu",  # Load to CPU first
                trust_remote_code=True
            )
            
            print("   2. Converting to HookedTransformer...")
            model = HookedTransformer.from_pretrained(
                base_model_name,  # Still need base_name for TL to build computation graph
                hf_model=hf_model,
                device=device,
                fold_ln=False,
                center_writing_weights=False,
                center_unembed=False,
                dtype=dtype
            )
            del hf_model

        torch.cuda.empty_cache()

    # --- Case 3: Standard State Dict (.pt / .pth file) ---
    elif model_path.endswith('.pt') or model_path.endswith('.pth'):
        print("   [Mode] Standard State Dict (.pt/.pth)")
        # Initialize structure from pretrained
        model = HookedTransformer.from_pretrained(
            base_model_name,
            device="cpu",
            fold_ln=False,
            center_writing_weights=False,
            center_unembed=False,
            dtype=dtype,
            #hf_model_args={"trust_remote_code": True}
        )
        
        # Load weights
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
        model.to(device)

    else:
        raise ValueError(f"Unknown model path format: {model_path}")

    # --- Common Configuration ---
    model.cfg.use_attn_in = True
    model.cfg.use_split_qkv_input = True
    model.cfg.use_attn_result = True
    model.cfg.use_hook_mlp_in = True

    # Unified tokenizer handling (compatible with Llama and Qwen)
    if model.tokenizer.pad_token is None:
        model.tokenizer.pad_token = model.tokenizer.eos_token
    
    # Force left padding (TransformerLens preference)
    model.tokenizer.padding_side = "left"

    return model
# ==========================================
# 5. EAP 执行逻辑
# ==========================================
def get_important_edges(model, dataset, metric, top_k, GraphClass, attribute_fn, task):
    # 1. 验证数据
    dataset = validate_dataset_tokenization(model, dataset,task)

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
    parser.add_argument("--task", type=str,choices=['yelp', 'sst2','qa', 'mt', 'coqa','squad', 'kde4', 'tatoeba'],default='sst2')
    parser.add_argument("--model_name", type=str, default="llama2")  
    parser.add_argument("--output_dir", type=str, default="/users/sglli24/UnderstandingFineTuningViaMI/output/EAP_edges")
    
    # Selcet mode
    parser.add_argument("--mode", type=str, default="finetuned", choices=['finetuned', 'pretrained', 'compare'])
                      
    parser.add_argument("--data_path", type=str, default='/users/sglli24/UnderstandingFineTuningViaMI/output/corrupted_data/sst2_corrupted.csv')
    parser.add_argument("--ft_model_path", type=str,help="Path to fientuned model directory")#default="/mnt/scratch/users/sglli24/fine-tuning-project/fine_tuned_model/qwen2-0.5b-coqa-full-20251125-182058/checkpoint-4500/"
    parser.add_argument("--base_model_name", type=str, default="Qwen/Qwen2-0.5B",help="HF Hub name for config/tokenizer") #"meta-llama/Llama-2-7b-hf" default="Qwen/Qwen2-0.5B"
    parser.add_argument("--top_k", type=int, default=400)
    parser.add_argument("--batch_size", type=int, default=1)
    
    
    # [Compare Mode]
    parser.add_argument("--edge_file_1", type=str, default=None, help="Path to first edge csv (e.g., finetuned)")
    parser.add_argument("--edge_file_2", type=str, default=None, help="Path to second edge csv (e.g., pretrained)")
    
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 2. 加载 CSV 数据
    print(f"Reading data: {args.data_path}")
    df = pd.read_csv(args.data_path)
    raw_dataset = batch_dataset(df, batch_size=args.batch_size)

    # 3. 分析微调模型
    edges_ft = None
    if args.ft_model_path and args.mode == 'finetuned':
        print("Analyzing FINETUNED Model...")
        model_ft = load_model_unified(args.ft_model_path, args.base_model_name)
        edges_ft = get_important_edges(model_ft, raw_dataset, prob_diff, args.top_k, Graph, attribute.attribute,args.task)

        save_path = os.path.join(args.output_dir, f"{args.model_name}_{args.task}_finetuned_edges.csv")
        pd.DataFrame(list(edges_ft.items()), columns=['edge', 'score']).to_csv(save_path, index=False)
        print(f"   Saved to {save_path}")
        del model_ft
        torch.cuda.empty_cache()

    # 4. 分析预训练模型
    edges_pt = None
    if args.mode == 'pretrained':
        print("Analyzing PRETRAINED Model...")
        model_pt = load_model_unified(None, args.base_model_name)
        edges_pt = get_important_edges(model_pt, raw_dataset, prob_diff, args.top_k, Graph, attribute.attribute,args.task)
        
        pretrain_dir = os.path.join(args.output_dir, "pretrained")
        os.makedirs(pretrain_dir, exist_ok=True)
        save_path = os.path.join(pretrain_dir, f"{args.model_name}_{args.task}_pretrained_edges.csv")
        pd.DataFrame(list(edges_pt.items()), columns=['edge', 'score']).to_csv(save_path, index=False)
        print(f"   Saved to {save_path}")
        del model_pt
        torch.cuda.empty_cache()

    # 5. Calculate overlap
    if args.mode == 'compare':
        if not args.edge_file_1 or not args.edge_file_2:
            raise ValueError("In 'compare' mode, --edge_file_1 and --edge_file_2 are required.")
        
        print(f"Comparing:\n  File 1: {args.edge_file_1}\n  File 2: {args.edge_file_2}")
        
        # 1. Read CSV
        df1 = pd.read_csv(args.edge_file_1)
        df2 = pd.read_csv(args.edge_file_2)
        edges_1 = set(df1['edge'].tolist())
        edges_2 = set(df2['edge'].tolist())
        common = edges_1 & edges_2
        print(f"  Intersection (Common): {len(common)}")
        
        scores_1 = dict(zip(df1['edge'], df1['score']))
        scores_2 = dict(zip(df2['edge'], df2['score']))
        
        common_list = [{'edge': e, 'score_1': scores_1[e], 'score_2': scores_2[e]} for e in common]
        
        overlap_dir = os.path.join(args.output_dir, "overlap")
        os.makedirs(overlap_dir, exist_ok=True)
        
        # --- Dynamic Filename Generation Logic ---
        # Extract filename without extension (e.g., "gpt2_coqa_finetuned_edges")
        name1 = os.path.splitext(os.path.basename(args.edge_file_1))[0]
        name2 = os.path.splitext(os.path.basename(args.edge_file_2))[0]
        
        # Create combined name: "name1&name2_overlap.csv"
        # Example: gpt2_coqa_finetuned_edges&llama2_squad_finetuned_edges_overlap.csv
        save_filename = f"{name1}&{name2}_overlap.csv"
        save_path = os.path.join(overlap_dir, save_filename)
        
        pd.DataFrame(common_list).to_csv(save_path, index=False)
        print(f"Saved overlap details to: {save_path}")

if __name__ == "__main__":
    main()