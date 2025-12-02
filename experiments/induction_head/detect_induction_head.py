import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import gc
import logging
from tqdm import tqdm
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM
from peft import PeftModel

# 配置
logging.basicConfig(level=logging.INFO)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class ModelLoader:
    @staticmethod
    def load(model_key: str, is_finetuned: bool, task_name: str) -> HookedTransformer:
        device = UserConfig.DEVICE

        # 1. 确定精度
        use_bf16 = (model_key == "llama2" and task_name == "mt_kde4" and
                    torch.cuda.is_available() and torch.cuda.is_bf16_supported())
        dtype = torch.bfloat16 if use_bf16 else (torch.float16 if "llama" in model_key else torch.float32)

        base_model_name = SysConfig.BASE_MODELS[model_key]
        logging.info(f"⬇️ Loading {model_key} ({'Fine-tuned' if is_finetuned else 'Base'}) with {dtype}...")

        # === 情况 A: 加载纯基座模型 ===
        if not is_finetuned:
            return HookedTransformer.from_pretrained(
                base_model_name, device=device, torch_dtype=dtype,
                fold_ln=False, center_writing_weights=False, center_unembed=False
            )

        # === 情况 B: 加载微调模型 ===
        else:
            folder_name = UserConfig.FT_MODEL_MAP.get(model_key, {}).get(task_name)
            if not folder_name:
                raise ValueError(f"Config Error: No folder name for {model_key} on {task_name}")

            full_path = os.path.join(UserConfig.MODEL_ROOT_DIR, folder_name)

            if not os.path.exists(full_path):
                logging.error(f"❌ Path NOT FOUND: {full_path}")
                raise FileNotFoundError(f"Path not found: {full_path}")

            # --- 关键修改开始 ---

            # 1. 如果是文件夹 (通常是 QLoRA Adapter)
            if os.path.isdir(full_path):
                is_lora = os.path.exists(os.path.join(full_path, "adapter_config.json"))
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
                    # 步骤 2: 加载 Adapter 并合并
                    try:
                        hf_model = PeftModel.from_pretrained(hf_base, full_path)
                        hf_model = hf_model.merge_and_unload()  # 合并权重
                        logging.info("   Merge complete.")
                    except Exception as e:
                        logging.warning(f"   Failed to load as PEFT adapter ({e}). Trying as full HF model...")
                        # 如果不是 Adapter，可能是全量保存的模型，直接加载
                        del hf_base
                        hf_model = AutoModelForCausalLM.from_pretrained(
                            full_path, torch_dtype=dtype, device_map="cpu"
                        )

                    # 步骤 3: 传入 HookedTransformer
                    # 注意：第一个参数必须是官方名称 (base_model_name)，然后通过 hf_model 参数传入实际权重对象
                    model = HookedTransformer.from_pretrained(
                        base_model_name,  # 这里传 "meta-llama/Llama-2-7b-hf"
                        hf_model=hf_model,  # 这里传合并后的模型对象
                        device=device,
                        torch_dtype=dtype,
                        fold_ln=False, center_writing_weights=False, center_unembed=False
                    )
                    del hf_base, hf_model
                else:
                    print("   [Mode] Directory detected as Full Fine-Tuned Model (Qwen2 style)")
                    # Load complete model directly from directory
                    print("   1. Loading Full Model from directory...")
                    hf_model = AutoModelForCausalLM.from_pretrained(
                        full_path,
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
                    # 清理临时变量
                    del hf_model

                torch.cuda.empty_cache()
                return model

            # 2. 如果是 .pt 文件 (State Dict)
            else:
                logging.info(f"   Loading state_dict from .pt file onto {base_model_name}...")

                # 先加载 Base HookedTransformer
                model = HookedTransformer.from_pretrained(
                    base_model_name, device=device, torch_dtype=dtype,
                    fold_ln=False, center_writing_weights=False, center_unembed=False
                )
                # 覆盖权重
                state_dict = torch.load(full_path, map_location=device)
                model.load_state_dict(state_dict, strict=False)
                return model

# ==============================================================================
# [区域 1] 用户配置控制台 (USER CONFIGURATION)
# ==============================================================================
class UserConfig:
    # 1. 你想测哪个模型？
    TARGET_MODEL =  ["llama2","gpt2","llama3","qwen2"]       # 选项: "gpt2", "llama2", "llama3", "qwen2"
    
    # 2. 你想测这个模型的哪个版本？
    # True  = 测微调后的版本 (需要下方 TARGET_TASK 有对应的文件夹)
    # False = 测原始基座版本 (HuggingFace 原版)
    USE_FINETUNED = False
    
    # 3. 如果测微调版，是哪个任务的微调版？
    TARGET_TASK = ['sentiment_yelp','qa_squad','mt_kde4','mt_tatoeba','qa_coqa']      # 选项: "sentiment_yelp", "qa_squad", "mt_kde4" 等
    
    # 4. 路径配置 (保持和你之前的一致)
    MODEL_ROOT_DIR = r"/mnt/data1/users/sglli24/fine-tuning-project-1/fine_tuned_models/"
    OUTPUT_DIR = r"/users/sglli24/UnderstandingFineTuningViaMI/experiments/induction_head/output/"
    # 5. 微调模型文件夹映射
    FT_MODEL_MAP = {
        "gpt2": {
            "sentiment_yelp": "gpt2-yelp.pt",
            "qa_squad":       "gpt2-squad.pt",
            "mt_kde4":        "gpt2-kde4.pt",
            "mt_tatoeba":     "gpt2-tatoeba.pt",
            "qa_coqa":        "gpt2-coqa.pt"
        },
        "llama3": {
            "sentiment_yelp": "llama3.2-yelp.pt",
            "qa_squad":       "llama3.2-squad.pt",
            "mt_kde4":        "llama3.2-kde4.pt",
            "qa_coqa":        "llama3.2-coqa.pt",
            "mt_tatoeba":     "llama3.2-tatoeba.pt"
        },
        "llama2": {
            "sentiment_yelp": "llama2-yelp",
            "qa_squad":       "llama2-squad",
            "qa_coqa":        "llama2-coqa",
            "mt_kde4":        "llama2-kde4",
            "mt_tatoeba":     "llama2-tatoeba",
        },
        "qwen2": {
            "sentiment_yelp": "qwen2-yelp",
            "qa_squad":       "qwen2-squad",
            "qa_coqa":        "qwen2-coqa",
            "mt_kde4":        "qwen2-kde4",
            "mt_tatoeba":     "qwen2-tatoeba",
        }
    }
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==============================================================================
# [区域 2] 系统配置 (System Config)
# ==============================================================================
class SysConfig:
    BASE_MODELS = {
        'gpt2':   'gpt2',
        'llama2': 'meta-llama/Llama-2-7b-hf',
        'llama3': 'meta-llama/Llama-3.2-1B',
        'qwen2': 'Qwen/Qwen2-0.5B',
    }                

class InductionDetector:
    @staticmethod
    def get_induction_score_matrix(model, seq_len=50, batch_size=1):
        """
        计算 Induction Score 矩阵
        """
        # 1. 动态生成随机序列 (根据模型词表大小)
        vocab_size = model.cfg.d_vocab
        # 避开特殊的 BOS/EOS token (通常在前几百个)
        random_tokens = torch.randint(100, vocab_size, (batch_size, seq_len)).to(UserConfig.DEVICE)
        
        # 重复拼接: [A, B, C] -> [A, B, C, A, B, C]
        # 如果模型需要 BOS，可以在这里手动加，但对于随机序列测试通常不需要
        repeated_tokens = torch.cat([random_tokens, random_tokens], dim=1)
        
        # 2. 运行模型
        # remove_batch_dim=False 保持维度 [batch, heads, q, k]
        _, cache = model.run_with_cache(repeated_tokens, remove_batch_dim=False)
        
        n_layers = model.cfg.n_layers
        n_heads = model.cfg.n_heads
        induction_scores = np.zeros((n_layers, n_heads))
        
        logging.info("Scanning layers for Induction Heads...")
        for layer in tqdm(range(n_layers), desc="Layers"):
            # 获取 Attention Pattern
            pattern = cache[f"blocks.{layer}.attn.hook_pattern"]
            # 对 batch 取平均 -> [heads, seq_len*2, seq_len*2]
            pattern_mean = pattern.mean(dim=0)
            
            for head in range(n_heads):
                head_attn = pattern_mean[head]
                
                score = 0
                count = 0
                
                # 遍历后半段 (第二次出现的部分)
                for i in range(seq_len, 2 * seq_len - 1):
                    # Induction Target: 上一次出现位置的后一位
                    # 当前位置 i，上一次位置 i - seq_len
                    # 目标位置 (Key) = i - seq_len + 1
                    target_pos = i - seq_len + 1
                    
                    # 获取该 Head 对该位置的关注度
                    prob = head_attn[i, target_pos].item()
                    score += prob
                    count += 1
                
                induction_scores[layer, head] = score / count
                
        # 清理
        del cache
        torch.cuda.empty_cache()
        return induction_scores

    @staticmethod
    def visualize_and_save(scores, model_key, task_name, is_ft):
        """可视化并保存结果"""
        save_dir = os.path.join(UserConfig.OUTPUT_DIR, model_key)
        os.makedirs(save_dir, exist_ok=True)
        
        status = "FineTuned" if is_ft else "Base"
        suffix = f"{task_name}" if is_ft else "Pretrained"
        
        # 1. 保存数据
        npy_path = os.path.join(save_dir, f"induction_scores_{status}_{suffix}.npy")
        np.save(npy_path, scores)
        
        # 2. 画图
        plt.figure(figsize=(12, 8))
        sns.set_theme(style="whitegrid")
        
        ax = sns.heatmap(
            scores, 
            cmap="Blues", 
            vmin=0, vmax=1.0,
            cbar_kws={'label': 'Induction Score'}
        )
        
        plt.title(f"Induction Heads: {model_key} ({status} - {suffix})", fontsize=16)
        plt.xlabel("Head Index", fontsize=12)
        plt.ylabel("Layer Index", fontsize=12)
        plt.gca().invert_yaxis()
        
        img_path = os.path.join(save_dir, f"heatmap_{status}_{suffix}.png")
        plt.savefig(img_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"✅ Results saved to {save_dir}")
        logging.info(f"   Top Score: {np.max(scores):.4f}")

# ==============================================================================
# [区域 5] 主程序
# ==============================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    
    model_keys = UserConfig.TARGET_MODEL
    is_ft = UserConfig.USE_FINETUNED
    tasks = UserConfig.TARGET_TASK
    
    try:
        for model_key in model_keys:
            for task in tasks:
                # 1. 加载模型
                model = ModelLoader.load(model_key, is_ft, task)
                
                # 2. 计算 Induction Scores
                scores = InductionDetector.get_induction_score_matrix(model)
                
                # 3. 保存和可视化
                InductionDetector.visualize_and_save(scores, model_key, task, is_ft)
                
                # 4. 清理
                del model
                gc.collect()
                torch.cuda.empty_cache()
        
    except Exception as e:
        logging.error(f"Failed: {e}")
        import traceback
        traceback.print_exc()