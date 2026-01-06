import os
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformer_lens import HookedTransformer
import torch
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm
import logging
import gc
import json
from transformers import AutoModelForCausalLM
from peft import PeftModel


# ==============================================================================
# [区域 1] 用户配置控制台 (USER CONFIGURATION)
# ==============================================================================

class UserConfig:
    # 1. 运行模式
    RUN_MODE = "SINGLE"  # "ALL" 或 "SINGLE"

    # 2. 单次运行的目标
    TARGET_MODEL = ["gpt2","llama2","llama3","qwen2"]
    TARGET_TASK =  "sentiment_sst2" # 可以是字符串或字符串列表

    # 3. 路径配置 (根据你的截图修正)
    MODEL_ROOT_DIR = r"/mnt/data1/users/sglli24/fine-tuning-project-1/fine_tuned_models/"
    OUTPUT_DIR = r"/users/sglli24/UnderstandingFineTuningViaMI/experiments/attention_matrix_analysis/attention_analysis_results/"

    # 4. 微调模型文件夹映射 (确保这里的文件名和你截图里的一模一样)
    FT_MODEL_MAP = {
        "gpt2": {
            "sentiment_yelp": "gpt2-yelp.pt",
            "sentiment_sst2": "gpt2-sst2",
            "qa_squad":       "gpt2-squad.pt",
            "mt_kde4":        "gpt2-kde4.pt",
            "mt_tatoeba":     "gpt2-tatoeba.pt",
            "qa_coqa":        "gpt2-coqa.pt"
        },
        "llama3": {
            "sentiment_yelp": "llama3.2-yelp.pt",
            "sentiment_sst2": "llama3.2-sst2",
            "qa_squad":       "llama3.2-squad.pt",
            "mt_kde4":        "llama3.2-kde4.pt",
            "qa_coqa":        "llama3.2-coqa.pt",
            "mt_tatoeba":     "llama3.2-tatoeba.pt"
        },
        "llama2": {
            "sentiment_yelp": "llama2-yelp",
            "sentiment_sst2": "llama2-sst2",
            "qa_squad":       "llama2-squad",
            "qa_coqa":        "llama2-coqa",
            "mt_kde4":        "llama2-kde4",
            "mt_tatoeba":     "llama2-tatoeba",
        },
        "qwen2": {
            "sentiment_yelp": "qwen2-yelp",
            "sentiment_sst2": "qwen2-sst2",
            "qa_squad":       "qwen2-squad",
            "qa_coqa":        "qwen2-coqa",
            "mt_kde4":        "qwen2-kde4",
            "mt_tatoeba":     "qwen2-tatoeba",
        }
    }
    # 5. 分析参数
    NUM_SAMPLES = 50
    DEFAULT_EPSILON = 1e-5
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==============================================================================
# [区域 2] 格式化器定义 (Prompt Formatters)
# ==============================================================================
class PromptFormatter:
    @staticmethod
    def simple_yelp(item): return f"Review: {item['text']}\nSentiment:"

    @staticmethod
    def simple_qa(item): return f"Context: {item['context']}\nQuestion: {item['question']}\nAnswer:"

    @staticmethod
    def simple_coqa(item): return f"Story: {item['story']}\nQuestion: {item['question']}\nAnswer:"

    @staticmethod
    def simple_mt(item): return f"English: {item['en']}\nFrench:"

    @staticmethod
    def llama2_yelp(item): return f"Review: {item['text']}\nSentiment:"

    @staticmethod
    def llama2_squad(item):
        return f"### Context:\n{item['context']}\n\n### Question:\n{item['question']}\n\n### Answer:"

    @staticmethod
    def llama2_coqa(item):
        return f"### Context:\n{item['story']}\n\n### Chat History:\nNone\n\n### Current Question:\n{item['question']}\n\n### Current Answer:"

    @staticmethod
    def llama2_kde4(item):
        return f"Translate Technical English to French.\n\n### Technical English:\n{item['en']}\n\n### Technical French:"

    @staticmethod
    def llama2_tatoeba(item):
        return f"Translate English to French.\n\n### English:\n{item['en']}\n\n### French:"


# ==============================================================================
# [区域 3] 系统内部配置
# ==============================================================================
class SysConfig:
    BASE_MODELS = {
        'gpt2': 'gpt2',
        'llama2': 'meta-llama/Llama-2-7b-hf',
        'llama3': 'meta-llama/Llama-3.2-1B',
        'qwen2': 'Qwen/Qwen2-0.5B'
    }

    TASKS = {
        "sentiment_yelp": {
            "dataset": ("yelp_polarity", None),
            "split": "test", "range": (0, 100),
            "processor": lambda d: [{"text": s["text"]} for s in d],
            "formatters": {"llama2": PromptFormatter.llama2_yelp, "gpt2": PromptFormatter.simple_yelp,
                           "llama3": PromptFormatter.simple_yelp, "qwen2": PromptFormatter.llama2_yelp }
        },
        "qa_squad": {
            "dataset": ("squad", None),
            "split": "validation", "range": (0, 100),
            "processor": lambda d: [{"context": s["context"], "question": s["question"]} for s in d],
            "formatters": {"llama2": PromptFormatter.llama2_squad, "qwen2": PromptFormatter.llama2_squad,
                           "gpt2": PromptFormatter.simple_qa, "llama3": PromptFormatter.simple_qa}
        },
        "qa_coqa": {
            "dataset": ("stanfordnlp/coqa", None),
            "split": "validation", "range": (0, 100),
            "processor": lambda d: [{"story": s["story"], "question": s["questions"][0]} for s in d],
            "formatters": {"llama2": PromptFormatter.llama2_coqa, "qwen2": PromptFormatter.llama2_coqa,
                           "gpt2": PromptFormatter.simple_coqa,"llama3": PromptFormatter.simple_coqa}
        },
        "mt_kde4": {
            "dataset": ("kde4", ["en", "fr"]),
            "split": "train", "range": (30000, 30100),
            "processor": lambda d: [{"en": s["translation"]["en"]} for s in d],
            "formatters": {"llama2": PromptFormatter.llama2_kde4,"qwen2": PromptFormatter.llama2_kde4,
                           "gpt2": PromptFormatter.simple_mt, "llama3": PromptFormatter.simple_mt}
        },
        "mt_tatoeba": {
            "dataset": ("tatoeba", ["en", "fr"]),
            "split": "train", "range": (40000, 40100),
            "processor": lambda d: [{"en": s["translation"]["en"]} for s in d],
            "formatters": {"llama2": PromptFormatter.llama2_tatoeba, "qwen2": PromptFormatter.llama2_tatoeba,
                           "gpt2": PromptFormatter.simple_mt, "llama3": PromptFormatter.simple_mt}
        }
    }


# ==============================================================================
# [区域 4] 核心逻辑 (带内存优化)
# ==============================================================================

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

class DataLoader:
    @staticmethod
    def get_prompts(model_key: str, task_name: str) -> List[str]:
        task_cfg = SysConfig.TASKS[task_name]
        formatter = task_cfg["formatters"].get(model_key, task_cfg["formatters"].get("gpt2"))

        ds_name, ds_cfg = task_cfg["dataset"]
        if ds_cfg:
            ds = load_dataset(ds_name, lang1=ds_cfg[0], lang2=ds_cfg[1], split=task_cfg["split"],
                              trust_remote_code=True)
        else:
            ds = load_dataset(ds_name, split=task_cfg["split"])

        start, end = task_cfg["range"]
        ds = ds.select(range(min(start, len(ds)), min(end, len(ds))))

        raw_items = task_cfg["processor"](ds)
        prompts = [formatter(item) for item in raw_items]
        return prompts[:UserConfig.NUM_SAMPLES]


class AnalysisEngine:
    @staticmethod
    def run_pair(model_key: str, task_name: str):
        logging.info(f"\n{'=' * 40}\nStarting: {model_key} | {task_name}\n{'=' * 40}")

        try:
            # 1. 准备数据
            prompts = DataLoader.get_prompts(model_key, task_name)

            # 2. Base Model
            base_model = ModelLoader.load(model_key, False, task_name)
            base_patterns = AnalysisEngine._extract_patterns(base_model, prompts)

            # === 关键优化：彻底释放 Base Model 显存 ===
            del base_model
            gc.collect()
            torch.cuda.empty_cache()
            logging.info("Base model unloaded. Memory cleared.")

            # 3. FT Model
            ft_model = ModelLoader.load(model_key, True, task_name)
            ft_patterns = AnalysisEngine._extract_patterns(ft_model, prompts)

            # 4. 计算 KL
            logging.info("Calculating KL Divergence...")
            kl_matrix = AnalysisEngine._calculate_kl(base_patterns, ft_patterns)

            # 5. 保存结果 (只保存 Matrix)
            AnalysisEngine._save_results(model_key, task_name, kl_matrix)

            # 6. 清理 FT Model
            del ft_model
            gc.collect()
            torch.cuda.empty_cache()

        except Exception as e:
            logging.error(f"Failed to run {model_key} on {task_name}: {e}")
            import traceback
            traceback.print_exc()

    @staticmethod
    def _extract_patterns(model: HookedTransformer, prompts: List[str]):
        """批量提取 Attention"""
        patterns_all_samples = []
        for text in tqdm(prompts, desc="Extracting Attn"):
            tokens = model.to_tokens(text, prepend_bos=True)
            _, cache = model.run_with_cache(tokens, remove_batch_dim=True)

            sample_patterns = []
            for layer in range(model.cfg.n_layers):
                # 关键：立即转到 CPU 并 detach，防止显存堆积
                p = cache["pattern", layer].detach().cpu().to(torch.float32)
                sample_patterns.append(p)
            patterns_all_samples.append(sample_patterns)

            del cache  # 删除 cache 引用
        return patterns_all_samples

    @staticmethod
    def _calculate_kl(base_list, ft_list):
        n_samples = len(base_list)
        n_layers = len(base_list[0])
        n_heads = base_list[0][0].shape[0]
        total_kl_matrix = np.zeros((n_layers, n_heads))
        epsilon = UserConfig.DEFAULT_EPSILON

        for i in range(n_samples):
            for layer in range(n_layers):
                p = base_list[i][layer] + epsilon
                q = ft_list[i][layer] + epsilon
                p = p / p.sum(dim=-1, keepdim=True)
                q = q / q.sum(dim=-1, keepdim=True)
                kl_val = (p * (torch.log(p) - torch.log(q))).sum(dim=-1)
                total_kl_matrix[layer] += kl_val.mean(dim=-1).numpy()

        return total_kl_matrix / n_samples

    @staticmethod
    def _save_results(model_key, task_name, kl_matrix):
        save_path = os.path.join(UserConfig.OUTPUT_DIR, model_key, task_name)
        os.makedirs(save_path, exist_ok=True)

        # 保存 CSV
        df_head = pd.DataFrame(kl_matrix)
        df_head.index.name = 'Layer'
        df_head.columns = [f'Head_{i}' for i in range(kl_matrix.shape[1])]
        df_head.to_csv(os.path.join(save_path, "kl_divergence_heads.csv"), index=True)

        # 保存 NPY
        np.save(os.path.join(save_path, "kl_divergence_heads.npy"), kl_matrix)
        logging.info(f"✅ Results saved to: {save_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    if UserConfig.RUN_MODE == "ALL":
        for model_key, tasks_map in UserConfig.FT_MODEL_MAP.items():
            for task_name, folder_name in tasks_map.items():
                if folder_name:
                    AnalysisEngine.run_pair(model_key, task_name)
    elif UserConfig.RUN_MODE == "SINGLE":
        #判断task是list还是str
        target_tasks = UserConfig.TARGET_TASK
        if isinstance(target_tasks, str):
            AnalysisEngine.run_pair(UserConfig.TARGET_MODEL, target_tasks)
        else:
            for task in target_tasks:
                AnalysisEngine.run_pair(UserConfig.TARGET_MODEL, task)
