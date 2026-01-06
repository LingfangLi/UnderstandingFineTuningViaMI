import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import gc
import json
import logging
from tqdm import tqdm
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM
from peft import PeftModel

# Configuration
logging.basicConfig(level=logging.INFO)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class ModelLoader:
    @staticmethod
    def load(model_key: str, is_finetuned: bool, task_name: str) -> HookedTransformer:
        device = UserConfig.DEVICE

        # 1. Determine precision
        use_bf16 = (model_key == "llama2" and task_name == "mt_kde4" and
                    torch.cuda.is_available() and torch.cuda.is_bf16_supported())
        dtype = torch.bfloat16 if use_bf16 else (torch.float16 if "llama" in model_key else torch.float32)

        base_model_name = SysConfig.BASE_MODELS[model_key]
        logging.info(f"⬇️ Loading {model_key} ({'Fine-tuned' if is_finetuned else 'Base'}) with {dtype}...")

        # === Case A: Load base model ===
        if not is_finetuned:
            return HookedTransformer.from_pretrained(
                base_model_name, device=device, torch_dtype=dtype,
                fold_ln=False, center_writing_weights=False, center_unembed=False
            )

        # === Case B: Load fine-tuned model ===
        else:
            folder_name = UserConfig.FT_MODEL_MAP.get(model_key, {}).get(task_name)
            if not folder_name:
                raise ValueError(f"Config Error: No folder name for {model_key} on {task_name}")

            full_path = os.path.join(UserConfig.MODEL_ROOT_DIR, folder_name)

            if not os.path.exists(full_path):
                logging.error(f"❌ Path NOT FOUND: {full_path}")
                raise FileNotFoundError(f"Path not found: {full_path}")

            # 1. If it's a directory (usually QLoRA Adapter)
            if os.path.isdir(full_path):
                is_lora = os.path.exists(os.path.join(full_path, "adapter_config.json"))
                if is_lora:
                    print("   [Mode] Directory detected as PEFT/LoRA Adapter")
                    print("   1. Loading Base Model...")
                    hf_base = AutoModelForCausalLM.from_pretrained(
                        base_model_name,
                        torch_dtype=dtype,
                        device_map="cpu", 
                        trust_remote_code=True
                    )

                    print("   2. Loading & Merging Adapter...")
                    try:
                        hf_model = PeftModel.from_pretrained(hf_base, full_path)
                        hf_model = hf_model.merge_and_unload() 
                        logging.info("   Merge complete.")
                    except Exception as e:
                        logging.warning(f"   Failed to load as PEFT adapter ({e}). Trying as full HF model...")
                        del hf_base
                        hf_model = AutoModelForCausalLM.from_pretrained(
                            full_path, torch_dtype=dtype, device_map="cpu"
                        )

                    # Pass to HookedTransformer
                    model = HookedTransformer.from_pretrained(
                        base_model_name, 
                        hf_model=hf_model, 
                        device=device,
                        torch_dtype=dtype,
                        fold_ln=False, center_writing_weights=False, center_unembed=False
                    )
                    del hf_base, hf_model
                else:
                    print("   [Mode] Directory detected as Full Fine-Tuned Model")
                    print("   1. Loading Full Model from directory...")
                    hf_model = AutoModelForCausalLM.from_pretrained(
                        full_path,
                        torch_dtype=dtype,
                        device_map="cpu",
                        trust_remote_code=True
                    )

                    print("   2. Converting to HookedTransformer...")
                    model = HookedTransformer.from_pretrained(
                        base_model_name, 
                        hf_model=hf_model,
                        device=device,
                        fold_ln=False,
                        center_writing_weights=False,
                        center_unembed=False,
                        dtype=dtype
                    )
                    del hf_model

                torch.cuda.empty_cache()
                return model

            # 2. If it's a .pt file (State Dict)
            else:
                logging.info(f"   Loading state_dict from .pt file onto {base_model_name}...")
                model = HookedTransformer.from_pretrained(
                    base_model_name, device=device, torch_dtype=dtype,
                    fold_ln=False, center_writing_weights=False, center_unembed=False
                )
                state_dict = torch.load(full_path, map_location=device)
                model.load_state_dict(state_dict, strict=False)
                return model

# ==============================================================================
# [Section 1] User Configuration Console (USER CONFIGURATION)
# ==============================================================================
class UserConfig:
    # 1. Which model do you want to test?
    TARGET_MODEL =  ["gpt2", "llama3","qwen2","llama2"]        
    
    # 2. Which version of this model do you want to test?
    USE_FINETUNED = True
    
    # 3. If testing fine-tuned version, which task's fine-tuned version?
    TARGET_TASK = ['sentiment_yelp','sentiment_sst2','qa_squad', 'qa_coqa','mt_kde4','mt_tatoeba'] 
    
    # 4. Path configuration
    MODEL_ROOT_DIR = r"/mnt/data1/users/sglli24/fine-tuning-project-1/fine_tuned_models/"
    OUTPUT_DIR = r"/users/sglli24/UnderstandingFineTuningViaMI/experiments/induction_head/output/"
    
    # 5. Fine-tuned model folder mapping
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
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==============================================================================
# [Section 2] System Configuration (System Config)
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
        Calculate Induction Score matrix
        """
        vocab_size = model.cfg.d_vocab
        # Avoid special BOS/EOS tokens
        random_tokens = torch.randint(100, vocab_size, (batch_size, seq_len)).to(UserConfig.DEVICE)
        
        # Repeat concatenation: [A, B, C] -> [A, B, C, A, B, C]
        repeated_tokens = torch.cat([random_tokens, random_tokens], dim=1)
        
        # Run model
        _, cache = model.run_with_cache(repeated_tokens, remove_batch_dim=False)
        
        n_layers = model.cfg.n_layers
        n_heads = model.cfg.n_heads
        induction_scores = np.zeros((n_layers, n_heads))
        
        logging.info("Scanning layers for Induction Heads...")
        for layer in tqdm(range(n_layers), desc="Layers"):
            pattern = cache[f"blocks.{layer}.attn.hook_pattern"]
            # Average over batch
            pattern_mean = pattern.mean(dim=0)
            
            for head in range(n_heads):
                head_attn = pattern_mean[head]
                
                score = 0
                count = 0
                
                # Traverse the second half
                for i in range(seq_len, 2 * seq_len - 1):
                    target_pos = i - seq_len + 1
                    prob = head_attn[i, target_pos].item()
                    score += prob
                    count += 1
                
                induction_scores[layer, head] = score / count
                
        del cache
        torch.cuda.empty_cache()
        return induction_scores

    # ==========================================================================
    # [New Feature] Automatic elbow detection + threshold cutoff
    # ==========================================================================
    @staticmethod
    def get_dynamic_cutoff(induction_scores, min_threshold=0.1):
        """
        Algorithm:
        1. Sort scores
        2. Geometrically find curve inflection point (Elbow Point)
        3. Combine with 0.1 hard threshold to prevent collapsed models from selecting noise
        """
        n_layers, n_heads = induction_scores.shape
        # 1. Flatten and record indices
        flat_scores = []
        for l in range(n_layers):
            for h in range(n_heads):
                flat_scores.append( (induction_scores[l, h], l, h) )
        
        # Sort in descending order of scores
        flat_scores.sort(key=lambda x: x[0], reverse=True)
        sorted_scores = np.array([x[0] for x in flat_scores])
        
        # 2. Elbow detection (Elbow Method)
        n_points = len(sorted_scores)
        if n_points < 2: return [], 0, 0
        
        all_coords = np.vstack((range(n_points), sorted_scores)).T
        first_point = all_coords[0]
        line_vec = all_coords[-1] - all_coords[0]
        line_vec_norm = line_vec / np.sqrt(np.sum(line_vec**2))
        vec_from_first = all_coords - first_point
        scalar_product = np.sum(vec_from_first * np.tile(line_vec_norm, (n_points, 1)), axis=1)
        vec_from_first_parallel = np.outer(scalar_product, line_vec_norm)
        vec_to_line = vec_from_first - vec_from_first_parallel
        dist_to_line = np.sqrt(np.sum(vec_to_line ** 2, axis=1))
        
        # Limit search range to first 20% (Induction Heads won't be too many)
        search_limit = max(5, int(n_points * 0.2))
        elbow_idx = np.argmax(dist_to_line[:search_limit])
        
        # 3. Safety threshold filter
        valid_mask = sorted_scores > min_threshold
        threshold_idx = np.sum(valid_mask) # Number of values above threshold
        
        # 4. Final decision
        # Take the smaller value (stricter), but if Elbow is too conservative (e.g., only selects 1 but there are actually 5 strong heads), allow relaxation
        final_k = min(elbow_idx, threshold_idx)
        
        # Fallback: If scores are really high (>0.3), select them anyway
        strong_heads_count = np.sum(sorted_scores > 0.3)
        final_k = max(final_k, strong_heads_count)
        
        # Extract results
        selected_heads = flat_scores[:final_k]
        # Format: [(score, layer, head), ...]
        
        return selected_heads, final_k, sorted_scores

    @staticmethod
    def visualize_and_save(scores, model_key, task_name, is_ft):
        """Visualize and save results (enhanced version)"""
        save_dir = os.path.join(UserConfig.OUTPUT_DIR, model_key)
        os.makedirs(save_dir, exist_ok=True)
        
        status = "FineTuned" if is_ft else "Base"
        suffix = f"{task_name}" if is_ft else "Pretrained"
        
        # 1. Run automatic detection algorithm
        selected_heads, k_val, sorted_scores = InductionDetector.get_dynamic_cutoff(scores, min_threshold=0.1)
        
        # Prepare list of Heads to save (Layer, Head)
        head_list = [{"layer": int(h[1]), "head": int(h[2]), "score": float(h[0])} for h in selected_heads]
        
        logging.info(f"[Detection] Found {k_val} induction heads for {model_key}-{task_name}")
        
        # 2. Save data
        # (A) Raw matrix
        np.save(os.path.join(save_dir, f"induction_scores_{status}_{suffix}.npy"), scores)
        # (B) Detected Heads list (for Overlap calculation)
        with open(os.path.join(save_dir, f"detected_heads_{status}_{suffix}.json"), 'w') as f:
            json.dump({"model": model_key, "task": task_name, "k": int(k_val), "heads": head_list}, f, indent=4)
        
        # 3. Plot (two side-by-side plots: left Heatmap, right Curve)
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        # --- Left plot: Heatmap ---
        sns.heatmap(
            scores, 
            cmap="Blues", 
            vmin=0, vmax=1.0,
            ax=axes[0],
            cbar_kws={'label': 'Induction Score'}
        )
        axes[0].set_title(f"Induction Matrix: {model_key} ({status})", fontsize=14)
        axes[0].set_xlabel("Head Index")
        axes[0].set_ylabel("Layer Index")
        axes[0].invert_yaxis()
        
        # --- Right plot: Elbow Curve (Rank vs Score) ---
        axes[1].plot(sorted_scores, linewidth=2, label='Score Curve')
        axes[1].set_title(f"Head Distribution (Top-K={k_val} Selected)", fontsize=14)
        axes[1].set_xlabel("Rank (Head)")
        axes[1].set_ylabel("Induction Score")
        axes[1].grid(True, alpha=0.3)
        
        # Mark cutoff point
        if k_val > 0:
            cutoff_score = sorted_scores[k_val-1]
            axes[1].axvline(x=k_val, color='r', linestyle='--', label=f'Cutoff K={k_val}')
            axes[1].axhline(y=cutoff_score, color='g', linestyle='--', label=f'Threshold={cutoff_score:.2f}')
            axes[1].scatter([k_val], [cutoff_score], color='red', s=100, zorder=5)
            # Add text on plot
            axes[1].text(k_val+5, cutoff_score, f" K={k_val}\n Score={cutoff_score:.3f}", color='red')
        else:
             axes[1].text(0.5, 0.5, "NO INDUCTION HEADS FOUND\n(Mechanism Collapse)", 
                          horizontalalignment='center', verticalalignment='center', 
                          transform=axes[1].transAxes, color='red', fontsize=14, fontweight='bold')

        axes[1].legend()
        
        img_path = os.path.join(save_dir, f"analysis_{status}_{suffix}.png")
        plt.savefig(img_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Analysis saved to {img_path}")

# ==============================================================================
# [Section 5] Main Program
# ==============================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    
    model_keys = UserConfig.TARGET_MODEL
    is_ft = UserConfig.USE_FINETUNED
    tasks = UserConfig.TARGET_TASK
    
    try:
        for model_key in model_keys:
            for task in tasks:
                print("load models")
                # 1. Load model
                model = ModelLoader.load(model_key, is_ft, task)
                
                # 2. Calculate Induction Scores
                scores = InductionDetector.get_induction_score_matrix(model)
                
                # 3. Automatic detection + visualization + save Json
                InductionDetector.visualize_and_save(scores, model_key, task, is_ft)
                
                # 4. Cleanup
                del model
                gc.collect()
                torch.cuda.empty_cache()
        
    except Exception as e:
        logging.error(f"Failed: {e}")
        import traceback
        traceback.print_exc()