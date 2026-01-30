import os
import torch
import json
import gc
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from datasets import load_dataset
from transformer_lens import HookedTransformer
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


# Configuration

# Options: "qwen2", "llama3", "llama2", "gpt2"
MODEL_KEY = "llama3"

BASE_MODEL_PATH = r"/mnt/data1/users/sglli24/fine-tuning-project-1/old_version_finetuned_models"

MODEL_CONFIGS = {
    "qwen2": {
        "hf_name": "Qwen/Qwen2.5-0.5B",
        "ft_path": os.path.join(BASE_MODEL_PATH, "qwen2_kde4.pt"),
        "folder_name": "Qwen2_0.5B"
    },
    "llama3": {
        "hf_name": "meta-llama/Llama-3.2-1B",
        "ft_path": os.path.join(BASE_MODEL_PATH, "llama3.2-kde4.pt"),
        "folder_name": "Llama3_1B"
    },
    "llama2": {
        "hf_name": "meta-llama/Llama-2-7b-hf",
        "ft_path": os.path.join(BASE_MODEL_PATH, "llama2-kde4"),
        "folder_name": "Llama2_7B"
    }
}

BATCH_SIZE = 32
NUM_SAMPLES = 10
MAX_NEW_TOKENS = 64

# Helper Functions

def get_kde4_dataset():
    """Load KDE4 dataset and select test range (30000+)."""
    print("Loading KDE4 dataset...")
    try:
        dataset = load_dataset('kde4', lang1="en", lang2="fr")['train']
        test_data = dataset.select(range(30000, 30000 + NUM_SAMPLES))

        samples = []
        for item in test_data:
            if 'translation' in item:
                samples.append({'en': item['translation']['en'], 'fr': item['translation']['fr']})
            elif 'en' in item and 'fr' in item:
                samples.append(item)
        return samples
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return []

def calculate_bleu(ref, hyp):
    """Robust BLEU score calculation."""
    ref = ref.strip().lower().split()
    hyp = hyp.strip().lower().split()
    if not hyp: return 0.0
    smoothing = SmoothingFunction().method1
    return sentence_bleu([ref], hyp, smoothing_function=smoothing)

# Core: Pipeline Batch Processing

def run_phase_batched(model, samples, phase_name):
    print(f"\n[Phase: {phase_name}] Processing {len(samples)} samples...")

    # Model health check
    try:
        print(f"\n[{phase_name}] Performing Weight Health Check...")

        W_Q = model.blocks[0].attn.W_Q.data
        mean_val = W_Q.mean().item()
        std_val = W_Q.std().item()
        print(f"   Layer 0 Attn Weights -> Mean: {mean_val:.4f}, Std: {std_val:.4f}")

        if torch.isnan(W_Q).any():
            print("    CRITICAL: Weights contain NaNs! Model is broken.")
        elif std_val < 1e-6:
            print("    CRITICAL: Weights are all zeros/constant! Model learned nothing.")
        else:
            print("   Weights look statistically healthy.")
    except Exception as e:
        print(f"   Could not run health check: {e}")

    results = []


    model.tokenizer.padding_side = "left"
    if model.tokenizer.pad_token is None:
        model.tokenizer.pad_token = model.tokenizer.eos_token


    if "widest" not in locals():
        try:
            mystery_token = model.tokenizer.decode([81339])
            print(f"    Token 81339 decodes to: '{mystery_token}'")
        except:
            pass

    for i in tqdm(range(0, len(samples), BATCH_SIZE), desc=f"Running {phase_name}"):
        batch_samples = samples[i : i + BATCH_SIZE]


        batch_prompts = [
            f"Translate English to French. English: {s['en']}\nFrench:"
            for s in batch_samples
        ]

        batch_data = []

        try:
            with torch.no_grad():
                inputs = model.tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                    add_special_tokens=True
                ).to(model.cfg.device)

                output_ids = model.generate(
                    inputs.input_ids,
                    max_new_tokens=20,
                    do_sample=True,
                    temperature=0.6,
                    stop_at_eos=False,
                )

                input_len = inputs.input_ids.shape[1]
                new_tokens = output_ids[:, input_len:]
                decoded_preds = model.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

                for pred in decoded_preds:
                    batch_data.append({"prediction": pred.strip()})

        except Exception as e:
            print(f"Generation Error: {e}")
            for _ in batch_samples: batch_data.append({"prediction": "ERROR"})

        # Feature Extraction
        try:
            with torch.no_grad():
                tokens = model.to_tokens(batch_prompts)
                _, cache = model.run_with_cache(tokens, remove_batch_dim=False)
                batch_vectors = []
                for L in range(model.cfg.n_layers):
                    layer_out = cache[f"blocks.{L}.hook_resid_post"]
                    last_token_vecs = layer_out[:, -1, :].cpu().numpy()
                    last_token_vecs = np.nan_to_num(last_token_vecs)
                    batch_vectors.append(last_token_vecs)
                del cache
                for b_idx in range(len(batch_samples)):
                    sample_vectors = [batch_vectors[L][b_idx] for L in range(model.cfg.n_layers)]
                    batch_data[b_idx]["vectors"] = sample_vectors
        except Exception as e:
            print(f"Vectors Error: {e}")

        results.extend(batch_data)

    return results

def load_custom_model(cfg):
    print(f"Loading base model: {cfg['hf_name']}...")


    hf_model = AutoModelForCausalLM.from_pretrained(
        cfg['hf_name'],
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,

    )

    path = cfg['ft_path']
    device = "cuda" if torch.cuda.is_available() else "cpu"


    print("Converting Base Model to HookedTransformer...")
    tl_model = HookedTransformer.from_pretrained(
        cfg['hf_name'],
        hf_model=hf_model,
        device=device,
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
        dtype=torch.float16
    )

    del hf_model
    gc.collect()


    print(f"Loading finetuned weights from {path}...")
    state_dict = torch.load(path, map_location="cpu")


    missing, unexpected = tl_model.load_state_dict(state_dict, strict=False)
    print(f"Weight Loading Report:")
    print(f"  Missing Keys: {len(missing)}")
    print(f"  Unexpected Keys: {len(unexpected)}")


    if "embed.W_E" in state_dict and "unembed.W_U" not in state_dict:
        print(" Detected Tied Embeddings mismatch! Syncing W_E to W_U...")
        with torch.no_grad():
            tl_model.unembed.W_U.data = tl_model.embed.W_E.data.T.clone()
    else:
        print(" Forcing Tied Embeddings sync for Llama-3.2-1B...")
        with torch.no_grad():
            tl_model.unembed.W_U.data = tl_model.embed.W_E.data.T.clone()

    torch.cuda.empty_cache()
    return tl_model

# Main Execution

def main():
    cfg = MODEL_CONFIGS[MODEL_KEY]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"./Results/MT/{cfg['folder_name']}/{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    print(f"=== Starting Fast Analysis for {MODEL_KEY} ===")
    print(f"Output Directory: {output_dir}")

    samples = get_kde4_dataset()
    if not samples: return

    # Step 1: Pretrained Model
    print("\n--- Step 1: Pretrained Model ---")
    print(f"Loading Base Model: {cfg['hf_name']}")
    base_tl = HookedTransformer.from_pretrained(cfg['hf_name'], device="cuda", dtype=torch.float16)

    res_pre = run_phase_batched(base_tl, samples, "Pretrained")

    del base_tl
    gc.collect()
    torch.cuda.empty_cache()

    # Step 2: Finetuned Model
    print("\n--- Step 2: Finetuned Model ---")
    ft_tl = load_custom_model(cfg)

    res_ft = run_phase_batched(ft_tl, samples, "Finetuned")

    del ft_tl
    gc.collect()
    torch.cuda.empty_cache()

    # Step 3: Offline Calculation
    print("\n--- Step 3: Calculating PCA & Metrics ---")

    final_output = []
    n_layers = len(res_pre[0]['vectors'])

    print("Training PCA...")
    pcas = {}
    scalers = {}
    for L in tqdm(range(n_layers), desc="Fitting PCA"):
        X = np.array([s['vectors'][L] for s in res_pre])

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        pca = PCA(n_components=2)
        pca.fit(X_scaled)

        scalers[L] = scaler
        pcas[L] = pca

    print("Computing Distances & BLEU...")
    for i in tqdm(range(len(samples))):
        s = samples[i]
        pre = res_pre[i]
        ft = res_ft[i]

        bleu_before = calculate_bleu(s['fr'], pre['prediction'])
        bleu_after = calculate_bleu(s['fr'], ft['prediction'])

        layer_dists = []
        for L in range(n_layers):
            v_pre = pre['vectors'][L]
            v_ft = ft['vectors'][L]

            p_pre = pcas[L].transform(scalers[L].transform([v_pre]))[0]
            p_ft = pcas[L].transform(scalers[L].transform([v_ft]))[0]

            dist_before = np.linalg.norm(p_pre)
            dist_after = np.linalg.norm(p_ft)
            dist_change = dist_after - dist_before

            layer_dists.append({
                "layer": L,
                "dist_before": float(dist_before),
                "dist_after": float(dist_after),
                "dist_change": float(dist_change),
                "is_decreased": bool(dist_after < dist_before)
            })

        final_output.append({
            "sample_idx": i,
            "en": s['en'],
            "fr": s['fr'],
            "prediction_before": pre['prediction'],
            "prediction_after": ft['prediction'],
            "bleu_before": bleu_before,
            "bleu_after": bleu_after,
            "bleu_improvement": bleu_after - bleu_before,
            "layer_distances": layer_dists,
            "avg_dist_change": np.mean([d['dist_change'] for d in layer_dists])
        })

    save_path = os.path.join(output_dir, "batch_1_summary.json")
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)

    print(f"\n Analysis Done! Results saved to: {save_path}")
    print("Next: Run your filter script (filter_samples_bleu.py) on this folder.")

if __name__ == "__main__":
    main()
