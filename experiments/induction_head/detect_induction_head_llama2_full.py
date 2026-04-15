"""Detect induction heads in Llama-2-7B FULL fine-tuning models (one per task)
plus the base pretrained model. Saves induction score matrices, detected head
lists, and analysis plots to experiments/induction_head/output/llama2/.

This is the full-FT counterpart of detect_induction_head.py, which uses the
older QLoRA models. Outputs OVERWRITE the corresponding files in
output/llama2/; the previous QLoRA-era data has already been backed up to
output/llama2_qlora/.

Twitter is intentionally excluded.
"""
import gc
import json
import logging
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# Configuration
PROJECT_ROOT = "/users/sglli24/UnderstandingFineTuningViaMI"
MODEL_ROOT = "/mnt/scratch/users/sglli24/fine-tuning-project/fine_tuned_model"
OUTPUT_DIR = f"{PROJECT_ROOT}/experiments/induction_head/output"

MODEL_KEY = "llama2"
BASE_MODEL_NAME = "meta-llama/Llama-2-7b-hf"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Six canonical tasks (no twitter). Folder names follow the standardized
# llama2-7b-{task}-full convention. Task names match overlap_analysis.py.
FT_MODEL_MAP = {
    "sentiment_yelp": "llama2-7b-yelp-full",
    "sentiment_sst2": "llama2-7b-sst2-full",
    "qa_squad":       "llama2-7b-squad-full",
    "qa_coqa":        "llama2-7b-coqa-full",
    "mt_kde4":        "llama2-7b-kde4-full",
    "mt_tatoeba":     "llama2-7b-tatoeba-full",
}

# Run base detection only when needed (set RERUN_BASE=1 to force).
RERUN_BASE = os.environ.get("RERUN_BASE", "0") == "1"

# Optional task filter (comma-separated env var, e.g. TASKS=sentiment_yelp,qa_squad)
TASK_FILTER_ENV = os.environ.get("TASKS", "").strip()
TASK_FILTER = TASK_FILTER_ENV.split(",") if TASK_FILTER_ENV else None


def load_full_ft_hookedtransformer(model_subdir: str) -> HookedTransformer:
    """Load a Llama-2-7B full FT directory and convert to HookedTransformer."""
    full_path = os.path.join(MODEL_ROOT, model_subdir)
    if not os.path.isdir(full_path):
        raise FileNotFoundError(f"Full FT model dir not found: {full_path}")

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    logging.info(f"Loading full FT model from {full_path} ({dtype})")

    hf_model = AutoModelForCausalLM.from_pretrained(
        full_path,
        torch_dtype=dtype,
        device_map="cpu",
        trust_remote_code=True,
    )

    logging.info("Converting to HookedTransformer...")
    model = HookedTransformer.from_pretrained(
        BASE_MODEL_NAME,
        hf_model=hf_model,
        device=DEVICE,
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
        dtype=dtype,
    )
    del hf_model
    torch.cuda.empty_cache()
    return model


def load_base_hookedtransformer() -> HookedTransformer:
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    logging.info(f"Loading base pretrained {BASE_MODEL_NAME} ({dtype})")
    return HookedTransformer.from_pretrained(
        BASE_MODEL_NAME,
        device=DEVICE,
        torch_dtype=dtype,
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
    )


def get_induction_score_matrix(model: HookedTransformer, seq_len: int = 50, batch_size: int = 1) -> np.ndarray:
    """Run a synthetic [A,B,...,A,B,...] sequence through the model and measure
    each head's induction score: P(attend to position i-seq_len+1 from position i)
    in the second half of the repeated sequence."""
    vocab_size = model.cfg.d_vocab
    random_tokens = torch.randint(100, vocab_size, (batch_size, seq_len)).to(DEVICE)
    repeated = torch.cat([random_tokens, random_tokens], dim=1)

    _, cache = model.run_with_cache(repeated, remove_batch_dim=False)

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    scores = np.zeros((n_layers, n_heads))

    logging.info("Scanning layers for induction heads...")
    for layer in tqdm(range(n_layers), desc="Layers"):
        pattern = cache[f"blocks.{layer}.attn.hook_pattern"]
        pattern_mean = pattern.mean(dim=0)
        for head in range(n_heads):
            head_attn = pattern_mean[head]
            score, count = 0.0, 0
            for i in range(seq_len, 2 * seq_len - 1):
                target_pos = i - seq_len + 1
                score += head_attn[i, target_pos].item()
                count += 1
            scores[layer, head] = score / count

    del cache
    torch.cuda.empty_cache()
    return scores


def get_dynamic_cutoff(scores: np.ndarray, min_threshold: float = 0.1):
    """Elbow detection on sorted scores + min_threshold guard.
    Always include strong heads (>0.3) as a safety floor."""
    n_layers, n_heads = scores.shape
    flat = [(scores[l, h], l, h) for l in range(n_layers) for h in range(n_heads)]
    flat.sort(key=lambda x: x[0], reverse=True)
    sorted_scores = np.array([x[0] for x in flat])

    n_points = len(sorted_scores)
    if n_points < 2:
        return [], 0, sorted_scores

    coords = np.vstack((range(n_points), sorted_scores)).T
    line_vec = coords[-1] - coords[0]
    line_vec_norm = line_vec / np.sqrt(np.sum(line_vec ** 2))
    vec_from_first = coords - coords[0]
    scalar = np.sum(vec_from_first * np.tile(line_vec_norm, (n_points, 1)), axis=1)
    parallel = np.outer(scalar, line_vec_norm)
    perp = vec_from_first - parallel
    dist_to_line = np.sqrt(np.sum(perp ** 2, axis=1))

    search_limit = max(5, int(n_points * 0.2))
    elbow_idx = int(np.argmax(dist_to_line[:search_limit]))

    valid_mask = sorted_scores > min_threshold
    threshold_idx = int(np.sum(valid_mask))
    final_k = min(elbow_idx, threshold_idx)
    final_k = max(final_k, int(np.sum(sorted_scores > 0.3)))

    return flat[:final_k], final_k, sorted_scores


def visualize_and_save(scores: np.ndarray, task_name: str, is_ft: bool) -> None:
    save_dir = os.path.join(OUTPUT_DIR, MODEL_KEY)
    os.makedirs(save_dir, exist_ok=True)

    status = "FineTuned" if is_ft else "Base"
    suffix = task_name if is_ft else "Pretrained"

    selected, k_val, sorted_scores = get_dynamic_cutoff(scores, min_threshold=0.1)
    head_list = [{"layer": int(h[1]), "head": int(h[2]), "score": float(h[0])} for h in selected]

    logging.info(f"[Detection] Found {k_val} induction heads for {MODEL_KEY}-{task_name}")

    np.save(os.path.join(save_dir, f"induction_scores_{status}_{suffix}.npy"), scores)
    with open(os.path.join(save_dir, f"detected_heads_{status}_{suffix}.json"), "w") as f:
        json.dump({"model": MODEL_KEY, "task": task_name, "k": int(k_val), "heads": head_list}, f, indent=4)

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    sns.heatmap(
        scores, cmap="Blues", vmin=0, vmax=1.0,
        ax=axes[0], cbar_kws={"label": "Induction Score"},
    )
    axes[0].set_title(f"Induction Matrix: {MODEL_KEY} ({status} {suffix})", fontsize=14)
    axes[0].set_xlabel("Head Index")
    axes[0].set_ylabel("Layer Index")
    axes[0].invert_yaxis()

    axes[1].plot(sorted_scores, linewidth=2, label="Score Curve")
    axes[1].set_title(f"Head Distribution (Top-K={k_val} Selected)", fontsize=14)
    axes[1].set_xlabel("Rank (Head)")
    axes[1].set_ylabel("Induction Score")
    axes[1].grid(True, alpha=0.3)
    if k_val > 0:
        cutoff = sorted_scores[k_val - 1]
        axes[1].axvline(x=k_val, color="r", linestyle="--", label=f"Cutoff K={k_val}")
        axes[1].axhline(y=cutoff, color="g", linestyle="--", label=f"Threshold={cutoff:.2f}")
        axes[1].scatter([k_val], [cutoff], color="red", s=100, zorder=5)
        axes[1].text(k_val + 5, cutoff, f" K={k_val}\n Score={cutoff:.3f}", color="red")
    else:
        axes[1].text(0.5, 0.5, "NO INDUCTION HEADS FOUND",
                     ha="center", va="center", transform=axes[1].transAxes,
                     color="red", fontsize=14, fontweight="bold")
    axes[1].legend()

    img_path = os.path.join(save_dir, f"analysis_{status}_{suffix}.png")
    fig.savefig(img_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"Saved: {img_path}")


def main():
    save_dir = os.path.join(OUTPUT_DIR, MODEL_KEY)
    os.makedirs(save_dir, exist_ok=True)

    # Optional base model run
    base_npy = os.path.join(save_dir, "induction_scores_Base_Pretrained.npy")
    if RERUN_BASE or not os.path.exists(base_npy):
        logging.info("=== Base pretrained run ===")
        model = load_base_hookedtransformer()
        scores = get_induction_score_matrix(model)
        visualize_and_save(scores, task_name="Pretrained", is_ft=False)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    else:
        logging.info(f"Skipping Base run (existing file: {base_npy})")

    # Fine-tuned runs
    tasks = list(FT_MODEL_MAP.keys())
    if TASK_FILTER:
        tasks = [t for t in tasks if t in TASK_FILTER]
    if not tasks:
        logging.warning("No tasks selected after filtering; nothing to do.")
        return

    logging.info(f"Tasks to run: {tasks}")
    for task_name in tasks:
        try:
            logging.info(f"=== Full-FT run: {task_name} ===")
            subdir = FT_MODEL_MAP[task_name]
            model = load_full_ft_hookedtransformer(subdir)
            scores = get_induction_score_matrix(model)
            visualize_and_save(scores, task_name=task_name, is_ft=True)
            del model
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:
            logging.error(f"Failed on {task_name}: {e}")
            import traceback
            traceback.print_exc()
            continue


if __name__ == "__main__":
    main()
