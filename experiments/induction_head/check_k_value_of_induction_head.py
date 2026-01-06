#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator, ScalarFormatter, AutoMinorLocator
# ------------------------------------------------------------------
# 1. Helper: analyse one .npy and save two figures (Improved)
# ------------------------------------------------------------------
def analyse_distribution(npy_path: str, save_prefix: str = "plot", top_n_view: int = 500):
    """
    Analyse induction scores with adaptive visualization.
    Args:
        top_n_view: Number of top heads to show in the sorted plot (default: 500)
    """
    print(f"[INFO] Processing: {npy_path}")
    scores = np.load(npy_path)
    flat = scores.flatten()

    sns.set_theme(style="whitegrid", context="talk")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))  # Slightly wider canvas
    
    d_max = np.max(flat)
    
    # -------------------------------------------------------
    # 1. Left Plot: Histogram (unchanged, added transparency for refinement)
    # -------------------------------------------------------
    counts, bin_edges = np.histogram(flat, bins=50)
    max_count_idx = np.argmax(counts)
    mode_score = (bin_edges[max_count_idx] + bin_edges[max_count_idx+1]) / 2
    max_count_val = counts[max_count_idx]

    sns.histplot(flat, bins=50, kde=False, color="#4c72b0", alpha=0.9, edgecolor=None, ax=ax1)
    ax1.set_title("Distribution of Induction Scores")
    ax1.set_xlabel("Induction Score")
    ax1.set_ylabel("Count of Heads")
    ax1.set_yscale("log")
    
    # Mark the mode
    ax1.axvline(x=mode_score, color='green', linestyle=':', alpha=0.6, linewidth=1.5)
    ax1.text(mode_score, max_count_val * 1.1, "Peak", color='green', fontsize=10, ha='center')

    # -------------------------------------------------------
    # 2. Right Plot: Sorted Curve (extensively refined)
    # -------------------------------------------------------
    sorted_scores = np.sort(flat)[::-1]
    
    # Actual display count is the minimum (prevents error if total heads < top_n_view)
    actual_n = min(top_n_view, len(sorted_scores))
    
    # --- [Adaptive Styling Strategy] ---
    # Automatically adjust marker size and line width based on number of points
    if actual_n > 400:
        mk_size = 1.5  # Many points: use very small markers
        lw = 1.0
    elif actual_n > 200:
        mk_size = 2.5
        lw = 1.5
    else:
        mk_size = 4.0  # Fewer points: larger markers look better
        lw = 2.0

    ax2.plot(range(actual_n), sorted_scores[:actual_n],
             marker='o', markersize=mk_size, linewidth=lw, 
             color="#c44e52", markeredgewidth=0)  # Remove marker edge for smoother appearance
    
    ax2.set_title(f"Top {actual_n} Heads (Sorted)")
    ax2.set_xlabel("Rank (K)")
    ax2.set_ylabel("Induction Score")
    
    # --- [Automated X-axis Ticks] ---
    # No longer hardcoding range(0, n, 25); let Matplotlib auto-select up to 8 major ticks
    ax2.xaxis.set_major_locator(MaxNLocator(nbins=8))
    # Add minor ticks for a more professional look
    ax2.xaxis.set_minor_locator(AutoMinorLocator())
    
    # --- [Automated Y-axis Ticks] ---
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=6))
    if d_max < 0.01:
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-2, 2))
        ax2.yaxis.set_major_formatter(formatter)
    else:
        from matplotlib.ticker import FormatStrFormatter
        ax2.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))

    # Fine-tune grid lines
    ax2.grid(visible=True, which='major', linestyle='-', alpha=0.4)
    ax2.grid(visible=True, which='minor', linestyle=':', alpha=0.2)  # Minor grid is more subtle

    # =======================================================
    # Annotation Logic (adapted for large ranges)
    # =======================================================
    
    # 1. Annotate the highest score (Rank 0)
    max_score = sorted_scores[0]
    ax2.annotate(f"Max: {max_score:.4f}", 
                 xy=(0, max_score), 
                 xytext=(actual_n * 0.05, max_score),  # Dynamic offset
                 fontsize=11, fontweight='bold', color='#c44e52',
                 arrowprops=dict(arrowstyle="->", color='#c44e52', alpha=0.6))

    # 2. Annotate the mode (peak)
    rank_of_mode = (np.abs(sorted_scores - mode_score)).argmin()
    val_at_mode = sorted_scores[rank_of_mode]

    if rank_of_mode < actual_n:
        ax2.annotate(f"Mode (Peak)\nRank: {rank_of_mode}\nScore: {val_at_mode:.4f}", 
                     xy=(rank_of_mode, val_at_mode), 
                     xytext=(rank_of_mode + actual_n * 0.1, val_at_mode),  # Dynamic offset
                     fontsize=9, color='green', fontweight='bold',
                     arrowprops=dict(arrowstyle="->", color='green', alpha=0.8))
        ax2.plot(rank_of_mode, val_at_mode, 'o', color='green', markersize=mk_size*2, alpha=0.8)

    # 3. Reference dashed lines (K=10, 50, 100, 200, 500...)
    # Only draw if K is within the current display range
    potential_ks = [10, 50, 100, 200, 500, 1000]
    for k in potential_ks:
        if k < actual_n:
            s = sorted_scores[k-1]
            ax2.axvline(x=k, ls='--', c='gray', alpha=0.4, linewidth=1)
            
            # Optimize annotation position: stagger display to prevent crowding
            # Use slightly smaller font if K is large
            font_s = 9 if actual_n < 500 else 8
            ax2.text(k + actual_n * 0.01, s, f"K={k}\n{s:.4f}", fontsize=font_s, color='#555555')

    plt.tight_layout()
    for suffix in ("hist", "sorted"):
        fig.savefig(f"{save_prefix}_{suffix}.png", dpi=300)
    plt.close(fig)
    print(f"[SAVE] {save_prefix}_{{hist,sorted}}.png")
import re
from pathlib import Path
# ------------------------------------------------------------------
# 2. Batch walker (Logic Updated for Pretrained)
# ------------------------------------------------------------------
def batch_analyse(root_dir: str, out_dir: str = "plots"):
    """
    Traverse root_dir and analyse induction_scores_*.npy.
    """
    root = Path(root_dir).expanduser().resolve()
    out = Path(out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    # Regex pattern:
    # Group 1 (cond): Base, FineTuned, Pretrained
    # Group 2 (suffix): remaining part (could be task name or "Pretrained")
    rex = re.compile(r"induction_scores_(?P<cond>Base|FineTuned|Pretrained)_(?P<suffix>.+)\.npy")
    print(f"Scanning directory: {root}")
    
    count = 0
    for model_dir in root.iterdir():
        if not model_dir.is_dir():
            continue
            
        model = model_dir.name  # e.g., llama2, gpt2
        
        for npy in model_dir.rglob("induction_scores_*.npy"):
            m = rex.match(npy.name)
            
            if not m:
                continue
                
            cond_raw = m.group("cond")     # e.g., "Base", "FineTuned"
            suffix_raw = m.group("suffix") # e.g., "Pretrained", "qa_squad"
            
            # --- Logic handling ---
            if cond_raw in ["Base", "Pretrained"] or suffix_raw == "Pretrained":
                # This is a pretrained base model
                cond_display = "pretrained"
                task_display = "base"  # or "none", "general"
            else:
                # This is a fine-tuned model
                cond_display = "finetuned"
                task_display = suffix_raw
                
            # Build output prefix
            # e.g. plots/llama2_pretrained_base
            # e.g. plots/llama2_finetuned_qa_squad
            prefix = str(out / f"{model}_{cond_display}_{task_display}")
            
            try:
                analyse_distribution(str(npy), prefix)
                count += 1
            except Exception as e:
                print(f"  [Error] Failed to process {npy.name}: {e}")
    print(f"\nProcessing complete. Generated {count} sets of plots in '{out_dir}'.")

# ------------------------------------------------------------------
# 3. Entry point
# ------------------------------------------------------------------
if __name__ == "__main__":
    ROOT = "/users/sglli24/UnderstandingFineTuningViaMI/experiments/induction_head/output"
    SAVE = "induction_heads_distribution_plots"
    batch_analyse(ROOT, SAVE)