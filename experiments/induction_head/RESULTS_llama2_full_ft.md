# Induction Head Analysis — Llama-2-7B Full Fine-Tuning Results

Scope: Induction head detection and induction-vs-EAP overlap for Llama-2-7B full
fine-tuning across 6 tasks.

## File layout

| Path | Contents |
|---|---|
| `output/llama2/induction_scores_FineTuned_{task}.npy` | 32×32 induction score matrix per task |
| `output/llama2/detected_heads_FineTuned_{task}.json` | Heads with induction score above threshold (0.3) |
| `output/llama2/heatmap_FineTuned_{task}.png` | Heatmap of induction scores, FT model |
| `output/llama2/heatmap_Base_Pretrained.png` | Heatmap of induction scores, base (PT) model |
| `output/llama2/scatter_{task}.{png,pdf}` | PT vs FT induction-score scatter |
| `output/llama2/analysis_FineTuned_{task}.png` | Per-head analysis panel |
| `output/induction_overlap_stats_edges400.csv` | Overlap with EAP top-400 (all models, includes llama2) |
| `output/induction_overlap_topK_sweep.csv` | Overlap vs top-K sweep (K=400,1000,2000,5000) |
| `NOTES_llama2_full_ft_overlap.md` | Detailed analysis notes |
| `detect_induction_head_llama2_full.py` | Detection script for llama2 full-FT |
| `submit_induction_llama2_full.sh` | SLURM submission wrapper |
| `overlap_analysis.py` | EAP × induction overlap script |

## 1. Induction head counts (threshold 0.3)

| Task | # Induction heads | Notes |
|---|---|---|
| sentiment_yelp | 53 | Normal, max score ~0.98 |
| sentiment_sst2 | 52 | Normal, max score ~0.98 |
| qa_coqa        | 61 | Normal, max score ~0.98 |
| **qa_squad**   | **15** | **Suppressed** — max score only 0.503 |
| mt_kde4        | 58 | Normal |
| mt_tatoeba     | 49 | Normal |

**Finding.** SQuAD FT uniquely collapses the induction-head population. All other tasks keep
49–61 heads with induction scores above 0.9, whereas SQuAD FT leaves only 15 heads and even
the best of them sits at 0.503. This is consistent with the EAP finding that SQuAD
top-400 edges concentrate in early layers (L2–L6) rather than the induction band (L6–L26).

## 2. Induction × EAP overlap (top-400 edges)

Source: `output/induction_overlap_stats_edges400.csv` (llama2 rows only)

| Task | # Induction | # EAP-important heads | Overlap | Recall (%) | Precision (%) |
|---|---|---|---|---|---|
| yelp    | 53 | 263 | 4 | 7.55 | 1.52 |
| sst2    | 52 | 94  | **0** | **0.00** | **0.00** |
| squad   | 15 | 270 | 2 | 13.33 | 0.74 |
| coqa    | 61 | 227 | 12 | 19.67 | 5.29 |
| kde4    | 58 | 166 | **0** | **0.00** | **0.00** |
| tatoeba | 49 | 88  | **0** | **0.00** | **0.00** |

**Finding.** Three of six tasks (sst2, kde4, tatoeba) show **zero overlap** between
canonical induction heads and EAP-important heads. Even where overlap is non-zero,
recall stays below 20%.

Contrast with other models on the same overlap metric (same CSV):
- **gpt2 qa_coqa**: recall 79.2%, precision 17.6%
- **llama3 qa_coqa**: recall 21.0%, precision 8.0%
- **qwen2 mt_kde4**: recall 30.3%, precision 8.6%

Llama-2 full-FT is the only model family where the induction and EAP head sets are
essentially disjoint for most tasks.

## 3. Top-K sweep — is it really disjoint, or just hidden below top-400?

Source: `output/induction_overlap_topK_sweep.csv` (K ∈ {400, 1000, 2000, 5000})

Sweeping K does not materially move the llama-2 numbers. The canonical high-score induction
heads (L11.H15, L8.H26, L16.H19, L6.H9, L7.H4, L21.H30, L17.H22, L11.H2) remain absent
from the EAP important set even at K=5000. See `NOTES_llama2_full_ft_overlap.md` for the
full analysis, including per-task spatial scatterplots showing the two head populations
live in disjoint layer bands.

## Cross-reference

- Raw EAP edges: `../../output/EAP_edges/finetuned/llama2_*_finetuned_edges.csv`
- EAP layer distribution: `../../output/EAP_edges/RESULTS_llama2_full_ft.md`
