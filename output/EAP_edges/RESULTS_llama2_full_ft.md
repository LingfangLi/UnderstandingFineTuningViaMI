# EAP Edges — Llama-2-7B Full Fine-Tuning Results

Scope: Edge Attribution Patching (EAP) results for Llama-2-7B **full fine-tuning** across 6 tasks
(`sentiment_yelp`, `sentiment_sst2`, `qa_squad`, `qa_coqa`, `mt_kde4`, `mt_tatoeba`).

## File layout

| Path | Contents |
|---|---|
| `finetuned/llama2_{task}_finetuned_edges.csv` | Top-400 most important edges for FT model, same task |
| `cross_task_edges/llama2_Finetuned-{A}_Corrupted-Data_{B}_finetuned_edges.csv` | Top-400 edges for FT task A when corrupted with task B data (30 combinations) |
| `overlap/llama2_{task}_overlap.csv` | Per-edge overlap between PT base and FT top-400 |
| `overlap/llama2_pt_vs_ft_overlap_summary.csv` | Aggregate Jaccard/retention stats per task |

## 1. PT vs FT circuit overlap (top-400 edges)

Source: `overlap/llama2_pt_vs_ft_overlap_summary.csv`

| Task  | n_pt | n_ft | Common | Jaccard | % PT retained | % FT from PT |
|-------|------|------|--------|---------|---------------|--------------|
| squad   | 401 | 404 | 260 | **0.477** | 64.8% | 64.4% |
| yelp    | 400 | 400 | 222 | 0.384 | 55.5% | 55.5% |
| tatoeba | 400 | 403 | 217 | 0.370 | 54.3% | 53.8% |
| coqa    | 400 | 400 | 200 | 0.333 | 50.0% | 50.0% |
| kde4    | 401 | 405 | 172 | 0.271 | 42.9% | 42.5% |
| sst2    | 401 | 400 | 120 | **0.176** | 29.9% | 30.0% |

**Finding.** Circuit continuity from PT → FT varies dramatically by task. SQuAD FT stays closest to the pretrained circuit (~65% retained), whereas SST-2 FT rebuilds the circuit almost entirely (only 30% retained). KDE4 and SST-2 show the largest structural rewrite.

## 2. Layer distribution of top-400 EAP heads

Head layer indices were extracted from every `a<L>.h<H>` token in the edge strings and counted per task.

| Task | Top-5 layers (layer: head-mentions) | Pattern |
|---|---|---|
| yelp    | L2:41, L4:35, L3:34, L5:33, L6:29 | **Early-layer heavy (L2–L6)** |
| squad   | L2:45, L3:39, L4:38, L5:34, L6:34 | **Early-layer heavy (L2–L6)** |
| coqa    | L2:45, L4:44, L5:43, L3:39, L6:37 | **Early-layer heavy (L2–L6)** |
| sst2    | L19:25, L31:21, L23:16, L20:15, L16:14 | Mid-to-late, dispersed |
| kde4    | **L31:81**, L0:34, L3:32, L2:30, L4:20 | **Boundary layers (L0 + L31)** |
| tatoeba | **L31:153**, L0:119, L1:74, L30:12, L28:2 | **Extreme boundary (L0/L1 + L30/L31)** |

**Finding.** Two circuit archetypes emerge:
- **QA / sentiment family** (yelp, squad, coqa) — attention importance concentrates in early layers L2–L6.
- **MT family** (kde4, tatoeba) — importance collapses to the first (L0/L1) and last (L30/L31) layers, with L31 dominating in tatoeba (153 head-mentions in a single layer).

SST-2 sits in its own class: no early-layer concentration and no boundary collapse, instead spreading across mid-to-late layers (L16–L31).

## 3. Cross-task EAP edges

30 CSVs (6 FT tasks × 5 corrupted-data tasks) under `cross_task_edges/`. These answer
*"Is the important-edge set for FT task A stable when the corrupting data comes from task B?"*
Pairwise Jaccard between cross-task edge sets has not yet been summarized — TODO if needed.

## Raw derived summary

See `../experiments/component_distribution/RESULTS_llama2_full_ft.md` for the top-400 edge
component-type breakdown (heads vs MLPs vs IO).
