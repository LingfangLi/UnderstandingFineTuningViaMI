# Component Distribution — Llama-2-7B Full Fine-Tuning Results

Scope: Per-task component breakdown (attention heads / MLPs / IO) of the top-400 EAP edges
for Llama-2-7B full fine-tuning across 6 tasks.

## File layout

| Path | Contents |
|---|---|
| `figure/llama2_{task}_top_400_edges_component_distribution.pdf` | Component distribution figure per task |
| `component_distribution_analysis.py` | Analysis script |

Six figures produced, one per task:
`llama2_yelp`, `llama2_sst2`, `llama2_squad`, `llama2_coqa`, `llama2_kde4`, `llama2_tatoeba`.

## How to read the figures

Each figure breaks down the top-400 edges of the corresponding FT task into component types
(attention-head source/target, MLP source/target, model input/output) and visualizes their
relative frequency.

## Cross-reference

- Raw edge CSVs driving these figures: `../../output/EAP_edges/finetuned/llama2_{task}_finetuned_edges.csv`
- Layer distribution of the same edges: `../../output/EAP_edges/RESULTS_llama2_full_ft.md`
- Component-type breakdown for other models (gpt2, llama3.2, qwen2) is available in the same `figure/` dir for comparison.
