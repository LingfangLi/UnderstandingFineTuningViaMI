# Llama-2-7B Full-FT: Induction Heads vs. EAP Top-400 Edges

**Date:** 2026-04-11
**Source:**
- `experiments/induction_head/output/llama2/detected_heads_FineTuned_*.json`
- `output/EAP_edges/finetuned/llama2_*_finetuned_edges.csv`
- Overlap summary: `experiments/induction_head/output/induction_overlap_stats_edges400.csv`

## TL;DR

For Llama-2-7B full fine-tuned models, the set of "induction heads" (detected
via synthetic `[A,B,…,A,B,…]` sequences) and the set of "task-critical heads"
(heads touched by the top-400 EAP edges) are **almost disjoint**. Three of six
tasks show **zero overlap**.

This is **not a detection bug**. Induction heads and EAP-important heads live
in **completely different layer regions**, and full-FT's task gains appear to
flow through the input/output boundary heads rather than through the
mid-layer induction lookup mechanism.

## Overlap Table (Top-400 edges, llama-2 rows)

| Task | #Induction | #Heads-in-top400 | Overlap | Recall | Precision |
|---|---|---|---|---|---|
| sentiment_yelp | 53 | 263 | 4 | 7.5% | 1.5% |
| sentiment_sst2 | 52 | 94  | **0** | **0.0%** | **0.0%** |
| qa_squad       | **15** | 270 | 2 | 13.3% | 0.7% |
| qa_coqa        | 61 | 227 | 12 | 19.7% | 5.3% |
| mt_kde4        | 58 | 166 | **0** | **0.0%** | **0.0%** |
| mt_tatoeba     | 49 | 88  | **0** | **0.0%** | **0.0%** |

## Where the two sets actually live

### Induction heads (stable across tasks)

llama-2 induction heads cluster in **middle layers L6–L26**, and the same
heads re-appear across sst2 / kde4 / tatoeba:

- L11.H15  (0.90–0.98, strongest)
- L8.H26
- L16.H19
- L6.H9
- L7.H4
- L21.H30, L17.H22

→ These are the model's canonical induction heads from pretraining, and
**full-FT on sentiment / MT does not break them**.

### EAP-top heads (highly task-dependent, always at the boundary)

| Task | Where EAP top-10 heads sit | Order of magnitude |
|---|---|---|
| mt_kde4    | **Almost entirely L31** (L31.H6/H24/H14/H4/…) | ~1–4 |
| mt_tatoeba | **L31 + L0/L1**                             | **~14–47** |
| sentiment_sst2 | Late layers L18–L31 (L19.H28, L20.H16, L31.H11, …) | ~0.2–0.8 |
| qa_squad   | **Early layers L2–L8** (L3.H17, L4.H5, L6.H6/H8, L2.H21, …) | ~2–4 |

- **MT → final-layer "logit writer" heads dominate**, with tatoeba's L31.H15
  accumulated score at **47**, ~10× anything in mid-layers. Plus the
  embedding-reading L0/L1 heads.
- **Sentiment → late-layer readout**, classic classifier pattern reading
  `[pos/neg]` out of the residual stream.
- **SQuAD → early-layer context encoding**, qualitatively different from the
  other tasks. See SQuAD anomaly note below.

## Interpretation

Two non-exclusive readings of the same data:

**Reading A — "Induction is not the bottleneck"**
Full-FT's task gains on llama-2 flow through **boundary heads**
(embedding-read at L0/L1, logit-write at L31, classifier readout at L18–L31),
not through re-tuning the mid-layer induction lookup. EAP's gradient sees the
steep payoff at the boundary and ignores the stable mid-layer machinery.

**Reading B — "Induction is preserved, readout is re-wired"**
The induction heads are still doing their pretrained job — they are just not
on the critical path that the task loss uses. Full-FT **stacks a
task-specific readout on top of an intact induction stack**, rather than
changing the induction layer itself.

Both readings are consistent with the zero overlaps on sst2 / kde4 / tatoeba
and with the very high induction scores (0.9+) still observed on those tasks.

## SQuAD anomaly (worth a separate look)

- Only **15** induction heads detected for llama-2 squad (other tasks: 49–61).
- Top induction score is **0.503**, versus 0.90–0.98 for every other task.
- EAP top-400 heads on squad concentrate in **L2–L8**, not L31 like the MT
  tasks and not L18–L31 like sst2.

→ SQuAD is the **only** llama-2 task where full-FT appears to
**actively suppress induction behavior**, and it's also the only task whose
EAP profile is dominated by **early** layers. These two facts are probably
related: SQuAD requires re-encoding long context aggressively, so full-FT
re-allocates capacity out of the mid-layer induction channel and into early
context encoding.

This is a potential standalone observation for the paper: **QA-style full-FT
changes the induction mechanism itself, while sentiment/MT full-FT leaves it
untouched and routes around it.**

## Caveats

- These are overlaps at **top-400 edges** only. A head missing from the top
  400 does not mean it's irrelevant — it may still matter at rank 500–2000.
  **Important discovery (2026-04-11)**: the saved CSVs in
  `output/EAP_edges/finetuned/*.csv` are **pre-truncated to 400 rows** at
  EAP attribution time, not at save time. `src/EAP/eap_unified.py:281-283`
  uses `scores[-top_k]` as a hard threshold and filters the graph *before*
  saving. The CLI default is `--top_k 400` and every CSV we have was
  generated with this value. A post-hoc top-K sweep on the existing CSVs is
  therefore **impossible** — the data past rank 400 was never computed.
  Resolving the rank-500–2000 question requires **re-running EAP with a
  larger `--top_k`** (e.g. 2000) on each (model, task) pair. This is GPU
  work: ~20 min per run on A100, ~24 runs total if done for every
  (model, task).

## Top-K sweep follow-up (2026-04-11, llama-2 only, top_k=2000)

Re-ran EAP with `--top_k 2000` on llama-2 for the three zero-overlap tasks
(sst2, kde4, tatoeba) to test whether induction heads were "hidden" at
rank 401–2000. CSVs saved to `output/EAP_edges/finetuned_top2000/`. Result:

| Task | K=400 | K=800 | K=1200 | K=1600 | K=2000 |
|---|---|---|---|---|---|
| sentiment_sst2 | 0.0% | 3.8% | 7.7% | 7.7% | 7.7% |
| mt_kde4        | 0.0% | 1.7% | 6.9% | 6.9% | 10.3% |
| mt_tatoeba     | 0.0% | 2.0% | 6.1% | 10.2% | 14.3% |

(Recall = induction heads found ∩ EAP top-K head set.)

Key observations:

1. **Recall saturates well below 15%**. Widening the search 5× only raised
   overlap by 4–7 heads per task. sst2 recall completely plateaus at 7.7%
   from K=1200 onward.

2. **The canonical, highest-scoring induction heads are PERMANENTLY missing
   from top-2000**, across all three tasks. The same ~7 heads are invisible
   to EAP regardless of task type:

   | Head | sst2 | kde4 | tatoeba | Status |
   |---|---|---|---|---|
   | **L11.H15** | 0.981 | 0.974 | 0.900 | Never in top-2000 |
   | **L8.H26**  | 0.871 | 0.924 | 0.798 | Never in top-2000 |
   | **L16.H19** | 0.732 | 0.751 | 0.692 | Never in top-2000 |
   | **L6.H9**   | 0.703 | 0.794 | 0.638 | Never in top-2000 |
   | **L7.H4**   | 0.667 | 0.692 | 0.589 | Never in top-2000 |
   | L21.H30     | 0.554 | 0.601 | 0.463 | Never in top-2000 |
   | L17.H22     | 0.525 | 0.463 | 0.461 | Never in top-2000 |
   | L11.H2      | 0.433 | 0.513 | 0.450 | Never in top-2000 |

3. **The 4–7 overlap heads that DO appear at top-2000 are all weak induction
   heads** (induction score < 0.5, typically ~0.1–0.4). They are not the
   real induction machinery — they're marginal detections that happen to
   also be marginally on the task path.

4. **Layer distribution of newly added heads (rank 401-2000) is broad and
   NOT concentrated in the mid-layer induction zone** (L6–L21). For the MT
   tasks, the new heads are actually concentrated more in early layers
   (L1, L4–L10) than in the induction zone. EAP is spending its
   rank-401-2000 "budget" on expanding encoding/routing paths, not on
   filling in induction heads.

### Updated conclusion

The "rank-ordering noise" explanation for zero overlap is **rejected**.

Full-FT llama-2's task-gradient path **categorically bypasses** the canonical
mid-layer induction heads. L11.H15 and its siblings retain their 0.9+
induction scores (so the mechanism is intact), but under EAP attribution for
the task loss their contribution is effectively zero — they don't appear at
rank 1, rank 400, or rank 2000.

This materially strengthens **Reading A** ("induction is not the
bottleneck"): llama-2's induction heads are a stable pretraining substrate,
and full-FT builds its task-specific machinery on paths that don't route
through them. Reading B ("induction preserved but readout rewired on top")
is still consistent, but only in a weak form — the induction→readout path
would have to be completely non-gradient-visible from the task loss.

The definitive test remains the ablation study: patch out L11.H15 etc.
during eval and see whether sst2/kde4/tatoeba accuracy drops. If it doesn't,
Reading A is fully confirmed: these heads are decoration, not load-bearing,
for full-FT llama-2 on these tasks.
- The induction detection uses synthetic random tokens (`seq_len=50`,
  `batch_size=1`). The 0.9+ scores across five of six llama-2 tasks give
  strong evidence that the detection is stable, but squad's k=15 / max=0.503
  should be double-checked by re-running detection with a larger batch size
  before drawing strong conclusions from the squad anomaly.
- Functional importance is **not tested** here. Ablation of the top-10
  induction heads (L11.H15, L8.H26, L16.H19, L6.H9, L7.H4, …) on downstream
  task loss would answer whether these heads are causally necessary, even
  when they don't appear in the top-400 edges.

## Suggested follow-ups

1. **Top-K sweep**: re-run `overlap_analysis.py` at top-1000 and top-2000
   edges. Plot recall as a function of K.
2. **SQuAD diff heatmap**:
   `induction_scores_FineTuned_qa_squad − induction_scores_Base_Pretrained`
   to visualize which heads lost induction behavior.
3. **Induction-head ablation study**: patch out the top-10 induction heads
   during eval on each task. If sst2/kde4/tatoeba accuracy/BLEU drops, then
   induction is functionally necessary even though it's not EAP-selected.
