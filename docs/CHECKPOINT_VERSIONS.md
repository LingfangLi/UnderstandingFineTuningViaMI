# Checkpoint & EAP-Edge Version Provenance

This project has TWO fine-tuning pipelines, TWO checkpoint storage locations, and
overlapping EAP-edge outputs. Mixing them up silently produces "Perf∆ vs Overlap"
tables where the left and right columns describe **different models**. This
document maps every checkpoint source, pipeline, and downstream artifact so
future work can audit provenance before publishing.

> Maintained by: LingfangLi.  Last major audit: 2026-04-24.

---

## 0. The naming trap — "old" does NOT mean "non-SFT"

The repository has a `finetuned/` vs `old-version-finetuned/` split that sounds like "new vs old architecture" but is actually:

- **`old-version-finetuned/`**: from `/mnt/data1/.../old_version_finetuned_models/*.pt` — ckpts
  trained via a **manual `HookedTransformer` training loop** (Pipeline A). Saved as a single
  `.pt` state_dict file per task. **Still full SFT** (all parameters updated).
- **`finetuned/`**: from `/mnt/data1/.../fine_tuned_models/<task>/` — ckpts trained via
  **`trl.SFTTrainer`** (Pipeline B-style). Saved as standard HF directory. Also full SFT.

Verified 2026-04-24 by rerunning EAP with today's code on both source candidates and
comparing shared-edge scores against GitHub `finetuned/gpt2_yelp_finetuned_edges.csv`:

  `finetuned/` vs `fine_tuned_models/gpt2-yelp` (HF dir):    180/400 overlap, Pearson r=0.923
  `finetuned/` vs `old_version_finetuned_models/gpt2-yelp.pt`: ~116/400 overlap, Pearson uncorr.
  `old-version-finetuned/` vs `.pt` (same ckpt):             306/400 overlap, Pearson r=0.981

Both directories are SFT. The split records **two different training pipelines / recipes**,
not "SFT vs non-SFT". Anything involving Llama-2 QLoRA is separate (`cross_task_edges_qlora/`
and `qlora-finetuned/`).

### Cross-task counterpart of each

- `cross_task_edges/` (non-llama2 files, Jan 2026) pairs with **`old-version-finetuned/`**
  (both Pipeline A `.pt`). Verified: 76.5% overlap / r=0.981 when rerunning with `.pt` + current code.
- `cross_task_edges/` (llama2_* files, Apr 2026) is Pipeline B (full-FT).
- There is **no GitHub cross-task counterpart for `finetuned/`** (Pipeline B) for GPT-2/Qwen2/Llama-3.2.
  The `gpt2_cross_task_hfdir/` (2026-04-24, run on `fine_tuned_models/<task>` dirs) is the
  local equivalent; `llama3_all_edges/` and `qwen2_all_edges/` use the April `fine_tuned_model/*-full-ft-*`
  checkpoints (a different Pipeline B snapshot).

---

## 1. Two fine-tuning pipelines exist

### 1.1  Pipeline A — old HookedTransformer / QLoRA (2025-Q3 → early 2026-Q1)

- **Implementation**: manual training loop with `HookedTransformer` for small
  models; QLoRA for Llama-2.
- **Scripts** (historical, still in repo): `experiments/text_complexity/fine_tune/qwen2_{sentiment,qa,mt}_train.py`, `llama2_{sentiment,qa,mt}_train.py`, etc.
- **Output format**: single `.pt` state-dict file per model
  (e.g. `gpt2-yelp.pt`, `qwen2-squad.pt`); Llama-2 is a directory of QLoRA adapters.
- **Storage**:
  `/mnt/data1/users/sglli24/fine-tuning-project-1/old_version_finetuned_models/`
  (NB: `/mnt/data1/`, not `/mnt/scratch/` — different mount point).
- **Epochs / data**: variable per task; see original scripts' `PARAM_MAP`.
- **Timestamps**: 2025-11-21 → 2026-01-13 (most ckpts).

### 1.1b  `/mnt/data1/.../fine_tuned_models/` is the canonical "best" (user-selected)

Distinct from `/mnt/scratch/.../fine_tuned_model/*-full-ft-*/`:

- `/mnt/scratch/.../fine_tuned_model/gpt2-sst2-full-ft-20251205-172809/` top-level
  `model.safetensors` = what `SFTTrainer` saved via `load_best_model_at_end=True`
  (best by eval_loss — typically step 600).
- `/mnt/scratch/.../fine_tuned_model/gpt2-sst2-full-ft-20251205-172809/gpt2-sst2/`
  nested subdir `model.safetensors` = end-of-training (step 939).
- **`/mnt/data1/.../fine_tuned_models/gpt2-sst2/model.safetensors`** = the nested subdir
  copied over by the user (verified 2026-04-24: MD5 matches). **This is the user's
  "canonical best" — the checkpoint chosen for publication based on downstream
  accuracy, not `eval_loss`.**

Implication: if publication uses GitHub `finetuned/` (which was generated from
`/mnt/data1/.../fine_tuned_models/`), then **any new EAP run that points to
`/mnt/scratch/.../fine_tuned_model/<dir>/` top-level uses a DIFFERENT ckpt** and
will not match published results. Pair EAP runs with `/mnt/data1/.../fine_tuned_models/`
sources to stay consistent with GitHub `finetuned/` and the paper.

Yelp note: GPT-2 and Llama-3.2 Yelp were retrained on 2026-04-15 and saved to
`/mnt/scratch/.../fine_tuned_model/{gpt2-small,llama3.2-1b}-yelp-full-ft-20260415-*/`.
These April retrains were **not migrated** to `/mnt/data1/`. So `/mnt/data1/gpt2-yelp/`
(mtime 2026-01-05) is the older, published Yelp; scratch-April is a separate,
unpublished retrain.

### 1.2  Pipeline B — SFTTrainer full fine-tuning (late 2025-Q4 onwards)

- **Implementation**: standard HuggingFace `AutoModelForCausalLM` +
  `trl.SFTTrainer`, bf16, no LoRA/PEFT. This is "SFT" in the standard paper
  sense.
- **Scripts**:
  - `src/Fine_tune/Sentiment_classification/*-full.py`
  - `experiments/text_complexity/fine_tune/llama2_*_train_full.py`
    (and the Apr-2026 `qwen2_*_train_full.py` variants generated from them)
- **Output format**: standard HF directory with `model.safetensors` +
  `config.json` + `experiment_config.json`.
- **Storage**:
  `/mnt/scratch/users/sglli24/fine-tuning-project/fine_tuned_model/`
- **Timestamps**: 2025-11-24 → 2026-04-16 (staggered per task).

### 1.3  Why the two pipelines coexist

- Pipeline A was the original project implementation, tightly coupled to
  `HookedTransformer`.
- Pipeline B was introduced (late 2025) to produce standard HF checkpoints that
  downstream tools (`transformers`, `trl`, EAP) can load uniformly, and to
  obtain stronger / cleaner full-FT baselines for the paper's main tables.
- Both pipelines still coexist because early EAP runs were already done with A
  and reproducing everything under B is expensive.

---

## 2. Where each checkpoint currently lives

### 2.1  Pipeline A ckpts (used for old EAP runs Jan 15–20, 2026)

Location: `/mnt/data1/users/sglli24/fine-tuning-project-1/old_version_finetuned_models/`

| Model | Per-task files | Notes |
|---|---|---|
| GPT-2   | `gpt2-{yelp,sst2,squad,coqa,kde4,tatoeba}.pt`       | `.pt` state-dict |
| Qwen2   | `qwen2-{yelp,sst2,squad,coqa,kde4,tatoeba}.pt`       | `.pt` state-dict |
| Llama-3.2 | `llama3.2-{yelp,sst2,squad,coqa,kde4,tatoeba}.pt`  | `.pt` state-dict |
| Llama-2 | `llama2-{yelp,sst2,squad,coqa,kde4,tatoeba}/`        | QLoRA adapter dirs |

### 2.2  Pipeline B ckpts (used for current Perf∆ matrices and Apr-2026 EAP runs)

Location: `/mnt/scratch/users/sglli24/fine-tuning-project/fine_tuned_model/`

| Model | Per-task dir pattern | Notable dates |
|---|---|---|
| GPT-2   | `gpt2-{small-,}{TASK}-full-ft-YYYYMMDD-HHMMSS/` | Yelp **2026-04-15**, others Dec 2025 – Jan 2026 |
| Qwen2   | `qwen2-0.5b-{TASK}-full[-ft]-YYYYMMDD-HHMMSS/`   | All Nov–Dec 2025 (no Apr refresh needed) |
| Llama-3.2 | `llama3.2-1b-{TASK}-full[-ft]-YYYYMMDD-HHMMSS/` | Yelp **2026-04-15**, others Dec 2025 – Jan 2026 |
| Llama-2 | `llama2-{7b-,}{TASK}-full[-YYYYMMDD-HHMMSS]/`   | All 2026-04-16 full-FT refresh |

### 2.3  Known bugged ckpt

- `llama2-7b-yelp-full-BAD-was-squad/` (2026-04-03): data-loader bug, the
  "Yelp" model was actually trained on SQuAD. **Must not be used.** Renamed
  with `BAD-was-squad` suffix to signal this; full replacement at
  `llama2-yelp-full-20260416-135323`.

### 2.4  Why Yelp was retrained on 2026-04-15 for GPT-2 / Llama-3.2

- The Llama-2 bug (§2.3) prompted a broader audit of Yelp pipelines. Even
  though GPT-2 and Llama-3.2 did **not** have the same bug, their earlier
  Pipeline A ckpts were superseded on 2026-04-15 with fresh Pipeline B
  (SFTTrainer) runs for consistency with Llama-2's fixed Yelp FT.
- The earlier Pipeline A GPT-2/Llama-3.2 Yelp ckpts remain in
  `old_version_finetuned_models/` and are **not the current versions**.

---

## 3. EAP edge outputs — which pipeline each dir corresponds to

Location prefix: `output/EAP_edges/`

| Directory | Model coverage | Pipeline source | Ckpt date range | Notes |
|---|---|---|---|---|
| `old-version-finetuned/`   | GPT-2, Qwen2, Llama-3.2, Llama-2 | A (`.pt` / QLoRA) | Jan 2026 | Own-task FT edges only |
| `cross_task_edges/` (GPT-2/Qwen2/Llama-3.2 files) | GPT-2, Qwen2, Llama-3.2 | **A** (verified 2026-04-24 — reran with `.pt` + current code on GPT-2 Yelp-FT × SST-2: 76.5% top-400 match, shared-edge Pearson r=0.98; vs Pipeline B ckpt: only 28.5% / r=-0.32) | 2026-01-15 – 2026-01-20 | top-400. Cross-task (FT-on-X × eval-on-Y). Code was pre-Jan-30 `attribute_mem.py` (before `FORCE_MAX_LENGTH=256` truncation and hook refactor) |
| `cross_task_edges/` (llama2_* files) | Llama-2 | **B full-FT** | 2026-04-10 / 2026-04-17 | top-400 |
| `cross_task_edges_qlora/`  | Llama-2 | A (QLoRA) | Jan 2026 | top-400 |
| `finetuned/`               | all 4 models | Mixed — **verify per file** | Jan 2026 – Apr 2026 | Own-task FT; some files may be A, some B |
| `finetuned_top2000/`       | Llama-2 | B full-FT | 2026-04 | top-2000 variant |
| `pretrained/`              | all 4 models | N/A (base model, no FT) | Mixed dates | Base-model EAP runs |
| `gpt2_all_edges/`          | GPT-2 | **B full-FT** | **2026-04-24** | **ALL edges** (no top-K). Includes 6 pretrained + 6 own-task FT + 30 cross-task FT = 42 files of 32,491 edges each. **Most reliable GPT-2 source.** |

### 3.1  Overlap summary tables

Location: `output/EAP_edges/cross_task_edges/summary_tables/`

| File | What it aggregates | Implied source |
|---|---|---|
| `CrossTask_Overlap_NewEdges_Combined.csv` | overlap rows for `gpt2 / qwen2 / llama3.2 / llama2_qlora / llama2_full_ft` | Draws from `cross_task_edges/` → so GPT-2/Qwen2/Llama-3.2 rows are **Pipeline A**, `llama2_qlora` from `cross_task_edges_qlora/`, `llama2_full_ft` from `cross_task_edges/llama2_*` (April Pipeline B files) |

**Consequence**: in `src/Fine_tune/cross_eval/perf_overlap_table.tex`
(generated 2026-04-24), the **Perf∆ column** and the **Overlap column** for
GPT-2 / Qwen2 / Llama-3.2 describe slightly different ckpts:

- Perf∆ uses Pipeline B (current SFT) numbers
- Overlap uses Pipeline A (old `.pt`) edges

Llama-2 rows are internally consistent because both columns explicitly use
the QLoRA variant (or both use full-FT, depending on which row the reader looks
at).

---

## 4. Action items / outstanding work

| # | Action | Status | Rationale |
|---|---|---|---|
| 1 | Re-generate GPT-2 cross-task EAP edges from Pipeline B ckpts | **Done 2026-04-24** — see `gpt2_all_edges/` (all-edges variant). top-400 derivable via a 1-line pandas filter. | GPT-2 old edges used Pipeline A; Perf∆ uses Pipeline B |
| 2 | Re-generate Llama-3.2 cross-task EAP edges from Pipeline B ckpts | **Pending** — needs a runner analogous to `src/EAP/run_gpt2_all_edges.sh`. Est. 20–40 min on an L4. | Same inconsistency as GPT-2 |
| 3 | Re-generate Qwen2 cross-task EAP edges from Pipeline B ckpts | **Pending** — same | Same inconsistency as GPT-2 |
| 4 | Re-build `CrossTask_Overlap_NewEdges_Combined.csv` after #1–#3 | Pending | Needed before any new `perf_overlap_table.tex` can claim full pipeline consistency |
| 5 | Document this on a per-table-caption basis in the paper | Pending | Reviewers will notice if Perf∆ and Overlap use different model snapshots |

### 4.1  How to re-generate (template)

- For GPT-2, use the already-finished `output/EAP_edges/gpt2_all_edges/` and
  derive top-400 via:
  ```python
  import pandas as pd
  d = pd.read_csv(PATH)
  d['abs'] = d['score'].abs()
  top400 = d.nlargest(400, 'abs')[['edge', 'score']]
  top400.to_csv(OUT_PATH, index=False)
  ```
- For Llama-3.2 / Qwen2: copy `src/EAP/run_gpt2_all_edges.sh` → adjust
  `GPT2_FT` paths to the Pipeline B `fine_tuned_model/llama3.2-...` /
  `fine_tuned_model/qwen2-...` dirs, change `base_model` to
  `meta-llama/Llama-3.2-1B` / `Qwen/Qwen2-0.5B`. Output to a parallel
  `llama3_all_edges/` / `qwen2_all_edges/` dir.

---

## 5. How to inspect which pipeline a ckpt belongs to

```bash
# Rule of thumb:
#   Pipeline A (.pt files):     ls <path>.pt
#   Pipeline B (HF safetensors): ls <dir>/model.safetensors  + config.json + experiment_config.json
#   Llama-2 QLoRA:               ls <dir>/adapter_config.json + adapter_model.safetensors

ls <model_dir> | head
cat <model_dir>/experiment_config.json 2>/dev/null | head -20
```

If unsure which ckpt generated a given EAP edge CSV:
1. Check the CSV's `mtime` (the ckpt must have existed before that date).
2. Look at git log around that date for the EAP shell script (e.g.
   `git log --all --since=... -p src/EAP/run_cross_task_generation.sh |
   grep MODEL_DIR`) — the script literally records `MODEL_DIR=...` so the
   ckpt location is explicit.
3. As a last resort, re-run EAP on the candidate ckpt and check top-400 edge
   overlap with the CSV: 100 % = same ckpt; <40 % = different.

---

## 6. Glossary

- **Pipeline A** / **"old"** / **"old_version_finetuned_models"**: early training via `HookedTransformer` manual loop (small models) or QLoRA (Llama-2). `.pt` files at `/mnt/data1/.../old_version_finetuned_models/`.
- **Pipeline B** / **"new"** / **"full-FT"**: `SFTTrainer` full fine-tuning; standard HF dirs at `/mnt/scratch/.../fine_tuned_model/`.
- **SFT**: supervised fine-tuning (both A's HookedTransformer loops and B's SFTTrainer are SFT in this broad sense — only Llama-2 QLoRA is not).
- **Perf∆ (pp)**: `(m_FT − m_Base) × 100`, absolute percentage-point change. Used in `perf_overlap_table.tex`.
- **Overlap (%)**: `|edges_FT-on-F ∩ edges_FT-on-T| / 400 × 100`, both sides computed on the same corrupted-row-task data T. Diagonals are 100 % by construction.
