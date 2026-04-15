# Text Complexity Control — Llama-2-7B Full Fine-Tuning Results

Scope: 3×3 cross evaluation of Llama-2-7B full fine-tuning, splitting training and test
sets by lexical complexity (simple / complex), on 3 tasks.

## Experimental design

- **Tasks covered**: `yelp` (sentiment), `squad` (QA), `tatoeba` (MT, en→fr)
- **Training regimes**: `base` (no FT, zero-shot baseline), `simple-FT` (trained on lexically simple subset), `complex-FT` (trained on lexically complex subset)
- **Test splits**: `test_simple`, `test_complex`
- Full fine-tuning (not QLoRA)

## File layout

| Path | Contents |
|---|---|
| `fine_tune/results_full/summary_long.csv` | Long-format eval table (25 rows, 3 tasks × train × test × metric) |
| `fine_tune/results_full/eval_results_*.json` | Per-run raw evaluation JSON |
| `fine_tune/llama2_{sentiment,qa,mt}_train_full.py` | Full-FT training scripts |
| `fine_tune/llama2_{sentiment,qa,mt}_eval_full.py` | Full-FT evaluation scripts |
| `fine_tune/submit_tc_full_ft_train.sh` | SLURM training submission |
| `fine_tune/submit_tc_full_ft_eval.sh` | SLURM evaluation submission |
| `fine_tune/aggregate_eval_results.py` | Builds `summary_long.csv` from JSON |
| `matrix_analysis/*.txt` | Lexical-complexity filter indices per task |
| `run_complexity.py`, `run_stanza.py` | Complexity scoring entrypoints |

## Results (excerpted from `summary_long.csv`)

### Yelp — accuracy

| Train \ Test | simple | complex |
|---|---|---|
| base        | 0.036 | 0.029 |
| simple-FT   | 0.936 | 0.940 |
| complex-FT  | 0.939 | 0.945 |

### SQuAD — F1 / EM

| Train \ Test | F1 (simple) | F1 (complex) | EM (simple) | EM (complex) |
|---|---|---|---|---|
| base        | 0.513 | 0.537 | 0.337 | 0.374 |
| simple-FT   | 0.757 | 0.844 | 0.603 | 0.717 |
| complex-FT  | 0.757 | 0.839 | 0.602 | 0.705 |

### Tatoeba — BLEU (nltk)

| Train \ Test | simple | complex |
|---|---|---|
| base        | 0.268 | 0.267 |
| simple-FT   | 0.352 | 0.376 |
| complex-FT  | 0.345 | 0.361 |

## Findings

1. **Training complexity does not meaningfully affect post-FT performance.** For all three tasks, simple-trained and complex-trained models perform within 0.01–0.02 of each other on every test split.
2. **The effect direction is inconsistent across tasks:**
   - Yelp: complex-FT ≥ simple-FT on both test splits
   - SQuAD: simple-FT slightly beats complex-FT on F1 (0.844 vs 0.839 on test-complex)
   - Tatoeba: simple-FT beats complex-FT on test-complex (0.376 vs 0.361)
3. **No "complexity preference" phenomenon** — the hypothesis that training on one complexity tier would particularly boost in-distribution performance is not supported. The FT signal swamps whatever small bias the complexity split induces.

## Known caveat

The lexically complex subset is ~2× larger than the simple subset and was not rebalanced (accepted explicitly — not a bug). This imbalance does not invalidate the finding above, but readers should be aware when comparing raw sample counts.
