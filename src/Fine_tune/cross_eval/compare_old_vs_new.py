"""Compare old vs new pp-Delta matrices per model and summarize differences.

Old sources:
  - gpt2, llama3.2:  old_raw_data_gpt2_llama3.csv  (paper's original runs)
  - qwen2:           qwen_matrix_results.csv       (old full-FT checkpoint)
  - llama2:          llama2_matrix_results.csv     (QLoRA version)

New sources:
  - gpt2, llama3.2:  gpt2_/llama3_matrix_results_full.csv (current full-FT)
  - qwen2:           qwen2_matrix_results_full.csv        (current full-FT rerun)
  - llama2:          llama2_full_ft_matrix_results.csv    (full fine-tuning)

For old-vs-new tasks: the old GPT-2 / Llama-3.2 runs used Twitter (not SST-2),
so only the 5-task subset {yelp, squad, coqa, kde4, tatoeba} is comparable.
Qwen2 old and Llama-2 QLoRA already use SST-2, so 6x6 comparable.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

HERE = Path(__file__).resolve().parent
TASK_METRIC = {"sst2": "Accuracy", "yelp": "Accuracy",
               "squad": "F1", "coqa": "F1",
               "kde4": "BLEU", "tatoeba": "BLEU"}


def read_matrix(path: Path) -> dict:
    out = defaultdict(dict)
    if not path.exists():
        return out
    with path.open() as f:
        for row in csv.DictReader(f):
            src = row["Model_Source"].strip()
            evl = row["Eval_Task"].strip()
            cell = {}
            for k in ("Accuracy", "F1", "EM", "BLEU"):
                v = (row.get(k) or "").strip()
                if v:
                    try:
                        cell[k] = float(v)
                    except ValueError:
                        pass
            if cell:
                out[src][evl] = cell
    return out


def read_old_raw(model: str) -> dict:
    out = defaultdict(dict)
    with (HERE / "old_raw_data_gpt2_llama3.csv").open() as f:
        for line in f:
            if line.startswith("#") or line.startswith("Model,"):
                continue
            parts = [p.strip() for p in line.rstrip("\n").split(",")]
            if len(parts) < 7:
                continue
            m, src, evl, acc, f1, em, bleu = parts
            if m != model:
                continue
            cell = {}
            for name, val in [("Accuracy", acc), ("F1", f1),
                              ("EM", em), ("BLEU", bleu)]:
                if val:
                    try:
                        cell[name] = float(val)
                    except ValueError:
                        pass
            if cell:
                out[src][evl] = cell
    return out


def pp_matrix(perf: dict, tasks: list[str]) -> dict:
    """Return {(src, evl): pp}."""
    out = {}
    for src in tasks:
        for evl in tasks:
            metric = TASK_METRIC[evl]
            base = perf.get("Base_Model", {}).get(evl, {}).get(metric)
            ft = perf.get(src, {}).get(evl, {}).get(metric)
            if base is None or ft is None:
                continue
            out[(src, evl)] = (ft - base) * 100.0
    return out


COMPARISONS = [
    ("GPT-2 Small", "old raw", "new full-FT",
     read_old_raw("gpt2"),
     read_matrix(HERE / "gpt2_matrix_results_full.csv"),
     ["yelp", "squad", "coqa", "kde4", "tatoeba"]),
    ("Llama-3.2-1B", "old raw", "new full-FT",
     read_old_raw("llama3.2"),
     read_matrix(HERE / "llama3_matrix_results_full.csv"),
     ["yelp", "squad", "coqa", "kde4", "tatoeba"]),
    ("Qwen2-0.5B", "old full-FT", "new full-FT",
     read_matrix(HERE / "qwen_matrix_results.csv"),
     read_matrix(HERE / "qwen2_matrix_results_full.csv"),
     ["yelp", "sst2", "squad", "coqa", "kde4", "tatoeba"]),
    ("Llama-2-7B", "QLoRA (paper)", "full-FT (current)",
     read_matrix(HERE / "llama2_matrix_results.csv"),
     read_matrix(HERE / "llama2_full_ft_matrix_results.csv"),
     ["yelp", "sst2", "squad", "coqa", "kde4", "tatoeba"]),
]


def diag_off_split(pp_map: dict):
    diag = [v for (s, e), v in pp_map.items() if s == e]
    off = [v for (s, e), v in pp_map.items() if s != e]
    return diag, off


def sign_label(v):
    return "↑" if v > 0 else ("↓" if v < 0 else "·")


print("=" * 100)
print(f"{'Model':<15}  {'Variant':<20}  {'Diag μ±σ':<18}  {'Off-diag μ±σ':<20}  {'|diag| forget rate'}")
print("=" * 100)
for name, old_label, new_label, perf_old, perf_new, tasks in COMPARISONS:
    pp_old = pp_matrix(perf_old, tasks)
    pp_new = pp_matrix(perf_new, tasks)

    for label, pp in [(old_label, pp_old), (new_label, pp_new)]:
        diag, off = diag_off_split(pp)
        dm = mean(diag) if diag else float('nan')
        ds = pstdev(diag) if len(diag) > 1 else 0
        om = mean(off) if off else float('nan')
        os_ = pstdev(off) if len(off) > 1 else 0
        forget_rate = sum(1 for v in off if v < -3) / len(off) if off else 0
        print(f"{name:<15}  {label:<20}  {dm:+6.2f}±{ds:5.2f}      {om:+6.2f}±{os_:5.2f}        {forget_rate*100:.1f}% (|pp|>3 drops)")
    print("-" * 100)

print()
print("=" * 100)
print("CELL-BY-CELL DIFFERENCES (pp_new - pp_old), per model")
print("=" * 100)
for name, old_label, new_label, perf_old, perf_new, tasks in COMPARISONS:
    pp_old = pp_matrix(perf_old, tasks)
    pp_new = pp_matrix(perf_new, tasks)
    common = sorted(set(pp_old.keys()) & set(pp_new.keys()))
    if not common:
        continue
    diffs = [(src, evl, pp_new[(src, evl)] - pp_old[(src, evl)]) for (src, evl) in common]
    abs_diffs = [abs(d) for _, _, d in diffs]
    diag_diffs = [d for s, e, d in diffs if s == e]
    off_diffs = [d for s, e, d in diffs if s != e]

    # Rank biggest |diff|
    ranked = sorted(diffs, key=lambda x: abs(x[2]), reverse=True)
    print(f"\n[{name}]  {old_label} → {new_label}   ({len(common)} common cells)")
    print(f"  mean |Δpp| = {mean(abs_diffs):.2f}   max |Δpp| = {max(abs_diffs):.2f}")
    print(f"  diag Δ μ = {mean(diag_diffs):+.2f}   off-diag Δ μ = {mean(off_diffs):+.2f}")
    print(f"  top 8 shifts (|Δpp|):")
    for src, evl, d in ranked[:8]:
        old_v = pp_old[(src, evl)]
        new_v = pp_new[(src, evl)]
        cell_type = "DIAG" if src == evl else "OFF "
        print(f"    {cell_type}  FT={src:<8} eval={evl:<8}  "
              f"old={old_v:+7.2f}  new={new_v:+7.2f}  Δ={d:+7.2f}")

print()
print("=" * 100)
print("DIAGONALS: own-task pp (old vs new) side-by-side")
print("=" * 100)
for name, old_label, new_label, perf_old, perf_new, tasks in COMPARISONS:
    pp_old = pp_matrix(perf_old, tasks)
    pp_new = pp_matrix(perf_new, tasks)
    print(f"\n[{name}]")
    print(f"  {'task':<10}  {old_label:<22}  {new_label:<22}  delta")
    for t in tasks:
        k = (t, t)
        o = pp_old.get(k)
        n = pp_new.get(k)
        if o is None and n is None:
            continue
        os_ = f"{o:+6.2f}" if o is not None else "  --  "
        ns_ = f"{n:+6.2f}" if n is not None else "  --  "
        ds_ = f"{(n-o):+6.2f}" if (o is not None and n is not None) else "  --  "
        print(f"  {t:<10}  {os_:<22}  {ns_:<22}  {ds_}")
