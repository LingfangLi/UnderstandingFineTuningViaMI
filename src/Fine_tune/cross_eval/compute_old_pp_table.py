"""
Using the raw values from the original paper draft (old_raw_data_gpt2_llama3.csv),
recompute the Perf-Delta cells in pp (percentage-point) form.

Rule: all values in the CSV are stored as fractions in [0, 1]:
  - Accuracy (Yelp, Twitter): fractional accuracy, e.g. 0.3714 for 37.14%
  - F1, EM (SQuAD, CoQA):     fractional F1 score
  - BLEU (KDE4, Tatoeba):     fractional BLEU score
Therefore pp = (FT - Base) * 100 applies uniformly.

Also cross-check a few values against the original published table to prove
the mix of formulas used there:
  - Sentiment cells  ≡  pp              (e.g. GPT-2 Yelp/Yelp = 34.04)
  - QA cells         ≡  rel-% on F1     (e.g. GPT-2 SQuAD/SQuAD = 535)
  - MT cells         ≡  rel-% on BLEU   (e.g. GPT-2 KDE4/KDE4 = 441)
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "old_raw_data_gpt2_llama3.csv"
OUT_CSV = HERE / "old_perf_delta_pp.csv"

TASKS = ["yelp", "twitter", "squad", "coqa", "kde4", "tatoeba"]
METRIC_KEY = {"yelp": "Accuracy", "twitter": "Accuracy",
              "squad": "F1", "coqa": "F1",
              "kde4": "BLEU", "tatoeba": "BLEU"}

# ---------------------------------------------------------------------------
# Load raw fractions from CSV (skip comment lines starting with '#')
# ---------------------------------------------------------------------------
raw: dict = defaultdict(lambda: defaultdict(dict))
with SRC.open() as f:
    for line in f:
        if line.startswith("#") or line.startswith("Model,"):
            continue
        parts = [p.strip() for p in line.rstrip("\n").split(",")]
        if len(parts) < 7:
            continue
        model, src, evl, acc, f1, em, bleu = parts
        for col, val in [("Accuracy", acc), ("F1", f1), ("EM", em), ("BLEU", bleu)]:
            if val == "":
                continue
            try:
                raw[model][src][(evl, col)] = float(val)
            except ValueError:
                pass


# ---------------------------------------------------------------------------
# Build pp matrix: rows = source FT, cols = eval task
# ---------------------------------------------------------------------------
def value_for(model: str, src: str, evl: str) -> float | None:
    key = (evl, METRIC_KEY[evl])
    return raw[model][src].get(key)

pp_table = defaultdict(dict)     # [model][(src, evl)] = pp value (FT - Base)*100
rel_table = defaultdict(dict)    # for comparison

for model in ("gpt2", "llama3.2"):
    for src in ["Base_Model"] + [t for t in TASKS if t != "Base_Model"]:
        for evl in TASKS:
            base = value_for(model, "Base_Model", evl)
            ft = value_for(model, src, evl)
            if base is None or ft is None:
                continue
            pp = (ft - base) * 100.0
            pp_table[model][(src, evl)] = pp
            if base > 1e-9:
                rel_table[model][(src, evl)] = (ft - base) / base * 100.0

# ---------------------------------------------------------------------------
# Cross-check: reproduce a few cells from the original published table
# ---------------------------------------------------------------------------
print("=== Cross-check against original-paper values ===\n")
checks = [
    ("gpt2",     "yelp",    "yelp",    "pp",    34.04),
    ("gpt2",     "squad",   "squad",   "rel",   535.0),
    ("gpt2",     "coqa",    "coqa",    "rel",   287.0),
    ("gpt2",     "kde4",    "kde4",    "rel",   441.0),
    ("gpt2",     "twitter", "twitter", "pp",    52.81),
    ("llama3.2", "yelp",    "yelp",    "pp",    33.81),
    ("llama3.2", "squad",   "squad",   "rel",   417.54),
    ("llama3.2", "kde4",    "kde4",    "rel",   188.0),
]
for m, src, evl, kind, original in checks:
    computed = (pp_table[m].get((src, evl)) if kind == "pp"
                else rel_table[m].get((src, evl)))
    ok = abs(computed - original) < 2.5 if computed is not None else False
    print(f"  {m:<8} FT={src:<8} eval={evl:<8} kind={kind:<3}  "
          f"original={original:>7.2f}  recomputed={computed:>7.2f}  {'OK' if ok else 'MISMATCH'}")

# ---------------------------------------------------------------------------
# Dump pp matrix to CSV for later consumption by the table builder
# ---------------------------------------------------------------------------
with OUT_CSV.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Model", "Source_FT", "Eval_Task", "PP_Delta_vs_Base"])
    for model in ("gpt2", "llama3.2"):
        for src, evl in sorted(pp_table[model].keys()):
            w.writerow([model, src, evl, f"{pp_table[model][(src, evl)]:.4f}"])

print(f"\n[wrote] {OUT_CSV.relative_to(HERE.parent.parent.parent)}")

# ---------------------------------------------------------------------------
# Print pp matrix in readable 6x6 form (skip Base row, skip Twitter col later
# if desired)
# ---------------------------------------------------------------------------
for model in ("gpt2", "llama3.2"):
    print(f"\n=== {model.upper()} pp-Delta matrix (rows=source FT, cols=eval task) ===")
    hdr = ["src\\eval"] + [t.upper() for t in TASKS]
    print("  " + " | ".join(f"{h:>9}" for h in hdr))
    for src in [t for t in TASKS]:
        row = [src.upper()]
        for evl in TASKS:
            v = pp_table[model].get((src, evl))
            row.append("--" if v is None else f"{v:+7.2f}")
        print("  " + " | ".join(f"{c:>9}" for c in row))
