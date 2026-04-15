"""Aggregate the 18 text-complexity full-FT eval JSONs in results_full/ into:

  1. results_full/summary_long.csv     - one row per (task, train, test, metric)
  2. results_full/summary_wide.csv     - one row per (task, train), columns per test
  3. results_full/summary.tex          - LaTeX table for paper inclusion
  4. stdout                            - human-readable cross-tables per task

Usage:
    python aggregate_eval_results.py [results_dir]

If no path is given, defaults to ./results_full next to this script.

Tolerant to missing files: cells with no JSON show "-" / NaN.
"""
import json
import os
import sys
from collections import defaultdict

import pandas as pd

# Per-task primary metric used for the headline cross-tables
PRIMARY_METRIC = {
    "yelp":    "accuracy",
    "squad":   "avg_f1",
    "tatoeba": "avg_bleu",
}
# All metrics to record per task in the long CSV
ALL_METRICS = {
    "yelp":    ["accuracy"],
    "squad":   ["avg_f1", "avg_em"],
    "tatoeba": ["avg_bleu"],
}
TASKS = ["yelp", "squad", "tatoeba"]
SUBSETS = ["simple", "complex"]
TRAIN_VARIANTS = ["base", "simple", "complex"]  # row order in cross-tables

DISPLAY_TASK = {"yelp": "Yelp", "squad": "SQuAD", "tatoeba": "Tatoeba"}
DISPLAY_TRAIN = {"base": "Base (no FT)", "simple": "FT$_{\\text{simple}}$", "complex": "FT$_{\\text{complex}}$"}


def find_json(results_dir: str, task: str, train: str, test: str) -> str:
    """Locate eval JSON for a given (task, train, test) combination.

    For tatoeba the suffix is `_nltk.json`; for others `.json`.
    train can be "base" (no train subset), "simple", or "complex".
    """
    suffix = "_nltk.json" if task == "tatoeba" else ".json"
    if train == "base":
        name = f"eval_results_{task}_test_{test}_base_full{suffix}"
    else:
        name = f"eval_results_{task}_train_{train}_test_{test}_full{suffix}"
    path = os.path.join(results_dir, name)
    return path if os.path.exists(path) else ""


def load_metrics(json_path: str) -> dict:
    with open(json_path) as f:
        d = json.load(f)
    return d.get("metrics", {})


def build_long_table(results_dir: str) -> pd.DataFrame:
    rows = []
    for task in TASKS:
        for train in TRAIN_VARIANTS:
            for test in SUBSETS:
                path = find_json(results_dir, task, train, test)
                metrics = load_metrics(path) if path else {}
                for m_name in ALL_METRICS[task]:
                    rows.append({
                        "task": task,
                        "train_subset": train,
                        "test_subset": test,
                        "metric": m_name,
                        "value": metrics.get(m_name),
                        "json_path": path or None,
                    })
    return pd.DataFrame(rows)


def build_wide_table(long_df: pd.DataFrame) -> pd.DataFrame:
    """Wide table: rows = (task, train_subset), columns = test_subset, primary metric only."""
    rows = []
    for task in TASKS:
        prim = PRIMARY_METRIC[task]
        for train in TRAIN_VARIANTS:
            row = {"task": task, "train_subset": train, "metric": prim}
            for test in SUBSETS:
                v = long_df[
                    (long_df.task == task)
                    & (long_df.train_subset == train)
                    & (long_df.test_subset == test)
                    & (long_df.metric == prim)
                ]["value"]
                row[f"test_{test}"] = v.iloc[0] if len(v) and pd.notna(v.iloc[0]) else None
            rows.append(row)
    return pd.DataFrame(rows)


def print_cross_tables(long_df: pd.DataFrame) -> None:
    """Per-task cross-tables (3 rows x 2 cols) showing primary metric."""
    for task in TASKS:
        prim = PRIMARY_METRIC[task]
        print()
        print(f"=== {DISPLAY_TASK[task]} ({prim}) ===")
        header = f"{'':<22} | {'test=simple':>11} | {'test=complex':>12}"
        print(header)
        print("-" * len(header))
        for train in TRAIN_VARIANTS:
            row_label = {
                "base": "BASE (no FT)",
                "simple": "FT (train=simple)",
                "complex": "FT (train=complex)",
            }[train]
            cells = []
            for test in SUBSETS:
                v = long_df[
                    (long_df.task == task)
                    & (long_df.train_subset == train)
                    & (long_df.test_subset == test)
                    & (long_df.metric == prim)
                ]["value"]
                v = v.iloc[0] if len(v) else None
                cells.append(f"{v:.4f}" if pd.notna(v) else "  ---  ")
            print(f"{row_label:<22} | {cells[0]:>11} | {cells[1]:>12}")


def build_latex(long_df: pd.DataFrame) -> str:
    """Single LaTeX table covering 3 tasks x 3 train variants x 2 test subsets."""
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Llama-2-7B full fine-tuning on lexical-complexity-filtered training subsets, evaluated on both subsets. Yelp metric: accuracy; SQuAD: token-F1; Tatoeba: BLEU. Bold marks best per column.}")
    lines.append(r"\label{tab:tc_full_ft}")
    lines.append(r"\begin{tabular}{llcc}")
    lines.append(r"\toprule")
    lines.append(r"Task & Model & Test=Simple & Test=Complex \\")
    lines.append(r"\midrule")

    for task in TASKS:
        prim = PRIMARY_METRIC[task]
        sub = long_df[(long_df.task == task) & (long_df.metric == prim)]
        # Find best per test column for bolding
        best_simple = max(
            (v for v in sub[sub.test_subset == "simple"]["value"] if pd.notna(v)),
            default=None,
        )
        best_complex = max(
            (v for v in sub[sub.test_subset == "complex"]["value"] if pd.notna(v)),
            default=None,
        )
        for i, train in enumerate(TRAIN_VARIANTS):
            task_label = DISPLAY_TASK[task] if i == 0 else ""
            row = sub[sub.train_subset == train]
            v_s = row[row.test_subset == "simple"]["value"]
            v_c = row[row.test_subset == "complex"]["value"]
            v_s = v_s.iloc[0] if len(v_s) else None
            v_c = v_c.iloc[0] if len(v_c) else None

            def fmt(v, best):
                if v is None or pd.isna(v):
                    return "---"
                s = f"{v:.4f}"
                return f"\\textbf{{{s}}}" if best is not None and abs(v - best) < 1e-9 else s

            lines.append(
                f"{task_label} & {DISPLAY_TRAIN[train]} & {fmt(v_s, best_simple)} & {fmt(v_c, best_complex)} \\\\"
            )
        if task != TASKS[-1]:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def main():
    results_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "results_full")
    if not os.path.isdir(results_dir):
        print(f"[ERROR] results_dir not found: {results_dir}")
        sys.exit(1)

    print(f"Reading JSONs from: {results_dir}")
    long_df = build_long_table(results_dir)

    # Report missing files
    missing = long_df[long_df["json_path"].isna()]
    if len(missing):
        unique_missing = missing[["task", "train_subset", "test_subset"]].drop_duplicates()
        print(f"[WARN] {len(unique_missing)} missing eval JSON(s):")
        for _, r in unique_missing.iterrows():
            print(f"   - task={r.task} train={r.train_subset} test={r.test_subset}")

    # 1. long CSV
    long_csv = os.path.join(results_dir, "summary_long.csv")
    long_df.to_csv(long_csv, index=False)
    print(f"\nSaved: {long_csv}")

    # 2. wide CSV
    wide_df = build_wide_table(long_df)
    wide_csv = os.path.join(results_dir, "summary_wide.csv")
    wide_df.to_csv(wide_csv, index=False)
    print(f"Saved: {wide_csv}")

    # 3. LaTeX table
    tex = build_latex(long_df)
    tex_path = os.path.join(results_dir, "summary.tex")
    with open(tex_path, "w") as f:
        f.write(tex + "\n")
    print(f"Saved: {tex_path}")

    # 4. cross-tables to stdout
    print_cross_tables(long_df)

    print("\nDone.")


if __name__ == "__main__":
    main()
