"""Circuit-stability overlap: for each fine-tuned model, compare its
**own-task** circuit against its circuit on each **other task's corrupted
data**. Answers the question: "how invariant is the FT circuit across
different input distributions, holding the fine-tuned weights constant?"

  Cell (row=Data_Task D, col=FT_Task F):
      overlap of top-400 edges between
        * F-finetuned model on F's corrupted data   (own-task circuit)
        * F-finetuned model on D's corrupted data   (same model, different data)
      where D != F.  Diagonals are trivially 100% and are left blank.

Layout: 6 rows x 6 cols, diagonals empty, so 30 populated cells per model.

Source edges (top-400), unified:
  output/EAP_edges/cross_task_edges_v2/   (contains all 4 models' files, flat)

Output: output/EAP_edges/cross_task_edges_v2/same_ft_cross_data_overlap/
  <model>_same_ft_cross_data_overlap.csv      (wide 6x6)
  combined_long.csv                           (all 4 models, long format)
  overlap_edges/
    <model>_FT-<ft>_data-<data>_overlap.csv   (per-cell edge list,
                                               30 cells x 4 models = 120 files)
"""
from __future__ import annotations

import csv
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EDGE_DIR = PROJECT_ROOT / "output/EAP_edges/cross_task_edges_v2"
OUT_DIR = EDGE_DIR / "same_ft_cross_data_overlap"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TASKS = ["yelp", "sst2", "squad", "coqa", "kde4", "tatoeba"]

# All four models live flat in EDGE_DIR with a model prefix in each filename.
# (model_key, filename_prefix)
MODELS = [
    ("gpt2",     "gpt2"),
    ("llama3.2", "llama3.2"),
    ("qwen2",    "qwen2"),
    ("llama2",   "llama2"),
]


def load_top_edges_df(csv_path: Path, top_k: int = 400) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if len(df) > top_k:
        df = df.reindex(df["score"].abs().sort_values(ascending=False).index).head(top_k)
    return df.reset_index(drop=True)


def load_top_edges(csv_path: Path, top_k: int = 400) -> set:
    return set(load_top_edges_df(csv_path, top_k)["edge"])


def own_task_path(prefix: str, task: str) -> Path:
    return EDGE_DIR / f"{prefix}_{task}_finetuned_edges.csv"


def cross_task_path(prefix: str, ft_task: str, data_task: str) -> Path:
    return EDGE_DIR / f"{prefix}_Finetuned-{ft_task}_Corrupted-Data_{data_task}_finetuned_edges.csv"


def compute_for_model(model_key, prefix, edges_out_dir: Path):

    matrix = pd.DataFrame(index=TASKS, columns=TASKS, dtype=float)
    matrix.index.name = "Data_Task"
    matrix.columns.name = "FT_Task"

    long_rows = []
    missing = []

    for ft in TASKS:
        own_path = own_task_path(prefix, ft)
        if not own_path.exists():
            missing.append(str(own_path))
            continue
        own_df = load_top_edges_df(own_path)
        own_edges = set(own_df["edge"])

        for data_task in TASKS:
            if data_task == ft:
                continue
            cross_path = cross_task_path(prefix, ft, data_task)
            if not cross_path.exists():
                missing.append(str(cross_path))
                continue
            other_df = load_top_edges_df(cross_path)
            other_edges = set(other_df["edge"])

            common = own_edges & other_edges
            overlap_pct = 100.0 * len(common) / len(own_edges)
            matrix.loc[data_task, ft] = overlap_pct
            long_rows.append({
                "Model": model_key, "FT_Task": ft, "Data_Task": data_task,
                "Overlap_Pct": round(overlap_pct, 2), "N_Shared_Edges": len(common),
            })

            # Save the overlapping edges themselves (with both score columns)
            own_scores = own_df.set_index("edge")["score"]
            cross_scores = other_df.set_index("edge")["score"]
            overlap_df = pd.DataFrame({
                "edge": sorted(common),
                "score_own":   [own_scores[e] for e in sorted(common)],
                "score_cross": [cross_scores[e] for e in sorted(common)],
            })
            # order by descending |score_own| so top-of-own-circuit comes first
            overlap_df["abs_own"] = overlap_df["score_own"].abs()
            overlap_df = overlap_df.sort_values("abs_own", ascending=False).drop(columns=["abs_own"])
            edge_out = edges_out_dir / f"{model_key}_FT-{ft}_data-{data_task}_overlap.csv"
            overlap_df.to_csv(edge_out, index=False)

    if missing:
        print(f"[{model_key}] missing {len(missing)} files; first few:")
        for m in missing[:3]:
            print(f"   - {m}")

    matrix = matrix.round(2)
    return matrix, long_rows


def main():
    edges_out_dir = OUT_DIR / "overlap_edges"
    edges_out_dir.mkdir(parents=True, exist_ok=True)

    combined = []
    for model_key, prefix in MODELS:
        print(f"\n=== {model_key} ===")
        matrix, long_rows = compute_for_model(model_key, prefix, edges_out_dir)
        out_csv = OUT_DIR / f"{model_key}_same_ft_cross_data_overlap.csv"
        matrix.to_csv(out_csv)
        print(f"[wrote] {out_csv.relative_to(PROJECT_ROOT)}")
        print(matrix.to_string())
        combined.extend(long_rows)

    combined_df = pd.DataFrame(combined)
    combined_path = OUT_DIR / "combined_long.csv"
    combined_df.to_csv(combined_path, index=False)
    print(f"\n[wrote] {combined_path.relative_to(PROJECT_ROOT)} ({len(combined_df)} rows)")


if __name__ == "__main__":
    main()
