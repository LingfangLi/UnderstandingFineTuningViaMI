"""Per-layer table joining attention-KL (existing) and EAP-score (summed from
top-K edges, grouped by destination-layer) for each task. One CSV per task.

Current scope: GPT-2 (12 layers). Uses the fresh Pipeline B edges at
output/EAP_edges/gpt2_all_edges/.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
KL_CSV_DEFAULT = PROJECT_ROOT / "experiments/attention_matrix_analysis/attention_analysis_results/gpt2/gpt2_layer_wise_summary.csv"
EAP_DIR_DEFAULT = PROJECT_ROOT / "output/EAP_edges/gpt2_all_edges"
OUT_DIR_DEFAULT = PROJECT_ROOT / "experiments/attention_matrix_analysis/layer_kl_vs_eap/gpt2"

# KL columns (task family prefix) -> canonical task id
KL_COL_TO_TASK = {
    "mt_kde4": "kde4",
    "mt_tatoeba": "tatoeba",
    "qa_coqa": "coqa",
    "qa_squad": "squad",
    "sentiment_sst2": "sst2",
    "sentiment_yelp": "yelp",
}

# "a{L}.h{H}<qkv?>" or "m{L}" or "logits" or "input"
_NODE_RE = re.compile(r"^(?:a(\d+)\.h\d+(?:<[qkv]>)?|m(\d+)|(logits)|(input))$")


def dst_layer(edge_name: str, n_layers: int) -> int | None:
    """Extract destination-layer index from 'src->dst'. Return None if edge
    lands on `input` (shouldn't happen in EAP). `logits` is mapped to the
    virtual "post-final" layer = n_layers (so it's captured separately)."""
    try:
        _src, dst = edge_name.split("->", 1)
    except ValueError:
        return None
    m = _NODE_RE.match(dst.strip())
    if not m:
        return None
    if m.group(1) is not None:
        return int(m.group(1))
    if m.group(2) is not None:
        return int(m.group(2))
    if m.group(3) is not None:   # logits
        return n_layers
    return None                   # input as destination: skip


def build_for_task(
    task: str,
    kl_layer_series: pd.Series,
    edges_csv: Path,
    top_k: int,
    n_layers: int,
) -> pd.DataFrame:
    df = pd.read_csv(edges_csv)
    df["abs_score"] = df["score"].abs()
    top = df.nlargest(top_k, "abs_score").copy()
    top["dst_layer"] = top["edge"].map(lambda e: dst_layer(e, n_layers))

    per_layer = (
        top.groupby("dst_layer")
           .agg(eap_score_sum=("score", "sum"),
                eap_abs_score_sum=("abs_score", "sum"),
                eap_edge_count=("score", "size"))
           .reindex(range(n_layers + 1), fill_value=0.0)
    )
    per_layer["eap_edge_count"] = per_layer["eap_edge_count"].astype(int)
    per_layer.index.name = "layer"

    kl_full = kl_layer_series.reindex(range(n_layers)).astype("float64")
    kl_full.loc[n_layers] = float("nan")
    kl_full.index.name = "layer"
    kl_full.name = "attention_kl"

    out = pd.concat([kl_full, per_layer], axis=1).reset_index()
    out["layer"] = out["layer"].astype(str)
    out.loc[out.index[-1], "layer"] = "logits"
    return out[["layer", "attention_kl",
                "eap_score_sum", "eap_abs_score_sum", "eap_edge_count"]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kl-csv", type=Path, default=KL_CSV_DEFAULT)
    ap.add_argument("--eap-dir", type=Path, default=EAP_DIR_DEFAULT)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR_DEFAULT)
    ap.add_argument("--top-k", type=int, default=400,
                    help="Take top-K edges by |score| before summing per layer")
    ap.add_argument("--n-layers", type=int, default=12,
                    help="Model depth (GPT-2 Small = 12)")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    kl_df = pd.read_csv(args.kl_csv, index_col=0)  # index = layer, cols = tasks

    print(f"KL columns: {list(kl_df.columns)}")
    print(f"EAP dir: {args.eap_dir}")
    print(f"top-K = {args.top_k}, n_layers = {args.n_layers}\n")

    for kl_col, task in KL_COL_TO_TASK.items():
        if kl_col not in kl_df.columns:
            print(f"[skip] {task}: KL column '{kl_col}' not in {args.kl_csv.name}")
            continue
        edges_csv = args.eap_dir / f"gpt2_{task}_finetuned_edges.csv"
        if not edges_csv.exists():
            print(f"[skip] {task}: {edges_csv.name} missing")
            continue

        out_df = build_for_task(
            task,
            kl_df[kl_col],
            edges_csv,
            top_k=args.top_k,
            n_layers=args.n_layers,
        )
        out_path = args.out_dir / f"gpt2_{task}_layer_kl_vs_eap.csv"
        out_df.to_csv(out_path, index=False, float_format="%.6g")
        print(f"[wrote] {out_path.relative_to(PROJECT_ROOT)}")
        print(out_df.to_string(index=False))
        print()


if __name__ == "__main__":
    main()
