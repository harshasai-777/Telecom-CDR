"""
benchmark.py
============
Performance benchmarking across dataset sizes (small / medium / large).

Measures
--------
- Data loading time (Dask vs Pandas)
- Preprocessing time
- Feature engineering time
- K-Means clustering time
- XGBoost training time
- Total pipeline time

Results are saved as a CSV report and a bar chart.

Usage:
    python benchmark.py
"""

from __future__ import annotations
import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATASET_SIZES, REPORT_DIR
from data_generator import generate_cdr, save_dataset
from preprocessing import run_pipeline
from clustering import run_clustering


def _time_block(label: str, fn, *args, **kwargs):
    """Run fn(*args, **kwargs), return (result, elapsed_seconds)."""
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - t0
    print(f"      {label}: {elapsed:.2f}s")
    return result, elapsed


def run_benchmark(sizes: list[str] | None = None) -> pd.DataFrame:
    if sizes is None:
        sizes = list(DATASET_SIZES.keys())

    rows = []

    for size in sizes:
        n = DATASET_SIZES[size]
        print(f"\n{'='*50}")
        print(f"  Benchmarking: {size.upper()} ({n:,} records)")
        print(f"{'='*50}")

        # ── Generate data if needed ─────────────────────────────────────────
        from config import DATA_DIR
        path = os.path.join(DATA_DIR, f"cdr_{size}.csv")
        if not os.path.exists(path):
            print("  Generating dataset ...")
            df_raw = generate_cdr(n)
            save_dataset(df_raw, size, "csv")

        row = {"size": size, "n_records": n}

        # ── Preprocessing (Dask + feature engineering + aggregation) ────────
        (raw_df, sub_df), t_pre = _time_block(
            "Preprocessing (Dask->Pandas + feature engineering)",
            run_pipeline, size, False
        )
        row["t_preprocess_s"] = round(t_pre, 3)
        row["n_subscribers"]  = len(sub_df)

        # ── K-Means clustering ───────────────────────────────────────────────
        (sub_labelled, *_), t_km = _time_block(
            "K-Means clustering",
            run_clustering, sub_df, 5, False
        )
        row["t_kmeans_s"] = round(t_km, 3)

        # ── XGBoost training ─────────────────────────────────────────────────
        from xgb_classifier import run_xgb_pipeline
        (_, metrics), t_xgb = _time_block(
            "XGBoost training + evaluation",
            run_xgb_pipeline, sub_df, False
        )
        row["t_xgb_s"]       = round(t_xgb, 3)
        row["xgb_accuracy"]  = metrics["accuracy"]
        row["xgb_roc_auc"]   = metrics["roc_auc"]
        row["xgb_f1"]        = metrics["f1"]

        row["t_total_s"] = round(t_pre + t_km + t_xgb, 3)
        rows.append(row)

    bench_df = pd.DataFrame(rows)
    path = os.path.join(REPORT_DIR, "benchmark_results.csv")
    bench_df.to_csv(path, index=False)
    print(f"\nBenchmark results saved -> {path}")
    print(bench_df.to_string(index=False))

    # ── Plot ─────────────────────────────────────────────────────────────────
    _plot_benchmark(bench_df)

    return bench_df


def _plot_benchmark(bench_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Timing breakdown
    time_cols = ["t_preprocess_s", "t_kmeans_s", "t_xgb_s"]
    labels    = ["Preprocessing", "K-Means", "XGBoost"]
    colors    = ["#2196F3", "#4CAF50", "#FF9800"]

    x = range(len(bench_df))
    bottoms = [0] * len(bench_df)

    for col, label, color in zip(time_cols, labels, colors):
        vals = bench_df[col].values
        axes[0].bar(x, vals, bottom=bottoms, label=label, color=color, alpha=0.85)
        bottoms = [b + v for b, v in zip(bottoms, vals)]

    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels([f"{r['size']}\n({r['n_records']:,})" for _, r in bench_df.iterrows()])
    axes[0].set_ylabel("Time (seconds)")
    axes[0].set_title("Pipeline Execution Time by Dataset Size")
    axes[0].legend()
    axes[0].grid(True, axis="y", alpha=0.3)

    # XGBoost performance
    ax2 = axes[1]
    width = 0.25
    xs = list(range(len(bench_df)))
    ax2.bar([xi - width for xi in xs], bench_df["xgb_accuracy"], width, label="Accuracy", color="#2196F3")
    ax2.bar(xs,                        bench_df["xgb_roc_auc"],  width, label="ROC-AUC",  color="#4CAF50")
    ax2.bar([xi + width for xi in xs], bench_df["xgb_f1"],       width, label="F1 Score", color="#FF9800")
    ax2.set_xticks(xs)
    ax2.set_xticklabels([r["size"] for _, r in bench_df.iterrows()])
    ax2.set_ylabel("Score")
    ax2.set_ylim(0, 1.1)
    ax2.set_title("XGBoost Performance by Dataset Size")
    ax2.legend()
    ax2.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(REPORT_DIR, "benchmark_chart.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"Chart saved -> {path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sizes", nargs="+",
        choices=["small", "medium", "large"],
        default=["small", "medium"],
        help="Dataset sizes to benchmark",
    )
    args = parser.parse_args()
    run_benchmark(args.sizes)
