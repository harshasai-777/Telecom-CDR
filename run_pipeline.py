"""
run_pipeline.py
===============
End-to-end pipeline script — runs every stage in sequence.
Supports both internal simulated sizes AND any external custom CDR CSV file.

Usage:
    python run_pipeline.py --size small
    python run_pipeline.py --file my_external_cdr.csv
    python run_pipeline.py --size medium --no-elbow
"""

import argparse
import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA_DIR, DATASET_SIZES, DEFAULT_SIZE, N_CLUSTERS


def _banner(text: str) -> None:
    width = 60
    print(f"\n{'='*width}")
    print(f"  {text}")
    print(f"{'='*width}")


def main():
    parser = argparse.ArgumentParser(description="Full CDR analysis pipeline")
    parser.add_argument("--size",          default=DEFAULT_SIZE,
                        choices=["small", "medium", "large"])
    parser.add_argument("--file",          type=str, default=None,
                        help="Path to any custom external CDR CSV file")
    parser.add_argument("--k",             type=int, default=N_CLUSTERS,
                        help="Number of K-Means clusters")
    parser.add_argument("--no-elbow",      action="store_true",
                        help="Skip elbow analysis (faster)")
    parser.add_argument("--skip-benchmark",action="store_true",
                        help="Skip cross-size benchmarking")
    parser.add_argument("--format",        choices=["csv","parquet"], default="csv")
    args = parser.parse_args()

    t_global = time.perf_counter()

    # ── Step 1: Ingestion — User CSV Required ────────────────────────────────
    custom_csv = args.file
    if not custom_csv:
        uploaded_fallback = os.path.join(DATA_DIR, "uploaded_cdr.csv")
        if os.path.exists(uploaded_fallback):
            print(f"\nNo --file argument provided. Detected active CSV: {uploaded_fallback}")
            try:
                ans = input("Press [Enter] to use this CSV, or type a custom path: ").strip()
                if not ans:
                    custom_csv = uploaded_fallback
                elif os.path.exists(ans):
                    custom_csv = ans
            except (EOFError, KeyboardInterrupt):
                custom_csv = uploaded_fallback

    if not custom_csv:
        try:
            custom_csv = input("Enter path to your CDR / Call Details CSV file: ").strip()
        except (EOFError, KeyboardInterrupt):
            pass

    if not custom_csv or not os.path.exists(custom_csv):
        print("\n[ERROR] No valid CSV file provided.")
        print("Usage: python run_pipeline.py --file <path_to_your_dataset.csv>")
        sys.exit(1)

    _banner(f"Step 1 / 5 — Ingesting User CSV: {custom_csv}")

    # ── Step 2: Preprocessing ────────────────────────────────────────────────
    _banner("Step 2 / 5 — Dask Preprocessing & Schema Adaptation")
    from preprocessing import run_pipeline
    raw_df, sub_df = run_pipeline(size=args.size, custom_path=custom_csv, verbose=True)

    # ── Step 3: Clustering ───────────────────────────────────────────────────
    _banner("Step 3 / 5 — K-Means Clustering & Dynamic Profiling")
    from clustering import run_clustering, profile_clusters
    sub_df, km, scaler, pca, elbow = run_clustering(
        sub_df,
        n_clusters = args.k,
        run_elbow  = not args.no_elbow,
    )

    # ── Step 4: Anomaly Detection ────────────────────────────────────────────
    _banner("Step 4 / 5 — Isolation Forest Anomaly Detection")
    from anomaly_detection import run_anomaly_pipeline
    sub_df = run_anomaly_pipeline(sub_df)

    # ── Step 5: XGBoost ──────────────────────────────────────────────────────
    _banner("Step 5 / 5 — XGBoost Fraud Classifier")
    from xgb_classifier import run_xgb_pipeline
    clf, metrics = run_xgb_pipeline(sub_df, run_cv=True)

    # ── Optional benchmark ───────────────────────────────────────────────────
    if not args.skip_benchmark and not args.file:
        _banner("Bonus — Performance Benchmark (small + medium)")
        from benchmark import run_benchmark
        run_benchmark(["small"] if args.size == "small" else ["small", "medium"])

    # ── Summary ──────────────────────────────────────────────────────────────
    total = time.perf_counter() - t_global
    _banner("Pipeline Complete")
    dataset_desc = args.file if args.file else f"{args.size} ({DATASET_SIZES[args.size]:,} records)"
    print(f"  Dataset source   : {dataset_desc}")
    print(f"  Total records    : {len(raw_df):,}")
    print(f"  Subscribers      : {len(sub_df):,}")
    print(f"  Clusters         : {args.k}")
    print(f"  XGBoost Accuracy : {metrics['accuracy']:.4f}")
    print(f"  XGBoost ROC-AUC  : {metrics['roc_auc']:.4f}")
    print(f"  Anomalies found  : {sub_df['is_anomaly'].sum():,}")
    print(f"  Total wall-clock : {total:.1f}s")
    print(f"\n  Reports saved to : reports/")
    print(f"  Models  saved to : models/")
    print(f"\n  Launch dashboard : streamlit run dashboard.py")


if __name__ == "__main__":
    main()
