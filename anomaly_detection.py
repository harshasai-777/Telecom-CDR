"""
anomaly_detection.py
====================
Unsupervised anomaly detection on subscriber behaviour using Isolation Forest.

Purpose
-------
Anomaly detection complements supervised classification when ground-truth
labels are unavailable (real-world CDRs rarely come pre-labelled).

Outputs
-------
- anomaly_score  : continuous score (lower = more anomalous)
- is_anomaly     : binary flag (-1 = anomaly, 1 = normal in sklearn convention)
- Visualisation plots

Usage:
    python anomaly_detection.py --size small
"""
from __future__ import annotations

import os
import sys
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    ANOMALY_RANDOM_STATE,
    DEFAULT_SIZE,
    ISOLATION_FOREST_CONTAMINATION,
    MODEL_DIR,
    REPORT_DIR,
)
from clustering import CLUSTER_FEATURES


# ── Train Isolation Forest ─────────────────────────────────────────────────────

def train_isolation_forest(X: np.ndarray) -> IsolationForest:
    print("  Fitting Isolation Forest ...")
    iso = IsolationForest(
        n_estimators = 200,
        contamination= ISOLATION_FOREST_CONTAMINATION,
        random_state = ANOMALY_RANDOM_STATE,
        n_jobs       = -1,
    )
    iso.fit(X)
    return iso


# ── Score and flag anomalies ───────────────────────────────────────────────────

def score_subscribers(
    iso: IsolationForest,
    sub_df: pd.DataFrame,
    X: np.ndarray,
) -> pd.DataFrame:
    sub_df = sub_df.copy()
    # Raw anomaly score (negative; more negative = more anomalous)
    raw_score = iso.score_samples(X)
    # Normalise to [0, 1] where 1 = most normal
    sub_df["anomaly_score"]  = raw_score
    sub_df["anomaly_flag"]   = iso.predict(X)   # -1 = anomaly, 1 = normal
    sub_df["is_anomaly"]     = (sub_df["anomaly_flag"] == -1).astype(int)
    return sub_df


# ── Evaluation (if ground-truth labels exist) ──────────────────────────────────

def evaluate_anomaly(sub_df: pd.DataFrame) -> dict | None:
    if "is_fraud" not in sub_df.columns:
        print("  No ground-truth labels available — skipping evaluation.")
        return None

    y_true = sub_df["is_fraud"].values
    y_pred = sub_df["is_anomaly"].values

    # Convert Isolation Forest prediction to 0/1
    print("\nAnomaly Detection vs. Ground-Truth Fraud Labels:")
    try:
        classes_present = np.unique(np.concatenate([y_true, y_pred]))
        if len(classes_present) == 2:
            print(classification_report(y_true, y_pred, labels=[0, 1], target_names=["Normal", "Anomaly"], zero_division=0))
        elif len(classes_present) == 1 and classes_present[0] == 0:
            print(classification_report(y_true, y_pred, labels=[0], target_names=["Normal"], zero_division=0))
        elif len(classes_present) == 1 and classes_present[0] == 1:
            print(classification_report(y_true, y_pred, labels=[1], target_names=["Anomaly"], zero_division=0))
        else:
            print(classification_report(y_true, y_pred, zero_division=0))
    except Exception as e:
        print(f"  Anomaly evaluation report note: {e}")

    # AUC using normalised score
    try:
        if len(np.unique(y_true)) > 1:
            auc = roc_auc_score(y_true, -sub_df["anomaly_score"].values)
            print(f"  ROC-AUC (anomaly score vs. fraud): {auc:.4f}")
        else:
            auc = 1.0
    except Exception:
        auc = None

    return {"auc": auc}


# ── Visualisation ──────────────────────────────────────────────────────────────

def plot_anomalies(sub_df: pd.DataFrame) -> None:
    """Scatter plot of PCA-reduced space with anomalies highlighted."""
    if "pca_x" not in sub_df.columns or "pca_y" not in sub_df.columns:
        print("  PCA columns not found — run clustering first.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Anomaly overlay on PCA scatter
    colors = sub_df["is_anomaly"].map({0: "#4CAF50", 1: "#F44336"})
    axes[0].scatter(
        sub_df["pca_x"], sub_df["pca_y"],
        c=colors, alpha=0.6, s=30, edgecolors="none",
    )
    axes[0].set_title("Anomaly Detection — PCA Space")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#4CAF50", label="Normal"),
        Patch(facecolor="#F44336", label="Anomaly"),
    ]
    axes[0].legend(handles=legend_elements)

    # Plot 2: Distribution of anomaly scores
    normal  = sub_df[sub_df["is_anomaly"] == 0]["anomaly_score"]
    anomaly = sub_df[sub_df["is_anomaly"] == 1]["anomaly_score"]
    axes[1].hist(normal,  bins=30, alpha=0.7, color="#4CAF50", label="Normal")
    axes[1].hist(anomaly, bins=30, alpha=0.7, color="#F44336", label="Anomaly")
    axes[1].set_xlabel("Isolation Forest Score")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Score Distribution")
    axes[1].legend()

    plt.tight_layout()
    path = os.path.join(REPORT_DIR, "anomaly_plot.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  Plot saved -> {path}")


# ── Save ───────────────────────────────────────────────────────────────────────

def save_model(iso: IsolationForest) -> None:
    with open(os.path.join(MODEL_DIR, "isolation_forest.pkl"), "wb") as f:
        pickle.dump(iso, f)
    print(f"  Model saved to {MODEL_DIR}")


# ── Full pipeline ──────────────────────────────────────────────────────────────

def run_anomaly_pipeline(sub_df: pd.DataFrame) -> pd.DataFrame:
    t0 = time.perf_counter()
    print("\n=== Anomaly Detection Pipeline ===")

    X = sub_df[CLUSTER_FEATURES].values
    iso = train_isolation_forest(X)
    sub_df = score_subscribers(iso, sub_df, X)
    evaluate_anomaly(sub_df)
    plot_anomalies(sub_df)
    save_model(iso)

    n_anom = sub_df["is_anomaly"].sum()
    print(f"\n  Flagged {n_anom} anomalous subscribers "
          f"({n_anom / len(sub_df):.1%} of {len(sub_df)} total)")
    print(f"Anomaly pipeline complete in {time.perf_counter() - t0:.2f}s")

    return sub_df


# ── Standalone ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    from preprocessing import run_pipeline
    from clustering import run_clustering

    parser = argparse.ArgumentParser()
    parser.add_argument("--size", default=DEFAULT_SIZE, choices=["small", "medium", "large"])
    args = parser.parse_args()

    _, sub_df = run_pipeline(args.size, verbose=True)
    sub_df, *_ = run_clustering(sub_df, run_elbow=False)
    run_anomaly_pipeline(sub_df)
