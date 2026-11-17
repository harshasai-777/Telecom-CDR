"""
clustering.py
=============
K-Means clustering of subscriber behaviour profiles.

Steps
-----
1. Scale features with StandardScaler.
2. Reduce dimensions with PCA (for visualisation only â€“ not for clustering).
3. Determine optimal k using the Elbow method (WCSS) and Silhouette scores.
4. Fit K-Means with the configured k.
5. Attach cluster labels and dynamic behavioural names to subscribers.
6. Save the fitted model artefacts.

Usage:
    python clustering.py --size small
"""

import os
import sys
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    DEFAULT_SIZE,
    KMEANS_INIT,
    KMEANS_MAX_ITER,
    KMEANS_N_INIT,
    MODEL_DIR,
    N_CLUSTERS,
    RANDOM_SEED,
    REPORT_DIR,
)


# â”€â”€ Feature columns used for clustering â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
CLUSTER_FEATURES = [
    "total_calls",
    "total_duration_min",
    "avg_duration_min",
    "max_duration_min",
    "std_duration_min",
    "pct_voice",
    "pct_sms",
    "pct_data",
    "pct_night_calls",
    "pct_peak_calls",
    "unique_hours_active",
    "pct_international",
    "pct_roaming",
    "unique_towers",
    "unique_callees",
    "total_charge_usd",
    "avg_charge_usd",
    "pct_outgoing",
    "unique_days_active",
]


# â”€â”€ Helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _prepare_features(sub_df: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    X = sub_df[CLUSTER_FEATURES].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


def _reduce_pca(X_scaled: np.ndarray, n_components: int = 2) -> np.ndarray:
    pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
    return pca.fit_transform(X_scaled), pca


# â”€â”€ Elbow / Silhouette analysis â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def elbow_analysis(
    X_scaled: np.ndarray,
    k_range: range = range(2, 11),
    save_plot: bool = True,
) -> dict:
    """Compute WCSS and silhouette score for each k."""
    print("  Running elbow analysis ...")
    wcss   = []
    sil    = []

    for k in k_range:
        km = KMeans(
            n_clusters    = k,
            init          = KMEANS_INIT,
            n_init        = KMEANS_N_INIT,
            max_iter      = KMEANS_MAX_ITER,
            random_state  = RANDOM_SEED,
        )
        labels = km.fit_predict(X_scaled)
        wcss.append(km.inertia_)
        sil.append(silhouette_score(X_scaled, labels, sample_size=min(2000, len(X_scaled))))
        print(f"    k={k:2d}  WCSS={km.inertia_:,.0f}  Silhouette={sil[-1]:.4f}")

    results = {"k": list(k_range), "wcss": wcss, "silhouette": sil}

    if save_plot:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(k_range, wcss, "bo-", linewidth=2)
        axes[0].set_xlabel("Number of Clusters (k)")
        axes[0].set_ylabel("WCSS (Inertia)")
        axes[0].set_title("Elbow Method")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(k_range, sil, "rs-", linewidth=2)
        axes[1].set_xlabel("Number of Clusters (k)")
        axes[1].set_ylabel("Silhouette Score")
        axes[1].set_title("Silhouette Analysis")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        path = os.path.join(REPORT_DIR, "elbow_silhouette.png")
        plt.savefig(path, dpi=120)
        plt.close()
        print(f"  Plot saved -> {path}")

    return results


# â”€â”€ Cluster fitting â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def fit_kmeans(X_scaled: np.ndarray, n_clusters: int = N_CLUSTERS) -> KMeans:
    print(f"  Fitting K-Means (k={n_clusters}) ...")
    km = KMeans(
        n_clusters   = n_clusters,
        init         = KMEANS_INIT,
        n_init       = KMEANS_N_INIT,
        max_iter     = KMEANS_MAX_ITER,
        random_state = RANDOM_SEED,
    )
    km.fit(X_scaled)
    print(f"  Inertia: {km.inertia_:,.2f}")
    return km


def label_subscribers(
    sub_df: pd.DataFrame,
    km: KMeans,
    X_scaled: np.ndarray,
) -> tuple[pd.DataFrame, PCA]:
    """Attach cluster labels and 2-D PCA coordinates to the subscriber table."""
    sub_df = sub_df.copy()
    sub_df["cluster"] = km.labels_

    X_2d, pca = _reduce_pca(X_scaled, 2)
    sub_df["pca_x"] = X_2d[:, 0].round(4)
    sub_df["pca_y"] = X_2d[:, 1].round(4)

    return sub_df, pca


# â”€â”€ Dynamic Cluster profiling â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def assign_cluster_names(profile: pd.DataFrame) -> dict[int, str]:
    """
    Dynamically assign human-readable labels to clusters based on their statistical traits:
    - Highest fraud rate -> Suspicious / High Fraud Risk
    - Highest international ratio -> International Travellers
    - Highest night activity -> Night-Shift / Off-Peak Users
    - Highest total calls / duration -> High-Volume Callers
    - Lowest volume -> Light / Casual Users
    """
    labels = {}
    assigned = set()

    # 1. High-risk fraud cluster
    if "fraud_rate_pct" in profile.columns:
        fraud_c = profile["fraud_rate_pct"].idxmax()
        if profile.loc[fraud_c, "fraud_rate_pct"] > 10.0:
            labels[fraud_c] = "Suspicious / High-Risk"
            assigned.add(fraud_c)

    # 2. International Travellers
    intl_candidates = [c for c in profile.index if c not in assigned]
    if intl_candidates:
        intl_c = profile.loc[intl_candidates, "pct_international"].idxmax()
        labels[intl_c] = "International Travellers"
        assigned.add(intl_c)

    # 3. Night-Shift Users
    night_candidates = [c for c in profile.index if c not in assigned]
    if night_candidates:
        night_c = profile.loc[night_candidates, "pct_night_calls"].idxmax()
        labels[night_c] = "Night-Shift / Off-Peak"
        assigned.add(night_c)

    # 4. Heavy Callers
    remaining = [c for c in profile.index if c not in assigned]
    if remaining:
        high_c = profile.loc[remaining, "total_calls"].idxmax()
        labels[high_c] = "Heavy Callers"
        assigned.add(high_c)

    # 5. Remaining
    for c in profile.index:
        if c not in labels:
            labels[c] = "Standard / Casual Users"

    return labels


def profile_clusters(sub_df: pd.DataFrame) -> pd.DataFrame:
    """Return a summary table of key metrics per cluster with dynamic names."""
    profile_cols = [
        "total_calls", "avg_duration_min", "pct_night_calls",
        "pct_international", "pct_roaming", "total_charge_usd",
        "unique_callees", "is_fraud",
    ]
    profile = sub_df.groupby("cluster")[profile_cols].mean().round(3)
    profile["n_subscribers"] = sub_df.groupby("cluster").size()
    profile["fraud_rate_pct"] = (profile["is_fraud"] * 100).round(1)
    profile.drop(columns=["is_fraud"], inplace=True)

    label_dict = assign_cluster_names(profile)
    profile["segment_name"] = profile.index.map(label_dict)
    return profile


# â”€â”€ Saving artefacts â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def save_artefacts(km: KMeans, scaler: StandardScaler, pca: PCA, cluster_names: dict | None = None) -> None:
    with open(os.path.join(MODEL_DIR, "kmeans.pkl"),  "wb") as f: pickle.dump(km,     f)
    with open(os.path.join(MODEL_DIR, "scaler.pkl"),  "wb") as f: pickle.dump(scaler, f)
    with open(os.path.join(MODEL_DIR, "pca.pkl"),     "wb") as f: pickle.dump(pca,    f)
    if cluster_names:
        with open(os.path.join(MODEL_DIR, "cluster_names.pkl"), "wb") as f:
            pickle.dump(cluster_names, f)
    print(f"  Artefacts saved to {MODEL_DIR}")


# â”€â”€ Full clustering pipeline â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def run_clustering(
    sub_df: pd.DataFrame,
    n_clusters: int = N_CLUSTERS,
    run_elbow: bool = True,
) -> tuple[pd.DataFrame, KMeans, StandardScaler, PCA, dict | None]:
    t0 = time.perf_counter()
    print("\n=== Clustering Pipeline ===")

    X_scaled, scaler = _prepare_features(sub_df)

    elbow_results = None
    if run_elbow:
        elbow_results = elbow_analysis(X_scaled)

    km      = fit_kmeans(X_scaled, n_clusters)
    sub_df, pca = label_subscribers(sub_df, km, X_scaled)

    # Dynamic names
    profile = profile_clusters(sub_df)
    names = assign_cluster_names(profile)
    sub_df["segment_name"] = sub_df["cluster"].map(names)

    save_artefacts(km, scaler, pca, names)

    elapsed = time.perf_counter() - t0
    print(f"\nClustering complete in {elapsed:.2f}s")
    print(profile.to_string())

    return sub_df, km, scaler, pca, elbow_results


# â”€â”€ Standalone execution â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if __name__ == "__main__":
    import argparse
    from preprocessing import run_pipeline

    parser = argparse.ArgumentParser()
    parser.add_argument("--size",       default=DEFAULT_SIZE, choices=["small", "medium", "large"])
    parser.add_argument("--k",          type=int, default=N_CLUSTERS)
    parser.add_argument("--no-elbow",   action="store_true")
    args = parser.parse_args()

    _, sub_df = run_pipeline(args.size, verbose=True)
    run_clustering(sub_df, n_clusters=args.k, run_elbow=not args.no_elbow)
