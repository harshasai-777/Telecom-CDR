"""
config.py
=========
Central configuration for the Telecom CDR Analysis project.
Edit this file to change dataset sizes, paths, or model hyperparameters.
"""
import os

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
REPORT_DIR = os.path.join(BASE_DIR, "reports")

for _d in (DATA_DIR, MODEL_DIR, REPORT_DIR):
    os.makedirs(_d, exist_ok=True)

# ── Dataset sizes (number of CDR records) ─────────────────────────────────────
DATASET_SIZES = {
    "small":  10_000,
    "medium": 50_000,
    "large":  200_000,
}

# Default size used when scripts are run without arguments
DEFAULT_SIZE = "small"

# ── Telecom simulation parameters ─────────────────────────────────────────────
NUM_SUBSCRIBERS     = 500        # unique phone numbers in the simulation
NUM_CELL_TOWERS     = 50         # unique cell tower IDs
FRAUD_FRACTION      = 0.05       # fraction of subscribers flagged as suspicious
INTERNATIONAL_RATIO = 0.08       # fraction of calls that are international
ROAMING_RATIO       = 0.06       # fraction of calls while roaming
RANDOM_SEED         = 42

# ── Feature engineering ───────────────────────────────────────────────────────
NIGHT_HOUR_START  = 22   # 10 PM
NIGHT_HOUR_END    = 6    #  6 AM
PEAK_HOUR_START   = 9
PEAK_HOUR_END     = 18

# ── Clustering ────────────────────────────────────────────────────────────────
N_CLUSTERS        = 5    # K-Means number of clusters
KMEANS_INIT       = "k-means++"
KMEANS_N_INIT     = 3    # cloud-safe: was 10
KMEANS_MAX_ITER   = 100  # cloud-safe: was 300

# ── XGBoost (fraud classifier) ────────────────────────────────────────────────
XGB_PARAMS = dict(
    n_estimators     = 80,   # cloud-safe: was 200
    max_depth        = 4,    # cloud-safe: was 6
    learning_rate    = 0.1,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    eval_metric      = "logloss",
    random_state     = RANDOM_SEED,
    n_jobs           = 1,    # cloud-safe: was -1 (no parallel overload)
)
TEST_SIZE   = 0.2
SCALE_POS_WEIGHT = int((1 - FRAUD_FRACTION) / FRAUD_FRACTION)  # handle imbalance

# ── Anomaly detection ─────────────────────────────────────────────────────────
ISOLATION_FOREST_CONTAMINATION = 0.05
ANOMALY_RANDOM_STATE           = RANDOM_SEED

# ── Dask ──────────────────────────────────────────────────────────────────────
DASK_NPARTITIONS = 2   # cloud-safe: was 4 (reduces parallel overhead on free tier)

# ── Streamlit ─────────────────────────────────────────────────────────────────
PAGE_TITLE = "Telecom CDR Analytics Platform"
PAGE_ICON  = "📡"
