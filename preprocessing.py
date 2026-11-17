"""
preprocessing.py
================
Dask-powered data loading, cleaning, and feature engineering pipeline.
Supports both internal simulated CDR datasets and ANY external custom CDR CSV file.

Key outputs
-----------
- Cleaned DataFrame with derived temporal, behavioural, and charge features.
- Per-subscriber aggregated feature table used for clustering and classification.
- Intelligent schema adapter for external CDR datasets (Kaggle, GitHub, Operator dumps).

Run standalone to see a quick summary:
    python preprocessing.py
"""

import os
import sys
import time

import dask.dataframe as dd
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    DATA_DIR,
    DASK_NPARTITIONS,
    DEFAULT_SIZE,
    NIGHT_HOUR_END,
    NIGHT_HOUR_START,
    PEAK_HOUR_END,
    PEAK_HOUR_START,
)


# â”€â”€ Intelligent Schema Normalizer for External CSVs â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def adapt_external_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Intelligently maps common column names from ANY external CDR CSV
    to the standard platform schema.
    """
    col_map = {
        # Caller
        "caller": "caller_id", "calling_number": "caller_id", "source": "caller_id",
        "from": "caller_id", "msisdn": "caller_id", "phone_number": "caller_id",
        "phone": "caller_id", "phonenumber": "caller_id", "user_id": "caller_id",
        "subscriber_id": "caller_id", "customer_id": "caller_id", "sub_id": "caller_id",
        "imsi": "caller_id", "calling_no": "caller_id", "src": "caller_id",
        
        # Callee
        "callee": "callee_id", "called_number": "callee_id", "destination": "callee_id",
        "to": "callee_id", "dst": "callee_id", "called_no": "callee_id", "recipient": "callee_id",
        
        # Datetime
        "timestamp": "call_start", "datetime": "call_start", "date_time": "call_start",
        "start_time": "call_start", "time": "call_start", "date": "call_start", "call_date": "call_start",
        
        # Duration
        "duration": "duration_sec", "call_duration": "duration_sec", "length": "duration_sec",
        "seconds": "duration_sec", "sec": "duration_sec", "duration_seconds": "duration_sec",
        
        # Type
        "type": "call_type", "service": "call_type", "traffic_type": "call_type", "event_type": "call_type",
        
        # Direction
        "call_direction": "direction", "dir": "direction", "flow": "direction",
        
        # Flags
        "international": "is_international", "intl": "is_international", "is_intl": "is_international",
        "roaming": "is_roaming", "is_roam": "is_roaming",
        
        # Tower
        "tower_id": "cell_tower_id", "tower": "cell_tower_id", "cell_id": "cell_tower_id",
        "bts_id": "cell_tower_id", "location_id": "cell_tower_id",
        
        # Charge
        "charge": "charge_usd", "cost": "charge_usd", "amount": "charge_usd",
        "price": "charge_usd", "bill": "charge_usd", "tariff": "charge_usd",
        
        # Label / Churn
        "fraud": "is_fraud", "label": "is_fraud", "is_suspicious": "is_fraud",
        "target": "is_fraud", "class": "is_fraud", "churn": "is_fraud", "churned": "is_fraud"
    }
    
    rename_dict = {}
    for c in df.columns:
        norm = str(c).strip().lower().replace(" ", "_").replace("-", "_").replace(".", "_")
        if norm in col_map:
            rename_dict[c] = col_map[norm]
        elif norm in [v for v in col_map.values()]:
            rename_dict[c] = norm
            
    df = df.rename(columns=rename_dict)
    
    # Fill in mandatory baseline defaults if omitted in external file
    n = len(df)
    if "record_id" not in df.columns:
        df["record_id"] = [f"CDR{i:09d}" for i in range(n)]
    if "caller_id" not in df.columns:
        # Dynamic subscriber allocation if no caller identifier is present
        n_subs = min(n, 5000)
        df["caller_id"] = [f"+91-98000{i%n_subs:05d}" for i in range(n)]
    if "callee_id" not in df.columns:
        n_unique_callers = max(1, df["caller_id"].nunique())
        df["callee_id"] = [f"+91-99000{(i*7)%n_unique_callers:05d}" for i in range(n)]
    if "call_start" not in df.columns:
        df["call_start"] = pd.date_range("2024-01-01", periods=n, freq="min")
    else:
        df["call_start"] = pd.to_datetime(df["call_start"], errors="coerce").fillna(pd.Timestamp("2024-01-01"))

    # Compute duration if minutes columns exist
    mins_cols = [c for c in df.columns if any(k in str(c).lower() for k in ["day mins", "day_mins", "eve mins", "eve_mins", "night mins", "night_mins", "intl mins", "intl_mins"])]
    if mins_cols and "duration_sec" not in df.columns:
        df["duration_sec"] = (df[mins_cols].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1) * 60).astype(int)
    elif "duration_sec" not in df.columns:
        df["duration_sec"] = 120
    else:
        df["duration_sec"] = pd.to_numeric(df["duration_sec"], errors="coerce").fillna(60).astype(int)

    # Compute charge if charge columns exist
    charge_cols = [c for c in df.columns if any(k in str(c).lower() for k in ["day charge", "day_charge", "eve charge", "eve_charge", "night charge", "night_charge", "intl charge", "intl_charge"])]
    if charge_cols and "charge_usd" not in df.columns:
        df["charge_usd"] = df[charge_cols].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1).round(2)
    elif "charge_usd" not in df.columns:
        df["charge_usd"] = (df["duration_sec"] * 0.002).round(2)
    else:
        df["charge_usd"] = pd.to_numeric(df["charge_usd"], errors="coerce").fillna(0.05).round(2)

    # Compute international flag
    intl_cols = [c for c in df.columns if "intl" in str(c).lower()]
    if intl_cols and "is_international" not in df.columns:
        df["is_international"] = (df[intl_cols].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1) > 0).astype(int)
    elif "is_international" not in df.columns:
        df["is_international"] = 0
    else:
        df["is_international"] = pd.to_numeric(df["is_international"], errors="coerce").fillna(0).clip(0, 1).astype(int)

    if "call_type" not in df.columns:
        df["call_type"] = "voice"
    else:
        df["call_type"] = df["call_type"].astype(str).str.lower().map(lambda x: x if x in ["voice", "sms", "data"] else "voice")
    if "direction" not in df.columns:
        df["direction"] = "outgoing"
    else:
        df["direction"] = df["direction"].astype(str).str.lower().map(lambda x: x if x in ["outgoing", "incoming"] else "outgoing")
    if "is_roaming" not in df.columns:
        df["is_roaming"] = 0
    else:
        df["is_roaming"] = pd.to_numeric(df["is_roaming"], errors="coerce").fillna(0).clip(0, 1).astype(int)
    if "cell_tower_id" not in df.columns:
        df["cell_tower_id"] = "TWR0001"
    if "cell_lat" not in df.columns:
        df["cell_lat"] = 28.6139
    if "cell_lon" not in df.columns:
        df["cell_lon"] = 77.2090
    if "is_fraud" not in df.columns:
        df["is_fraud"] = 0
    else:
        # Handle booleans, strings, or numbers
        df["is_fraud"] = df["is_fraud"].map({True: 1, False: 0, "True": 1, "False": 0, "true": 1, "false": 0, "1": 1, "0": 0, 1: 1, 0: 0}).fillna(0).astype(int)

    return df


# â”€â”€ Loading â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def load_raw_dask(size: str = DEFAULT_SIZE, custom_path: str | None = None) -> dd.DataFrame:
    """
    Load a CDR CSV into a Dask DataFrame for parallel processing.
    Can load standard dataset or ANY external custom CSV path.
    """
    if custom_path:
        path = custom_path
    else:
        path = os.path.join(DATA_DIR, f"cdr_{size}.csv")

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            f"Run:  python data_generator.py --size {size}"
        )

    # First inspect and normalize schema if it's an external custom file
    pdf = pd.read_csv(path)
    pdf = adapt_external_dataframe(pdf)
    
    if not pd.api.types.is_datetime64_any_dtype(pdf["call_start"]):
        pdf["call_start"] = pd.to_datetime(pdf["call_start"], errors="coerce").fillna(pd.Timestamp("2024-01-01"))

    # Convert in-memory directly to partitioned Dask DataFrame (fast, zero disk I/O overhead)
    ddf = dd.from_pandas(pdf, npartitions=DASK_NPARTITIONS)
    return ddf


# â”€â”€ Cleaning â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def clean_dask(ddf: dd.DataFrame) -> dd.DataFrame:
    """
    Apply basic data-quality rules:
    - Drop records with null critical fields.
    - Remove zero-duration voice calls.
    - Ensure numeric flags are 0/1.
    """
    ddf = ddf.dropna(subset=["caller_id", "callee_id", "call_start"])
    ddf = ddf[~((ddf["call_type"] == "voice") & (ddf["duration_sec"] == 0))]
    ddf["duration_sec"] = ddf["duration_sec"].clip(lower=1)
    ddf["is_international"] = ddf["is_international"].clip(lower=0, upper=1)
    ddf["is_roaming"]       = ddf["is_roaming"].clip(lower=0, upper=1)
    return ddf


# â”€â”€ Feature Engineering â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _add_temporal_features(ddf: dd.DataFrame) -> dd.DataFrame:
    """Add hour, day-of-week, night-time and peak-hour columns."""
    ddf["hour"]        = ddf["call_start"].dt.hour.astype("int8")
    ddf["day_of_week"] = ddf["call_start"].dt.dayofweek.astype("int8")   # 0=Mon
    ddf["is_night"]    = (
        (ddf["hour"] >= NIGHT_HOUR_START) | (ddf["hour"] < NIGHT_HOUR_END)
    ).astype("int8")
    ddf["is_peak"]     = (
        (ddf["hour"] >= PEAK_HOUR_START) & (ddf["hour"] < PEAK_HOUR_END)
    ).astype("int8")
    ddf["month"]       = ddf["call_start"].dt.month.astype("int8")
    ddf["week"]        = ddf["call_start"].dt.isocalendar().week.astype("int8")
    return ddf


def engineer_features_dask(ddf: dd.DataFrame) -> dd.DataFrame:
    """Full feature-engineering pipeline (runs on Dask partitions)."""
    ddf = _add_temporal_features(ddf)
    type_map      = {"voice": 0, "sms": 1, "data": 2}
    direction_map = {"outgoing": 0, "incoming": 1}
    ddf["call_type_enc"] = ddf["call_type"].map(type_map).fillna(0).astype("int8")
    ddf["direction_enc"] = ddf["direction"].map(direction_map).fillna(0).astype("int8")
    ddf["duration_min"]  = (ddf["duration_sec"] / 60).round(2).astype("float32")
    return ddf


# â”€â”€ Per-subscriber Aggregation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def aggregate_per_subscriber(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a rich feature vector for each subscriber that summarises their
    calling behaviour over the entire observation window.
    """
    g = df.groupby("caller_id")
    agg = pd.DataFrame()

    agg["total_calls"]           = g["record_id"].count()
    agg["total_duration_min"]    = g["duration_min"].sum()
    agg["avg_duration_min"]      = g["duration_min"].mean().round(2)
    agg["max_duration_min"]      = g["duration_min"].max()
    agg["std_duration_min"]      = g["duration_min"].std().fillna(0).round(2)

    agg["pct_voice"] = (
        g.apply(lambda x: (x["call_type"] == "voice").sum() / max(len(x), 1), include_groups=False)
    ).round(3)
    agg["pct_sms"]   = (
        g.apply(lambda x: (x["call_type"] == "sms").sum()   / max(len(x), 1), include_groups=False)
    ).round(3)
    agg["pct_data"]  = (
        g.apply(lambda x: (x["call_type"] == "data").sum()  / max(len(x), 1), include_groups=False)
    ).round(3)

    agg["pct_night_calls"] = (
        g.apply(lambda x: x["is_night"].sum() / max(len(x), 1), include_groups=False)
    ).round(3)
    agg["pct_peak_calls"]  = (
        g.apply(lambda x: x["is_peak"].sum()  / max(len(x), 1), include_groups=False)
    ).round(3)
    agg["unique_hours_active"] = g["hour"].nunique()

    agg["pct_international"] = g["is_international"].mean().round(3)
    agg["pct_roaming"]       = g["is_roaming"].mean().round(3)

    agg["unique_towers"]   = g["cell_tower_id"].nunique()
    agg["unique_callees"]  = g["callee_id"].nunique()

    agg["total_charge_usd"] = g["charge_usd"].sum().round(2)
    agg["avg_charge_usd"]   = g["charge_usd"].mean().round(4)
    agg["max_charge_usd"]   = g["charge_usd"].max()

    agg["pct_outgoing"] = (
        g.apply(lambda x: (x["direction"] == "outgoing").sum() / max(len(x), 1), include_groups=False)
    ).round(3)
    agg["unique_days_active"] = g["day_of_week"].nunique()

    agg["is_fraud"] = g["is_fraud"].max()

    agg = agg.reset_index()
    agg = agg.fillna(0)
    return agg


# â”€â”€ Full pipeline â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def run_pipeline(
    size: str = DEFAULT_SIZE,
    custom_path: str | None = None,
    verbose: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Execute the complete preprocessing pipeline.
    """
    t0 = time.perf_counter()

    target_name = os.path.basename(custom_path) if custom_path else f"'{size}'"
    if verbose:
        print(f"[1/4] Loading {target_name} dataset via Dask ...")
    ddf = load_raw_dask(size=size, custom_path=custom_path)

    if verbose:
        print("[2/4] Cleaning & standardizing ...")
    ddf = clean_dask(ddf)

    if verbose:
        print("[3/4] Engineering temporal & behavioural features ...")
    ddf = engineer_features_dask(ddf)

    if verbose:
        print("[4/4] Computing Dask graph -> Pandas ...")
    raw_df = ddf.compute()

    if verbose:
        print("      Aggregating per-subscriber multidimensional metrics ...")
    sub_df = aggregate_per_subscriber(raw_df)

    elapsed = time.perf_counter() - t0
    if verbose:
        print(f"\nPipeline complete in {elapsed:.2f}s")
        print(f"  Records  : {len(raw_df):,}")
        print(f"  Subscribers: {len(sub_df):,}")
        print(f"  Fraud rate : {sub_df['is_fraud'].mean():.1%}")

    return raw_df, sub_df


# â”€â”€ Standalone execution â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--size", default=DEFAULT_SIZE, choices=["small", "medium", "large"])
    parser.add_argument("--file", type=str, default=None, help="Custom external CDR CSV file")
    args = parser.parse_args()

    raw_df, sub_df = run_pipeline(args.size, custom_path=args.file, verbose=True)
    print("\nSample subscriber features:")
    print(sub_df.head(3).to_string())
