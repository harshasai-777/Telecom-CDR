"""
data_generator.py
=================
Generates a realistic synthetic Telecom CDR dataset.

Usage:
    python data_generator.py --size small          # 10 000 records
    python data_generator.py --size medium         # 50 000 records
    python data_generator.py --size large          # 200 000 records
    python data_generator.py --size small --format parquet

Each record represents one phone call with the following fields:

| Column              | Description                                      |
|---------------------|--------------------------------------------------|
| record_id           | Unique CDR identifier                            |
| caller_id           | Anonymised caller MSISDN (phone number)          |
| callee_id           | Anonymised callee MSISDN                         |
| call_start          | Datetime the call started                        |
| duration_sec        | Call duration in seconds                         |
| call_type           | voice / sms / data                               |
| direction           | outgoing / incoming                              |
| is_international    | 1 if the call crossed a national boundary        |
| is_roaming          | 1 if caller was roaming                          |
| cell_tower_id       | Tower that handled the call                      |
| cell_lat / cell_lon | Approximate geographic position of the tower     |
| charge_usd          | Computed call charge                             |
| is_fraud            | Ground-truth label (1 = suspicious subscriber)  |
"""

import argparse
from __future__ import annotations
import os
import random
import sys

import numpy as np
import pandas as pd
from faker import Faker

# Allow running from any location
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    DATA_DIR,
    DATASET_SIZES,
    DEFAULT_SIZE,
    FRAUD_FRACTION,
    INTERNATIONAL_RATIO,
    NUM_CELL_TOWERS,
    NUM_SUBSCRIBERS,
    RANDOM_SEED,
    ROAMING_RATIO,
)

fake = Faker()
rng  = np.random.default_rng(RANDOM_SEED)
random.seed(RANDOM_SEED)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_msisdn(n: int) -> list[str]:
    """Generate n anonymised MSISDN strings like '+91-XXXXXXXXXX'."""
    return [f"+91-{rng.integers(7000000000, 9999999999)}" for _ in range(n)]


def _make_towers(n: int) -> pd.DataFrame:
    """Create a lookup table of cell tower positions (India-ish lat/lon)."""
    return pd.DataFrame({
        "tower_id": [f"TWR{i:04d}" for i in range(n)],
        "lat":      rng.uniform(8.0, 35.0, n).round(4),
        "lon":      rng.uniform(68.0, 97.0, n).round(4),
    })


def _call_duration(is_fraud: bool) -> int:
    """
    Normal subscribers: exponential distribution (mean ≈ 180 s).
    Fraud subscribers:  mix of very short (probing) and very long calls.
    """
    if is_fraud:
        if rng.random() < 0.5:
            return int(rng.integers(1, 10))        # very short probing call
        else:
            return int(rng.integers(1200, 7200))   # unusually long call
    else:
        dur = rng.exponential(180)
        return max(1, min(int(dur), 3600))


def _charge(duration_sec: int, call_type: str, is_international: bool, is_roaming: bool) -> float:
    base_rate = {"voice": 0.002, "sms": 0.01, "data": 0.005}[call_type]
    rate = base_rate
    if is_international:
        rate *= 4.5
    if is_roaming:
        rate *= 2.0
    if call_type == "voice":
        return round(rate * duration_sec, 4)
    elif call_type == "sms":
        return round(rate, 4)
    else:  # data (duration_sec treated as KB)
        return round(rate * (duration_sec / 10), 4)


# ── Main generator ─────────────────────────────────────────────────────────────

def generate_cdr(n_records: int) -> pd.DataFrame:
    print(f"  Generating {n_records:,} CDR records …")

    # Subscriber pool
    subscribers = _make_msisdn(NUM_SUBSCRIBERS)
    fraud_set   = set(random.sample(subscribers, max(1, int(NUM_SUBSCRIBERS * FRAUD_FRACTION))))

    # Tower lookup
    towers = _make_towers(NUM_CELL_TOWERS)

    # Sample caller / callee indices
    caller_idx = rng.integers(0, NUM_SUBSCRIBERS, n_records)
    callee_idx = rng.integers(0, NUM_SUBSCRIBERS, n_records)

    callers = [subscribers[i] for i in caller_idx]
    callees = [subscribers[i] for i in callee_idx]
    is_fraud = np.array([1 if c in fraud_set else 0 for c in callers])

    # Timestamps: random within last 90 days
    base_ts   = pd.Timestamp("2024-01-01")
    offsets   = rng.integers(0, 90 * 24 * 3600, n_records)  # seconds
    call_start = [base_ts + pd.Timedelta(seconds=int(s)) for s in offsets]

    # Duration
    durations = np.array([_call_duration(bool(f)) for f in is_fraud])

    # Call type
    call_types = rng.choice(["voice", "sms", "data"], n_records, p=[0.65, 0.25, 0.10])

    # Flags
    is_intl    = (rng.random(n_records) < INTERNATIONAL_RATIO).astype(int)
    is_roaming = (rng.random(n_records) < ROAMING_RATIO).astype(int)

    # Fraud subscribers call internationally far more often
    intl_boost = rng.random(n_records) < 0.60
    is_intl    = np.where(is_fraud & intl_boost, 1, is_intl)

    # Tower assignment
    tower_idx = rng.integers(0, NUM_CELL_TOWERS, n_records)
    tower_ids = towers["tower_id"].values[tower_idx]
    tower_lat = towers["lat"].values[tower_idx]
    tower_lon = towers["lon"].values[tower_idx]

    # Charges
    charges = [
        _charge(int(d), ct, bool(ii), bool(ir))
        for d, ct, ii, ir in zip(durations, call_types, is_intl, is_roaming)
    ]

    df = pd.DataFrame({
        "record_id":       [f"CDR{i:09d}" for i in range(n_records)],
        "caller_id":       callers,
        "callee_id":       callees,
        "call_start":      call_start,
        "duration_sec":    durations,
        "call_type":       call_types,
        "direction":       rng.choice(["outgoing", "incoming"], n_records, p=[0.55, 0.45]),
        "is_international":is_intl,
        "is_roaming":      is_roaming,
        "cell_tower_id":   tower_ids,
        "cell_lat":        tower_lat.round(4),
        "cell_lon":        tower_lon.round(4),
        "charge_usd":      charges,
        "is_fraud":        is_fraud,
    })

    # Sort by timestamp for realism
    df = df.sort_values("call_start").reset_index(drop=True)
    return df


# ── Save helpers ───────────────────────────────────────────────────────────────

def save_dataset(df: pd.DataFrame, name: str, fmt: str = "csv") -> str:
    fname = os.path.join(DATA_DIR, f"cdr_{name}.{fmt}")
    if fmt == "parquet":
        df.to_parquet(fname, index=False)
    else:
        df.to_csv(fname, index=False)
    size_mb = os.path.getsize(fname) / 1_048_576
    print(f"  Saved -> {fname}  ({size_mb:.1f} MB)")
    return fname


# ── CLI entry point ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic Telecom CDR data.")
    parser.add_argument("--size",   choices=list(DATASET_SIZES.keys()), default=DEFAULT_SIZE)
    parser.add_argument("--format", choices=["csv", "parquet"], default="csv")
    parser.add_argument("--all",    action="store_true", help="Generate all dataset sizes")
    args = parser.parse_args()

    if args.all:
        for size_name, n in DATASET_SIZES.items():
            print(f"\n[{size_name.upper()}]")
            df = generate_cdr(n)
            save_dataset(df, size_name, args.format)
    else:
        n = DATASET_SIZES[args.size]
        print(f"\n[{args.size.upper()}]")
        df = generate_cdr(n)
        save_dataset(df, args.size, args.format)

    print("\nDone.")


if __name__ == "__main__":
    main()
