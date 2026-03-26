"""
build_dataset.py
----------------
Orchestrates Phase 1: loads all datasets, runs feature extraction,
and saves a feature matrix CSV for each patient group.

Outputs:
    features/group1_features.csv   — EMG features, Group 1 (Stroke)
    features/group2_features.csv   — IMU features, Group 2 (Parkinson's)

Usage:
    python build_dataset.py

Make sure you have downloaded the datasets first (see README.md).
Edit the DATA_DIR_* paths below to match where you saved each dataset.
"""

import os
import sys
import numpy as np
import pandas as pd

# ── Configure paths here ───────────────────────────────────────────────────────
DATA_DIR_GRABMYO = "data/grabmyo"   # ← point to your GRABMyo download
DATA_DIR_PADS    = "data/pads"      # ← point to your PADS download
OUTPUT_DIR       = "features"
# ─────────────────────────────────────────────────────────────────────────────

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from loaders.grabmyo_loader import load_grabmyo
from loaders.pads_loader import load_pads
from feature_extraction.emg_features import extract_emg_features_multichannel
from feature_extraction.imu_features import extract_imu_features


def build_group1_dataset() -> pd.DataFrame:
    """
    Group 1 (Stroke/EMG): Load GRABMyo, extract EMG features.
    All GRABMyo subjects are healthy (label=0).
    One row per recording (per subject × session × gesture × trial).
    """
    print("\n── Group 1: Loading GRABMyo ──────────────────────────────────────")
    records = load_grabmyo(DATA_DIR_GRABMYO)

    rows = []
    for i, rec in enumerate(records):
        if (i + 1) % 100 == 0:
            print(f"  Processing record {i+1}/{len(records)}...")

        feats = extract_emg_features_multichannel(rec["emg"], rec["fs"])
        row = {
            "subject_id": rec["subject_id"],
            "session":    rec["session"],
            "gesture":    rec["gesture"],
            "trial":      rec["trial"],
            "label":      rec["label"],
            **feats,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    # Drop rows where all features are NaN (too-short recordings)
    feat_cols = ["emg_rms", "median_freq", "fatigue_index",
                 "tremor_band_power", "peak_tremor_freq"]
    df = df.dropna(subset=feat_cols, how="all")
    print(f"  → {len(df)} rows, {df['subject_id'].nunique()} subjects, "
          f"label distribution:\n{df['label'].value_counts().to_dict()}")
    return df


def build_group2_dataset() -> pd.DataFrame:
    """
    Group 2 (Parkinson's/IMU): Load PADS, extract IMU features.
    Labels: 0=Healthy, 1=Mild, 2=Severe.
    One row per recording (per subject × task).
    """
    print("\n── Group 2: Loading PADS ─────────────────────────────────────────")
    records = load_pads(DATA_DIR_PADS)

    rows = []
    for i, rec in enumerate(records):
        if (i + 1) % 50 == 0:
            print(f"  Processing record {i+1}/{len(records)}...")

        feats = extract_imu_features(rec["imu"], rec["fs"])
        row = {
            "subject_id": rec["subject_id"],
            "condition":  rec["condition"],
            "task":       rec["task"],
            "label":      rec["label"],
            **feats,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    feat_cols = ["tremor_power", "tremor_frequency", "orientation_stability",
                 "range_of_motion", "hold_stability", "drift_rate"]
    df = df.dropna(subset=feat_cols, how="all")
    print(f"  → {len(df)} rows, {df['subject_id'].nunique()} subjects, "
          f"label distribution:\n{df['label'].value_counts().to_dict()}")
    return df


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    errors = []

    # Group 1
    try:
        df1 = build_group1_dataset()
        out1 = os.path.join(OUTPUT_DIR, "group1_features.csv")
        df1.to_csv(out1, index=False)
        print(f"\n✓ Saved Group 1 features → {out1}")
    except FileNotFoundError as e:
        print(f"\n[SKIP] Group 1: {e}")
        errors.append("group1")

    # Group 2
    try:
        df2 = build_group2_dataset()
        out2 = os.path.join(OUTPUT_DIR, "group2_features.csv")
        df2.to_csv(out2, index=False)
        print(f"✓ Saved Group 2 features → {out2}")
    except FileNotFoundError as e:
        print(f"\n[SKIP] Group 2: {e}")
        errors.append("group2")

    if errors:
        print(f"\n⚠ Skipped: {errors}. Download the missing datasets (see README.md).")
    else:
        print("\n✓ Phase 1 complete. Ready for Phase 2 (train_svm.py).")


if __name__ == "__main__":
    main()
