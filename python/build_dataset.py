"""
build_dataset.py
----------------
Orchestrates Phase 1: loads all datasets, runs feature extraction,
and saves a feature matrix CSV for each patient group.

Group 1 (Stroke/EMG):
    - Stroke patients (ST files): impaired arm → label 1
    - Healthy subjects (HS files): dominant arm → label 0
    Source: Kinematic & EMG dataset (Figshare)

Group 2 (Parkinson's/IMU):
    - PADS smartwatch dataset
    Source: PhysioNet

Outputs:
    features/group1_features.csv
    features/group2_features.csv

Usage:
    python build_dataset.py
"""

import os
import sys
import numpy as np
import pandas as pd

# ── Configure paths ────────────────────────────────────────────────────────────
DATA_DIR_STROKE = "data/stroke"
DATA_DIR_PADS   = "data/pads"
OUTPUT_DIR      = "features"
# ──────────────────────────────────────────────────────────────────────────────

sys.path.insert(0, os.path.dirname(__file__))

from loaders.stroke_loader import load_stroke
from loaders.pads_loader   import load_pads
from feature_extraction.emg_features import extract_emg_features_multichannel
from feature_extraction.imu_features import extract_imu_features


def build_group1_dataset() -> pd.DataFrame:
    """
    Group 1 (Stroke/EMG):
    Labels: 0 = Healthy, 1 = Stroke
    """
    print("\n── Group 1: Loading Stroke EMG dataset ───────────────────────────")
    records = load_stroke(DATA_DIR_STROKE)

    rows = []
    for i, rec in enumerate(records):
        if (i + 1) % 50 == 0:
            print(f"  Processing record {i+1}/{len(records)}...")

        feats = extract_emg_features_multichannel(rec["emg"], rec["fs"])
        row = {
            "subject_id": rec["subject_id"],
            "task":       rec["task"],
            "label":      rec["label"],
            **feats,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    feat_cols = ["emg_rms", "median_freq", "fatigue_index",
                 "tremor_band_power", "peak_tremor_freq"]
    df = df.dropna(subset=feat_cols, how="all")
    print(f"  → {len(df)} rows, label distribution: "
          f"{df['label'].value_counts().to_dict()}")
    return df


def build_group2_dataset() -> pd.DataFrame:
    """
    Group 2 (Parkinson's/IMU):
    Labels: 0 = Healthy, 1 = Parkinson's
    """
    print("\n── Group 2: Loading PADS ─────────────────────────────────────────")
    records = load_pads(DATA_DIR_PADS)

    rows = []
    for i, rec in enumerate(records):
        if (i + 1) % 50 == 0:
            print(f"  Processing record {i+1}/{len(records)}...")

        feats = extract_imu_features(rec["imu"], rec["fs"])
        label = 1 if rec["label"] > 0 else 0
        row = {
            "subject_id": rec["subject_id"],
            "condition":  rec["condition"],
            "task":       rec["task"],
            "label":      label,
            **feats,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    feat_cols = ["tremor_power", "tremor_frequency", "orientation_stability",
                 "range_of_motion", "hold_stability", "drift_rate"]
    df = df.dropna(subset=feat_cols, how="all")
    print(f"  → {len(df)} rows, label distribution: "
          f"{df['label'].value_counts().to_dict()}")
    return df


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    errors = []

    # ── Group 1 ───────────────────────────────────────────────────────────────
    try:
        df1 = build_group1_dataset()
        out1 = os.path.join(OUTPUT_DIR, "group1_features.csv")
        df1.to_csv(out1, index=False)
        print(f"\n✓ Saved Group 1 features → {out1}")
    except FileNotFoundError as e:
        print(f"\n[SKIP] Group 1: {e}")
        errors.append("group1")

    # ── Group 2 commented out — already generated, no need to re-run ──────────
    # try:
    #     df2 = build_group2_dataset()
    #     out2 = os.path.join(OUTPUT_DIR, "group2_features.csv")
    #     df2.to_csv(out2, index=False)
    #     print(f"✓ Saved Group 2 features → {out2}")
    # except FileNotFoundError as e:
    #     print(f"\n[SKIP] Group 2: {e}")
    #     errors.append("group2")

    if errors:
        print(f"\n⚠ Skipped: {errors}.")
    else:
        print("\n✓ Phase 1 complete. Ready for Phase 2 (train_svm.py).")


if __name__ == "__main__":
    main()