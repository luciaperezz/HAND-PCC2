"""
gait_loader.py
--------------
Loads the Gait in Parkinson's Disease dataset from PhysioNet.

Filename convention:
  GaCo##_##.txt  → Gait Control (Healthy)   → label 0
  GaPt##_##.txt  → Gait Patient (Parkinson's) → label 1

19 columns: timestamp + 18 force sensor channels at 100Hz.

Returns a list of dicts:
    {
        'subject_id': str,
        'gait':       np.ndarray  shape (n_samples, 18),
        'fs':         int         (100),
        'label':      int         (0=Healthy, 1=Parkinson's)
    }
"""

import os
import glob
import re
from pathlib import Path
import numpy as np
import pandas as pd

FS = 100


def load_gait(data_dir: str) -> list[dict]:
    records = []
    # Allow calling from different working directories by resolving relative paths
    # relative to the repo root (one level above this file's parent folder).
    data_path = Path(data_dir)
    if not data_path.is_absolute():
        repo_root = Path(__file__).resolve().parents[2]
        data_path = (repo_root / data_path).resolve()

    txt_files = sorted(glob.glob(str(data_path / "*.txt")))

    if not txt_files:
        raise FileNotFoundError(
            f"No .txt files found in {data_path}.\n"
            "Make sure you downloaded the Gait in Parkinson's dataset."
        )

    seen_subjects = set()

    for txt_path in txt_files:
        basename = os.path.basename(txt_path)

        # Determine label from filename
        if basename.startswith("GaCo"):
            label = 0  # Healthy control
        elif basename.startswith("GaPt"):
            label = 1  # Parkinson's patient
        else:
            continue  # skip unknown files

        # Extract subject ID (e.g. GaCo01 from GaCo01_02.txt)
        match = re.match(r"(Ga(?:Co|Pt)\d+)", basename)
        subject_id = match.group(1) if match else basename
        seen_subjects.add(subject_id)

        try:
            df = pd.read_csv(txt_path, sep=r"\s+", header=None, engine="python")
        except Exception as e:
            print(f"  [WARN] Could not read {txt_path}: {e}")
            continue

        # Drop timestamp column (first column), keep 18 force channels
        if df.shape[1] >= 19:
            gait = df.iloc[:, 1:19].values.astype(np.float32)
        elif df.shape[1] == 18:
            gait = df.values.astype(np.float32)
        else:
            print(f"  [WARN] Unexpected columns in {basename}: {df.shape[1]}")
            continue

        # Skip if less than 1 second of data
        if len(gait) < FS:
            continue

        records.append({
            'subject_id': f"gait_{subject_id}",
            'gait':       gait,
            'fs':         FS,
            'label':      label,
        })

    print(f"[Gait] Loaded {len(records)} recordings from {len(seen_subjects)} subjects.")
    return records


if __name__ == "__main__":
    recs = load_gait("data/gait")
    if recs:
        r = recs[0]
        print(f"  First: subject={r['subject_id']}, gait={r['gait'].shape}, label={r['label']}")