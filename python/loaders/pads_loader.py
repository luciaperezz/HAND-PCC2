"""
pads_loader.py
--------------
Loads the PADS (Parkinson's Disease Smartwatch) dataset from PhysioNet.

Real dataset structure on disk:
  data/pads/
    preprocessed/
      file_list.csv          ← one row per patient: id, condition, label (0/1/2)
    patients/
      patient_001.json       ← metadata per patient
      patient_002.json
      ...
    movement/
      timeseries/
        001_Relaxed_LeftWrist.txt
        001_Relaxed_RightWrist.txt
        001_RelaxedTask_LeftWrist.txt
        ...                  ← columns: time, ax, ay, az, gx, gy, gz (no header)
      observation_001.json   ← lists all timeseries files for each patient

Label mapping (from file_list.csv):
  label == 0  → Healthy
  label == 1  → Parkinson's
  label == 2  → Other Movement Disorders → EXCLUDED (ambiguous)

Tasks included:
  We use the 4 tasks most relevant to tremor and postural stability,
  matching the kind of exercises performed with the HAND device:
    - Relaxed      (rest tremor)
    - RelaxedTask  (rest tremor during cognitive task)
    - StretchHold  (postural tremor, arms outstretched)
    - LiftHold     (postural tremor, arms lifted)

  Both wrists are included per task, giving up to 8 recordings per patient.

Returns a list of dicts:
    {
        'subject_id': str,
        'condition':  str,
        'task':       str,
        'wrist':      str   ('LeftWrist' or 'RightWrist')
        'imu':        np.ndarray  shape (n_samples, 6)  [ax,ay,az,gx,gy,gz],
        'fs':         int         (100),
        'label':      int         (0=Healthy, 1=Parkinson's)
    }
"""

import os
import glob
import numpy as np
import pandas as pd
from pathlib import Path

FS = 100  # Hz

# Tasks most clinically relevant to hand/wrist tremor assessment
# These match the static hold and postural tasks performed with the HAND device
RELEVANT_TASKS = [
    "Relaxed",       # seated at rest — captures resting tremor
    "RelaxedTask",   # seated at rest during cognitive distraction
    "StretchHold",   # arms outstretched — captures postural tremor
    "LiftHold",      # arms lifted — captures postural tremor
    "HoldWeight",    # holding a weight — closest to grip task
    "PointFinger",   # pointing — captures action tremor
]

WRISTS = ["LeftWrist", "RightWrist"]


def _load_timeseries(txt_path: str) -> np.ndarray | None:
    """
    Load a single PADS timeseries .txt file.
    Format: comma-separated, no header.
    Columns: time, ax, ay, az, gx, gy, gz

    Returns array of shape (n_samples, 6) — drops the time column.
    Returns None on failure.
    """
    try:
        data = np.loadtxt(txt_path, delimiter=",")
    except Exception as e:
        print(f"  [WARN] Could not read {txt_path}: {e}")
        return None

    if data.ndim != 2 or data.shape[1] != 7:
        print(f"  [WARN] Unexpected shape {data.shape} in {txt_path}")
        return None

    # Drop time column (col 0), keep ax ay az gx gy gz (cols 1-6)
    return data[:, 1:].astype(np.float32)


def load_pads(data_dir: str, max_subjects: int = None) -> list[dict]:
    """
    Load PADS recordings from `data_dir`.

    Parameters
    ----------
    data_dir     : path to the pads root folder
                   (the folder that contains preprocessed/, patients/, movement/)
    max_subjects : if set, only load this many subjects (useful for quick testing)

    Returns
    -------
    List of recording dicts (see module docstring).
    """
    data_path = Path(data_dir).resolve()

    # ── Load file_list.csv — the index of all patients and their labels ───────
    file_list_path = data_path / "preprocessed" / "file_list.csv"
    if not file_list_path.exists():
        raise FileNotFoundError(
            f"Could not find {file_list_path}.\n"
            f"Make sure data_dir points to the PADS root folder."
        )

    file_list = pd.read_csv(file_list_path)

    # Keep only Healthy (0) and Parkinson's (1) — exclude Other (2)
    file_list = file_list[file_list["label"].isin([0, 1])].reset_index(drop=True)
    n_excluded = pd.read_csv(file_list_path)["label"].eq(2).sum()

    ts_dir = data_path / "movement" / "timeseries"
    if not ts_dir.exists():
        raise FileNotFoundError(
            f"Could not find timeseries folder at {ts_dir}."
        )

    records = []
    n_subjects = 0
    n_missing  = 0

    for _, row in file_list.iterrows():
        patient_id = str(row["id"]).zfill(3)   # e.g. "1" → "001"
        label      = int(row["label"])
        condition  = str(row["condition"])

        subject_records = []

        for task in RELEVANT_TASKS:
            for wrist in WRISTS:
                fname    = f"{patient_id}_{task}_{wrist}.txt"
                ts_path  = ts_dir / fname

                if not ts_path.exists():
                    n_missing += 1
                    continue

                imu = _load_timeseries(str(ts_path))

                if imu is None or len(imu) < FS:  # need at least 1 second
                    continue

                subject_records.append({
                    "subject_id": f"pads_{patient_id}",
                    "condition":  condition,
                    "task":       task,
                    "wrist":      wrist,
                    "imu":        imu,   # shape (n_samples, 6): ax,ay,az,gx,gy,gz
                    "fs":         FS,
                    "label":      label,
                })

        if subject_records:
            records.extend(subject_records)
            n_subjects += 1

        if max_subjects is not None and n_subjects >= max_subjects:
            break

    label_counts = {
        0: sum(1 for r in records if r["label"] == 0),
        1: sum(1 for r in records if r["label"] == 1),
    }

    print(f"[PADS] Loaded {len(records)} recordings from {n_subjects} subjects.")
    print(f"       Label distribution: {label_counts}")
    print(f"       Excluded (Other Movement Disorders): {n_excluded} subjects")
    if n_missing > 0:
        print(f"       Missing timeseries files (skipped): {n_missing}")

    return records


if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/pads"

    print(f"Loading PADS from: {data_dir}")
    recs = load_pads(data_dir, max_subjects=5)

    if recs:
        r = recs[0]
        print(f"\nFirst record:")
        print(f"  subject_id : {r['subject_id']}")
        print(f"  condition  : {r['condition']}")
        print(f"  task       : {r['task']}")
        print(f"  wrist      : {r['wrist']}")
        print(f"  imu shape  : {r['imu'].shape}")
        print(f"  label      : {r['label']}")
        print(f"  first row  : {r['imu'][0]}")