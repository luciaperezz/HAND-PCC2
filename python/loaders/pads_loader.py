"""
pads_loader.py
--------------
Loads the PADS (Parkinson's Disease Smartwatch) dataset from PhysioNet.

Dataset structure on disk:
  data/pads/
    patients/
      patient_001.json      ← metadata: condition, age, UPDRS scores, etc.
      patient_002.json
      ...
    timeseries/
      patient_001/
        <task_name>.txt     ← columns: timestamp, ax, ay, az, gx, gy, gz  (100 Hz)
      patient_002/
      ...
    samples.csv             ← optional overview file listing all samples

Label mapping:
  condition == "Healthy"       → 0
  condition == "Parkinson's"   → 1 (Mild) if UPDRS tremor <= 1, else 2 (Severe)
  condition == "DD" or other   → excluded (differential diagnosis, ambiguous)

Returns a list of dicts:
    {
        'subject_id': str,
        'condition':  str,
        'task':       str,
        'imu':        np.ndarray  shape (n_samples, 6)  [ax,ay,az,gx,gy,gz],
        'fs':         int         (100),
        'label':      int         (0=Healthy, 1=Mild, 2=Severe)
    }
"""

import os
import json
import glob
import numpy as np
import pandas as pd

FS = 100  # Hz — PADS sampling frequency

# Tasks most relevant to tremor and postural stability (from the 11-task protocol)
# We use all tasks; filter later if needed.
INCLUDE_ALL_TASKS = True


def _parse_label(patient_meta: dict) -> int | None:
    """
    Map PADS patient metadata to our 3-class label.
    Returns None if the subject should be excluded.
    """
    condition = patient_meta.get("condition", "").strip()

    if condition == "Healthy":
        return 0

    if condition == "Parkinson's":
        # Use UPDRS tremor subscore if available
        updrs = patient_meta.get("updrs_tremor_score", None)
        if updrs is None:
            # Fall back to overall severity if tremor-specific score missing
            updrs = patient_meta.get("updrs_score", None)
        if updrs is not None:
            return 1 if float(updrs) <= 1.5 else 2
        else:
            # No UPDRS score — still include as mild (conservative)
            return 1

    # Differential diagnosis or unknown → exclude
    return None


def _load_timeseries(ts_path: str) -> np.ndarray | None:
    """
    Load a single timeseries .txt file.
    Expected columns: timestamp, ax, ay, az, gx, gy, gz
    Returns array of shape (n_samples, 6) or None on failure.
    """
    try:
        df = pd.read_csv(ts_path, sep=r"\s+|,", engine="python", header=None)
    except Exception:
        return None

    # Drop first column if it looks like a timestamp (very large integers)
    if df.shape[1] == 7:
        data = df.iloc[:, 1:].values  # drop timestamp
    elif df.shape[1] == 6:
        data = df.values
    else:
        return None

    return data.astype(np.float32)


def load_pads(data_dir: str, max_subjects: int = None) -> list[dict]:
    """
    Load PADS recordings from `data_dir`.

    Parameters
    ----------
    data_dir     : path to pads root folder (contains patients/ and timeseries/)
    max_subjects : if set, only load this many subjects (useful for quick testing)

    Returns
    -------
    List of recording dicts (see module docstring).
    """
    patients_dir = os.path.join(data_dir, "patients")
    ts_root = os.path.join(data_dir, "timeseries")

    if not os.path.isdir(patients_dir):
        raise FileNotFoundError(
            f"Could not find {patients_dir}.\n"
            "Make sure you downloaded PADS and pointed data_dir at the right folder."
        )

    meta_files = sorted(glob.glob(os.path.join(patients_dir, "patient_*.json")))
    if not meta_files:
        raise FileNotFoundError(f"No patient JSON files found in {patients_dir}.")

    records = []
    n_subjects = 0
    n_excluded = 0

    for meta_path in meta_files:
        with open(meta_path) as f:
            meta = json.load(f)

        label = _parse_label(meta)
        if label is None:
            n_excluded += 1
            continue

        patient_id = str(meta.get("id", os.path.splitext(os.path.basename(meta_path))[0]))
        subject_key = f"pads_{patient_id}"

        # Find timeseries folder for this patient
        ts_folder = os.path.join(ts_root, f"patient_{patient_id}")
        if not os.path.isdir(ts_folder):
            # Try alternate naming
            ts_folder = os.path.join(ts_root, patient_id)
        if not os.path.isdir(ts_folder):
            print(f"  [WARN] No timeseries folder for patient {patient_id}, skipping.")
            continue

        task_files = sorted(glob.glob(os.path.join(ts_folder, "*.txt")))
        if not task_files:
            print(f"  [WARN] No .txt files for patient {patient_id}, skipping.")
            continue

        for ts_path in task_files:
            task_name = os.path.splitext(os.path.basename(ts_path))[0]
            imu = _load_timeseries(ts_path)
            if imu is None or len(imu) < FS:  # skip if < 1 second of data
                continue

            records.append({
                "subject_id": subject_key,
                "condition":  meta.get("condition", "unknown"),
                "task":       task_name,
                "imu":        imu,   # shape (n_samples, 6): ax,ay,az,gx,gy,gz
                "fs":         FS,
                "label":      label,
            })

        n_subjects += 1
        if max_subjects is not None and n_subjects >= max_subjects:
            break

    print(f"[PADS] Loaded {len(records)} recordings from {n_subjects} subjects "
          f"({n_excluded} excluded as differential diagnosis).")
    return records


if __name__ == "__main__":
    recs = load_pads("data/pads", max_subjects=3)
    r = recs[0]
    print(f"  First record: subject={r['subject_id']}, condition={r['condition']}, "
          f"task={r['task']}, imu shape={r['imu'].shape}, label={r['label']}")
