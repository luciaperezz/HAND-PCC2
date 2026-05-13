"""
pads_loader.py
--------------
Loads the PADS (Parkinson's Disease Smartwatch) dataset from PhysioNet.

Actual dataset structure on disk:
  data/pads/pads-parkinsons-disease-.../
    patients/
      patient_001.json      ← metadata: condition, UPDRS scores, etc.
      patient_002.json
      ...
    movement/
      timeseries/
        460_Relaxed_LeftWrist.txt
        460_Relaxed_RightWrist.txt
        ...
      observation_001.json  ← lists all tasks + file paths for one subject
      observation_002.json
      ...

Label mapping:
  condition == "Healthy"       → 0
  condition == "Parkinson's"   → 1 (Mild) if UPDRS tremor <= 1.5, else 2 (Severe)
  condition == "DD" or other   → excluded
"""

import os
import json
import glob
import numpy as np
import pandas as pd

FS = 100  # Hz


def _parse_label(patient_meta: dict) -> int | None:
    condition = patient_meta.get("condition", "").strip()
    if condition == "Healthy":
        return 0
    if condition == "Parkinson's":
        updrs = patient_meta.get("updrs_tremor_score", None)
        if updrs is None:
            updrs = patient_meta.get("updrs_score", None)
        if updrs is not None:
            return 1 if float(updrs) <= 1.5 else 2
        else:
            return 1
    return None


def _load_timeseries(ts_path: str) -> np.ndarray | None:
    try:
        df = pd.read_csv(ts_path, sep=r"\s+|,", engine="python", header=None)
    except Exception:
        return None
    if df.shape[1] == 7:
        data = df.iloc[:, 1:].values  # drop timestamp
    elif df.shape[1] == 6:
        data = df.values
    else:
        return None
    return data.astype(np.float32)


def load_pads(data_dir: str, max_subjects: int = None) -> list[dict]:
    # Handle extra nested folder (e.g. pads-parkinsons-disease-smartwatch-...)
    subdirs = [d for d in os.listdir(data_dir)
               if os.path.isdir(os.path.join(data_dir, d))]
    if subdirs and not os.path.isdir(os.path.join(data_dir, "patients")):
        data_dir = os.path.join(data_dir, subdirs[0])

    patients_dir  = os.path.join(data_dir, "patients")
    movement_dir  = os.path.join(data_dir, "movement")
    ts_dir        = os.path.join(movement_dir, "timeseries")

    if not os.path.isdir(patients_dir):
        raise FileNotFoundError(
            f"Could not find {patients_dir}.\n"
            "Make sure you downloaded PADS and pointed data_dir at the right folder."
        )

    # Build a lookup: subject_id (str) → label
    label_map = {}
    for meta_path in glob.glob(os.path.join(patients_dir, "patient_*.json")):
        with open(meta_path) as f:
            meta = json.load(f)
        subject_id = str(meta.get("id", "")).strip().lstrip("0") or str(meta.get("id", ""))
        label = _parse_label(meta)
        if label is not None:
            # store both zero-padded and plain versions
            label_map[subject_id] = label
            label_map[subject_id.zfill(3)] = label

    # Load observation JSONs from movement/
    obs_files = sorted(glob.glob(os.path.join(movement_dir, "observation_*.json")))
    if not obs_files:
        raise FileNotFoundError(f"No observation JSON files found in {movement_dir}.")

    records = []
    n_subjects = 0
    n_excluded = 0

    for obs_path in obs_files:
        with open(obs_path) as f:
            obs = json.load(f)

        subject_id = str(obs.get("subject_id", "")).strip()
        subject_id_plain = subject_id.lstrip("0") or subject_id

        # Look up label
        label = label_map.get(subject_id) or label_map.get(subject_id_plain)
        if label is None:
            n_excluded += 1
            continue

        for session in obs.get("session", []):
            task_name = session.get("record_name", "unknown")
            for rec in session.get("records", []):
                file_ref = rec.get("file_name", "")
                # file_name is relative to movement/, e.g. "timeseries/460_Relaxed_LeftWrist.txt"
                ts_path = os.path.join(movement_dir, file_ref.replace("/", os.sep))
                if not os.path.isfile(ts_path):
                    continue

                imu = _load_timeseries(ts_path)
                if imu is None or len(imu) < FS:
                    continue

                location = rec.get("device_location", "unknown")
                records.append({
                    "subject_id": f"pads_{subject_id}",
                    "condition":  label_map.get(subject_id, "unknown"),
                    "task":       f"{task_name}_{location}",
                    "imu":        imu,
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
    if recs:
        r = recs[0]
        print(f"  First record: subject={r['subject_id']}, task={r['task']}, "
              f"imu shape={r['imu'].shape}, label={r['label']}")