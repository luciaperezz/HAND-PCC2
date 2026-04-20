"""
stroke_loader.py
----------------
Loads the Kinematic & EMG dataset for stroke and healthy subjects.

Stroke files (ST##.mat):
  - DataULpleg:    impaired arm EMG   → label 1 (Stroke)
  - DataULnonpleg: healthy arm EMG    → label 0 (Healthy)

Healthy files (HS##.mat):
  - DataULdom:     dominant arm EMG   → label 0 (Healthy)

Returns a list of dicts:
    {
        'subject_id': str,
        'emg':        np.ndarray  shape (n_samples, n_channels),
        'fs':         int,
        'label':      int   (0=Healthy, 1=Stroke)
    }
"""

import os
import glob
from pathlib import Path
import numpy as np
import scipy.io


def _extract_emg_from_arm(data_ul, fs, subject_id, arm_label, label):
    """Extract EMG recordings from one arm field (DataULpleg, DataULnonpleg, or DataULdom)."""
    records = []
    n_tasks = data_ul.shape[1]

    for task_idx in range(n_tasks):
        try:
            task = data_ul[0, task_idx]
            emg = task['EMG']

            # EMG can be nested one more level
            if emg.dtype == object:
                emg = emg[0, 0]

            if emg is None or emg.size == 0:
                continue

            emg = np.asarray(emg)

            # Convert (n_channels, n_samples) → (n_samples, n_channels)
            if emg.ndim == 2 and emg.shape[0] < emg.shape[1]:
                emg = emg.T

            emg = emg.astype(np.float32, copy=False)

            # Skip if less than 0.5 seconds of data
            if emg.ndim != 2 or emg.shape[0] < fs // 2:
                continue

            records.append({
                'subject_id': f"stroke_{subject_id}_{arm_label}",
                'emg':        emg,
                'fs':         fs,
                'label':      label,
                'task':       task_idx,
            })
        except Exception as e:
            print(f"  [WARN] {subject_id} {arm_label} task {task_idx}: {e}")
            continue

    return records


def load_stroke(data_dir: str) -> list[dict]:
    records = []

    data_path = Path(data_dir)
    if not data_path.is_absolute():
        repo_root = Path(__file__).resolve().parents[2]
        data_path = (repo_root / data_path).resolve()

    mat_files = sorted(glob.glob(str(data_path / "*.mat")))

    if not mat_files:
        raise FileNotFoundError(
            f"No .mat files found in {data_path}.\n"
            "Make sure you downloaded the stroke dataset."
        )

    for mat_path in mat_files:
        subject_id = os.path.splitext(os.path.basename(mat_path))[0]
        is_healthy = subject_id.upper().startswith("HS")
        is_stroke  = subject_id.upper().startswith("ST")

        try:
            mat = scipy.io.loadmat(mat_path)
            s = mat['s'][0, 0]
        except Exception as e:
            print(f"  [WARN] Could not read {mat_path}: {e}")
            continue

        # Get sampling frequency
        try:
            fs = int(s['EmgFreq'][0, 0])
        except Exception:
            fs = 2000

        if is_healthy:
            # Healthy subjects — one arm field only
            try:
                data_ul = s['DataULdom']
                records += _extract_emg_from_arm(data_ul, fs, subject_id, "dom", 0)
            except Exception as e:
                print(f"  [WARN] {subject_id} DataULdom: {e}")

        elif is_stroke:
            # Stroke patients — impaired arm (label 1) and healthy arm (label 0)
            for arm_field, arm_label, label in [
                ('DataULpleg',    'pleg',    1),
                ('DataULnonpleg', 'nonpleg', 0),
            ]:
                try:
                    data_ul = s[arm_field]
                    records += _extract_emg_from_arm(data_ul, fs, subject_id, arm_label, label)
                except Exception as e:
                    print(f"  [WARN] {subject_id} {arm_field}: {e}")

        else:
            print(f"  [SKIP] Unknown file format: {subject_id}")

    print(f"[Stroke] Loaded {len(records)} recordings from {len(mat_files)} patients.")
    return records


if __name__ == "__main__":
    recs = load_stroke("data/stroke")
    if recs:
        r = recs[0]
        print(f"  First: subject={r['subject_id']}, emg={r['emg'].shape}, label={r['label']}")
        labels = [r['label'] for r in recs]
        print(f"  Label distribution: {{0: {labels.count(0)}, 1: {labels.count(1)}}}")