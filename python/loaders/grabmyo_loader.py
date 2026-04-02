"""
grabmyo_loader.py
-----------------
Loads the GRABMyo dataset from PhysioNet WFDB format.

Dataset structure on disk (after wget download):
  data/grabmyo/
    Session1/
      session1_participant1/
        session1_participant1_gesture1_trial1.dat
        session1_participant1_gesture1_trial1.hea
        ...
    Session2/
    Session3/

All 43 participants are healthy controls → label 0.
Each recording has up to 16 forearm channels + 12 wrist channels at 2048 Hz.
We use only the forearm channels (indices 0–15) to match our EMG sensor placement.

Returns a list of dicts:
    {
        'subject_id': str,
        'session':    int,
        'gesture':    int,
        'trial':      int,
        'emg':        np.ndarray  shape (n_samples, n_channels),
        'fs':         int         (2048),
        'label':      int         (0 = Healthy)
    }
"""

import os
import glob
import re
import numpy as np

try:
    import wfdb
except ImportError:
    raise ImportError("Install wfdb: pip install wfdb")


# GRABMyo forearm electrode count (16 channels on the forearm ring)
FOREARM_CHANNELS = 16
FS = 2048  # Hz


def load_grabmyo(data_dir: str, max_subjects: int = None) -> list[dict]:
    """
    Load GRABMyo recordings from `data_dir`.

    Parameters
    ----------
    data_dir    : path to the grabmyo root folder (contains Session1/, Session2/, Session3/)
    max_subjects: if set, only load this many subjects (useful for quick testing)

    Returns
    -------
    List of recording dicts (see module docstring).
    """
    records = []
    pattern = os.path.join(data_dir, "Session*", "session*_participant*",
                           "session*_participant*_gesture*_trial*.hea")
    files = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(
            f"No GRABMyo .hea files found under {data_dir}.\n"
            "Make sure you downloaded the dataset and pointed data_dir at the right folder."
        )

    seen_subjects = set()

    for hea_path in files:
        # Parse filename: session{s}_participant{p}_gesture{g}_trial{t}.hea
        basename = os.path.splitext(os.path.basename(hea_path))[0]
        match = re.match(
            r"session(\d+)_participant(\d+)_gesture(\d+)_trial(\d+)", basename
        )
        if not match:
            continue

        session, subject, gesture, trial = [int(x) for x in match.groups()]
        seen_subjects.add(subject)

        if max_subjects is not None and len(seen_subjects) > max_subjects:
            break

        record_path = os.path.splitext(hea_path)[0]  # wfdb wants path without extension
        try:
            record = wfdb.rdrecord(record_path)
        except Exception as e:
            print(f"  [WARN] Could not read {record_path}: {e}")
            continue

        signal = record.p_signal  # shape: (n_samples, n_channels)

        # Keep only forearm channels (first FOREARM_CHANNELS columns)
        n_ch = min(FOREARM_CHANNELS, signal.shape[1])
        emg = signal[:, :n_ch].astype(np.float32)

        records.append({
            "subject_id": f"grabmyo_p{subject:02d}",
            "session":    session,
            "gesture":    gesture,
            "trial":      trial,
            "emg":        emg,
            "fs":         FS,
            "label":      0,  # all GRABMyo subjects are healthy
        })

    print(f"[GRABMyo] Loaded {len(records)} recordings from {len(seen_subjects)} subjects.")
    return records


if __name__ == "__main__":
    # Quick sanity check — update path to your download location
    recs = load_grabmyo("data/grabmyo", max_subjects=2)
    r = recs[0]
    print(f"  First record: subject={r['subject_id']}, gesture={r['gesture']}, "
          f"trial={r['trial']}, emg shape={r['emg'].shape}, label={r['label']}")
