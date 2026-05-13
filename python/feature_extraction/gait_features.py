"""
gait_features.py
----------------
Extracts features from the Gait in Parkinson's Disease dataset.

The dataset contains 18 force sensor channels (left/right foot pressure sensors)
at 100 Hz. Features capture gait rhythm, symmetry, and variability — all known
to differ between healthy controls and Parkinson's patients.

Features computed:
  1. stride_regularity   — autocorrelation peak (gait rhythm regularity)
  2. gait_asymmetry      — difference between left and right foot total force
  3. force_variability   — coefficient of variation of total vertical force
  4. cadence             — estimated steps per minute from dominant frequency
  5. freeze_index        — ratio of freeze-band to locomotion-band power
                           (elevated in Parkinson's freezing of gait)

All functions expect a 2D numpy array of shape (n_samples, 18) and fs=100.
"""

import numpy as np
from scipy.signal import welch, butter, filtfilt


# ── Filter helpers ────────────────────────────────────────────────────────────

def _lowpass(signal: np.ndarray, cutoff: float, fs: int, order: int = 4) -> np.ndarray:
    nyq = fs / 2
    b, a = butter(order, cutoff / nyq, btype="low")
    return filtfilt(b, a, signal)


# ── Feature functions ─────────────────────────────────────────────────────────

def compute_stride_regularity(total_force: np.ndarray, fs: int) -> float:
    """
    Stride regularity via autocorrelation.
    A regular gait produces a strong periodic autocorrelation peak.
    Parkinson's patients show reduced regularity.
    """
    signal = total_force - np.mean(total_force)
    if np.std(signal) == 0:
        return 0.0
    # Normalized autocorrelation
    autocorr = np.correlate(signal, signal, mode='full')
    autocorr = autocorr[len(autocorr) // 2:]  # keep positive lags
    autocorr /= autocorr[0]  # normalize

    # Look for first peak between 0.3s and 2.5s (typical stride range)
    min_lag = int(0.3 * fs)
    max_lag = int(2.5 * fs)
    max_lag = min(max_lag, len(autocorr) - 1)

    if min_lag >= max_lag:
        return 0.0

    peak_val = float(np.max(autocorr[min_lag:max_lag]))
    return peak_val


def compute_gait_asymmetry(gait: np.ndarray) -> float:
    """
    Asymmetry between left and right foot total force.
    The dataset has 18 channels — first 8 are left foot sensors,
    next 8 are right foot sensors (columns 0-7 and 8-15),
    last 2 are total left/right force (columns 16-17).
    We use columns 16 and 17 directly.
    """
    if gait.shape[1] < 18:
        return 0.0
    left_total  = np.mean(gait[:, 16])
    right_total = np.mean(gait[:, 17])
    total = left_total + right_total
    if total == 0:
        return 0.0
    asymmetry = abs(left_total - right_total) / total
    return float(asymmetry)


def compute_force_variability(total_force: np.ndarray) -> float:
    """
    Coefficient of variation (CV) of total vertical force.
    Higher variability indicates less stable gait — common in Parkinson's.
    """
    mean_force = np.mean(total_force)
    if mean_force == 0:
        return 0.0
    return float(np.std(total_force) / mean_force)


def compute_cadence(total_force: np.ndarray, fs: int) -> float:
    """
    Estimated cadence (steps per minute) from dominant frequency
    of the total force signal in the 0.5–3 Hz range.
    """
    freqs, psd = welch(total_force, fs=fs, nperseg=min(len(total_force), 256))
    mask = (freqs >= 0.5) & (freqs <= 3.0)
    if not np.any(mask):
        return 0.0
    dominant_freq = freqs[mask][np.argmax(psd[mask])]
    # Convert Hz to steps per minute (* 60)
    return float(dominant_freq * 60)


def compute_freeze_index(total_force: np.ndarray, fs: int) -> float:
    """
    Freeze index — ratio of power in freeze band (3–8 Hz) to
    locomotion band (0.5–3 Hz). Elevated in Parkinson's freezing of gait.
    """
    freqs, psd = welch(total_force, fs=fs, nperseg=min(len(total_force), 256))

    locomotion_mask = (freqs >= 0.5) & (freqs < 3.0)
    freeze_mask     = (freqs >= 3.0) & (freqs <= 8.0)

    locomotion_power = np.sum(psd[locomotion_mask])
    freeze_power     = np.sum(psd[freeze_mask])

    if locomotion_power == 0:
        return 0.0
    return float(freeze_power / locomotion_power)


# ── Main interface ────────────────────────────────────────────────────────────

def extract_gait_features(gait: np.ndarray, fs: int) -> dict:
    """
    Full gait feature extraction pipeline.

    Parameters
    ----------
    gait : 2D array of shape (n_samples, 18) — force sensor channels
    fs   : sampling frequency in Hz (100 for this dataset)

    Returns
    -------
    dict with keys: stride_regularity, gait_asymmetry, force_variability,
                    cadence, freeze_index
    """
    nan_result = {k: np.nan for k in [
        "stride_regularity", "gait_asymmetry", "force_variability",
        "cadence", "freeze_index"
    ]}

    if gait.ndim != 2 or gait.shape[0] < fs:  # need at least 1 second
        return nan_result

    # Total force = sum across all 16 foot sensor channels (cols 0-15)
    total_force = np.sum(gait[:, :16], axis=1).astype(np.float64)

    # Smooth slightly to reduce noise
    total_force_smooth = _lowpass(total_force, cutoff=10.0, fs=fs)

    return {
        "stride_regularity": compute_stride_regularity(total_force_smooth, fs),
        "gait_asymmetry":    compute_gait_asymmetry(gait),
        "force_variability": compute_force_variability(total_force_smooth),
        "cadence":           compute_cadence(total_force_smooth, fs),
        "freeze_index":      compute_freeze_index(total_force_smooth, fs),
    }


if __name__ == "__main__":
    # Quick test with synthetic data
    fs = 100
    t = np.linspace(0, 10, fs * 10)
    # Simulate 18 force channels with a 1Hz gait cycle
    fake_gait = np.random.randn(len(t), 18) * 10 + 50
    fake_gait[:, 16] = np.abs(np.sin(2 * np.pi * 1.0 * t)) * 200  # left total
    fake_gait[:, 17] = np.abs(np.sin(2 * np.pi * 1.0 * t + 0.5)) * 200  # right total
    feats = extract_gait_features(fake_gait, fs)
    print("Gait features from synthetic signal:")
    for k, v in feats.items():
        print(f"  {k}: {v:.4f}")