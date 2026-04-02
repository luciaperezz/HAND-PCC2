"""
imu_features.py
---------------
Extracts the IMU features defined in the HAND report from raw accelerometer + gyroscope data.

Features computed (matching Section 4.3.3 of the report):
  1. tremor_power         — total spectral energy in 4–12 Hz band (acceleration magnitude)
  2. tremor_frequency     — dominant frequency within 4–12 Hz
  3. orientation_stability— variance of angular position estimate during hold phase
  4. range_of_motion      — max − min angle during a movement
  5. hold_stability       — SD of angular velocity magnitude during hold
  6. drift_rate           — linear trend in orientation over time

All functions accept:
  accel : np.ndarray shape (n_samples, 3)  — ax, ay, az in g
  gyro  : np.ndarray shape (n_samples, 3)  — gx, gy, gz in rad/s
  fs    : int — sampling frequency in Hz
"""

import numpy as np
from scipy.signal import butter, filtfilt, welch


# ── Filter helpers ─────────────────────────────────────────────────────────────

def _lowpass(signal: np.ndarray, cutoff: float, fs: int, order: int = 4) -> np.ndarray:
    nyq = fs / 2
    b, a = butter(order, cutoff / nyq, btype="low")
    return filtfilt(b, a, signal, axis=0)


# ── Signal magnitudes ──────────────────────────────────────────────────────────

def _accel_magnitude(accel: np.ndarray) -> np.ndarray:
    """Euclidean magnitude of 3-axis acceleration."""
    return np.sqrt(np.sum(accel ** 2, axis=1))


def _gyro_magnitude(gyro: np.ndarray) -> np.ndarray:
    """Euclidean magnitude of 3-axis angular velocity."""
    return np.sqrt(np.sum(gyro ** 2, axis=1))


# ── Orientation from gyro (simple integration) ────────────────────────────────

def _integrate_gyro(gyro: np.ndarray, fs: int) -> np.ndarray:
    """
    Estimate cumulative angle by integrating gyroscope signal.
    Returns orientation magnitude array (shape: n_samples).
    Simple trapezoidal integration — good enough for feature extraction
    over short windows.
    """
    dt = 1.0 / fs
    angle = np.cumsum(_gyro_magnitude(gyro)) * dt
    return angle


# ── Feature functions ──────────────────────────────────────────────────────────

def compute_tremor_power(accel: np.ndarray, fs: int,
                          f_low: float = 4.0, f_high: float = 12.0) -> float:
    """
    Tremor power: total PSD energy in [f_low, f_high] Hz (Eq. 13 in report).
    Computed on acceleration magnitude.
    """
    amag = _accel_magnitude(accel)
    freqs, psd = welch(amag, fs=fs, nperseg=min(len(amag), 256))
    mask = (freqs >= f_low) & (freqs <= f_high)
    if not np.any(mask):
        return 0.0
    df = freqs[1] - freqs[0]
    return float(np.sum(psd[mask]) * df)


def compute_tremor_frequency(accel: np.ndarray, fs: int,
                              f_low: float = 4.0, f_high: float = 12.0) -> float:
    """
    Dominant tremor frequency within [f_low, f_high] Hz (Eq. 14 in report).
    """
    amag = _accel_magnitude(accel)
    freqs, psd = welch(amag, fs=fs, nperseg=min(len(amag), 256))
    mask = (freqs >= f_low) & (freqs <= f_high)
    if not np.any(mask):
        return 0.0
    peak_idx = np.argmax(psd[mask])
    return float(freqs[mask][peak_idx])


def compute_orientation_stability(gyro: np.ndarray, fs: int) -> float:
    """
    Orientation stability: variance of integrated angle during the recording (Eq. 15 in report).
    Lower = more stable posture.
    """
    angle = _integrate_gyro(gyro, fs)
    return float(np.var(angle))


def compute_range_of_motion(gyro: np.ndarray, fs: int) -> float:
    """
    Range of motion: max − min cumulative angle (Eq. 16 in report).
    Higher ROM during grip = compensation strategy.
    """
    angle = _integrate_gyro(gyro, fs)
    return float(np.max(angle) - np.min(angle))


def compute_hold_stability(gyro: np.ndarray) -> float:
    """
    Hold stability: standard deviation of angular velocity magnitude (Eq. 17 in report).
    Higher SD = more tremor or co-contraction during static hold.
    """
    wmag = _gyro_magnitude(gyro)
    return float(np.std(wmag))


def compute_drift_rate(gyro: np.ndarray, fs: int) -> float:
    """
    Drift rate: linear trend (slope) in cumulative orientation over time (Eq. 18 in report).
    Reflects fatigue or loss of postural control.
    """
    angle = _integrate_gyro(gyro, fs)
    n = len(angle)
    if n < 2:
        return 0.0
    t = np.arange(n) / fs
    # Least-squares linear fit → slope is drift rate
    slope = np.polyfit(t, angle, 1)[0]
    return float(slope)


# ── Main interface ─────────────────────────────────────────────────────────────

def extract_imu_features(imu: np.ndarray, fs: int) -> dict:
    """
    Full IMU feature extraction pipeline.

    Parameters
    ----------
    imu : array of shape (n_samples, 6) — columns: ax, ay, az, gx, gy, gz
    fs  : sampling frequency in Hz

    Returns
    -------
    dict with keys: tremor_power, tremor_frequency, orientation_stability,
                    range_of_motion, hold_stability, drift_rate
    """
    if imu.shape[0] < fs:  # need at least 1 second
        return {k: np.nan for k in
                ["tremor_power", "tremor_frequency", "orientation_stability",
                 "range_of_motion", "hold_stability", "drift_rate"]}

    accel = imu[:, :3].astype(np.float64)  # ax, ay, az
    gyro  = imu[:, 3:].astype(np.float64)  # gx, gy, gz

    # Low-pass filter both at 20 Hz (as per report Section 4.2.1)
    accel_filt = _lowpass(accel, 20.0, fs)
    gyro_filt  = _lowpass(gyro,  20.0, fs)

    return {
        "tremor_power":          compute_tremor_power(accel_filt, fs),
        "tremor_frequency":      compute_tremor_frequency(accel_filt, fs),
        "orientation_stability": compute_orientation_stability(gyro_filt, fs),
        "range_of_motion":       compute_range_of_motion(gyro_filt, fs),
        "hold_stability":        compute_hold_stability(gyro_filt),
        "drift_rate":            compute_drift_rate(gyro_filt, fs),
    }


if __name__ == "__main__":
    # Quick test with synthetic IMU signal
    fs = 100
    t = np.linspace(0, 5, fs * 5)
    # Simulate: slow drift + tremor at 5 Hz + noise
    ax = np.sin(2 * np.pi * 5 * t) * 0.3 + np.random.randn(len(t)) * 0.05
    ay = np.cos(2 * np.pi * 5 * t) * 0.2 + np.random.randn(len(t)) * 0.05
    az = np.ones(len(t)) * 1.0 + np.random.randn(len(t)) * 0.02  # ~1g gravity
    gx = np.sin(2 * np.pi * 5 * t) * 0.5 + np.random.randn(len(t)) * 0.1
    gy = np.cos(2 * np.pi * 5 * t) * 0.4 + np.random.randn(len(t)) * 0.1
    gz = np.random.randn(len(t)) * 0.05
    imu = np.stack([ax, ay, az, gx, gy, gz], axis=1)

    feats = extract_imu_features(imu, fs)
    print("IMU features from synthetic signal:")
    for k, v in feats.items():
        print(f"  {k}: {v:.4f}")
