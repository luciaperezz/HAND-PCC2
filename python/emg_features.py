"""
emg_features.py
---------------
Extracts the EMG features defined in the HAND report from a raw EMG signal.

Features computed (matching Section 4.3.2 of the report):
  1. emg_rms            — overall muscle activation level
  2. median_freq        — spectral median frequency (fatigue indicator)
  3. fatigue_index      — RMS first-half vs second-half decline
  4. tremor_band_power  — spectral energy in the 2–12 Hz envelope band
  5. peak_tremor_freq   — dominant frequency within the tremor band

All functions expect a 1D numpy array of raw EMG samples and the sampling frequency fs.
"""

import numpy as np
from scipy.signal import butter, filtfilt, welch


# ── Filter helpers ────────────────────────────────────────────────────────────

def _bandpass(signal: np.ndarray, lowcut: float, highcut: float, fs: int,
              order: int = 4) -> np.ndarray:
    nyq = fs / 2
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    return filtfilt(b, a, signal)


def _lowpass(signal: np.ndarray, cutoff: float, fs: int, order: int = 4) -> np.ndarray:
    nyq = fs / 2
    b, a = butter(order, cutoff / nyq, btype="low")
    return filtfilt(b, a, signal)


def _notch(signal: np.ndarray, notch_freq: float, fs: int, Q: float = 30) -> np.ndarray:
    """Simple IIR notch filter at notch_freq Hz."""
    from scipy.signal import iirnotch
    b, a = iirnotch(notch_freq / (fs / 2), Q)
    return filtfilt(b, a, signal)


def preprocess_emg(raw: np.ndarray, fs: int) -> np.ndarray:
    """
    Standard EMG preprocessing pipeline:
      1. Remove DC offset
      2. Notch filter at 50 Hz
      3. Bandpass 20–450 Hz
    Returns cleaned signal.
    """
    signal = raw - np.mean(raw)
    if fs >= 110:  # only notch if sampling rate allows it
        signal = _notch(signal, 50.0, fs)
    signal = _bandpass(signal, 20.0, min(450.0, fs / 2 - 1), fs)
    return signal


def emg_envelope(signal: np.ndarray, fs: int, window_ms: int = 50) -> np.ndarray:
    """
    Compute EMG amplitude envelope:
      rectify → moving RMS with `window_ms` ms window.
    """
    rectified = np.abs(signal)
    window = max(1, int(fs * window_ms / 1000))
    # Moving RMS via convolution
    kernel = np.ones(window) / window
    rms_env = np.sqrt(np.convolve(rectified ** 2, kernel, mode="same"))
    return rms_env


# ── Feature functions ─────────────────────────────────────────────────────────

def compute_emg_rms(signal: np.ndarray) -> float:
    """
    EMG RMS — overall activation level (Eq. 8 in report).
    Input: preprocessed EMG (not rectified).
    """
    return float(np.sqrt(np.mean(signal ** 2)))


def compute_median_frequency(signal: np.ndarray, fs: int) -> float:
    """
    Spectral median frequency of the EMG signal.
    The frequency that divides the power spectrum into two equal halves.
    Decrease over a contraction indicates fatigue.
    """
    freqs, psd = welch(signal, fs=fs, nperseg=min(len(signal), 256))
    cumulative = np.cumsum(psd)
    total = cumulative[-1]
    if total == 0:
        return 0.0
    idx = np.searchsorted(cumulative, total / 2)
    return float(freqs[min(idx, len(freqs) - 1)])


def compute_fatigue_index(signal: np.ndarray) -> float:
    """
    Fatigue index based on RMS decline from first to second half (Eq. 9 in report).
    Negative value = signal decays = fatigue present.
    """
    mid = len(signal) // 2
    rms_start = float(np.sqrt(np.mean(signal[:mid] ** 2)))
    rms_end = float(np.sqrt(np.mean(signal[mid:] ** 2)))
    # Delta fatigue = RMS_end - RMS_start (negative = fatiguing)
    if rms_start == 0:
        return 0.0
    return float(rms_end - rms_start)


def compute_tremor_band_power(signal: np.ndarray, fs: int,
                               f_low: float = 2.0, f_high: float = 12.0) -> float:
    """
    Tremor band power in the EMG envelope (Eq. 10 in report).
    Steps: bandpass EMG → rectify → lowpass envelope → PSD → integrate [f_low, f_high].
    """
    envelope = emg_envelope(signal, fs)
    freqs, psd = welch(envelope, fs=fs, nperseg=min(len(envelope), 256))
    mask = (freqs >= f_low) & (freqs <= f_high)
    if not np.any(mask):
        return 0.0
    df = freqs[1] - freqs[0]
    return float(np.sum(psd[mask]) * df)


def compute_peak_tremor_frequency(signal: np.ndarray, fs: int,
                                   f_low: float = 2.0, f_high: float = 12.0) -> float:
    """
    Peak tremor frequency in the envelope spectrum (Eq. 11 in report).
    Returns the dominant frequency in [f_low, f_high] Hz.
    """
    envelope = emg_envelope(signal, fs)
    freqs, psd = welch(envelope, fs=fs, nperseg=min(len(envelope), 256))
    mask = (freqs >= f_low) & (freqs <= f_high)
    if not np.any(mask):
        return 0.0
    peak_idx = np.argmax(psd[mask])
    return float(freqs[mask][peak_idx])


# ── Main interface ─────────────────────────────────────────────────────────────

def extract_emg_features(raw_emg: np.ndarray, fs: int) -> dict:
    """
    Full EMG feature extraction pipeline.

    Parameters
    ----------
    raw_emg : 1D array of raw EMG samples (single channel)
    fs      : sampling frequency in Hz

    Returns
    -------
    dict with keys: emg_rms, median_freq, fatigue_index,
                    tremor_band_power, peak_tremor_freq
    """
    if len(raw_emg) < fs:  # need at least 1 second
        return {k: np.nan for k in
                ["emg_rms", "median_freq", "fatigue_index",
                 "tremor_band_power", "peak_tremor_freq"]}

    signal = preprocess_emg(raw_emg, fs)

    return {
        "emg_rms":           compute_emg_rms(signal),
        "median_freq":       compute_median_frequency(signal, fs),
        "fatigue_index":     compute_fatigue_index(signal),
        "tremor_band_power": compute_tremor_band_power(signal, fs),
        "peak_tremor_freq":  compute_peak_tremor_frequency(signal, fs),
    }


def extract_emg_features_multichannel(emg: np.ndarray, fs: int) -> dict:
    """
    Run extract_emg_features on each channel and return mean across channels.
    Input: emg shape (n_samples, n_channels)
    """
    if emg.ndim == 1:
        return extract_emg_features(emg, fs)

    all_features = [extract_emg_features(emg[:, ch], fs) for ch in range(emg.shape[1])]
    keys = all_features[0].keys()
    return {
        k: float(np.nanmean([f[k] for f in all_features]))
        for k in keys
    }


if __name__ == "__main__":
    # Quick test with synthetic signal
    fs = 2048
    t = np.linspace(0, 3, fs * 3)
    # Simulate EMG: broadband noise + tremor component at 5 Hz
    fake_emg = (np.random.randn(len(t)) * 0.5 +
                np.sin(2 * np.pi * 5 * t) * 0.1)
    feats = extract_emg_features(fake_emg, fs)
    print("EMG features from synthetic signal:")
    for k, v in feats.items():
        print(f"  {k}: {v:.4f}")
