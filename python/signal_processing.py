import numpy as np
from scipy.signal import butter, filtfilt, welch

# ========================================================================================================================================================================================================
#                                                                                                       FILTERS
# ========================================================================================================================================================================================================

def lowpass_filter(signal, cutoff=20, fs=1000, order=2):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low')
    return filtfilt(b, a, signal)


def bandpass_filter(signal, fs, lowcut=20, highcut=450, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)


# ========================================================================================================================================================================================================
#                                                                                                       PRESSURE
# ========================================================================================================================================================================================================

def process_pressure(force, fs=1000):
    # Low-pass filter
    force_filtered = lowpass_filter(force, cutoff=20, fs=fs)

    # Baseline correction
    force_filtered = force_filtered - np.mean(force_filtered[:20])

    return force_filtered


# ========================================================================================================================================================================================================
#                                                                                                          EMG
# ========================================================================================================================================================================================================

def process_emg(emg, fs=1000):
    # Band-pass filter
    emg_filt = bandpass_filter(emg, fs)

    # Rectification
    emg_rect = np.abs(emg_filt)

    # RMS window
    window = int(0.1 * fs)  # 100 ms
    emg_rms = np.sqrt(
        np.convolve(emg_rect**2, np.ones(window)/window, mode='same')
    )

    return emg_rms


# ========================================================================================================================================================================================================
#                                                                                                           IMU
# ========================================================================================================================================================================================================

def compute_magnitude(x, y, z):
    return np.sqrt(x**2 + y**2 + z**2)


def compute_psd(signal, fs):
    f, Pxx = welch(signal, fs=fs, nperseg=128)
    return f, Pxx


def process_imu(ax, ay, az, gx, gy, gz, fs=100):
    
    # Magnitudes
    acc_mag = compute_magnitude(ax, ay, az)
    gyro_mag = compute_magnitude(gx, gy, gz)

    # Low-pass filter
    acc_filt = lowpass_filter(acc_mag, cutoff=20, fs=fs)
    gyro_filt = lowpass_filter(gyro_mag, cutoff=20, fs=fs)

    # PSD
    f, Pxx = compute_psd(acc_filt, fs)

    # Tremor band (4–12 Hz)
    mask = (f >= 4) & (f <= 12)

    tremor_power = np.sum(Pxx[mask])
    peak_freq = f[mask][np.argmax(Pxx[mask])] if np.any(mask) else 0

    return {
        "tremor_power": tremor_power,
        "peak_freq": peak_freq
    }