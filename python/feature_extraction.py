#calculates metrics
#maybe it is better to divide them by exercises and desseases


import numpy as np
from scipy.signal import butter, filtfilt, welch
from signal_processing import process_pressure, process_emg, compute_magnitude, compute_psd, process_imu

sampling_rate = 1000  
dt = 1 / sampling_rate  # 0.001 s

# ------------------- EXERCISE 1 STROKE -------------------
def extract_features_ex1_stroke(signal, mvc):
    pressure_signal = signal['pressure']
    emg_signal = signal['emg']

#---- Pressure features ----
    # Peak force (MVC)
    peak_force = np.max(pressure_signal)
    normalized_peak = peak_force /mvc if mvc != 0 else np.nan

    # Rate of force Development (RFD)
    rfd = np.diff(pressure_signal).max() / dt if len(pressure_signal) > 1 else np.nan

    # Time to peak Force
    time_to_peak = np.argmax(pressure_signal) / sampling_rate if len(pressure_signal) > 0 else np.nan

#------ EMG features -------
    # Peak activation (RMS)
    rms_emg = np.sqrt(np.mean(emg_signal**2)) if len(emg_signal) > 0 else np.nan

    # Electromechanical Delay (EMD)
    emg_threshold = 0.05 * np.max(emg_signal) if len(emg_signal) > 0 else np.nan
    force_threshold = 0.05 * np.max(pressure_signal) if len(pressure_signal) > 0 else np.nan
    emg_indices = np.where(emg_signal >= emg_threshold)[0] if len(emg_signal) > 0 else []
    force_indices = np.where(pressure_signal >= force_threshold)[0] if len(pressure_signal) > 0 else []
    emd = (force_indices[0] - emg_indices[0]) * dt if len(emg_indices) > 0 and len(force_indices) > 0 else np.nan

    features = {
        'peak_force': peak_force,
        'normalized_peak': normalized_peak,
        'time_to_peak': time_to_peak,
        'rfd': rfd,
        'rms_emg': rms_emg,
        'emd' : emd
    }

    return features

# ------------------- EXERCISE 2 STROKE -------------------
def extract_features_ex2_stroke(signal, mvc):
    # compute features for Exercise 2, Stroke
    pressure_signal = signal['pressure']
    emg_signal = signal['emg']

#----- Pressure features
    # Force variavility (SD/CV)
    sd_force = np.std(pressure_signal) if len(pressure_signal) > 0 else np.nan
    mean_force = np.mean(pressure_signal) if len(pressure_signal) > 0 else np.nan
    cv_force = sd_force / mean_force if mean_force != 0 else np.nan
    
    # Fatigue Index
    window = max(1, int(len(pressure_signal)/10))
    f_max_start = np.max(pressure_signal[:window]) if len(pressure_signal) >= window else np.nan
    f_max_end = np.max(pressure_signal[-window:]) if len(pressure_signal) >= window else np.nan
    fatigue_index = (f_max_start - f_max_end) / f_max_start if f_max_start != 0 else np.nan

    # Time to target Band (20% MVC)
    target_force = 0.2 * mvc
    indices = np.where(pressure_signal >= target_force)[0] if len(pressure_signal) > 0 else []
    time_to_target = indices[0]*dt if len(indices) > 0 else np.nan

# --- EMG features ---
    # Peak activation (RMS)
    rms_emg = np.sqrt(np.mean(emg_signal**2)) if len(emg_signal) > 0 else np.nan

    # Median Frequency shift (fatigue)
    f, Pxx = welch(emg_signal, fs=sampling_rate) if len(emg_signal) > 0 else ([], [])
    if len(Pxx) > 0:
        cum_power = np.cumsum(Pxx)
        median_freq_global = f[np.searchsorted(cum_power, cum_power[-1]/2)]
        # Median frequency shift: inicio vs final
        mid = len(emg_signal)//2
        f_start, Pxx_start = welch(emg_signal[:mid], fs=sampling_rate)
        f_end, Pxx_end = welch(emg_signal[mid:], fs=sampling_rate)
        median_freq_start = f_start[np.searchsorted(np.cumsum(Pxx_start), np.cumsum(Pxx_start)[-1]/2)]
        median_freq_end = f_end[np.searchsorted(np.cumsum(Pxx_end), np.cumsum(Pxx_end)[-1]/2)]
        median_freq_shift = median_freq_start - median_freq_end
    else:
        median_freq_global = np.nan
        median_freq_shift = np.nan

    # Co-activation index

    features = {
        'cv_force': cv_force,
        'fatigue_index': fatigue_index,
        'time_to_target': time_to_target,
        'rms_emg': rms_emg,
        'median_freq_global': median_freq_global,
        'median_freq_shift': median_freq_shift
    }
    return features

def extract_features_ex1_tremor(signal, mvc):
    # compute features for Exercise 1, Tremor
    pressure_signal = signal['pressure']
    imu_signal = signal['imu']
    ax = imu_signal['ax']
    ay = imu_signal['ay']
    az = imu_signal['az']
    magnitude = np.sqrt(ax**2 + ay**2 + az**2) if len(ax) > 0 else np.array([])
# --- Pressure features ---
    # Tremor Amplitude (force oscillation)
    tremor_amplitude = np.std(pressure_signal) if len(pressure_signal) > 0 else np.nan

    # Force variavility (SD/CV)
    sd_force = np.std(pressure_signal) if len(pressure_signal) > 0 else np.nan
    mean_force = np.mean(pressure_signal) if len(pressure_signal) > 0 else np.nan
    cv_force = sd_force / mean_force if mean_force != 0 else np.nan
    # Spectral power in tremor freq band
    f, Pxx = welch(pressure_signal, fs=sampling_rate) if len(pressure_signal) > 0 else ([], [])
    if len(Pxx) > 0:
        tremor_band = (f >= 4) & (f <= 12)
        spectral_power_tremor = np.sum(Pxx[tremor_band])
    else:
        spectral_power_tremor = np.nan

#----- IMU ---------------
    # Tremor amplitude
    imu_tremor_amplitude = np.std(magnitude) if len(magnitude) > 0 else np.nan

    # Dominant Frequency
    if len(magnitude) > 0:
        f_imu, Pxx_imu = welch(magnitude, fs=sampling_rate)
        dominant_freq = f_imu[np.argmax(Pxx_imu)]
    else:
        dominant_freq = np.nan
    # Orientation Stability
    orientation_stability = 1 / (np.std(ax) + np.std(ay) + np.std(az)) if len(ax) > 0 else np.nan

    features = {
        'tremor_amplitude_force': tremor_amplitude,
        'cv_force': cv_force,
        'spectral_power_tremor': spectral_power_tremor,
        'imu_tremor_amplitude': imu_tremor_amplitude,
        'dominant_freq': dominant_freq,
        'orientation_stability': orientation_stability
    }
    
    return features

def extract_features_ex2_tremor(signal, mvc):
    # compute features for Exercise 2, Tremor
    pressure_signal = signal['pressure']
    imu_signal = signal['imu']
    ax = imu_signal['ax']
    ay = imu_signal['ay']
    az = imu_signal['az']
    magnitude = np.sqrt(ax**2 + ay**2 + az**2) if len(ax) > 0 else np.array([])
#---------- Pressure ---------
    # Cycle timing
    peaks = np.where(pressure_signal > 0.5 * mvc)[0] if len(pressure_signal) > 0 else []
    if len(peaks) > 1:
        cycle_intervals = np.diff(peaks) * dt
        mean_cycle_time = np.mean(cycle_intervals)
        peak_force_consistency = np.std(pressure_signal[peaks])
    else:
        mean_cycle_time = np.nan
        peak_force_consistency = np.nan
    # Peak Force consistency
    # Amplitude Regularity
    amplitude_regularity = np.std(pressure_signal[peaks])/np.mean(pressure_signal[peaks]) if len(peaks) > 0 else np.nan

#---------- IMU ---------------
    # Movement Smoothness = 1 /std of acceleration
    movement_smoothness = 1 / (np.std(magnitude)) if len(magnitude) > 0 else np.nan

    # Timing Consistency = std of cycles interval
    timing_consistency = np.std(cycle_intervals) if len(peaks) > 1 else np.nan

    # Amplitude Regularity
    amplitude_regularity_imu = np.std(magnitude)/np.mean(magnitude) if len(magnitude) > 0 else np.nan

    features = {
        'mean_cycle_time': mean_cycle_time,
        'peak_force_consistency': peak_force_consistency,
        'amplitude_regularity_force': amplitude_regularity,
        'movement_smoothness': movement_smoothness,
        'timing_consistency': timing_consistency,
        'amplitude_regularity_imu': amplitude_regularity_imu
    }
    
    return features


