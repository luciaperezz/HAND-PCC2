# main.py
# This file is the entry point for Phase 1
# It calls all functions: reading, processing, extraction, rules, and visualization
"""
import time
from serial_reader import read_serial
from signal_processing import process_pressure, process_emg, process_imu
from feature_extraction import extract_pressure_features, extract_emg_features, extract_imu_features
from rules_feedback import evaluate_rules

def run_session():

    Main flow for an exercise session.


    #Inicializar variables o buffers
    buffer_time = [] # Stores timestamps
    buffer_force = [] # Stores pressure
    buffer_emg = [] # Stores EMG signal
    buffer_imu_acc = [] # Stores IMU acceleration
    buffer_imu_gyro = [] # Stores IMU angular velocity

    print("Starting test session... Press Ctrl+C to exit")

    try:
        while True:
            #Read data from the Arduino
            time_stamp, force, emg, imu_acc, imu_gyro = read_serial()
            
            # Store in the buffers
            buffer_time.append(time_stamp)
            buffer_force.append(force)
            buffer_emg.append(emg)
            buffer_imu_acc.append(imu_acc)
            buffer_imu_gyro.append(imu_gyro)

            #Process the signal
            force_proc = process_pressure(buffer_force)
            emg_proc = process_emg(buffer_emg)
            imu_proc = process_imu(buffer_imu_acc, buffer_imu_gyro)

            # Extact the features
            features_pressure = extract_pressure_features(force_proc)
            features_emg = extract_emg_features(emg_proc)
            features_imu = extract_imu_features(imu_proc)

            #Evaluate rules and generate feedback
            feedback_msg = evaluate_rules(features_pressure, features_emg, features_imu)

            #Show feedback
            plot_realtime(force_proc, emg_proc, imu_proc, feedback_msg)

            # Wait a short interval to avoid saturation
            time.sleep(0.05)  # 50 ms → 20 Hz
    except KeyboardInterrupt:
        print("Sesion Finished.")
        # You can save the data as a CSV file here if you want.
        # save_data(buffer_time, buffer_force, buffer_emg, buffer_imu)

if __name__ == "__main__":
    run_session()
        """
# main.py
import asyncio
from collections import deque
from ble_client import run_ble

# Buffers
buffer_pressure = deque(maxlen=200)
buffer_imu_acc = deque(maxlen=200)
buffer_imu_gyro = deque(maxlen=200)

def parse_line(line):
    """Convierte la línea BLE en diccionario"""
    try:
        parts = line.split(",")
        return {
            "pressure": int(parts[0]),
            "ax": int(parts[1]),
            "ay": int(parts[2]),
            "az": int(parts[3]),
            "gx": int(parts[4]),
            "gy": int(parts[5]),
            "gz": int(parts[6])
        }
    except Exception:
        return None

def on_data(line):
    data = parse_line(line)
    if not data:
        return

    buffer_pressure.append(data["pressure"])
    buffer_imu_acc.append((data["ax"], data["ay"], data["az"]))
    buffer_imu_gyro.append((data["gx"], data["gy"], data["gz"]))

    # Para test rápido
    print(f"P:{data['pressure']}  AX:{data['ax']} AY:{data['ay']} AZ:{data['az']}")

if __name__ == "__main__":
    asyncio.run(run_ble(on_data))