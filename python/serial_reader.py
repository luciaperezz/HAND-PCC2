#function to read from the Arduino

# read_serial ()
#connect via Bluethooth, parse CSV and return the array

def read_serial():
    """
    Reads current data from Arduino.
    Returns: time_stamp, force, emg, imu_acc, imu_gyro
    For now, we're using example values for testing without Arduino.

    """
    import time
    import random

    time_stamp = time.time()
    force = random.uniform(0, 10)       # simulate pressure
    emg = random.uniform(0, 1)          # simulate EMG
    imu_acc = [random.uniform(-1,1) for _ in range(3)]
    imu_gyro = [random.uniform(-1,1) for _ in range(3)]

    return time_stamp, force, emg, imu_acc, imu_gyro