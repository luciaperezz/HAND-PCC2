def parse_line(line):
    """
    Converts a raw data line into a dictionary ready for processing.
    Expected CSV format: pressure, emg, ax, ay, az, gx, gy, gz
    """
    try:
        parts = line.strip().split(",")

        if len(parts) < 8:
            return None  # incomplete line

        # ── Pressure ──
        pressure = int(parts[0])

        # ── EMG ──
        emg = int(parts[1])

        # ── Accelerometer ──
        ax = int(parts[2]) / 1000.0
        ay = int(parts[3]) / 1000.0
        az = int(parts[4]) / 1000.0

        # ── Gyroscope ──
        gx = int(parts[5]) / 1000.0
        gy = int(parts[6]) / 1000.0
        gz = int(parts[7]) / 1000.0

        return {
            "pressure": pressure,
            "emg": emg,
            "ax": ax,
            "ay": ay,
            "az": az,
            "gx": gx,
            "gy": gy,
            "gz": gz
        }

    except ValueError:
        # Error converting some value
        return None