def parse_line(line):
    try:
        parts = line.split(",")

        pressure = int(parts[0])

        ax = int(parts[1]) / 1000.0
        ay = int(parts[2]) / 1000.0
        az = int(parts[3]) / 1000.0

        gx = int(parts[4]) / 1000.0
        gy = int(parts[5]) / 1000.0
        gz = int(parts[6]) / 1000.0

        return {
            "pressure": pressure,
            "acc": (ax, ay, az),
            "gyro": (gx, gy, gz)
        }

    except:
        return None