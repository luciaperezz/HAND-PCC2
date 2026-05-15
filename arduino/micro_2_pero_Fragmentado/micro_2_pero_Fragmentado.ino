#include <bluefruit.h>
#include <LSM6DS3.h>
#include <Wire.h>  // I2C communication

/////////////////////////////////// 
// MICRO 2 - PRESSURE + IMU (I2C) + BLE (1 frame)
///////////////////////////////////

LSM6DS3 imu(I2C_MODE, 0x6A);   // IMU over I2C (internal XIAO Sense)
const int PRESSURE_PIN = D0;    // Analog pressure sensor
BLEUart bleuart;                // BLE UART service

String emg_buffer = "";         // Buffer for incoming EMG data (from Serial1)
unsigned long last_frame_time = 0; // Timer to control frame rate
int frame_id = 0;               // Frame counter

//----------------------------------
// Start BLE advertising
//----------------------------------
void startAdv() {
  Bluefruit.Advertising.addFlags(BLE_GAP_ADV_FLAGS_LE_ONLY_GENERAL_DISC_MODE);
  Bluefruit.Advertising.addTxPower();
  Bluefruit.Advertising.addService(bleuart);
  Bluefruit.ScanResponse.addName();
  Bluefruit.Advertising.restartOnDisconnect(true);
  Bluefruit.Advertising.setInterval(32, 244);
  Bluefruit.Advertising.start(0);
}

//----------------------------------
// Setup
//----------------------------------
void setup() {
  Serial.begin(115200);
  delay(2000);

  Serial1.begin(115200);       // EMG data comes from micro_1 via UART
  pinMode(PRESSURE_PIN, INPUT);

  // Initialize IMU
  if (imu.begin() == 0) {
    Serial.println("✅ IMU I2C OK");
    imu.writeRegister(0x10, 0x60);  // Accel: 104 Hz, ±2g
    imu.writeRegister(0x11, 0x60);  // Gyro: 104 Hz, ±250 dps
    imu.writeRegister(0x12, 0x04);  // Enable interrupts
  } else {
    Serial.println("❌ IMU I2C fail");
  }

  // Initialize BLE
  Bluefruit.begin();
  Bluefruit.setName("SensorNode");
  bleuart.begin();
  startAdv();

  Serial.println("🚀 micro_2: Pressure + IMU + EMG → BLE (1 frame)");
}

//----------------------------------
// Main loop
//----------------------------------
void loop() {
  // Read EMG from Serial1
  int emg = 0;
  while (Serial1.available()) {
    char c = Serial1.read();

    if (c == '\n') {
      emg_buffer.trim();
      if (emg_buffer.startsWith("EMG:")) {
        emg = atoi(emg_buffer.substring(4).c_str());
      }
      emg_buffer = "";
    } else {
      emg_buffer += c;
      if (emg_buffer.length() > 50) emg_buffer = ""; // prevent overflow
    }
  }

  // Send one frame every 40 ms (~25 Hz)
  if (millis() - last_frame_time >= 40) {
    last_frame_time = millis();

    // Read pressure sensor
    int pressure = analogRead(PRESSURE_PIN)/ 5.12;;

    // Read IMU
    float ax = imu.readFloatAccelX();
    float ay = imu.readFloatAccelY();
    float az = imu.readFloatAccelZ();
    float gx = imu.readFloatGyroX();
    float gy = imu.readFloatGyroY();
    float gz = imu.readFloatGyroZ();

    // Construct single frame string
    String msg = "F,";
    msg += frame_id++; msg += ",";
    msg += pressure; msg += ",";
    msg += emg; msg += ",";
    msg += String(ax, 2); msg += ",";
    msg += String(ay, 2); msg += ",";
    msg += String(az, 2); msg += ",";
    msg += String(gx, 2); msg += ",";
    msg += String(gy, 2); msg += ",";
    msg += String(gz, 2);
    msg += "\n";

    // Send over BLE
    bleuart.print(msg);

    // Debug
    Serial.print("📤 SENT: ");
    Serial.print(msg);
  }
}