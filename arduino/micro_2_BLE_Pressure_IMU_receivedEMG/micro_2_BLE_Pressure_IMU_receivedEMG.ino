#include <bluefruit.h>
#include <LSM6DS3.h>
#include <Wire.h>  // I2C communication

/////////////////////////////////// 
// MICRO 2 - PRESSURE + IMU (I2C) + BLE
///////////////////////////////////

LSM6DS3 imu(I2C_MODE, 0x6A);   // IMU over I2C (internal XIAO Sense)
const int PRESSURE_PIN = D0;   // Analog pressure sensor
BLEUart bleuart;               // BLE UART service

String emg_buffer = "";        // Buffer for incoming EMG data (from Serial1)
unsigned long last_imu_time = 0; // Timer to control IMU sampling rate

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

    // Configure accelerometer and gyroscope
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

  Serial.println("🚀 micro_2: Pressure + IMU + EMG → BLE");
}

//----------------------------------
// Main loop
//----------------------------------
void loop() {

  // Read pressure sensor (scaled)
  float pressure = analogRead(PRESSURE_PIN) / 5.12;

  //----------------------------------
  // Read EMG from Serial1
  //----------------------------------
  float emg = 0;

  if (Serial1.available()) {
    emg_buffer = Serial1.readStringUntil('\n'); // Read full line
    emg_buffer.trim();

    // Expected format: "EMG:xxx"
    if (emg_buffer.startsWith("EMG:")) {
      emg = emg_buffer.substring(4).toFloat();
    }
  }

  //----------------------------------
  // Read IMU at ~25 Hz (every 40 ms)
  //----------------------------------
  float ax=0, ay=0, az=1.0, gx=0, gy=0, gz=0;

  if (millis() - last_imu_time > 40) {
    ax = imu.readFloatAccelX();
    ay = imu.readFloatAccelY();
    az = imu.readFloatAccelZ();

    gx = imu.readFloatGyroX();
    gy = imu.readFloatGyroY();
    gz = imu.readFloatGyroZ();

    last_imu_time = millis();
  }

  //----------------------------------
  // Scale values to integers
  // (reduces payload size for BLE transmission)
  //----------------------------------
  int p    = (int)(pressure * 10);

  int ax_i = (int)(ax * 100);
  int ay_i = (int)(ay * 100);
  int az_i = (int)(az * 100);

  int gx_i = (int)(gx * 100);
  int gy_i = (int)(gy * 100);
  int gz_i = (int)(gz * 100);

  //----------------------------------
  // BLE transmission (split into 3 packets)
  //----------------------------------

  // Packet 1: Header + Pressure + EMG
  bleuart.printf("S,%d,%d\n", p, (int)emg);

  // Packet 2: Accelerometer
  bleuart.printf("A,%d,%d,%d\n", ax_i, ay_i, az_i);

  // Packet 3: Gyroscope
  bleuart.printf("G,%d,%d,%d\n", gx_i, gy_i, gz_i);

  //----------------------------------
  // Loop rate control (~25 Hz)
  //----------------------------------
  delay(40);
}