#include <LSM6DS3.h>
#include <Wire.h>
#include <Adafruit_TinyUSB.h>
#include <bluefruit.h>

const int PRESSURE_PIN = D2;   // Replace with your pressure sensor pin
const int EMG_PIN = D0;        // EMG sensor pin

LSM6DS3 myIMU(I2C_MODE, 0x6A);
BLEUart bleuart;

// ---------------- BLE Setup ----------------
void startAdv() {
  Bluefruit.Advertising.addFlags(BLE_GAP_ADV_FLAGS_LE_ONLY_GENERAL_DISC_MODE);
  Bluefruit.Advertising.addTxPower();
  Bluefruit.Advertising.addService(bleuart);
  Bluefruit.ScanResponse.addName();
  Bluefruit.Advertising.restartOnDisconnect(true);
  Bluefruit.Advertising.setInterval(32, 244);
  Bluefruit.Advertising.start(0);
}

void setup() {
  Serial.begin(115200);
  Wire.begin();

  if (myIMU.begin() != 0) while(1); // IMU check

  pinMode(PRESSURE_PIN, INPUT);
  pinMode(EMG_PIN, INPUT);

  Bluefruit.begin();
  Bluefruit.setName("SensorNode");
  bleuart.begin();
  startAdv();
}

// ---------------- EMG Filtering ----------------
float x[3] = {0, 0, 0};
float y[3] = {0, 0, 0};
float mean = 0;
float alpha = 0.01;

// ---------------- Loop ----------------
int bleCounter = 0;
void loop() {
  // -------- Pressure ----------
  uint16_t pressure = analogRead(PRESSURE_PIN);

  // -------- EMG ----------
  float raw_emg = analogRead(EMG_PIN);
  mean = (1 - alpha)*mean + alpha*raw_emg;
  float emg_signal = raw_emg - mean;

  x[0] = emg_signal;
  float b0 = 0.9722, b1 = -1.6180, b2 = 0.9722;
  float a1 = -1.6180, a2 = 0.9445;
  y[0] = b0*x[0] + b1*x[1] + b2*x[2] - a1*y[1] - a2*y[2];
  x[2] = x[1]; x[1] = x[0];
  y[2] = y[1]; y[1] = y[0];

  // -------- IMU ----------
  float ax = myIMU.readFloatAccelX();
  float ay = myIMU.readFloatAccelY();
  float az = myIMU.readFloatAccelZ();
  float gx = myIMU.readFloatGyroX();
  float gy = myIMU.readFloatGyroY();
  float gz = myIMU.readFloatGyroZ();

  int16_t ax_i = (int16_t)(ax * 1000);
  int16_t ay_i = (int16_t)(ay * 1000);
  int16_t az_i = (int16_t)(az * 1000);
  int16_t gx_i = (int16_t)(gx * 1000);
  int16_t gy_i = (int16_t)(gy * 1000);
  int16_t gz_i = (int16_t)(gz * 1000);

  // -------- BLE send 10 Hz ----------
  bleCounter++;
  if (bleCounter >= 10) {
    bleCounter = 0;
    bleuart.print(pressure); bleuart.print(",");
    bleuart.print(y[0], 2); bleuart.print(",");      // EMG
    bleuart.print(ax_i); bleuart.print(",");
    bleuart.print(ay_i); bleuart.print(",");
    bleuart.print(az_i); bleuart.print(",");
    bleuart.print(gx_i); bleuart.print(",");
    bleuart.print(gy_i); bleuart.print(",");
    bleuart.print(gz_i); bleuart.print("\n");
  }

  delayMicroseconds(1000);  // ~1 kHz sampling
}