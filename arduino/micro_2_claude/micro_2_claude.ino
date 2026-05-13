// ══════════════════════════════════════════════════════════════════
// MICRO_2 - CÓDIGO CORREGIDO (MATCH CON MICRO_1)
// ══════════════════════════════════════════════════════════════════

#include <bluefruit.h>
#include <LSM6DS3.h>
#include <Wire.h>

LSM6DS3 imu(I2C_MODE, 0x6A);
const int PRESSURE_PIN = D0;
BLEUart bleuart;

String emg_buffer = "";
unsigned long last_frame_time = 0;
int frame_id = 0;
int emg = 0;

void setup() {
  Serial.begin(115200);
  delay(2000);
  
  Serial1.begin(115200);
  pinMode(PRESSURE_PIN, INPUT);
  
  if (imu.begin() == 0) {
    Serial.println("✅ IMU OK");
    imu.writeRegister(0x10, 0x60);
    imu.writeRegister(0x11, 0x60);
  } else {
    Serial.println("❌ IMU ERROR");
  }
  
  Bluefruit.begin();
  Bluefruit.setName("SensorNode");
  bleuart.begin();
  startAdv();
  
  Serial.println("🚀 Micro_2 listo");
}

void loop() {
  
  // ════════════════════════════════════════════════════════════
  // LEER EMG desde Serial1 (formato: E,256)
  // ════════════════════════════════════════════════════════════
  while (Serial1.available()) {
    char c = Serial1.read();

    if (c == '\n') {
      emg_buffer.trim();

      // ✅ CAMBIO: Buscar "E," en lugar de "EMG:"
      if (emg_buffer.startsWith("E,")) {
        String emg_str = emg_buffer.substring(2);  // Saltar "E,"
        emg = emg_str.toInt();
        
        // Debug cada 100 frames
        if (frame_id % 100 == 0) {
          Serial.print("📥 EMG: ");
          Serial.println(emg);
        }
      }

      emg_buffer = "";
    }
    else if (c != '\r') {
      emg_buffer += c;
      if (emg_buffer.length() > 50) emg_buffer = "";
    }
  }
  
  // ════════════════════════════════════════════════════════════
  // ENVIAR FRAME cada 40ms (25 Hz)
  // ════════════════════════════════════════════════════════════
  
  if (millis() - last_frame_time >= 40) {
    last_frame_time = millis();
    
    int pressure = analogRead(PRESSURE_PIN) / 16;  // 0-4095 → 0-255
    
    float ax = imu.readFloatAccelX();
    float ay = imu.readFloatAccelY();
    float az = imu.readFloatAccelZ();
    float gx = imu.readFloatGyroX();
    float gy = imu.readFloatGyroY();
    float gz = imu.readFloatGyroZ();
    
    // ✅ FRAME COMPLETO: 10 campos
    String msg = "F,";
    msg += frame_id; msg += ",";
    msg += pressure; msg += ",";
    msg += emg; msg += ",";
    msg += String(ax, 2); msg += ",";
    msg += String(ay, 2); msg += ",";
    msg += String(az, 2); msg += ",";
    msg += String(gx, 2); msg += ",";
    msg += String(gy, 2); msg += ",";
    msg += String(gz, 2);
    msg += "\n";
    
    bleuart.print(msg);
    
    // Debug cada 100 frames
    if (frame_id % 100 == 0) {
      Serial.print("📤 Frame ");
      Serial.print(frame_id);
      Serial.print(": P=");
      Serial.print(pressure);
      Serial.print(" E=");
      Serial.print(emg);
      Serial.print(" Az=");
      Serial.println(az, 2);
    }
    
    frame_id++;
  }
}

void startAdv() {
  Bluefruit.Advertising.addFlags(BLE_GAP_ADV_FLAGS_LE_ONLY_GENERAL_DISC_MODE);
  Bluefruit.Advertising.addTxPower();
  Bluefruit.Advertising.addService(bleuart);
  Bluefruit.ScanResponse.addName();
  Bluefruit.Advertising.restartOnDisconnect(true);
  Bluefruit.Advertising.setInterval(32, 244);
  Bluefruit.Advertising.start(0);
}

// ══════════════════════════════════════════════════════════════════
// CAMBIOS RESPECTO AL CÓDIGO ANTERIOR:
// 1. Busca "E," en lugar de "EMG:" (línea 49)
// 2. substring(2) en lugar de substring(4) (línea 50)
// 3. División por 16 en lugar de 5.12 para presión (más rápido)
// 4. Debug reducido (cada 100 frames) para no saturar Serial
// ══════════════════════════════════════════════════════════════════
