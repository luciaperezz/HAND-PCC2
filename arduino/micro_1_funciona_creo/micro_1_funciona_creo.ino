#include <bluefruit.h> // (no hace falta aquí, pero no rompe nada)

const int EMG_PIN = D0;

float x[3] = {0,0,0}, y[3] = {0,0,0};
float mean = 0;
float alpha = 0.01;

void setup() {
  Serial.begin(115200);
  Serial1.begin(115200);
  pinMode(EMG_PIN, INPUT);

  Serial.println("micro_1: EMG → Serial1 OK");
}

void loop() {
  float raw_emg = analogRead(EMG_PIN);

  // quitar DC
  mean = (1 - alpha)*mean + alpha*raw_emg;
  float emg_signal = raw_emg - mean;

  // filtro
  float b0 = 0.9722, b1 = -1.6180, b2 = 0.9722;
  float a1 = -1.6180, a2 = 0.9445;

  x[0] = emg_signal;
  y[0] = b0*x[0] + b1*x[1] + b2*x[2] - a1*y[1] - a2*y[2];

  x[2] = x[1]; x[1] = x[0];
  y[2] = y[1]; y[1] = y[0];

  int emg_int = (int)y[0];

  // 🔥 FORMATO CORRECTO
  Serial1.print("E,");
  Serial1.println(emg_int);

  // Debug
  Serial.print("EMG: ");
  Serial.println(emg_int);

  delay(40); // 25 Hz → perfecto
}