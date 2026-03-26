#include <bluefruit.h>
///////////////////////////////////////// MICRO_1 : EMG + ENVIO A MICRO_2
// micro_1: EMG → Serial1 a micro_2
const int EMG_PIN = D0;
float x[3] = {0,0,0}, y[3] = {0,0,0}, mean = 0, alpha = 0.01;

void setup() {
  Serial.begin(115200);
  Serial1.begin(115200);  // → micro_2
  pinMode(EMG_PIN, INPUT);
  Serial.println("micro_1: EMG → Serial1 OK");
}

void loop() {
  // FILTRO EMG (tu código original)
  float raw_emg = analogRead(EMG_PIN);
  mean = (1 - alpha)*mean + alpha*raw_emg;
  float emg_signal = raw_emg - mean;
  
  x[0] = emg_signal;
  float b0 = 0.9722, b1 = -1.6180, b2 = 0.9722;
  float a1 = -1.6180, a2 = 0.9445;
  y[0] = b0*x[0] + b1*x[1] + b2*x[2] - a1*y[1] - a2*y[2];
  x[2] = x[1]; x[1] = x[0];
  y[2] = y[1]; y[1] = y[0];
  
  // ENVÍA EMG a micro_2
  Serial1.print("EMG:");
  Serial1.println(y[0], 0);
  
  Serial.print("EMG:"); Serial.println(y[0], 0);  // Debug
  
  delay(40);
}
