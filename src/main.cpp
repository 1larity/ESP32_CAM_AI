// esp32cam_main.cpp
#include "Arduino.h"
#include "WiFiManager.h"
#include "CameraServer.h"
#include <ArduinoOTA.h>
#include "Utils.h"
#include "PTZ.h"  

static bool g_inConfigPortal = false;
static bool g_otaReady = false;

void setup() {
  setCpuFrequencyMhz(240);
  psramInit();
  Serial.begin(115200);
  delay(2000);
  Serial.println("\nBooting ESP32_CAM_AI...");
  disableBrownout();
  Serial.println("Brownout disabled");

  Serial.println("Connecting to stored WiFi...");
  if (!connectToStoredWiFi()) {
    g_inConfigPortal = true;
    Serial.println("No stored WiFi, starting config portal");
    startConfigPortal();
    Serial.println("Config portal started");
    return;
  }
  Serial.println("WiFi connected");

  Serial.println("Init camera...");
  setupCamera();
  Serial.println("Init PTZ...");
  ptzInit();
  Serial.println("Start camera server...");
  startCameraServer();

  Serial.println("Init OTA...");
  ArduinoOTA.begin();
  g_otaReady = true;
  Serial.println("Setup complete");
}

void loop() {
  if (!g_inConfigPortal && g_otaReady) {
      ArduinoOTA.handle();;
  }
}
