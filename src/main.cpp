// esp32cam_main.cpp
#include "Arduino.h"
#include "WiFiManager.h"
#include "CameraServer.h"
#include "OTAHandler.h"
#include "Utils.h"

void setup() {
  Serial.begin(115200);
  disableBrownout();

  setupServos();  // hardware-safe init

  // Start Wi-Fi
  if (!connectToStoredWiFi()) {
    startConfigPortal();  // blocks until user enters credentials
  }

  // Now that WiFi is confirmed working
  setupCamera();          // Camera init includes startStreamServer()
  startCameraServer();    // Web UI (controls only)
  setupOTA();             // OTA updater
}

void loop() {
  handleOTA();             // optional depending on OTA lib
}
