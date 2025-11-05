// esp32cam_main.cpp
#include "Arduino.h"
#include "WiFiManager.h"
#include "CameraServer.h"
#include "OTAHandler.h"
#include "Utils.h"
#include "PTZ.h"  

void setup() {
  Serial.begin(115200);
  disableBrownout();

  // Removed early setupServos() call.
  // Servos are initialised inside setupCamera() after camera init.
  // setupServos();

  // Start Wi-Fi; if it fails, launch configuration portal and skip rest
  if (!connectToStoredWiFi()) {
    startConfigPortal();
    return;  // remain in portal mode until restart
  }

  // Now that WiFi is confirmed working
  setupCamera();          // Camera init includes startStreamServer() and setupServos()
  ptzInit(); // Init PTZ and register its routes on the shared server
  startCameraServer();    // Web UI (controls only)

  setupOTA();             // OTA updater
}

void loop() {
  handleOTA();             // optional depending on OTA lib
}
