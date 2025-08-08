// esp32cam_main.cpp
#include "Arduino.h"
#include <WiFi.h>           // needed for WiFi.localIP() print
#include "WiFiManager.h"
#include "CameraServer.h"
#include "OTAHandler.h"
#include "Utils.h"

void setup() {
  Serial.begin(115200);
  delay(200);                      // allow Serial to settle
  disableBrownout();

  // Wi‑Fi first
  if (!connectToStoredWiFi()) {
    startConfigPortal();           // blocks until user enters credentials
  }

  // Register camera UI routes onto the shared Async server
  startCameraServer();

  // Start the :80 server once, safely (works for both STA/AP paths)
  ensureWebServerStarted();

  // Start camera + MJPEG stream on :81
  setupCamera();

  // Announce where to browse
  Serial.printf("Web UI:    http://%s/  (try /ping)\n", WiFi.localIP().toString().c_str());
  Serial.printf("MJPEG stream: http://%s:81/stream\n", WiFi.localIP().toString().c_str());

  // OTA last
  setupOTA();
}

void loop() {
  handleOTA();
  delay(2);
}
