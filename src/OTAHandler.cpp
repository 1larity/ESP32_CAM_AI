#include <ArduinoOTA.h>
#include <WiFi.h>
#include "StreamServer.h"          // **CHANGED** to pause stream during OTA

void setupOTA() {
  ArduinoOTA.setHostname("ESP32Cam");
  ArduinoOTA.setPassword("change_me");   // **CHANGED** basic auth

  ArduinoOTA.onStart([]() {
    String type = ArduinoOTA.getCommand() == U_FLASH ? "sketch" : "filesystem";
    Serial.println("Start updating " + type);
    stopStreamServer();                  // **CHANGED** prevent contention
  });

  ArduinoOTA.onEnd([]() {
    Serial.println("\nUpdate complete.");
    // Option 1: restart stream, or simply restart device.
    // **startStreamServer();**              // optional
  });

  ArduinoOTA.onProgress([](unsigned int progress, unsigned int total) {
    Serial.printf("Progress: %u%%\r", (progress * 100) / total);
  });

  ArduinoOTA.onError([](ota_error_t error) {
    Serial.printf("Error[%u]: ", error);
    if (error == OTA_AUTH_ERROR) Serial.println("Auth Failed");
    else if (error == OTA_BEGIN_ERROR) Serial.println("Begin Failed");
    else if (error == OTA_CONNECT_ERROR) Serial.println("Connect Failed");
    else if (error == OTA_RECEIVE_ERROR) Serial.println("Receive Failed");
    else if (error == OTA_END_ERROR) Serial.println("End Failed");
  });

  ArduinoOTA.begin();
  Serial.println("OTA Ready");
  Serial.print("IP address: ");
  Serial.println(WiFi.localIP());
}

void handleOTA() {
  ArduinoOTA.handle();
}
