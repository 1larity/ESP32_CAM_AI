#include <ArduinoOTA.h>
#include <WiFi.h>
#include "esp_camera.h" // Needed for esp_camera_deinit()

void setupOTA() {
  // Set the device name for OTA updates
  ArduinoOTA.setHostname("ESP32Cam");

  // Called when the OTA update starts
  ArduinoOTA.onStart([]() {
    // Deinit camera right before OTA begins to avoid memory/camera conflicts
    esp_camera_deinit();
    String type = ArduinoOTA.getCommand() == U_FLASH ? "sketch" : "filesystem";
    Serial.println("Start updating " + type);
  });

  // Called when the OTA update finishes
  ArduinoOTA.onEnd([]() {
    Serial.println("\nUpdate complete.");
  });

  // Report OTA update progress
  ArduinoOTA.onProgress([](unsigned int progress, unsigned int total) {
    Serial.printf("Progress: %u%%\r", (progress * 100) / total);
  });

  // Handle OTA update errors
  ArduinoOTA.onError([](ota_error_t error) {
    Serial.printf("Error[%u]: ", error);
    if (error == OTA_AUTH_ERROR) Serial.println("Auth Failed");
    else if (error == OTA_BEGIN_ERROR) Serial.println("Begin Failed");
    else if (error == OTA_CONNECT_ERROR) Serial.println("Connect Failed");
    else if (error == OTA_RECEIVE_ERROR) Serial.println("Receive Failed");
    else if (error == OTA_END_ERROR) Serial.println("End Failed");
  });

  // Start OTA service
  ArduinoOTA.begin();
  Serial.println("OTA Ready");
  Serial.print("IP address: ");
  Serial.println(WiFi.localIP());
}

void handleOTA() {
  ArduinoOTA.handle();
}
