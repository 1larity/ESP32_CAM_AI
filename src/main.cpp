    // esp32cam_main.cpp
    #include "Arduino.h"
    #include "WiFiManager.h"
    #include "CameraServer.h"
    #include "OTAHandler.h"
    #include "Utils.h"
    
    void setup() {
      Serial.begin(115200);
    
      disableBrownout();
      setupServos();
      setupCamera();
    
      if (!connectToStoredWiFi()) {
        startConfigPortal();
      }
    
      startCameraServer();
      setupOTA();
    }
    
    void loop() {
      handleOTA();
    }