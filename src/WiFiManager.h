// WiFiManager.h
#ifndef WIFI_MANAGER_H
#define WIFI_MANAGER_H

#include <ESPAsyncWebServer.h>

// Global web server used by both the WiFi config portal and the camera UI
extern AsyncWebServer server;

// Connect to WiFi using credentials saved in preferences.
bool connectToStoredWiFi();

// Start an access point and serve a configuration portal for new credentials.
void startConfigPortal();

#endif
