#ifndef WIFI_MANAGER_H
#define WIFI_MANAGER_H

#include <ESPAsyncWebServer.h>

bool connectToStoredWiFi();
void startConfigPortal();

AsyncWebServer& getWebServer();     // expose singleton server
void ensureWebServerStarted();      // idempotent starter

#endif
