// WiFiManager.h
#ifndef WIFI_MANAGER_H
#define WIFI_MANAGER_H

#pragma once

// Forward declarations to avoid pulling in ESPAsyncWebServer everywhere
class AsyncWebServer;
class AsyncWebServerRequest;

// Expose the shared web server for other modules (e.g., CameraServer.cpp)
AsyncWebServer& getWebServer();

// Ensure the web server is started exactly once
void ensureWebServerStarted();

// Existing functions you already had
bool connectToStoredWiFi();
void startConfigPortal();

// Authentication helpers (Basic Auth + token)
bool isAuthEnabled();
bool isAuthorized(AsyncWebServerRequest* req);        // For AsyncWebServer (port 80)
bool isAuthorizedBasicHeader(const char* header);     // For esp_http_server (port 81)
bool isValidTokenParam(const char* token);            // For either server
#endif
