#ifndef WIFI_MANAGER_H
#define WIFI_MANAGER_H

// Connect to WiFi using credentials saved in preferences.
bool connectToStoredWiFi();

// Start an access point and serve a configuration portal for new credentials.
void startConfigPortal();

#endif
