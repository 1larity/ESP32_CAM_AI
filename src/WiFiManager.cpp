#include "WiFiManager.h"
#include <WiFi.h>
#include <Preferences.h>
#include <ESPAsyncWebServer.h>

Preferences preferences;
AsyncWebServer server(80);
String ssid, password;

bool connectToStoredWiFi() {
  preferences.begin("wifi", true);
  ssid = preferences.getString("ssid", "");
  password = preferences.getString("pass", "");
  preferences.end();

  if (ssid == "") return false;

  WiFi.begin(ssid.c_str(), password.c_str());
  Serial.printf("Connecting to %s", ssid.c_str());

  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 20) {
    delay(500);
    Serial.print(".");
    attempts++;
  }
  Serial.println();
  return WiFi.status() == WL_CONNECTED;
}

void startConfigPortal() {
  WiFi.softAP("ESP32Cam-Setup", "12345678");
  IPAddress IP = WiFi.softAPIP();
  Serial.print("AP IP address: ");
  Serial.println(IP);

  server.on("/", HTTP_GET, [](AsyncWebServerRequest *req){
    String html = "<form action='/save'>SSID:<input name='ssid'><br>Password:<input name='password' type='password'><br><input type='submit'></form>";
    req->send(200, "text/html", html);
  });

  server.on("/save", HTTP_GET, [](AsyncWebServerRequest *req){
    if (req->hasParam("ssid") && req->hasParam("password")) {
      preferences.begin("wifi", false);
      preferences.putString("ssid", req->getParam("ssid")->value());
      preferences.putString("pass", req->getParam("password")->value());
      preferences.end();
      req->send(200, "text/html", "Saved. Rebooting...");
      delay(2000);
      ESP.restart();
    } else {
      req->send(400, "text/plain", "Missing parameters");
    }
  });

  server.begin();
}
