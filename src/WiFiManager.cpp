#include "WiFiManager.h"
#include <WiFi.h>
#include <Preferences.h>
#include <ESPAsyncWebServer.h>

static AsyncWebServer server(80);            // single shared Async server
static bool webStarted = false;
static Preferences preferences;

static String ssid;
static String password;

AsyncWebServer& getWebServer() {
  return server;
}

// start once, safe to call repeatedly
static void beginWebOnce() {
  if (webStarted) return;
  server.begin();
  webStarted = true;
}

// public, idempotent wrapper
void ensureWebServerStarted() {
  beginWebOnce();
}

static void buildConfigRoutes() {
  server.on("/", HTTP_GET, [](AsyncWebServerRequest* req){
    String html =
      "<html><head><meta charset='utf-8'></head><body>"
      "<h2>ESP32-CAM Wi-Fi Setup</h2>"
      "<form action='/save' method='post'>"
      "SSID: <input name='ssid'><br>"
      "Password: <input id='pw' name='password' type='password'> "
      "<button type='button' onclick=\"pw.type=pw.type==='password'?'text':'password'\">Show</button><br><br>"
      "<button type='submit'>Save & Connect</button>"
      "</form>"
      "</body></html>";
    req->send(200, "text/html", html);
  });

  server.on("/save", HTTP_POST, [](AsyncWebServerRequest* req){
    String ns, np;
    if (req->hasParam("ssid", true)) ns = req->getParam("ssid", true)->value();
    if (req->hasParam("password", true)) np = req->getParam("password", true)->value();

    preferences.begin("wifi", false);
    preferences.putString("ssid", ns);
    preferences.putString("pass", np);
    preferences.end();

    req->send(200, "text/plain", "Saved. Rebooting to connect...");
    delay(200);
    ESP.restart();
  });
}

bool connectToStoredWiFi() {
  preferences.begin("wifi", true);
  ssid     = preferences.getString("ssid", "");
  password = preferences.getString("pass", "");
  preferences.end();

  if (ssid.isEmpty()) return false;

  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);
  WiFi.begin(ssid.c_str(), password.c_str());

  Serial.printf("Connecting to %s", ssid.c_str());
  for (int i = 0; i < 30; ++i) {
    if (WiFi.status() == WL_CONNECTED) {
      Serial.printf("\nConnected: %s\n", WiFi.localIP().toString().c_str());
      // Ensure AP is off if previously on:
      WiFi.softAPdisconnect(true);
      return true;
    }
    delay(250);
    Serial.print(".");
  }
  Serial.println();
  return false;
}

void startConfigPortal() {
  // AP fallback
  WiFi.disconnect(true, true);
  WiFi.mode(WIFI_AP);
  WiFi.softAP("ESP32Cam-Setup", "setup1234", 6, false, 4);

  buildConfigRoutes();
  ensureWebServerStarted();  // start :80 once, safely

  Serial.printf("Config portal at http://%s/\n", WiFi.softAPIP().toString().c_str());

  // Block here until station connects (simple loop)
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    // optional: add timeout/auto-retry
  }
}
