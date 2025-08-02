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

  bool connected = WiFi.status() == WL_CONNECTED;

  preferences.begin("wifi", false);
  int failCount = preferences.getInt("fails", 0);

  if (!connected) {
    failCount++;
    preferences.putInt("fails", failCount);
    preferences.end();

    if (failCount >= 3) {
      Serial.println("WiFi failed 3 times. Resetting credentials and starting AP config.");
      preferences.begin("wifi", false);
      preferences.clear();  // Erase bad credentials
      preferences.end();
      startConfigPortal();
      return false;
    }
  } else {
    preferences.putInt("fails", 0);  // Reset counter on success
    preferences.end();
  }

  return connected;
}

void startConfigPortal() {
  WiFi.softAP("ESP32Cam-Setup", "12345678");
  IPAddress IP = WiFi.softAPIP();
  Serial.print("AP IP address: ");
  Serial.println(IP);

  server.on("/", HTTP_GET, [](AsyncWebServerRequest *req){
    String html = R"rawliteral(
      <html>
      <head>
        <title>WiFi Config</title>
      </head>
      <body>
        <form action='/save'>
          SSID:<input name='ssid'><br>
          Password:<input name='password' id='pass' type='password'>
          <input type='checkbox' onclick='togglePass()'> Show Password<br>
          <input type='submit'>
        </form>
        <script>
          function togglePass() {
            var x = document.getElementById('pass');
            x.type = x.type === 'password' ? 'text' : 'password';
          }
        </script>
      </body>
      </html>
    )rawliteral";
    req->send(200, "text/html", html);
  });

  server.on("/save", HTTP_GET, [](AsyncWebServerRequest *req){
    if (req->hasParam("ssid") && req->hasParam("password")) {
      preferences.begin("wifi", false);
      preferences.putString("ssid", req->getParam("ssid")->value());
      preferences.putString("pass", req->getParam("password")->value());
      preferences.putInt("fails", 0);  // Reset on new save
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
