## Prompt for Copilot
```
This goes with the python app
```
# Packaged Project (C/C++)
- **Project root**: C:\Users\stellaris\Documents\PlatformIO\Projects\ESP32_CAM_AI\src
- **Generated**: 2025-11-04 16:28:36 +0000
- **Tool**: Copilot Python Packager v1.6
## Table of contents
- `CameraServer.cpp`
- `CameraServer.h`
- `main.cpp`
- `OTAHandler.cpp`
- `OTAHandler.h`
- `StreamServer.cpp`
- `StreamServer.h`
- `Utils.cpp`
- `Utils.h`
- `WiFiManager.cpp`
- `WiFiManager.h`
---
## `CameraServer.cpp`
**Absolute path**: C:\Users\stellaris\Documents\PlatformIO\Projects\ESP32_CAM_AI\src\CameraServer.cpp
**Size**: 12030 bytes
**Modified**: 2025-10-23 23:09:24 +0100
**SHA256**: 7c190707f23dcb9a45e3bc0d1e3869f871ccf2a3b418495a6018d551c0ade118
``````cpp#include "WiFiManager.h"
#include "CameraServer.h"
#include "OTAHandler.h"
#include "Utils.h"

#include "esp_camera.h"
#include <WiFi.h>
#include <ESPAsyncWebServer.h>
#include <Preferences.h>
#include <vector>
#include <memory>

#include "StreamServer.h"

// ===== PTZ pins =====
#define SERVO_1 14
#define SERVO_2 15

// ===== PTZ state (servo objects are defined in Utils.cpp) =====
// servo1 and servo2 are declared extern in Utils.h and defined in Utils.cpp
static int servo1Pos = 90;
static int servo2Pos = 90;

// ===== Resolution options with actual pixel sizes =====
struct ResolutionOption {
  framesize_t size;
  const char* name;
  uint16_t    w;
  uint16_t    h;
};

// From highest to lowest (safe set for OV2640)
static const std::vector<ResolutionOption> resolutionOptions = {
  {FRAMESIZE_UXGA,  "UXGA",   1600,1200},
  {FRAMESIZE_SXGA,  "SXGA",   1280,1024},
  {FRAMESIZE_XGA,   "XGA",    1024,768},
  {FRAMESIZE_SVGA,  "SVGA",   800, 600},
  {FRAMESIZE_VGA,   "VGA",    640, 480},
  {FRAMESIZE_CIF,   "CIF",    352, 288},
  {FRAMESIZE_QVGA,  "QVGA",   320, 240},
  {FRAMESIZE_HQVGA, "HQVGA",  240, 176},
  {FRAMESIZE_QQVGA, "QQVGA",  160, 120},
};

static inline int clamp180(int v) {
  if (v < 0) return 0;
  if (v > 180) return 180;
  return v;
}

// ===== Persistence =====
static Preferences camPrefs;
static const char* CAM_NS  = "camera";
static const char* KEY_RES = "res";  // framesize_t stored as uint8_t

static const ResolutionOption* findOptionByName(const String& n) {
  for (const auto& o : resolutionOptions) {
    if (n.equalsIgnoreCase(o.name)) return &o;
  }
  return nullptr;
}

static const ResolutionOption* findOptionBySize(framesize_t fs) {
  for (const auto& o : resolutionOptions) if (o.size == fs) return &o;
  return nullptr;
}

// ===== Simple pale-blue UI theme (CSS only; no stream changes) =====
static const char* kStyle = R"CSS(
  <style>
    :root{
      --bg:#eaf4ff; --panel:#ffffff; --text:#102a43; --muted:#486581; --accent:#2b6cb0;
      --line:#cfe3ff; --shadow:0 1px 4px rgba(16,42,67,.08);
    }
    *{box-sizing:border-box}
    html,body{margin:0;padding:0;background:var(--bg);color:var(--text);
      font-family:system-ui,-apple-system,"Segoe UI",Roboto,Ubuntu,"Helvetica Neue",Arial,sans-serif}
    .wrap{max-width:980px;margin:20px auto;padding:0 12px}
    .nav{display:flex;gap:8px;justify-content:flex-end;margin:6px 0 16px 0}
    .btn{appearance:none;border:1px solid var(--line);padding:8px 12px;border-radius:8px;
      background:#d9ebff;cursor:pointer;font-weight:600;box-shadow:var(--shadow);text-decoration:none;color:inherit}
    .btn:hover{background:#cfe3ff}
    .panel{background:var(--panel);border:1px solid var(--line);border-radius:12px;
      box-shadow:var(--shadow);padding:14px 14px 16px 14px;margin:14px 0}
    .panel h3{margin:0 0 10px 0;color:var(--accent)}
    .row{margin:8px 0}
    select,button{font-size:1rem;padding:6px 10px;border-radius:8px;border:1px solid var(--line)}
    img.stream{max-width:100%;height:auto;border:1px solid var(--line);border-radius:10px}
    pre{background:#f7fbff;border:1px solid var(--line);border-radius:8px;padding:8px;display:block}
    label{color:var(--muted);margin-right:6px}
  </style>
)CSS";

// ===== Web UI & API =====
static void renderCameraPage(AsyncWebServerRequest* request) {
  sensor_t* s = esp_camera_sensor_get();
  framesize_t cur = s ? s->status.framesize : FRAMESIZE_VGA;

  String html;
  html.reserve(7000);
  html += "<!doctype html><html><head><meta charset='utf-8'><title>ESP32-CAM</title>";
  html += kStyle;
  html += "</head><body><div class='wrap'>";

  // Small nav: Wi-Fi Settings
  html += "<div class='nav'>"
          "<a class='btn' href='/wifi'>Wi-Fi Settings</a>"
          "</div>";

  // PANEL: Controls
  html += "<div class='panel'><h3>Camera Controls</h3>";
  html += "<div class='row'><label for='res'>Resolution</label>"
          "<select id='res' onchange=\"fetch('/resolution?set='+this.value).then(()=>location.reload())\">";
  for (const auto& o : resolutionOptions) {
    bool sel = (o.size == cur);
    html += "<option value='" + String(o.name) + "'" + (sel ? " selected" : "") + ">"
            + String(o.name) + " (" + String(o.w) + "&times;" + String(o.h) + ")</option>";
  }
  html += "</select></div>";

  html += "<div class='row'>"
          "<button class='btn' onclick=\"fetch('/action?go=up')\">Up</button> "
          "<button class='btn' onclick=\"fetch('/action?go=down')\">Down</button> "
          "<button class='btn' onclick=\"fetch('/action?go=left')\">Left</button> "
          "<button class='btn' onclick=\"fetch('/action?go=right')\">Right</button>"
          "</div>";
  html += "</div>"; // panel

  // PANEL: Live Video
  html += "<div class='panel'><h3>Live Video</h3>";
  {
    // Use token when auth is enabled so the <img> can load cross-port
    String streamURL = String("http://") + WiFi.localIP().toString() + ":81/stream";
    if (isAuthEnabled()) {
      String tok = getAuthTokenParam();
      if (tok.length() > 0) streamURL += "?token=" + tok;
    }
    html += "<img class='stream' src='" + streamURL + "' alt='Video stream'>";
  }
  html += "</div>";

  // PANEL: Status
  html += "<div class='panel'><h3>Status</h3><pre id='st'>Loading…</pre></div>"
          "<script>fetch('/api/status').then(r=>r.json()).then(j=>{"
          "document.getElementById('st').textContent=JSON.stringify(j,null,2);});</script>";

  html += "</div></body></html>";
  request->send(200, "text/html", html);
}

void startCameraServer() {
  // Reuse singleton AsyncWebServer from WiFiManager (port 80)
  AsyncWebServer& camServer = getWebServer();

  // Main page: render skinned UI
  camServer.on("/", HTTP_GET, [](AsyncWebServerRequest *request){
    if (!isAuthorized(request)) {
      auto* r = request->beginResponse(401, "text/plain", "Unauthorized");
      r->addHeader("WWW-Authenticate", "Basic realm=\"ESP32Cam\"");
      request->send(r);
      return;
    }
    renderCameraPage(request);
  });

  // Alias so AP page can link back as "/cam" later without touching other files
  camServer.on("/cam", HTTP_GET, [](AsyncWebServerRequest *request){
    if (!isAuthorized(request)) {
      auto* r = request->beginResponse(401, "text/plain", "Unauthorized");
      r->addHeader("WWW-Authenticate", "Basic realm=\"ESP32Cam\"");
      request->send(r);
      return;
    }
    renderCameraPage(request);
  });

  // Change resolution + persist
  camServer.on("/resolution", HTTP_GET, [](AsyncWebServerRequest *request){
    if (!isAuthorized(request)) {
      auto* r = request->beginResponse(401, "text/plain", "Unauthorized");
      r->addHeader("WWW-Authenticate", "Basic realm=\"ESP32Cam\"");
      request->send(r);
      return;
    }
    if (!request->hasParam("set")) {
      request->send(400, "text/plain", "Missing ?set=");
      return;
    }

    const String target = request->getParam("set")->value();
    const auto* opt = findOptionByName(target);
    if (!opt) { request->send(400, "text/plain", "Invalid resolution"); return; }

    sensor_t* s = esp_camera_sensor_get();
    if (!s) { request->send(500, "text/plain", "Sensor unavailable"); return; }

    s->set_framesize(s, opt->size);

    camPrefs.begin(CAM_NS, false);
    camPrefs.putUChar(KEY_RES, static_cast<uint8_t>(opt->size));
    camPrefs.end();

    request->send(200, "text/plain", "OK");
  });

  // PTZ actions
  camServer.on("/action", HTTP_GET, [](AsyncWebServerRequest *request){
    if (!isAuthorized(request)) {
      auto* r = request->beginResponse(401, "text/plain", "Unauthorized");
      r->addHeader("WWW-Authenticate", "Basic realm=\"ESP32Cam\"");
      request->send(r);
      return;
    }
    if (request->hasParam("go")) {
      String dir = request->getParam("go")->value();
      if (dir == "up")          servo1Pos = clamp180(servo1Pos + 10);
      else if (dir == "down")   servo1Pos = clamp180(servo1Pos - 10);
      else if (dir == "left")   servo2Pos = clamp180(servo2Pos + 10);
      else if (dir == "right")  servo2Pos = clamp180(servo2Pos - 10);
      servo1.write(servo1Pos);
      servo2.write(servo2Pos);
    }
    request->send(200, "text/plain", "OK");
  });

  // JSON status (handy for your PC viewer)
  camServer.on("/api/status", HTTP_GET, [](AsyncWebServerRequest* req){
    if (!isAuthorized(req)) {
      auto* r = req->beginResponse(401, "text/plain", "Unauthorized");
      r->addHeader("WWW-Authenticate", "Basic realm=\"ESP32Cam\"");
      req->send(r);
      return;
    }
    sensor_t* s = esp_camera_sensor_get();
    framesize_t fs = s ? s->status.framesize : FRAMESIZE_INVALID;
    const auto* cur = findOptionBySize(fs);

    String json = "{";
    json += "\"ip\":\"" + WiFi.localIP().toString() + "\"";
    if (cur) {
      json += ",\"resolution\":\"" + String(cur->name) + "\"";
      json += ",\"width\":" + String(cur->w) + ",\"height\":" + String(cur->h);
    }
    json += "}";
    req->send(200, "application/json", json);
  });

  // Discovery endpoint (no auth) for local scans
  camServer.on("/api/advertise", HTTP_GET, [](AsyncWebServerRequest* req){
    // No auth; minimal info only
    String ip = WiFi.localIP().toString();
    String json = "{";
    json += "\"name\":\"ESP32-CAM\",";
    json += "\"ip\":\"" + ip + "\",";
    json += "\"stream\":\"http://" + ip + ":81/stream\"";
    json += "}";
    auto* r = req->beginResponse(200, "application/json", json);
    r->addHeader("Access-Control-Allow-Origin", "*");
    req->send(r);
  });

  // Ensure the shared web server is listening (STA mode path)
  ensureWebServerStarted();
}

// ===== Camera init (unchanged) =====
void setupCamera() {
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer   = LEDC_TIMER_0;
  config.pin_d0       = 5;
  config.pin_d1       = 18;
  config.pin_d2       = 19;
  config.pin_d3       = 21;
  config.pin_d4       = 36;
  config.pin_d5       = 39;
  config.pin_d6       = 34;
  config.pin_d7       = 35;
  config.pin_xclk     = 0;
  config.pin_pclk     = 22;
  config.pin_vsync    = 25;
  config.pin_href     = 23;
  config.pin_sccb_sda = 26;
  config.pin_sccb_scl = 27;
  config.pin_pwdn     = 32;
  config.pin_reset    = -1;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;

  bool psram = psramFound();
  bool found = false;

  // Try from high to low until one initialises
  for (const auto& option : resolutionOptions) {
    config.frame_size   = option.size;
    config.jpeg_quality = psram ? 10 : 12;
    config.fb_count     = psram ? 2 : 1;

    // ensure clean state between attempts
    esp_camera_deinit();
    delay(50);

    esp_err_t err = esp_camera_init(&config);
    if (err == ESP_OK) {
      Serial.printf("Camera initialized with resolution: %s\n", option.name);
      found = true;
      break;
    } else {
      Serial.printf("Failed to init with %s (%d), trying lower...\n", option.name, err);
    }
  }

  if (!found) {
    Serial.println("Camera init failed for all resolutions.");
    return;
  }

  // Apply saved framesize (if any) AFTER successful init
  camPrefs.begin(CAM_NS, true);
  uint8_t saved = camPrefs.getUChar(KEY_RES, 0xFF);
  camPrefs.end();
  if (saved != 0xFF) {
    sensor_t* s = esp_camera_sensor_get();
    if (s) {
      s->set_framesize(s, static_cast<framesize_t>(saved));
      Serial.printf("Applied saved framesize id: %u\n", saved);
    }
  }

  // PTZ init
  setupServos();                 // sets 50Hz on both (from Utils.cpp)
  servo1.attach(SERVO_1, 1000, 2000);
  servo2.attach(SERVO_2, 1000, 2000);
  servo1.write(servo1Pos);
  servo2.write(servo2Pos);

  // Start MJPEG stream on :81 (unchanged)
  startStreamServer();
}
## `CameraServer.h`
**Absolute path**: C:\Users\stellaris\Documents\PlatformIO\Projects\ESP32_CAM_AI\src\CameraServer.h
**Size**: 245 bytes
**Modified**: 2025-08-09 00:53:38 +0100
**SHA256**: 848b618bb54a0f163c34e84f7b3bfa22c6202a06301bffc4280278cda472eba9
``````c#ifndef CAMERA_SERVER_H
#define CAMERA_SERVER_H

// Configures camera hardware, PSRAM settings, and servo motors
void setupCamera();

// Launches the web interface for streaming and controlling servos
void startCameraServer();

#endif
## `main.cpp`
**Absolute path**: C:\Users\stellaris\Documents\PlatformIO\Projects\ESP32_CAM_AI\src\main.cpp
**Size**: 843 bytes
**Modified**: 2025-10-23 23:10:53 +0100
**SHA256**: 621e05aa722610a998f62119c958d204c440bf1740a53b97351b7d410f92dd86
``````cpp// esp32cam_main.cpp
#include "Arduino.h"
#include "WiFiManager.h"
#include "CameraServer.h"
#include "OTAHandler.h"
#include "Utils.h"

void setup() {
  Serial.begin(115200);
  disableBrownout();

  // Removed early setupServos() call.
  // Servos are initialised inside setupCamera() after camera init.
  // setupServos();

  // Start Wi-Fi; if it fails, launch configuration portal and skip rest
  if (!connectToStoredWiFi()) {
    startConfigPortal();
    return;  // remain in portal mode until restart
  }

  // Now that WiFi is confirmed working
  setupCamera();          // Camera init includes startStreamServer() and setupServos()
  startCameraServer();    // Web UI (controls only)
  setupOTA();             // OTA updater
}

void loop() {
  handleOTA();             // optional depending on OTA lib
}
## `OTAHandler.cpp`
**Absolute path**: C:\Users\stellaris\Documents\PlatformIO\Projects\ESP32_CAM_AI\src\OTAHandler.cpp
**Size**: 1489 bytes
**Modified**: 2025-08-27 23:38:07 +0100
**SHA256**: fa0dfdf915e44d1ea9ea520ccddd3f59687483ebacfbbd111422951cc38c0bb4
``````cpp#include <ArduinoOTA.h>
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
## `OTAHandler.h`
**Absolute path**: C:\Users\stellaris\Documents\PlatformIO\Projects\ESP32_CAM_AI\src\OTAHandler.h
**Size**: 160 bytes
**Modified**: 2025-08-09 00:53:38 +0100
**SHA256**: a435ddaa5d35f270f01bb25cf98e994ff8cca0db5fd6eee4e6749831a9186cb8
``````c#ifndef OTA_HANDLER_H
#define OTA_HANDLER_H

// Initialize OTA functionality
void setupOTA();

// Process OTA update events
void handleOTA();

#endif
## `StreamServer.cpp`
**Absolute path**: C:\Users\stellaris\Documents\PlatformIO\Projects\ESP32_CAM_AI\src\StreamServer.cpp
**Size**: 6354 bytes
**Modified**: 2025-08-29 18:39:14 +0100
**SHA256**: d45be7e3e3dc0a413382ca5855b9bd5bd1270832b27f6daac57ad135e25c2bcd
``````cpp#include "StreamServer.h"
#include "esp_camera.h"
#include "esp_http_server.h"
#include "WiFiManager.h"

httpd_handle_t stream_httpd = NULL;

// MIME type for MJPEG stream with frame boundary marker
static const char* _STREAM_CONTENT_TYPE = "multipart/x-mixed-replace; boundary=frame";
// Delimiter between JPEG frames in the multipart response
static const char* _STREAM_BOUNDARY = "\r\n--frame\r\n";
// Template for each part's headers indicating JPEG data and its length
static const char* _STREAM_PART = "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n";

static esp_err_t stream_handler(httpd_req_t *req) {
  camera_fb_t *fb = NULL;
  esp_err_t res = ESP_OK;
  char part_buf[64];

  // Authorization: allow Basic header or ?token= param
  if (isAuthEnabled()) {
    bool ok = false;
    // Check token in query string
    int ql = httpd_req_get_url_query_len(req);
    if (ql > 0) {
      char* q = (char*)malloc(ql+1);
      if (!q) return ESP_ERR_NO_MEM;
      if (httpd_req_get_url_query_str(req, q, ql+1) == ESP_OK) {
        char tbuf[128];
        if (httpd_query_key_value(q, "token", tbuf, sizeof(tbuf)) == ESP_OK) {
          ok = isValidTokenParam(tbuf);
        }
      }
      free(q);
    }
    // Check Basic header if not ok yet
    if (!ok) {
      size_t alen = httpd_req_get_hdr_value_len(req, "Authorization");
      if (alen > 0) {
        char* abuf = (char*)malloc(alen+1);
        if (!abuf) return ESP_ERR_NO_MEM;
        if (httpd_req_get_hdr_value_str(req, "Authorization", abuf, alen+1) == ESP_OK) {
          ok = isAuthorizedBasicHeader(abuf);
        }
        free(abuf);
      }
    }
    if (!ok) {
      httpd_resp_set_status(req, "401 Unauthorized");
      httpd_resp_set_hdr(req, "WWW-Authenticate", "Basic realm=\"ESP32Cam\"");
      httpd_resp_set_type(req, "text/plain");
      httpd_resp_send(req, "Unauthorized", HTTPD_RESP_USE_STRLEN);
      return ESP_OK;
    }
  }

  // Tell the client to expect multipart MJPEG stream + friendly headers
  httpd_resp_set_hdr(req, "Cache-Control", "no-store, no-cache, must-revalidate, max-age=0");
  httpd_resp_set_hdr(req, "Pragma", "no-cache");
  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
  res = httpd_resp_set_type(req, _STREAM_CONTENT_TYPE);
  if (res != ESP_OK) return res;

  while (true) {
    // Grab a frame from the camera
    fb = esp_camera_fb_get();
    if (!fb) {
      res = ESP_FAIL;
      break;
    }

    // Prepare JPEG buffer
    const uint8_t* buf = nullptr;
    size_t len = 0;
    uint8_t* jpg_buf = nullptr;
    size_t jpg_len = 0;

    if (fb->format == PIXFORMAT_JPEG) {
      buf = fb->buf;
      len = fb->len;
    } else {
      // Convert frame buffer to JPEG if not already
      bool ok = frame2jpg(fb, 80, &jpg_buf, &jpg_len);
      if (!ok) {
        esp_camera_fb_return(fb);
        res = ESP_FAIL;
        break;
      }
      buf = jpg_buf;
      len = jpg_len;
    }

    // Send the JPEG as a multipart HTTP response:
    // first the part header, then the image data, then the boundary marker
    size_t hlen = snprintf(part_buf, sizeof(part_buf), _STREAM_PART, (unsigned)len);
    res = httpd_resp_send_chunk(req, part_buf, hlen);
    if (res == ESP_OK)
      res = httpd_resp_send_chunk(req, (const char*)buf, len);
    if (res == ESP_OK)
      res = httpd_resp_send_chunk(req, _STREAM_BOUNDARY, strlen(_STREAM_BOUNDARY));

    // Return/free buffers
    if (fb) {
      esp_camera_fb_return(fb);
      fb = NULL;
    }
    if (jpg_buf) {
      free(jpg_buf);
      jpg_buf = NULL;
    }

    if (res != ESP_OK) break;
  }

  return res;
}

// Snapshot endpoint for quick diagnostics
static esp_err_t snapshot_handler(httpd_req_t *req) {
  if (isAuthEnabled()) {
    bool ok = false;
    int ql = httpd_req_get_url_query_len(req);
    if (ql > 0) {
      char* q = (char*)malloc(ql+1);
      if (!q) return ESP_ERR_NO_MEM;
      if (httpd_req_get_url_query_str(req, q, ql+1) == ESP_OK) {
        char tbuf[128];
        if (httpd_query_key_value(q, "token", tbuf, sizeof(tbuf)) == ESP_OK) {
          ok = isValidTokenParam(tbuf);
        }
      }
      free(q);
    }
    if (!ok) {
      size_t alen = httpd_req_get_hdr_value_len(req, "Authorization");
      if (alen > 0) {
        char* abuf = (char*)malloc(alen+1);
        if (!abuf) return ESP_ERR_NO_MEM;
        if (httpd_req_get_hdr_value_str(req, "Authorization", abuf, alen+1) == ESP_OK) {
          ok = isAuthorizedBasicHeader(abuf);
        }
        free(abuf);
      }
    }
    if (!ok) {
      httpd_resp_set_status(req, "401 Unauthorized");
      httpd_resp_set_hdr(req, "WWW-Authenticate", "Basic realm=\"ESP32Cam\"");
      httpd_resp_set_type(req, "text/plain");
      httpd_resp_send(req, "Unauthorized", HTTPD_RESP_USE_STRLEN);
      return ESP_OK;
    }
  }
  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) {
    httpd_resp_set_status(req, "503 Service Unavailable");
    httpd_resp_set_type(req, "text/plain");
    httpd_resp_send(req, "Camera unavailable", HTTPD_RESP_USE_STRLEN);
    return ESP_OK;
  }
  const uint8_t* img = nullptr; size_t len = 0; uint8_t* jpg_buf = nullptr; size_t jpg_len = 0;
  if (fb->format == PIXFORMAT_JPEG) { img = fb->buf; len = fb->len; }
  else {
    if (!frame2jpg(fb, 80, &jpg_buf, &jpg_len)) { esp_camera_fb_return(fb); return ESP_FAIL; }
    img = jpg_buf; len = jpg_len;
  }
  httpd_resp_set_type(req, "image/jpeg");
  httpd_resp_send(req, (const char*)img, len);
  if (jpg_buf) free(jpg_buf);
  esp_camera_fb_return(fb);
  return ESP_OK;
}

/**
 * Start HTTP server dedicated to MJPEG streaming.
 * Uses port 81 and registers stream_handler for the "/stream" URI.
 */
void startStreamServer() {
  httpd_config_t config = HTTPD_DEFAULT_CONFIG();
  config.server_port = 81; // Use a port different from the main web server

  httpd_uri_t stream_uri = {
    .uri       = "/stream",
    .method    = HTTP_GET,
    .handler   = stream_handler,
    .user_ctx  = NULL
  };

  if (httpd_start(&stream_httpd, &config) == ESP_OK) {
    // Register handler that serves the stream when /stream is requested
    httpd_register_uri_handler(stream_httpd, &stream_uri);
    // Also provide /snap for quick diagnostics
    httpd_uri_t snap_uri = { .uri = "/snap", .method = HTTP_GET, .handler = snapshot_handler, .user_ctx = NULL };
    httpd_register_uri_handler(stream_httpd, &snap_uri);
  }
}
## `StreamServer.h`
**Absolute path**: C:\Users\stellaris\Documents\PlatformIO\Projects\ESP32_CAM_AI\src\StreamServer.h
**Size**: 41 bytes
**Modified**: 2025-08-09 00:53:38 +0100
**SHA256**: fff3c163400dd12f9a360984776df7675f26958cfaf5dd7be0a7f7f1363d2afd
``````c#pragma once
void startStreamServer();
## `Utils.cpp`
**Absolute path**: C:\Users\stellaris\Documents\PlatformIO\Projects\ESP32_CAM_AI\src\Utils.cpp
**Size**: 813 bytes
**Modified**: 2025-10-23 23:08:14 +0100
**SHA256**: 65f873eae658ea636f6678cc962144109e40e36154c38e692c23c1e7580bebb0
``````cpp#include "Utils.h"
#include "soc/soc.h"
#include "soc/rtc_cntl_reg.h"

// Define servo globals here to ensure they are constructed
// before any call to setupServos() from other translation units.
Servo servo1;
Servo servo2;

void disableBrownout() {
  // The ESP32-CAM is sensitive to brief voltage drops when peripherals
  // such as the camera or servos draw peak current. Those dips can
  // trigger the on-chip brownout detector and cause an unexpected
  // reset, so the detector is disabled to keep the device running.
  WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0);
}

void setupServos() {
  // Standard hobby servos expect a 20ms refresh period (50 Hz). Using
  // this rate provides full range of motion and avoids jitter.
  servo1.setPeriodHertz(50);
  servo2.setPeriodHertz(50);
}
## `Utils.h`
**Absolute path**: C:\Users\stellaris\Documents\PlatformIO\Projects\ESP32_CAM_AI\src\Utils.h
**Size**: 558 bytes
**Modified**: 2025-10-23 23:06:30 +0100
**SHA256**: b4750c8af9a905afbfa58f6d4bfe91ae1b806260624b1f086df793ab8e0724f3
``````c#ifndef UTILS_H
#define UTILS_H

#include <ESP32Servo.h>

/**
 * Globals for servo control. Defined in Utils.cpp to guarantee
 * construction order before any call to setupServos().
 */
extern Servo servo1;
extern Servo servo2;

/**
 * Disable the ESP32's brownout detector to prevent unwanted resets
 * during brief voltage dips caused by high current draw.
 */
void disableBrownout();

/**
 * Configure the servos to run at a 50 Hz PWM frequency, the standard
 * refresh rate for most hobby servos.
 */
void setupServos();

#endif
## `WiFiManager.cpp`
**Absolute path**: C:\Users\stellaris\Documents\PlatformIO\Projects\ESP32_CAM_AI\src\WiFiManager.cpp
**Size**: 30662 bytes
**Modified**: 2025-10-23 23:09:39 +0100
**SHA256**: faa94292f893c2507819f0e9fc29604751b038719bfa5bba29d5a6793b4948e6
``````cpp#include "WiFiManager.h"
#include <WiFi.h>
#include <Preferences.h>
#include <ESPAsyncWebServer.h>
#include <mbedtls/sha256.h>
#include <mbedtls/base64.h>
#include <esp_system.h>

// Define the global web server here (single definition for the whole program)
static bool g_serverStarted = false;
static AsyncWebServer g_server(80);
static bool g_wifiRoutesAdded = false;
static bool g_authRoutesAdded = false;

// New auth (hashed + salted)
static bool    g_authEnabled = false;
static String  g_authUser;
static uint8_t g_authSalt[16];
static uint8_t g_authHash[32];
static String  g_authToken; // Base64(user:pass) saved at set time for convenience

// Stored creds
static Preferences preferences;
static String ssid, password;
// Optional static IP configuration (stored in the same prefs namespace "wifi")
static bool useStaticIP = false;
static String ipStr, gwStr, snStr, dnsStr;

// Attempts allowed before AP fallback
static const int MAX_CONNECT_ATTEMPTS = 20;
static const int MRU_MAX = 5; // keep top-5 most recently used networks

// ===== MRU helpers (top-5 networks) =====
static void loadMRU(String ssids[], String passes[], int &count) {
  count = 0;
  preferences.begin("wifi", true);
  // Read MRU slots ssid0..ssid4 / pass0..pass4
  for (int i = 0; i < MRU_MAX; ++i) {
    String skey = String("ssid") + i;
    String pkey = String("pass") + i;
    String s = preferences.getString(skey.c_str(), String());
    String p = preferences.getString(pkey.c_str(), String());
    if (s.length() > 0) {
      ssids[count] = s;
      passes[count] = p;
      ++count;
    }
  }
  // Backward compatibility: if no MRU stored, fall back to legacy keys
  if (count == 0) {
    String s = preferences.getString("ssid", "");
    String p = preferences.getString("pass", "");
    if (s.length() > 0) {
      ssids[0] = s; passes[0] = p; count = 1;
    }
  }
  preferences.end();
}

static void saveMRUList(const String ssids[], const String passes[], int count) {
  preferences.begin("wifi", false);
  // Persist MRU slots
  for (int i = 0; i < MRU_MAX; ++i) {
    if (i < count) {
      String skey = String("ssid") + i;
      String pkey = String("pass") + i;
      preferences.putString(skey.c_str(), ssids[i]);
      preferences.putString(pkey.c_str(), passes[i]);
    } else {
      String skey = String("ssid") + i;
      String pkey = String("pass") + i;
      preferences.putString(skey.c_str(), String());
      preferences.putString(pkey.c_str(), String());
    }
  }
  // Update legacy keys to the most-recent one for UI/back-compat
  preferences.putString("ssid", count > 0 ? ssids[0] : "");
  preferences.putString("pass", count > 0 ? passes[0] : "");
  preferences.end();
}

static void mruMoveToFront(String ssids[], String passes[], int &count, int idx) {
  if (idx <= 0 || idx >= count) return;
  String s = ssids[idx];
  String p = passes[idx];
  for (int i = idx; i > 0; --i) {
    ssids[i] = ssids[i-1];
    passes[i] = passes[i-1];
  }
  ssids[0] = s; passes[0] = p;
}

static void mruInsertFrontUnique(String ssids[], String passes[], int &count, const String& s, const String& p) {
  if (s.length() == 0) return;
  // Find existing
  int found = -1;
  for (int i = 0; i < count; ++i) { if (ssids[i] == s) { found = i; break; } }
  if (found >= 0) {
    // Update pass and move to front
    passes[found] = p;
    mruMoveToFront(ssids, passes, count, found);
    return;
  }
  // Shift down (cap at MRU_MAX-1)
  int newCount = count < MRU_MAX ? count + 1 : MRU_MAX;
  for (int i = newCount - 1; i > 0; --i) {
    ssids[i] = ssids[i-1];
    passes[i] = passes[i-1];
  }
  ssids[0] = s; passes[0] = p; count = newCount;
}

// ===== Shared pale-blue theme =====
static const char* kStyle = R"CSS(
  <style>
    :root{
      --bg:#eaf4ff; --panel:#ffffff; --text:#102a43; --muted:#486581; --accent:#2b6cb0;
      --line:#cfe3ff; --shadow:0 1px 4px rgba(16,42,67,.08);
    }
    *{box-sizing:border-box}
    html,body{margin:0;padding:0;background:var(--bg);color:var(--text);
      font-family:system-ui,-apple-system,"Segoe UI",Roboto,Ubuntu,"Helvetica Neue",Arial,sans-serif}
    .wrap{max-width:980px;margin:20px auto;padding:0 12px}
    .nav{display:flex;gap:8px;justify-content:flex-end;margin:6px 0 16px 0}
    .btn{appearance:none;border:1px solid var(--line);padding:8px 12px;border-radius:8px;
      background:#d9ebff;cursor:pointer;font-weight:600;box-shadow:var(--shadow);text-decoration:none;color:inherit}
    .btn:hover{background:#cfe3ff}
    .panel{background:var(--panel);border:1px solid var(--line);border-radius:12px;
      box-shadow:var(--shadow);padding:14px 14px 16px 14px;margin:14px 0}
    .panel h3{margin:0 0 10px 0;color:var(--accent)}
    .row{margin:10px 0;display:flex;gap:12px;align-items:center;flex-wrap:wrap}
    label{min-width:120px;color:var(--muted)}
    input[type=text],input[type=password]{flex:1 1 320px;padding:8px 10px;border-radius:8px;border:1px solid var(--line)}
    .hint{color:var(--muted);font-size:.9rem}
    .right{display:flex;gap:8px;justify-content:flex-end}
    .danger{background:#ffd9d9;border-color:#ffc5c5}
    .danger:hover{background:#ffcccc}
    .ok{background:#d6ffe1;border-color:#c0f2cd}
    .ok:hover{background:#c8f5d4}
    .mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
  </style>
)CSS";

// ===== Auth helpers (hashed + salted; Basic Auth + token) =====
static void sha256(const uint8_t* data, size_t len, uint8_t out[32]) {
  mbedtls_sha256_context ctx; mbedtls_sha256_init(&ctx);
  mbedtls_sha256_starts_ret(&ctx, 0);
  mbedtls_sha256_update_ret(&ctx, data, len);
  mbedtls_sha256_finish_ret(&ctx, out);
  mbedtls_sha256_free(&ctx);
}

static void loadAuthHashed() {
  Preferences p; p.begin("auth", true);
  g_authUser = p.getString("user", "");
  g_authToken = p.getString("tok", "");
  size_t sl = p.getBytesLength("salt");
  size_t hl = p.getBytesLength("hash");
  if (sl == sizeof(g_authSalt) && hl == sizeof(g_authHash)) {
    p.getBytes("salt", g_authSalt, sizeof(g_authSalt));
    p.getBytes("hash", g_authHash, sizeof(g_authHash));
    g_authEnabled = true;
  } else {
    g_authEnabled = false;
  }
  // Back-compat: if no stored token but legacy plain password exists, synthesize token
  if (g_authToken.length() == 0) {
    String legacy = p.getString("pwd", "");
    if (legacy.length() > 0 && g_authUser.length() > 0) {
      String up = g_authUser + ":" + legacy;
      size_t outcap = (up.length()*4)/3 + 8; size_t olen=0;
      std::unique_ptr<unsigned char[]> out(new unsigned char[outcap]);
      if (mbedtls_base64_encode(out.get(), outcap, &olen, (const unsigned char*)up.c_str(), up.length()) == 0) {
        g_authToken = String((const char*)out.get(), olen);
      }
    }
  }
  p.end();
}

static void disableAuth() {
  Preferences p; p.begin("auth", false);
  p.remove("user"); p.remove("salt"); p.remove("hash"); p.remove("pwd"); p.remove("tok");
  p.end();
  g_authUser = String();
  g_authEnabled = false;
  g_authToken = String();
}

static void saveAuth(const String& user, const String& pass) {
  if (pass.isEmpty()) { disableAuth(); return; }
  String u = user.length() ? user : (g_authUser.length()? g_authUser : String("admin"));
  for (size_t i=0;i<sizeof(g_authSalt);i++) g_authSalt[i] = (uint8_t)esp_random();
  String up = u + ":" + pass + ":";
  const size_t L = up.length();
  std::unique_ptr<uint8_t[]> buf(new uint8_t[L + sizeof(g_authSalt)]);
  memcpy(buf.get(), up.c_str(), L);
  memcpy(buf.get()+L, g_authSalt, sizeof(g_authSalt));
  sha256(buf.get(), L + sizeof(g_authSalt), g_authHash);
  g_authUser = u;
  g_authEnabled = true;
  Preferences p; p.begin("auth", false);
  p.putString("user", g_authUser);
  p.putBytes("salt", g_authSalt, sizeof(g_authSalt));
  p.putBytes("hash", g_authHash, sizeof(g_authHash));
  p.remove("pwd");
  // Save Base64 token for stream embedding on port 81
  // Warning: stores reversible credentials for convenience.
  {
    size_t inlen = u.length() + 1 + pass.length();
    std::unique_ptr<unsigned char[]> in(new unsigned char[inlen]);
    memcpy(in.get(), u.c_str(), u.length()); in.get()[u.length()] = ':';
    memcpy(in.get()+u.length()+1, pass.c_str(), pass.length());
    size_t outcap = (inlen * 4) / 3 + 8; size_t olen = 0;
    std::unique_ptr<unsigned char[]> out(new unsigned char[outcap]);
    if (mbedtls_base64_encode(out.get(), outcap, &olen, in.get(), inlen) == 0) {
      g_authToken = String((const char*)out.get(), olen);
      p.putString("tok", g_authToken);
    } else {
      g_authToken = String(); p.remove("tok");
    }
  }
  p.end();
}

static bool verifyUserPass(const String& u, const String& p) {
  if (!g_authEnabled) return true;
  if (u != g_authUser) return false;
  String up = u + ":" + p + ":";
  const size_t L = up.length();
  std::unique_ptr<uint8_t[]> buf(new uint8_t[L + sizeof(g_authSalt)]);
  memcpy(buf.get(), up.c_str(), L);
  memcpy(buf.get()+L, g_authSalt, sizeof(g_authSalt));
  uint8_t h[32]; sha256(buf.get(), L + sizeof(g_authSalt), h);
  return memcmp(h, g_authHash, 32) == 0;
}

bool isValidTokenParam(const char* token) {
  if (!isAuthEnabled()) return true;
  if (!token) return false;
  // token is Base64(user:pass)
  String b64(token);
  size_t out_len = 0; size_t buflen = (b64.length()*3)/4 + 4;
  std::unique_ptr<uint8_t[]> out(new uint8_t[buflen]);
  if (mbedtls_base64_decode(out.get(), buflen, &out_len, (const unsigned char*)b64.c_str(), b64.length()) != 0) return false;
  String pair((const char*)out.get(), out_len);
  int sep = pair.indexOf(':'); if (sep < 0) return false;
  String u = pair.substring(0, sep);
  String p = pair.substring(sep+1);
  return verifyUserPass(u, p);
}

bool isAuthorizedBasicHeader(const char* header) {
  if (!isAuthEnabled()) return true;
  if (!header) return false;
  String h(header);
  if (!h.startsWith("Basic ")) return false;
  String b64 = h.substring(6);
  size_t out_len = 0; size_t buflen = (b64.length()*3)/4 + 4;
  std::unique_ptr<uint8_t[]> out(new uint8_t[buflen]);
  if (mbedtls_base64_decode(out.get(), buflen, &out_len, (const unsigned char*)b64.c_str(), b64.length()) != 0) return false;
  String pair((const char*)out.get(), out_len);
  int sep = pair.indexOf(':'); if (sep < 0) return false;
  String u = pair.substring(0, sep);
  String p = pair.substring(sep+1);
  return verifyUserPass(u, p);
}

bool isAuthEnabled() {
  static bool loaded=false;
  if (!loaded) { loadAuthHashed(); loaded=true; }
  Preferences p; p.begin("auth", true); String legacy = p.getString("pwd", ""); p.end();
  return g_authEnabled || legacy.length() > 0;
}

String getAuthTokenParam() {
  // Ensure token is loaded if present
  if (g_authToken.length()==0 && !g_authEnabled) {
    loadAuthHashed();
  }
  return g_authToken;
}

// Removed cookie-based bypass: only token or Authorization header are valid.
bool isAuthorized(AsyncWebServerRequest* req) {
  if (!isAuthEnabled()) return true;
  if (req->hasParam("token")) { if (isValidTokenParam(req->getParam("token")->value().c_str())) return true; }
  if (req->hasHeader("Authorization")) { if (isAuthorizedBasicHeader(req->getHeader("Authorization")->value().c_str())) return true; }
  return false;
}

// Helper: consider AP mode as "open" for the wifi setup pages so user can recover device
static bool isAuthorizedOrAP(AsyncWebServerRequest* req) {
  if ((WiFi.getMode() & WIFI_AP) != 0) return true;
  return isAuthorized(req);
}

// Simple HTML escape for user-provided values echoed into pages
static String htmlEscape(const String& s){
  String o; o.reserve(s.length()+8);
  for (size_t i=0;i<s.length();++i){
    char c = s[i];
    if (c == '&') o += "&amp;";
    else if (c == '<') o += "&lt;";
    else if (c == '>') o += "&gt;";
    else if (c == '\"') o += "&quot;";
    else if (c == '\'') o += "&#39;";
    else o += c;
  }
  return o;
}

// ===== Helper: load/save stored creds =====
static void loadStoredCreds() {
  // Prefer MRU[0]; fall back to legacy keys handled in loadMRU
  String ssids[MRU_MAX]; String passes[MRU_MAX]; int n = 0;
  loadMRU(ssids, passes, n);
  ssid = (n > 0) ? ssids[0] : String();
  password = (n > 0) ? passes[0] : String();

  preferences.begin("wifi", true);
  useStaticIP = preferences.getBool("static", false);
  ipStr  = preferences.getString("ip",  "");
  gwStr  = preferences.getString("gw",  "");
  snStr  = preferences.getString("sn",  "");
  dnsStr = preferences.getString("dns", "");
  preferences.end();
}

static void saveCreds(const String& s, const String& p) {
  // Insert or move to front of MRU, then persist
  String ssids[MRU_MAX]; String passes[MRU_MAX]; int n = 0;
  loadMRU(ssids, passes, n);
  mruInsertFrontUnique(ssids, passes, n, s, p);
  saveMRUList(ssids, passes, n);
}

// ===== Public: give access to the shared server (used by CameraServer.cpp) =====
AsyncWebServer& getWebServer() { return g_server; }

// Forward-declare route registration for Wi-Fi pages
static void registerWiFiRoutes();
// static void registerAuthRoutes(); // removed: Basic Auth replaces login UI

// Legacy cookie/login stubs retained for build compatibility (routes not registered)
static void loadAuthPass() {}
static String g_authPass;
static String authToken() { return String(); }
static void sendRedirectWithCookie(AsyncWebServerRequest* req, const String& location, const String& cookie) {
  AsyncWebServerResponse *res = req->beginResponse(302);
  res->addHeader("Location", location);
  if (cookie.length()) res->addHeader("Set-Cookie", cookie);
  req->send(res);
}

// ===== Start the shared server once =====
void ensureWebServerStarted() {
  if (g_serverStarted) return;

  // A tiny health check is handy during bring-up
  g_server.on("/ping", HTTP_GET, [](AsyncWebServerRequest* req){
    req->send(200, "text/plain", "pong");
  });

  g_server.begin();
  g_serverStarted = true;

  // Make Wi-Fi settings and auth pages available even in STA mode
  registerWiFiRoutes();
  /* registerAuthRoutes(); */

  // Nice console hints
  IPAddress ip = (WiFi.status() == WL_CONNECTED) ? WiFi.localIP() : WiFi.softAPIP();
  Serial.printf("Web UI:    http://%s/  (try /ping)\n", ip.toString().c_str());
}

// ===== Connect to stored Wi-Fi (STA) =====
bool connectToStoredWiFi() {
  // Load static IP settings (and seed ssid/password for UI)
  loadStoredCreds();
  // Load MRU list (or legacy single entry)
  String ssids[MRU_MAX]; String passes[MRU_MAX]; int n = 0;
  loadMRU(ssids, passes, n);
  if (n == 0) return false;

  WiFi.mode(WIFI_STA);
  // Apply static IP if configured and valid
  if (useStaticIP) {
    IPAddress ip, gw, sn, dns;
    bool ok = ip.fromString(ipStr) && gw.fromString(gwStr) && sn.fromString(snStr);
    if (!dns.fromString(dnsStr)) dns = gw; // default DNS to gateway if not provided
    if (ok) {
      Serial.printf("Using static IP: %s gw %s sn %s dns %s\n",
        ip.toString().c_str(), gw.toString().c_str(), sn.toString().c_str(), dns.toString().c_str());
      WiFi.config(ip, gw, sn, dns);
    } else {
      Serial.println("Static IP config invalid; falling back to DHCP");
    }
  }

  // Try each MRU candidate until connected
  for (int i = 0; i < n; ++i) {
    const String& s = ssids[i];
    const String& p = passes[i];
    if (s.isEmpty()) continue;
    Serial.printf("Connecting to %s", s.c_str());
    WiFi.begin(s.c_str(), p.c_str());

    int attempts = 0;
    while (WiFi.status() != WL_CONNECTED && attempts < MAX_CONNECT_ATTEMPTS) {
      delay(300);
      Serial.print(".");
      ++attempts;
    }
    Serial.println();

    if (WiFi.status() == WL_CONNECTED) {
      Serial.printf("Connected: %s\n", WiFi.localIP().toString().c_str());
      // Move successful network to front if not already
      if (i != 0) {
        mruMoveToFront(ssids, passes, n, i);
        saveMRUList(ssids, passes, n);
      }
      // Update exported vars for UI
      ssid = ssids[0];
      password = passes[0];
      return true;
    }

    // Clean up before next attempt
    WiFi.disconnect(true);
    delay(200);
  }

  return false;
}

// ===== Render the /wifi page (AP config) =====
static void renderWiFiPage(AsyncWebServerRequest* req, const String& msg = "") {
  loadStoredCreds();

  String html;
  html.reserve(9000);
  html += "<!doctype html><html><head><meta charset='utf-8'><title>ESP32-CAM • Wi-Fi</title>";
  html += kStyle;
  html += "</head><body><div class='wrap'>";

  // Nav between pages
  html += "<div class='nav'>"
          "<a class='btn' href='/wifi'>Wi-Fi Settings</a>"
          "<a class='btn' href='/cam'>Camera</a>"
          "</div>";

  // Panel: Wi-Fi Settings (status)
  html += "<div class='panel'><h3>Wi-Fi Settings</h3>";
  html += "<div class='row'><span class='hint'>Mode: <b>";
  if (WiFi.getMode() & WIFI_AP) html += "Access Point";
  else html += "Station";
  html += "</b></span></div>";
  html += "<div class='row'><span class='hint'>Device IP: <span class='mono'><b>";
  if (WiFi.status() == WL_CONNECTED) html += WiFi.localIP().toString();
  else html += WiFi.softAPIP().toString();
  html += "</b></span></span></div>";
  if (!msg.isEmpty()) {
    html += "<div class='row'><span class='hint'><b>" + htmlEscape(msg) + "</b></span></div>";
  }
  html += "</div>";

  // Panel: Credentials form
  html += "<div class='panel'><h3>Credentials</h3>";
  html += "<form method='POST' action='/wifi/save' onsubmit=\"document.getElementById('saving').style.display='block'\">";
  html += "<div class='row'><label for='ssid'>SSID</label>"
          "<input id='ssid' name='ssid' type='text' value='" + htmlEscape(ssid) + "' required></div>";
  html += "<div class='row'><label for='pass'>Password</label>"
          "<input id='pass' name='pass' type='password' value='" + htmlEscape(password) + "'>"
          "<button type='button' class='btn' onclick=\"const p=document.getElementById('pass');p.type=p.type==='password'?'text':'password'\">Show/Hide</button>"
          "</div>";
  // Network (DHCP / Static)
  html += "<div class='row'><label for='ustatic'>Use Static IP</label>"
          "<input id='ustatic' name='ustatic' type='checkbox' " + String(useStaticIP?"checked":"") + "></div>";
  html += "<div class='row'><label for='ip'>IP Address</label>"
          "<input id='ip' name='ip' type='text' value='" + htmlEscape(ipStr) + "' placeholder='e.g. 192.168.1.50'></div>";
  html += "<div class='row'><label for='gw'>Gateway</label>"
          "<input id='gw' name='gw' type='text' value='" + htmlEscape(gwStr) + "' placeholder='e.g. 192.168.1.1'></div>";
  html += "<div class='row'><label for='sn'>Subnet</label>"
          "<input id='sn' name='sn' type='text' value='" + htmlEscape(snStr) + "' placeholder='e.g. 255.255.255.0'></div>";
  html += "<div class='row'><label for='dns'>DNS</label>"
          "<input id='dns' name='dns' type='text' value='" + htmlEscape(dnsStr) + "' placeholder='(optional, defaults to gateway)'></div>";
  html += "<div class='row'><label for='auser'>Access Username</label>"
          "<input id='auser' name='auser' type='text' value='" + htmlEscape(g_authUser) + "' placeholder='(default admin)'>"
          "</div>";
  html += "<div class='row'><label for='apass'>Access Password</label>"
          "<input id='apass' name='apass' type='password' placeholder='" + String(isAuthEnabled()?"(leave empty to keep current)":"(set to enable protection)") + "'>"
          "<button type='button' class='btn' onclick=\"const a=document.getElementById('apass');a.type=a.type==='password'?'text':'password'\">Show/Hide</button>"
          "</div>";
  // Explicit clear option to avoid accidental resets when changing Wi-Fi
  html += "<div class='row'><label for='aclear'>Clear Password</label>"
          "<input id='aclear' name='aclear' type='checkbox'>"
          "<span class='hint'>(check to remove credentials)</span>"
          "</div>";
  // Token (read-only)
  {
    String tok = getAuthTokenParam();
    html += "<div class='row'><label for='atok'>Stream Token</label>";
    html += "<input id='atok' type='text' readonly value='" + htmlEscape(tok) + "' placeholder='(generated from user:pass)'>";
    html += "<button type='button' class='btn' onclick=\"(function(){var el=document.getElementById('atok');el.focus();el.select();try{document.execCommand('copy');}catch(e){}})()\">Copy</button>";
    html += "</div>";
  }
  // Current stream URL (clickable)
  {
    IPAddress ip = (WiFi.status() == WL_CONNECTED) ? WiFi.localIP() : WiFi.softAPIP();
    String url = String("http://") + ip.toString() + ":81/stream";
    String tok = getAuthTokenParam();
    if (isAuthEnabled() && tok.length()>0) url += "?token=" + tok;
    html += "<div class='row'><label>Stream URL</label>";
    html += "<a class='btn' href='" + htmlEscape(url) + "' target='_blank'>Open Stream</a>";
    html += "<input type='text' readonly value='" + htmlEscape(url) + "' style='flex:1 1 320px'>";
    html += "</div>";
  }
  if (isAuthEnabled()) html += "<div class='row hint'>Auth <b>enabled</b>. Browser prompts via Basic Auth. Also supports ?token=…</div>";
  else html += "<div class='row hint'>Auth <b>disabled</b>. Set credentials to protect the camera.</div>";
  html += "<div class='right'><button class='btn ok' type='submit'>Save &amp; Reboot</button></div>";
  html += "<div id='saving' class='row hint' style='display:none'>Saving… Rebooting…</div>";
  html += "</form></div>";

  // Panel: Known Networks (MRU)
  {
    String ssids[MRU_MAX]; String passes[MRU_MAX]; int n = 0;
    loadMRU(ssids, passes, n);
    html += "<div class='panel'><h3>Known Networks</h3>";
    if (n == 0) {
      html += "<div class='row hint'>No saved networks yet.</div>";
    } else {
      for (int i = 0; i < n; ++i) {
        String tag = (i == 0) ? String("<b>(current)</b>") : String("");
        html += "<div class='row'>";
        html += String("<label>") + String(i+1) + String(".</label>");
        html += "<span class='mono'>" + htmlEscape(ssids[i]) + "</span> ";
        if (i == 0) {
          html += tag;
        } else {
          html += tag;
          // Small inline form to select this entry as active
          html += "<form method='POST' action='/wifi/select' style='margin:0'>";
          html += String("<input type='hidden' name='sel' value='") + String(i) + String("'>");
          html += "<button class='btn' type='submit'>Make Active</button>";
          html += "</form>";
        }
        html += "</div>";
      }
      html += "<div class='row hint'>Selecting a network moves it to the top and reboots, connecting to it on startup.</div>";
    }
    html += "</div>"; // panel
  }

  // Panel: Actions
  html += "<div class='panel'><h3>Actions</h3>";
  html += "<div class='row'>"
          "<a class='btn' href='/cam'>Open Camera</a>"
          "<a class='btn danger' href='/wifi/reboot'>Reboot</a>"
          "</div>";
  html += "</div>";

  html += "</div></body></html>";
  req->send(200, "text/html", html);
}

// ===== Start AP config portal (and register routes) =====
void startConfigPortal() {
  WiFi.mode(WIFI_AP);
  String apName = "ESP32Cam-Setup";
  WiFi.softAP(apName.c_str());
  IPAddress apIP = WiFi.softAPIP();
  Serial.printf("AP started: %s  IP: %s\n", apName.c_str(), apIP.toString().c_str());

  // Register routes (can do this before or after begin)
  g_server.on("/", HTTP_GET, [](AsyncWebServerRequest* req){
    // In AP mode, land users on /wifi
    req->redirect("/wifi");
  });
  // Ensure Wi-Fi settings routes exist
  registerWiFiRoutes();
  /* registerAuthRoutes(); */

  // Make sure the server is actually listening
  ensureWebServerStarted();
}

// ===== Register Wi-Fi routes once =====
static void registerWiFiRoutes() {
  if (g_wifiRoutesAdded) return;

  g_server.on("/wifi", HTTP_GET, [](AsyncWebServerRequest* req){
    // Allow access to the wifi setup page when device is in AP mode
    if (!isAuthorizedOrAP(req)) {
      AsyncWebServerResponse* r = req->beginResponse(401, "text/plain", "Unauthorized");
      r->addHeader("WWW-Authenticate", "Basic realm=\"ESP32Cam\"");
      req->send(r);
      return;
    }
    renderWiFiPage(req);
  });

  g_server.on("/wifi/save", HTTP_POST, [](AsyncWebServerRequest* req){
    // Allow saving in AP mode even if auth previously set
    if (!isAuthorizedOrAP(req) && isAuthEnabled()) {
      AsyncWebServerResponse* r = req->beginResponse(401, "text/plain", "Unauthorized");
      r->addHeader("WWW-Authenticate", "Basic realm=\"ESP32Cam\"");
      req->send(r);
      return;
    }
    String newSsid, newPass;
    String newAuthUser, newAuthPass;
    bool   clearAuth = false;
    bool   newUseStatic = false;
    String newIP, newGW, newSN, newDNS;

    if (req->hasParam("ssid", true)) newSsid = req->getParam("ssid", true)->value();
    if (req->hasParam("pass", true)) newPass = req->getParam("pass", true)->value();
    if (req->hasParam("auser", true)) newAuthUser = req->getParam("auser", true)->value();
    if (req->hasParam("apass", true)) newAuthPass = req->getParam("apass", true)->value();
    if (req->hasParam("aclear", true)) clearAuth = true;
    if (req->hasParam("ustatic", true)) newUseStatic = true;
    if (req->hasParam("ip",   true)) newIP  = req->getParam("ip",  true)->value();
    if (req->hasParam("gw",   true)) newGW  = req->getParam("gw",  true)->value();
    if (req->hasParam("sn",   true)) newSN  = req->getParam("sn",  true)->value();
    if (req->hasParam("dns",  true)) newDNS = req->getParam("dns", true)->value();

    // Save Wi-Fi MRU (does not touch other namespaces)
    saveCreds(newSsid, newPass);
    // Auth changes are explicit-only now: either clear, or set a new password
    if (clearAuth) {
      disableAuth();
    } else if (req->hasParam("apass", true) && newAuthPass.length() > 0) {
      // If password field provided and non-empty, set/replace credentials
      saveAuth(newAuthUser, newAuthPass);
    } // else: leave existing auth untouched
    // Persist network settings
    preferences.begin("wifi", false);
    preferences.putBool("static", newUseStatic);
    preferences.putString("ip",  newIP);
    preferences.putString("gw",  newGW);
    preferences.putString("sn",  newSN);
    preferences.putString("dns", newDNS);
    preferences.end();

    // Feedback page while rebooting
    String html;
    html.reserve(2000);
    html += "<!doctype html><html><head><meta charset='utf-8'><title>Saved</title>";
    html += kStyle;
    html += "<meta http-equiv='refresh' content='3;url=/wifi'>";
    html += "</head><body><div class='wrap'>";
    html += "<div class='nav'><a class='btn' href='/cam'>Camera</a></div>";
    html += "<div class='panel'><h3>Saved</h3><div class='row'><span class='hint'>Credentials saved. Rebooting…</span></div></div>";
    html += "</div></body></html>";
    req->send(200, "text/html", html);

    delay(500);
    ESP.restart();
  });

  g_server.on("/wifi/reboot", HTTP_GET, [](AsyncWebServerRequest* req){
    if (!isAuthorizedOrAP(req)) {
      AsyncWebServerResponse* r = req->beginResponse(401, "text/plain", "Unauthorized");
      r->addHeader("WWW-Authenticate", "Basic realm=\"ESP32Cam\"");
      req->send(r);
      return;
    }
    String html;
    html.reserve(2000);
    html += "<!doctype html><html><head><meta charset='utf-8'><title>Rebooting…</title>";
    html += kStyle;
    html += "</head><body><div class='wrap'>";
    html += "<div class='panel'><h3>Rebooting</h3><div class='row'><span class='hint'>Device is rebooting…</span></div></div>";
    html += "</div></body></html>";
    req->send(200, "text/html", html);
    delay(300);
    ESP.restart();
  });

  // Select an MRU entry to make active (move to front and reboot)
  g_server.on("/wifi/select", HTTP_POST, [](AsyncWebServerRequest* req){
    if (!isAuthorizedOrAP(req)) {
      AsyncWebServerResponse* r = req->beginResponse(401, "text/plain", "Unauthorized");
      r->addHeader("WWW-Authenticate", "Basic realm=\"ESP32Cam\"");
      req->send(r);
      return;
    }

    if (!req->hasParam("sel", true)) {
      req->send(400, "text/plain", "Missing selection");
      return;
    }
    int idx = req->getParam("sel", true)->value().toInt();
    String ssids[MRU_MAX]; String passes[MRU_MAX]; int n = 0;
    loadMRU(ssids, passes, n);
    if (idx < 0 || idx >= n) {
      req->send(400, "text/plain", "Invalid selection");
      return;
    }
    if (idx != 0) {
      mruMoveToFront(ssids, passes, n, idx);
      saveMRUList(ssids, passes, n);
    }

    String html;
    html.reserve(2000);
    html += "<!doctype html><html><head><meta charset='utf-8'><title>Switching…</title>";
    html += kStyle;
    html += "<meta http-equiv='refresh' content='3;url=/wifi'>";
    html += "</head><body><div class='wrap'>";
    html += "<div class='nav'><a class='btn' href='/cam'>Camera</a></div>";
    html += "<div class='panel'><h3>Switching Network</h3><div class='row'><span class='hint'>Rebooting to connect…</span></div></div>";
    html += "</div></body></html>";
    req->send(200, "text/html", html);

    delay(500);
    ESP.restart();
  });

  g_wifiRoutesAdded = true;
}

// ===== Auth routes (login/logout) =====
static void registerAuthRoutes() {
  if (g_authRoutesAdded) return;

  g_server.on("/login", HTTP_GET, [](AsyncWebServerRequest* req){
    String html;
    html.reserve(3000);
    html += "<!doctype html><html><head><meta charset='utf-8'><title>Login</title>";
    html += kStyle;
    html += "</head><body><div class='wrap'>";
    html += "<div class='nav'>"
            "<a class='btn' href='/wifi'>Wi-Fi Settings</a>"
            "<a class='btn' href='/cam'>Camera</a>"
            "</div>";
    html += "<div class='panel'><h3>Login</h3>";
    if (!isAuthEnabled()) {
      html += "<div class='row hint'>Authentication is not enabled.</div>";
    } else {
      html += "<div class='row hint'>Use your configured credentials.</div>";
    }
    html += "</div></div></body></html>";
    req->send(200, "text/html", html);
  });

  g_authRoutesAdded = true;
}
## `WiFiManager.h`
**Absolute path**: C:\Users\stellaris\Documents\PlatformIO\Projects\ESP32_CAM_AI\src\WiFiManager.h
**Size**: 943 bytes
**Modified**: 2025-10-23 23:09:45 +0100
**SHA256**: 5c8386f501664c2d63ec045e8db8d882b6b571295b963e61d71a3f2ec46e22b1
``````c// WiFiManager.h
#ifndef WIFI_MANAGER_H
#define WIFI_MANAGER_H

#pragma once
#include <Arduino.h> // for String

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
// Expose saved Base64 token (user:pass) for building intra-page links
String getAuthTokenParam();
#endif
