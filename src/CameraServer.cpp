// CameraServer.cpp
#include "WiFiManager.h"
#include "CameraServer.h"
#include "OTAHandler.h"
#include "Utils.h"

#include "esp_camera.h"
#include <WiFi.h>
#include <ESPAsyncWebServer.h>
#include <ESP32Servo.h>
#include <Preferences.h>
#include <vector>
#include <memory>

#include "StreamServer.h"

// ===== PTZ pins =====
#define SERVO_1 14
#define SERVO_2 15

// ===== PTZ state (servo objects must be global to match Utils.cpp's externs) =====
Servo servo1;
Servo servo2;
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
  html += "<!doctype html><html><head><meta charset='utf-8'><title>ESP32‑CAM</title>";
  html += kStyle;
  html += "</head><body><div class='wrap'>";

  // Small nav: Wi‑Fi Settings
  html += "<div class='nav'>"
          "<a class='btn' href='/wifi'>Wi‑Fi Settings</a>"
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

  // PANEL: Video (reference style: direct :81/stream without token)
  html += "<div class='panel'><h3>Live Video</h3>";
  html += "<img class='stream' src='http://" + WiFi.localIP().toString() + ":81/stream' alt='Video stream'>";
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

  // MJPEG stream on port 80 (proxy endpoint using shared broker)
#if STREAM_BROKER && defined(ENABLE_PORT80_STREAM) && (ENABLE_PORT80_STREAM==1)
  camServer.on("/stream", HTTP_GET, [](AsyncWebServerRequest *request){
    if (!isAuthorized(request)) {
      auto* r = request->beginResponse(401, "text/plain", "Unauthorized");
      r->addHeader("WWW-Authenticate", "Basic realm=\"ESP32Cam\"");
      request->send(r);
      return;
    }
    // Use the same multipart format as :81 handler (header->image->boundary)
    struct RespState { std::vector<uint8_t> pending; uint32_t last_seq = 0; };
    auto state = std::make_shared<RespState>();
    auto filler = [state](uint8_t* out, size_t maxLen, size_t index) mutable -> size_t {
      extern SemaphoreHandle_t g_frameMutex;
      extern std::vector<uint8_t> g_lastJpg;
      extern volatile uint32_t g_frameSeq;
      // Wait for a new frame
      uint32_t start = millis();
      while (state->last_seq == g_frameSeq) { vTaskDelay(5 / portTICK_PERIOD_MS); if (millis() - start > 1000) break; }
      // Copy current frame
      std::vector<uint8_t> local;
      if (g_frameMutex && xSemaphoreTake(g_frameMutex, 10 / portTICK_PERIOD_MS) == pdTRUE) {
        local = g_lastJpg; state->last_seq = g_frameSeq;
        xSemaphoreGive(g_frameMutex);
      }
      if (local.empty() && state->pending.empty()) return 0;
      const char* b = "\r\n--frame\r\n";
      char hdr[64]; int hlen = snprintf(hdr, sizeof(hdr), "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n", (unsigned)local.size());
      // Build pending buffer if empty
      if (state->pending.empty() && !local.empty()) {
        state->pending.reserve(hlen + local.size() + strlen(b));
        state->pending.insert(state->pending.end(), (const uint8_t*)hdr, (const uint8_t*)hdr+hlen);
        state->pending.insert(state->pending.end(), local.begin(), local.end());
        state->pending.insert(state->pending.end(), (const uint8_t*)b, (const uint8_t*)b+strlen(b));
      }
      if (state->pending.empty()) return 0;
      size_t n = state->pending.size() < maxLen ? state->pending.size() : maxLen;
      memcpy(out, state->pending.data(), n);
      // Erase sent bytes
      state->pending.erase(state->pending.begin(), state->pending.begin()+n);
      return n;
    };
    auto* resp = request->beginChunkedResponse("multipart/x-mixed-replace; boundary=frame", filler);
    resp->addHeader("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0");
    resp->addHeader("Pragma", "no-cache");
    resp->addHeader("Access-Control-Allow-Origin", "*");
    request->send(resp);
  });
#endif

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
