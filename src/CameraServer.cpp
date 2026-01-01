// CameraServer.cpp
#include "WiFiManager.h"
#include "CameraServer.h"
#include "OTAHandler.h"
#include "esp_camera.h"
#include <WiFi.h>
#include <ESPAsyncWebServer.h>
#include <Preferences.h>
#include "StreamServer.h"
#include "PTZ.h"

// ===== Resolution options with actual pixel sizes =====
struct ResolutionOption {
  framesize_t size;
  const char* name;
  uint16_t    w;
  uint16_t    h;
};
static Preferences camPrefs;
static const char* CAM_NS  = "camera";
static const char* KEY_RES = "res";
static const char* KEY_FLASH = "flash";
static const char* KEY_FLASH_LEVEL = "flash_level";

static const ResolutionOption resolutionOptions[] = {
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
// AI Thinker pin map
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27

#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22
#define FLASH_LED_PIN      4

static bool g_flash_on = false;
static int  g_flash_level = 0;  // 0-1023 (10-bit PWM)

static void setFlash(bool on) {
  // Flash disabled to avoid interfering with the stream.
  (void)on;
  g_flash_on = false;
  g_flash_level = 0;
  digitalWrite(FLASH_LED_PIN, LOW);
}

static void setFlashLevel(int level) {
  // Flash disabled to avoid interfering with the stream.
  (void)level;
  g_flash_on = false;
  g_flash_level = 0;
  digitalWrite(FLASH_LED_PIN, LOW);
}

void setupCamera() {
  camera_config_t config = {};
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer   = LEDC_TIMER_0;
  config.pin_d0       = Y2_GPIO_NUM;
  config.pin_d1       = Y3_GPIO_NUM;
  config.pin_d2       = Y4_GPIO_NUM;
  config.pin_d3       = Y5_GPIO_NUM;
  config.pin_d4       = Y6_GPIO_NUM;
  config.pin_d5       = Y7_GPIO_NUM;
  config.pin_d6       = Y8_GPIO_NUM;
  config.pin_d7       = Y9_GPIO_NUM;
  config.pin_xclk     = XCLK_GPIO_NUM;
  config.pin_pclk     = PCLK_GPIO_NUM;
  config.pin_vsync    = VSYNC_GPIO_NUM;
  config.pin_href     = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn     = PWDN_GPIO_NUM;
  config.pin_reset    = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;

  // PSRAM tuning
  if (psramFound()) {
    config.frame_size   = FRAMESIZE_VGA; // will override below from prefs
    config.jpeg_quality = 12;            // lower = better quality
    config.fb_count     = 2;
    config.grab_mode    = CAMERA_GRAB_LATEST;
  } else {
    config.frame_size   = FRAMESIZE_QVGA;
    config.jpeg_quality = 15;
    config.fb_count     = 1;
  }

  // Init camera
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed: 0x%X\n", err);
    return;
  }

  // Force flash pin off (no PWM setup to avoid stream interference).
  pinMode(FLASH_LED_PIN, OUTPUT);
  digitalWrite(FLASH_LED_PIN, LOW);

  // Apply persisted framesize if present
  framesize_t fs = config.frame_size;
  camPrefs.begin(CAM_NS, true);
  if (camPrefs.isKey(KEY_RES)) {
    fs = (framesize_t)camPrefs.getUChar(KEY_RES, (uint8_t)config.frame_size);
  }
  camPrefs.end();

  sensor_t* s = esp_camera_sensor_get();
  if (s) s->set_framesize(s, fs);

  // Start port-81 MJPEG stream server
  startStreamServer();

  Serial.println("Camera ready");
}
static inline const ResolutionOption* findByName(const String& n) {
  for (auto& o : resolutionOptions) if (n.equalsIgnoreCase(o.name)) return &o;
  return nullptr;
}

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

static void send401(AsyncWebServerRequest* req) {
  auto* r = req->beginResponse(401, "text/plain", "Unauthorized");
  r->addHeader("WWW-Authenticate", "Basic realm=\"ESP32Cam\"");
  req->send(r);
}

static void renderCameraPage(AsyncWebServerRequest* request) {
  sensor_t* s = esp_camera_sensor_get();
  framesize_t cur = s ? s->status.framesize : FRAMESIZE_VGA;

  String html;
  html.reserve(7000);
  html += "<!doctype html><html><head><meta charset='utf-8'><title>ESP32-CAM</title>";
  html += kStyle;
  html += "</head><body><div class='wrap'>";

  html += "<div class='nav'>"
          "<a class='btn' href='/wifi'>Wi-Fi Settings</a>"
          "<a class='btn' href='/ptz/home'>PTZ Home</a>"
          "</div>";

  // Controls
  html += "<div class='panel'><h3>Camera Controls</h3>";
  html += "<div class='row'><label for='res'>Resolution</label>"
          "<select id='res' onchange=\"fetch('/resolution?set='+this.value).then(()=>location.reload())\">";
  for (auto& o : resolutionOptions) {
    bool sel = (s && o.size == cur);
    html += String("<option value='") + o.name + "'" + (sel ? " selected" : "") + ">"
          + o.name + " (" + o.w + "&times;" + o.h + ")</option>";
  }
  html += "</select></div>";

  // PTZ buttons call /ptz/step
  html += "<div class='row'>"
          "<button class='btn' onclick=\"fetch('/ptz/step?dy=10')\">Up</button> "
          "<button class='btn' onclick=\"fetch('/ptz/step?dy=-10')\">Down</button> "
          "<button class='btn' onclick=\"fetch('/ptz/step?dx=-10')\">Left</button> "
          "<button class='btn' onclick=\"fetch('/ptz/step?dx=10')\">Right</button>"
          "</div>";
  html += "</div>";

  // Live video
  html += "<div class='panel'><h3>Live Video</h3>";
  String streamURL = String("http://") + WiFi.localIP().toString() + ":81/stream";
  if (isAuthEnabled()) {
    String tok = getAuthTokenParam();
    if (tok.length() > 0) streamURL += "?token=" + tok;
  }
  html += "<img class='stream' src='" + streamURL + "' alt='Video stream'>";
  html += "</div>";

  // Status block
  html += "<div class='panel'><h3>Status</h3><pre id='st'>Loading...</pre></div>"
          "<script>"
          "function applyStatus(j){document.getElementById('st').textContent=JSON.stringify(j,null,2);}"
          "fetch('/api/status').then(r=>r.json()).then(applyStatus);"
          "</script>";

  html += "</div></body></html>";
  request->send(200, "text/html", html);
}

void startCameraServer() {
  AsyncWebServer& srv = getWebServer();

  srv.on("/", HTTP_GET, [](AsyncWebServerRequest *req){
    if (!isAuthorized(req)) { send401(req); return; }
    renderCameraPage(req);
  });
  srv.on("/cam", HTTP_GET, [](AsyncWebServerRequest *req){
    if (!isAuthorized(req)) { send401(req); return; }
    renderCameraPage(req);
  });

  // Change resolution and persist
  srv.on("/resolution", HTTP_GET, [](AsyncWebServerRequest *req){
    if (!isAuthorized(req)) { send401(req); return; }
    if (!req->hasParam("set")) { req->send(400, "text/plain", "Missing ?set="); return; }
    const String target = req->getParam("set")->value();
    const auto* opt = findByName(target);
    if (!opt) { req->send(400, "text/plain", "Invalid resolution"); return; }

    sensor_t* s = esp_camera_sensor_get();
    if (!s) { req->send(500, "text/plain", "Sensor unavailable"); return; }
    s->set_framesize(s, opt->size);

    Preferences camPrefs; camPrefs.begin("camera", false);
    camPrefs.putUChar("res", static_cast<uint8_t>(opt->size));
    camPrefs.end();

    req->send(200, "text/plain", "OK");
  });

  // Composite status JSON for page
  srv.on("/api/status", HTTP_GET, [](AsyncWebServerRequest* req){
    if (!isAuthorized(req)) { send401(req); return; }
    sensor_t* s = esp_camera_sensor_get();
    framesize_t fs = s ? s->status.framesize : FRAMESIZE_VGA;
    int pan=0, tilt=0; ptzGet(pan, tilt);

    char buf[256];
    snprintf(buf, sizeof(buf),
      "{"
      "\"ip\":\"%s\","
      "\"framesize\":%d,"
      "\"ptz\":{\"pan\":%d,\"tilt\":%d}"
      "}",
      WiFi.localIP().toString().c_str(),
      (int)fs,
      pan, tilt);
    req->send(200, "application/json", buf);
  });
}
