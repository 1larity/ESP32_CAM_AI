#include "PTZ.h"

#include <ESP32Servo.h>
#include <Preferences.h>
#include <ESPAsyncWebServer.h>

#include "WiFiManager.h"   // isAuthorized(), getWebServer(), ensureWebServerStarted()

// ===== Hardware pins (AI Thinker defaults; change if needed) =====
#ifndef PTZ_SERVO_PAN_PIN
#define PTZ_SERVO_PAN_PIN 14
#endif
#ifndef PTZ_SERVO_TILT_PIN
#define PTZ_SERVO_TILT_PIN 15
#endif

// ===== Limits and defaults =====
static const int kMinDeg = 0;
static const int kMaxDeg = 180;
static const int kStepDefault = 10;

static Servo g_pan;
static Servo g_tilt;

static int g_panDeg = 90;
static int g_tiltDeg = 90;

static Preferences g_prefs; // namespace "ptz"

// Helpers
static inline int clamp(int v, int lo, int hi) {
  if (v < lo) return lo;
  if (v > hi) return hi;
  return v;
}
static void writeServos() {
  g_pan.write(clamp(g_panDeg, kMinDeg, kMaxDeg));
  g_tilt.write(clamp(g_tiltDeg, kMinDeg, kMaxDeg));
}

void ptzInit() {
  // Configure frequency first to avoid jitter
  g_pan.setPeriodHertz(50);
  g_tilt.setPeriodHertz(50);

  // Attach once; on ESP32 you can attach without explicit channel if not used elsewhere
  g_pan.attach(PTZ_SERVO_PAN_PIN, 500, 2500);  // 0..180 -> 0.5..2.5ms
  g_tilt.attach(PTZ_SERVO_TILT_PIN, 500, 2500);

  // Load persisted angles; fall back to 90/90
  g_prefs.begin("ptz", false);
  g_panDeg  = clamp(g_prefs.getInt("pan",  90), kMinDeg, kMaxDeg);
  g_tiltDeg = clamp(g_prefs.getInt("tilt", 90), kMinDeg, kMaxDeg);
  // Optional separate home values; default to current if missing
  int homePan  = g_prefs.getInt("homePan",  g_panDeg);
  int homeTilt = g_prefs.getInt("homeTilt", g_tiltDeg);
  // Ensure current is within range
  g_panDeg  = clamp(g_panDeg,  kMinDeg, kMaxDeg);
  g_tiltDeg = clamp(g_tiltDeg, kMinDeg, kMaxDeg);
  g_prefs.end();

  // Move to current (which equals home on first boot)
  writeServos();

  // Ensure routes exist
  ensureWebServerStarted();
  ptzRegisterRoutes(getWebServer());
}

bool ptzSet(int panDeg, int tiltDeg) {
  int np = clamp(panDeg,  kMinDeg, kMaxDeg);
  int nt = clamp(tiltDeg, kMinDeg, kMaxDeg);
  bool changed = (np != g_panDeg) || (nt != g_tiltDeg);
  g_panDeg = np; g_tiltDeg = nt;
  writeServos();
  // Persist current for warm reboots
  g_prefs.begin("ptz", false);
  g_prefs.putInt("pan", g_panDeg);
  g_prefs.putInt("tilt", g_tiltDeg);
  g_prefs.end();
  return changed;
}

bool ptzStep(int dPan, int dTilt) {
  return ptzSet(g_panDeg + dPan, g_tiltDeg + dTilt);
}

void ptzHome() {
  g_prefs.begin("ptz", true);
  int hp = g_prefs.getInt("homePan",  90);
  int ht = g_prefs.getInt("homeTilt", 90);
  g_prefs.end();
  ptzSet(hp, ht);
}

void ptzSaveHome(int panDeg, int tiltDeg) {
  int np = clamp(panDeg,  kMinDeg, kMaxDeg);
  int nt = clamp(tiltDeg, kMinDeg, kMaxDeg);
  g_prefs.begin("ptz", false);
  g_prefs.putInt("homePan",  np);
  g_prefs.putInt("homeTilt", nt);
  g_prefs.end();
}

void ptzGet(int& panOut, int& tiltOut) {
  panOut = g_panDeg; tiltOut = g_tiltDeg;
}

// ===== HTTP routes (port 80 via AsyncWebServer) =====
static void send401(AsyncWebServerRequest* req) {
  auto* r = req->beginResponse(401, "text/plain", "Unauthorized");
  r->addHeader("WWW-Authenticate", "Basic realm=\"ESP32Cam\"");
  req->send(r);
}

void ptzRegisterRoutes(AsyncWebServer& srv) {
  // Absolute set: /ptz?pan=..&tilt=.. [&savehome=1]
  srv.on("/ptz", HTTP_GET, [](AsyncWebServerRequest* req){
    if (!isAuthorized(req)) { send401(req); return; }
    bool had = false;
    int pan = g_panDeg, tilt = g_tiltDeg;
    if (req->hasParam("pan"))  { pan  = req->getParam("pan")->value().toInt();  had = true; }
    if (req->hasParam("tilt")) { tilt = req->getParam("tilt")->value().toInt(); had = true; }
    if (had) ptzSet(pan, tilt);

    if (req->hasParam("savehome")) {
      ptzSaveHome(g_panDeg, g_tiltDeg);
    }

    int hp=90, ht=90;
    Preferences p; p.begin("ptz", true);
    hp = p.getInt("homePan", 90);
    ht = p.getInt("homeTilt", 90);
    p.end();

    char buf[128];
    snprintf(buf, sizeof(buf),
      "{\"pan\":%d,\"tilt\":%d,\"homePan\":%d,\"homeTilt\":%d}",
      g_panDeg, g_tiltDeg, hp, ht);
    req->send(200, "application/json", buf);
  });

  // Relative step: /ptz/step?dx=..&dy=..
  srv.on("/ptz/step", HTTP_GET, [](AsyncWebServerRequest* req){
    if (!isAuthorized(req)) { send401(req); return; }
    int dx = kStepDefault;
    int dy = 0;
    if (req->hasParam("dx")) dx = req->getParam("dx")->value().toInt();
    if (req->hasParam("dy")) dy = req->getParam("dy")->value().toInt();
    ptzStep(dx, dy);
    char buf[64];
    snprintf(buf, sizeof(buf), "{\"pan\":%d,\"tilt\":%d}", g_panDeg, g_tiltDeg);
    req->send(200, "application/json", buf);
  });

  // Go home: /ptz/home
  srv.on("/ptz/home", HTTP_GET, [](AsyncWebServerRequest* req){
    if (!isAuthorized(req)) { send401(req); return; }
    ptzHome();
    char buf[64];
    snprintf(buf, sizeof(buf), "{\"pan\":%d,\"tilt\":%d}", g_panDeg, g_tiltDeg);
    req->send(200, "application/json", buf);
  });

  // PTZ status: /api/ptz
  srv.on("/api/ptz", HTTP_GET, [](AsyncWebServerRequest* req){
    if (!isAuthorized(req)) { send401(req); return; }
    char buf[64];
    snprintf(buf, sizeof(buf), "{\"pan\":%d,\"tilt\":%d}", g_panDeg, g_tiltDeg);
    req->send(200, "application/json", buf);
  });
}
