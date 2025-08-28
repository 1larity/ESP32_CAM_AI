#include "WiFiManager.h"
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

// Stored creds
static Preferences preferences;
static String ssid, password;
// Optional static IP configuration (stored in the same prefs namespace "wifi")
static bool useStaticIP = false;
static String ipStr, gwStr, snStr, dnsStr;

// Attempts allowed before AP fallback
static const int MAX_CONNECT_ATTEMPTS = 20;

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
  size_t sl = p.getBytesLength("salt");
  size_t hl = p.getBytesLength("hash");
  if (sl == sizeof(g_authSalt) && hl == sizeof(g_authHash)) {
    p.getBytes("salt", g_authSalt, sizeof(g_authSalt));
    p.getBytes("hash", g_authHash, sizeof(g_authHash));
    g_authEnabled = true;
  } else {
    g_authEnabled = false;
  }
  p.end();
}

static void disableAuth() {
  Preferences p; p.begin("auth", false);
  p.remove("user"); p.remove("salt"); p.remove("hash"); p.remove("pwd");
  p.end();
  g_authUser = String();
  g_authEnabled = false;
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

bool isAuthorized(AsyncWebServerRequest* req) {
  if (!isAuthEnabled()) return true;
  if (req->hasParam("token")) { if (isValidTokenParam(req->getParam("token")->value().c_str())) return true; }
  if (req->hasHeader("Authorization")) { if (isAuthorizedBasicHeader(req->getHeader("Authorization")->value().c_str())) return true; }
  if (req->hasHeader("Cookie")) { String ck=req->getHeader("Cookie")->value(); if (ck.indexOf("cam_auth=")>=0) return true; }
  return false;
}

// ===== Helper: load/save stored creds =====
static void loadStoredCreds() {
  preferences.begin("wifi", true);
  ssid     = preferences.getString("ssid", "");
  password = preferences.getString("pass", "");
  useStaticIP = preferences.getBool("static", false);
  ipStr  = preferences.getString("ip",  "");
  gwStr  = preferences.getString("gw",  "");
  snStr  = preferences.getString("sn",  "");
  dnsStr = preferences.getString("dns", "");
  preferences.end();
}

static void saveCreds(const String& s, const String& p) {
  preferences.begin("wifi", false);
  preferences.putString("ssid", s);
  preferences.putString("pass", p);
  preferences.end();
}

// ===== Public: give access to the shared server (used by CameraServer.cpp) =====
AsyncWebServer& getWebServer() { return g_server; }

// Forward-declare route registration for Wi‑Fi pages
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

  // Make Wi‑Fi settings and auth pages available even in STA mode
  registerWiFiRoutes();
  /* registerAuthRoutes(); */

  // Nice console hints
  IPAddress ip = (WiFi.status() == WL_CONNECTED) ? WiFi.localIP() : WiFi.softAPIP();
  Serial.printf("Web UI:    http://%s/  (try /ping)\n", ip.toString().c_str());
}

// ===== Connect to stored Wi‑Fi (STA) =====
bool connectToStoredWiFi() {
  loadStoredCreds();
  if (ssid.isEmpty()) return false;

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
  WiFi.begin(ssid.c_str(), password.c_str());
  Serial.printf("Connecting to %s", ssid.c_str());

  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < MAX_CONNECT_ATTEMPTS) {
    delay(500);
    Serial.print(".");
    attempts++;
  }
  Serial.println();

  if (WiFi.status() == WL_CONNECTED) {
    Serial.printf("Connected: %s\n", WiFi.localIP().toString().c_str());
    return true;
  }
  return false;
}

// ===== Render the /wifi page (AP config) =====
static void renderWiFiPage(AsyncWebServerRequest* req, const String& msg = "") {
  loadStoredCreds();

  String html;
  html.reserve(9000);
  html += "<!doctype html><html><head><meta charset='utf-8'><title>ESP32‑CAM • Wi‑Fi</title>";
  html += kStyle;
  html += "</head><body><div class='wrap'>";

  // Nav between pages
  html += "<div class='nav'>"
          "<a class='btn' href='/wifi'>Wi‑Fi Settings</a>"
          "<a class='btn' href='/cam'>Camera</a>"
          "</div>";

  // Panel: Wi‑Fi Settings (status)
  html += "<div class='panel'><h3>Wi‑Fi Settings</h3>";
  html += "<div class='row'><span class='hint'>Mode: <b>";
  if (WiFi.getMode() & WIFI_AP) html += "Access Point";
  else html += "Station";
  html += "</b></span></div>";
  html += "<div class='row'><span class='hint'>Device IP: <span class='mono'><b>";
  if (WiFi.status() == WL_CONNECTED) html += WiFi.localIP().toString();
  else html += WiFi.softAPIP().toString();
  html += "</b></span></span></div>";
  if (!msg.isEmpty()) {
    html += "<div class='row'><span class='hint'><b>" + msg + "</b></span></div>";
  }
  html += "</div>";

  // Panel: Credentials form
  html += "<div class='panel'><h3>Credentials</h3>";
  html += "<form method='POST' action='/wifi/save' onsubmit=\"document.getElementById('saving').style.display='block'\">";
  html += "<div class='row'><label for='ssid'>SSID</label>"
          "<input id='ssid' name='ssid' type='text' value='" + ssid + "' required></div>";
  html += "<div class='row'><label for='pass'>Password</label>"
          "<input id='pass' name='pass' type='password' value='" + password + "'>"
          "<button type='button' class='btn' onclick=\"const p=document.getElementById('pass');p.type=p.type==='password'?'text':'password'\">Show/Hide</button>"
          "</div>";
  // Network (DHCP / Static)
  html += "<div class='row'><label for='ustatic'>Use Static IP</label>"
          "<input id='ustatic' name='ustatic' type='checkbox' " + String(useStaticIP?"checked":"") + "></div>";
  html += "<div class='row'><label for='ip'>IP Address</label>"
          "<input id='ip' name='ip' type='text' value='" + ipStr + "' placeholder='e.g. 192.168.1.50'></div>";
  html += "<div class='row'><label for='gw'>Gateway</label>"
          "<input id='gw' name='gw' type='text' value='" + gwStr + "' placeholder='e.g. 192.168.1.1'></div>";
  html += "<div class='row'><label for='sn'>Subnet</label>"
          "<input id='sn' name='sn' type='text' value='" + snStr + "' placeholder='e.g. 255.255.255.0'></div>";
  html += "<div class='row'><label for='dns'>DNS</label>"
          "<input id='dns' name='dns' type='text' value='" + dnsStr + "' placeholder='(optional, defaults to gateway)'></div>";
  html += "<div class='row'><label for='auser'>Access Username</label>"
          "<input id='auser' name='auser' type='text' placeholder='(default admin)'>"
          "</div>";
  html += "<div class='row'><label for='apass'>Access Password</label>"
          "<input id='apass' name='apass' type='password' placeholder='" + String(isAuthEnabled()?"(set to change or clear)":"(leave empty to keep open)") + "'>"
          "<button type='button' class='btn' onclick=\"const a=document.getElementById('apass');a.type=a.type==='password'?'text':'password'\">Show/Hide</button>"
          "</div>";
  if (isAuthEnabled()) html += "<div class='row hint'>Auth <b>enabled</b>. Browser prompts via Basic Auth. Also supports ?token=…</div>";
  else html += "<div class='row hint'>Auth <b>disabled</b>. Set credentials to protect the camera.</div>";
  html += "<div class='right'><button class='btn ok' type='submit'>Save &amp; Reboot</button></div>";
  html += "<div id='saving' class='row hint' style='display:none'>Saving… Rebooting…</div>";
  html += "</form></div>";

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
  // Ensure Wi‑Fi settings routes exist
  registerWiFiRoutes();
  /* registerAuthRoutes(); */

  // Make sure the server is actually listening
  ensureWebServerStarted();
}

// ===== Register Wi‑Fi routes once =====
static void registerWiFiRoutes() {
  if (g_wifiRoutesAdded) return;

  g_server.on("/wifi", HTTP_GET, [](AsyncWebServerRequest* req){
    if (!isAuthorized(req)) {
      AsyncWebServerResponse* r = req->beginResponse(401, "text/plain", "Unauthorized");
      r->addHeader("WWW-Authenticate", "Basic realm=\"ESP32Cam\"");
      req->send(r);
      return;
    }
    renderWiFiPage(req);
  });

  g_server.on("/wifi/save", HTTP_POST, [](AsyncWebServerRequest* req){
    if (!isAuthorized(req) && isAuthEnabled()) {
      AsyncWebServerResponse* r = req->beginResponse(401, "text/plain", "Unauthorized");
      r->addHeader("WWW-Authenticate", "Basic realm=\"ESP32Cam\"");
      req->send(r);
      return;
    }
    String newSsid, newPass;
    String newAuthUser, newAuthPass;
    bool   newUseStatic = false;
    String newIP, newGW, newSN, newDNS;

    if (req->hasParam("ssid", true)) newSsid = req->getParam("ssid", true)->value();
    if (req->hasParam("pass", true)) newPass = req->getParam("pass", true)->value();
    if (req->hasParam("auser", true)) newAuthUser = req->getParam("auser", true)->value();
    if (req->hasParam("apass", true)) newAuthPass = req->getParam("apass", true)->value();
    if (req->hasParam("ustatic", true)) newUseStatic = true;
    if (req->hasParam("ip",   true)) newIP  = req->getParam("ip",  true)->value();
    if (req->hasParam("gw",   true)) newGW  = req->getParam("gw",  true)->value();
    if (req->hasParam("sn",   true)) newSN  = req->getParam("sn",  true)->value();
    if (req->hasParam("dns",  true)) newDNS = req->getParam("dns", true)->value();

    saveCreds(newSsid, newPass);
    if (req->hasParam("apass", true)) {
      if (newAuthPass.length() == 0) {
        // disable auth (hashed + legacy)
        disableAuth();
      } else {
        saveAuth(newAuthUser, newAuthPass);
      }
    }
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
            "<a class='btn' href='/wifi'>Wi‑Fi Settings</a>"
            "<a class='btn' href='/cam'>Camera</a>"
            "</div>";
    html += "<div class='panel'><h3>Login</h3>";
    if (!isAuthEnabled()) {
      html += "<div class='row hint'>No password set. Access is open.</div>";
    }
    html += "<form method='POST' action='/login' onsubmit=\"document.getElementById('saving').style.display='block'\">";
    html += "<div class='row'><label for='pass'>Password</label>"
            "<input id='pass' name='pass' type='password' required>"
            "<button type='button' class='btn' onclick=\"const p=document.getElementById('pass');p.type=p.type==='password'?'text':'password'\">Show/Hide</button>"
            "</div>";
    String ret = "/";
    if (req->hasParam("return")) ret = req->getParam("return")->value();
    html += "<input type='hidden' name='return' value='" + ret + "'>";
    html += "<div class='right'><button class='btn ok' type='submit'>Login</button></div>";
    html += "<div id='saving' class='row hint' style='display:none'>Signing in…</div>";
    html += "</form></div>";
    html += "</div></body></html>";
    req->send(200, "text/html", html);
  });

  g_server.on("/login", HTTP_POST, [](AsyncWebServerRequest* req){
    String pass;
    String ret = "/";
    if (req->hasParam("pass", true)) pass = req->getParam("pass", true)->value();
    if (req->hasParam("return", true)) ret = req->getParam("return", true)->value();

    loadAuthPass();
    if (!isAuthEnabled() || pass == g_authPass) {
      String cookie = String("cam_auth=") + authToken() + "; Path=/; Max-Age=2592000; HttpOnly";
      sendRedirectWithCookie(req, ret, cookie);
    } else {
      sendRedirectWithCookie(req, String("/login?return=") + ret, String());
    }
  });

  g_server.on("/logout", HTTP_GET, [](AsyncWebServerRequest* req){
    String cookie = String("cam_auth=; Path=/; Max-Age=0; HttpOnly");
    sendRedirectWithCookie(req, "/login", cookie);
  });

  g_authRoutesAdded = true;
}
