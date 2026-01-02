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

static void mruDeleteIndex(String ssids[], String passes[], int &count, int idx) {
  if (idx < 0 || idx >= count) return;
  for (int i = idx; i < count - 1; ++i) {
    ssids[i] = ssids[i + 1];
    passes[i] = passes[i + 1];
  }
  if (count > 0) --count;
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

bool isAuthEnabled() {
  static bool loaded=false;
  if (!loaded) { loadAuthHashed(); loaded=true; }
  Preferences p; p.begin("auth", true); String legacy = p.getString("pwd", ""); p.end();
  return g_authEnabled || legacy.length() > 0;
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
          "<a class='btn' href='/cam' title='Open camera controls'>Camera</a>"
          "<a class='btn' href='/about' title='About this firmware'>About</a>"
          "</div>";

  // Panel: Wi-Fi Settings (status)
  html += "<div class='panel'><h3>Wi-Fi Settings</h3>";
  html += "<div class='row'><span class='hint' title='Current Wi-Fi operating mode'>Mode: <b>";
  if (WiFi.getMode() & WIFI_AP) html += "Access Point";
  else html += "Station";
  html += "</b></span></div>";
  html += "<div class='row'><span class='hint' title='Current IP address'>Device IP: <span class='mono'><b>";
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
          "<input id='ssid' name='ssid' type='text' value='" + htmlEscape(ssid) + "' required title='Network name to connect to'></div>";
  html += "<div class='row'><label for='pass'>Password</label>"
          "<input id='pass' name='pass' type='password' value='" + htmlEscape(password) + "' title='Wi-Fi password (leave blank for open networks)'>"
          "<button type='button' class='btn' title='Show or hide password' onclick=\"const p=document.getElementById('pass');p.type=p.type==='password'?'text':'password'\">Show/Hide</button>"
          "</div>";
  // Network (DHCP / Static)
  html += "<div class='row'><label for='ustatic'>Use Static IP</label>"
          "<input id='ustatic' name='ustatic' type='checkbox' " + String(useStaticIP?"checked":"") + " title='Enable static IP instead of DHCP'></div>";
  html += "<div class='row'><label for='ip'>IP Address</label>"
          "<input id='ip' name='ip' type='text' value='" + htmlEscape(ipStr) + "' placeholder='e.g. 192.168.1.50' title='Static IP for the camera'></div>";
  html += "<div class='row'><label for='gw'>Gateway</label>"
          "<input id='gw' name='gw' type='text' value='" + htmlEscape(gwStr) + "' placeholder='e.g. 192.168.1.1' title='Gateway/router address'></div>";
  html += "<div class='row'><label for='sn'>Subnet</label>"
          "<input id='sn' name='sn' type='text' value='" + htmlEscape(snStr) + "' placeholder='e.g. 255.255.255.0' title='Subnet mask'></div>";
  html += "<div class='row'><label for='dns'>DNS</label>"
          "<input id='dns' name='dns' type='text' value='" + htmlEscape(dnsStr) + "' placeholder='(optional, defaults to gateway)' title='DNS server address'></div>";
  html += "<div class='row'><label for='auser'>Access Username</label>"
          "<input id='auser' name='auser' type='text' value='" + htmlEscape(g_authUser) + "' placeholder='(default admin)' title='Username for web access protection'>"
          "</div>";
  html += "<div class='row'><label for='apass'>Access Password</label>"
          "<input id='apass' name='apass' type='password' placeholder='" + String(isAuthEnabled()?"(leave empty to keep current)":"(set to enable protection)") + "' title='Set or change web access password'>"
          "<button type='button' class='btn' title='Show or hide password' onclick=\"const a=document.getElementById('apass');a.type=a.type==='password'?'text':'password'\">Show/Hide</button>"
          "</div>";
  // Explicit clear option to avoid accidental resets when changing Wi-Fi
  html += "<div class='row'><label for='aclear'>Clear Password</label>"
          "<input id='aclear' name='aclear' type='checkbox' title='Remove stored web access password'>"
          "<span class='hint'>(check to remove credentials)</span>"
          "</div>";
  // Token (read-only)
  {
    String tok = getAuthTokenParam();
    html += "<div class='row'><label for='atok'>Stream Token</label>";
    html += "<input id='atok' type='text' readonly value='" + htmlEscape(tok) + "' placeholder='(generated from user:pass)' title='Base64 stream token derived from credentials'>";
    html += "<button type='button' class='btn' title='Copy stream token' onclick=\"(function(){var el=document.getElementById('atok');el.focus();el.select();try{document.execCommand('copy');}catch(e){}})()\">Copy</button>";
    html += "</div>";
  }
  // Current stream URL (clickable)
  {
    IPAddress ip = (WiFi.status() == WL_CONNECTED) ? WiFi.localIP() : WiFi.softAPIP();
    String url = String("http://") + ip.toString() + ":81/stream";
    String tok = getAuthTokenParam();
    if (isAuthEnabled() && tok.length()>0) url += "?token=" + tok;
    html += "<div class='row'><label>Stream URL</label>";
    html += "<a class='btn' href='" + htmlEscape(url) + "' target='_blank' title='Open the MJPEG stream'>Open Stream</a>";
    html += "<input type='text' readonly value='" + htmlEscape(url) + "' style='flex:1 1 320px' title='Direct stream URL'>";
    html += "</div>";
  }
  if (isAuthEnabled()) html += "<div class='row hint'>Auth <b>enabled</b>. Browser prompts via Basic Auth. Also supports ?token=…</div>";
  else html += "<div class='row hint'>Auth <b>disabled</b>. Set credentials to protect the camera.</div>";
  html += "<div class='right'><button class='btn ok' type='submit' title='Save settings and reboot to apply'>Save &amp; Reboot</button></div>";
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
        html += tag;
        if (i != 0) {
          // Inline form to select this entry as active
          html += "<form method='POST' action='/wifi/select' style='margin:0'>";
          html += String("<input type='hidden' name='sel' value='") + String(i) + String("'>");
          html += "<button class='btn' type='submit' title='Move this network to the top and reboot'>Make Active</button>";
          html += "</form>";
        }
        // Inline form to delete this entry
        html += "<form method='POST' action='/wifi/delete' style='margin:0'>";
        html += String("<input type='hidden' name='del' value='") + String(i) + String("'>");
        html += "<button class='btn danger' type='submit' title='Remove this saved network'>Remove</button>";
        html += "</form>";
        html += "</div>";
      }
      html += "<div class='row hint'>Selecting a network moves it to the top and reboots, connecting to it on startup.</div>";
    }
    html += "</div>"; // panel
  }

  // Panel: Actions
  html += "<div class='panel'><h3>Actions</h3>";
  html += "<div class='row'>"
          "<a class='btn' href='/cam' title='Go to camera controls'>Open Camera</a>"
          "<a class='btn danger' href='/wifi/reboot' title='Reboot the device'>Reboot</a>"
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

  // Delete an MRU entry
  g_server.on("/wifi/delete", HTTP_POST, [](AsyncWebServerRequest* req){
    if (!isAuthorizedOrAP(req)) {
      AsyncWebServerResponse* r = req->beginResponse(401, "text/plain", "Unauthorized");
      r->addHeader("WWW-Authenticate", "Basic realm=\"ESP32Cam\"");
      req->send(r);
      return;
    }
    if (!req->hasParam("del", true)) {
      req->send(400, "text/plain", "Missing selection");
      return;
    }
    int idx = req->getParam("del", true)->value().toInt();
    String ssids[MRU_MAX]; String passes[MRU_MAX]; int n = 0;
    loadMRU(ssids, passes, n);
    if (idx < 0 || idx >= n) {
      req->send(400, "text/plain", "Invalid selection");
      return;
    }
    mruDeleteIndex(ssids, passes, n, idx);
    saveMRUList(ssids, passes, n);

    String html;
    html.reserve(2000);
    html += "<!doctype html><html><head><meta charset='utf-8'><title>Removed...</title>";
    html += kStyle;
    html += "<meta http-equiv='refresh' content='1;url=/wifi'>";
    html += "</head><body><div class='wrap'>";
    html += "<div class='nav'><a class='btn' href='/cam'>Camera</a></div>";
    html += "<div class='panel'><h3>Removed</h3><div class='row'><span class='hint'>Network removed from saved list.</span></div></div>";
    html += "</div></body></html>";
    req->send(200, "text/html", html);
  });

  // About page
  g_server.on("/about", HTTP_GET, [](AsyncWebServerRequest* req){
    if (!isAuthorizedOrAP(req)) {
      AsyncWebServerResponse* r = req->beginResponse(401, "text/plain", "Unauthorized");
      r->addHeader("WWW-Authenticate", "Basic realm=\"ESP32Cam\"");
      req->send(r);
      return;
    }
    String html;
    html.reserve(4000);
    html += "<!doctype html><html><head><meta charset='utf-8'><title>About</title>";
    html += kStyle;
    html += "</head><body><div class='wrap'>";
    html += "<div class='nav'>"
            "<a class='btn' href='/wifi' title='Edit Wi-Fi and access settings'>Wi-Fi Settings</a>"
            "<a class='btn' href='/cam' title='Open camera controls'>Camera</a>"
            "</div>";
    html += "<div class='panel'><h3>About</h3>";
    html += "<div class='row hint' title='License statement'>This firmware UI is provided under a Creative Commons Attribution-NonCommercial-ShareAlike license.</div>";
    html += "<div class='row hint' title='Attribution'>CC Richard Beech 2023-2026</div>";
    html += "<div class='row hint' title='Github Repository'>Github: <a href='https://github.com/1larity/ESP32_CAM_AI' target='_blank'>https://github.com/1larity/ESP32_CAM_AI</a></div>";
    html += "<div class='row hint' title='Non-commercial clause'>You may share and adapt for non-commercial purposes with attribution and share-alike.</div>";
    html += "</div>";
    html += "</div></body></html>";
    req->send(200, "text/html", html);
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
