#include "RtspServer.h"
#include "esp_camera.h"
#include <WiFi.h>
#include <lwip/sockets.h>
#include <lwip/netdb.h>
#include <string.h>
#include <strings.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include <esp_system.h>
#include <esp_timer.h>
#include <unistd.h>

// Minimal RTSP MJPEG server (RFC2435-ish) for VLC compatibility.
// Single client, RTP over TCP (interleaved). No auth. One packet per frame.

static TaskHandle_t s_task = nullptr;
static int s_server_fd = -1;
static bool s_running = false;
static uint16_t s_port = 8554;
static uint32_t s_ssrc = 0;
static uint16_t s_seq = 1;
static uint32_t s_ts = 0;
static uint32_t s_session = 0;

static ssize_t send_all(int fd, const void* buf, size_t len) {
  const uint8_t* p = (const uint8_t*)buf;
  size_t sent = 0;
  while (sent < len) {
    int n = send(fd, p + sent, len - sent, 0);
    if (n <= 0) return n;
    sent += n;
  }
  return (ssize_t)sent;
}

static bool recv_request(int fd, char* out, size_t out_len) {
  size_t off = 0;
  while (off + 1 < out_len) {
    int n = recv(fd, out + off, 1, 0);
    if (n <= 0) return false;
    off += n;
    if (off >= 4 && strncmp(out + off - 4, "\r\n\r\n", 4) == 0) {
      out[off] = 0;
      return true;
    }
  }
  out[out_len - 1] = 0;
  return true;
}

static const char* find_hdr(const char* req, const char* key) {
  const char* p = strcasestr(req, key);
  if (!p) return nullptr;
  p += strlen(key);
  while (*p == ' ' || *p == ':') p++;
  return p;
}

static void send_rtsp(int fd, const char* cseq, const char* body, const char* extra) {
  char resp[512];
  size_t n = 0;
  n += snprintf(resp + n, sizeof(resp) - n, "RTSP/1.0 200 OK\r\n");
  n += snprintf(resp + n, sizeof(resp) - n, "CSeq: %s\r\n", cseq ? cseq : "1");
  n += snprintf(resp + n, sizeof(resp) - n, "Session: %u\r\n", (unsigned)s_session);
  if (extra && *extra) n += snprintf(resp + n, sizeof(resp) - n, "%s", extra);
  if (body) {
    n += snprintf(resp + n, sizeof(resp) - n, "Content-Type: application/sdp\r\n");
    n += snprintf(resp + n, sizeof(resp) - n, "Content-Length: %u\r\n", (unsigned)strlen(body));
  }
  n += snprintf(resp + n, sizeof(resp) - n, "\r\n");
  send_all(fd, resp, n);
  if (body) send_all(fd, body, strlen(body));
}

static void stream_loop(int fd) {
  s_seq = 1; s_ts = 0; if (s_ssrc == 0) s_ssrc = esp_random();
  while (s_running) {
    camera_fb_t* fb = esp_camera_fb_get();
    if (!fb) break;
    uint16_t w = fb->width;
    uint16_t h = fb->height;

    // RTP interleaved header (2 bytes ch + 2 bytes length)
    size_t mjpeg_hdr_len = 8; // RFC2435 main header without qtables
    size_t rtp_len = 12 + mjpeg_hdr_len + fb->len;
    uint8_t pre[4 + 12 + 8];
    pre[0] = '$'; pre[1] = 0; pre[2] = (rtp_len >> 8) & 0xFF; pre[3] = rtp_len & 0xFF;
    // RTP header
    pre[4] = 0x80;
    pre[5] = 0x80 | 26; // M=1, PT=26 JPEG
    pre[6] = (s_seq >> 8) & 0xFF; pre[7] = s_seq & 0xFF; s_seq++;
    pre[8] = (s_ts >> 24) & 0xFF; pre[9] = (s_ts >> 16) & 0xFF; pre[10] = (s_ts >> 8) & 0xFF; pre[11] = s_ts & 0xFF; s_ts += 9000; // ~10 fps
    pre[12] = (s_ssrc >> 24) & 0xFF; pre[13] = (s_ssrc >> 16) & 0xFF; pre[14] = (s_ssrc >> 8) & 0xFF; pre[15] = s_ssrc & 0xFF;
    // JPEG payload header (no fragmentation, type 1 baseline)
    pre[16] = 0; pre[17] = 0; pre[18] = 0; // type-specific + frag offset (24-bit)
    pre[19] = 0; // frag offset low byte
    pre[20] = 1; // type
    pre[21] = 255; // Q-factor (use internal tables)
    pre[22] = (uint8_t)(w / 8);
    pre[23] = (uint8_t)(h / 8);

    if (!s_running) { esp_camera_fb_return(fb); break; }
    if (send_all(fd, pre, sizeof(pre)) <= 0) { esp_camera_fb_return(fb); break; }
    if (send_all(fd, fb->buf, fb->len) <= 0) { esp_camera_fb_return(fb); break; }
    esp_camera_fb_return(fb);
  }
}

static void handle_client(int fd) {
  char req[1024];
  bool setup = false;
  uint16_t w = 640, h = 480;
  if (s_ssrc == 0) s_ssrc = esp_random();
  if (s_session == 0) s_session = esp_random();
  String ip = WiFi.localIP().toString();
  char sdp[256];
  snprintf(sdp, sizeof(sdp),
    "v=0\r\n"
    "o=- 0 0 IN IP4 %s\r\n"
    "s=ESP32-CAM\r\n"
    "c=IN IP4 %s\r\n"
    "t=0 0\r\n"
    "m=video 0 RTP/AVP 26\r\n"
    "a=rtpmap:26 JPEG/90000\r\n"
    "a=control:trackID=1\r\n",
    ip.c_str(), ip.c_str());

  while (s_running) {
    if (!recv_request(fd, req, sizeof(req))) break;
    const char* cseq = find_hdr(req, "CSeq");
    if (strstr(req, "OPTIONS")) {
      send_rtsp(fd, cseq, nullptr, "Public: OPTIONS, DESCRIBE, SETUP, PLAY, TEARDOWN\r\n");
    } else if (strstr(req, "DESCRIBE")) {
      char base[128];
      snprintf(base, sizeof(base), "Content-Base: rtsp://%s:%u/\r\n", ip.c_str(), (unsigned)s_port);
      send_rtsp(fd, cseq, sdp, base);
    } else if (strstr(req, "SETUP")) {
      setup = true;
      const char* extra = "Transport: RTP/AVP/TCP;unicast;interleaved=0-1\r\n";
      send_rtsp(fd, cseq, nullptr, extra);
    } else if (strstr(req, "PLAY")) {
      if (!setup) { send_rtsp(fd, cseq, nullptr, nullptr); continue; }
      char rinfo[128];
      snprintf(rinfo, sizeof(rinfo), "RTP-Info: url=rtsp://%s:%u/trackID=1;seq=1;rtptime=0\r\n", ip.c_str(), (unsigned)s_port);
      send_rtsp(fd, cseq, nullptr, rinfo);
      stream_loop(fd);
      break;
    } else if (strstr(req, "TEARDOWN")) {
      send_rtsp(fd, cseq, nullptr, nullptr);
      break;
    } else {
      send_rtsp(fd, cseq, nullptr, nullptr);
    }
  }
}

static void rtsp_task(void* arg) {
  (void)arg;
  s_server_fd = socket(AF_INET, SOCK_STREAM, IPPROTO_IP);
  if (s_server_fd < 0) { s_running = false; vTaskDelete(nullptr); return; }
  int yes = 1; setsockopt(s_server_fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
  struct sockaddr_in addr; memset(&addr, 0, sizeof(addr)); addr.sin_family = AF_INET; addr.sin_port = htons(s_port); addr.sin_addr.s_addr = htonl(INADDR_ANY);
  if (bind(s_server_fd, (struct sockaddr*)&addr, sizeof(addr)) != 0) { close(s_server_fd); s_server_fd=-1; s_running=false; vTaskDelete(nullptr); return; }
  listen(s_server_fd, 1);
  while (s_running) {
    struct sockaddr_in raddr; socklen_t rlen = sizeof(raddr);
    int cfd = accept(s_server_fd, (struct sockaddr*)&raddr, &rlen);
    if (cfd < 0) continue;
    handle_client(cfd);
    close(cfd);
  }
  if (s_server_fd >= 0) { close(s_server_fd); s_server_fd = -1; }
  s_running = false;
  s_task = nullptr;
  vTaskDelete(nullptr);
}

void startRtspServer(uint16_t port) {
  stopRtspServer();
  s_port = port;
  s_running = true;
  s_session = esp_random();
  xTaskCreatePinnedToCore(rtsp_task, "rtsp", 4096, nullptr, 3, &s_task, 1);
}

void stopRtspServer() {
  s_running = false;
  if (s_server_fd >= 0) { close(s_server_fd); s_server_fd = -1; }
  if (s_task) { vTaskDelete(s_task); s_task = nullptr; }
}

bool isRtspRunning() {
  return s_running;
}
