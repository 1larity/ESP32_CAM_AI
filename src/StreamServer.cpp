#include "StreamServer.h"
#include "esp_camera.h"
#include "esp_http_server.h"
#include <string.h>

static httpd_handle_t stream_httpd = NULL;  // keep handle private here

static const char* _STREAM_CONTENT_TYPE = "multipart/x-mixed-replace; boundary=frame";
static const char* _STREAM_BOUNDARY     = "\r\n--frame\r\n";
static const char* _STREAM_PART         = "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n";

static esp_err_t stream_handler(httpd_req_t *req) {
  camera_fb_t *fb = NULL;

  esp_err_t res = httpd_resp_set_type(req, _STREAM_CONTENT_TYPE);
  if (res != ESP_OK) return res;

  // initial boundary
  res = httpd_resp_send_chunk(req, _STREAM_BOUNDARY, strlen(_STREAM_BOUNDARY));
  if (res != ESP_OK) return res;

  char part_buf[64];

  while (true) {
    fb = esp_camera_fb_get();
    if (!fb) { res = ESP_FAIL; break; }

    if (fb->format != PIXFORMAT_JPEG) {
      uint8_t *jpg_buf = nullptr;
      size_t   jpg_len = 0;
      bool ok = frame2jpg(fb, 80, &jpg_buf, &jpg_len);
      esp_camera_fb_return(fb);
      fb = nullptr;
      if (!ok || !jpg_buf) { res = ESP_FAIL; break; }

      size_t hlen = snprintf(part_buf, sizeof(part_buf), _STREAM_PART, (unsigned)jpg_len);
      res = httpd_resp_send_chunk(req, part_buf, hlen);
      if (res == ESP_OK) res = httpd_resp_send_chunk(req, (const char*)jpg_buf, jpg_len);
      free(jpg_buf);
    } else {
      size_t hlen = snprintf(part_buf, sizeof(part_buf), _STREAM_PART, (unsigned)fb->len);
      res = httpd_resp_send_chunk(req, part_buf, hlen);
      if (res == ESP_OK) res = httpd_resp_send_chunk(req, (const char*)fb->buf, fb->len);
      esp_camera_fb_return(fb);
      fb = nullptr;
    }

    if (res == ESP_OK)
      res = httpd_resp_send_chunk(req, _STREAM_BOUNDARY, strlen(_STREAM_BOUNDARY));

    if (res != ESP_OK) break;
    vTaskDelay(1);
  }

  if (fb) esp_camera_fb_return(fb);
  return res;
}

void startStreamServer() {
  if (stream_httpd) return;

  httpd_config_t config = HTTPD_DEFAULT_CONFIG();
  config.server_port = 81;

  if (httpd_start(&stream_httpd, &config) == ESP_OK) {
    httpd_uri_t stream_uri = {
      .uri      = "/stream",
      .method   = HTTP_GET,
      .handler  = stream_handler,
      .user_ctx = NULL
    };
    httpd_register_uri_handler(stream_httpd, &stream_uri);
  }
}

void stopStreamServer() {
  if (stream_httpd) {
    httpd_stop(stream_httpd);
    stream_httpd = NULL;
  }
}

bool isStreamServerRunning() {
  return stream_httpd != NULL;
}
