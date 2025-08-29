#include "StreamServer.h"
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

  // Tell the client to expect multipart MJPEG stream
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
  }
}
