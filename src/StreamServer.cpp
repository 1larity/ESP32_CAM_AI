#include "StreamServer.h"
#include "esp_camera.h"
#include "esp_http_server.h"

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

    if (fb->format != PIXFORMAT_JPEG) {
      // Convert frame buffer to JPEG if not already
      bool jpeg_converted = frame2jpg(fb, 80, (uint8_t**)&fb->buf, &fb->len);
      if (!jpeg_converted) {
        esp_camera_fb_return(fb);
        res = ESP_FAIL;
        break;
      }
    }

    // Send the JPEG as a multipart HTTP response:
    // first the part header, then the image data, then the boundary marker
    size_t hlen = snprintf(part_buf, sizeof(part_buf), _STREAM_PART, fb->len);
    res = httpd_resp_send_chunk(req, part_buf, hlen);
    if (res == ESP_OK)
      res = httpd_resp_send_chunk(req, (const char*)fb->buf, fb->len);
    if (res == ESP_OK)
      res = httpd_resp_send_chunk(req, _STREAM_BOUNDARY, strlen(_STREAM_BOUNDARY));

    esp_camera_fb_return(fb);
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
