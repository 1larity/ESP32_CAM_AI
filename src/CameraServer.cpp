// esp32cam_main.cpp
#include "WiFiManager.h"
#include "CameraServer.h"
#include "OTAHandler.h"
#include "Utils.h"
#include "esp_camera.h"
#include <WiFi.h>
#include <ESPAsyncWebServer.h>
#include <ESP32Servo.h>


#include "StreamServer.h"

#define SERVO_1      14
#define SERVO_2      15

Servo servo1, servo2;
int servo1Pos = 90;
int servo2Pos = 90;

AsyncWebServer camServer(80);

void startCameraServer() {
  camServer.on("/", HTTP_GET, [](AsyncWebServerRequest *request){
    // Serve a simple page with buttons that tilt and pan the camera
    String html = "<html><body><h2>ESP32-CAM Control</h2>";
    html += "<button onclick=\"fetch('/action?go=up')\">Up</button> ";
    html += "<button onclick=\"fetch('/action?go=down')\">Down</button><br>";
    html += "<button onclick=\"fetch('/action?go=left')\">Left</button> ";
    html += "<button onclick=\"fetch('/action?go=right')\">Right</button><br><br>";
    html += "<img src='http://" + WiFi.localIP().toString() + ":81/stream' width='80%'><br>";

    html += "</body></html>";
    request->send(200, "text/html", html);
  });

  camServer.on("/action", HTTP_GET, [](AsyncWebServerRequest *request){
    // Adjust servo positions according to the requested direction
    if (request->hasParam("go")) {
      String dir = request->getParam("go")->value();
      if (dir == "up" && servo1Pos < 180) servo1Pos += 10;        // tilt up
      else if (dir == "down" && servo1Pos > 0) servo1Pos -= 10;    // tilt down
      else if (dir == "left" && servo2Pos < 180) servo2Pos += 10;  // pan left
      else if (dir == "right" && servo2Pos > 0) servo2Pos -= 10;   // pan right
      servo1.write(servo1Pos); // apply new vertical angle
      servo2.write(servo2Pos); // apply new horizontal angle
    }
    request->send(200, "text/plain", "OK");
  });

  camServer.begin();
}

void setupCamera() {
  camera_config_t config;                                  // camera configuration structure
  config.ledc_channel = LEDC_CHANNEL_0;                    // timer channel for LEDC
  config.ledc_timer = LEDC_TIMER_0;                        // use timer 0 for LEDC
  config.pin_d0 = 5;                                       // data bit 0
  config.pin_d1 = 18;                                      // data bit 1
  config.pin_d2 = 19;                                      // data bit 2
  config.pin_d3 = 21;                                      // data bit 3
  config.pin_d4 = 36;                                      // data bit 4
  config.pin_d5 = 39;                                      // data bit 5
  config.pin_d6 = 34;                                      // data bit 6
  config.pin_d7 = 35;                                      // data bit 7
  config.pin_xclk = 0;                                    // external clock pin
  config.pin_pclk = 22;                                   // pixel clock pin
  config.pin_vsync = 25;                                  // vertical sync pin
  config.pin_href = 23;                                   // horizontal reference pin
  config.pin_sccb_sda = 26;                               // SCCB data pin
  config.pin_sccb_scl = 27;                               // SCCB clock pin
  config.pin_pwdn = 32;                                   // power-down pin
  config.pin_reset = -1;                                  // reset pin not used
  config.xclk_freq_hz = 20000000;                         // 20 MHz clock
  config.pixel_format = PIXFORMAT_JPEG;                   // JPEG output

  // Choose resolution and buffer count based on PSRAM availability
  if(psramFound()){
    config.frame_size = FRAMESIZE_VGA;                    // higher resolution with PSRAM
    config.jpeg_quality = 10;
    config.fb_count = 2;                                  // double buffer
  } else {
    config.frame_size = FRAMESIZE_CIF;                    // lower resolution without PSRAM
    config.jpeg_quality = 12;
    config.fb_count = 1;                                  // single buffer
  }

  esp_camera_init(&config);                               // initialize camera with above settings

  // Initialize servos for camera orientation
  servo1.setPeriodHertz(50);                              // standard 50 Hz servo frequency
  servo2.setPeriodHertz(50);
  servo1.attach(SERVO_1, 1000, 2000);                     // attach vertical servo
  servo2.attach(SERVO_2, 1000, 2000);                     // attach horizontal servo
  servo1.write(servo1Pos);                                // center vertical servo
  servo2.write(servo2Pos);                                // center horizontal servo

  startStreamServer();                                    // begin MJPEG stream server
}
