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
    String html = "<html><body><h2>ESP32-CAM Control</h2>";
    html += "<img src='http://" + WiFi.localIP().toString() + ":81/stream' width='100%'><br>";
    html += "<button onclick=\"fetch('/action?go=up')\">Up</button> ";
    html += "<button onclick=\"fetch('/action?go=down')\">Down</button><br>";
    html += "<button onclick=\"fetch('/action?go=left')\">Left</button> ";
    html += "<button onclick=\"fetch('/action?go=right')\">Right</button>";
    html += "</body></html>";
    request->send(200, "text/html", html);
  });

  camServer.on("/action", HTTP_GET, [](AsyncWebServerRequest *request){
    if (request->hasParam("go")) {
      String dir = request->getParam("go")->value();
      if (dir == "up" && servo1Pos < 180) servo1Pos += 10;
      else if (dir == "down" && servo1Pos > 0) servo1Pos -= 10;
      else if (dir == "left" && servo2Pos < 180) servo2Pos += 10;
      else if (dir == "right" && servo2Pos > 0) servo2Pos -= 10;
      servo1.write(servo1Pos);
      servo2.write(servo2Pos);
    }
    request->send(200, "text/plain", "OK");
  });

  camServer.begin();
}

void setupCamera() {
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = 5;
  config.pin_d1 = 18;
  config.pin_d2 = 19;
  config.pin_d3 = 21;
  config.pin_d4 = 36;
  config.pin_d5 = 39;
  config.pin_d6 = 34;
  config.pin_d7 = 35;
  config.pin_xclk = 0;
  config.pin_pclk = 22;
  config.pin_vsync = 25;
  config.pin_href = 23;
  config.pin_sccb_sda = 26;
  config.pin_sccb_scl = 27;
  config.pin_pwdn = 32;
  config.pin_reset = -1;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;

  if(psramFound()){
    config.frame_size = FRAMESIZE_VGA;
    config.jpeg_quality = 10;
    config.fb_count = 2;
  } else {
    config.frame_size = FRAMESIZE_CIF;
    config.jpeg_quality = 12;
    config.fb_count = 1;
  }

  esp_camera_init(&config);

  servo1.setPeriodHertz(50);
  servo2.setPeriodHertz(50);
  servo1.attach(SERVO_1, 1000, 2000);
  servo2.attach(SERVO_2, 1000, 2000);
  servo1.write(servo1Pos);
  servo2.write(servo2Pos);

  startStreamServer();
}
