#include "Utils.h"
#include <ESP32Servo.h>
#include "soc/soc.h"
#include "soc/rtc_cntl_reg.h"

extern Servo servo1;
extern Servo servo2;

void disableBrownout() {
  WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0); // Disable brownout detector
}

void setupServos() {
  servo1.setPeriodHertz(50); // 50Hz standard servo
  servo2.setPeriodHertz(50);
}
