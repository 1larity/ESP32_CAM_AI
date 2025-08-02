#include "Utils.h"
#include <ESP32Servo.h>
#include "soc/soc.h"
#include "soc/rtc_cntl_reg.h"

extern Servo servo1;
extern Servo servo2;

void disableBrownout() {
  // The ESP32-CAM is sensitive to brief voltage drops when peripherals
  // such as the camera or servos draw peak current. Those dips can
  // trigger the on-chip brownout detector and cause an unexpected
  // reset, so the detector is disabled to keep the device running.
  WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0);
}

void setupServos() {
  // Standard hobby servos expect a 20ms refresh period (50 Hz). Using
  // this rate provides full range of motion and avoids jitter.
  servo1.setPeriodHertz(50);
  servo2.setPeriodHertz(50);
}
