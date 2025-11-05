#include "Utils.h"
#include "soc/soc.h"
#include "soc/rtc_cntl_reg.h"



void disableBrownout() {
  // The ESP32-CAM is sensitive to brief voltage drops when peripherals
  // such as the camera or servos draw peak current. Those dips can
  // trigger the on-chip brownout detector and cause an unexpected
  // reset, so the detector is disabled to keep the device running.
  WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0);
}
