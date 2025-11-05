#ifndef UTILS_H
#define UTILS_H

#include <ESP32Servo.h>

/**
 * Globals for servo control. Defined in Utils.cpp to guarantee
 * construction order before any call to setupServos().
 */
extern Servo servo1;
extern Servo servo2;

/**
 * Disable the ESP32's brownout detector to prevent unwanted resets
 * during brief voltage dips caused by high current draw.
 */
void disableBrownout();

#endif
