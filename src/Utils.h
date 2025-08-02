#ifndef UTILS_H
#define UTILS_H

/**
 * Disable the ESP32's brownout detector to prevent unwanted resets
 * during brief voltage dips caused by high current draw.
 */
void disableBrownout();

/**
 * Configure the servos to run at a 50 Hz PWM frequency, the standard
 * refresh rate for most hobby servos.
 */
void setupServos();

#endif
