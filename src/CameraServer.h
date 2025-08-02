#ifndef CAMERA_SERVER_H
#define CAMERA_SERVER_H

// Configures camera hardware, PSRAM settings, and servo motors
void setupCamera();

// Launches the web interface for streaming and controlling servos
void startCameraServer();

#endif
