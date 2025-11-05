#pragma once
#include <Arduino.h>

class AsyncWebServer;

// Initialise servos, load persisted home/current angles, move to home.
void ptzInit();

// Register all PTZ HTTP routes on the shared AsyncWebServer.
// Safe to call multiple times; it ensures the server exists.
void ptzRegisterRoutes(AsyncWebServer& srv);

// Set absolute angles (0..180). Returns false if clamped or unchanged.
bool ptzSet(int panDeg, int tiltDeg);

// Step by deltas (can be negative). Returns final angles applied.
bool ptzStep(int dPan, int dTilt);

// Reset to persisted home.
void ptzHome();

// Read current angles.
void ptzGet(int& panOut, int& tiltOut);

// Persist the provided angles as new home.
void ptzSaveHome(int panDeg, int tiltDeg);
