#pragma once
#include <stdint.h>

void startRtspServer(uint16_t port);
void stopRtspServer();
bool isRtspRunning();
