#ifndef STREAM_SERVER_H
#define STREAM_SERVER_H

// Opaque API: avoid pulling esp_http_server.h into other translation units
void startStreamServer();
void stopStreamServer();
bool isStreamServerRunning();

#endif
