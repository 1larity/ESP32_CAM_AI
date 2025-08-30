ESP32_CAM_AI

Overview

ESP32_CAM_AI is a small end‑to‑end project for streaming and analyzing video from an ESP32‑CAM (OV2640) module. It includes:

- ESP32 firmware with a clean web UI (port 80), MJPEG stream server (port 81), PTZ controls, and Wi‑Fi/auth management.
- A desktop viewer app (PySide6/Qt) that can display multiple cameras, record clips, and optionally run lightweight CV (YOLO object/face/pet recognition).

License

License: CC BY-NC 4.0 https://creativecommons.org/licenses/by-nc/4.0/

This project is licensed under the Creative Commons Attribution–NonCommercial 4.0 International (CC BY‑NC 4.0) license.

- You may share and adapt the material for non‑commercial purposes, as long as you give appropriate credit and indicate if changes were made.
- Commercial use is not permitted without prior written permission.

Full license text and human‑readable summary: https://creativecommons.org/licenses/by-nc/4.0/

Attribution

When using this project, please attribute with a link back to this repository (e.g., “Built with ESP32_CAM_AI: https://github.com/1larity/ESP32_CAM_AI”).



Key Features

- MJPEG streaming on port 81 with a reference, widely compatible framing.
- Built‑in snapshot endpoint for quick diagnostics.
- Web UI (port 80) with resolution control, PTZ (two servos), and Wi‑Fi/auth pages.
- Basic Auth with hashed storage; optional token parameter for cross‑port embed.
- Desktop MDI viewer with pre‑buffered recording and optional overlays.

Hardware

- ESP32‑CAM board (OV2640 sensor).
- Optional two servos for PTZ: pins 14 and 15 (see `src/CameraServer.cpp`).

Build & Flash

- Tooling: PlatformIO
- Environment: `esp32cam` (see `platformio.ini`)


Stream Quickstart

- MJPEG stream (port 81):
  - No auth: `http://<ip>:81/stream`
  - Basic Auth (curl): `curl -v -u user:pass http://<ip>:81/stream`
- Snapshot (single JPEG): `http://<ip>:81/snap`
- Web UI (with embedded live video): `http://<ip>/`
- Wi‑Fi/Auth settings: `http://<ip>/wifi`

Desktop Viewer (MDI)

- Title Bar: Each camera subwindow shows `Name [IP]` and appends `(REC)` while recording.
- AI Controls: YOLO and Face toggles live under the camera toolbar’s `AI` dropdown menu.
- Recording: Defaults to AVI/MJPG on Windows for reliability; falls back to MP4V if needed. Frames are resized to even dimensions automatically.
- Camera Persistence: The app de‑duplicates cameras by host (IP).
  - Add/Load/Save remove duplicates by IP; closing a camera removes it by IP.
- Tools menu:
  - Scan For Cameras: Probes the local /24 for `http://<ip>/api/advertise` and adds new devices.
  - Manage Cameras: View/remove cameras from the saved set.

Status API
- Endpoint: `GET http://<ip>/api/status`
- Returns: JSON with device/network and current camera resolution.
- Example response:
  `{"ip":"192.168.1.130","resolution":"VGA","width":640,"height":480}`
- Notes:
  - Served by the main web UI (port 80), follows the same auth rules (Basic or cookie).
  - Useful for the desktop app to discover stream size before recording/overlays.

Auth & Token
- When auth is enabled, the page embeds the stream with `?token=<Base64(user:pass)>` so the `<img>` can load across ports.
- Token is generated when you save credentials on `/wifi` and is shown read‑only on that page.
- Token helpers:
  - PowerShell: `[Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes('user:pass'))`
  - Bash: `echo -n 'user:pass' | base64`
  - Python: `import base64; print(base64.b64encode(b'user:pass').decode())`

Troubleshooting
- If the page video doesn’t load with auth enabled, confirm the Stream Token is non‑empty on `/wifi` and try the “Open Stream” link shown there.
- Test the camera path via `http://<ip>:81/snap`.
- Use curl to inspect HTTP status and headers if available.
- Reduce resolution (VGA/SVGA) if Wi‑Fi bandwidth causes stalls.
- Power‑cycle the device after flashing major changes.

Repo Structure
- Firmware (PlatformIO):
  - `src/StreamServer.cpp` — MJPEG stream and snapshot (port 81)
  - `src/CameraServer.cpp` — Web UI (port 80), PTZ, resolution control
  - `src/WiFiManager.cpp|.h` — Wi‑Fi, auth, token handling, UI
- Desktop Viewer:
  - `AI/mdi_app.py` — PySide6 MDI viewer; optional CV via OpenCV/ONNX
    - Tools → Scan For Cameras (uses `/api/advertise`)
    - Tools → Manage Cameras (remove from saved set)
    - Camera toolbar → AI dropdown (YOLO/Face)

