Plan:

1. Clarify goals

* Multi-camera MDI app on PC with zoom/pan, “Fit” and “100%”.
* Real-time overlays: people, dogs, cats, faces, named pets.
* Pre-event recording buffer per camera.
* PTZ control to ESP32-CAM servos. Motion tracking later.

2. Triage current breakages

* Fix numpy/Qt bug: replace deprecated rgb.stride(0) with rgb.strides[0] anywhere it still exists and ensure QImage uses bytesPerLine = channels * width.
* Restore missing panels: Tools, Log, Recognition toggles. Re-enable their actions and signals. Make them per-camera and optional in a global View menu.
* Correct fit-to-window: resize only the camera viewport widget, not the QMdiArea or MDI subwindow.

3. Video pipeline hardening

* Keep a clean BGR for saving and a copy for overlays.
* Decouple network decode from UI thread via a reader thread with a bounded deque prebuffer.
* Compute FPS from timestamps, cap writer FPS, and pad to even dimensions for encoders.

4. Zoom and pan UX

* Ctrl+wheel = zoom in/out about cursor. Clamp 10–400%.
* Left-drag = pan when zoom > Fit/100%.
* Buttons: Fit, 100%, +, −. Status text shows zoom %, pan offset, FPS.
* Per-window state persists across sessions.

5. Overlay renderer

* Draw detections and labels after scaling with zoom/pan transform.
* Show dog ID when score passes threshold and IOU matches dog box.
* Toggle overlays: YOLO, Faces, Dog ID, Tracks.
* Render order: boxes → labels → crosshair for current aim.

6. Detection loop

* Worker thread runs at fixed interval (e.g., 8–12 Hz) on the latest frame snapshot.
* YOLO: CPU OpenCV DNN, configurable model path and thresholds.
* Faces: Haar cascade + LBPH if available, ORB fallback otherwise.
* Pets: ORB matcher on species folders.
* Emit results to the UI with timestamps for de-stale checks.

7. Enrollment flows

* Face enroll: collect N samples from tracked face with debounce. Progress dialog shows collected/target count. Auto-train after completion.
* Pet enroll: pick largest dog/cat box, collect N samples. Auto-load into DB.

8. Presence and event log

* Simple tracker with TTL. Enter/exit events with hysteresis.
* Event sidebar per camera with last N events and a “Open recordings folder” action.
* Write plain-text JSONL logs per camera.

9. Recording with prebuffer

* Start Rec: flush prebuffer to file then append live frames.
* Stop Rec: close writer cleanly. File naming: camName_host_YYYYMMDD_HHMMSS.ext
* AVI/MJPG primary, MP4/mp4v fallback.

10. PTZ plumbing

* UI buttons: Up/Down/Left/Right. Optional keyboard arrows when a camera window focused.
* Aim timer: steer toward last tracked primary target center with dead-zone and rate limit.
* Configurable mapping from screen offset to servo delta per tick.

11. Settings and persistence

* Global JSON for defaults (models, thresholds, detection interval).
* Per-camera JSON (name, host, auth, last window geometry, zoom/pan, enabled overlays).
* Safe load/save with versioning.

12. ESP32-CAM firmware alignment

* Keep /stream on port 81 with Basic or token auth.
* Ensure /action PTZ endpoints exist and match UI.
* Confirm /api/status returns resolution and IP for diagnostics.

13. Testing checklist

* Single camera: connect, zoom/pan, overlays visible, enroll face/pet, record with prebuffer.
* Two to four cameras: CPU load, UI responsiveness, independent settings, close/reopen windows.
* Auth permutations: token, user/pass, no auth.
* Error paths: stream drop, reconnection, writer failure, model missing.

14. Performance guardrails

* Drop frames in UI path when busy. Detection thread always reads last frame only.
* Optional downscale for detector while rendering full-res.
* Timing overlay to display decode ms, detect ms, and render ms.

15. Deliverables

* Updated mdi_app.py with modular CameraWidget and restored panels.
* helpers: tools.py, gallery dialog retained. Config and models folder structure.
* README with setup, models download, and workflow for enrollment.

 implement in this order: (a) bugfixes + restored panels, (b) zoom/pan + fit/100%, (c) stable recording with prebuffer, (d) detector thread + overlays, (e) enrollment and presence log, (f) PTZ aim loop, (g) settings persistence.
 check work so far for bugs then continue plan
