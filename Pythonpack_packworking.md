Primary goals Preserve stability and functionality unless requested.
Do not alter code unrelated to the task at hand.
Always provide the full file contents.
Try to keep files under 150 lines, if required create new files for new functionality or to delegate responsabilities.

Remove the translucent fill on the recognition frames.
Event view needs to have an autoscroll checkbox, when enabled, the display will follow the tail of newest events.
log events should be time/datestamped.
Even though i am correctly identified, that does not show up in the log. possibly over written by enter/exit person loglines.

# Project: AI

## Structure
```
cam discovery.py
camera_recorder.py
camera_widget.py
detection_packet.py
detectors/
  __init__.py
  config.py
  faces_backend.py
  lbph_model.py
  packet.py
  thread.py
  yolo_backend.py
discovery_dialog.py
enrollment.py
enrollment_service.py
events_pane.py
face_params.py
face_tuner.py
gallery.py
graphics_view.py
image_manager.py
ip_cam_dialog.py
mdi_app.py
models.py
overlays.py
presence.py
ptz.py
recorder.py
refactor/
  ai_viewer/
    __init__.py
    ai/
      __init__.py
      detectors/
        __init__.py
        yolo.py
      recognition/
        __init__.py
        face_db.py
        pets_db.py
    core/
      __init__.py
      config.py
      stream.py
      tracking.py
    tools/
      dupes.py
      gallery.py
    ui/
      __init__.py
      camera_widget.py
      windows.py
  run_ai_viewer.py
settings.py
src/
  app/
    main.py
  camera/
    detection.py
    stream.py
  ui/
    dialogs.py
    widgets.py
stream/
  __init__.py
  capture.py
  mjpeg_backend.py
  opencv_backend.py
tools.py
utils.py
```


## FILE: AI/cam discovery.py
```text
#!/usr/bin/env python3
"""
discover_cams.py — Discover ESP32-CAMs on the local LAN.

Strategy
- Determine local /24 automatically via a UDP socket trick (override with --cidr).
- Probe http://IP/ping (your firmware replies "pong").
- Optionally probe :81/stream to confirm MJPEG boundary and auth requirements.
- Concurrency with ThreadPoolExecutor. Safe timeouts.

Usage
  python discover_cams.py                # auto /24 from primary NIC
  python discover_cams.py --cidr 192.168.1.0/24
  python discover_cams.py --check-stream
  python discover_cams.py --workers 256 --timeout 0.6
"""
from __future__ import annotations
import argparse
import ipaddress
import socket
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests


def guess_primary_ipv4() -> str | None:
    """Pick the local IPv4 by opening a UDP socket to a public IP."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.2)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return None

def default_cidr() -> str:
    return "192.168.1.0/24"

# def default_cidr() -> str:
#     ip = guess_primary_ipv4()
#     if not ip:
#         return "192.168.1.0/24"
#     # assume /24
#     parts = ip.split(".")
#     return f"{parts[0]}.{parts[1]}.{parts[2]}.0/24"


def probe_host(ip: str, timeout: float, check_stream: bool, token: str | None) -> dict | None:
    base = f"http://{ip}"
    session = requests.Session()
    session.headers.update({"User-Agent": "ESP32-CAM-Discovery/1.0"})
    # 1) /ping
    try:
        r = session.get(f"{base}/ping", timeout=timeout)
        text = (r.text or "").strip().lower()
        if r.status_code == 200 and text.startswith("pong"):
            info = {
                "ip": ip,
                "ping": True,
                "auth": False,
                "stream_ok": None,
                "notes": "",
            }
            # try / (may 401 if auth on)
            try:
                r0 = session.get(base + "/", timeout=timeout)
                if r0.status_code == 401:
                    info["auth"] = True
                elif r0.ok:
                    # maybe capture title if present
                    t = r0.text.lower()
                    if "<title" in t:
                        info["notes"] = "web ui reachable"
            except Exception:
                pass

            # 2) :81/stream (optional)
            if check_stream:
                stream_url = f"http://{ip}:81/stream"
                if token:
                    sep = "&" if "?" in stream_url else "?"
                    stream_url = f"{stream_url}{sep}token={token}"
                try:
                    r1 = session.get(stream_url, stream=True, timeout=timeout)
                    if r1.status_code == 401:
                        info["stream_ok"] = False
                        info["auth"] = True
                    elif r1.ok:
                        ctype = r1.headers.get("Content-Type", "")
                        # read a small chunk to confirm boundary, then close
                        try:
                            next(r1.iter_content(chunk_size=512))
                        except Exception:
                            pass
                        info["stream_ok"] = ("multipart/x-mixed-replace" in ctype.lower())
                    else:
                        info["stream_ok"] = False
                except Exception:
                    info["stream_ok"] = False
            return info
    except requests.exceptions.RequestException:
        return None
    return None


def main():
    ap = argparse.ArgumentParser(description="Discover ESP32-CAM devices on LAN")
    ap.add_argument("--cidr", default=default_cidr(), help="CIDR to scan, e.g. 192.168.1.0/24")
    ap.add_argument("--workers", type=int, default=128, help="Max concurrent probes")
    ap.add_argument("--timeout", type=float, default=0.8, help="Per-request timeout seconds")
    ap.add_argument("--check-stream", action="store_true", help="Also probe :81/stream")
    ap.add_argument("--token", default=None, help="Optional Base64 user:pass token for /stream")
    args = ap.parse_args()

    try:
        net = ipaddress.ip_network(args.cidr, strict=False)
    except Exception as e:
        print(f"Invalid CIDR: {e}", file=sys.stderr)
        sys.exit(2)

    hosts = [str(ip) for ip in net.hosts()]
    print(f"Scanning {len(hosts)} hosts in {args.cidr}…")
    t0 = time.time()

    found = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(probe_host, ip, args.timeout, args.check_stream, args.token): ip
            for ip in hosts
        }
        for fut in as_completed(futs):
            res = fut.result()
            if res:
                found.append(res)

    dt = time.time() - t0
    if not found:
        print("No ESP32-CAMs found.")
        print(f"Done in {dt:.1f}s")
        return

    # Output table
    print("\nDiscovered devices:")
    print("{:<15} {:<6} {:<6} {}".format("IP", "PING", "STRM", "NOTES/AUTH"))
    print("-" * 54)
    for d in sorted(found, key=lambda x: x["ip"]):
        ping = "ok" if d["ping"] else "-"
        if d["stream_ok"] is None:
            strm = "-"
        else:
            strm = "ok" if d["stream_ok"] else "fail"
        notes = d["notes"]
        if d["auth"]:
            notes = (notes + " | auth").strip(" |")
        print("{:<15} {:<6} {:<6} {}".format(d["ip"], ping, strm, notes))

    print(f"\nDone in {dt:.1f}s")


if __name__ == "__main__":
    main()

```


## FILE: AI/camera_recorder.py
```text

"""
CameraRecorder: pre‑event buffered video recorder for PySide6 apps.

- Ring buffer of (ts_ms, frame_bgr) in RAM.
- On start(): flushes last N seconds then appends live frames.
- Background writer thread with target FPS and codec fallback.
- Safe write to .tmp then rename on stop().
"""

from __future__ import annotations
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Optional, Tuple, List

import numpy as np
import cv2


@dataclass
class RecorderConfig:
    out_dir: Path
    prebuffer_sec: int = 5
    fps: int = 20
    codec_primary: str = "MJPG"
    codec_fallback: str = "mp4v"
    max_ram_mb: int = 256  # soft cap for ring buffer


class CameraRecorder:
    def __init__(self, name: str, cfg: RecorderConfig):
        self.name = name
        self.cfg = cfg
        self.cfg.out_dir.mkdir(parents=True, exist_ok=True)
        (self.cfg.out_dir / ".tmp").mkdir(parents=True, exist_ok=True)

        self._ring: Deque[Tuple[int, np.ndarray]] = deque()
        self._ring_lock = threading.Lock()
        self._ring_bytes = 0

        self._writer = None
        self._writer_path_tmp: Optional[Path] = None
        self._writer_path_final: Optional[Path] = None
        self._writer_lock = threading.Lock()
        self._running = False
        self._write_thread: Optional[threading.Thread] = None
        self._q: Deque[Tuple[int, np.ndarray]] = deque()
        self._q_lock = threading.Lock()

        self._last_written_ts = 0
        self._target_dt = int(1000 / max(1, self.cfg.fps))

        self._w = None
        self._h = None

    # ---- ingestion ----
    def ingest_frame(self, frame_bgr: np.ndarray, ts_ms: Optional[int] = None):
        if frame_bgr is None or frame_bgr.size == 0:
            return
        if ts_ms is None:
            ts_ms = int(time.time() * 1000)

        h, w = frame_bgr.shape[:2]
        if self._w is None:
            self._w, self._h = int(w), int(h)

        # update ring buffer
        b = int(frame_bgr.nbytes) + 16
        with self._ring_lock:
            self._ring.append((ts_ms, frame_bgr.copy()))
            self._ring_bytes += b
            # trim by time
            cutoff = ts_ms - self.cfg.prebuffer_sec * 1000
            while self._ring and self._ring[0][0] < cutoff:
                _, old = self._ring.popleft()
                self._ring_bytes -= int(old.nbytes) + 16
            # trim by RAM
            cap_bytes = self.cfg.max_ram_mb * 1024 * 1024
            while self._ring and self._ring_bytes > cap_bytes:
                _, old = self._ring.popleft()
                self._ring_bytes -= int(old.nbytes) + 16

        # enqueue to writer if recording
        if self._running:
            with self._q_lock:
                self._q.append((ts_ms, frame_bgr.copy()))

    # ---- control ----
    def start(self) -> Path:
        if self._running:
            return self._writer_path_final or self._writer_path_tmp or Path()

        now = time.localtime()
        stamp = time.strftime("%Y%m%d-%H%M%S", now)
        base = f"{self._sanitize(self.name)}_{stamp}"
        tmp = self.cfg.out_dir / ".tmp" / f"{base}.avi"
        final = self.cfg.out_dir / f"{base}.avi"
        self._writer_path_tmp = tmp
        self._writer_path_final = final

        # open writer lazily after we know size
        if self._w is None or self._h is None:
            # fallback default if no frames yet
            self._w, self._h = 640, 480

        w_even = self._w - (self._w % 2)
        h_even = self._h - (self._h % 2)
        if w_even <= 0 or h_even <= 0:
            w_even, h_even = 640, 480

        self._open_writer(tmp, w_even, h_even)

        # seed queue with ring buffer content near target FPS
        seed = []
        with self._ring_lock:
            seed = list(self._ring)
        seed = self._resample(seed, self.cfg.fps)
        with self._q_lock:
            self._q.extend(seed)

        self._running = True
        self._last_written_ts = 0
        self._write_thread = threading.Thread(target=self._writer_loop, name=f"{self.name}-recorder", daemon=True)
        self._write_thread.start()
        return final

    def stop(self) -> Optional[Path]:
        if not self._running:
            return self._writer_path_final
        self._running = False
        # wait for thread
        if self._write_thread:
            self._write_thread.join(timeout=5.0)
        self._write_thread = None

        # close writer
        with self._writer_lock:
            try:
                if self._writer:
                    self._writer.release()
            finally:
                self._writer = None

        # rename tmp -> final if tmp exists and is non-empty
        try:
            if self._writer_path_tmp and self._writer_path_tmp.exists():
                if self._writer_path_tmp.stat().st_size > 0:
                    if self._writer_path_final:
                        try:
                            self._writer_path_final.unlink()
                        except FileNotFoundError:
                            pass
                        self._writer_path_tmp.replace(self._writer_path_final)
                        return self._writer_path_final
        except Exception:
            pass
        return self._writer_path_tmp

    # ---- internals ----
    def _open_writer(self, path: Path, w: int, h: int):
        fourcc = cv2.VideoWriter_fourcc(*self.cfg.codec_primary)
        writer = cv2.VideoWriter(str(path), fourcc, float(self.cfg.fps), (w, h))
        if not writer.isOpened():
            # fallback to mp4 in tmp folder
            alt = path.with_suffix(".mp4")
            fourcc2 = cv2.VideoWriter_fourcc(*self.cfg.codec_fallback)
            writer2 = cv2.VideoWriter(str(alt), fourcc2, float(self.cfg.fps), (w, h))
            if writer2.isOpened():
                self._writer_path_tmp = alt
                writer = writer2
            else:
                raise RuntimeError("Failed to open VideoWriter for both MJPG and mp4v")
        with self._writer_lock:
            self._writer = writer

    def _writer_loop(self):
        target_dt = self._target_dt
        while self._running or self._q:
            item = None
            with self._q_lock:
                if self._q:
                    item = self._q.popleft()
            if item is None:
                time.sleep(0.005)
                continue

            ts, frame = item
            # resample live to target FPS using timestamps
            if self._last_written_ts == 0 or ts - self._last_written_ts >= target_dt:
                self._write_frame(frame)
                self._last_written_ts = ts
            else:
                # drop frame
                continue

    def _write_frame(self, frame: np.ndarray):
        if frame is None or frame.size == 0:
            return
        h, w = frame.shape[:2]
        w_even = w - (w % 2)
        h_even = h - (h % 2)
        if (w_even != w) or (h_even != h):
            frame = cv2.resize(frame, (w_even, h_even), interpolation=cv2.INTER_AREA)
        with self._writer_lock:
            if self._writer:
                self._writer.write(frame)

    @staticmethod
    def _resample(samples: List[Tuple[int, np.ndarray]], fps: int):
        """Return a list of frames at ~fps from timestamped samples."""
        if not samples:
            return []
        target_dt = int(1000 / max(1, fps))
        out: List[Tuple[int, np.ndarray]] = []
        last_ts = 0
        for ts, frame in samples:
            if not out:
                out.append((ts, frame))
                last_ts = ts
            else:
                if ts - last_ts >= target_dt:
                    out.append((ts, frame))
                    last_ts = ts
        return out

    @staticmethod
    def _sanitize(name: str) -> str:
        keep = []
        for ch in name:
            if ch.isalnum() or ch in ("-", "_"):
                keep.append(ch)
            elif ch == " ":
                keep.append("_")
        return "".join(keep)[:64]

```


## FILE: AI/camera_widget.py
```text
# camera_widget.py
from __future__ import annotations
from typing import Optional
import time
import cv2
from PyQt6 import QtCore, QtGui, QtWidgets

from detectors import DetectorThread, DetectorConfig, DetectionPacket
from overlays import OverlayFlags, draw_overlays
from recorder import PrebufferRecorder
from presence import PresenceBus
from settings import AppSettings, CameraSettings
from utils import qimage_from_bgr
from stream import StreamCapture
from graphics_view import GraphicsView
from enrollment_service import EnrollmentService


class CameraWidget(QtWidgets.QWidget):
    def __init__(
        self,
        cam_cfg: CameraSettings,
        app_cfg: AppSettings,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        super().__init__(parent)
        self.cam_cfg = cam_cfg
        self.app_cfg = app_cfg

        # scene + view
        self._scene = QtWidgets.QGraphicsScene(self)
        self._pixmap_item = QtWidgets.QGraphicsPixmapItem()
        self._scene.addItem(self._pixmap_item)
        self.view = GraphicsView(self._scene, self)

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)

        # toolbar row
        bar = QtWidgets.QHBoxLayout()
        self.btn_rec = QtWidgets.QPushButton("● REC")
        self.btn_snap = QtWidgets.QPushButton("Snapshot")

        # underlying checkboxes still hold state but are hidden
        self.cb_ai = QtWidgets.QCheckBox("AI")
        self.cb_yolo = QtWidgets.QCheckBox("YOLO")
        self.cb_faces = QtWidgets.QCheckBox("Faces")
        self.cb_pets = QtWidgets.QCheckBox("Pets")
        self.cb_ai.setChecked(True)
        self.cb_yolo.setChecked(True)
        self.cb_faces.setChecked(True)
        self.cb_pets.setChecked(True)
        self.cb_ai.hide()
        self.cb_yolo.hide()
        self.cb_faces.hide()
        self.cb_pets.hide()

        # AI dropdown menu
        self.btn_ai_menu = QtWidgets.QToolButton()
        self.btn_ai_menu.setText("AI")
        self.btn_ai_menu.setPopupMode(
            QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup
        )
        self.menu_ai = QtWidgets.QMenu(self)
        self.act_ai_enable = self.menu_ai.addAction("Enable AI")
        self.act_ai_enable.setCheckable(True)
        self.act_ai_enable.setChecked(True)
        self.act_yolo = self.menu_ai.addAction("YOLO objects")
        self.act_yolo.setCheckable(True)
        self.act_yolo.setChecked(True)
        self.act_faces = self.menu_ai.addAction("Face recognition")
        self.act_faces.setCheckable(True)
        self.act_faces.setChecked(True)
        self.act_pets = self.menu_ai.addAction("Pet recognition")
        self.act_pets.setCheckable(True)
        self.act_pets.setChecked(True)
        self.btn_ai_menu.setMenu(self.menu_ai)

        # View dropdown menu (Fit / 100% / Fit window to video + lock)
        self.btn_view_menu = QtWidgets.QToolButton()
        self.btn_view_menu.setText("View")
        self.btn_view_menu.setPopupMode(
            QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup
        )
        self.menu_view = QtWidgets.QMenu(self)
        self.act_fit = self.menu_view.addAction("Fit in window")
        self.act_100 = self.menu_view.addAction("100%")
        self.act_fit_win = self.menu_view.addAction("Fit window to video")
        self.menu_view.addSeparator()
        self.act_lock = self.menu_view.addAction("Lock window configuration")
        self.act_lock.setCheckable(True)
        self.btn_view_menu.setMenu(self.menu_view)

        bar.addWidget(self.btn_rec)
        bar.addWidget(self.btn_snap)
        bar.addSpacing(12)
        bar.addWidget(self.btn_ai_menu)
        bar.addStretch(1)
        bar.addWidget(self.btn_view_menu)

        lay.addLayout(bar)
        lay.addWidget(self.view)

        self._overlays = OverlayFlags()
        self._ai_enabled = True
        self._last_bgr: Optional[cv2.Mat] = None
        self._last_ts = 0
        self._locked = False
        self._last_pkt: Optional[DetectionPacket] = None

        self._recorder = PrebufferRecorder(
            cam_name=self.cam_cfg.name,
            out_dir=self.app_cfg.output_dir,
            fps=25,
            pre_ms=self.app_cfg.prebuffer_ms,
        )
        self._presence = PresenceBus(self.cam_cfg.name, self.app_cfg.logs_dir)

        det_cfg = DetectorConfig.from_app(self.app_cfg)
        self._detector = DetectorThread(det_cfg, self.cam_cfg.name)
        self._detector.resultsReady.connect(self._on_detections)

        self._capture = StreamCapture(self.cam_cfg)

        self._frame_timer = QtCore.QTimer(self)
        self._frame_timer.setInterval(30)
        self._frame_timer.timeout.connect(self._poll_frame)

        # View actions
        self.act_fit.triggered.connect(self.fit_to_window)
        self.act_100.triggered.connect(self.zoom_100)   
        self.act_fit_win.triggered.connect(self.fit_window_to_video)

        # Buttons / overlay wiring
        self.btn_snap.clicked.connect(self._snapshot)
        self.btn_rec.clicked.connect(self._toggle_recording)
        self.cb_ai.toggled.connect(self._on_ai_toggled)
        self.cb_yolo.toggled.connect(self._on_overlay_changed)
        self.cb_faces.toggled.connect(self._on_overlay_changed)
        self.cb_pets.toggled.connect(self._on_overlay_changed)
        self.act_ai_enable.toggled.connect(self.cb_ai.setChecked)
        self.act_yolo.toggled.connect(self.cb_yolo.setChecked)
        self.act_faces.toggled.connect(self.cb_faces.setChecked)
        self.act_pets.toggled.connect(self.cb_pets.setChecked)
        self.act_lock.toggled.connect(self._on_lock_toggled)

        # Initialise overlay flags from current UI state
        self._on_overlay_changed()

        self._detector.start()

    # lifecycle
    def start(self):
        self._capture.start()
        self._frame_timer.start()

    def stop(self):
        self._frame_timer.stop()
        self._capture.stop()
        self._detector.stop()
        self._recorder.close()

    # frame path
    def _poll_frame(self):
        ok, frame, ts_ms = self._capture.read()
        if not ok or frame is None:
            return
        self._last_bgr = frame
        self._last_ts = ts_ms

        self._recorder.on_frame(frame, ts_ms)
        if self._ai_enabled:
            self._detector.submit_frame(self.cam_cfg.name, frame, ts_ms)

        pkt = self._last_pkt if (self._ai_enabled and self._last_pkt is not None) else None
        self._update_pixmap(frame, pkt)

    def _update_pixmap(self, bgr, pkt: Optional[DetectionPacket]):
        qimg = qimage_from_bgr(bgr)
        pixmap = QtGui.QPixmap.fromImage(qimg)
        if pkt is not None and self._ai_enabled:
            p = QtGui.QPainter(pixmap)
            try:
                draw_overlays(p, pkt, self._overlays, self.cam_cfg.name, time.time())
            finally:
                p.end()
        self._pixmap_item.setPixmap(pixmap)
        self._scene.setSceneRect(QtCore.QRectF(pixmap.rect()))

    @QtCore.pyqtSlot(object)
    def _on_detections(self, pkt_obj):
        pkt = pkt_obj
        if not isinstance(pkt, DetectionPacket):
            return
        if pkt.name != self.cam_cfg.name:
            return

        # Presence bus / events
        self._presence.update(pkt)

        # Store packet so that future frames can be drawn with overlays without flicker
        self._last_pkt = pkt

        # Feed enrollment service (if active) with this camera's detections
        svc = EnrollmentService.instance()
        if self._last_bgr is not None:
            svc.on_detections(self.cam_cfg.name, self._last_bgr, pkt)

    # helpers
    def _snapshot(self):
        if self._last_bgr is None:
            return
        stamp = time.strftime("%Y%m%d_%H%M%S")
        fname = f"{self.cam_cfg.name}_{stamp}.jpg"
        out = self.app_cfg.output_dir / fname
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), self._last_bgr)

    def _toggle_recording(self):
        if self._recorder.writer is None:
            self._recorder.start()
            self.btn_rec.setText("■ STOP")
        else:
            self._recorder.stop()
            self.btn_rec.setText("● REC")

    def _on_ai_toggled(self, checked: bool):
        self._ai_enabled = bool(checked)
        if not self._ai_enabled:
            self._last_pkt = None
            if self._last_bgr is not None:
                self._update_pixmap(self._last_bgr, None)

    def _on_overlay_changed(self):
        # map UI toggles into OverlayFlags
        self._overlays.yolo = self.cb_yolo.isChecked()
        self._overlays.faces = self.cb_faces.isChecked()
        self._overlays.pets = self.cb_pets.isChecked()
        # hud / crosshair / info are in OverlayFlags if you’re using them elsewhere

    # view helpers
    def _mdi_subwindow(self) -> Optional[QtWidgets.QMdiSubWindow]:
        w = self.parentWidget()
        while w is not None:
            if isinstance(w, QtWidgets.QMdiSubWindow):
                return w
            w = w.parentWidget()
        return None

    def fit_to_window(self):
        self.view.fitInView(
            self._scene.sceneRect(),
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
        )
        self.view._scale = 1.0

    def zoom_100(self):
        self.view.resetTransform()
        self.view._scale = 1.0

    def fit_window_to_video(self):
        if self._last_bgr is None:
            return

        # reset zoom so one image pixel == one screen pixel
        self.zoom_100()

        pix = self._pixmap_item.pixmap()
        if pix.isNull():
            return

        img_w = pix.width()
        img_h = pix.height()

        sub = self._mdi_subwindow()
        if sub is None:
            # not inside an MDI subwindow; just resize our own widget
            self.resize(img_w, img_h)
            return

        # Measure current viewport and chrome sizes
        vp = self.view.viewport()
        cur_view_w = vp.width() or 1
        cur_view_h = vp.height() or 1
        cur_self_w = self.width()
        cur_self_h = self.height()
        cur_sub_w = sub.width()
        cur_sub_h = sub.height()

        # How much bigger the widget is than the viewport,
        # and the subwindow is than the widget
        chrome_self_w = cur_self_w - cur_view_w
        chrome_self_h = cur_self_h - cur_view_h
        chrome_sub_w = cur_sub_w - cur_self_w
        chrome_sub_h = cur_sub_h - cur_self_h

        # Target: image size + chrome, plus a tiny safety margin
        target_self_w = img_w + chrome_self_w + 2
        target_self_h = img_h + chrome_self_h + 2
        target_sub_w = target_self_w + chrome_sub_w
        target_sub_h = target_self_h + chrome_sub_h

        sub.resize(target_sub_w, target_sub_h)

    def _on_lock_toggled(self, locked: bool):
        self._locked = bool(locked)
        self.btn_ai_menu.setEnabled(not locked)
        # allow unlocking even when locked
        self.btn_view_menu.setEnabled(True)
        for cb in (self.cb_ai, self.cb_yolo, self.cb_faces, self.cb_pets):
            cb.setEnabled(not locked)

```


## FILE: AI/detection_packet.py
```text
# detection_packet.py
from dataclasses import dataclass, field
from typing import List, Tuple, Dict

@dataclass
class DetBox:
    cls: str
    score: float
    xyxy: Tuple[int, int, int, int]

@dataclass
class DetectionPacket:
    name: str
    ts_ms: int
    size: Tuple[int, int]
    yolo: List[DetBox] = field(default_factory=list)
    faces: List[DetBox] = field(default_factory=list)
    pets: List[DetBox] = field(default_factory=list)
    timing_ms: Dict[str, int] = field(default_factory=dict)

```


## FILE: AI/detectors\__init__.py
```text
# detectors/__init__.py
from .packet import DetBox, DetectionPacket
from .config import DetectorConfig, COCO_ID_TO_NAME
from .thread import DetectorThread

__all__ = [
    "DetBox",
    "DetectionPacket",
    "DetectorConfig",
    "COCO_ID_TO_NAME",
    "DetectorThread",
]

```


## FILE: AI/detectors\config.py
```text
# detectors/config.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple

# COCO subset we care about
COCO_ID_TO_NAME: Dict[int, str] = {0: "person", 15: "cat", 16: "dog"}


@dataclass
class DetectorConfig:
    yolo_model: str
    yolo_conf: float = 0.35
    yolo_nms: float = 0.45
    interval_ms: int = 100
    face_cascade: str | None = None

    @classmethod
    def from_app(cls, app_cfg):
        m = app_cfg.models_dir
        return cls(
            yolo_model=str((m / "yolov8n.onnx").resolve()),
            yolo_conf=app_cfg.thresh_yolo,
            yolo_nms=0.45,
            interval_ms=getattr(app_cfg, "detect_interval_ms", 100),
            face_cascade=str((m / "haarcascade_frontalface_default.xml").resolve()),
        )


def letterbox_square(
    img, new_shape: int = 640, color: int = 114
) -> tuple:
    """Square letterbox to new_shape x new_shape with padding."""
    import cv2
    import numpy as np

    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)
    nh, nw = int(h * r), int(w * r)
    resized = cv2.resize(img, (nw, nh))
    canvas = np.full((new_shape, new_shape, 3), color, np.uint8)
    top = (new_shape - nh) // 2
    left = (new_shape - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas, r, left, top

```


## FILE: AI/detectors\faces_backend.py
```text
# detectors/faces_backend.py
from __future__ import annotations
from typing import Optional
import os

import cv2 as cv

from .packet import DetBox, DetectionPacket
from .config import DetectorConfig
from .lbph_model import LBPHModel


class FaceBackend:
    def __init__(self, cfg: DetectorConfig, models_dir: str):
        self.cascade: Optional[cv.CascadeClassifier] = None
        self.lbph = LBPHModel.load(models_dir)
        if cfg.face_cascade and os.path.exists(cfg.face_cascade):
            try:
                self.cascade = cv.CascadeClassifier(cfg.face_cascade)
            except Exception as e:
                print(f"[Face] Haar load failed: {e}")
                self.cascade = None
        else:
            print(f"[Face] Haar not found at {cfg.face_cascade}")

    def run(self, bgr, pkt: DetectionPacket):
        if self.cascade is None:
            return
        try:
            gray = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)
            try:
                eq = cv.createCLAHE(2.0, (8, 8)).apply(gray)
            except Exception:
                eq = cv.equalizeHist(gray)

            minsz = max(40, int(0.12 * min(gray.shape[:2])))
            faces = self.cascade.detectMultiScale(eq, 1.1, 4, minSize=(minsz, minsz))
            if len(faces) == 0:
                faces = self.cascade.detectMultiScale(eq, 1.05, 3, minSize=(minsz, minsz))

            for (fx, fy, fw, fh) in faces:
                roi = gray[fy:fy + fh, fx:fx + fw]
                name, score = self.lbph.predict_name(roi)
                x1, y1, x2, y2 = fx, fy, fx + fw, fy + fh
                pkt.faces.append(DetBox(name, float(score), (x1, y1, x2, y2)))
        except Exception as e:
            print(f"[Face] error: {e}")


```


## FILE: AI/detectors\lbph_model.py
```text
# detectors/lbph_model.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import json
import os

import cv2 as cv
import numpy as np


@dataclass
class LBPHModel:
    recognizer: Optional[cv.face_LBPHFaceRecognizer]  # type: ignore
    labels: Dict[int, str]
    accept_conf: float = 95.0  # distance threshold

    @classmethod
    def load(cls, models_dir: str) -> "LBPHModel":
        model_path = os.path.join(models_dir, "lbph_faces.xml")
        labels_path = os.path.join(models_dir, "labels_faces.json")
        rec = None
        labels: Dict[int, str] = {}
        try:
            if os.path.exists(model_path):
                rec = cv.face.LBPHFaceRecognizer_create()  # type: ignore[attr-defined]
                rec.read(model_path)
            if os.path.exists(labels_path):
                with open(labels_path, "r", encoding="utf-8") as fp:
                    m = json.load(fp)
                labels = {int(v): k for k, v in m.items()}
        except Exception as e:
            print(f"[LBPH] disabled: {e}")
            rec = None
            labels = {}
        return cls(rec, labels)

    def predict_name(self, roi_gray) -> Tuple[str, float]:
        if self.recognizer is None:
            return "face", 0.6
        try:
            roi = cv.resize(roi_gray, (160, 160))
            pred, dist = self.recognizer.predict(roi)
            if 0 <= pred and dist <= self.accept_conf and pred in self.labels:
                name = self.labels[int(pred)]
                score = max(0.0, min(1.0, 1.0 - (dist / self.accept_conf)))
            else:
                name = "unknown"
                score = 0.4
            print(f"[LBPH] pred={pred} name={self.labels.get(int(pred), '?')} dist={dist:.1f} -> {name}")
            return name, float(score)
        except Exception as e:
            print(f"[LBPH] predict error: {e}")
            return "face", 0.6

```


## FILE: AI/detectors\packet.py
```text
# detectors/packet.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Dict


@dataclass
class DetBox:
    cls: str
    score: float
    xyxy: Tuple[int, int, int, int]


@dataclass
class DetectionPacket:
    name: str
    ts_ms: int
    size: Tuple[int, int]
    yolo: List[DetBox] = field(default_factory=list)
    faces: List[DetBox] = field(default_factory=list)
    pets: List[DetBox] = field(default_factory=list)
    timing_ms: Dict[str, int] = field(default_factory=dict)

```


## FILE: AI/detectors\thread.py
```text
# detectors/thread.py
from __future__ import annotations
import os
import threading
import time
from typing import Optional, Tuple

import numpy as np
from PyQt6 import QtCore

from .config import DetectorConfig
from .packet import DetectionPacket
from .yolo_backend import load_yolo, run_yolo
from .faces_backend import FaceBackend
from utils import monotonic_ms


class DetectorThread(QtCore.QThread):
    resultsReady = QtCore.pyqtSignal(object)

    def __init__(self, cfg: DetectorConfig, name: str):
        super().__init__()
        self.cfg = cfg
        self.name = name
        self._latest: Optional[Tuple[np.ndarray, int]] = None
        self._lock = threading.RLock()
        self._stop = threading.Event()

        self.models_dir = os.path.dirname(self.cfg.yolo_model)
        self._net = load_yolo(self.cfg)
        self._faces = FaceBackend(self.cfg, self.models_dir)

        print(
            f"[Detector:{self.name}] init: "
            f"net={'OK' if self._net is not None else 'NONE'}, "
            f"face={'OK' if self._faces.cascade is not None else 'NONE'}, "
            f"lbph={'OK' if self._faces.lbph.recognizer is not None else 'NONE'}, "
            f"yolo_conf={self.cfg.yolo_conf}"
        )

    def submit_frame(self, cam_name: str, bgr: np.ndarray, ts_ms: int):
        if cam_name != self.name:
            return
        with self._lock:
            self._latest = (bgr.copy(), ts_ms)

    def stop(self):
        self._stop.set()

    def run(self):
        next_due = 0
        while not self._stop.is_set():
            now = monotonic_ms()
            if now < next_due:
                time.sleep(max(0, (next_due - now) / 1000.0))
                continue
            next_due = now + self.cfg.interval_ms

            with self._lock:
                snap = self._latest
            if snap is None:
                continue

            bgr, ts_ms = snap
            H, W = bgr.shape[:2]
            pkt = DetectionPacket(self.name, ts_ms, (W, H))

            t0 = monotonic_ms()
            run_yolo(self._net, self.cfg, bgr, pkt)
            t1 = monotonic_ms()
            self._faces.run(bgr, pkt)
            t2 = monotonic_ms()

            pkt.timing_ms["yolo"] = int(t1 - t0)
            pkt.timing_ms["faces"] = int(t2 - t1)

            self.resultsReady.emit(pkt)

```


## FILE: AI/detectors\yolo_backend.py
```text
# detectors/yolo_backend.py
from __future__ import annotations
from typing import List, Tuple
import os

import cv2
import numpy as np

from .packet import DetBox, DetectionPacket
from .config import DetectorConfig, COCO_ID_TO_NAME, letterbox_square


def load_yolo(cfg: DetectorConfig):
    if not os.path.exists(cfg.yolo_model):
        print(f"[YOLO] model not found at {cfg.yolo_model}")
        return None
    try:
        net = cv2.dnn.readNetFromONNX(cfg.yolo_model)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        return net
    except Exception as e:
        print(f"[YOLO] load failed: {e}")
        return None


def run_yolo(net, cfg: DetectorConfig, bgr, pkt: DetectionPacket) -> int:
    if net is None:
        return 0
    H, W = bgr.shape[:2]
    img, r, dx, dy = letterbox_square(bgr, new_shape=640)
    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (640, 640), swapRB=True, crop=False)
    net.setInput(blob)
    out = net.forward()
    out = np.squeeze(out)

    if out.ndim == 2 and out.shape[0] in (84, 85):
        out = out.T
    elif out.ndim == 3:
        o = out[0]
        out = o.T if o.shape[0] in (84, 85) else o

    boxes: List[Tuple[float, float, float, float]] = []
    scores: List[float] = []
    ids: List[int] = []

    for det in out:
        det = np.asarray(det).ravel()
        if det.shape[0] < 5:
            continue
        cx, cy, w, h = det[:4]
        if det.shape[0] >= 85:
            obj = float(det[4])
            cls_scores = det[5:]
            c = int(np.argmax(cls_scores))
            conf = obj * float(cls_scores[c])
        else:
            c = int(det[4])
            conf = float(det[5]) if det.shape[0] > 5 else 0.0
        if conf < cfg.yolo_conf:
            continue
        boxes.append((float(cx), float(cy), float(w), float(h)))
        scores.append(conf)
        ids.append(c)

    for (cx, cy, w, h), conf, cid in zip(boxes, scores, ids):
        if cid not in COCO_ID_TO_NAME:
            continue
        cx0 = (cx - dx) / r
        cy0 = (cy - dy) / r
        w0 = w / r
        h0 = h / r
        x1 = max(0, int(cx0 - w0 / 2))
        y1 = max(0, int(cy0 - h0 / 2))
        x2 = min(W - 1, int(cx0 + w0 / 2))
        y2 = min(H - 1, int(cy0 + h0 / 2))
        label = COCO_ID_TO_NAME[cid]
        box = DetBox(label, float(conf), (x1, y1, x2, y2))
        pkt.yolo.append(box)
        if label in ("cat", "dog"):
            pkt.pets.append(box)

    # no easy split timing here; we return 0 and let caller time externally
    return 0

```


## FILE: AI/discovery_dialog.py
```text
# discovery_dialog.py
# Local subnet scanner for ESP32-CAM. Looks for /api/status (preferred),
# then /status, /stream, / on ports 80 and 81.

from __future__ import annotations
import socket
import threading
import queue
from typing import Set

from PyQt6 import QtWidgets, QtCore
import requests


def _guess_primary_ipv4() -> str | None:
    """
    Try to get the primary IPv4 by opening a UDP socket to a public IP.
    This avoids cases where gethostname() resolves to 127.0.x.x.
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.2)
        # We don't actually connect, just use routing table
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return None


def _default_subnet() -> str:
    """
    Try to guess the local /24 from the primary IPv4; fall back to 192.168.1.x.
    """
    ip = _guess_primary_ipv4()
    if ip:
        parts = ip.split(".")
        if len(parts) == 4:
            return ".".join(parts[:3]) + "."
    # Fallback
    return "192.168.1."


class DiscoveryDialog(QtWidgets.QDialog):
    # idx, total, ip
    progress = QtCore.pyqtSignal(int, int, str)
    # label
    addItemSignal = QtCore.pyqtSignal(str)
    # scan finished (renamed earlier to avoid clashing with QDialog.done())
    scanFinished = QtCore.pyqtSignal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Discover ESP32-CAM")

        # --- UI ---
        self.edit_subnet = QtWidgets.QLineEdit(_default_subnet())
        self.edit_range_from = QtWidgets.QSpinBox()
        self.edit_range_from.setRange(1, 254)
        self.edit_range_from.setValue(1)

        self.edit_range_to = QtWidgets.QSpinBox()
        self.edit_range_to.setRange(1, 254)
        self.edit_range_to.setValue(254)

        self.btn_scan = QtWidgets.QPushButton("Scan")
        self.btn_stop = QtWidgets.QPushButton("Stop")
        self.btn_stop.setEnabled(False)

        self.list = QtWidgets.QListWidget()
        self.lbl = QtWidgets.QLabel(
            "Finds devices responding on /api/status (preferred), "
            "/status, /stream, or / on ports 80/81.\n"
            "401 Unauthorized is treated as a hit (auth-only /api/status).\n"
            "Scanning uses a concurrent worker pool for speed."
        )
        self.lbl_progress = QtWidgets.QLabel("Idle")
        self.pb = QtWidgets.QProgressBar()
        self.pb.setMinimum(0)
        self.pb.setMaximum(0)
        self.pb.setValue(0)

        form = QtWidgets.QFormLayout()
        form.addRow("Subnet prefix", self.edit_subnet)
        form.addRow("Range", self._range_row())

        btns = QtWidgets.QHBoxLayout()
        btns.addWidget(self.btn_scan)
        btns.addWidget(self.btn_stop)
        btns.addStretch(1)

        lay = QtWidgets.QVBoxLayout(self)
        lay.addLayout(form)
        lay.addLayout(btns)
        lay.addWidget(self.lbl_progress)
        lay.addWidget(self.pb)
        lay.addWidget(self.list)
        lay.addWidget(self.lbl)

        # --- state ---
        self._stop = threading.Event()
        self._seen_keys: Set[str] = set()
        self._lock = threading.Lock()  # protects _seen_keys and progress counter

        # --- wire ---
        self.btn_scan.clicked.connect(self._start)
        self.btn_stop.clicked.connect(self._cancel)

        self.progress.connect(self._on_progress)
        self.addItemSignal.connect(self._add_item)
        self.scanFinished.connect(self._done)

    # ------------------------------------------------------------------ UI helpers

    def _range_row(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(self.edit_range_from)
        h.addWidget(QtWidgets.QLabel("to"))
        h.addWidget(self.edit_range_to)
        h.addStretch(1)
        return w

    # ------------------------------------------------------------------ control

    def _start(self) -> None:
        self.list.clear()
        self._seen_keys.clear()
        self._stop.clear()

        self.btn_scan.setEnabled(False)
        self.btn_stop.setEnabled(True)

        subnet = self.edit_subnet.text().strip()
        a = int(self.edit_range_from.value())
        b = int(self.edit_range_to.value())
        total = max(0, b - a + 1)

        self.pb.setMinimum(0)
        if total > 0:
            self.pb.setMaximum(total)
            self.pb.setValue(0)
            self.lbl_progress.setText(f"Scanning {subnet}{a} to {subnet}{b} (0/{total})")
        else:
            self.pb.setMaximum(0)
            self.lbl_progress.setText("Scanning...")

        # Run the IP scanning in a single manager thread which spawns a worker pool.
        t = threading.Thread(
            target=self._scan_range,
            args=(subnet, a, b, total),
            daemon=True,
        )
        t.start()

    def _cancel(self) -> None:
        self._stop.set()
        self.btn_scan.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.lbl_progress.setText("Scan cancelled.")

    # ------------------------------------------------------------------ scanning (concurrent workers over IPs)

    def _scan_range(self, subnet: str, a: int, b: int, total: int) -> None:
        # Prepare queue of IPs to scan
        ips = [f"{subnet}{i}" for i in range(a, b + 1)]
        q: queue.Queue[str] = queue.Queue()
        for ip in ips:
            q.put(ip)

        total = len(ips)
        if total == 0:
            self.scanFinished.emit()
            return

        # Shared progress counter
        idx = 0

        # Prioritise /api/status on port 80, then other combinations.
        paths = ("/api/status", "/status", "/stream", "/")
        ports = (80, 81)

        def worker() -> None:
            nonlocal idx
            sess = requests.Session()
            sess.headers.update({"User-Agent": "ESP32-CAM-Discovery/1.0"})
            while not self._stop.is_set():
                try:
                    ip = q.get_nowait()
                except queue.Empty:
                    break

                found_for_ip = False

                for port in ports:
                    if self._stop.is_set():
                        break

                    for path in paths:
                        if self._stop.is_set():
                            break

                        key = f"{ip}:{port}"
                        # If we've already got this IP:port, skip further paths
                        with self._lock:
                            if key in self._seen_keys:
                                found_for_ip = True
                                break

                        url = f"http://{ip}:{port}{path}"
                        try:
                            r = sess.get(url, timeout=0.6)
                        except Exception:
                            continue

                        # Accept 2xx and 401 as hits
                        if (200 <= r.status_code < 300) or (r.status_code == 401):
                            with self._lock:
                                if key in self._seen_keys:
                                    # Another worker already recorded it
                                    found_for_ip = True
                                    break
                                self._seen_keys.add(key)

                            label = f"{ip}:{port}  {path}"

                            # If this is /api/status and NOT 401, try to pull a name from JSON
                            if path == "/api/status" and (200 <= r.status_code < 300):
                                try:
                                    data = r.json()
                                    nm = data.get("name") or data.get("camera")
                                    if nm:
                                        label = f"{ip}:{port}  {nm} (/api/status)"
                                except Exception:
                                    pass

                            # emit to GUI thread
                            self.addItemSignal.emit(label)
                            found_for_ip = True
                            break

                    if found_for_ip:
                        break

                # progress update
                with self._lock:
                    idx += 1
                    cur_idx = idx

                self.progress.emit(cur_idx, total, ip)
                q.task_done()

        # Spawn worker pool
        num_workers = min(32, total)
        threads = []
        for _ in range(num_workers):
            t = threading.Thread(target=worker, daemon=True)
            t.start()
            threads.append(t)

        # Wait for workers to finish
        for t in threads:
            t.join()

        self.scanFinished.emit()

    # ------------------------------------------------------------------ slots (GUI thread)

    @QtCore.pyqtSlot()
    def _done(self) -> None:
        self.btn_scan.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.lbl_progress.setText("Scan complete.")

    @QtCore.pyqtSlot(int, int, str)
    def _on_progress(self, idx: int, total: int, ip: str) -> None:
        if total > 0:
            self.pb.setMaximum(total)
            self.pb.setValue(min(idx, total))
            self.lbl_progress.setText(f"Scanning {ip} ({idx}/{total})")
        else:
            self.pb.setMaximum(0)
            self.lbl_progress.setText(f"Scanning {ip}")

    @QtCore.pyqtSlot(str)
    def _add_item(self, label: str) -> None:
        self.list.addItem(label)

```


## FILE: AI/enrollment.py
```text
# enrollment.py
from __future__ import annotations
from PyQt6 import QtWidgets, QtCore
from pathlib import Path
from settings import AppSettings, BASE_DIR
from enrollment_service import EnrollmentService


class EnrollDialog(QtWidgets.QDialog):
    def __init__(self, app_cfg: AppSettings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Face Enrollment")
        self.app_cfg = app_cfg
        self.svc = EnrollmentService.instance()
        self.svc.status_changed.connect(self._on_status)

        self.name = QtWidgets.QLineEdit()
        self.target = QtWidgets.QSpinBox()
        self.target.setRange(5, 200)
        self.target.setValue(25)

        # Camera selection
        self.cam_combo = QtWidgets.QComboBox()
        cam_names = [c.name for c in self.app_cfg.cameras]
        if cam_names:
            self.cam_combo.addItems(cam_names)
        else:
            self.cam_combo.addItem("(no cameras)")
            self.cam_combo.setEnabled(False)

        self.status = QtWidgets.QLabel("Idle")
        self.pb = QtWidgets.QProgressBar()
        self.pb.setRange(0, 100)
        self.pb.setValue(0)

        # Buttons: Start / Abort / Close
        self.btn_start = QtWidgets.QPushButton("Start Face Enrollment")
        self.btn_abort = QtWidgets.QPushButton("Abort")
        self.btn_close = QtWidgets.QPushButton("Close")

        self.btn_start.clicked.connect(self._start)
        self.btn_abort.clicked.connect(self._abort)
        self.btn_close.clicked.connect(self.close)

        form = QtWidgets.QFormLayout()
        form.addRow("Name", self.name)
        form.addRow("Camera", self.cam_combo)
        form.addRow("Samples target", self.target)

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addWidget(self.btn_start)
        btn_row.addWidget(self.btn_abort)
        btn_row.addStretch(1)
        btn_row.addWidget(self.btn_close)

        lay = QtWidgets.QVBoxLayout(self)
        lay.addLayout(form)
        lay.addWidget(self.status)
        lay.addWidget(self.pb)
        lay.addLayout(btn_row)

        # Ensure initial button state matches current service state
        self._update_buttons()

    # ------------------------------------------------------------------ control

    def _set_running(self, running: bool) -> None:
        """Enable/disable buttons based on whether enrollment is active."""
        self.btn_start.setEnabled(not running)
        # Abort makes sense only while active
        self.btn_abort.setEnabled(running)

    def _start(self):
        nm = self.name.text().strip()
        if not nm:
            QtWidgets.QMessageBox.warning(self, "Enroll", "Enter a name.")
            return
        if not self.cam_combo.isEnabled() or self.cam_combo.count() == 0:
            QtWidgets.QMessageBox.warning(self, "Enroll", "No cameras available for enrollment.")
            return
        cam_name = self.cam_combo.currentText().strip() or None
        # force faces dir under BASE_DIR/data/faces
        faces_root = Path(BASE_DIR) / "data" / "faces"
        faces_root.mkdir(parents=True, exist_ok=True)
        self.svc.faces_dir = str(faces_root)
        self.svc.begin_face(nm, int(self.target.value()), cam_name)
        self.status.setText("Running…")
        self._set_running(True)

    def _abort(self):
        """User-initiated abort of the current enrollment run."""
        if self.svc.active:
            self.svc.end()
        # service will emit status_changed, but ensure UI looks idle
        self._set_running(False)

    def _update_buttons(self):
        self._set_running(self.svc.active)

    @QtCore.pyqtSlot()
    def _on_status(self):
        st = self.svc.status_text
        self.status.setText(st)
        if self.svc.samples_needed:
            pct = int(100 * self.svc.samples_got / self.svc.samples_needed)
        else:
            pct = 0
        self.pb.setValue(pct)

        # Keep buttons in sync with service state
        self._update_buttons()

        # Auto-close on successful completion
        if (not self.svc.active) and ("complete" in st.lower()):
            self.accept()

```


## FILE: AI/enrollment_service.py
```text
# enrollment_service.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List
from pathlib import Path
import json

import cv2 as cv
import numpy as np
from PyQt6 import QtCore

from detectors import DetectionPacket
from settings import BASE_DIR


@dataclass
class _EnrollState:
    active: bool = False
    kind: str = "face"          # only "face" for now
    name: str = ""
    samples_needed: int = 0
    samples_got: int = 0
    faces_dir: str = ""
    status: str = "Idle"
    start_index: int = 0


class EnrollmentService(QtCore.QObject):
    """
    Singleton service handling face enrollment and LBPH model training.

    Usage:
      svc = EnrollmentService.instance()
      svc.begin_face(name, n_samples, cam_name)
      ...
      svc.on_detections(cam_name, bgr, pkt)  # called by CameraWidget
    """
    status_changed = QtCore.pyqtSignal()
    model_rebuilt = QtCore.pyqtSignal()

    _instance: Optional["EnrollmentService"] = None

    def __init__(self):
        super().__init__()
        self.state = _EnrollState()
        self.faces_dir = str(Path(BASE_DIR) / "data" / "faces")
        self.models_dir = str(Path(BASE_DIR) / "models")
        self.cam_filter: Optional[str] = None

    # ------------------------------------------------------------------ #
    # singleton access
    # ------------------------------------------------------------------ #
    @classmethod
    def instance(cls) -> "EnrollmentService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------ #
    # public API from UI
    # ------------------------------------------------------------------ #
    def begin_face(self, name: str, n: int, cam_name: Optional[str] = None):
        """
        Called by EnrollDialog. Starts collecting face samples.
        """
        self.start(name, n, cam_name)

    def start(self, name: str, n: int, cam_name: Optional[str] = None):
        s = self.state
        s.active = True
        s.kind = "face"
        s.name = name
        s.samples_needed = n
        s.samples_got = 0
        # compute starting index based on existing images so we don't overwrite
        faces_root = Path(self.faces_dir)
        person_dir = faces_root / name
        person_dir.mkdir(parents=True, exist_ok=True)
        existing = sorted(person_dir.glob("*.png"))
        s.start_index = len(existing)
        s.status = f"Collecting samples for {name} ({n} needed)"
        self.cam_filter = cam_name
        self.status_changed.emit()

    def end(self):
        s = self.state
        s.active = False
        s.status = "Idle"
        self.cam_filter = None
        self.status_changed.emit()

    # ------------------------------------------------------------------ #
    # properties / helpers
    # ------------------------------------------------------------------ #
    @property
    def active(self) -> bool:
        return self.state.active

    @property
    def status_text(self) -> str:
        return self.state.status

    @property
    def samples_needed(self) -> int:
        return self.state.samples_needed

    @property
    def samples_got(self) -> int:
        return self.state.samples_got

    def _emit_status(self, text: str):
        self.state.status = text
        self.status_changed.emit()

    # ------------------------------------------------------------------ #
    # detection hook
    # ------------------------------------------------------------------ #
    def on_detections(self, cam_name: str, bgr: np.ndarray, pkt: DetectionPacket):
        """
        Called from CameraWidget whenever there are new detections.
        """
        s = self.state
        if not s.active or s.kind != "face":
            return

        # Only accept packets from the selected camera, if any
        if self.cam_filter is not None and cam_name != self.cam_filter:
            return

        if not pkt.faces:
            return

        # choose largest face by area
        best_face = None
        best_area = 0
        for f in pkt.faces:
            x1, y1, x2, y2 = f.xyxy
            area = (x2 - x1) * (y2 - y1)
            if area > best_area:
                best_area = area
                best_face = f

        if best_face is None:
            return

        x1, y1, x2, y2 = best_face.xyxy
        x1 = max(int(x1), 0)
        y1 = max(int(y1), 0)
        x2 = min(int(x2), bgr.shape[1])
        y2 = min(int(y2), bgr.shape[0])
        if x2 <= x1 or y2 <= y1:
            return

        roi = bgr[y1:y2, x1:x2]
        gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
        gray = cv.resize(gray, (128, 128))

        faces_root = Path(self.faces_dir)
        faces_root.mkdir(parents=True, exist_ok=True)
        person_dir = faces_root / s.name
        person_dir.mkdir(parents=True, exist_ok=True)

        idx = s.start_index + s.samples_got + 1
        out_path = person_dir / f"{idx:04d}.png"
        cv.imwrite(str(out_path), gray)

        s.samples_got += 1
        self._emit_status(
            f"Collected {s.samples_got}/{s.samples_needed} for {s.name}"
        )

        if s.samples_got >= s.samples_needed:
            self._emit_status("Training LBPH model…")
            ok = self._maybe_train_and_save()
            if not ok:
                self._emit_status("Training failed or no data.")
            self.end()

    # ------------------------------------------------------------------ #
    # training
    # ------------------------------------------------------------------ #
    def _maybe_train_and_save(self) -> bool:
        """
        Train LBPH face recogniser from faces_dir and save into models_dir.
        Returns True on success, False otherwise.
        """
        faces_root = Path(self.faces_dir)
        if not faces_root.exists():
            self._emit_status("No faces directory to train from.")
            return False

        images: List[np.ndarray] = []
        labels: List[int] = []
        label_map: dict[str, int] = {}
        label_id = 0

        for person_dir in sorted(faces_root.iterdir()):
            if not person_dir.is_dir():
                continue
            person_name = person_dir.name
            if person_name not in label_map:
                label_map[person_name] = label_id
                label_id += 1
            lbl = label_map[person_name]
            for img_path in sorted(person_dir.glob("*.png")):
                img = cv.imread(str(img_path), cv.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                images.append(img)
                labels.append(lbl)

        if not images:
            self._emit_status("No images found for training.")
            return False

        recognizer = cv.face.LBPHFaceRecognizer_create()
        labels_np = np.array(labels, dtype=np.int32)
        recognizer.train(images, labels_np)

        models_dir = Path(self.models_dir)
        models_dir.mkdir(parents=True, exist_ok=True)
        model_path = models_dir / "lbph_faces.xml"
        recognizer.write(str(model_path))

        labels_path = models_dir / "labels_faces.json"
        with labels_path.open("w", encoding="utf-8") as f:
            json.dump(label_map, f, indent=2)

        self._emit_status(f"Trained LBPH model with {len(label_map)} labels.")
        self.model_rebuilt.emit()
        return True

    def _train_lbph(self) -> bool:
        """
        Public helper used by the menu action 'Rebuild face model from disk…'.
        """
        return self._maybe_train_and_save()

```


## FILE: AI/events_pane.py
```text
# events_pane.py
# Dockable pane that tails JSONL event logs and shows a live list.
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List
from PyQt6 import QtWidgets, QtCore

class EventsPane(QtWidgets.QWidget):
    def __init__(self, logs_dir: Path, parent=None):
        super().__init__(parent)
        self.logs_dir = Path(logs_dir)
        self.list = QtWidgets.QListWidget()
        self.btn_open = QtWidgets.QPushButton("Open Logs Folder")
        self.btn_clear = QtWidgets.QPushButton("Clear View")
        btns = QtWidgets.QHBoxLayout(); btns.addWidget(self.btn_open); btns.addStretch(1); btns.addWidget(self.btn_clear)
        lay = QtWidgets.QVBoxLayout(self); lay.addWidget(self.list); lay.addLayout(btns)
        self.btn_open.clicked.connect(self._open_logs)
        self.btn_clear.clicked.connect(self.list.clear)
        self._pos: Dict[Path, int] = {}
        self._timer = QtCore.QTimer(self); self._timer.timeout.connect(self._poll); self._timer.start(500)

    def _open_logs(self):
        from utils import open_folder_or_warn
        open_folder_or_warn(self, self.logs_dir)

    def _poll(self):
        if not self.logs_dir.exists(): return
        for p in sorted(self.logs_dir.glob("*.jsonl")):
            last = self._pos.get(p, 0)
            try:
                with p.open("rb") as fp:
                    fp.seek(last)
                    for line in fp:
                        try:
                            rec = json.loads(line.decode("utf-8", "ignore"))
                        except Exception:
                            continue
                        ts = rec.get("ts")
                        cam = rec.get("camera")
                        ev = rec.get("event")
                        typ = rec.get("type")
                        self.list.addItem(f"{cam} — {ev} {typ} @ {ts}")
                    self._pos[p] = fp.tell()
            except FileNotFoundError:
                self._pos.pop(p, None)

```


## FILE: AI/face_params.py
```text
# face_params.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Optional
import json, os

@dataclass
class FaceParams:
    accept_conf: float = 95.0      # LBPH accept if conf <= accept_conf
    roi_size: int = 128            # square ROI for LBPH
    eq_hist: bool = True           # apply cv.equalizeHist on ROI
    min_face_px: int = 48          # ignore faces smaller than this
    smooth_n: int = 3              # EMA for displayed score/conf

    @staticmethod
    def load(models_dir: str) -> "FaceParams":
        p = os.path.join(models_dir, "face_recog.json")
        if not os.path.exists(p):
            return FaceParams()
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)
        return FaceParams(**{k: d.get(k, getattr(FaceParams, k)) for k in FaceParams.__annotations__.keys()})

    def save(self, models_dir: str) -> None:
        os.makedirs(models_dir, exist_ok=True)
        p = os.path.join(models_dir, "face_recog.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)

```


## FILE: AI/face_tuner.py
```text
# face_tuner.py
from __future__ import annotations
from PySide6 import QtWidgets, QtCore
from face_params import FaceParams

class FaceRecTunerDialog(QtWidgets.QDialog):
    def __init__(self, models_dir: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Face Recognizer Tuner")
        self.models_dir = models_dir
        self.params = FaceParams.load(models_dir)

        layout = QtWidgets.QFormLayout(self)

        self.s_conf = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.s_conf.setRange(40, 200)
        self.s_conf.setValue(int(self.params.accept_conf))
        self.s_conf.setTickInterval(5); self.s_conf.setTickPosition(QtWidgets.QSlider.TickPosition.TicksBelow)
        layout.addRow("Accept confidence (≤):", self.s_conf)

        self.s_roi = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.s_roi.setRange(64, 256)
        self.s_roi.setValue(int(self.params.roi_size))
        self.s_roi.setTickInterval(16); self.s_roi.setTickPosition(QtWidgets.QSlider.TickPosition.TicksBelow)
        layout.addRow("ROI size (px):", self.s_roi)

        self.s_min = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.s_min.setRange(24, 160)
        self.s_min.setValue(int(self.params.min_face_px))
        self.s_min.setTickInterval(8); self.s_min.setTickPosition(QtWidgets.QSlider.TickPosition.TicksBelow)
        layout.addRow("Min face size (px):", self.s_min)

        self.cb_eq = QtWidgets.QCheckBox("Equalize histogram")
        self.cb_eq.setChecked(self.params.eq_hist)
        layout.addRow(self.cb_eq)

        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Save |
            QtWidgets.QDialogButtonBox.StandardButton.Cancel |
            QtWidgets.QDialogButtonBox.StandardButton.Apply
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        btns.button(QtWidgets.QDialogButtonBox.StandardButton.Apply).clicked.connect(self._apply_only)
        layout.addRow(btns)

    def _apply_only(self):
        self._collect_and_save()
        QtWidgets.QMessageBox.information(self, "Face Tuner", "Parameters saved. Live detectors will pick up changes on next frame.")

    def accept(self):
        self._collect_and_save()
        super().accept()

    def _collect_and_save(self):
        self.params.accept_conf = float(self.s_conf.value())
        self.params.roi_size = int(self.s_roi.value())
        self.params.min_face_px = int(self.s_min.value())
        self.params.eq_hist = bool(self.cb_eq.isChecked())
        self.params.save(self.models_dir)

```


## FILE: AI/gallery.py
```text
# gallery.py
# (unchanged from previous message except for a small guard to display empty dirs gracefully)
from __future__ import annotations
from pathlib import Path
from typing import List
import cv2 as cv, numpy as np
from PyQt6 import QtWidgets, QtGui, QtCore

def _thumb(path: Path, max_size: int = 160) -> QtGui.QPixmap:
    im = cv.imread(str(path), cv.IMREAD_UNCHANGED)
    if im is None:
        return QtGui.QPixmap(160,160)
    if im.ndim == 2:
        im = cv.cvtColor(im, cv.COLOR_GRAY2RGB)
    elif im.shape[2] == 4:
        im = cv.cvtColor(im, cv.COLOR_BGRA2RGB)
    else:
        im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
    h, w = im.shape[:2]
    s = max_size / float(max(h, w)) if max(h, w) else 1.0
    im = cv.resize(im, (max(1, int(w*s)), max(1, int(h*s))), interpolation=cv.INTER_AREA)
    qimg = QtGui.QImage(im.data, im.shape[1], im.shape[0], int(im.strides[0]), QtGui.QImage.Format.Format_RGB888).copy()
    return QtGui.QPixmap.fromImage(qimg)

class GalleryDialog(QtWidgets.QDialog):
    def __init__(self, folder: Path, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Gallery — {Path(folder).name}")
        self.setMinimumSize(900, 900)
        self.folder = Path(folder)
        self.view = QtWidgets.QListWidget()
        self.view.setResizeMode(QtWidgets.QListView.ResizeMode.Adjust)
        self.view.setViewMode(QtWidgets.QListView.ViewMode.IconMode)
        self.view.setIconSize(QtCore.QSize(160, 160))
        self.view.setMovement(QtWidgets.QListView.Movement.Static)
        self.view.setSpacing(8)
        self.view.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.btn_del = QtWidgets.QPushButton("Delete Selected")
        self.btn_prune = QtWidgets.QPushButton("Self-Prune Near Duplicates")
        self.btn_refresh = QtWidgets.QPushButton("Refresh")

        btns = QtWidgets.QHBoxLayout()
        btns.addWidget(self.btn_refresh); btns.addStretch(1); btns.addWidget(self.btn_prune); btns.addWidget(self.btn_del)

        lay = QtWidgets.QVBoxLayout(self)
        self.resize(900, 900)
        lay.addWidget(self.view); lay.addLayout(btns)

        self.btn_refresh.clicked.connect(self._load)
        self.btn_del.clicked.connect(self._delete_selected)
        self.btn_prune.clicked.connect(self._self_prune)
        self._load()

    def _load(self):
        self.view.clear()
        if not self.folder.exists():
            return
        pats = ["*.png", "*.PNG", "*.jpg", "*.JPG", "*.jpeg", "*.JPEG"]
        files: List[Path] = []
        for p in pats:
            files += sorted(self.folder.glob(p))
        for f in files:
            pm = _thumb(f)
            it = QtWidgets.QListWidgetItem(QtGui.QIcon(pm), f.name)
            it.setData(QtCore.Qt.ItemDataRole.UserRole, str(f))
            self.view.addItem(it)

    def _delete_selected(self):
        sel = self.view.selectedItems()
        for it in sel:
            Path(it.data(QtCore.Qt.ItemDataRole.UserRole)).unlink(missing_ok=True)
            self.view.takeItem(self.view.row(it))

    def _self_prune(self):
        files = [Path(self.view.item(i).data(QtCore.Qt.ItemDataRole.UserRole)) for i in range(self.view.count())]
        imgs = []
        for f in files:
            im = cv.imread(str(f), cv.IMREAD_GRAYSCALE)
            if im is None: continue
            imgs.append((f, cv.resize(im, (160, 160), interpolation=cv.INTER_AREA)))
        if len(imgs) < 2:
            QtWidgets.QMessageBox.information(self, "Self-Prune", "Not enough images to compare.")
            return
        orb = cv.ORB_create()
        kept = []
        pruned = 0
        for f, im in imgs:
            k, d = orb.detectAndCompute(im, None)
            if d is None or len(k) < 10:
                kept.append((f, k, d)); continue
            drop = False
            for fk, kk, dk in kept:
                if dk is None: continue
                bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
                m = bf.match(d, dk)
                if not m: continue
                dmean = float(np.mean([mm.distance for mm in m]))
                sim = 1.0 - (dmean / 100.0)
                if sim >= 0.82:
                    try: f.unlink()
                    except Exception: pass
                    pruned += 1
                    drop = True
                    break
            if not drop:
                kept.append((f, k, d))
        self._load()
        QtWidgets.QMessageBox.information(self, "Self-Prune", f"Removed {pruned} near-duplicates.")

```


## FILE: AI/graphics_view.py
```text
# graphics_view.py
from __future__ import annotations
from typing import Optional
from PyQt6 import QtCore, QtGui, QtWidgets


class GraphicsView(QtWidgets.QGraphicsView):
    zoomChanged = QtCore.pyqtSignal(float)

    def __init__(self, scene: QtWidgets.QGraphicsScene,
                 parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(scene, parent)
        self.setRenderHints(
            QtGui.QPainter.RenderHint.Antialiasing
            | QtGui.QPainter.RenderHint.SmoothPixmapTransform
            | QtGui.QPainter.RenderHint.TextAntialiasing
        )
        self.setDragMode(QtWidgets.QGraphicsView.DragMode.ScrollHandDrag)
        self._scale = 1.0
        self.setTransformationAnchor(QtWidgets.QGraphicsView.ViewportAnchor.NoAnchor)
        self.setResizeAnchor(QtWidgets.QGraphicsView.ViewportAnchor.NoAnchor)
        self.setMouseTracking(True)

    def wheelEvent(self, e: QtGui.QWheelEvent):
        if QtWidgets.QApplication.keyboardModifiers() == QtCore.Qt.KeyboardModifier.ControlModifier:
            factor = 1.0 + (0.0015 * e.angleDelta().y())
            self._scale = float(max(0.1, min(8.0, self._scale * factor)))
            target = self.mapToScene(e.position().toPoint())
            self.setTransformationAnchor(QtWidgets.QGraphicsView.ViewportAnchor.NoAnchor)
            self.setResizeAnchor(QtWidgets.QGraphicsView.ViewportAnchor.NoAnchor)
            self.setTransform(QtGui.QTransform())
            self.scale(self._scale, self._scale)
            self.centerOn(target)
            self.zoomChanged.emit(self._scale)
        else:
            super().wheelEvent(e)

```


## FILE: AI/image_manager.py
```text
# image_manager.py
# Faces and Pets manager with rename, delete, open folder, and Gallery launcher.
from __future__ import annotations
from pathlib import Path
from PyQt6 import QtWidgets
from utils import open_folder_or_warn
from settings import BASE_DIR
from gallery import GalleryDialog

class ImageManagerDialog(QtWidgets.QDialog):
    def __init__(self, app_cfg, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Image Manager")
        self.faces_root = BASE_DIR / "data" / "faces"
        self.pets_root  = BASE_DIR / "data" / "pets"

        self.tabs = QtWidgets.QTabWidget()
        self.faces_tab = self._build_tab(self.faces_root, is_pets=False)
        self.pets_tab  = self._build_tab(self.pets_root,  is_pets=True)
        self.tabs.addTab(self.faces_tab, "Faces")
        self.tabs.addTab(self.pets_tab,  "Pets")

        lay = QtWidgets.QVBoxLayout(self); lay.addWidget(self.tabs)

    def _build_tab(self, root: Path, is_pets: bool) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        lst = QtWidgets.QListWidget(); lst.setObjectName("list")
        btn_open = QtWidgets.QPushButton("Open Folder"); btn_refresh = QtWidgets.QPushButton("Refresh")
        btn_rename = QtWidgets.QPushButton("Rename…"); btn_delete = QtWidgets.QPushButton("Delete…")
        btn_gallery = QtWidgets.QPushButton("Open Gallery…")

        buttons = QtWidgets.QHBoxLayout()
        for b in (btn_open, btn_refresh, btn_rename, btn_delete, btn_gallery): buttons.addWidget(b)
        buttons.addStretch(1)

        lay = QtWidgets.QVBoxLayout(w); lay.addWidget(lst); lay.addLayout(buttons)

        def _refresh():
            lst.clear(); root.mkdir(parents=True, exist_ok=True)
            for d in sorted(root.glob("*")):
                if d.is_dir(): lst.addItem(d.name)
        def _sel_path():
            it = lst.currentItem()
            if not it: return None
            return root / it.text()
        def _open():
            open_folder_or_warn(self, root)
        def _rename():
            p = _sel_path()
            if not p: return
            new, ok = QtWidgets.QInputDialog.getText(self, "Rename", f"New name for '{p.name}':", text=p.name)
            if not ok or not new.strip(): return
            (root / new.strip()).mkdir(parents=True, exist_ok=True)
            for f in p.glob("*"): f.rename(root / new.strip() / f.name)
            try: p.rmdir()
            except Exception: pass
            _refresh()
        def _delete():
            p = _sel_path()
            if not p: return
            if QtWidgets.QMessageBox.question(self, "Delete", f"Delete '{p.name}' and all samples?") != QtWidgets.QMessageBox.StandardButton.Yes:
                return
            for f in p.glob("**/*"): f.unlink(missing_ok=True)
            try: p.rmdir()
            except Exception: pass
            _refresh()
        def _gallery():
            p = _sel_path()
            if not p: return
            GalleryDialog(p, parent=self).exec()

        btn_open.clicked.connect(_open)
        btn_refresh.clicked.connect(_refresh)
        btn_rename.clicked.connect(_rename)
        btn_delete.clicked.connect(_delete)
        btn_gallery.clicked.connect(_gallery)

        _refresh()
        return w

```


## FILE: AI/ip_cam_dialog.py
```text
# ip_cam_dialog.py
from __future__ import annotations
from typing import Optional
from PyQt6 import QtWidgets
from settings import CameraSettings

class AddIpCameraDialog(QtWidgets.QDialog):
    def __init__(self, app_cfg, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Camera by IP")
        self._app_cfg = app_cfg
        self._camera: Optional[CameraSettings] = None

        self.edit_name = QtWidgets.QLineEdit()
        self.edit_ip   = QtWidgets.QLineEdit()
        self.edit_user = QtWidgets.QLineEdit()
        self.edit_pass = QtWidgets.QLineEdit(); self.edit_pass.setEchoMode(QtWidgets.QLineEdit.EchoMode.Password)
        self.edit_token = QtWidgets.QLineEdit()

        # Sensible defaults
        default_name = f"Cam-{len(self._app_cfg.cameras) + 1}"
        self.edit_name.setText(default_name)

        form = QtWidgets.QFormLayout()
        form.addRow("Name", self.edit_name)
        form.addRow("IP address", self.edit_ip)
        form.addRow("Username (optional)", self.edit_user)
        form.addRow("Password (optional)", self.edit_pass)
        form.addRow("Token (optional)", self.edit_token)

        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok |
            QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self._on_accept)
        btns.rejected.connect(self.reject)

        lay = QtWidgets.QVBoxLayout(self)
        lay.addLayout(form)
        lay.addWidget(btns)

    def _on_accept(self):
        name = self.edit_name.text().strip()
        ip   = self.edit_ip.text().strip()
        user = self.edit_user.text().strip() or None
        pw   = self.edit_pass.text().strip() or None
        token = self.edit_token.text().strip() or None

        if not ip:
            QtWidgets.QMessageBox.warning(self, "Add Camera", "Enter an IP address.")
            return
        if not name:
            name = ip

        cam = CameraSettings.from_ip(name=name, host=ip, user=user, password=pw, token=token)
        self._camera = cam
        self.accept()

    def get_camera(self) -> Optional[CameraSettings]:
        return self._camera

```


## FILE: AI/mdi_app.py
```text
# mdi_app.py
from __future__ import annotations
import sys
from PyQt6 import QtCore, QtGui, QtWidgets

from settings import AppSettings, CameraSettings, load_settings, save_settings
from utils import open_folder_or_warn
from image_manager import ImageManagerDialog
from models import ModelManager
from enrollment import EnrollDialog
from enrollment_service import EnrollmentService
from events_pane import EventsPane
from discovery_dialog import DiscoveryDialog
from ip_cam_dialog import AddIpCameraDialog
from camera_widget import CameraWidget


class CameraSubWindow(QtWidgets.QMdiSubWindow):
    def __init__(
        self,
        cam_cfg: CameraSettings,
        app_cfg: AppSettings,
        settings: QtCore.QSettings,
        parent=None,
    ):
        super().__init__(parent)
        self._cam_name = cam_cfg.name
        self._settings = settings
        self.app_cfg = app_cfg
        self.setWindowTitle(cam_cfg.name)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, True)

        self.widget = CameraWidget(cam_cfg, app_cfg, self)
        self.setWidget(self.widget)

        geom = self._settings.value(f"cams/{self._cam_name}/geometry", None)
        if isinstance(geom, QtCore.QByteArray) and not geom.isEmpty():
            self.restoreGeometry(geom)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self._settings.setValue(f"cams/{self._cam_name}/geometry", self.saveGeometry())
        super().closeEvent(event)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, app_cfg: AppSettings, parent=None):
        super().__init__(parent)
        self.app_cfg = app_cfg
        self.setWindowTitle("ESP32-CAM AI Viewer")

        self.settings = QtCore.QSettings("ESP32CAM", "AIViewer")

        self.mdi = QtWidgets.QMdiArea()
        self.setCentralWidget(self.mdi)

        self.events_pane = EventsPane(self.app_cfg.logs_dir, parent=self)
        self.dock_events = QtWidgets.QDockWidget("Events", self)
        self.dock_events.setWidget(self.events_pane)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, self.dock_events)
        self.dock_events.hide()
        self._load_ui_state()
        self._build_menus()
        self._load_initial_cameras()

    def _load_ui_state(self):
        geom = self.settings.value("main/geometry")
        if isinstance(geom, QtCore.QByteArray) and not geom.isEmpty():
            self.restoreGeometry(geom)
        state = self.settings.value("main/windowState")
        if isinstance(state, QtCore.QByteArray) and not state.isEmpty():
            self.restoreState(state)

    def _save_ui_state(self):
        self.settings.setValue("main/geometry", self.saveGeometry())
        self.settings.setValue("main/windowState", self.saveState())
        # also persist each camera subwindow’s geometry
        for sub in self.mdi.subWindowList():
            if isinstance(sub, CameraSubWindow):
                self.settings.setValue(f"cams/{sub._cam_name}/geometry", sub.saveGeometry())

    # cameras
    def _load_initial_cameras(self):
        for cam in self.app_cfg.cameras:
            self._add_camera_window(cam)

    def _add_camera_window(self, cam_cfg: CameraSettings):
        sub = CameraSubWindow(cam_cfg, self.app_cfg, self.settings, self)
        self.mdi.addSubWindow(sub)
        sub.show()
        sub.widget.start()

    def _current_cam_window(self) -> CameraSubWindow | None:
        sub = self.mdi.currentSubWindow()
        if sub is None:
            return None
        if not isinstance(sub, CameraSubWindow):
            return None
        return sub

    # menus / actions
    def _build_menus(self):
        mb = self.menuBar()

        m_file = mb.addMenu("File")
        m_file.addAction("Open logs folder…").triggered.connect(
            lambda: open_folder_or_warn(self, self.app_cfg.logs_dir)
        )
        act_save = m_file.addAction("Save settings")
        act_save.triggered.connect(self._save_all_settings)
        m_file.addSeparator()
        m_file.addAction("Quit").triggered.connect(self.close)

        m_cams = mb.addMenu("Cameras")
        m_cams.addAction("Add camera by IP…").triggered.connect(self._add_camera_by_ip)

        m_tools = mb.addMenu("Tools")
        m_tools.addAction("Face enrollment…").triggered.connect(self._open_enrollment)
        m_tools.addAction("Open image manager…").triggered.connect(self._open_image_manager)
        m_tools.addSeparator()
        m_tools.addAction("Download default models…").triggered.connect(
            lambda: ModelManager.fetch_defaults(self, self.app_cfg)
        )
        m_tools.addSeparator()
        m_tools.addAction("Discover ESP32-CAMs…").triggered.connect(self._discover_esp32)
        m_tools.addAction("Delete Current Camera…").triggered.connect(self._delete_current_camera)

        m_view = mb.addMenu("View")
        m_view.addAction("Events pane").triggered.connect(self._toggle_events_pane)
        m_view.addSeparator()
        m_view.addAction("Tile Subwindows").triggered.connect(self.mdi.tileSubWindows)
        m_view.addAction("Cascade Subwindows").triggered.connect(self.mdi.cascadeSubWindows)
        m_view.addSeparator()
        m_view.addAction("Fit windows to video").triggered.connect(self._fit_windows_to_video_all)
        m_view.addAction("Fit All").triggered.connect(self._fit_all)
        m_view.addAction("100% All").triggered.connect(self._zoom_100_all)

    def _fit_windows_to_video_all(self):
        for sub in self.mdi.subWindowList():
            if isinstance(sub, CameraSubWindow):
                sub.widget.fit_window_to_video()

    # actions
    def _add_camera_by_ip(self):
        dlg = AddIpCameraDialog(self.app_cfg, self)
        if dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            cam_cfg = dlg.get_camera()
            if cam_cfg:
                self.app_cfg.add_or_update_camera(cam_cfg)
                save_settings(self.app_cfg)
                self._add_camera_window(cam_cfg)

    def _fit_windows_to_video_all(self):
        for sub in self.mdi.subWindowList():
            if isinstance(sub, CameraSubWindow):
                sub.widget.fit_window_to_video()

    def _delete_current_camera(self):
        sub = self._current_cam_window()
        if sub is None:
            return
        cam_name = sub._cam_name
        reply = QtWidgets.QMessageBox.question(
            self,
            "Delete Camera",
            f"Delete camera '{cam_name}' from configuration?",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
        )
        if reply != QtWidgets.QMessageBox.StandardButton.Yes:
            return

        # remove from config
        self.app_cfg.cameras = [c for c in self.app_cfg.cameras if c.name != cam_name]
        save_settings(self.app_cfg)

        # close window
        sub.close()

    def _toggle_events_pane(self):
        if self.dock_events.isVisible():
            self.dock_events.hide()
        else:
            self.dock_events.show()

    def _fit_all(self):
        for sub in self.mdi.subWindowList():
            if isinstance(sub, CameraSubWindow):
                sub.widget.fit_to_window()

    def _zoom_100_all(self):
        for sub in self.mdi.subWindowList():
            if isinstance(sub, CameraSubWindow):
                sub.widget.zoom_100()
    
    def _save_all_settings(self):
        # save window / dock / subwindow layout
        self._save_ui_state()
        # then persist app configuration (cameras, paths, etc.)
        save_settings(self.app_cfg)

    def _open_enrollment(self):
        EnrollDialog(self.app_cfg, self).exec()

    def _open_image_manager(self):
        ImageManagerDialog(self.app_cfg, self).exec()

    def _discover_esp32(self):
        DiscoveryDialog(self).exec()


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("ESP32-CAM AI Viewer")
    app_cfg = load_settings()
    win = MainWindow(app_cfg)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

```


## FILE: AI/models.py
```text
# models.py
# Uses settings.BASE_DIR-backed folders; unchanged logic with explicit BASE_DIR use.
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
import requests
from PyQt6 import QtWidgets
from settings import AppSettings, BASE_DIR
from utils import ensure_dir

YOLO_URL_DEFAULT = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.onnx"
HAAR_URL_DEFAULT = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"

@dataclass
class ModelPaths:
    yolo: Path
    haar: Path

class ModelManager:
    @staticmethod
    def paths(cfg: AppSettings) -> ModelPaths:
        return ModelPaths(
            yolo=Path(cfg.models_dir) / "yolov8n.onnx",
            haar=Path(cfg.models_dir) / "haarcascade_frontalface_default.xml"
        )

    @staticmethod
    def ensure_models(parent: QtWidgets.QWidget, cfg: AppSettings):
        p = ModelManager.paths(cfg)
        missing = []
        if not p.yolo.exists():
            missing.append(("YOLOv8n ONNX", p.yolo, cfg.yolo_url or YOLO_URL_DEFAULT))
        if not p.haar.exists():
            missing.append(("Haar face cascade", p.haar, cfg.haar_url or HAAR_URL_DEFAULT))
        if not missing:
            return
        mb = QtWidgets.QMessageBox(parent)
        mb.setWindowTitle("Models missing")
        mb.setText(f"Models folder:\n{cfg.models_dir}\n\nDownload defaults now?")
        mb.setStandardButtons(QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No)
        if mb.exec() == QtWidgets.QMessageBox.StandardButton.Yes:
            ModelManager._download_many(parent, missing)

    @staticmethod
    def fetch_defaults(parent: QtWidgets.QWidget, cfg: AppSettings):
        p = ModelManager.paths(cfg)
        todo = []
        if not p.yolo.exists():
            todo.append(("YOLOv8n ONNX", p.yolo, cfg.yolo_url or YOLO_URL_DEFAULT))
        if not p.haar.exists():
            todo.append(("Haar face cascade", p.haar, cfg.haar_url or HAAR_URL_DEFAULT))
        if not todo:
            QtWidgets.QMessageBox.information(parent, "Models", "All default models already present.")
            return
        ModelManager._download_many(parent, todo)

    @staticmethod
    def _download_many(parent: QtWidgets.QWidget, items: list[Tuple[str, Path, str]]):
        for label, path, url in items:
            ensure_dir(path.parent)
            prog = _DownloadDialog(parent, f"Downloading {label}…", url, path)
            ok = prog.exec_and_download()
            if not ok:
                QtWidgets.QMessageBox.warning(parent, "Download failed", f"Could not fetch {label}.\nURL: {url}")

class _DownloadDialog(QtWidgets.QDialog):
    def __init__(self, parent, title: str, url: str, path: Path):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.url = url
        self.path = path
        from PyQt6 import QtWidgets
        self.pb = QtWidgets.QProgressBar()
        self.lbl = QtWidgets.QLabel(url)
        self.btn = QtWidgets.QPushButton("Cancel")
        self.btn.clicked.connect(self.reject)
        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(self.lbl); lay.addWidget(self.pb); lay.addWidget(self.btn)
        self._ok = False

    def exec_and_download(self) -> bool:
        from PyQt6 import QtWidgets
        try:
            with requests.get(self.url, stream=True, timeout=(5, 60)) as r:
                r.raise_for_status()
                total = int(r.headers.get("Content-Length", 0)) or None
                got = 0
                with self.path.open("wb") as fp:
                    for chunk in r.iter_content(chunk_size=65536):
                        if not chunk: continue
                        fp.write(chunk); got += len(chunk)
                        if total:
                            self.pb.setMaximum(total); self.pb.setValue(got)
                        QtWidgets.QApplication.processEvents()
            self._ok = True
        except Exception:
            self._ok = False
        self.accept()
        return self._ok

```


## FILE: AI/overlays.py
```text
# overlays.py
# Overlay renderer with toggles and render order: boxes → labels → crosshair + HUD.
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import time

from PyQt6 import QtGui, QtCore
from detectors import DetectionPacket, DetBox


@dataclass
class OverlayFlags:
    yolo: bool = True
    faces: bool = True
    pets: bool = True
    tracks: bool = True
    hud: bool = True          # debug HUD (counts) top-left
    crosshair: bool = True    # central crosshair
    info: bool = True         # camera name + timestamp


def _pen(width: int, rgb: Tuple[int, int, int]) -> QtGui.QPen:
    p = QtGui.QPen(QtGui.QColor(*rgb))
    p.setWidth(width)
    return p


def _brush(rgb: Tuple[int, int, int], a: int) -> QtGui.QBrush:
    c = QtGui.QColor(*rgb)
    c.setAlpha(a)
    return QtGui.QBrush(c)


def draw_overlays(
    p: QtGui.QPainter,
    pkt: DetectionPacket,
    flags: OverlayFlags,
    cam_name: str | None = None,
    wall_ts: float | None = None,
):
    # Debug HUD (counts) in top-left
    if flags.hud:
        hud = f"yolo={len(pkt.yolo)} faces={len(pkt.faces)} pets={len(pkt.pets)}"
        p.setPen(_pen(2, (255, 255, 0)))
        p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        p.drawText(10, 20, hud)

    # Boxes
    if flags.yolo:
        for b in pkt.yolo:
            _draw_box(p, b, (0, 255, 0))
    if flags.faces:
        for b in pkt.faces:
            _draw_box(p, b, (0, 200, 255))
    if flags.pets:
        for b in pkt.pets:
            _draw_box(p, b, (255, 200, 0))

    # Labels
    if flags.yolo:
        for b in pkt.yolo:
            _draw_label(p, b.cls, b.score, b.xyxy, (0, 255, 0))
    if flags.faces:
        for b in pkt.faces:
            _draw_label(p, b.cls, b.score, b.xyxy, (0, 200, 255))
    if flags.pets:
        for b in pkt.pets:
            _draw_label(p, b.cls, b.score, b.xyxy, (255, 200, 0))

    # Crosshair at image center
    if flags.crosshair:
        cx, cy = pkt.size[0] // 2, pkt.size[1] // 2
        p.setPen(_pen(2, (255, 0, 255)))
        p.drawLine(cx - 16, cy, cx + 16, cy)
        p.drawLine(cx, cy - 16, cx, cy + 16)

    # Camera name + wall-clock timestamp at bottom-left
    if flags.info:
        label = cam_name or pkt.name
        if wall_ts is not None:
            stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(wall_ts))
            text = f"{label}  {stamp}"
        else:
            text = label

        fm = QtGui.QFontMetrics(p.font())
        tw = fm.horizontalAdvance(text) + 8
        th = fm.height() + 6
        img_w, img_h = pkt.size
        rect = QtCore.QRectF(10, img_h - th - 10, tw, th)

        p.setPen(QtCore.Qt.PenStyle.NoPen)
        p.setBrush(_brush((0, 0, 0), 160))
        p.drawRect(rect)

        p.setPen(_pen(1, (255, 255, 255)))
        p.drawText(
            rect,
            QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter,
            text,
        )


def _draw_box(p: QtGui.QPainter, b: DetBox, rgb):
    x1, y1, x2, y2 = b.xyxy
    rect = QtCore.QRectF(x1, y1, x2 - x1, y2 - y1)
    # Thicker pen and translucent fill to make it obvious
    p.setPen(_pen(3, rgb))
    p.setBrush(_brush(rgb, 60))
    p.drawRect(rect)


def _draw_label(p: QtGui.QPainter, name: str, score: float, xyxy, rgb):
    x1, y1, x2, y2 = xyxy
    text = f"{name} {score:.2f}"
    fm = QtGui.QFontMetrics(p.font())
    tw = fm.horizontalAdvance(text) + 6
    th = fm.height() + 4
    r = QtCore.QRectF(x1, max(0, y1 - th), tw, th)

    p.setPen(QtCore.Qt.PenStyle.NoPen)
    p.setBrush(_brush(rgb, 200))
    p.drawRect(r)

    p.setPen(_pen(1, (0, 0, 0)))
    p.drawText(
        r,
        QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter,
        text,
    )

```


## FILE: AI/presence.py
```text
# presence.py
# Uses YOLO or faces to generate events; prevents “no events” when YOLO is absent.
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Set
from detectors import DetectionPacket
from utils import ensure_dir

class PresenceBus:
    def __init__(self, cam_name: str, logs_dir: Path, ttl_ms: int = 2500):
        self.cam = cam_name
        self.logs_dir = Path(logs_dir)
        self.ttl = ttl_ms
        self.last_seen: Dict[str, int] = {}
        self.present: Set[str] = set()

    def update(self, pkt: DetectionPacket):
        now = pkt.ts_ms
        seen = set()
        # Prefer YOLO
        for b in pkt.yolo:
            if b.cls in ("person", "dog", "cat"):
                seen.add(b.cls)
        # Fallback: any face counts as a person presence
        if not seen and pkt.faces:
            seen.add("person")

        # update timestamps
        for k in seen:
            self.last_seen[k] = now

        # exit events
        for k in list(self.present):
            if now - self.last_seen.get(k, 0) > self.ttl:
                self.present.remove(k)
                self._write({"ts": now, "camera": self.cam, "event": "exit", "type": k})

        # enter events
        for k in seen:
            if k not in self.present:
                self.present.add(k)
                self._write({"ts": now, "camera": self.cam, "event": "enter", "type": k})

    def _write(self, rec: Dict):
        ensure_dir(self.logs_dir)
        f = self.logs_dir / f"{self.cam}.jsonl"
        with f.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(rec) + "\n")

```


## FILE: AI/ptz.py
```text
# ptz.py
# PTZ client stub.
from __future__ import annotations
from settings import CameraSettings

class PTZClient:
    def __init__(self, cam: CameraSettings):
        self.cam = cam

    def nudge(self, dx: int, dy: int):
        return

```


## FILE: AI/recorder.py
```text
# recorder.py
# Per-camera prebuffer recorder with flush-on-start.
from __future__ import annotations
from collections import deque
from pathlib import Path
import cv2 as cv
from typing import Optional
import numpy as np
from utils import timestamp_name, ensure_dir

class PrebufferRecorder:
    def __init__(self, cam_name: str, out_dir: Path, fps: int = 25, pre_ms: int = 3000):
        self.cam_name = cam_name
        self.out_dir = Path(out_dir)
        self.fps = fps
        self.pre_ms = pre_ms
        self.buf = deque()  # (ts_ms, bgr)
        self.writer: Optional[cv.VideoWriter] = None
        self.size = None

    def on_frame(self, bgr: np.ndarray, ts_ms: int):
        self.size = (bgr.shape[1], bgr.shape[0])
        self.buf.append((ts_ms, bgr.copy()))
        # Trim buffer
        while self.buf and ts_ms - self.buf[0][0] > self.pre_ms:
            self.buf.popleft()
        if self.writer is not None:
            self.writer.write(bgr)

    def start(self):
        if self.writer is not None:
            return
        ensure_dir(self.out_dir)
        fname = f"{self.cam_name}_{timestamp_name()}.avi"
        path = str(self.out_dir / fname)
        fourcc = cv.VideoWriter_fourcc(*"MJPG")
        self.writer = cv.VideoWriter(path, fourcc, self.fps, self.size)
        for _, b in self.buf:
            self.writer.write(b)

    def stop(self):
        if self.writer is None:
            return
        self.writer.release()
        self.writer = None

    def close(self):
        self.stop()
        self.buf.clear()

```


## FILE: AI/refactor\ai_viewer\__init__.py
```text
# ai_viewer package

```


## FILE: AI/refactor\ai_viewer\ai\__init__.py
```text
# package

```


## FILE: AI/refactor\ai_viewer\ai\detectors\__init__.py
```text
# package
__all__ = ["YOLODetector"]

```


## FILE: AI/refactor\ai_viewer\ai\detectors\yolo.py
```text
#!/usr/bin/env python3
"""
ESP32-CAM MDI Viewer (Qt)

Multi-document interface (MDI) master application to manage multiple
ESP32-CAM feeds as independent, floating, resizable windows inside a
single main window. Includes a standard toolbar with camera management
and recording controls. Supports basic MJPEG streaming with optional
Basic-Auth or token, plus pre-buffered video capture per camera.

Dependencies (install on your PC):
  - pip install PySide6 requests opencv-python numpy

Notes:
  - This is a first pass skeleton designed to get the MDI scaffolding,
    multi-camera streaming, and pre-buffered recording in place.
  - Face/pet recognition and the advanced UI from cam_ai.py can be
    integrated in phased steps by adding overlays and per-camera tool
    panels.
"""

from __future__ import annotations
import os
import sys
import time
import threading
from collections import deque
from dataclasses import dataclass
from typing import Optional, Deque, Tuple

import requests
import numpy as np
import cv2

from PySide6 import QtCore, QtGui, QtWidgets
import sys as _sys
_sys.path.append(os.path.dirname(__file__))  # allow local module imports
from gallery import GalleryDialog
import tools



class YOLODetector:
    """Lightweight YOLO (ONNX) wrapper for COCO classes.
    Uses OpenCV DNN and letterbox preprocessing.
    """
    COCO = [
        'person','bicycle','car','motorcycle','airplane','bus','train','truck','boat','traffic light',
        'fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow','elephant',
        'bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee','skis','snowboard',
        'sports ball','kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket','bottle',
        'wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich','orange','broccoli',
        'carrot','hot dog','pizza','donut','cake','chair','couch','potted plant','bed','dining table','toilet',
        'tv','laptop','mouse','remote','keyboard','cell phone','microwave','oven','toaster','sink','refrigerator',
        'book','clock','vase','scissors','teddy bear','hair drier','toothbrush'
    ]
    def __init__(self, model_path='ai/models/yolo.onnx', input_size=640, conf=0.35, iou=0.45):
        self.net = None
        self.size = input_size
        self.conf = conf
        self.iou = iou
        if os.path.exists(model_path):
            try:
                self.net = cv2.dnn.readNetFromONNX(model_path)
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            except Exception as e:
                print('[YOLO] load failed:', e)


```


## FILE: AI/refactor\ai_viewer\ai\recognition\__init__.py
```text
# package
__all__ = ["FaceDB", "PetsDB"]

```


## FILE: AI/refactor\ai_viewer\ai\recognition\face_db.py
```text
#!/usr/bin/env python3
"""
ESP32-CAM MDI Viewer (Qt)

Multi-document interface (MDI) master application to manage multiple
ESP32-CAM feeds as independent, floating, resizable windows inside a
single main window. Includes a standard toolbar with camera management
and recording controls. Supports basic MJPEG streaming with optional
Basic-Auth or token, plus pre-buffered video capture per camera.

Dependencies (install on your PC):
  - pip install PySide6 requests opencv-python numpy

Notes:
  - This is a first pass skeleton designed to get the MDI scaffolding,
    multi-camera streaming, and pre-buffered recording in place.
  - Face/pet recognition and the advanced UI from cam_ai.py can be
    integrated in phased steps by adding overlays and per-camera tool
    panels.
"""

from __future__ import annotations
import os
import sys
import time
import threading
from collections import deque
from dataclasses import dataclass
from typing import Optional, Deque, Tuple

import requests
import numpy as np
import cv2

from PySide6 import QtCore, QtGui, QtWidgets
import sys as _sys
_sys.path.append(os.path.dirname(__file__))  # allow local module imports
from gallery import GalleryDialog
import tools



class FaceDB:
    """Face recognition store.
    - Trains LBPH if contrib available; falls back to ORB matching.
    - Persists samples as images on disk under ai/data/faces/<name>.
    """
    def __init__(self, base='ai/data/faces'):
        self.base = base
        os.makedirs(self.base, exist_ok=True)
        self.size = (160,160)
        self.model = None
        try:
            self.model = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)
        except Exception:
            self.model = None
        self.labels=[]
        self.orb = cv2.ORB_create(nfeatures=1000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.db_descs={}
        self.cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


```


## FILE: AI/refactor\ai_viewer\ai\recognition\pets_db.py
```text
#!/usr/bin/env python3
"""
ESP32-CAM MDI Viewer (Qt)

Multi-document interface (MDI) master application to manage multiple
ESP32-CAM feeds as independent, floating, resizable windows inside a
single main window. Includes a standard toolbar with camera management
and recording controls. Supports basic MJPEG streaming with optional
Basic-Auth or token, plus pre-buffered video capture per camera.

Dependencies (install on your PC):
  - pip install PySide6 requests opencv-python numpy

Notes:
  - This is a first pass skeleton designed to get the MDI scaffolding,
    multi-camera streaming, and pre-buffered recording in place.
  - Face/pet recognition and the advanced UI from cam_ai.py can be
    integrated in phased steps by adding overlays and per-camera tool
    panels.
"""

from __future__ import annotations
import os
import sys
import time
import threading
from collections import deque
from dataclasses import dataclass
from typing import Optional, Deque, Tuple

import requests
import numpy as np
import cv2

from PySide6 import QtCore, QtGui, QtWidgets
import sys as _sys
_sys.path.append(os.path.dirname(__file__))  # allow local module imports
from gallery import GalleryDialog
import tools



class PetsDB:
    """Pet recognition store (dogs/cats) using ORB descriptors.
    - Persists samples as grayscale images under ai/data/pets/{dogs,cats}/<name>.
    """
    def __init__(self, base='ai/data/pets'):
        self.base=base; os.makedirs(os.path.join(base,'dogs'),exist_ok=True); os.makedirs(os.path.join(base,'cats'),exist_ok=True)
        self.orb=cv2.ORB_create(nfeatures=1000); self.bf=cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.db={'dogs':{},'cats':{}}


```


## FILE: AI/refactor\ai_viewer\core\__init__.py
```text
# package
__all__ = ["CameraConfig", "CameraStreamThread", "SimpleTracker"]

```


## FILE: AI/refactor\ai_viewer\core\config.py
```text
#!/usr/bin/env python3
"""
ESP32-CAM MDI Viewer (Qt)

Multi-document interface (MDI) master application to manage multiple
ESP32-CAM feeds as independent, floating, resizable windows inside a
single main window. Includes a standard toolbar with camera management
and recording controls. Supports basic MJPEG streaming with optional
Basic-Auth or token, plus pre-buffered video capture per camera.

Dependencies (install on your PC):
  - pip install PySide6 requests opencv-python numpy

Notes:
  - This is a first pass skeleton designed to get the MDI scaffolding,
    multi-camera streaming, and pre-buffered recording in place.
  - Face/pet recognition and the advanced UI from cam_ai.py can be
    integrated in phased steps by adding overlays and per-camera tool
    panels.
"""

from __future__ import annotations
import os
import sys
import time
import threading
from collections import deque
from dataclasses import dataclass
from typing import Optional, Deque, Tuple

import requests
import numpy as np
import cv2

from PySide6 import QtCore, QtGui, QtWidgets
import sys as _sys
_sys.path.append(os.path.dirname(__file__))  # allow local module imports
from gallery import GalleryDialog
import tools



class CameraConfig:
    name: str
    host: str                 # ip[:port] for port 80
    user: Optional[str] = None
    password: Optional[str] = None
    token: Optional[str] = None  # Base64 of user:pass


```


## FILE: AI/refactor\ai_viewer\core\stream.py
```text
#!/usr/bin/env python3
"""
ESP32-CAM MDI Viewer (Qt)

Multi-document interface (MDI) master application to manage multiple
ESP32-CAM feeds as independent, floating, resizable windows inside a
single main window. Includes a standard toolbar with camera management
and recording controls. Supports basic MJPEG streaming with optional
Basic-Auth or token, plus pre-buffered video capture per camera.

Dependencies (install on your PC):
  - pip install PySide6 requests opencv-python numpy

Notes:
  - This is a first pass skeleton designed to get the MDI scaffolding,
    multi-camera streaming, and pre-buffered recording in place.
  - Face/pet recognition and the advanced UI from cam_ai.py can be
    integrated in phased steps by adding overlays and per-camera tool
    panels.
"""

from __future__ import annotations
import os
import sys
import time
import threading
from collections import deque
from dataclasses import dataclass
from typing import Optional, Deque, Tuple

import requests
import numpy as np
import cv2

from PySide6 import QtCore, QtGui, QtWidgets
import sys as _sys
_sys.path.append(os.path.dirname(__file__))  # allow local module imports
from gallery import GalleryDialog
import tools



class CameraStreamThread(QtCore.QThread):
    frameReady = QtCore.Signal(np.ndarray, float)  # (bgr_frame, timestamp)


```


## FILE: AI/refactor\ai_viewer\core\tracking.py
```text
#!/usr/bin/env python3
"""
ESP32-CAM MDI Viewer (Qt)

Multi-document interface (MDI) master application to manage multiple
ESP32-CAM feeds as independent, floating, resizable windows inside a
single main window. Includes a standard toolbar with camera management
and recording controls. Supports basic MJPEG streaming with optional
Basic-Auth or token, plus pre-buffered video capture per camera.

Dependencies (install on your PC):
  - pip install PySide6 requests opencv-python numpy

Notes:
  - This is a first pass skeleton designed to get the MDI scaffolding,
    multi-camera streaming, and pre-buffered recording in place.
  - Face/pet recognition and the advanced UI from cam_ai.py can be
    integrated in phased steps by adding overlays and per-camera tool
    panels.
"""

from __future__ import annotations
import os
import sys
import time
import threading
from collections import deque
from dataclasses import dataclass
from typing import Optional, Deque, Tuple

import requests
import numpy as np
import cv2

from PySide6 import QtCore, QtGui, QtWidgets
import sys as _sys
_sys.path.append(os.path.dirname(__file__))  # allow local module imports
from gallery import GalleryDialog
import tools



class SimpleTracker:
    """Very simple IOU-based tracker with TTL for object permanence.
    Tracks: list of dicts with bbox, cls, last_ts.
    """
    def __init__(self, ttl: float = 1.0, iou_thresh: float = 0.3):
        self.ttl = ttl
        self.iou_thresh = iou_thresh
        self.tracks = []


```


## FILE: AI/refactor\ai_viewer\tools\dupes.py
```text
"""Utility tools for the Qt MDI app.
 - Image ingestion from a source directory into face/pet stores
 - Simple near-duplicate culling via average-hash (aHash)
"""
from __future__ import annotations
import os
import time
from typing import Tuple
import numpy as np
import cv2


def ingest_images(src_dir: str, dest_dir: str, size: Tuple[int,int], gray: bool=False) -> int:
    os.makedirs(dest_dir, exist_ok=True)
    count=0
    for root,_,files in os.walk(src_dir):
        for fn in files:
            if not fn.lower().endswith(('.jpg','.jpeg','.png')):
                continue
            try:
                img=cv2.imread(os.path.join(root,fn))
                if img is None: continue
                if gray:
                    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                img=cv2.resize(img, size)
                now=time.time(); ms=int((now-int(now))*1000); ds=time.strftime('%Y%m%d_%H%M%S', time.localtime(now))
                out=os.path.join(dest_dir, f'{ds}_{ms:03d}.jpg')
                cv2.imwrite(out, img)
                count+=1
            except Exception:
                pass
    return count


def cull_similar_in_dir(target: str, hash_size: int = 8, hamming_thresh: int = 4) -> int:
    """Remove near-duplicate images in a folder using aHash similarity.
    Returns number of removed files.
    """
    if not os.path.isdir(target):
        return 0
    files=[os.path.join(target,f) for f in os.listdir(target) if f.lower().endswith(('.jpg','.jpeg','.png'))]
    files.sort(key=lambda p: os.path.getmtime(p))
    hashes=[]; removed=0
    def ahash(path):
        try:
            img=cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None: return None
            img=cv2.resize(img,(hash_size,hash_size))
            avg=img.mean(); bits=(img>avg).astype(np.uint8)
            return bits
        except Exception:
            return None
    for fp in files:
        h=ahash(fp)
        if h is None: continue
        dup=False
        for hh in hashes:
            dist = int((h^hh).sum())
            if dist <= hamming_thresh:
                try:
                    os.remove(fp); removed+=1
                except Exception:
                    pass
                dup=True
                break
        if not dup:
            hashes.append(h)
    return removed


def find_similar_in_dir(target: str, hash_size: int = 8, hamming_thresh: int = 4):
    """Analyze a directory for near-duplicates by aHash.
    Returns (files:list[str], remove_indices:set[int]) where remove_indices
    contains indices in files suggested for deletion.
    """
    if not os.path.isdir(target):
        return [], set()
    files=[os.path.join(target,f) for f in os.listdir(target) if f.lower().endswith(('.jpg','.jpeg','.png'))]
    files.sort(key=lambda p: os.path.getmtime(p))
    hashes=[]; remove=set()
    def ahash(path):
        try:
            img=cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None: return None
            img=cv2.resize(img,(hash_size,hash_size))
            avg=img.mean(); bits=(img>avg).astype(np.uint8)
            return bits
        except Exception:
            return None
    for idx, fp in enumerate(files):
        h=ahash(fp)
        if h is None: continue
        for hh in hashes:
            dist = int((h^hh).sum())
            if dist <= hamming_thresh:
                remove.add(idx)
                break
        else:
            hashes.append(h)
    return files, remove

```


## FILE: AI/refactor\ai_viewer\tools\gallery.py
```text
"""Qt thumbnail gallery dialog for selecting and deleting images."""
from __future__ import annotations
import os
from PySide6 import QtCore, QtGui, QtWidgets


class GalleryDialog(QtWidgets.QDialog):
    def __init__(self, dir_path: str, title: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(900, 600)
        self.dir_path = dir_path
        v = QtWidgets.QVBoxLayout(self)
        self.list = QtWidgets.QListWidget()
        self.list.setViewMode(QtWidgets.QListView.ViewMode.IconMode)
        self.list.setIconSize(QtCore.QSize(160, 120))
        self.list.setResizeMode(QtWidgets.QListView.ResizeMode.Adjust)
        self.list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        v.addWidget(self.list, 1)
        btns = QtWidgets.QDialogButtonBox()
        self.btn_del = btns.addButton('Delete Selected', QtWidgets.QDialogButtonBox.ButtonRole.ActionRole)
        self.btn_close = btns.addButton(QtWidgets.QDialogButtonBox.StandardButton.Close)
        v.addWidget(btns)
        self.btn_del.clicked.connect(self.do_delete)
        self.btn_close.clicked.connect(self.accept)
        self.populate()

    def populate(self):
        self.list.clear()
        if not os.path.isdir(self.dir_path):
            return
        files = [os.path.join(self.dir_path,f) for f in os.listdir(self.dir_path)
                 if f.lower().endswith(('.jpg','.jpeg','.png'))]
        files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        for fp in files:
            item = QtWidgets.QListWidgetItem(os.path.basename(fp))
            try:
                pix = QtGui.QPixmap(fp)
                if not pix.isNull():
                    item.setIcon(QtGui.QIcon(pix.scaled(160,120, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)))
            except Exception:
                pass
            item.setData(QtCore.Qt.ItemDataRole.UserRole, fp)
            self.list.addItem(item)

    def do_delete(self):
        items = self.list.selectedItems()
        if not items:
            return
        if QtWidgets.QMessageBox.question(self, 'Delete', f'Delete {len(items)} images?') != QtWidgets.QMessageBox.Yes:
            return
        cnt=0
        for it in items:
            fp = it.data(QtCore.Qt.ItemDataRole.UserRole)
            try:
                os.remove(fp)
                cnt+=1
            except Exception:
                pass
        self.populate()
        QtWidgets.QMessageBox.information(self, 'Manage', f'Deleted {cnt} files')


```


## FILE: AI/refactor\ai_viewer\ui\__init__.py
```text
# package
__all__ = ["AddCameraDialog", "CameraWidget", "CollectionDialog", "CullDialog", "DetectionThread", "MainWindow", "SettingsDialog"]

```


## FILE: AI/refactor\ai_viewer\ui\camera_widget.py
```text
#!/usr/bin/env python3
"""
ESP32-CAM MDI Viewer (Qt)

Multi-document interface (MDI) master application to manage multiple
ESP32-CAM feeds as independent, floating, resizable windows inside a
single main window. Includes a standard toolbar with camera management
and recording controls. Supports basic MJPEG streaming with optional
Basic-Auth or token, plus pre-buffered video capture per camera.

Dependencies (install on your PC):
  - pip install PySide6 requests opencv-python numpy

Notes:
  - This is a first pass skeleton designed to get the MDI scaffolding,
    multi-camera streaming, and pre-buffered recording in place.
  - Face/pet recognition and the advanced UI from cam_ai.py can be
    integrated in phased steps by adding overlays and per-camera tool
    panels.
"""

from __future__ import annotations
import os
import sys
import time
import threading
from collections import deque
from dataclasses import dataclass
from typing import Optional, Deque, Tuple

import requests
import numpy as np
import cv2

from PySide6 import QtCore, QtGui, QtWidgets
import sys as _sys
_sys.path.append(os.path.dirname(__file__))  # allow local module imports
from gallery import GalleryDialog
import tools



class CollectionDialog(QtWidgets.QDialog):
    stopClicked = QtCore.Signal()
    def __init__(self, parent=None, title: str = 'Collection'):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(False)
        self.setWindowFlag(QtCore.Qt.WindowType.WindowStaysOnTopHint, True)
        self.resize(360, 120)
        v = QtWidgets.QVBoxLayout(self)
        self.lbl = QtWidgets.QLabel('Starting...')
        self.lbl.setWordWrap(True)
        v.addWidget(self.lbl, 1)
        btns = QtWidgets.QDialogButtonBox()
        self.btn_stop = btns.addButton('Stop', QtWidgets.QDialogButtonBox.ActionRole)
        self.btn_close = btns.addButton(QtWidgets.QDialogButtonBox.Close)
        v.addWidget(btns)
        self.btn_stop.clicked.connect(lambda: self.stopClicked.emit())
        self.btn_close.clicked.connect(self.accept)
    def set_status(self, text: str):
        try:
            self.lbl.setText(text)
        except Exception:
            pass

class CameraWidget(QtWidgets.QWidget):
    closed = QtCore.Signal(dict)
    eventLogged = QtCore.Signal(str)
    def __init__(self, cfg: CameraConfig, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self.setWindowTitle(f"{cfg.name} [{cfg.host}]")
        self.label = QtWidgets.QLabel('Connecting…')
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setMinimumSize(320, 240)
        self.label.setStyleSheet('background:#000; color:#9cf;')

class DetectionThread(QtCore.QThread):
    resultsReady = QtCore.Signal(list, list)  # dets, faces
    def __init__(self, widget: CameraWidget, interval_ms: int = 200):
        super().__init__(widget)
        self.w = widget
        self.interval = interval_ms
        self._stop = threading.Event()
        self._skip_cycles = 0
        self.max_skip_cycles = 1  # skip this many cycles if tracks are active


```


## FILE: AI/refactor\ai_viewer\ui\windows.py
```text
#!/usr/bin/env python3
"""
ESP32-CAM MDI Viewer (Qt)

Multi-document interface (MDI) master application to manage multiple
ESP32-CAM feeds as independent, floating, resizable windows inside a
single main window. Includes a standard toolbar with camera management
and recording controls. Supports basic MJPEG streaming with optional
Basic-Auth or token, plus pre-buffered video capture per camera.

Dependencies (install on your PC):
  - pip install PySide6 requests opencv-python numpy

Notes:
  - This is a first pass skeleton designed to get the MDI scaffolding,
    multi-camera streaming, and pre-buffered recording in place.
  - Face/pet recognition and the advanced UI from cam_ai.py can be
    integrated in phased steps by adding overlays and per-camera tool
    panels.
"""

from __future__ import annotations
import os
import sys
import time
import threading
from collections import deque
from dataclasses import dataclass
from typing import Optional, Deque, Tuple

import requests
import numpy as np
import cv2

from PySide6 import QtCore, QtGui, QtWidgets
import sys as _sys
_sys.path.append(os.path.dirname(__file__))  # allow local module imports
from gallery import GalleryDialog
import tools



class AddCameraDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, initial: Optional[CameraConfig]=None, title: str='Add Camera'):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        form = QtWidgets.QFormLayout(self)
        self.ed_name = QtWidgets.QLineEdit(initial.name if initial else 'Camera')
        self.ed_host = QtWidgets.QLineEdit(initial.host if initial else '192.168.1.100')
        self.ed_user = QtWidgets.QLineEdit(initial.user or '' if initial else '')
        self.ed_pass = QtWidgets.QLineEdit(initial.password or '' if initial else '')
        self.ed_pass.setEchoMode(QtWidgets.QLineEdit.Password)
        self.ed_token = QtWidgets.QLineEdit(initial.token or '' if initial else '')
        form.addRow('Name', self.ed_name)
        form.addRow('Host (ip[:port])', self.ed_host)
        form.addRow('User', self.ed_user)
        form.addRow('Password', self.ed_pass)
        form.addRow('Token (Base64 user:pass)', self.ed_token)
        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        form.addRow(btns)

class SettingsDialog(QtWidgets.QDialog):
    """App settings: detector interval, aHash grid size + Hamming, PTZ aim timing, deadzone."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Settings')
        self.resize(420, 240)
        form = QtWidgets.QFormLayout(self)
        self.s_det_interval = QtWidgets.QSpinBox(); self.s_det_interval.setRange(50, 2000); self.s_det_interval.setSingleStep(50); self.s_det_interval.setSuffix(' ms')
        self.s_hash_size = QtWidgets.QSpinBox(); self.s_hash_size.setRange(4, 16)
        self.s_hamming = QtWidgets.QSpinBox(); self.s_hamming.setRange(0, 32)
        self.s_ptz_interval = QtWidgets.QSpinBox(); self.s_ptz_interval.setRange(100, 2000); self.s_ptz_interval.setSingleStep(50); self.s_ptz_interval.setSuffix(' ms')
        self.s_deadzone = QtWidgets.QSpinBox(); self.s_deadzone.setRange(2, 20); self.s_deadzone.setSuffix(' %')
        form.addRow('Detector interval', self.s_det_interval)
        form.addRow('aHash grid size', self.s_hash_size)
        form.addRow('Hamming threshold', self.s_hamming)
        form.addRow('PTZ aim interval', self.s_ptz_interval)
        form.addRow('PTZ deadzone', self.s_deadzone)
        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept); btns.rejected.connect(self.reject)
        form.addRow(btns)

class CullDialog(QtWidgets.QDialog):
    """Preview duplicates with highlight before deletion.
    Includes a tolerance slider to adjust the near-duplicate matching threshold.
    Lower values are stricter; higher tolerate more difference.
    """
    def __init__(self, dir_path: str, files: list[str], remove: set[int], title: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(1000, 700)
        self.dir_path = dir_path
        self.files = files or []
        self.remove = set(remove or set())
        # Matching parameters (aHash on 8x8 grid → Hamming distance 0..64).
        # Practical duplicate tolerance range ~0..16; start at 4 by default.
        self.hash_size = 8
        self.thresh = 4
        v = QtWidgets.QVBoxLayout(self)
        self.view = QtWidgets.QListWidget()
        self.view.setViewMode(QtWidgets.QListView.ViewMode.IconMode)
        self.view.setIconSize(QtCore.QSize(160, 120))
        self.view.setResizeMode(QtWidgets.QListView.ResizeMode.Adjust)
        self.view.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        v.addWidget(self.view, 1)
        # Controls
        h = QtWidgets.QHBoxLayout()
        # Tolerance slider block
        tol_box = QtWidgets.QHBoxLayout()
        self.lbl_tol = QtWidgets.QLabel('Tolerance:')
        self.sld_tol = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.sld_tol.setMinimum(0)
        self.sld_tol.setMaximum(16)
        self.sld_tol.setValue(self.thresh)
        self.sld_tol.setTickInterval(1)
        self.sld_tol.setSingleStep(1)
        self.lbl_tol_val = QtWidgets.QLabel(str(self.thresh))
        tol_box.addWidget(self.lbl_tol)
        tol_box.addWidget(self.sld_tol)
        tol_box.addWidget(self.lbl_tol_val)
        tol_box_w = QtWidgets.QWidget(); tol_box_w.setLayout(tol_box)
        h.addWidget(tol_box_w, 2)
        # Info + actions
        self.lbl_info = QtWidgets.QLabel()
        h.addWidget(self.lbl_info, 1)
        self.btn_confirm = QtWidgets.QPushButton('Confirm Delete')
        self.btn_cancel = QtWidgets.QPushButton('Cancel')
        h.addWidget(self.btn_confirm)
        h.addWidget(self.btn_cancel)
        v.addLayout(h)
        self.btn_confirm.clicked.connect(self.on_confirm)
        self.btn_cancel.clicked.connect(self.reject)
        self.sld_tol.valueChanged.connect(self.on_tol_changed)
        self.populate()

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('ESP32-CAM MDI')
        self.resize(1200, 800)
        self.mdi = QtWidgets.QMdiArea()
        self.setCentralWidget(self.mdi)
        # Events sidebar (shows events for active camera)
        self.eventsDock = QtWidgets.QDockWidget('Events', self)
        self.eventsView = QtWidgets.QListWidget()
        self.eventsDock.setWidget(self.eventsView)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.eventsDock)
        self._active_cam_ref = None
        try:
            self.mdi.subWindowActivated.connect(self.on_sub_activated)
        except Exception:
            pass


def main():
    app = QtWidgets.QApplication(sys.argv)
    mw = MainWindow()
    mw.show()
    sys.exit(app.exec())

```


## FILE: AI/refactor\run_ai_viewer.py
```text
#!/usr/bin/env python3
from ai_viewer.ui.windows import main

if __name__ == "__main__":
    main()
```


## FILE: AI/settings.py
```text
# settings.py
# Base-path aware settings. Paths are anchored to the AI folder (this file's parent).
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Optional
from pathlib import Path
import json
import os

# --- Project base directory (…/ESP32_CAM_AI/AI) ---
BASE_DIR = Path(__file__).resolve().parent

SETTINGS_FILE = BASE_DIR / "config" / "app_settings.json"

def _abs_under_base(p: Path) -> Path:
    # Make absolute under BASE_DIR if relative
    if not p.is_absolute():
        p = BASE_DIR / p
    return p

@dataclass
class CameraSettings:
    name: str
    stream_url: str
    user: Optional[str] = None
    password: Optional[str] = None
    token: Optional[str] = None

    @classmethod
    def from_ip(cls, name: str, host: str, user: Optional[str] = None,
                password: Optional[str] = None, token: Optional[str] = None):
        host = host.strip()
        base = f"http://{host}:81/stream"
        if token:
            sep = "&" if "?" in base else "?"
            base = f"{base}{sep}token={token}"
        return cls(name=name, stream_url=base, user=user, password=password, token=token)

    def effective_url(self) -> str:
        return self.stream_url

@dataclass
class AppSettings:
    models_dir: Path = Path("models")
    output_dir: Path = Path("recordings")
    logs_dir: Path = Path("logs")
    detect_interval_ms: int = 100
    thresh_yolo: float = 0.35
    prebuffer_ms: int = 3000
    yolo_url: Optional[str] = None
    haar_url: Optional[str] = None
    cameras: List[CameraSettings] = field(default_factory=list)

def load_settings() -> AppSettings:
    if SETTINGS_FILE.exists():
        with SETTINGS_FILE.open("r", encoding="utf-8") as fp:
            raw = json.load(fp)
        cams = [CameraSettings(**c) for c in raw.get("cameras", [])]
        cfg = AppSettings(
            models_dir=Path(raw.get("models_dir", "models")),
            output_dir=Path(raw.get("output_dir", "recordings")),
            logs_dir=Path(raw.get("logs_dir", "logs")),
            detect_interval_ms=int(raw.get("detect_interval_ms", 100)),
            thresh_yolo=float(raw.get("thresh_yolo", 0.35)),
            prebuffer_ms=int(raw.get("prebuffer_ms", 3000)),
            yolo_url=raw.get("yolo_url"),
            haar_url=raw.get("haar_url"),
            cameras=cams
        )
    else:
        cfg = AppSettings()

    # Anchor to BASE_DIR
    cfg.models_dir = _abs_under_base(cfg.models_dir)
    cfg.output_dir = _abs_under_base(cfg.output_dir)
    cfg.logs_dir   = _abs_under_base(cfg.logs_dir)
    return cfg

def save_settings(cfg: AppSettings):
    SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    data = {
        # store as relative-to-base where possible for portability
        "models_dir": str(Path(cfg.models_dir).resolve().relative_to(BASE_DIR) if str(cfg.models_dir).startswith(str(BASE_DIR)) else str(cfg.models_dir)),
        "output_dir": str(Path(cfg.output_dir).resolve().relative_to(BASE_DIR) if str(cfg.output_dir).startswith(str(BASE_DIR)) else str(cfg.output_dir)),
        "logs_dir":   str(Path(cfg.logs_dir).resolve().relative_to(BASE_DIR) if str(cfg.logs_dir).startswith(str(BASE_DIR)) else str(cfg.logs_dir)),
        "detect_interval_ms": cfg.detect_interval_ms,
        "thresh_yolo": cfg.thresh_yolo,
        "prebuffer_ms": cfg.prebuffer_ms,
        "yolo_url": cfg.yolo_url,
        "haar_url": cfg.haar_url,
        "cameras": [asdict(c) for c in cfg.cameras],
    }
    with SETTINGS_FILE.open("w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=2)

```


## FILE: AI/src\app\main.py
```text
# app/main.py

import sys
from PySide6.QtWidgets import QApplication
from ui.main_window import MainWindow

def main() -> None:
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()

```


## FILE: AI/src\camera\detection.py
```text
# camera/detection.py

from PySide6 import QtCore
import threading
import time
from typing import List, Tuple

class DetectionThread(QtCore.QThread):
    resultsReady = QtCore.Signal(list, list)  # dets, faces

    def __init__(self, widget: 'CameraWidget', interval_ms: int = 200, parent=None):
        super().__init__(parent)
        self.w = widget
        self.interval = interval_ms
        self._stop = threading.Event()
        self.max_skip_cycles = 1

    # … (same implementation as before) …
```


## FILE: AI/src\camera\stream.py
```text
# camera/stream.py

import requests
import numpy as np
import cv2
from collections import deque
from typing import Deque, Tuple
from PySide6 import QtCore
import os
import time
import threading

class CameraStreamThread(QtCore.QThread):
    frameReady = QtCore.Signal(np.ndarray, float)  # (frame, ts)

    def __init__(self, cfg: CameraConfig, prebuffer_seconds: float = 5.0, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self.prebuffer: Deque[Tuple[np.ndarray, float]] = deque(
            maxlen=int(prebuffer_seconds * 20))
        self._stop = threading.Event()
        self._session = None
        self._resp = None
        self._buf = bytearray()

    # … (same implementation as before, only moved into its own file) …
```


## FILE: AI/src\ui\dialogs.py
```text
# ui/dialogs.py

from PySide6 import QtWidgets
from ..core.config import CameraConfig

class AddCameraDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, initial: CameraConfig | None = None, title: str = 'Add Camera'):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        form = QtWidgets.QFormLayout(self)

        self.ed_name   = QtWidgets.QLineEdit(initial.name if initial else 'Camera')
        self.ed_host   = QtWidgets.QLineEdit(initial.host if initial else '192.168.1.100')
        self.ed_user   = QtWidgets.QLineEdit(initial.user or '')
        self.ed_pass   = QtWidgets.QLineEdit(initial.password or '')
        self.ed_pass.setEchoMode(QtWidgets.QLineEdit.Password)
        self.ed_token  = QtWidgets.QLineEdit(initial.token or '')

        form.addRow('Name',   self.ed_name)
        form.addRow('Host',   self.ed_host)
        form.addRow('User',   self.ed_user)
        form.addRow('Pass',   self.ed_pass)
        form.addRow('Token',  self.ed_token)

        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        form.addRow(btns)

    def get_config(self) -> CameraConfig | None:
        if self.exec() == QtWidgets.QDialog.Accepted:
            return CameraConfig(
                name=self.ed_name.text().strip() or 'Camera',
                host=self.ed_host.text().strip(),
                user=self.ed_user.text().strip() or None,
                password=self.ed_pass.text(),
                token=self.ed_token.text().strip() or None
            )
        return None
```


## FILE: AI/src\ui\widgets.py
```text
# ui/widgets.py

from PySide6 import QtCore, QtGui, QtWidgets
import cv2
import numpy as np
import os
import time
from ..ai.yolo import YOLODetector
from ..ai.facedb import FaceDB
from ..ai.petsdb import PetsDB
from ..ai.tracker import SimpleTracker
from ..camera.detection import DetectionThread
from ..camera.stream import CameraStreamThread
from ..core.config import CameraConfig
from ..core.settings_dialog import SettingsDialog

class CameraWidget(QtWidgets.QWidget):
    """Widget that displays a single ESP32‑CAM feed and handles all per‑camera UI."""
    closed = QtCore.Signal(dict)          # emitted on close
    eventLogged = QtCore.Signal(str)      # emits a log line

    def __init__(self, cfg: CameraConfig, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self.setWindowTitle(f"{cfg.name} [{cfg.host}]")
        self._setup_ui()
        self._init_ai_components()
        self._init_threads()
        self._recording = False
        self._writer = None
        self._aim_timer = QtCore.QTimer(self)
        self._aim_timer.setInterval(300)          # PTZ aim interval
        self._aim_timer.timeout.connect(self._aim_at_target)

    # ------------------------------------------------------------------
    #  UI set‑up (toolbar, labels, etc.)
    # ------------------------------------------------------------------
    def _setup_ui(self):
        self.lbl = QtWidgets.QLabel('Connecting…')
        self.lbl.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl.setMinimumSize(320, 240)
        self.lbl.setStyleSheet('background:#000; color:#9cf;')

        toolbar = QtWidgets.QToolBar()
        toolbar.setMovable(False)
        toolbar.addAction('Start', self.start_stream)
        toolbar.addAction('Stop',  self.stop_stream)
        toolbar.addSeparator()
        toolbar.addAction('Start Rec', self.start_recording)
        toolbar.addAction('Stop Rec',  self.stop_recording)

        # AI toggles
        self.chk_yolo = QtWidgets.QCheckBox('YOLO')
        self.chk_face = QtWidgets.QCheckBox('Face')
        self.chk_dogid = QtWidgets.QCheckBox('Dog ID')
        ai_btn = QtWidgets.QToolButton()
        ai_btn.setText('AI')
        ai_btn.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        ai_menu = QtWidgets.QMenu(ai_btn)
        ai_menu.addAction('YOLO', self.chk_yolo.toggle)
        ai_menu.addAction('Face', self.chk_face.toggle)
        ai_menu.addAction('Dog ID', self.chk_dogid.toggle)
        ai_btn.setMenu(ai_menu)

        self.lbl_status = QtWidgets.QLabel('Ready')
        self.lbl_status.setStyleSheet('color:#246; padding:2px 4px;')

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(toolbar)
        layout.addWidget(self.lbl, 1)
        layout.setContentsMargins(4,4,4,4)

    # ------------------------------------------------------------------
    #  AI / DB helpers
    # ------------------------------------------------------------------
    def _init_ai_components(self):
        self.yolo   = YOLODetector()
        self.facedb = FaceDB()
        self.facedb.load()
        self.pets   = PetsDB()
        self.pets.load()
        self.tracker = SimpleTracker(ttl=1.0)

    # ------------------------------------------------------------------
    #  Threads
    # ------------------------------------------------------------------
    def _init_threads(self):
        self.stream_thread = CameraStreamThread(self.cfg)
        self.stream_thread.frameReady.connect(self._on_frame)
        self.stream_thread.start()

        self.det_thr = DetectionThread(self)
        self.det_thr.resultsReady.connect(self._on_results)
        self.det_thr.start()

    # ------------------------------------------------------------------
    #  Public API (mostly UI actions)
    # ------------------------------------------------------------------
    def start_stream(self):
        if not self.stream_thread.isRunning():
            self.stream_thread._stop.clear()
            self.stream_thread.start()

    def stop_stream(self):
        if self.stream_thread.isRunning():
            self.stream_thread.stop()
            self.stream_thread.wait(700)   # graceful shutdown

    # … (recording, enrollment, collection, etc.) …
```


## FILE: AI/stream\__init__.py
```text
# stream/__init__.py
from .capture import StreamCapture

__all__ = ["StreamCapture"]

```


## FILE: AI/stream\capture.py
```text
# stream/capture.py
from __future__ import annotations

import threading
import queue
import time
from typing import Optional, Tuple
from urllib.parse import urlparse

import numpy as np

from settings import CameraSettings
from .opencv_backend import run_opencv_backend
from .mjpeg_backend import run_mjpeg_backend


class StreamCapture:
    """Unified capture for a single camera with backend switching."""
    def __init__(self, cam: CameraSettings):
        self.cam = cam
        self._stop = threading.Event()
        self._q: "queue.Queue[Tuple[bool, Optional[np.ndarray], int]]" = queue.Queue(maxsize=2)
        self._t: Optional[threading.Thread] = None
        self.last_backend: str = "init"

    # --- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        if self._t and self._t.is_alive():
            return
        self._stop.clear()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def stop(self) -> None:
        self._stop.set()
        if self._t:
            self._t.join(timeout=1.0)

    def read(self) -> Tuple[bool, Optional[np.ndarray], int]:
        """Return (ok, frame, ts_ms). If no frame is ready, ok is False."""
        try:
            ok, frame, ts = self._q.get(timeout=0.25)
            return ok, frame, ts
        except queue.Empty:
            return False, None, 0

    # --- internals ---------------------------------------------------------
    def _run(self) -> None:
        url = self.cam.effective_url()
        parsed = urlparse(url)

        while not self._stop.is_set():
            try:
                if parsed.scheme in ("rtsp",):
                    ok = run_opencv_backend(self, url)
                    if not ok:
                        self._fail_once("cv-no-rtsp")
                elif parsed.scheme in ("http", "https"):
                    ok = run_opencv_backend(self, url)
                    if not ok:
                        ok = run_mjpeg_backend(self, url)
                        if not ok:
                            self._fail_once("mjpeg-fail")
                else:
                    self._fail_once("bad-url")

                # short backoff before retrying a broken stream
                self._sleep_with_cancel(1.0)
            except Exception:
                # Generic safety net: report once and back off a little
                self._fail_once("exception")
                self._sleep_with_cancel(1.0)

    def _offer(self, ok: bool, frame: Optional[np.ndarray], ts_ms: int) -> None:
        """Push a frame (or failure) into the small queue, dropping the oldest."""
        if self._q.full():
            try:
                self._q.get_nowait()
            except queue.Empty:
                pass
        self._q.put((ok, frame, ts_ms))

    def _fail_once(self, tag: str) -> None:
        self.last_backend = f"disconnected:{tag}"
        self._offer(False, None, 0)

    def _sleep_with_cancel(self, sec: float) -> None:
        t0 = time.time()
        while not self._stop.is_set() and time.time() - t0 < sec:
            time.sleep(0.05)

```


## FILE: AI/stream\mjpeg_backend.py
```text
# stream/mjpeg_backend.py
from __future__ import annotations

from typing import Optional, TYPE_CHECKING

import cv2 as cv
import numpy as np
import requests

from utils import monotonic_ms

if TYPE_CHECKING:
    from .capture import StreamCapture


def run_mjpeg_backend(owner: "StreamCapture", url: str) -> bool:
    """HTTP MJPEG fallback for ESP32-CAM style endpoints."""
    owner.last_backend = "http-mjpeg"

    auth: Optional[tuple[str, str]] = None
    if owner.cam.user and owner.cam.password:
        auth = (owner.cam.user, owner.cam.password)

    headers = {
        "User-Agent": "ESP32-CAM-AI/1.0",
        "Accept": "*/*",
        "Connection": "keep-alive",
    }

    try:
        resp = requests.get(
            url,
            stream=True,
            timeout=(5.0, 10.0),
            auth=auth,
            headers=headers,
        )
        resp.raise_for_status()
    except requests.RequestException:
        return False

    buf = bytearray()
    first = True

    try:
        for chunk in resp.iter_content(chunk_size=1024):
            if owner._stop.is_set():
                break
            if not chunk:
                continue

            buf.extend(chunk)

            # Extract as many JPEG frames as are currently complete
            while True:
                start = buf.find(b"\xff\xd8")  # SOI
                if start == -1:
                    break
                end = buf.find(b"\xff\xd9", start + 2)  # EOI
                if end == -1:
                    break

                jpg = bytes(buf[start : end + 2])
                del buf[: end + 2]

                arr = np.frombuffer(jpg, dtype=np.uint8)
                frame = cv.imdecode(arr, cv.IMREAD_COLOR)
                if frame is None:
                    continue

                ts = monotonic_ms()
                owner._offer(True, frame, ts)
                first = False
    finally:
        resp.close()

    return not first

```


## FILE: AI/stream\opencv_backend.py
```text
# stream/opencv_backend.py
from __future__ import annotations

from typing import Optional, TYPE_CHECKING
from urllib.parse import urlparse, urlunparse

import cv2 as cv

from utils import monotonic_ms

if TYPE_CHECKING:  # only for type hints, avoids runtime cycle
    from .capture import StreamCapture


def _build_auth_url(url: str, user: Optional[str], password: Optional[str]) -> str:
    """Inject basic-auth credentials into the URL if both user and password are set."""
    if not user or not password:
        return url

    p = urlparse(url)
    if not p.hostname:
        return url

    netloc = f"{user}:{password}@{p.hostname}"
    if p.port:
        netloc += f":{p.port}"
    return urlunparse((p.scheme, netloc, p.path or "", p.params, p.query, p.fragment))


def run_opencv_backend(owner: "StreamCapture", url: str) -> bool:
    """RTSP/HTTP stream using OpenCV / FFMPEG."""
    owner.last_backend = "cv-ffmpeg"
    full_url = _build_auth_url(url, owner.cam.user, owner.cam.password)

    cap = cv.VideoCapture(full_url)
    if not cap.isOpened():
        cap.release()
        return False

    # Small buffer reduces latency and memory ping-pong
    try:
        cap.set(cv.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    first = True
    idle_limit_ms = 5000
    last_ts = monotonic_ms()

    try:
        while not owner._stop.is_set():
            ok, frame = cap.read()
            now = monotonic_ms()
            if not ok or frame is None:
                if now - last_ts > idle_limit_ms:
                    break
                continue

            ts = now
            last_ts = ts
            owner._offer(True, frame, ts)
            first = False
    finally:
        cap.release()

    return not first


```


## FILE: AI/tools.py
```text
"""Utility tools for the Qt MDI app.
 - Image ingestion from a source directory into face/pet stores
 - Simple near-duplicate culling via average-hash (aHash)
"""
from __future__ import annotations
import os
import time
from typing import Tuple
import numpy as np
import cv2


def ingest_images(src_dir: str, dest_dir: str, size: Tuple[int,int], gray: bool=False) -> int:
    os.makedirs(dest_dir, exist_ok=True)
    count=0
    for root,_,files in os.walk(src_dir):
        for fn in files:
            if not fn.lower().endswith(('.jpg','.jpeg','.png')):
                continue
            try:
                img=cv2.imread(os.path.join(root,fn))
                if img is None: continue
                if gray:
                    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                img=cv2.resize(img, size)
                now=time.time(); ms=int((now-int(now))*1000); ds=time.strftime('%Y%m%d_%H%M%S', time.localtime(now))
                out=os.path.join(dest_dir, f'{ds}_{ms:03d}.jpg')
                cv2.imwrite(out, img)
                count+=1
            except Exception:
                pass
    return count


def cull_similar_in_dir(target: str, hash_size: int = 8, hamming_thresh: int = 4) -> int:
    """Remove near-duplicate images in a folder using aHash similarity.
    Returns number of removed files.
    """
    if not os.path.isdir(target):
        return 0
    files=[os.path.join(target,f) for f in os.listdir(target) if f.lower().endswith(('.jpg','.jpeg','.png'))]
    files.sort(key=lambda p: os.path.getmtime(p))
    hashes=[]; removed=0
    def ahash(path):
        try:
            img=cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None: return None
            img=cv2.resize(img,(hash_size,hash_size))
            avg=img.mean(); bits=(img>avg).astype(np.uint8)
            return bits
        except Exception:
            return None
    for fp in files:
        h=ahash(fp)
        if h is None: continue
        dup=False
        for hh in hashes:
            dist = int((h^hh).sum())
            if dist <= hamming_thresh:
                try:
                    os.remove(fp); removed+=1
                except Exception:
                    pass
                dup=True
                break
        if not dup:
            hashes.append(h)
    return removed


def find_similar_in_dir(target: str, hash_size: int = 8, hamming_thresh: int = 4):
    """Analyze a directory for near-duplicates by aHash.
    Returns (files:list[str], remove_indices:set[int]) where remove_indices
    contains indices in files suggested for deletion.
    """
    if not os.path.isdir(target):
        return [], set()
    files=[os.path.join(target,f) for f in os.listdir(target) if f.lower().endswith(('.jpg','.jpeg','.png'))]
    files.sort(key=lambda p: os.path.getmtime(p))
    hashes=[]; remove=set()
    def ahash(path):
        try:
            img=cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None: return None
            img=cv2.resize(img,(hash_size,hash_size))
            avg=img.mean(); bits=(img>avg).astype(np.uint8)
            return bits
        except Exception:
            return None
    for idx, fp in enumerate(files):
        h=ahash(fp)
        if h is None: continue
        for hh in hashes:
            dist = int((h^hh).sum())
            if dist <= hamming_thresh:
                remove.add(idx)
                break
        else:
            hashes.append(h)
    return files, remove

```


## FILE: AI/utils.py
```text
# utils.py
# open_folder_or_warn uses absolute, existing paths; no change except BASE_DIR import removed.
from __future__ import annotations
import time
from pathlib import Path
import numpy as np, cv2 as cv
from PyQt6 import QtGui, QtWidgets, QtCore

def monotonic_ms() -> int:
    return int(time.monotonic() * 1000)

def timestamp_name() -> str:
    return time.strftime("%Y%m%d_%H%M%S")

def ensure_dir(p: Path):
    Path(p).mkdir(parents=True, exist_ok=True)

def qimage_from_bgr(bgr: np.ndarray) -> QtGui.QImage:
    h, w = bgr.shape[:2]
    rgb = cv.cvtColor(bgr, cv.COLOR_BGR2RGB)
    bpl = int(rgb.strides[0])
    return QtGui.QImage(rgb.data, w, h, bpl, QtGui.QImage.Format.Format_RGB888).copy()

def open_folder_or_warn(parent: QtWidgets.QWidget, path: Path):
    try:
        ensure_dir(path)
        ok = QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(Path(path).resolve())))
        if not ok:
            raise RuntimeError("OS refused to open path")
    except Exception as e:
        QtWidgets.QMessageBox.warning(parent, "Open folder failed", f"Could not open:\n{path}\n\n{e}")

```