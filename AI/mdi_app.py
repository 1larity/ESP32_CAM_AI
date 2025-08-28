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


@dataclass
class CameraConfig:
    name: str
    host: str                 # ip[:port] for port 80
    user: Optional[str] = None
    password: Optional[str] = None
    token: Optional[str] = None  # Base64 of user:pass

    def stream_url(self) -> str:
        base = self.host.split(':')[0]
        suffix = f"?token={self.token}" if self.token else ""
        return f"http://{base}:81/stream{suffix}"

    def auth_header(self) -> Optional[str]:
        if self.token:
            return None
        if self.user and self.password:
            import base64
            up = f"{self.user}:{self.password}".encode('utf-8')
            return "Basic " + base64.b64encode(up).decode('ascii')
        return None


class CameraStreamThread(QtCore.QThread):
    frameReady = QtCore.Signal(np.ndarray, float)  # (bgr_frame, timestamp)

    def __init__(self, cfg: CameraConfig, parent=None, prebuffer_seconds: float = 5.0):
        super().__init__(parent)
        self.cfg = cfg
        self._stop = threading.Event()
        self._session = None
        self._resp = None
        self._buf = bytearray()
        self._boundary = b"--frame"
        self.prebuffer: Deque[Tuple[np.ndarray, float]] = deque(maxlen=int(prebuffer_seconds * 20))  # assume ~20fps cap

    def stop(self):
        self._stop.set()
        try:
            if self._resp is not None:
                self._resp.close()
        except Exception:
            pass

    def run(self):
        headers = {}
        auth = self.cfg.auth_header()
        if auth:
            headers['Authorization'] = auth
        self._session = requests.Session()
        try:
            self._resp = self._session.get(self.cfg.stream_url(), headers=headers, stream=True, timeout=10)
            self._resp.raise_for_status()
        except Exception as e:
            print(f"[Stream] Failed to connect {self.cfg.name}: {e}")
            return

        for chunk in self._resp.iter_content(chunk_size=8192):
            if self._stop.is_set():
                break
            if not chunk:
                continue
            self._buf += chunk
            while True:
                hdr_end = self._buf.find(b"\r\n\r\n")
                if hdr_end == -1:
                    break
                headers_blob = self._buf[:hdr_end].decode('latin1', errors='ignore').lower()
                cl_idx = headers_blob.find('content-length:')
                if cl_idx == -1:
                    # resync to boundary
                    bidx = self._buf.find(self._boundary)
                    self._buf = self._buf[bidx:] if bidx != -1 else self._buf[hdr_end+4:]
                    continue
                try:
                    cl_line = headers_blob[cl_idx:].split('\r\n', 1)[0]
                    length = int(cl_line.split(':', 1)[1].strip())
                except Exception:
                    bidx = self._buf.find(self._boundary)
                    self._buf = self._buf[bidx:] if bidx != -1 else self._buf[hdr_end+4:]
                    continue
                start = hdr_end + 4
                if len(self._buf) < start + length:
                    break
                jpg = self._buf[start:start+length]
                tail = self._buf[start+length:]
                bmark = b"\r\n--frame\r\n"
                if tail.startswith(bmark):
                    self._buf = bytearray(tail[len(bmark):])
                else:
                    bpos = tail.find(bmark)
                    self._buf = bytearray(tail[bpos+len(bmark):] if bpos >= 0 else b"")

                # decode
                arr = np.frombuffer(jpg, np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is None:
                    continue
                ts = time.time()
                self.prebuffer.append((frame, ts))
                self.frameReady.emit(frame, ts)


class CameraWidget(QtWidgets.QWidget):
    def __init__(self, cfg: CameraConfig, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self.setWindowTitle(cfg.name)
        self.label = QtWidgets.QLabel('Connecting…')
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setMinimumSize(320, 240)
        self.label.setStyleSheet('background:#000; color:#9cf;')

        # Recording state
        self.recording = False
        self.writer: Optional[cv2.VideoWriter] = None
        self.out_dir = os.path.join('ai', 'recordings')
        os.makedirs(self.out_dir, exist_ok=True)
        self.target_fps = 20.0

        # Controls (local toolbar)
        btns = QtWidgets.QToolBar()
        act_start = btns.addAction('Start')
        act_stop = btns.addAction('Stop')
        btns.addSeparator()
        act_rec = btns.addAction('Start Rec')
        act_stoprec = btns.addAction('Stop Rec')

        act_start.triggered.connect(self.start_stream)
        act_stop.triggered.connect(self.stop_stream)
        act_rec.triggered.connect(self.start_recording)
        act_stoprec.triggered.connect(self.stop_recording)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4,4,4,4)
        layout.addWidget(btns)
        layout.addWidget(self.label, 1)

        # Stream thread
        self.thr = CameraStreamThread(cfg)
        self.thr.frameReady.connect(self.on_frame)
        self._last_ts = None

    def start_stream(self):
        if not self.thr.isRunning():
            self.thr._stop.clear()
            self.thr.start()

    def stop_stream(self):
        if self.thr.isRunning():
            self.thr.stop()
            self.thr.wait(1000)

    def start_recording(self):
        if self.recording:
            return
        # estimate FPS from prebuffer timing
        fps = self.target_fps
        if len(self.thr.prebuffer) >= 5:
            tspan = self.thr.prebuffer[-1][1] - self.thr.prebuffer[0][1]
            frames = len(self.thr.prebuffer)
            if tspan > 0:
                fps = max(5.0, min(30.0, frames / tspan))
        # open writer
        ts_str = time.strftime('%Y%m%d_%H%M%S')
        outfile = os.path.join(self.out_dir, f"{self.cfg.name}_{ts_str}.mp4")
        h, w = self.current_size()
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(outfile, fourcc, fps, (w, h))
        # dump prebuffer first
        for frm, _ in list(self.thr.prebuffer):
            if frm.shape[1] != w or frm.shape[0] != h:
                frm = cv2.resize(frm, (w, h))
            self.writer.write(frm)
        self.recording = True
        self.setWindowTitle(f"{self.cfg.name} (REC)")

    def stop_recording(self):
        if self.recording and self.writer is not None:
            try:
                self.writer.release()
            except Exception:
                pass
        self.recording = False
        self.writer = None
        self.setWindowTitle(self.cfg.name)

    def current_size(self) -> Tuple[int,int]:
        # return H, W for writer
        pix = self.label.pixmap()
        if pix and not pix.isNull():
            return pix.height(), pix.width()
        return 480, 640

    @QtCore.Slot(np.ndarray, float)
    def on_frame(self, bgr: np.ndarray, ts: float):
        # show
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        img = QtGui.QImage(rgb.data, w, h, ch*w, QtGui.QImage.Format.Format_RGB888)
        self.label.setPixmap(QtGui.QPixmap.fromImage(img).scaled(self.label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
        # write
        if self.recording and self.writer is not None:
            # ensure writer size consistency
            W = int(self.writer.get(cv2.CAP_PROP_FRAME_WIDTH))
            H = int(self.writer.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if bgr.shape[1] != W or bgr.shape[0] != H:
                bgr = cv2.resize(bgr, (W, H))
            self.writer.write(bgr)

    def closeEvent(self, e: QtGui.QCloseEvent) -> None:
        self.stop_recording()
        self.stop_stream()
        super().closeEvent(e)


class AddCameraDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Add Camera')
        self.setModal(True)
        form = QtWidgets.QFormLayout(self)
        self.ed_name = QtWidgets.QLineEdit('Camera')
        self.ed_host = QtWidgets.QLineEdit('192.168.1.100')
        self.ed_user = QtWidgets.QLineEdit()
        self.ed_pass = QtWidgets.QLineEdit(); self.ed_pass.setEchoMode(QtWidgets.QLineEdit.Password)
        self.ed_token = QtWidgets.QLineEdit()
        form.addRow('Name', self.ed_name)
        form.addRow('Host (ip[:port])', self.ed_host)
        form.addRow('User', self.ed_user)
        form.addRow('Password', self.ed_pass)
        form.addRow('Token (Base64 user:pass)', self.ed_token)
        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        form.addRow(btns)

    def get_config(self) -> Optional[CameraConfig]:
        if self.exec() == QtWidgets.QDialog.Accepted:
            return CameraConfig(
                name=self.ed_name.text().strip() or 'Camera',
                host=self.ed_host.text().strip(),
                user=self.ed_user.text().strip() or None,
                password=self.ed_pass.text(),
                token=self.ed_token.text().strip() or None,
            )
        return None


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('ESP32-CAM MDI')
        self.resize(1200, 800)
        self.mdi = QtWidgets.QMdiArea()
        self.setCentralWidget(self.mdi)

        tb = QtWidgets.QToolBar('Main')
        tb.setIconSize(QtCore.QSize(16,16))
        self.addToolBar(tb)

        act_add = tb.addAction('Add Camera')
        act_tile = tb.addAction('Tile')
        act_cascade = tb.addAction('Cascade')
        tb.addSeparator()
        act_rec_all = tb.addAction('Start Rec All')
        act_stop_all = tb.addAction('Stop Rec All')

        act_add.triggered.connect(self.add_camera)
        act_tile.triggered.connect(self.mdi.tileSubWindows)
        act_cascade.triggered.connect(self.mdi.cascadeSubWindows)
        act_rec_all.triggered.connect(self.start_rec_all)
        act_stop_all.triggered.connect(self.stop_rec_all)

        os.makedirs(os.path.join('ai','recordings'), exist_ok=True)

    def add_camera(self):
        dlg = AddCameraDialog(self)
        cfg = dlg.get_config()
        if not cfg:
            return
        w = CameraWidget(cfg)
        sub = QtWidgets.QMdiSubWindow()
        sub.setWidget(w)
        sub.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
        sub.setWindowTitle(cfg.name)
        self.mdi.addSubWindow(sub)
        sub.resize(500, 420)
        sub.show()
        w.start_stream()

    def start_rec_all(self):
        for sub in self.mdi.subWindowList():
            w = sub.widget()
            if isinstance(w, CameraWidget):
                w.start_recording()

    def stop_rec_all(self):
        for sub in self.mdi.subWindowList():
            w = sub.widget()
            if isinstance(w, CameraWidget):
                w.stop_recording()


def main():
    app = QtWidgets.QApplication(sys.argv)
    mw = MainWindow()
    mw.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()

