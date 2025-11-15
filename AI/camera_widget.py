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
    def __init__(self, cam_cfg: CameraSettings, app_cfg: AppSettings,
                 parent: Optional[QtWidgets.QWidget] = None):
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
        self.cb_ai = QtWidgets.QCheckBox("AI")
        self.cb_yolo = QtWidgets.QCheckBox("YOLO")
        self.cb_faces = QtWidgets.QCheckBox("Faces")
        self.cb_pets = QtWidgets.QCheckBox("Pets")
        self.btn_fit = QtWidgets.QPushButton("Fit")
        self.btn_100 = QtWidgets.QPushButton("100%")

        self.cb_ai.setChecked(True)
        self.cb_yolo.setChecked(True)
        self.cb_faces.setChecked(True)
        self.cb_pets.setChecked(True)

        bar.addWidget(self.btn_rec)
        bar.addWidget(self.btn_snap)
        bar.addSpacing(12)
        bar.addWidget(self.cb_ai)
        bar.addWidget(self.cb_yolo)
        bar.addWidget(self.cb_faces)
        bar.addWidget(self.cb_pets)
        bar.addStretch(1)
        bar.addWidget(self.btn_fit)
        bar.addWidget(self.btn_100)

        lay.addLayout(bar)
        lay.addWidget(self.view)

        self._overlays = OverlayFlags()
        self._ai_enabled = True
        self._last_bgr = None
        self._last_ts = 0

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

        self.btn_fit.clicked.connect(self.fit_to_window)
        self.btn_100.clicked.connect(self.zoom_100)
        self.btn_snap.clicked.connect(self._snapshot)
        self.btn_rec.clicked.connect(self._toggle_recording)
        self.cb_ai.toggled.connect(self._on_ai_toggled)
        self.cb_yolo.toggled.connect(self._on_overlay_changed)
        self.cb_faces.toggled.connect(self._on_overlay_changed)
        self.cb_pets.toggled.connect(self._on_overlay_changed)

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
        self._update_pixmap(frame, None)

    def _update_pixmap(self, bgr, pkt: Optional[DetectionPacket]):
        qimg = qimage_from_bgr(bgr)
        pixmap = QtGui.QPixmap.fromImage(qimg)
        if pkt is not None and self._ai_enabled:
            p = QtGui.QPainter(pixmap)
            try:
                draw_overlays(p, pkt, self._overlays)
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

        # DEBUG show that GUI is receiving packets
        print(
            f"[GUI:{self.cam_cfg.name}] recv pkt ts={pkt.ts_ms} "
            f"yolo={len(pkt.yolo)} faces={len(pkt.faces)} pets={len(pkt.pets)}"
        )

        self._presence.update(pkt)
        if self._last_bgr is not None:
            self._update_pixmap(self._last_bgr, pkt)

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

    def _on_overlay_changed(self):
        self._overlays.yolo = self.cb_yolo.isChecked()
        self._overlays.faces = self.cb_faces.isChecked()
        self._overlays.pets = self.cb_pets.isChecked()

    # view helpers
    def fit_to_window(self):
        self.view.fitInView(
            self._scene.sceneRect(),
            QtCore.Qt.AspectRatioMode.KeepAspectRatio
        )
        self.view._scale = 1.0

    def zoom_100(self):
        self.view.resetTransform()
        self.view._scale = 1.0
