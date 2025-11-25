# camera_widget.py
# Per-camera widget: video view, AI overlays, recording, snapshots.

from __future__ import annotations

from typing import Optional

from PyQt6 import QtCore, QtGui, QtWidgets

from settings import AppSettings, CameraSettings
from detectors import DetectorThread, DetectorConfig, DetectionPacket
from recorder import PrebufferRecorder
from presence import PresenceBus
from utils import qimage_from_bgr
from stream import StreamCapture
from enrollment_service import EnrollmentService
from UI.graphics_view import GraphicsView
from UI.overlays import OverlayFlags, draw_overlays


OVERLAY_PERSIST_MS = 500  # keep last detections for this long to reduce flicker


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

        # Graphics scene + view for zoom/pan
        self._scene = QtWidgets.QGraphicsScene(self)
        self._pixmap_item = QtWidgets.QGraphicsPixmapItem()
        self._scene.addItem(self._pixmap_item)
        self.view = GraphicsView(self._scene, self)

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)

        # --- toolbar row (per-camera controls + AI switches) ---
        tb = QtWidgets.QHBoxLayout()

        self.btn_rec = QtWidgets.QPushButton("● REC")
        self.btn_snap = QtWidgets.QPushButton("Snapshot")

        self.cb_ai = QtWidgets.QCheckBox("AI")
        self.cb_yolo = QtWidgets.QCheckBox("YOLO")
        self.cb_faces = QtWidgets.QCheckBox("Faces")
        self.cb_pets = QtWidgets.QCheckBox("Pets")

        self.cb_ai.setChecked(True)
        self.cb_yolo.setChecked(True)
        self.cb_faces.setChecked(True)
        self.cb_pets.setChecked(True)

        self.btn_fit = QtWidgets.QPushButton("Fit")
        self.btn_100 = QtWidgets.QPushButton("100%")

        # Overlays menu button
        self.btn_overlay_menu = QtWidgets.QToolButton()
        self.btn_overlay_menu.setText("Overlays")
        self.btn_overlay_menu.setPopupMode(
            QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup
        )
        self.menu_overlays = QtWidgets.QMenu(self)

        self.act_overlay_yolo = self.menu_overlays.addAction("YOLO boxes")
        self.act_overlay_yolo.setCheckable(True)
        self.act_overlay_yolo.setChecked(True)

        self.act_overlay_faces = self.menu_overlays.addAction("Face labels")
        self.act_overlay_faces.setCheckable(True)
        self.act_overlay_faces.setChecked(True)

        self.act_overlay_pets = self.menu_overlays.addAction("Pet labels")
        self.act_overlay_pets.setCheckable(True)
        self.act_overlay_pets.setChecked(True)

        self.act_overlay_hud = self.menu_overlays.addAction("HUD (cam + time)")
        self.act_overlay_hud.setCheckable(True)
        self.act_overlay_hud.setChecked(True)

        self.btn_overlay_menu.setMenu(self.menu_overlays)

        tb.addWidget(self.btn_rec)
        tb.addWidget(self.btn_snap)
        tb.addSpacing(12)
        tb.addWidget(self.cb_ai)
        tb.addWidget(self.cb_yolo)
        tb.addWidget(self.cb_faces)
        tb.addWidget(self.cb_pets)
        tb.addWidget(self.btn_overlay_menu)
        tb.addStretch(1)
        tb.addWidget(self.btn_fit)
        tb.addWidget(self.btn_100)

        lay.addLayout(tb)
        lay.addWidget(self.view)

        # overlay flags / AI state
        self._overlays = OverlayFlags()
        self._ai_enabled = True
        self._last_bgr: Optional[object] = None
        self._last_ts: int = 0
        self._last_pkt: Optional[DetectionPacket] = None
        self._last_pkt_ts: int = 0

        # Prebuffer recorder: per-camera
        self._recorder = PrebufferRecorder(
            cam_name=self.cam_cfg.name,
            out_dir=self.app_cfg.output_dir,
            fps=25,
            pre_ms=self.app_cfg.prebuffer_ms,
        )

        # Presence logging bus
        self._presence = PresenceBus(self.cam_cfg.name, self.app_cfg.logs_dir)

        # Detector thread
        det_cfg = DetectorConfig.from_app(self.app_cfg)
        self._detector = DetectorThread(det_cfg, self.cam_cfg.name)
        self._detector.resultsReady.connect(self._on_detections)

        # Stream capture backend
        self._capture = StreamCapture(self.cam_cfg)

        # Poll frames from StreamCapture
        self._frame_timer = QtCore.QTimer(self)
        self._frame_timer.setInterval(30)
        self._frame_timer.timeout.connect(self._poll_frame)

        # wire toolbar actions
        self.btn_fit.clicked.connect(self.fit_to_window)
        self.btn_100.clicked.connect(self.zoom_100)
        self.btn_snap.clicked.connect(self._snapshot)
        self.btn_rec.clicked.connect(self._toggle_recording)

        self.cb_ai.toggled.connect(self._on_ai_toggled)
        self.cb_yolo.toggled.connect(self._on_overlay_changed)
        self.cb_faces.toggled.connect(self._on_overlay_changed)
        self.cb_pets.toggled.connect(self._on_overlay_changed)

        self.act_overlay_yolo.toggled.connect(self._on_overlay_menu_changed)
        self.act_overlay_faces.toggled.connect(self._on_overlay_menu_changed)
        self.act_overlay_pets.toggled.connect(self._on_overlay_menu_changed)
        self.act_overlay_hud.toggled.connect(self._on_overlay_menu_changed)

        self._detector.start()

    # ---- lifecycle ----
    def start(self):
        self._capture.start()
        self._frame_timer.start()

    def stop(self):
        self._frame_timer.stop()
        self._capture.stop()
        self._detector.stop()
        self._recorder.close()

    # ---- frame handling ----
    def _poll_frame(self):
        ok, frame, ts_ms = self._capture.read()
        if not ok or frame is None:
            return
        self._last_bgr = frame
        self._last_ts = ts_ms

        # Feed recorder
        self._recorder.on_frame(frame, ts_ms)

        # Feed detector only if AI enabled
        if self._ai_enabled:
            self._detector.submit_frame(self.cam_cfg.name, frame, ts_ms)

        # Use last detection packet for a short window to reduce overlay flicker
        pkt_for_frame: Optional[DetectionPacket] = None
        if self._last_pkt is not None:
            age = ts_ms - self._last_pkt_ts
            if 0 <= age <= OVERLAY_PERSIST_MS:
                pkt_for_frame = self._last_pkt

        # Draw current frame (optionally with last overlays)
        self._update_pixmap(frame, pkt_for_frame)

    def _update_pixmap(self, bgr, pkt: Optional[DetectionPacket]):
        qimg = qimage_from_bgr(bgr)
        pixmap = QtGui.QPixmap.fromImage(qimg)

        if pkt is not None and self._ai_enabled:
            painter = QtGui.QPainter(pixmap)
            try:
                draw_overlays(painter, pkt, self._overlays)
            finally:
                painter.end()

        self._pixmap_item.setPixmap(pixmap)
        self._scene.setSceneRect(QtCore.QRectF(pixmap.rect()))

    @QtCore.pyqtSlot(object)
    def _on_detections(self, pkt_obj):
        if not self._ai_enabled:
            return
        pkt = pkt_obj
        if not isinstance(pkt, DetectionPacket):
            return
        if pkt.name != self.cam_cfg.name:
            return

        # Presence log
        self._presence.update(pkt)

        # Remember last packet for flicker-free overlays
        self._last_pkt = pkt
        self._last_pkt_ts = pkt.ts_ms

        if self._last_bgr is not None:
            # Feed enrollment service for the chosen (or any) camera
            EnrollmentService.instance().on_detections(
                self.cam_cfg.name, self._last_bgr, pkt
            )
            # Draw overlays immediately as well
            self._update_pixmap(self._last_bgr, pkt)

    # ---- recording / snapshot helpers ----
    def _snapshot(self):
        if self._last_bgr is None:
            return
        import cv2
        import time as _time

        fname = f"{self.cam_cfg.name}_{_time.strftime('%Y%m%d_%H%M%S')}.jpg"
        path = self.app_cfg.output_dir / fname
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), self._last_bgr)

    def _toggle_recording(self):
        if self._recorder.writer is None:
            self._recorder.start()
            self.btn_rec.setText("■ STOP")
        else:
            self._recorder.stop()
            self.btn_rec.setText("● REC")

    # ---- AI / overlay switches ----
    def _on_ai_toggled(self, checked: bool):
        self._ai_enabled = bool(checked)

    def _on_overlay_changed(self):
        """Checkboxes → flags + sync menu items."""
        self._overlays.yolo = self.cb_yolo.isChecked()
        self._overlays.faces = self.cb_faces.isChecked()
        self._overlays.pets = self.cb_pets.isChecked()
        # tracks flag left as-is (no UI yet)

        self.act_overlay_yolo.setChecked(self._overlays.yolo)
        self.act_overlay_faces.setChecked(self._overlays.faces)
        self.act_overlay_pets.setChecked(self._overlays.pets)
        self.act_overlay_hud.setChecked(self._overlays.hud)

    def _on_overlay_menu_changed(self):
        """Menu → flags + sync checkboxes."""
        self._overlays.yolo = self.act_overlay_yolo.isChecked()
        self._overlays.faces = self.act_overlay_faces.isChecked()
        self._overlays.pets = self.act_overlay_pets.isChecked()
        self._overlays.hud = self.act_overlay_hud.isChecked()

        self.cb_yolo.setChecked(self._overlays.yolo)
        self.cb_faces.setChecked(self._overlays.faces)
        self.cb_pets.setChecked(self._overlays.pets)

    # ---- view helpers for MainWindow ----
    def fit_to_window(self):
        self.view.fitInView(
            self._scene.sceneRect(),
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
        )
        self.view._scale = 1.0

    def zoom_100(self):
        self.view.resetTransform()
        self.view._scale = 1.0
