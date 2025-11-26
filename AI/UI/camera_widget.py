# camera_widget.py
# Per-camera widget: video view, AI overlays, recording, snapshots.

from __future__ import annotations

from typing import Optional
import time  # **CHANGED**: used for HUD clock

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


class CameraWidget(QtWidgets.QWidget):
    """
    One camera:
      - StreamCapture → frames
      - DetectorThread → DetectionPacket
      - PrebufferRecorder → videos with pre-roll
      - Overlays drawn via UI/overlays.py
    """

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

        # ------------------------------------------------------------------ toolbar row
        tb = QtWidgets.QHBoxLayout()

        self.btn_rec = QtWidgets.QPushButton("● REC")
        self.btn_snap = QtWidgets.QPushButton("Snapshot")

        self.btn_fit = QtWidgets.QPushButton("Fit")
        self.btn_100 = QtWidgets.QPushButton("100%")

        # Overlays menu button (rectangles + labels together, plus HUD)
        self.btn_overlay_menu = QtWidgets.QToolButton()
        self.btn_overlay_menu.setText("Overlays")
        self.btn_overlay_menu.setPopupMode(
            QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup
        )
        self.menu_overlays = QtWidgets.QMenu(self)

        # Single master toggle: all detection rectangles + labels
        self.act_overlay_detections = self.menu_overlays.addAction(
            "Detections (boxes + labels)"
        )
        self.act_overlay_detections.setCheckable(True)
        self.act_overlay_detections.setChecked(True)

        # HUD: camera name + timestamp
        self.act_overlay_hud = self.menu_overlays.addAction("HUD (cam + time)")
        self.act_overlay_hud.setCheckable(True)
        self.act_overlay_hud.setChecked(True)

        self.btn_overlay_menu.setMenu(self.menu_overlays)

        # AI menu button
        self.btn_ai_menu = QtWidgets.QToolButton()
        self.btn_ai_menu.setText("AI")
        self.btn_ai_menu.setPopupMode(
            QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup
        )
        self.menu_ai = QtWidgets.QMenu(self)

        # **CHANGED**: AI master + per-feature toggles
        self.act_ai_enabled = self.menu_ai.addAction("Enable AI")
        self.act_ai_enabled.setCheckable(True)
        self.act_ai_enabled.setChecked(True)

        self.act_ai_yolo = self.menu_ai.addAction("YOLO")
        self.act_ai_yolo.setCheckable(True)
        self.act_ai_yolo.setChecked(True)

        self.act_ai_faces = self.menu_ai.addAction("Faces")
        self.act_ai_faces.setCheckable(True)
        self.act_ai_faces.setChecked(True)

        self.act_ai_pets = self.menu_ai.addAction("Pets")
        self.act_ai_pets.setCheckable(True)
        self.act_ai_pets.setChecked(True)

        self.btn_ai_menu.setMenu(self.menu_ai)

        # Assemble toolbar
        tb.addWidget(self.btn_rec)
        tb.addWidget(self.btn_snap)
        tb.addSpacing(12)
        tb.addWidget(self.btn_ai_menu)
        tb.addWidget(self.btn_overlay_menu)
        tb.addStretch(1)
        tb.addWidget(self.btn_fit)
        tb.addWidget(self.btn_100)

        lay.addLayout(tb)
        lay.addWidget(self.view)

        # ------------------------------------------------------------------ state

        # Overlay flags / AI state
        self._overlays = OverlayFlags()
        self._ai_enabled = True

        # **CHANGED**: guard to avoid recursive master-toggle updates
        self._overlay_master_updating = False

        self._last_bgr: Optional[object] = None
        self._last_ts: int = 0
        self._last_pkt: Optional[DetectionPacket] = None
        self._last_pkt_ts: int = 0
        # keep overlays around a bit to avoid flicker
        self._overlay_ttl_ms: int = 750

        # Prebuffer recorder: per-camera
        self._recorder = PrebufferRecorder(
            cam_name=self.cam_cfg.name,
            out_dir=self.app_cfg.output_dir,
            fps=25,
            pre_ms=self.app_cfg.prebuffer_ms,
        )

        # Presence logging bus (per camera)
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

        # Wire actions
        self.btn_fit.clicked.connect(self.fit_to_window)
        self.btn_100.clicked.connect(self.zoom_100)
        self.btn_snap.clicked.connect(self._snapshot)
        self.btn_rec.clicked.connect(self._toggle_recording)

        # **CHANGED**: connect AI + overlay menu actions
        self.act_ai_enabled.toggled.connect(self._on_ai_toggled)
        self.act_ai_yolo.toggled.connect(self._on_ai_yolo_toggled)
        self.act_ai_faces.toggled.connect(self._on_ai_faces_toggled)
        self.act_ai_pets.toggled.connect(self._on_ai_pets_toggled)

        self.act_overlay_detections.toggled.connect(self._on_overlay_master_toggled)
        self.act_overlay_hud.toggled.connect(self._on_overlay_hud_toggled)

        self.setWindowTitle(self.cam_cfg.name)

    # ------------------------------------------------------------------ lifecycle

    def start(self):
        """Start streaming + detection."""
        self._capture.start()
        self._detector.start()
        self._frame_timer.start()

    def stop(self):
        """Stop timers, detector, recorder, capture."""
        self._frame_timer.stop()
        self._capture.stop()
        self._detector.stop()
        self._recorder.close()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # type: ignore[override]
        self.stop()
        event.accept()

    # ------------------------------------------------------------------ frame handling

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
            if 0 <= age <= self._overlay_ttl_ms:
                pkt_for_frame = self._last_pkt

        # Draw current frame (optionally with last overlays)
        self._update_pixmap(frame, pkt_for_frame)

    def _update_pixmap(self, bgr, pkt: Optional[DetectionPacket]):
        qimg = qimage_from_bgr(bgr)
        pixmap = QtGui.QPixmap.fromImage(qimg)

        # **CHANGED**: always paint; detections depend on AI, HUD does not
        painter = QtGui.QPainter(pixmap)
        try:
            # Detection boxes/labels only if:
            # - we have a packet
            # - AI is enabled
            # - at least one detection overlay type is active
            if pkt is not None and self._ai_enabled:
                detections_enabled = (
                    getattr(self._overlays, "yolo", False)
                    or getattr(self._overlays, "faces", False)
                    or getattr(self._overlays, "pets", False)
                    or getattr(self._overlays, "tracks", False)
                )
                if detections_enabled:
                    # Temporarily suppress HUD inside overlays.py; we draw HUD ourselves
                    orig_hud = getattr(self._overlays, "hud", False)
                    self._overlays.hud = False
                    draw_overlays(painter, pkt, self._overlays)
                    self._overlays.hud = orig_hud

            # HUD (camera name + date/time) is independent of AI on/off
            if getattr(self._overlays, "hud", False):
                self._draw_hud(painter)
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
            # Feed enrollment service for this camera (or any, depending on configuration)
            EnrollmentService.instance().on_detections(
                self.cam_cfg.name, self._last_bgr, pkt
            )
            # Draw overlays immediately on top of the last frame
            self._update_pixmap(self._last_bgr, pkt)

    # ------------------------------------------------------------------ recording / snapshot helpers

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
        # PrebufferRecorder exposes writer; use that as "is recording" flag.
        if self._recorder.writer is None:
            self._recorder.start()
            self.btn_rec.setText("■ STOP")
        else:
            self._recorder.stop()
            self.btn_rec.setText("● REC")

    # ------------------------------------------------------------------ AI / overlay switches

    def _on_ai_toggled(self, checked: bool):
        self._ai_enabled = bool(checked)

    # **CHANGED**: per-feature AI menu items control overlay flags
    def _on_ai_yolo_toggled(self, checked: bool):
        self._overlays.yolo = bool(checked)
        self._sync_overlay_master()

    def _on_ai_faces_toggled(self, checked: bool):
        self._overlays.faces = bool(checked)
        self._sync_overlay_master()

    def _on_ai_pets_toggled(self, checked: bool):
        self._overlays.pets = bool(checked)
        self._sync_overlay_master()

    def _sync_overlay_master(self):
        """Keep 'Detections (boxes + labels)' in sync with YOLO/Faces/Pets."""
        if self._overlay_master_updating:
            return
        any_on = (
            getattr(self._overlays, "yolo", False)
            or getattr(self._overlays, "faces", False)
            or getattr(self._overlays, "pets", False)
        )
        self._overlay_master_updating = True
        try:
            self.act_overlay_detections.setChecked(any_on)
        finally:
            self._overlay_master_updating = False

    def _on_overlay_master_toggled(self, checked: bool):
        """
        Single menu item to toggle all detection rectangles + labels together.
        """
        self._overlay_master_updating = True
        try:
            enabled = bool(checked)
            self._overlays.yolo = enabled
            self._overlays.faces = enabled
            self._overlays.pets = enabled

            # Keep AI menu items in sync
            self.act_ai_yolo.setChecked(enabled)
            self.act_ai_faces.setChecked(enabled)
            self.act_ai_pets.setChecked(enabled)
        finally:
            self._overlay_master_updating = False

    def _on_overlay_hud_toggled(self, checked: bool):
        """
        Toggle HUD (camera name + date/timestamp).
        """
        self._overlays.hud = bool(checked)

    # ------------------------------------------------------------------ HUD drawing (independent of AI)  **CHANGED**
    def _draw_hud(self, p: QtGui.QPainter):
        """
        Draw camera name + current wall-clock date/time in the top-left corner.
        This does not depend on AI or detection packets.
        """
        text = f"{self.cam_cfg.name}  {time.strftime('%Y-%m-%d %H:%M:%S')}"
        margin = 6
        p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255)))
        font = p.font()
        font.setPointSize(max(font.pointSize(), 9))
        p.setFont(font)

        # Simple text background
        metrics = QtGui.QFontMetrics(font)
        w = metrics.horizontalAdvance(text) + margin * 2
        h = metrics.height() + margin * 2
        rect = QtCore.QRect(margin, margin, w, h)

        bg = QtGui.QColor(0, 0, 0, 128)
        p.fillRect(rect, bg)
        p.drawText(
            rect.adjusted(margin, margin // 2, -margin, 0),
            QtCore.Qt.AlignmentFlag.AlignVCenter | QtCore.Qt.AlignmentFlag.AlignLeft,
            text,
        )

    # ------------------------------------------------------------------ view helpers for MainWindow

    def fit_to_window(self):
        self.view.fitInView(
            self._scene.sceneRect(),
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
        )
        self.view._scale = 1.0

    def zoom_100(self):
        self.view.resetTransform()
        self.view._scale = 1.0
