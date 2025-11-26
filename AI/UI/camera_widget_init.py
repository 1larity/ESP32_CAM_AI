# camera_widget_init.py
# Build UI, state and signal wiring for CameraWidget.

from __future__ import annotations

from PyQt6 import QtCore, QtWidgets

from detectors import DetectorThread, DetectorConfig
from recorder import PrebufferRecorder
from presence import PresenceBus
from stream import StreamCapture
from enrollment_service import EnrollmentService
from UI.graphics_view import GraphicsView
from UI.overlays import OverlayFlags


def init_camera_widget(self) -> None:
    """Build per-camera UI, backend helpers and signal wiring."""
    # Scene + view
    self._scene = QtWidgets.QGraphicsScene(self)
    self._pixmap_item = QtWidgets.QGraphicsPixmapItem()
    self._scene.addItem(self._pixmap_item)
    self.view = GraphicsView(self._scene, self)

    root_layout = QtWidgets.QVBoxLayout(self)
    root_layout.setContentsMargins(0, 0, 0, 0)

    # Toolbar
    tb = QtWidgets.QHBoxLayout()
    self.btn_rec = QtWidgets.QPushButton("● REC")
    self.btn_snap = QtWidgets.QPushButton("Snapshot")
    self.btn_fit = QtWidgets.QPushButton("Fit")
    self.btn_100 = QtWidgets.QPushButton("100%")
    self.btn_fit_window = QtWidgets.QPushButton("Fit win")
    self.btn_lock = QtWidgets.QToolButton()
    self.btn_lock.setText("Lock")
    self.btn_lock.setCheckable(True)

    # Overlays menu
    self.btn_overlay_menu = QtWidgets.QToolButton()
    self.btn_overlay_menu.setText("Overlays")
    self.btn_overlay_menu.setPopupMode(
        QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup
    )
    self.menu_overlays = QtWidgets.QMenu(self)
    self.act_overlay_detections = self.menu_overlays.addAction(
        "Detections (boxes + labels)"
    )
    self.act_overlay_detections.setCheckable(True)
    self.act_overlay_detections.setChecked(True)
    self.act_overlay_hud = self.menu_overlays.addAction("HUD (cam + time)")
    self.act_overlay_hud.setCheckable(True)
    self.act_overlay_hud.setChecked(True)
    self.btn_overlay_menu.setMenu(self.menu_overlays)

    # AI menu
    self.btn_ai_menu = QtWidgets.QToolButton()
    self.btn_ai_menu.setText("AI")
    self.btn_ai_menu.setPopupMode(
        QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup
    )
    self.menu_ai = QtWidgets.QMenu(self)

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
    tb.addWidget(self.btn_fit_window)
    tb.addWidget(self.btn_lock)

    root_layout.addLayout(tb)
    root_layout.addWidget(self.view)

    # State
    self._overlays = OverlayFlags()
    self._ai_enabled = True
    self._overlay_master_updating = False

    self._last_bgr = None
    self._last_ts = 0
    self._last_pkt = None
    self._last_pkt_ts = 0
    # keep overlays around a bit to avoid flicker
    self._overlay_ttl_ms = 750  # matches previous CameraWidget

    # Prebuffer recorder: per-camera
    self._recorder = PrebufferRecorder(
        cam_name=self.cam_cfg.name,
        out_dir=self.app_cfg.output_dir,
        fps=25,
        pre_ms=self.app_cfg.prebuffer_ms,
    )

    # Presence logging bus (per camera) – same as original
    self._presence = PresenceBus(self.cam_cfg.name, self.app_cfg.logs_dir)

    # Detector thread – same construction as before
    det_cfg = DetectorConfig.from_app(self.app_cfg)
    self._detector = DetectorThread(det_cfg, self.cam_cfg.name)
    self._detector.resultsReady.connect(self._on_detections)

    # Stream capture backend
    self._capture = StreamCapture(self.cam_cfg)

    # Poll frames from StreamCapture – keep previous cadence
    self._frame_timer = QtCore.QTimer(self)
    self._frame_timer.setInterval(
        getattr(self.app_cfg, "detect_interval_ms", 30)
    )

    # Lock state / MDI subwindow tracking
    self._locked = False
    self._subwindow = None
    self._locked_geometry = QtCore.QRect()

    # Wiring
    self._frame_timer.timeout.connect(self._poll_frame)
    self.btn_rec.clicked.connect(self._toggle_recording)
    self.btn_snap.clicked.connect(self._snapshot)
    self.btn_fit.clicked.connect(self.fit_to_window)
    self.btn_100.clicked.connect(self.zoom_100)
    self.btn_fit_window.clicked.connect(self.fit_window_to_video)
    self.btn_lock.toggled.connect(self._on_lock_toggled)

    # AI + overlay menu actions
    self.act_ai_enabled.toggled.connect(self._on_ai_toggled)
    self.act_ai_yolo.toggled.connect(self._on_ai_yolo_toggled)
    self.act_ai_faces.toggled.connect(self._on_ai_faces_toggled)
    self.act_ai_pets.toggled.connect(self._on_ai_pets_toggled)
    self.act_overlay_detections.toggled.connect(self._on_overlay_master_toggled)
    self.act_overlay_hud.toggled.connect(self._on_overlay_hud_toggled)

    # Enrollment service – singleton
    self._enrollment = EnrollmentService.instance()

    # Intercept view move/resize when locked
    self.view.installEventFilter(self)

    # Defaults
    self._overlays.hud = True
    self._overlays.yolo = True
    self._overlays.faces = True
    self._overlays.pets = True
