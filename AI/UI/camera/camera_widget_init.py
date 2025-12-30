# camera/camera_widget_init.py
# Build UI, state and signal wiring for CameraWidget.

from __future__ import annotations
from PySide6 import QtCore, QtWidgets
from detectors import DetectorThread, DetectorConfig
from recorder import PrebufferRecorder
from presence import PresenceBus
from stream import StreamCapture
from enrollment import EnrollmentService
from face_params import FaceParams
from ..graphics_view import GraphicsView
from ..overlays import OverlayFlags


def init_camera_widget(self) -> None:
    """Build per-camera UI, backend helpers and signal wiring."""

    # ------------------------------------------------------------------
    # Scene + view
    # ------------------------------------------------------------------
    self._scene = QtWidgets.QGraphicsScene(self)
    self._pixmap_item = QtWidgets.QGraphicsPixmapItem()
    self._scene.addItem(self._pixmap_item)
    self.view = GraphicsView(self._scene, self)

    root_layout = QtWidgets.QVBoxLayout(self)
    root_layout.setContentsMargins(0, 0, 0, 0)

    # ------------------------------------------------------------------
    # Toolbar
    # ------------------------------------------------------------------
    tb = QtWidgets.QHBoxLayout()
    self.btn_rec = QtWidgets.QPushButton("REC")
    self.btn_snap = QtWidgets.QPushButton("Snapshot")

    # View menu (replaces Fit / 100% / Fit win buttons)
    self.btn_view_menu = QtWidgets.QToolButton()
    self.btn_view_menu.setText("View")
    self.btn_view_menu.setPopupMode(
        QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup
    )
    self.menu_view = QtWidgets.QMenu(self)

    # View scaling actions
    self.act_view_fit = self.menu_view.addAction("Fit in view")
    self.act_view_100 = self.menu_view.addAction("100% (1:1)")
    self.act_view_fit_window = self.menu_view.addAction("Fit window to video")
    self.btn_view_menu.setMenu(self.menu_view)

    self.btn_lock = QtWidgets.QToolButton()
    self.btn_lock.setText("Lock")
    self.btn_lock.setCheckable(True)

    self.btn_info = QtWidgets.QToolButton()
    self.btn_info.setText("Info")

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
    self.act_overlay_stats = self.menu_overlays.addAction("Stats (FPS + counts)")
    self.act_overlay_stats.setCheckable(True)
    self.act_overlay_stats.setChecked(True)
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

    # Flash controls
    self.cb_flash = QtWidgets.QComboBox()
    self.cb_flash.addItems(["Off", "On", "Auto"])
    self.s_flash = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
    self.s_flash.setRange(0, 1023)
    self.s_flash.setSingleStep(8)
    self.s_flash.setFixedWidth(120)
    self.lbl_flash = QtWidgets.QLabel("0")
    flash_wrap = QtWidgets.QHBoxLayout()
    flash_wrap.setContentsMargins(0, 0, 0, 0)
    flash_wrap.addWidget(QtWidgets.QLabel("LED"))
    flash_wrap.addWidget(self.cb_flash)
    flash_wrap.addWidget(self.s_flash)
    flash_wrap.addWidget(self.lbl_flash)

    # Assemble toolbar (same order as before, but with View menu)
    tb.addWidget(self.btn_rec)
    tb.addWidget(self.btn_snap)
    tb.addSpacing(12)
    tb.addLayout(flash_wrap)
    tb.addSpacing(12)
    tb.addWidget(self.btn_ai_menu)
    tb.addWidget(self.btn_overlay_menu)
    tb.addStretch(1)
    tb.addWidget(self.btn_view_menu)
    tb.addWidget(self.btn_info)
    tb.addWidget(self.btn_lock)

    root_layout.addLayout(tb)
    root_layout.addWidget(self.view)

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------
    self._overlays = OverlayFlags()
    self._ai_enabled = True
    self._overlay_master_updating = False

    self._last_bgr = None
    self._last_ts = 0
    self._last_pkt = None
    self._last_pkt_ts = 0
    # keep overlays around a bit to avoid flicker
    self._overlay_ttl_ms = 750  # matches previous CameraWidget
    # Flash state
    self._flash_mode = getattr(self.cam_cfg, "flash_mode", "off") or "off"
    self._flash_level = int(getattr(self.cam_cfg, "flash_level", 512) or 512)
    self._flash_auto_target = int(getattr(self.cam_cfg, "flash_auto_target", 80) or 80)
    self._flash_auto_hyst = int(getattr(self.cam_cfg, "flash_auto_hyst", 15) or 15)
    self._flash_next_auto_ms = 0

    # Prebuffer recorder: per-camera
    self._recorder = PrebufferRecorder(
        cam_name=self.cam_cfg.name,
        out_dir=self.app_cfg.output_dir,
        fps=25,
        pre_ms=self.app_cfg.prebuffer_ms,
    )

    # Presence logging bus (per camera)
    # Presence logging bus (per camera) with configurable grace period
    face_params = FaceParams.load(str(self.app_cfg.models_dir))
    self._presence = PresenceBus(
        self.cam_cfg.name,
        self.app_cfg.logs_dir,
        ttl_ms=getattr(face_params, "presence_ttl_ms", 6000),
    )

    # Detector thread – use app-level settings (as in original)
    det_cfg = DetectorConfig.from_app(self.app_cfg)
    self._detector = DetectorThread(det_cfg, self.cam_cfg.name)
    self._detector.resultsReady.connect(self._on_detections)

    # Stream capture backend
    self._capture = StreamCapture(self.cam_cfg)

    # Poll frames from StreamCapture for UI at a video-friendly cadence (~30 FPS)
    self._frame_timer = QtCore.QTimer(self)
    self._frame_timer.setInterval(33)  # 33 ms ≈ 30 FPS for smoother UI

    # Lock state / MDI subwindow tracking
    self._locked = False
    self._subwindow = None
    self._locked_geometry = QtCore.QRect()

    # ------------------------------------------------------------------
    # Wiring
    # ------------------------------------------------------------------
    # Frame polling
    self._frame_timer.timeout.connect(self._poll_frame)

    # Recording / snapshot
    self.btn_rec.clicked.connect(self._toggle_recording)
    self.btn_snap.clicked.connect(self._snapshot)
    self.cb_flash.currentTextChanged.connect(self._on_flash_mode_changed)
    self.s_flash.valueChanged.connect(self._on_flash_level_changed)

    # View menu actions -> view helpers (provided by camera_widget_view.py)
    self.act_view_fit.triggered.connect(self.fit_to_window)
    self.act_view_100.triggered.connect(self.zoom_100)
    self.act_view_fit_window.triggered.connect(self.fit_window_to_video)

    # Lock
    self.btn_lock.toggled.connect(self._on_lock_toggled)
    self.btn_info.clicked.connect(self._show_info)

    # AI + overlay menu actions
    self.act_ai_enabled.toggled.connect(self._on_ai_toggled)
    self.act_ai_yolo.toggled.connect(self._on_ai_yolo_toggled)
    self.act_ai_faces.toggled.connect(self._on_ai_faces_toggled)
    self.act_ai_pets.toggled.connect(self._on_ai_pets_toggled)
    self.act_overlay_detections.toggled.connect(self._on_overlay_master_toggled)
    self.act_overlay_hud.toggled.connect(self._on_overlay_hud_toggled)
    self.act_overlay_stats.toggled.connect(self._on_overlay_stats_toggled)

    # Enrollment service – singleton
    self._enrollment = EnrollmentService.instance()

    # Intercept view move/resize when locked
    self.view.installEventFilter(self)

    # Defaults
    self._overlays.hud = True
    self._overlays.yolo = True
    self._overlays.faces = True
    self._overlays.pets = True
    self._overlays.stats = True

    # Initialize flash controls/state
    self.s_flash.blockSignals(True)
    self.cb_flash.blockSignals(True)
    self.s_flash.setValue(self._flash_level)
    self.lbl_flash.setText(str(self._flash_level))
    mode_idx = {"off": 0, "on": 1, "auto": 2}.get(self._flash_mode.lower(), 0)
    self.cb_flash.setCurrentIndex(mode_idx)
    self.cb_flash.blockSignals(False)
    self.s_flash.blockSignals(False)
    self._apply_flash_mode(initial=True)
