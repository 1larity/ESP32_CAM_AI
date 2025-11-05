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