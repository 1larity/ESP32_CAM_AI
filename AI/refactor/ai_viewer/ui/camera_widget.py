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

