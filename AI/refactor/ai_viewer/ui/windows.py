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
