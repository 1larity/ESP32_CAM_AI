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