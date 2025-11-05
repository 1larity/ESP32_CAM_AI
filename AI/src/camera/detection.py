# camera/detection.py

from PySide6 import QtCore
import threading
import time
from typing import List, Tuple

class DetectionThread(QtCore.QThread):
    resultsReady = QtCore.Signal(list, list)  # dets, faces

    def __init__(self, widget: 'CameraWidget', interval_ms: int = 200, parent=None):
        super().__init__(parent)
        self.w = widget
        self.interval = interval_ms
        self._stop = threading.Event()
        self.max_skip_cycles = 1

    # … (same implementation as before) …