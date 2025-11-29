# utils.py
# open_folder_or_warn uses absolute, existing paths; no change except BASE_DIR import removed.
from __future__ import annotations
import time
from pathlib import Path
import numpy as np, cv2 as cv
from PySide6 import QtGui, QtWidgets, QtCore

def monotonic_ms() -> int:
    return int(time.monotonic() * 1000)

def timestamp_name() -> str:
    return time.strftime("%Y%m%d_%H%M%S")

def ensure_dir(p: Path):
    Path(p).mkdir(parents=True, exist_ok=True)

def qimage_from_bgr(bgr: np.ndarray) -> QtGui.QImage:
    h, w = bgr.shape[:2]
    rgb = cv.cvtColor(bgr, cv.COLOR_BGR2RGB)
    bpl = int(rgb.strides[0])
    return QtGui.QImage(rgb.data, w, h, bpl, QtGui.QImage.Format.Format_RGB888).copy()

def open_folder_or_warn(parent: QtWidgets.QWidget, path: Path):
    try:
        ensure_dir(path)
        ok = QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(Path(path).resolve())))
        if not ok:
            raise RuntimeError("OS refused to open path")
    except Exception as e:
        QtWidgets.QMessageBox.warning(parent, "Open folder failed", f"Could not open:\n{path}\n\n{e}")
