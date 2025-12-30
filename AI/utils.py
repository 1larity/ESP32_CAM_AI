# utils.py
# Shared helpers: timing, paths, image conversion, and lightweight debug routing.
from __future__ import annotations

import time
from pathlib import Path
from enum import IntFlag
import numpy as np, cv2 as cv
from PySide6 import QtGui, QtWidgets, QtCore


class DebugMode(IntFlag):
    OFF = 0
    LOG = 1
    PRINT = 2
    BOTH = LOG | PRINT


# Global debug mode; tweak at runtime (e.g., utils.DEBUG_MODE = DebugMode.PRINT).
DEBUG_MODE: DebugMode = DebugMode.OFF
DEBUG_LOG_FILE = Path(__file__).resolve().parent / "logs" / "debug.log"


def monotonic_ms() -> int:
    return int(time.monotonic() * 1000)


def timestamp_name() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def ensure_dir(p: Path):
    Path(p).mkdir(parents=True, exist_ok=True)


def debug(msg: str) -> None:
    """
    Lightweight debug sink controlled by DEBUG_MODE.
    Modes:
      - OFF: no-op
      - LOG: append to AI/logs/debug.log (auto-creates folder)
      - PRINT: print to stdout
      - BOTH: log + print
    """
    mode = DEBUG_MODE
    if mode is DebugMode.OFF:
        return

    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"

    if mode & DebugMode.PRINT:
        print(line)
    if mode & DebugMode.LOG:
        try:
            ensure_dir(DEBUG_LOG_FILE.parent)
            with DEBUG_LOG_FILE.open("a", encoding="utf-8") as fp:
                fp.write(line + "\n")
        except Exception:
            # Avoid propagating debug failures into runtime
            pass


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
