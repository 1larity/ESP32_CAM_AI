# models.py
# Uses settings.BASE_DIR-backed folders; unchanged logic with explicit BASE_DIR use.
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
import requests
from PyQt6 import QtWidgets
from settings import AppSettings, BASE_DIR
from utils import ensure_dir

YOLO_URL_DEFAULT = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.onnx"
HAAR_URL_DEFAULT = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"

@dataclass
class ModelPaths:
    yolo: Path
    haar: Path

class ModelManager:
    @staticmethod
    def paths(cfg: AppSettings) -> ModelPaths:
        return ModelPaths(
            yolo=Path(cfg.models_dir) / "yolov8n.onnx",
            haar=Path(cfg.models_dir) / "haarcascade_frontalface_default.xml"
        )

    @staticmethod
    def ensure_models(parent: QtWidgets.QWidget, cfg: AppSettings):
        p = ModelManager.paths(cfg)
        missing = []
        if not p.yolo.exists():
            missing.append(("YOLOv8n ONNX", p.yolo, cfg.yolo_url or YOLO_URL_DEFAULT))
        if not p.haar.exists():
            missing.append(("Haar face cascade", p.haar, cfg.haar_url or HAAR_URL_DEFAULT))
        if not missing:
            return
        mb = QtWidgets.QMessageBox(parent)
        mb.setWindowTitle("Models missing")
        mb.setText(f"Models folder:\n{cfg.models_dir}\n\nDownload defaults now?")
        mb.setStandardButtons(QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No)
        if mb.exec() == QtWidgets.QMessageBox.StandardButton.Yes:
            ModelManager._download_many(parent, missing)

    @staticmethod
    def fetch_defaults(parent: QtWidgets.QWidget, cfg: AppSettings):
        p = ModelManager.paths(cfg)
        todo = []
        if not p.yolo.exists():
            todo.append(("YOLOv8n ONNX", p.yolo, cfg.yolo_url or YOLO_URL_DEFAULT))
        if not p.haar.exists():
            todo.append(("Haar face cascade", p.haar, cfg.haar_url or HAAR_URL_DEFAULT))
        if not todo:
            QtWidgets.QMessageBox.information(parent, "Models", "All default models already present.")
            return
        ModelManager._download_many(parent, todo)

    @staticmethod
    def _download_many(parent: QtWidgets.QWidget, items: list[Tuple[str, Path, str]]):
        for label, path, url in items:
            ensure_dir(path.parent)
            prog = _DownloadDialog(parent, f"Downloading {label}…", url, path)
            ok = prog.exec_and_download()
            if not ok:
                QtWidgets.QMessageBox.warning(parent, "Download failed", f"Could not fetch {label}.\nURL: {url}")

class _DownloadDialog(QtWidgets.QDialog):
    def __init__(self, parent, title: str, url: str, path: Path):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.url = url
        self.path = path
        from PyQt6 import QtWidgets
        self.pb = QtWidgets.QProgressBar()
        self.lbl = QtWidgets.QLabel(url)
        self.btn = QtWidgets.QPushButton("Cancel")
        self.btn.clicked.connect(self.reject)
        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(self.lbl); lay.addWidget(self.pb); lay.addWidget(self.btn)
        self._ok = False

    def exec_and_download(self) -> bool:
        from PyQt6 import QtWidgets
        try:
            with requests.get(self.url, stream=True, timeout=(5, 60)) as r:
                r.raise_for_status()
                total = int(r.headers.get("Content-Length", 0)) or None
                got = 0
                with self.path.open("wb") as fp:
                    for chunk in r.iter_content(chunk_size=65536):
                        if not chunk: continue
                        fp.write(chunk); got += len(chunk)
                        if total:
                            self.pb.setMaximum(total); self.pb.setValue(got)
                        QtWidgets.QApplication.processEvents()
            self._ok = True
        except Exception:
            self._ok = False
        self.accept()
        return self._ok
