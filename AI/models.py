# models.py
# Auto-verifies required models and downloads defaults without user prompts.
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import requests
from settings import AppSettings
from utils import ensure_dir

YOLO_URL_DEFAULT = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.onnx"
HAAR_URL_DEFAULT = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
YUNET_URL_DEFAULT = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"


@dataclass
class ModelPaths:
    yolo: Path
    haar: Path
    face: Path


class ModelManager:
    @staticmethod
    def paths(cfg: AppSettings) -> ModelPaths:
        return ModelPaths(
            yolo=Path(cfg.models_dir) / "yolov8n.onnx",
            haar=Path(cfg.models_dir) / "haarcascade_frontalface_default.xml",
            face=Path(cfg.face_model) if getattr(cfg, "face_model", None) else Path(cfg.models_dir) / "face_yunet.onnx",
        )

    @staticmethod
    def ensure_models(cfg: AppSettings, status_cb: Callable[[str], None] | None = None) -> None:
        """
        Ensure required models exist. Downloads missing ones silently (no dialogs).
        status_cb(text) can be provided to surface progress (e.g., loader screen).
        """
        p = ModelManager.paths(cfg)
        tasks: list[tuple[str, Path, str]] = []
        if not p.yolo.exists():
            tasks.append(("YOLOv8n ONNX", p.yolo, getattr(cfg, "yolo_url", None) or YOLO_URL_DEFAULT))
        if not p.haar.exists():
            tasks.append(("Haar face cascade", p.haar, getattr(cfg, "haar_url", None) or HAAR_URL_DEFAULT))
        if not p.face.exists():
            tasks.append(("YuNet face detector", p.face, YUNET_URL_DEFAULT))

        for label, path, url in tasks:
            if status_cb:
                status_cb(f"Downloading {label}...")
            ensure_dir(path.parent)
            ModelManager._download(url, path)
        if status_cb and not tasks:
            status_cb("Models present")

    @staticmethod
    def _download(url: str, dest: Path) -> None:
        resp = requests.get(url, stream=True, timeout=(5, 60))
        resp.raise_for_status()
        ensure_dir(dest.parent)
        with dest.open("wb") as fp:
            for chunk in resp.iter_content(chunk_size=65536):
                if chunk:
                    fp.write(chunk)
