# models.py
# Auto-verifies required models and downloads defaults without user prompts.
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import os
import time
import requests
from settings import AppSettings
from utils import ensure_dir, DEBUG_LOG_FILE

YOLO_MODEL_NAMES = ["yolo11n.onnx"]
YOLO_URL_DEFAULTS = [
    "https://github.com/ultralytics/assets/releases/latest/download/yolo11n.onnx",
    "https://github.com/ultralytics/assets/releases/download/v11.0.0/yolo11n.onnx",
    "https://github.com/ultralytics/assets/releases/download/v11.0.1/yolo11n.onnx",
]
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
            yolo=ModelManager._resolve_yolo_path(cfg),
            haar=Path(cfg.models_dir) / "haarcascade_frontalface_default.xml",
            face=Path(cfg.face_model) if getattr(cfg, "face_model", None) else Path(cfg.models_dir) / "face_yunet.onnx",
        )

    @staticmethod
    def _log(msg: str) -> None:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] [models] {msg}"
        try:
            ensure_dir(DEBUG_LOG_FILE.parent)
            with DEBUG_LOG_FILE.open("a", encoding="utf-8") as fp:
                fp.write(line + "\n")
        except Exception:
            pass

    @staticmethod
    def _is_valid_model_file(path: Path) -> bool:
        try:
            return path.is_file() and path.stat().st_size > 0
        except Exception:
            return False

    @staticmethod
    def _resolve_yolo_path(cfg: AppSettings) -> Path:
        models_dir = Path(cfg.models_dir)
        for name in YOLO_MODEL_NAMES:
            candidate = models_dir / name
            if ModelManager._is_valid_model_file(candidate):
                return candidate
        return models_dir / YOLO_MODEL_NAMES[0]

    @staticmethod
    def ensure_models(cfg: AppSettings, status_cb: Callable[[str], None] | None = None) -> None:
        """
        Ensure required models exist. Downloads missing ones silently (no dialogs).
        status_cb(text) can be provided to surface progress (e.g., loader screen).
        """
        p = ModelManager.paths(cfg)
        tasks: list[tuple[str, Path, list[str]]] = []
        if not ModelManager._is_valid_model_file(p.yolo):
            yolo_url = getattr(cfg, "yolo_url", None)
            urls = [yolo_url] if yolo_url else list(YOLO_URL_DEFAULTS)
            tasks.append(("YOLO11n ONNX", p.yolo, urls))
        if not ModelManager._is_valid_model_file(p.haar):
            haar_url = getattr(cfg, "haar_url", None)
            urls = [haar_url] if haar_url else [HAAR_URL_DEFAULT]
            tasks.append(("Haar face cascade", p.haar, urls))
        if not ModelManager._is_valid_model_file(p.face):
            tasks.append(("YuNet face detector", p.face, [YUNET_URL_DEFAULT]))

        if not tasks:
            if status_cb:
                status_cb("Models present")
            ModelManager._log("models present")
            return

        errors: list[str] = []
        for label, path, urls in tasks:
            if status_cb:
                status_cb(f"Downloading {label}...")
            success = False
            last_err: Exception | None = None
            for url in urls:
                ModelManager._log(f"download start: {label} from {url}")
                try:
                    ensure_dir(path.parent)
                    ModelManager._download(url, path)
                    if not ModelManager._is_valid_model_file(path):
                        raise RuntimeError("downloaded file missing or empty")
                    ModelManager._log(f"download ok: {label} -> {path}")
                    success = True
                    break
                except Exception as e:
                    last_err = e
                    ModelManager._log(f"download failed: {label} from {url} ({e})")
            if not success:
                err_msg = f"{label} ({last_err})" if last_err else label
                errors.append(err_msg)

        if errors:
            if status_cb:
                status_cb(f"Model download failed: {', '.join(errors)}")
            raise RuntimeError(f"Missing required models: {', '.join(errors)}")

    @staticmethod
    def _download(url: str, dest: Path) -> None:
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        resp = requests.get(url, stream=True, timeout=(5, 60))
        try:
            resp.raise_for_status()
            ensure_dir(dest.parent)
            with tmp.open("wb") as fp:
                for chunk in resp.iter_content(chunk_size=65536):
                    if chunk:
                        fp.write(chunk)
            os.replace(tmp, dest)
        finally:
            try:
                resp.close()
            finally:
                try:
                    if tmp.exists():
                        tmp.unlink()
                except Exception:
                    pass
