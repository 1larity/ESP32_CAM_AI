# settings.py
# Base-path aware settings. Paths are anchored to the AI folder (this file's parent).
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional
from pathlib import Path
import json
import os

# --- Project base directory (…/ESP32_CAM_AI/AI) ---
BASE_DIR = Path(__file__).resolve().parent

SETTINGS_FILE = BASE_DIR / "config" / "app_settings.json"

def _abs_under_base(p: Path) -> Path:
    # Make absolute under BASE_DIR if relative
    if not p.is_absolute():
        p = BASE_DIR / p
    return p

@dataclass
class CameraSettings:
    name: str
    stream_url: str
    user: Optional[str] = None
    password: Optional[str] = None
    token: Optional[str] = None

    @classmethod
    def from_ip(cls, name: str, host: str, user: Optional[str] = None,
                password: Optional[str] = None, token: Optional[str] = None):
        host = host.strip()
        base = f"http://{host}:81/stream"
        if token:
            sep = "&" if "?" in base else "?"
            base = f"{base}{sep}token={token}"
        return cls(name=name, stream_url=base, user=user, password=password, token=token)

    def effective_url(self) -> str:
        return self.stream_url

@dataclass
class AppSettings:
    models_dir: Path = Path("models")
    output_dir: Path = Path("recordings")
    logs_dir: Path = Path("logs")
    detect_interval_ms: int = 500
    thresh_yolo: float = 0.35
    prebuffer_ms: int = 3000
    yolo_url: Optional[str] = None
    haar_url: Optional[str] = None
    window_geometry: Optional[str] = None  # hex-encoded QByteArray
    window_state: Optional[str] = None     # hex-encoded QByteArray
    window_geometries: Dict[str, List[int]] = field(default_factory=dict)  # cam_name -> [x,y,w,h,maximized]
    cameras: List[CameraSettings] = field(default_factory=list)

def load_settings() -> AppSettings:
    if SETTINGS_FILE.exists():
        with SETTINGS_FILE.open("r", encoding="utf-8") as fp:
            raw = json.load(fp)
        cams = [CameraSettings(**c) for c in raw.get("cameras", [])]
        cfg = AppSettings(
            models_dir=Path(raw.get("models_dir", "models")),
            output_dir=Path(raw.get("output_dir", "recordings")),
            logs_dir=Path(raw.get("logs_dir", "logs")),
            detect_interval_ms=int(raw.get("detect_interval_ms", 500)),
            thresh_yolo=float(raw.get("thresh_yolo", 0.35)),
            prebuffer_ms=int(raw.get("prebuffer_ms", 3000)),
            yolo_url=raw.get("yolo_url"),
            haar_url=raw.get("haar_url"),
            window_geometry=raw.get("window_geometry"),
            window_state=raw.get("window_state"),
            window_geometries=raw.get("window_geometries", {}) or {},
            cameras=cams
        )
    else:
        cfg = AppSettings()

    # Anchor to BASE_DIR
    cfg.models_dir = _abs_under_base(cfg.models_dir)
    cfg.output_dir = _abs_under_base(cfg.output_dir)
    cfg.logs_dir   = _abs_under_base(cfg.logs_dir)
    return cfg

def save_settings(cfg: AppSettings):
    SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    data = {
        # store as relative-to-base where possible for portability
        "models_dir": str(Path(cfg.models_dir).resolve().relative_to(BASE_DIR) if str(cfg.models_dir).startswith(str(BASE_DIR)) else str(cfg.models_dir)),
        "output_dir": str(Path(cfg.output_dir).resolve().relative_to(BASE_DIR) if str(cfg.output_dir).startswith(str(BASE_DIR)) else str(cfg.output_dir)),
        "logs_dir":   str(Path(cfg.logs_dir).resolve().relative_to(BASE_DIR) if str(cfg.logs_dir).startswith(str(BASE_DIR)) else str(cfg.logs_dir)),
        "detect_interval_ms": cfg.detect_interval_ms,
        "thresh_yolo": cfg.thresh_yolo,
        "prebuffer_ms": cfg.prebuffer_ms,
        "yolo_url": cfg.yolo_url,
        "haar_url": cfg.haar_url,
        "window_geometry": cfg.window_geometry,
        "window_state": cfg.window_state,
        "window_geometries": cfg.window_geometries,
        "cameras": [asdict(c) for c in cfg.cameras],
    }
    with SETTINGS_FILE.open("w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=2)
