# settings.py
# Base-path aware settings. Paths are anchored to the AI folder (this file's parent).
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional
from pathlib import Path
import json
import os
from crypto_utils import encrypt, decrypt

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
    flash_mode: str = "off"         # off | on | auto
    flash_level: int = 128          # 0-255
    flash_auto_target: int = 80     # desired brightness (0-255)
    flash_auto_hyst: int = 15       # hysteresis band (0-255)
    # Per-camera recording overrides (fallback to app defaults if None)
    record_motion: Optional[bool] = None
    motion_sensitivity: Optional[int] = None
    # Per-camera AI toggles (fallback to app defaults of enabled)
    ai_enabled: Optional[bool] = None
    ai_yolo: Optional[bool] = None
    ai_faces: Optional[bool] = None
    ai_pets: Optional[bool] = None
    # Per-camera orientation
    rotation_deg: int = 0           # 0, 90, 180, 270
    flip_horizontal: bool = False
    flip_vertical: bool = False

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
    use_gpu: bool = False  # YOLO/face DNN backend target
    thresh_yolo: float = 0.35
    prebuffer_ms: int = 3000
    yolo_url: Optional[str] = None
    haar_url: Optional[str] = None
    face_model: Optional[str] = None
    window_geometry: Optional[str] = None  # hex-encoded QByteArray
    window_state: Optional[str] = None     # hex-encoded QByteArray
    window_geometries: Dict[str, List[int]] = field(default_factory=dict)  # cam_name -> [x,y,w,h,maximized]
    cameras: List[CameraSettings] = field(default_factory=list)
    collect_unknown_faces: bool = False
    collect_unknown_pets: bool = False
    ignore_enrollment_models: bool = False
    unknown_capture_limit: int = 50  # max images per class per cam
    auto_train_unknowns: bool = False
    # Security recording triggers
    record_person: bool = False
    record_unknown_person: bool = False
    record_pet: bool = False
    record_unknown_pet: bool = False
    record_motion: bool = False
    motion_sensitivity: int = 50  # 0-100

def load_settings() -> AppSettings:
    if SETTINGS_FILE.exists():
        with SETTINGS_FILE.open("r", encoding="utf-8") as fp:
            raw = json.load(fp)
        cams: List[CameraSettings] = []
        for c in raw.get("cameras", []):
            # Backward-compatible: support plaintext password or encrypted password_enc
            pwd = c.get("password")
            enc = c.get("password_enc")
            if not pwd and enc:
                ok, dec = decrypt(enc)
                pwd = dec if ok else None
            cams.append(
                CameraSettings(
                    name=c.get("name"),
                    stream_url=c.get("stream_url"),
                    user=c.get("user"),
                    password=pwd,
                    token=c.get("token"),
                    flash_mode=c.get("flash_mode", "off"),
                    flash_level=max(0, min(255, int(c.get("flash_level", 128)))),
                    flash_auto_target=int(c.get("flash_auto_target", 80)),
                    flash_auto_hyst=int(c.get("flash_auto_hyst", 15)),
                    record_motion=c.get("record_motion", raw.get("record_motion")),
                    motion_sensitivity=c.get("motion_sensitivity", raw.get("motion_sensitivity")),
                    ai_enabled=c.get("ai_enabled"),
                    ai_yolo=c.get("ai_yolo"),
                    ai_faces=c.get("ai_faces"),
                    ai_pets=c.get("ai_pets"),
                    rotation_deg=int(c.get("rotation_deg", 0) or 0),
                    flip_horizontal=bool(c.get("flip_horizontal", False)),
                    flip_vertical=bool(c.get("flip_vertical", False)),
                )
            )
        cfg = AppSettings(
            models_dir=Path(raw.get("models_dir", "models")),
            output_dir=Path(raw.get("output_dir", "recordings")),
            logs_dir=Path(raw.get("logs_dir", "logs")),
            detect_interval_ms=int(raw.get("detect_interval_ms", 500)),
            use_gpu=bool(raw.get("use_gpu", False)),
            thresh_yolo=float(raw.get("thresh_yolo", 0.35)),
            prebuffer_ms=int(raw.get("prebuffer_ms", 3000)),
            yolo_url=raw.get("yolo_url"),
            haar_url=raw.get("haar_url"),
            face_model=raw.get("face_model"),
            window_geometry=raw.get("window_geometry"),
            window_state=raw.get("window_state"),
            window_geometries=raw.get("window_geometries", {}) or {},
            cameras=cams,
            collect_unknown_faces=bool(raw.get("collect_unknown_faces", False)),
            collect_unknown_pets=bool(raw.get("collect_unknown_pets", False)),
            ignore_enrollment_models=bool(raw.get("ignore_enrollment_models", False)),
            unknown_capture_limit=int(raw.get("unknown_capture_limit", 50)),
            auto_train_unknowns=bool(raw.get("auto_train_unknowns", False)),
            record_person=bool(raw.get("record_person", False)),
            record_unknown_person=bool(raw.get("record_unknown_person", False)),
            record_pet=bool(raw.get("record_pet", False)),
            record_unknown_pet=bool(raw.get("record_unknown_pet", False)),
            record_motion=bool(raw.get("record_motion", False)),
            motion_sensitivity=int(raw.get("motion_sensitivity", 50)),
        )
    else:
        cfg = AppSettings()

    # Anchor to BASE_DIR
    cfg.models_dir = _abs_under_base(cfg.models_dir)
    cfg.output_dir = _abs_under_base(cfg.output_dir)
    cfg.logs_dir   = _abs_under_base(cfg.logs_dir)
    # Default face model to models/face_yunet.onnx if not set
    if not cfg.face_model:
        cfg.face_model = str(cfg.models_dir / "face_yunet.onnx")
    return cfg

def save_settings(cfg: AppSettings):
    SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    data = {
        # store as relative-to-base where possible for portability
        "models_dir": str(Path(cfg.models_dir).resolve().relative_to(BASE_DIR) if str(cfg.models_dir).startswith(str(BASE_DIR)) else str(cfg.models_dir)),
        "output_dir": str(Path(cfg.output_dir).resolve().relative_to(BASE_DIR) if str(cfg.output_dir).startswith(str(BASE_DIR)) else str(cfg.output_dir)),
        "logs_dir":   str(Path(cfg.logs_dir).resolve().relative_to(BASE_DIR) if str(cfg.logs_dir).startswith(str(BASE_DIR)) else str(cfg.logs_dir)),
        "detect_interval_ms": cfg.detect_interval_ms,
        "use_gpu": cfg.use_gpu,
        "thresh_yolo": cfg.thresh_yolo,
        "prebuffer_ms": cfg.prebuffer_ms,
        "yolo_url": cfg.yolo_url,
        "haar_url": cfg.haar_url,
        "face_model": str(cfg.face_model) if cfg.face_model else None,
        "window_geometry": cfg.window_geometry,
        "window_state": cfg.window_state,
        "window_geometries": cfg.window_geometries,
        "collect_unknown_faces": cfg.collect_unknown_faces,
        "collect_unknown_pets": cfg.collect_unknown_pets,
        "ignore_enrollment_models": cfg.ignore_enrollment_models,
        "unknown_capture_limit": cfg.unknown_capture_limit,
        "auto_train_unknowns": cfg.auto_train_unknowns,
        "record_person": cfg.record_person,
        "record_unknown_person": cfg.record_unknown_person,
        "record_pet": cfg.record_pet,
        "record_unknown_pet": cfg.record_unknown_pet,
        "record_motion": cfg.record_motion,
        "motion_sensitivity": cfg.motion_sensitivity,
        "cameras": [],
    }
    # Write cameras with encrypted password; avoid persisting plaintext.
    for c in cfg.cameras:
        entry = asdict(c)
        pwd = entry.pop("password", None)
        entry["password_enc"] = encrypt(pwd) if pwd else None
        data["cameras"].append(entry)
    with SETTINGS_FILE.open("w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=2)
