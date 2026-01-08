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
    mqtt_publish: bool = True
    alt_streams: list[str] = field(default_factory=list)  # optional alternates (e.g., substreams /102)
    view_scale: float = 1.0           # persisted zoom level
    # Per-camera recording overrides (fallback to app defaults if None)
    record_motion: Optional[bool] = None
    motion_sensitivity: Optional[int] = None
    # Per-camera AI toggles (fallback to app defaults of enabled)
    ai_enabled: Optional[bool] = None
    ai_yolo: Optional[bool] = None
    ai_faces: Optional[bool] = None
    ai_pets: Optional[bool] = None
    overlay_scale: Optional[float] = None  # text/box scale computed on first frame
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
    # MQTT / Home Assistant
    mqtt_enabled: bool = False
    mqtt_host: Optional[str] = None
    mqtt_port: int = 8883
    mqtt_client_id: Optional[str] = None
    mqtt_tls: bool = True
    mqtt_ca_path: Optional[str] = None
    mqtt_insecure: bool = False
    mqtt_keepalive: int = 60
    mqtt_base_topic: str = "esp32_cam_ai"
    mqtt_discovery_prefix: str = "homeassistant"
    # If True, discovery configs are published under mqtt_base_topic (legacy).
    # If False, discovery configs are published to mqtt_discovery_prefix directly (Home Assistant default).
    mqtt_discovery_under_base_topic: bool = False
    mqtt_user: Optional[str] = None
    # mqtt_password is kept in memory only; persisted as mqtt_password_enc
    mqtt_password: Optional[str] = None

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
                    mqtt_publish=bool(c.get("mqtt_publish", True)),
                    alt_streams=c.get("alt_streams", []) or [],
                    view_scale=float(c.get("view_scale", 1.0) or 1.0),
                    # flash fields removed
                    record_motion=c.get("record_motion", raw.get("record_motion")),
                    motion_sensitivity=c.get("motion_sensitivity", raw.get("motion_sensitivity")),
                    ai_enabled=c.get("ai_enabled"),
                    ai_yolo=c.get("ai_yolo"),
                    ai_faces=c.get("ai_faces"),
                    ai_pets=c.get("ai_pets"),
                    overlay_scale=c.get("overlay_scale"),
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
            mqtt_enabled=bool(raw.get("mqtt_enabled", False)),
            mqtt_host=raw.get("mqtt_host"),
            mqtt_port=int(raw.get("mqtt_port", 8883)),
            mqtt_client_id=raw.get("mqtt_client_id"),
            mqtt_tls=bool(raw.get("mqtt_tls", True)),
            mqtt_ca_path=raw.get("mqtt_ca_path"),
            mqtt_insecure=bool(raw.get("mqtt_insecure", False)),
            mqtt_keepalive=int(raw.get("mqtt_keepalive", 60)),
            mqtt_base_topic=raw.get("mqtt_base_topic", "esp32_cam_ai"),
            mqtt_discovery_prefix=raw.get("mqtt_discovery_prefix", "homeassistant"),
            mqtt_discovery_under_base_topic=bool(raw.get("mqtt_discovery_under_base_topic", False)),
            mqtt_user=raw.get("mqtt_user"),
            mqtt_password=decrypt(raw.get("mqtt_password_enc", "") or "")[1] if raw.get("mqtt_password_enc") else None,
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
        "mqtt_enabled": cfg.mqtt_enabled,
        "mqtt_host": cfg.mqtt_host,
        "mqtt_port": cfg.mqtt_port,
        "mqtt_client_id": cfg.mqtt_client_id,
        "mqtt_tls": cfg.mqtt_tls,
        "mqtt_ca_path": cfg.mqtt_ca_path,
        "mqtt_insecure": cfg.mqtt_insecure,
        "mqtt_keepalive": cfg.mqtt_keepalive,
        "mqtt_base_topic": cfg.mqtt_base_topic,
        "mqtt_discovery_prefix": cfg.mqtt_discovery_prefix,
        "mqtt_discovery_under_base_topic": cfg.mqtt_discovery_under_base_topic,
        "mqtt_user": cfg.mqtt_user,
        "mqtt_password_enc": encrypt(cfg.mqtt_password) if cfg.mqtt_password else None,
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
