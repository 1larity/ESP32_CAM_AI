# ha_discovery.py
# Publish Home Assistant discovery configs for MQTT entities.
from __future__ import annotations

import json
import re
import zlib
from pathlib import Path
from typing import Iterable

from mqtt_client import MqttService


def _device_block(cam_name: str) -> dict:
    ident = f"esp32_cam_ai_{cam_name}"
    return {
        "identifiers": [ident],
        "name": f"ESP32-CAM AI - {cam_name}",
        "manufacturer": "ESP32-CAM-AI",
        "model": "ESP32-CAM AI Viewer",
    }


def _pub_config(mqtt: MqttService, topic: str, payload: dict) -> None:
    payload_json = json.dumps(payload)
    fn = getattr(mqtt, "publish_discovery_config", None)
    if callable(fn):
        fn(topic, payload_json, retain=True)
    else:
        mqtt.publish(topic, payload_json, retain=True)


def _slug(s: str, fallback_prefix: str) -> str:
    """
    Home Assistant IDs must be URL-ish; keep it simple and stable.
    """
    raw = (s or "").strip()
    raw = raw.replace(" ", "_")
    out = re.sub(r"[^0-9A-Za-z_]+", "_", raw).strip("_").lower()
    if out:
        return out
    crc = zlib.crc32(raw.encode("utf-8")) & 0xFFFFFFFF
    return f"{fallback_prefix}_{crc:08x}"


def _load_face_labels(models_dir: object | None) -> list[str]:
    """
    Return known face label names from models_dir/labels_faces.json (written by LBPH training).

    File format is { "<name>": <id>, ... }.
    """
    if not models_dir:
        return []
    try:
        p = Path(str(models_dir))
        labels_path = p / "labels_faces.json"
        if not labels_path.exists():
            return []
        raw = json.loads(labels_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return []
        names = [str(k).strip() for k in raw.keys()]
        names = [n for n in names if n]
        # Stable ordering for predictable entity IDs.
        return sorted(set(names), key=str.casefold)
    except Exception:
        return []


def publish_discovery(mqtt: MqttService, cameras: Iterable[object], discovery_prefix: str, base_topic: str) -> None:
    if mqtt is None or not getattr(mqtt, "connected", False):
        return
    prefix = discovery_prefix.strip("/")
    avail = f"{base_topic}/status"
    face_labels = _load_face_labels(getattr(getattr(mqtt, "cfg", None), "models_dir", None))
    for cam in cameras:
        name = getattr(cam, "name", None) or "cam"
        dev = _device_block(name)

        obj_id = name.replace(" ", "_")
        topic_cam = obj_id

        # binary_sensor: person present
        cfg = {
            "name": f"{name} Person",
            "unique_id": f"{obj_id}_person_present",
            "state_topic": f"{base_topic}/{topic_cam}/presence/person",
            "payload_on": "ON",
            "payload_off": "OFF",
            "availability_topic": avail,
            "device_class": "occupancy",
            "device": dev,
        }
        _pub_config(mqtt, f"{prefix}/binary_sensor/{obj_id}_person/config", cfg)

        # binary_sensor: pet present
        cfg = {
            "name": f"{name} Pet",
            "unique_id": f"{obj_id}_pet_present",
            "state_topic": f"{base_topic}/{topic_cam}/presence/pet",
            "payload_on": "ON",
            "payload_off": "OFF",
            "availability_topic": avail,
            "device_class": "presence",
            "device": dev,
        }
        _pub_config(mqtt, f"{prefix}/binary_sensor/{obj_id}_pet/config", cfg)

        # sensor: person count
        cfg = {
            "name": f"{name} Person Count",
            "unique_id": f"{obj_id}_person_count",
            "state_topic": f"{base_topic}/{topic_cam}/counts/person",
            "availability_topic": avail,
            "device_class": "occupancy",
            "device": dev,
        }
        _pub_config(mqtt, f"{prefix}/sensor/{obj_id}_person_count/config", cfg)

        # sensor: pet count
        cfg = {
            "name": f"{name} Pet Count",
            "unique_id": f"{obj_id}_pet_count",
            "state_topic": f"{base_topic}/{topic_cam}/counts/pet",
            "availability_topic": avail,
            "device": dev,
        }
        _pub_config(mqtt, f"{prefix}/sensor/{obj_id}_pet_count/config", cfg)

        # sensor: recognized names
        cfg = {
            "name": f"{name} Recognized",
            "unique_id": f"{obj_id}_recognized",
            "state_topic": f"{base_topic}/{topic_cam}/recognized",
            "availability_topic": avail,
            "device": dev,
            "icon": "mdi:account-badge",
        }
        _pub_config(mqtt, f"{prefix}/sensor/{obj_id}_recognized/config", cfg)

        # device_tracker: person present (maps ON/OFF to home/not_home for HA "person" integration)
        cfg = {
            "name": f"{name} Person (Camera)",
            "unique_id": f"{obj_id}_person_tracker",
            "state_topic": f"{base_topic}/{topic_cam}/presence/person",
            "payload_home": "ON",
            "payload_not_home": "OFF",
            "availability_topic": avail,
            "source_type": "router",
            "device": dev,
        }
        _pub_config(mqtt, f"{prefix}/device_tracker/{obj_id}_person/config", cfg)

        # device_tracker: known recognised names (requires LBPH labels_faces.json)
        for label in face_labels:
            if label.casefold() in ("unknown", "face", "person"):
                continue
            label_id = _slug(label, "person")
            cfg = {
                "name": f"{label} ({name})",
                "unique_id": f"{obj_id}_person_{label_id}_tracker",
                "state_topic": f"{base_topic}/{topic_cam}/presence/person/{label}",
                "payload_home": "ON",
                "payload_not_home": "OFF",
                "availability_topic": avail,
                "source_type": "router",
                "device": dev,
                "icon": "mdi:account",
            }
            _pub_config(mqtt, f"{prefix}/device_tracker/{obj_id}_person_{label_id}/config", cfg)
