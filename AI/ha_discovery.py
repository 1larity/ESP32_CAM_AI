# ha_discovery.py
# Publish Home Assistant discovery configs for MQTT entities.
from __future__ import annotations

import json
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
    mqtt.publish(topic, json.dumps(payload), retain=True)


def publish_discovery(mqtt: MqttService, cameras: Iterable[object], discovery_prefix: str, base_topic: str) -> None:
    if mqtt is None or not getattr(mqtt, "connected", False):
        return
    prefix = discovery_prefix.strip("/")
    avail = f"{base_topic}/status"
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
