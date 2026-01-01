# mqtt_client.py
# Minimal MQTT client wrapper for Home Assistant integration.
from __future__ import annotations

import random
import string
import threading
from typing import Optional, List, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    try:
        from paho.mqtt.client import Client as MqttClient  # type: ignore
    except Exception:  # pragma: no cover - optional dep
        MqttClient = object  # fallback for type checkers

try:
    import paho.mqtt.client as mqtt
except Exception as e:
    print(f"[MQTT] paho-mqtt import failed: {e}")
    mqtt = None


def _clean_topic(s: Optional[str], default: str) -> str:
    if not s:
        return default
    t = str(s).strip().strip("/")
    return t or default


def _rand_id(prefix: str = "esp32-cam-ai") -> str:
    suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=6))
    return f"{prefix}-{suffix}"


class MqttService:
    """
    Lightweight MQTT client manager.

    Responsibilities:
      - Respect AppSettings MQTT fields
      - Handle TLS / auth / LWT
      - Expose publish() and stop()
    """

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        # Use a generic object to avoid type errors if paho isn't installed.
        self.client: Optional["MqttClient | object"] = None
        self.connected = False
        self._lock = threading.RLock()
        self._on_connect_cbs: List[Callable[["MqttService"], None]] = []
        self._base_topic = _clean_topic(getattr(cfg, "mqtt_base_topic", None), "esp32_cam_ai")
        self._avail_topic = f"{self._base_topic}/status"

    @property
    def base_topic(self) -> str:
        return self._base_topic

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        if not getattr(self.cfg, "mqtt_enabled", False):
            return
        if not getattr(self.cfg, "mqtt_host", None):
            print("[MQTT] enabled but host not set; skipping connect.")
            return
        if mqtt is None:
            print("[MQTT] paho-mqtt not installed; install 'paho-mqtt' to enable MQTT.")
            return

        client_id = getattr(self.cfg, "mqtt_client_id", None) or _rand_id()
        self.client = mqtt.Client(client_id=client_id, clean_session=True)

        user = getattr(self.cfg, "mqtt_user", None)
        pwd = getattr(self.cfg, "mqtt_password", None)
        if user:
            self.client.username_pw_set(user, pwd or None)

        # TLS
        if getattr(self.cfg, "mqtt_tls", True):
            ca = getattr(self.cfg, "mqtt_ca_path", None)
            try:
                self.client.tls_set(ca_certs=str(ca) if ca else None)
            except Exception as e:
                print(f"[MQTT] tls_set failed: {e}")
            if getattr(self.cfg, "mqtt_insecure", False):
                try:
                    self.client.tls_insecure_set(True)
                except Exception:
                    pass

        # LWT
        try:
            self.client.will_set(self._avail_topic, payload="offline", retain=True)
        except Exception:
            pass

        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect

        try:
            self.client.connect(
                host=self.cfg.mqtt_host,
                port=int(getattr(self.cfg, "mqtt_port", 8883)),
                keepalive=int(getattr(self.cfg, "mqtt_keepalive", 60)),
            )
            self.client.loop_start()
        except Exception as e:
            print(f"[MQTT] connect failed: {e}")

    def stop(self) -> None:
        with self._lock:
            if self.client is None:
                return
            try:
                if self.connected:
                    self.client.publish(self._avail_topic, payload="offline", retain=True)
                self.client.disconnect()
            except Exception:
                pass
            try:
                self.client.loop_stop()
            except Exception:
                pass
            self.client = None
            self.connected = False

    # ------------------------------------------------------------------ #
    # Publish helper
    # ------------------------------------------------------------------ #

    def publish(self, topic: str, payload: str | bytes, retain: bool = False, qos: int = 0) -> None:
        with self._lock:
            if not self.client or not self.connected:
                return
            try:
                full = f"{self._base_topic}/{topic.lstrip('/')}"
                self.client.publish(full, payload=payload, qos=int(qos), retain=retain)
            except Exception as e:
                print(f"[MQTT] publish error on {topic}: {e}")

    def add_on_connect(self, cb: Callable[["MqttService"], None]) -> None:
        with self._lock:
            self._on_connect_cbs.append(cb)

    # ------------------------------------------------------------------ #
    # Callbacks
    # ------------------------------------------------------------------ #

    def _on_connect(self, client, userdata, flags, rc):  # type: ignore[override]
        self.connected = True
        try:
            client.publish(self._avail_topic, payload="online", retain=True)
        except Exception:
            pass
        print(f"[MQTT] connected (rc={rc})")
        for cb in list(self._on_connect_cbs):
            try:
                cb(self)
            except Exception as e:
                print(f"[MQTT] on_connect callback error: {e}")

    def _on_disconnect(self, client, userdata, rc):  # type: ignore[override]
        self.connected = False
        print(f"[MQTT] disconnected (rc={rc})")
