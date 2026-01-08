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
        # De-spam: avoid publishing identical retained payloads repeatedly.
        # Keyed by (full_topic, retain, qos) to preserve semantics.
        self._last_published: dict[tuple[str, bool, int], bytes] = {}
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
        # Important: do not hold our lock while calling into paho; disconnect/loop_stop can
        # trigger callbacks on the network thread that also want this lock (deadlock on exit).
        with self._lock:
            client = self.client
            was_connected = bool(self.connected)
            # Prevent further publishes immediately.
            self.client = None
            self.connected = False
            self._last_published.clear()

        if client is None:
            return

        # Detach callbacks to avoid reentrancy into this object while stopping.
        try:
            client.on_connect = None
            client.on_disconnect = None
        except Exception:
            pass

        # Best-effort offline LWT publish (retained).
        if was_connected:
            try:
                client.publish(self._avail_topic, payload="offline", retain=True)
            except Exception:
                pass

        try:
            client.disconnect()
        except Exception:
            pass

        # Ensure the network loop thread can't keep the process alive on exit.
        try:
            try:
                client.loop_stop(force=True)
            except TypeError:
                client.loop_stop()
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # Publish helper
    # ------------------------------------------------------------------ #

    def _publish_full_locked(self, full_topic: str, payload: str | bytes, *, retain: bool, qos: int) -> None:
        if not self.client or not self.connected:
            return
        full = str(full_topic).strip().lstrip("/")
        if not full:
            return
        try:
            payload_bytes = (
                bytes(payload)
                if isinstance(payload, (bytes, bytearray))
                else str(payload).encode("utf-8")
            )
            key = (full, bool(retain), int(qos))
            if self._last_published.get(key) == payload_bytes:
                return
            self.client.publish(full, payload=payload, qos=int(qos), retain=retain)
            self._last_published[key] = payload_bytes
        except Exception as e:
            print(f"[MQTT] publish error on {full}: {e}")

    def publish(self, topic: str, payload: str | bytes, retain: bool = False, qos: int = 0) -> None:
        """Publish under the configured base topic (`mqtt_base_topic`)."""
        with self._lock:
            full = f"{self._base_topic}/{str(topic).lstrip('/')}"
            self._publish_full_locked(full, payload, retain=retain, qos=int(qos))

    def publish_absolute(self, full_topic: str, payload: str | bytes, retain: bool = False, qos: int = 0) -> None:
        """Publish to an absolute MQTT topic (no `mqtt_base_topic` prefix)."""
        with self._lock:
            self._publish_full_locked(str(full_topic), payload, retain=retain, qos=int(qos))

    def publish_discovery_config(self, topic: str, payload: str | bytes, retain: bool = True, qos: int = 0) -> None:
        """
        Publish a Home Assistant MQTT Discovery config payload.

        When `mqtt_discovery_under_base_topic` is enabled, publishes under `mqtt_base_topic`
        (legacy behaviour). Otherwise publishes to `mqtt_discovery_prefix` directly (HA default).
        """
        under_base = bool(getattr(self.cfg, "mqtt_discovery_under_base_topic", False))
        if under_base:
            self.publish(topic, payload, retain=retain, qos=int(qos))
        else:
            self.publish_absolute(topic, payload, retain=retain, qos=int(qos))

    def add_on_connect(self, cb: Callable[["MqttService"], None]) -> None:
        with self._lock:
            self._on_connect_cbs.append(cb)

    # ------------------------------------------------------------------ #
    # Callbacks
    # ------------------------------------------------------------------ #

    def _on_connect(self, client, userdata, flags, rc):  # type: ignore[override]
        self.connected = True
        # Allow first publish after reconnect to flow (broker may have restarted).
        with self._lock:
            self._last_published.clear()
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
        with self._lock:
            self._last_published.clear()
        print(f"[MQTT] disconnected (rc={rc})")
