from __future__ import annotations

from typing import Callable, Protocol


class _MqttServiceLike(Protocol):
    def add_on_connect(self, cb: Callable[[object], None]) -> None: ...

    def start(self) -> None: ...


def init_mqtt(
    app_cfg: object,
    mqtt: _MqttServiceLike,
    dlg: object,
    safe_update: Callable[[object, str], None],
) -> None:
    if getattr(app_cfg, "mqtt_enabled", False):
        host = getattr(app_cfg, "mqtt_host", None)
        port = getattr(app_cfg, "mqtt_port", 8883)
        if host:
            safe_update(dlg, f"MQTT: connecting to {host}:{port}...")
        else:
            safe_update(dlg, "MQTT: enabled but host not set; skipping connect.")
        try:
            mqtt.add_on_connect(lambda _svc: safe_update(dlg, "MQTT: connected"))
            mqtt.start()
        except Exception as e:
            safe_update(dlg, f"MQTT: failed to start ({e})")
    else:
        safe_update(dlg, "MQTT: disabled; skipping")
