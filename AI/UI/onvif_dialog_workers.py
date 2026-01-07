from __future__ import annotations

import threading
from typing import Callable, Optional

from PySide6 import QtWidgets

from onvif import discover_onvif, OnvifDiscoveryResult
from onvif.enrichment import enrich_onvif_device


def prompt_for_credentials(
    parent: QtWidgets.QWidget, *, title: str = "Camera credentials"
) -> Optional[tuple[str, str]]:
    user, ok = QtWidgets.QInputDialog.getText(
        parent,
        title,
        "Username:",
        QtWidgets.QLineEdit.EchoMode.Normal,
    )
    if not ok:
        return None
    pwd, ok = QtWidgets.QInputDialog.getText(
        parent,
        title,
        "Password:",
        QtWidgets.QLineEdit.EchoMode.Password,
    )
    if not ok:
        return None
    return user.strip(), pwd


def build_discovery_result(info: dict) -> OnvifDiscoveryResult:
    return OnvifDiscoveryResult(
        xaddr=info.get("xaddr"),
        epr=None,
        scopes=[],
        ip=info.get("ip"),
    )


def enrich_onvif_info(info: dict, user: Optional[str], pwd: Optional[str]) -> dict:
    return enrich_onvif_device(build_discovery_result(info), user, pwd)


def run_scan_worker(
    stop_event: threading.Event,
    *,
    on_info: Callable[[dict], None],
    on_finished: Callable[[], None],
    timeout: float = 2.0,
    retries: int = 2,
) -> None:
    try:
        hits = discover_onvif(timeout=timeout, retries=retries, stop_event=stop_event)
        for res in hits:
            if stop_event.is_set():
                break
            info = enrich_onvif_device(res, None, None)
            on_info(info)
    finally:
        on_finished()


__all__ = [
    "prompt_for_credentials",
    "build_discovery_result",
    "enrich_onvif_info",
    "run_scan_worker",
]

