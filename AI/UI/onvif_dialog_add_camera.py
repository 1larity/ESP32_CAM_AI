from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from onvif.enrichment import enrich_onvif_device
from onvif.rtsp import guess_fallback_urls, inject_auth
from settings import CameraSettings
from UI.onvif_dialog_format import format_onvif_label
from UI.onvif_dialog_workers import build_discovery_result, prompt_for_credentials


def build_camera_settings_from_selection(
    parent: QtWidgets.QWidget,
    item: QtWidgets.QListWidgetItem,
    info: dict,
) -> CameraSettings | None:
    stream = info.get("stream_uri")
    user = info.get("user")
    pwd = info.get("password")
    variants: list[str] = []

    def _add_variant(u: str | None) -> None:
        if u and u not in variants:
            variants.append(u)

    _add_variant(stream)
    for u in info.get("fallback_urls") or []:
        _add_variant(u)
    guesses = guess_fallback_urls(info.get("ip") or "")
    _add_variant(guesses[0] if guesses else None)

    if (not stream) or info.get("auth_required"):
        creds = prompt_for_credentials(parent)
        if creds is None:
            return None
        user, pwd = creds
        try:
            enriched = enrich_onvif_device(build_discovery_result(info), user or None, pwd)
            info.update(enriched)
            stream = info.get("stream_uri") or stream
            if info.get("stream_uri"):
                _add_variant(info.get("stream_uri"))
            for u in info.get("fallback_urls") or []:
                _add_variant(u)
            item.setText(format_onvif_label(info))
        except Exception as e:
            QtWidgets.QMessageBox.warning(
                parent, "Add Camera", f"Failed to fetch stream URI: {e}"
            )
            return None

    if not stream:
        # fall back to common RTSP paths
        candidates = info.get("fallback_urls") or guess_fallback_urls(info.get("ip") or "")
        if candidates:
            stream = candidates[0]
    # Embed credentials into URL if provided and not already present
    stream = inject_auth(stream, user, pwd)
    alt_streams = [u for u in variants if u and u != stream]
    name = info.get("name") or info.get("model") or info.get("ip") or "ONVIF-Camera"
    return CameraSettings(
        name=name,
        stream_url=stream,
        alt_streams=alt_streams,
        user=user or None,
        password=pwd or None,
    )


__all__ = ["build_camera_settings_from_selection"]

