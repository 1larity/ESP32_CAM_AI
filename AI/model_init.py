from __future__ import annotations

from typing import Callable


def init_models(
    app_cfg: object,
    dlg: object,
    safe_update: Callable[[object, str], None],
) -> None:
    def _status_cb(msg: str) -> None:
        safe_update(dlg, msg)

    try:
        from models import ModelManager

        ModelManager.ensure_models(app_cfg, status_cb=_status_cb)
    except Exception as e:
        safe_update(dlg, f"Model check failed: {e}")
