from __future__ import annotations

from .training import train_from_disk


def rebuild_lbph_model_from_disk(self) -> bool:
    """
    Scan `self.face_dir` and train LBPH models into `self.models_dir`.

    This is used both after enrollment completes and when the user
    requests a manual "rebuild from disk" via the controller.
    """
    ok = train_from_disk(self.face_dir, self.models_dir)
    if not ok:
        self._emit_status(last_error="No faces found on disk for training.")
    else:
        self._emit_status()
    return ok


def _train_now(self) -> None:
    """
    Train LBPH models from all faces on disk and update status.
    """
    self.rebuild_lbph_model_from_disk()


__all__ = ["rebuild_lbph_model_from_disk", "_train_now"]

