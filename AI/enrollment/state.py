from __future__ import annotations

from typing import Any, Dict


def _emit_status(self, **kwargs: Any) -> None:
    """
    Helper to emit a status dict.

    Base payload:
      active, target_name, samples_needed, samples_got, existing_count, done, last_error
    Extra keys may be merged in via kwargs.
    """
    data: Dict[str, Any] = {
        "active": self.active,
        "target_name": self.target_name,
        "samples_needed": self.samples_needed,
        "samples_got": self.samples_got,
        "existing_count": self._existing_count,
        "done": self.samples_got >= self.samples_needed and self.samples_needed > 0,
        "last_error": None,
    }
    data.update(kwargs)
    self.status_changed.emit(data)


def set_unknown_capture(
    self, faces: bool, pets: bool, limit: int | None = None, auto_train: bool | None = None
) -> None:
    self.collect_unknown_faces = bool(faces)
    self.collect_unknown_pets = bool(pets)
    if limit is not None:
        self.unknown_capture_limit = max(1, int(limit))
    if auto_train is not None:
        self.auto_train_unknowns = bool(auto_train)
    if self.auto_train_unknowns:
        self._bootstrap_auto_unknowns()


__all__ = ["_emit_status", "set_unknown_capture"]

