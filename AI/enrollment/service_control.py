# AI/enrollment/service_control.py
# EnrollmentService session control helpers (start/stop).

from __future__ import annotations

from typing import Optional


def start(
    self,
    name: str,
    total_samples: int,
    target_cam: Optional[str] = None,
) -> None:
    """
    Begin an enrollment session.

    name:
      Person / label name. Must be non-empty.
    total_samples:
      Number of samples to capture in this session.
    target_cam:
      If not None, only frames from this camera name will be accepted.
    """
    # Stop any existing session
    self.stop()

    self.target_name = name.strip()
    if not self.target_name:
        self._emit_status(last_error="Name is empty")
        return

    self.samples_needed = max(1, int(total_samples))
    self.samples_got = 0
    self._last_save_ms = 0
    self._last_gray = None
    self.target_cam = target_cam

    # Count any existing images so filenames continue from there
    person_dir = self.face_dir / self.target_name
    person_dir.mkdir(parents=True, exist_ok=True)
    self._existing_count = len(list(person_dir.glob("*.png")))

    self.active = True
    self._emit_status()


def stop(self) -> None:
    """
    Stop enrollment without triggering training.

    This is used when the user cancels / aborts enrollment.
    """
    if self.active:
        self.active = False
        self._emit_status()


__all__ = ["start", "stop"]

