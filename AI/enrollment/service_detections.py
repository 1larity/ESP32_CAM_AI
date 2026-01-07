# AI/enrollment/service_detections.py
# EnrollmentService detection hook (on_detections).

from __future__ import annotations

import time

import numpy as np

from .capture import capture_enrollment_sample


def on_detections(self, cam_name: str, bgr: np.ndarray, pkt) -> None:
    """
    Entry point from the video layer.

    Called from UI/CameraWidgetVideo for each `DetectionPacket`.

    cam_name:
      Name of the camera that produced this frame.
    bgr:
      BGR frame as a NumPy array.
    pkt:
      DetectionPacket with at least `.faces` containing DetBox objects.
    """
    if bgr is None or pkt is None:
        return

    now_ms = int(time.time() * 1000)

    # Collect unknowns even when not actively enrolling
    self._maybe_save_unknowns(cam_name, bgr, pkt, now_ms)

    if not self.active:
        return

    # Respect camera filter if set
    if self.target_cam is not None and cam_name != self.target_cam:
        return

    # Delegate the heavy lifting (face selection, debouncing, path creation)
    new_last_save_ms, last_gray, saved = capture_enrollment_sample(
        cam_name=cam_name,
        target_cam=self.target_cam,
        target_name=self.target_name,
        bgr=bgr,
        pkt=pkt,
        face_dir=self.face_dir,
        existing_count=self._existing_count,
        samples_got=self.samples_got,
        last_save_ms=self._last_save_ms,
        last_gray=self._last_gray,
        now_ms=now_ms,
    )

    # Update internal state from helper
    self._last_save_ms = new_last_save_ms
    self._last_gray = last_gray

    if not saved:
        # Nothing new persisted for this frame
        return

    # One more sample captured
    self.samples_got += 1
    done = self.samples_got >= self.samples_needed

    self._emit_status(done=done)

    if done:
        # Freeze further sampling and kick off training
        self.active = False
        self._train_now()


__all__ = ["on_detections"]

