# enrollment_service.py
# Progress signals, reloading helper, thin facade around EnrollmentService.

from pathlib import Path
import json
import time
from typing import Optional, Dict, Any

from PySide6 import QtCore
from PySide6.QtCore import Signal, Slot

from settings import BASE_DIR
from enrollment.service import EnrollmentService


class EnrollmentController(QtCore.QObject):
    """
    Thin controller for the enrollment UI.

    Exposes:
      - status_changed(dict): relay of EnrollmentService status
      - reload_models(): ask EnrollmentService to retrain LBPH
    """

    status_changed = Signal(dict)

    def __init__(self) -> None:
        super().__init__()
        self._svc = EnrollmentService.instance()
        self._svc.status_changed.connect(self._on_service_status)

    # ------------------------------------------------------------------ service hook

    @Slot(dict)
    def _on_service_status(self, data: Dict[str, Any]) -> None:
        self.status_changed.emit(data)

    # ------------------------------------------------------------------ public API

    def start(
        self,
        name: str,
        total_samples: int,
        target_cam: Optional[str] = None,
    ) -> None:
        self._svc.start(name=name, total_samples=total_samples, target_cam=target_cam)

    def stop(self) -> None:
        self._svc.stop()

    def rebuild_lbph_model_from_disk(self) -> bool:
        """
        Trigger a synchronous LBPH retrain from disk.
        UI code can run this in a worker thread.
        """
        return self._svc.rebuild_lbph_model_from_disk()
