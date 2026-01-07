from __future__ import annotations

from pathlib import Path

import numpy as np

from .auto_unknowns import bootstrap_auto_unknowns, promote_unknown
from .unknown_capture import count_unknown, maybe_save_unknowns


def _maybe_save_unknowns(self, cam_name: str, bgr: np.ndarray, pkt, now_ms: int) -> None:
    maybe_save_unknowns(self, cam_name, bgr, pkt, now_ms)


def _count_unknown(self, path: Path) -> int:
    return count_unknown(path)


def _promote_unknown(self, roi, cam_name: str, is_pet: bool) -> None:
    promote_unknown(self, roi, cam_name, is_pet=is_pet, train_now=self._train_now)


def _bootstrap_auto_unknowns(self) -> None:
    bootstrap_auto_unknowns(self, train_now=self._train_now)


__all__ = [
    "_maybe_save_unknowns",
    "_count_unknown",
    "_promote_unknown",
    "_bootstrap_auto_unknowns",
]

