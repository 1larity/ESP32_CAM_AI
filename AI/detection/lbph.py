from __future__ import annotations

# Facade module: keep import paths stable while logic lives in smaller modules.

from .lbph_loader import load_lbph
from .lbph_inference import LBPH_DEFAULT_THRESHOLD, run_faces, run_faces_dnn
from .lbph_training import train_lbph_models

__all__ = [
    "LBPH_DEFAULT_THRESHOLD",
    "load_lbph",
    "run_faces",
    "run_faces_dnn",
    "train_lbph_models",
]

