from __future__ import annotations

from pathlib import Path

from detection.lbph import train_lbph_models


def train_from_disk(face_dir: Path, models_dir: Path) -> bool:
    return train_lbph_models(str(face_dir), str(models_dir))

