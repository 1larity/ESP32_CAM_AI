# face_params.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Optional
import json, os

@dataclass
class FaceParams:
    accept_conf: float = 95.0      # LBPH accept if conf <= accept_conf
    roi_size: int = 128            # square ROI for LBPH
    eq_hist: bool = True           # apply cv.equalizeHist on ROI
    min_face_px: int = 48          # ignore faces smaller than this
    smooth_n: int = 3              # EMA for displayed score/conf
    presence_ttl_ms: int = 6000    # Grace period for presence enter/exit

    @staticmethod
    def load(models_dir: str) -> "FaceParams":
        p = os.path.join(models_dir, "face_recog.json")
        if not os.path.exists(p):
            return FaceParams()
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)
        return FaceParams(**{k: d.get(k, getattr(FaceParams, k)) for k in FaceParams.__annotations__.keys()})

    def save(self, models_dir: str) -> None:
        os.makedirs(models_dir, exist_ok=True)
        p = os.path.join(models_dir, "face_recog.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)
