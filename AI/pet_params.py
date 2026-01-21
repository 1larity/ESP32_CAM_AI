from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os


@dataclass
class PetParams:
    """
    Pet identification tuning parameters persisted under models_dir/pet_recog.json.
    """

    enabled: bool = True
    sim_thresh: float = 0.78
    sim_margin: float = 0.06
    min_box_px: int = 72
    reid_interval_ms: int = 1200
    track_expire_ms: int = 2500
    iou_match_thresh: float = 0.25
    include_auto_labels: bool = False
    max_samples_per_label: int = 200

    @staticmethod
    def load(models_dir: str) -> "PetParams":
        p = os.path.join(models_dir, "pet_recog.json")
        if not os.path.exists(p):
            return PetParams()
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)
        return PetParams(
            **{
                k: d.get(k, getattr(PetParams, k))
                for k in PetParams.__annotations__.keys()
            }
        )

    def save(self, models_dir: str) -> None:
        os.makedirs(models_dir, exist_ok=True)
        p = os.path.join(models_dir, "pet_recog.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)

