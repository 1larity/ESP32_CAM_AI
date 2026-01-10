from __future__ import annotations

import json
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np

from .features import extract_pet_embedding


def _iter_images(folder: Path) -> list[Path]:
    files: list[Path] = []
    for pat in ("*.png", "*.PNG", "*.jpg", "*.JPG", "*.jpeg", "*.JPEG"):
        files.extend(sorted(folder.glob(pat)))
    return files


def _dataset_signature(root: Path, *, include_auto: bool) -> tuple[int, int, int]:
    total = 0
    max_mtime_ns = 0
    labels: list[str] = []
    if not root.exists():
        return 0, 0, 0
    for child in root.iterdir():
        if not child.is_dir():
            continue
        name = child.name
        if not include_auto and name.startswith("auto_pet_"):
            continue
        if name.startswith("."):
            continue
        count = 0
        for img in _iter_images(child):
            total += 1
            count += 1
            try:
                max_mtime_ns = max(max_mtime_ns, int(img.stat().st_mtime_ns))
            except Exception:
                pass
        if count:
            labels.append(f"{name}:{count}")
    labels_sorted = sorted(labels, key=str.casefold)
    labels_sig = zlib.crc32("\n".join(labels_sorted).encode("utf-8")) & 0xFFFFFFFF
    return total, max_mtime_ns, int(labels_sig)


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _atomic_save_npz(path: Path, **arrays) -> None:
    # NOTE: np.savez_compressed appends ".npz" when passed a filename that doesn't
    # already end with ".npz", so write to a file handle to keep the tmp name stable.
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as fp:
        np.savez_compressed(fp, **arrays)
    tmp.replace(path)


@dataclass(frozen=True)
class PetGallery:
    labels: tuple[str, ...]
    centroids: np.ndarray  # shape (N, D), float32, L2-normalized
    sig_total: int
    sig_mtime_ns: int
    sig_labels_crc: int

    def match(self, feat: np.ndarray) -> tuple[str, float, float]:
        if feat is None or self.centroids.size == 0:
            return "unknown", 0.0, 0.0
        try:
            v = feat.astype(np.float32, copy=False).reshape(1, -1)
            sims = (self.centroids @ v.T).reshape(-1)
            if sims.size == 0:
                return "unknown", 0.0, 0.0
            best_idx = int(np.argmax(sims))
            best = float(sims[best_idx])
            second = float(np.partition(sims, -2)[-2]) if sims.size >= 2 else 0.0
            return self.labels[best_idx], best, second
        except Exception:
            return "unknown", 0.0, 0.0


def load_or_build_gallery(
    *,
    models_dir: Path,
    pets_root: Path,
    include_auto: bool,
    max_samples_per_label: int,
) -> PetGallery:
    models_dir = Path(models_dir)
    pets_root = Path(pets_root)
    models_dir.mkdir(parents=True, exist_ok=True)

    cache_path = models_dir / "pet_id_cache.npz"
    labels_path = models_dir / "labels_pets.json"

    sig_total, sig_mtime_ns, sig_labels_crc = _dataset_signature(pets_root, include_auto=include_auto)

    # Fast-path: load cache if signature matches.
    if cache_path.exists():
        try:
            npz = np.load(cache_path, allow_pickle=False)
            c_total = int(npz["sig_total"])
            c_mtime = int(npz["sig_mtime_ns"])
            c_labels = int(npz["sig_labels_crc"]) if "sig_labels_crc" in npz else -1
            if c_total == sig_total and c_mtime == sig_mtime_ns and c_labels == sig_labels_crc:
                labels = tuple(str(x) for x in npz["labels"].tolist())
                cents = np.asarray(npz["centroids"], dtype=np.float32)
                return PetGallery(
                    labels=labels,
                    centroids=cents,
                    sig_total=c_total,
                    sig_mtime_ns=c_mtime,
                    sig_labels_crc=c_labels,
                )
        except Exception:
            pass

    # Build from disk.
    try:
        import cv2
    except Exception:
        return PetGallery(
            labels=tuple(),
            centroids=np.zeros((0, 0), np.float32),
            sig_total=sig_total,
            sig_mtime_ns=sig_mtime_ns,
            sig_labels_crc=sig_labels_crc,
        )

    labels: list[str] = []
    centroids: list[np.ndarray] = []

    if pets_root.exists():
        for label_dir in sorted([p for p in pets_root.iterdir() if p.is_dir()], key=lambda p: p.name.casefold()):
            label = label_dir.name.strip()
            if not label:
                continue
            if label.startswith("."):
                continue
            if not include_auto and label.startswith("auto_pet_"):
                continue

            files = _iter_images(label_dir)
            if not files:
                continue

            # Prefer newest samples if there are many.
            if max_samples_per_label and len(files) > int(max_samples_per_label):
                try:
                    files = sorted(files, key=lambda p: int(p.stat().st_mtime_ns), reverse=True)[: int(max_samples_per_label)]
                except Exception:
                    files = files[: int(max_samples_per_label)]

            feats: list[np.ndarray] = []
            for f in files:
                img = cv2.imread(str(f), cv2.IMREAD_COLOR)
                if img is None:
                    continue
                feat = extract_pet_embedding(img)
                if feat is None:
                    continue
                feats.append(feat)

            if not feats:
                continue

            mat = np.stack(feats, axis=0).astype(np.float32, copy=False)
            c = mat.mean(axis=0)
            n = float(np.linalg.norm(c) + 1e-6)
            c = (c / n).astype(np.float32, copy=False)
            labels.append(label)
            centroids.append(c)

    if labels and centroids:
        cents = np.stack(centroids, axis=0).astype(np.float32, copy=False)
    else:
        cents = np.zeros((0, 0), np.float32)
        labels = []

    # Persist cache + labels.
    try:
        _atomic_save_npz(
            cache_path,
            sig_total=np.asarray(sig_total, dtype=np.int64),
            sig_mtime_ns=np.asarray(sig_mtime_ns, dtype=np.int64),
            sig_labels_crc=np.asarray(sig_labels_crc, dtype=np.int64),
            labels=np.asarray(labels, dtype=str),
            centroids=cents.astype(np.float32, copy=False),
        )
    except Exception:
        pass

    try:
        label_map = {name: idx for idx, name in enumerate(labels)}
        _atomic_write_text(labels_path, json.dumps(label_map, indent=2))
    except Exception:
        pass

    return PetGallery(
        labels=tuple(labels),
        centroids=cents,
        sig_total=sig_total,
        sig_mtime_ns=sig_mtime_ns,
        sig_labels_crc=sig_labels_crc,
    )


def list_pet_labels(
    *,
    pets_root: Path,
    include_auto: bool = False,
) -> list[str]:
    pets_root = Path(pets_root)
    if not pets_root.exists():
        return []
    names: list[str] = []
    for d in sorted([p for p in pets_root.iterdir() if p.is_dir()], key=lambda p: p.name.casefold()):
        name = d.name.strip()
        if not name or name.startswith("."):
            continue
        if not include_auto and name.startswith("auto_pet_"):
            continue
        names.append(name)
    return names


__all__ = ["PetGallery", "list_pet_labels", "load_or_build_gallery"]
