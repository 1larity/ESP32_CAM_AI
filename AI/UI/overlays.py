# overlays.py
# Overlay renderer with toggles and render order: boxes → labels → crosshair + HUD.
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import math
from PySide6 import QtGui, QtCore
from detectors import DetectionPacket, DetBox
import time


@dataclass
class OverlayFlags:
    yolo: bool = True
    faces: bool = True
    pets: bool = True
    tracks: bool = True
    hud: bool = True  # camera name + timestamp HUD
    stats: bool = True  # FPS + detection counts (bottom-left)

def _pen(width: int, rgb: Tuple[int, int, int]) -> QtGui.QPen:
    pen = QtGui.QPen(QtGui.QColor(*rgb))
    pen.setWidth(width)
    return pen


def _brush(rgb: Tuple[int, int, int], alpha: int = 80) -> QtGui.QBrush:
    c = QtGui.QColor(*rgb)
    c.setAlpha(alpha)
    return QtGui.QBrush(c)


def _draw_box(p: QtGui.QPainter, xyxy, rgb, *, scale: float):
    x1, y1, x2, y2 = xyxy
    r = QtCore.QRectF(x1, y1, x2 - x1, y2 - y1)
    p.save()
    try:
        p.setPen(_pen(max(1, int(round(2 * scale))), rgb))
        p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        p.drawRect(r)
    finally:
        p.restore()


def _draw_label(
    p: QtGui.QPainter,
    xyxy,
    text: str,
    rgb,
    *,
    scale: float,
    font_px: int | None = None,
):
    x1, y1, x2, y2 = xyxy
    # IMPORTANT: Use a stable baseline font size per call; avoid compounding font scaling
    # when multiple labels are drawn in a single frame.
    p.save()
    try:
        font = p.font()
        if font_px is not None and int(font_px) > 0:
            font.setPixelSize(int(max(8, int(font_px))))
        else:
            base_pt = 10
            font.setPointSize(int(max(8, round(base_pt * scale))))
        p.setFont(font)

        fm = QtGui.QFontMetrics(p.font())
        tw = fm.horizontalAdvance(text) + int(6 * scale)
        th = fm.height() + int(4 * scale)
        r = QtCore.QRectF(x1, max(0, y1 - th), tw, th)

        p.setPen(QtCore.Qt.PenStyle.NoPen)
        p.setBrush(_brush(rgb, 200))
        p.drawRect(r)

        p.setPen(_pen(max(1, int(round(1 * scale))), (0, 0, 0)))
        p.drawText(
            r,
            QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter,
            text,
        )
    finally:
        p.restore()


def _yolo_color(cls: str) -> Tuple[int, int, int]:
    if cls == "person":
        return (0, 255, 0)
    if cls in ("dog", "cat"):
        return (0, 128, 255)
    return (255, 255, 0)


def _face_color() -> Tuple[int, int, int]:
    return (255, 0, 255)


def _pet_color(cls: str) -> Tuple[int, int, int]:
    if "dog" in cls.lower():
        return (255, 128, 0)
    if "cat" in cls.lower():
        return (128, 0, 255)
    return (255, 0, 0)


def _draw_yolo(
    p: QtGui.QPainter,
    boxes: list[DetBox],
    *,
    scale: float,
    font_px: int | None = None,
):
    for d in boxes:
        rgb = _yolo_color(d.cls)
        _draw_box(p, d.xyxy, rgb, scale=scale)
        text = f"{d.cls} {d.score:.2f}"
        _draw_label(p, d.xyxy, text, rgb, scale=scale, font_px=font_px)


def _draw_faces(
    p: QtGui.QPainter,
    boxes: list[DetBox],
    *,
    scale: float,
    show_box_size: bool = False,
    font_px: int | None = None,
):
    for d in boxes:
        rgb = _face_color()
        _draw_box(p, d.xyxy, rgb, scale=scale)
        # For faces, DetBox.cls is usually the label (name) if recognised.
        # Show score as a percentage, but guard against pipelines that provide
        # a non-normalized "confidence" (e.g., an LBPH distance-like value).
        name = d.cls or "face"
        try:
            s = float(getattr(d, "score", 0.0))
        except Exception:
            s = 0.0

        if not math.isfinite(s):
            pct = 0.0
        elif 0.0 <= s <= 1.0:
            pct = s * 100.0
        elif 1.0 < s <= 100.0:
            pct = s
        else:
            # Interpret large values as "lower is better" and compress into 0..100.
            pct = 100.0 / (1.0 + (max(0.0, s) / 100.0))

        pct = max(0.0, min(100.0, pct))
        text = f"{name} {pct:.0f}%"
        _draw_label(p, d.xyxy, text, rgb, scale=scale, font_px=font_px)

        if show_box_size:
            try:
                x1, y1, x2, y2 = d.xyxy
                w = max(0, int(x2) - int(x1))
                h = max(0, int(y2) - int(y1))
                _draw_label(
                    p,
                    (x1, y2, x2, y2),
                    f"{w}x{h}px",
                    rgb,
                    scale=scale,
                    font_px=font_px,
                )
            except Exception:
                pass


def _draw_pets(
    p: QtGui.QPainter,
    boxes: list[DetBox],
    *,
    scale: float,
    font_px: int | None = None,
):
    for d in boxes:
        rgb = _pet_color(d.cls)
        _draw_box(p, d.xyxy, rgb, scale=scale)
        text = f"{d.cls} {d.score:.2f}"
        _draw_label(p, d.xyxy, text, rgb, scale=scale, font_px=font_px)


def _draw_hud(p: QtGui.QPainter, pkt: DetectionPacket):
    """Small HUD with camera name and current date/time."""
    cam_name = pkt.name
    now_str = time.strftime("%Y-%m-%d %H:%M:%S")
    text = f"{cam_name}  {now_str}"

    fm = QtGui.QFontMetrics(p.font())
    tw = fm.horizontalAdvance(text) + 12
    th = fm.height() + 6
    r = QtCore.QRectF(8, 8, tw, th)

    # Background
    p.setPen(QtCore.Qt.PenStyle.NoPen)
    p.setBrush(_brush((0, 0, 0), 160))
    p.drawRoundedRect(r, 6, 6)

    # Text
    p.setPen(_pen(1, (255, 255, 255)))
    p.drawText(r, QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter, text)


def draw_overlays(
    p: QtGui.QPainter,
    pkt: DetectionPacket,
    flags: OverlayFlags,
    *,
    scale: float = 1.0,
    show_face_box_size: bool = False,
    text_px: int | None = None,
):
    """
    Draw all overlays for a given detection packet.

    Order: boxes → labels → crosshair + HUD.
    """
    if flags.yolo and pkt.yolo:
        _draw_yolo(p, pkt.yolo, scale=scale, font_px=text_px)

    if flags.faces and pkt.faces:
        _draw_faces(
            p,
            pkt.faces,
            scale=scale,
            show_box_size=show_face_box_size,
            font_px=text_px,
        )

    if flags.pets and pkt.pets:
        _draw_pets(p, pkt.pets, scale=scale, font_px=text_px)

    # Tracks not implemented yet; left as a hook.
    # if flags.tracks: ...

    if flags.hud:
        _draw_hud(p, pkt)
