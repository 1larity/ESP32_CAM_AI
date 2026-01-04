# overlays.py
# Overlay renderer with toggles and render order: boxes → labels → crosshair + HUD.
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
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


def _draw_label(p: QtGui.QPainter, xyxy, text: str, rgb, *, scale: float):
    x1, y1, x2, y2 = xyxy
    # IMPORTANT: Use a stable baseline font size per call; avoid compounding font scaling
    # when multiple labels are drawn in a single frame.
    p.save()
    try:
        font = p.font()
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


def _draw_yolo(p: QtGui.QPainter, boxes: list[DetBox], *, scale: float):
    for d in boxes:
        rgb = _yolo_color(d.cls)
        _draw_box(p, d.xyxy, rgb, scale=scale)
        text = f"{d.cls} {d.score:.2f}"
        _draw_label(p, d.xyxy, text, rgb, scale=scale)


def _draw_faces(p: QtGui.QPainter, boxes: list[DetBox], *, scale: float):
    for d in boxes:
        rgb = _face_color()
        _draw_box(p, d.xyxy, rgb, scale=scale)
        # For faces, DetBox.cls is usually the label (name) if recognised.
        # Show score as a percentage to match the accept confidence UI.
        name = d.cls or "face"
        text = f"{name} {d.score * 100:.0f}%"
        _draw_label(p, d.xyxy, text, rgb, scale=scale)


def _draw_pets(p: QtGui.QPainter, boxes: list[DetBox], *, scale: float):
    for d in boxes:
        rgb = _pet_color(d.cls)
        _draw_box(p, d.xyxy, rgb, scale=scale)
        text = f"{d.cls} {d.score:.2f}"
        _draw_label(p, d.xyxy, text, rgb, scale=scale)


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


def draw_overlays(p: QtGui.QPainter, pkt: DetectionPacket, flags: OverlayFlags, *, scale: float = 1.0):
    """
    Draw all overlays for a given detection packet.

    Order: boxes → labels → crosshair + HUD.
    """
    if flags.yolo and pkt.yolo:
        _draw_yolo(p, pkt.yolo, scale=scale)

    if flags.faces and pkt.faces:
        _draw_faces(p, pkt.faces, scale=scale)

    if flags.pets and pkt.pets:
        _draw_pets(p, pkt.pets, scale=scale)

    # Tracks not implemented yet; left as a hook.
    # if flags.tracks: ...

    if flags.hud:
        _draw_hud(p, pkt)
