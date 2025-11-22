# overlays.py
# Overlay renderer with toggles and render order: boxes → labels → crosshair + HUD.
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
from PyQt6 import QtGui, QtCore
from detectors import DetectionPacket, DetBox


@dataclass
class OverlayFlags:
    yolo: bool = True
    faces: bool = True
    pets: bool = True
    tracks: bool = True


def _pen(width: int, rgb: Tuple[int, int, int]) -> QtGui.QPen:
    p = QtGui.QPen(QtGui.QColor(*rgb))
    p.setWidth(width)
    return p


def _brush(rgb: Tuple[int, int, int], a: int) -> QtGui.QBrush:
    c = QtGui.QColor(*rgb)
    c.setAlpha(a)
    return QtGui.QBrush(c)


def draw_overlays(p: QtGui.QPainter, pkt: DetectionPacket, flags: OverlayFlags):
    # Debug HUD first so we know it's being called
    hud = f"yolo={len(pkt.yolo)} faces={len(pkt.faces)} pets={len(pkt.pets)}"
    p.setPen(_pen(2, (255, 255, 0)))
    p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
    p.drawText(10, 20, hud)

    # Debug print in console too
    if pkt.yolo or pkt.faces or pkt.pets:
        print(
            f"[OVERLAY:{pkt.name}] drawing: "
            f"yolo={len(pkt.yolo)} faces={len(pkt.faces)} pets={len(pkt.pets)}"
        )

    # Boxes
    if flags.yolo:
        for b in pkt.yolo:
            _draw_box(p, b, (0, 255, 0))
    if flags.faces:
        for b in pkt.faces:
            _draw_box(p, b, (0, 200, 255))
    if flags.pets:
        for b in pkt.pets:
            _draw_box(p, b, (255, 200, 0))

    # Labels
    if flags.yolo:
        for b in pkt.yolo:
            _draw_label(p, b.cls, b.score, b.xyxy, (0, 255, 0))
    if flags.faces:
        for b in pkt.faces:
            _draw_label(p, b.cls, b.score, b.xyxy, (0, 200, 255))
    if flags.pets:
        for b in pkt.pets:
            _draw_label(p, b.cls, b.score, b.xyxy, (255, 200, 0))

    # Crosshair at image center
    cx, cy = pkt.size[0] // 2, pkt.size[1] // 2
    p.setPen(_pen(2, (255, 0, 255)))
    p.drawLine(cx - 16, cy, cx + 16, cy)
    p.drawLine(cx, cy - 16, cx, cy + 16)


def _draw_box(p: QtGui.QPainter, b: DetBox, rgb):
    x1, y1, x2, y2 = b.xyxy
    rect = QtCore.QRectF(x1, y1, x2 - x1, y2 - y1)
    # Thicker pen and translucent fill to make it obvious
    p.setPen(_pen(3, rgb))
    p.setBrush(_brush(rgb, 60))
    p.drawRect(rect)


def _draw_label(p: QtGui.QPainter, name: str, score: float, xyxy, rgb):
    x1, y1, x2, y2 = xyxy
    text = f"{name} {score:.2f}"
    fm = QtGui.QFontMetrics(p.font())
    tw = fm.horizontalAdvance(text) + 6
    th = fm.height() + 4
    r = QtCore.QRectF(x1, max(0, y1 - th), tw, th)

    p.setPen(QtCore.Qt.PenStyle.NoPen)
    p.setBrush(_brush(rgb, 200))
    p.drawRect(r)

    p.setPen(_pen(1, (0, 0, 0)))
    p.drawText(r, QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter, text)
