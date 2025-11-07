# overlays.py
# Overlay renderer with toggles and render order: boxes → labels → crosshair.
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
    if flags.yolo:
        for b in pkt.yolo:
            _draw_box(p, b, (0, 255, 0))
    if flags.faces:
        for b in pkt.faces:
            _draw_box(p, b, (0, 200, 255))
    if flags.pets:
        for b in pkt.pets:
            _draw_box(p, b, (255, 200, 0))

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
    p.setPen(_pen(1, (255, 255, 255)))
    p.drawLine(cx - 12, cy, cx + 12, cy)
    p.drawLine(cx, cy - 12, cx, cy + 12)


def _draw_box(p: QtGui.QPainter, b: DetBox, rgb):
    x1, y1, x2, y2 = b.xyxy
    rect = QtCore.QRectF(x1, y1, x2 - x1, y2 - y1)
    p.setPen(_pen(2, rgb))
    p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
    p.drawRect(rect)


def _draw_label(p: QtGui.QPainter, name: str, score: float, xyxy, rgb):
    x1, y1, x2, y2 = xyxy
    text = f"{name} {score:.2f}"
    fm = QtGui.QFontMetrics(p.font())
    tw = fm.horizontalAdvance(text) + 6
    th = fm.height() + 4
    r = QtCore.QRectF(x1, y1 - th, tw, th)
    p.setPen(QtCore.Qt.PenStyle.NoPen)
    p.setBrush(_brush(rgb, 160))
    p.drawRect(r)
    p.setPen(_pen(1, (0, 0, 0)))
    p.drawText(r.adjusted(3, 0, -3, 0), QtCore.Qt.AlignmentFlag.AlignVCenter, text)
