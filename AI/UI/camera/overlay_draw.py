from __future__ import annotations

import time

from PySide6 import QtCore, QtGui

from ..overlay_stats import YoloStats


def _overlay_scale_factor(self, w: int, h: int) -> float:
    """Scale overlays relative to video resolution so text remains readable."""
    base = max(1, min(w, h))
    # Quantize to suppress tiny stream size jitters (common on some ONVIF RTSP feeds).
    scale = round(base / 480.0, 2)
    return max(1.0, min(3.0, scale))


def _draw_hud(self, p: QtGui.QPainter) -> None:
    p.save()
    try:
        text = f"{self.cam_cfg.name}  {time.strftime('%Y-%m-%d %H:%M:%S')}"
        scale = float(getattr(self, "_overlay_scale", 1.0) or 1.0)
        margin = int(6 * scale)

        p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255)))
        font = QtGui.QFont(p.font())
        base_pt = 12
        scaled_pt = int(round(9 * scale))
        font.setPointSize(int(max(base_pt, scaled_pt)))
        p.setFont(font)

        fm = QtGui.QFontMetrics(font)
        rect = fm.boundingRect(text)
        rect = QtCore.QRectF(
            margin, margin, rect.width() + 8 * scale, rect.height() + 4 * scale
        )

        bg = QtGui.QColor(0, 0, 0, 128)
        p.fillRect(rect, bg)

        p.drawText(
            rect.adjusted(4 * scale, 0, -4 * scale, 0),
            QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter,
            text,
        )
    finally:
        p.restore()


def _draw_stats_line(
    self, p: QtGui.QPainter, fps: float, stats: YoloStats, width: int, height: int
) -> None:
    p.save()
    try:
        scale = float(getattr(self, "_overlay_scale", 1.0) or 1.0)
        margin = int(6 * scale)
        text = (
            f"FPS: {fps:4.1f} | "
            f"faces: {stats.faces} ({stats.known_faces} known) | "
            f"pets: {stats.pets} | total: {stats.total}"
        )

        font = QtGui.QFont(p.font())
        base_pt = 12
        scaled_pt = int(round(9 * scale))
        font.setPointSize(int(max(base_pt, scaled_pt)))
        p.setFont(font)

        fm = QtGui.QFontMetrics(font)
        text_width = fm.horizontalAdvance(text)
        text_height = fm.height()

        x = margin
        y = height - margin
        rect = QtCore.QRectF(
            x - 4 * scale,
            y - text_height - 2 * scale,
            text_width + 8 * scale,
            text_height + 4 * scale,
        )

        bg = QtGui.QColor(0, 0, 0, 128)
        p.fillRect(rect, bg)
        p.setPen(QtGui.QColor(255, 255, 255))
        p.drawText(
            rect.adjusted(4 * scale, 0, -4 * scale, 0),
            QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter,
            text,
        )
    finally:
        p.restore()


def _draw_rec_indicator(self, p: QtGui.QPainter, w: int, h: int) -> None:
    p.save()
    try:
        scale = float(getattr(self, "_overlay_scale", 1.0) or 1.0)
        margin = int(10 * scale)
        box_w, box_h = int(76 * scale), int(26 * scale)
        rect = QtCore.QRectF(w - box_w - margin, margin, box_w, box_h)
        bg = QtGui.QColor(180, 0, 0, 180)
        p.fillRect(rect, bg)

        dot_r = int(6 * scale)
        dot_center = QtCore.QPointF(rect.left() + 12 * scale, rect.center().y())
        p.setBrush(QtGui.QBrush(QtGui.QColor(255, 80, 80)))
        p.setPen(QtCore.Qt.PenStyle.NoPen)
        p.drawEllipse(dot_center, dot_r, dot_r)

        p.setPen(QtGui.QPen(QtGui.QColor(255, 230, 230)))
        font = QtGui.QFont(p.font())
        base_pt = 12
        scaled_pt = int(round(9 * scale))
        font.setPointSize(int(max(base_pt, scaled_pt)))
        font.setBold(True)
        p.setFont(font)
        p.drawText(
            rect.adjusted(24 * scale, 0, -6 * scale, 0),
            QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter,
            "REC",
        )
    finally:
        p.restore()


__all__ = [
    "_overlay_scale_factor",
    "_draw_hud",
    "_draw_stats_line",
    "_draw_rec_indicator",
]

