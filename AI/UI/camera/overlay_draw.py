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
        font_px = int(getattr(self, "_overlay_text_px", 0) or 0)
        if font_px > 0:
            font.setPixelSize(font_px)
        else:
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
        font_px = int(getattr(self, "_overlay_text_px", 0) or 0)
        if font_px > 0:
            font.setPixelSize(font_px)
        else:
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
        font_px = int(getattr(self, "_overlay_text_px", 0) or 0)
        if font_px > 0:
            font.setPixelSize(font_px)
        else:
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


def _draw_ptz_controls(self, p: QtGui.QPainter, w: int, h: int) -> None:
    """
    Draw a small PTZ control overlay (top-right) for PTZ-capable ONVIF cameras.

    Also stores hit-test rectangles in `self._ptz_hit_regions` for click handling.
    """
    if not bool(getattr(self, "_ptz_available", False)):
        try:
            self._ptz_hit_regions = {}
        except Exception:
            pass
        return

    p.save()
    try:
        scale = float(getattr(self, "_overlay_scale", 1.0) or 1.0)
        margin = int(10 * scale)
        gap = int(6 * scale)
        pad = int(6 * scale)

        dpad_size = int(72 * scale)
        zoom_btn_h = int(18 * scale)
        zoom_gap = int(4 * scale)
        has_zoom = bool(getattr(self, "_ptz_has_zoom", True))

        y = margin
        # If the REC indicator is present (top-right), place PTZ below it.
        if bool(getattr(self, "_rec_indicator_on", False)):
            y += int(26 * scale) + margin

        total_h = dpad_size
        if has_zoom:
            total_h += gap + zoom_btn_h * 2 + zoom_gap

        x = w - margin - dpad_size
        if x < margin:
            x = margin

        # Container
        outer = QtCore.QRectF(x - pad, y - pad, dpad_size + 2 * pad, total_h + 2 * pad)
        p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 120), max(1, int(1 * scale))))
        p.setBrush(QtGui.QBrush(QtGui.QColor(0, 0, 0, 110)))
        p.drawRoundedRect(outer, 6 * scale, 6 * scale)

        # D-pad button rects (3x3 grid)
        dpad = QtCore.QRectF(x, y, dpad_size, dpad_size)
        cell = dpad_size / 3.0
        r_up = QtCore.QRectF(dpad.left() + cell, dpad.top(), cell, cell)
        r_down = QtCore.QRectF(dpad.left() + cell, dpad.top() + 2 * cell, cell, cell)
        r_left = QtCore.QRectF(dpad.left(), dpad.top() + cell, cell, cell)
        r_right = QtCore.QRectF(dpad.left() + 2 * cell, dpad.top() + cell, cell, cell)

        # Zoom buttons
        regions: dict[str, QtCore.QRectF] = {
            "up": r_up,
            "down": r_down,
            "left": r_left,
            "right": r_right,
        }
        if has_zoom:
            y_zoom = dpad.bottom() + gap
            r_zi = QtCore.QRectF(dpad.left(), y_zoom, dpad.width(), zoom_btn_h)
            r_zo = QtCore.QRectF(dpad.left(), y_zoom + zoom_btn_h + zoom_gap, dpad.width(), zoom_btn_h)
            regions["zoom_in"] = r_zi
            regions["zoom_out"] = r_zo

        # Persist for hit testing
        try:
            self._ptz_hit_regions = regions
        except Exception:
            pass

        # Draw buttons + icons
        p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 140), max(1, int(1 * scale))))
        p.setBrush(QtGui.QBrush(QtGui.QColor(255, 255, 255, 30)))
        for r in (r_up, r_down, r_left, r_right):
            p.drawRoundedRect(r.adjusted(1, 1, -1, -1), 4 * scale, 4 * scale)

        def tri(points: list[QtCore.QPointF]) -> None:
            poly = QtGui.QPolygonF(points)
            p.setBrush(QtGui.QBrush(QtGui.QColor(255, 255, 255, 210)))
            p.setPen(QtCore.Qt.PenStyle.NoPen)
            p.drawPolygon(poly)
            p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 140), max(1, int(1 * scale))))

        # Arrow triangles
        tri([
            QtCore.QPointF(r_up.center().x(), r_up.top() + pad * 0.6),
            QtCore.QPointF(r_up.left() + pad, r_up.bottom() - pad),
            QtCore.QPointF(r_up.right() - pad, r_up.bottom() - pad),
        ])
        tri([
            QtCore.QPointF(r_down.center().x(), r_down.bottom() - pad * 0.6),
            QtCore.QPointF(r_down.left() + pad, r_down.top() + pad),
            QtCore.QPointF(r_down.right() - pad, r_down.top() + pad),
        ])
        tri([
            QtCore.QPointF(r_left.left() + pad * 0.6, r_left.center().y()),
            QtCore.QPointF(r_left.right() - pad, r_left.top() + pad),
            QtCore.QPointF(r_left.right() - pad, r_left.bottom() - pad),
        ])
        tri([
            QtCore.QPointF(r_right.right() - pad * 0.6, r_right.center().y()),
            QtCore.QPointF(r_right.left() + pad, r_right.top() + pad),
            QtCore.QPointF(r_right.left() + pad, r_right.bottom() - pad),
        ])

        if has_zoom:
            # Zoom buttons: rounded boxes with + / -
            for r in (regions["zoom_in"], regions["zoom_out"]):
                p.setBrush(QtGui.QBrush(QtGui.QColor(255, 255, 255, 30)))
                p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 140), max(1, int(1 * scale))))
                p.drawRoundedRect(r.adjusted(1, 1, -1, -1), 4 * scale, 4 * scale)

            pen = QtGui.QPen(QtGui.QColor(255, 255, 255, 220), max(1, int(2 * scale)))
            p.setPen(pen)
            # Plus
            r = regions["zoom_in"]
            cx, cy = r.center().x(), r.center().y()
            arm = min(r.width(), r.height()) * 0.25
            p.drawLine(QtCore.QPointF(cx - arm, cy), QtCore.QPointF(cx + arm, cy))
            p.drawLine(QtCore.QPointF(cx, cy - arm), QtCore.QPointF(cx, cy + arm))
            # Minus
            r = regions["zoom_out"]
            cx, cy = r.center().x(), r.center().y()
            arm = min(r.width(), r.height()) * 0.25
            p.drawLine(QtCore.QPointF(cx - arm, cy), QtCore.QPointF(cx + arm, cy))
    finally:
        p.restore()


__all__ = [
    "_overlay_scale_factor",
    "_draw_hud",
    "_draw_stats_line",
    "_draw_rec_indicator",
    "_draw_ptz_controls",
]
