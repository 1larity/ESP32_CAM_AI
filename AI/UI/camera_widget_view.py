# camera_widget_view.py
# Fit / zoom / lock / eventFilter helpers.

from __future__ import annotations

from PyQt6 import QtCore, QtWidgets


def attach_view_handlers(cls) -> None:
    """Inject fit/zoom/lock helpers into CameraWidget."""

    def fit_window_to_video(self) -> None:
        """
        Resize the *subwindow* so that the video is shown at 100% and the
        client area matches the current frame size.
        """
        pixmap = self._pixmap_item.pixmap()
        if pixmap.isNull():
            return

        # Ensure view is at 100% zoom
        self.zoom_100()

        video_w = pixmap.width()
        video_h = pixmap.height()

        # Compute overhead between camera widget and view (toolbars, margins)
        widget_size = self.size()
        view_size = self.view.size()
        overhead_w = widget_size.width() - view_size.width()
        overhead_h = widget_size.height() - view_size.height()

        desired_widget_w = video_w + overhead_w
        desired_widget_h = video_h + overhead_h

        # Apply to top-level window (QMdiSubWindow) if possible
        win = self.window()
        if isinstance(win, QtWidgets.QWidget):
            win.resize(desired_widget_w, desired_widget_h)

    def _on_lock_toggled(self, checked: bool) -> None:
        """
        When locked:
          - Disable all controls/menus within the subwindow.
          - Prevent the QMdiSubWindow from being moved by the user.
        """
        self._locked = bool(checked)

        # Locate the QMdiSubWindow container once we need it
        if self._subwindow is None:
            w = self.window()
            if isinstance(w, QtWidgets.QMdiSubWindow):
                self._subwindow = w
                self._subwindow.installEventFilter(self)

        # Remember geometry when locking
        if self._subwindow is not None and self._locked:
            self._locked_geometry = self._subwindow.geometry()

        self._update_lock_state()

    def _update_lock_state(self) -> None:
        """Enable/disable all interactive controls according to lock state."""
        locked = self._locked

        # These controls are disabled when locked
        for w in (
            self.btn_rec,
            self.btn_snap,
            self.btn_fit,
            self.btn_100,
            self.btn_fit_window,
            self.btn_ai_menu,
            self.btn_overlay_menu,
        ):
            w.setEnabled(not locked)

        # Lock button itself must remain enabled
        self.btn_lock.setEnabled(True)

        # Menu actions disabled when locked
        for act in (
            self.act_ai_enabled,
            self.act_ai_yolo,
            self.act_ai_faces,
            self.act_ai_pets,
            self.act_overlay_detections,
            self.act_overlay_hud,
        ):
            act.setEnabled(not locked)

    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:  # type: ignore[override]
        """
        Prevent the QMdiSubWindow from moving while locked.

        We restore the stored geometry whenever a Move/Resize event occurs
        on the subwindow.
        """
        if (
            obj is self._subwindow
            and self._locked
            and self._locked_geometry is not None
            and self._locked_geometry.isValid()
        ):
            et = event.type()
            if et in (QtCore.QEvent.Type.Move, QtCore.QEvent.Type.Resize):
                # Restore geometry on the next turn of the event loop
                def _restore(g=self._locked_geometry, w=self._subwindow):
                    if w is not None:
                        w.setGeometry(g)

                QtCore.QTimer.singleShot(0, _restore)
                return True

        # IMPORTANT: cannot use super() here because this function is attached
        # after class creation; call QWidget's implementation explicitly.
        return QtWidgets.QWidget.eventFilter(self, obj, event)

    def fit_to_window(self) -> None:
        self.view.fitInView(
            self._scene.sceneRect(),
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
        )
        self.view._scale = 1.0

    def zoom_100(self) -> None:
        self.view.resetTransform()
        self.view._scale = 1.0

    # Bind helpers
    cls.fit_window_to_video = fit_window_to_video
    cls._on_lock_toggled = _on_lock_toggled
    cls._update_lock_state = _update_lock_state
    cls.eventFilter = eventFilter
    cls.fit_to_window = fit_to_window
    cls.zoom_100 = zoom_100
