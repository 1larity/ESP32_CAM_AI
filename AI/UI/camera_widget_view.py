from __future__ import annotations

from PyQt6 import QtCore, QtWidgets, QtGui


def attach_view_handlers(cls) -> None:
    """
    Inject fit/zoom/lock/scrollbar helpers into CameraWidget.

    This file assumes the CameraWidget already defines:
      - self.view              : QGraphicsView
      - self._scene            : QGraphicsScene
      - self._pixmap_item      : QGraphicsPixmapItem
      - self.btn_*             : toolbar buttons
      - self.act_ai_*          : AI actions
      - self.act_overlay_*     : overlay actions
      - self._locked           : bool, initialised to False
      - self._subwindow        : QMdiSubWindow | None
      - self._locked_geometry  : QRect | None
    """

    def fit_window_to_video(self) -> None:
        """
        Resize the camera *subwindow* so that the visible video area matches
        the current frame size at the current zoom level.

        Also updates scrollbars so that when the window exactly fits the video,
        no scrollbars are shown.
        """
        pixmap = self._pixmap_item.pixmap()
        if pixmap.isNull():
            return

        scene_rect = self._scene.sceneRect()
        if scene_rect.isNull():
            return

        view = self.view

        # Map scene rect through current transform -> respects current zoom
        view_poly = view.mapFromScene(scene_rect)
        view_rect = view_poly.boundingRect()
        video_w = max(1, view_rect.width())
        video_h = max(1, view_rect.height())

        # --- size of camera widget (client) we need ---

        widget_size = self.size()
        view_size = view.size()

        # Overhead inside the camera widget (toolbars, margins, etc.)
        overhead_w = widget_size.width() - view_size.width()
        overhead_h = widget_size.height() - view_size.height()

        desired_widget_w = video_w + overhead_w
        desired_widget_h = video_h + overhead_h

        # --- locate our QMdiSubWindow container ---

        sub = getattr(self, "_subwindow", None)
        if sub is None:
            # Fallback: walk up parents until we find a QMdiSubWindow
            p = self.parentWidget()
            while p is not None and not isinstance(p, QtWidgets.QMdiSubWindow):
                p = p.parentWidget()
            if isinstance(p, QtWidgets.QMdiSubWindow):
                self._subwindow = sub = p

        # If we still don't have a subwindow, resize just this widget
        if not isinstance(sub, QtWidgets.QMdiSubWindow):
            self.resize(desired_widget_w, desired_widget_h)
            self._update_scrollbars()
            return

        # --- add subwindow frame overhead (title bar + frame) ---

        sub_size = sub.size()
        sub_client_size = widget_size  # our widget is the client area

        sub_over_w = sub_size.width() - sub_client_size.width()
        sub_over_h = sub_size.height() - sub_client_size.height()

        desired_sub_w = desired_widget_w + sub_over_w
        desired_sub_h = desired_widget_h + sub_over_h

        # Finally resize the QMdiSubWindow itself
        sub.resize(desired_sub_w, desired_sub_h)

        # Now that sizes are correct, adjust scrollbars
        self._update_scrollbars()

    def _on_lock_toggled(self, checked: bool) -> None:
        """
        When locked:
          - Disable all controls/menus within the subwindow.
          - Prevent the QMdiSubWindow from being moved by the user.
        """
        self._locked = bool(checked)

        # Locate the QMdiSubWindow container once we need it
        if getattr(self, "_subwindow", None) is None:
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
        locked = bool(getattr(self, "_locked", False))

        # These controls are disabled when locked
        for btn in (
            getattr(self, "btn_rec", None),
            getattr(self, "btn_snap", None),
            getattr(self, "btn_fit", None),
            getattr(self, "btn_100", None),
            getattr(self, "btn_fit_window", None),
            getattr(self, "btn_overlay_menu", None),
            getattr(self, "btn_ai_menu", None),
        ):
            if btn is not None:
                btn.setEnabled(not locked)

        # AI + overlay menu actions
        for act in (
            getattr(self, "act_ai_enabled", None),
            getattr(self, "act_ai_yolo", None),
            getattr(self, "act_ai_faces", None),
            getattr(self, "act_ai_pets", None),
            getattr(self, "act_overlay_detections", None),
            getattr(self, "act_overlay_hud", None),
        ):
            if act is not None:
                act.setEnabled(not locked)

    def _update_scrollbars(self) -> None:
        """
        Hide scrollbars when the view exactly fits the scaled video.
        Enable them only when required.
        """
        if not hasattr(self, "view"):
            return

        view = self.view
        pixmap = self._pixmap_item.pixmap()
        if pixmap.isNull():
            return

        scene_rect = self._scene.sceneRect()
        if scene_rect.isNull():
            return

        # Size actually needed to display the scaled pixmap
        mapped = view.mapFromScene(scene_rect).boundingRect()
        needed_w = mapped.width()
        needed_h = mapped.height()

        # Actual viewport size we have
        vp = view.viewport().size()
        vp_w = vp.width()
        vp_h = vp.height()

        # If video fits inside viewport, scrollbars OFF.
        # If video exceeds viewport, scrollbars ON.
        if needed_w <= vp_w:
            view.setHorizontalScrollBarPolicy(
                QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
            )
        else:
            view.setHorizontalScrollBarPolicy(
                QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded
            )

        if needed_h <= vp_h:
            view.setVerticalScrollBarPolicy(
                QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
            )
        else:
            view.setVerticalScrollBarPolicy(
                QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded
            )

    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:  # type: ignore[override]
        """
        Prevent the QMdiSubWindow from moving while locked.

        We restore the stored geometry whenever a Move/Resize event occurs
        on the subwindow.
        """
        sub = getattr(self, "_subwindow", None)
        if (
            obj is sub
            and bool(getattr(self, "_locked", False))
            and getattr(self, "_locked_geometry", None) is not None
            and self._locked_geometry.isValid()
        ):
            et = event.type()
            if et in (QtCore.QEvent.Type.Move, QtCore.QEvent.Type.Resize):
                # Restore geometry on the next turn of the event loop
                def _restore(g=self._locked_geometry, w=sub):
                    if w is not None:
                        w.setGeometry(g)

                QtCore.QTimer.singleShot(0, _restore)
                return True

        # Call base implementation with correct signature
        return QtWidgets.QWidget.eventFilter(self, obj, event)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # type: ignore[override]
        """
        Keep scrollbars in sync whenever the camera widget is resized.

        This is important so that:
          - After 'fit window to video', scrollbars are off.
          - When user later shrinks the window, scrollbars come back as needed.
        """
        QtWidgets.QWidget.resizeEvent(self, event)
        self._update_scrollbars()

    def fit_to_window(self) -> None:
        """Scale the view so the whole scene fits inside the view."""
        self.view.fitInView(
            self._scene.sceneRect(),
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
        )
        self.view._scale = 1.0
        self._update_scrollbars()

    def zoom_100(self) -> None:
        """Reset zoom to 100% and update scrollbars."""
        self.view.resetTransform()
        self.view._scale = 1.0
        self._update_scrollbars()

    # Bind helpers into the target class
    cls.fit_window_to_video = fit_window_to_video
    cls._on_lock_toggled = _on_lock_toggled
    cls._update_lock_state = _update_lock_state
    cls._update_scrollbars = _update_scrollbars
    cls.eventFilter = eventFilter
    cls.resizeEvent = resizeEvent
    cls.fit_to_window = fit_to_window
    cls.zoom_100 = zoom_100
