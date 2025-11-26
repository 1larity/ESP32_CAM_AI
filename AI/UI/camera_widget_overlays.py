# Camera_widget_overlays.py
# AI / overlay toggle helpers.

from __future__ import annotations


def attach_overlay_handlers(cls) -> None:
    """Inject AI / overlay toggle helpers into CameraWidget."""

    def _on_ai_toggled(self, checked: bool) -> None:
        self._ai_enabled = bool(checked)

    def _on_ai_yolo_toggled(self, checked: bool) -> None:
        self._overlays.yolo = bool(checked)
        self._sync_overlay_master()

    def _on_ai_faces_toggled(self, checked: bool) -> None:
        self._overlays.faces = bool(checked)
        self._sync_overlay_master()

    def _on_ai_pets_toggled(self, checked: bool) -> None:
        self._overlays.pets = bool(checked)
        self._sync_overlay_master()

    def _sync_overlay_master(self) -> None:
        """Keep 'Detections (boxes + labels)' in sync with YOLO/Faces/Pets."""
        if self._overlay_master_updating:
            return
        any_on = (
            getattr(self._overlays, "yolo", False)
            or getattr(self._overlays, "faces", False)
            or getattr(self._overlays, "pets", False)
        )
        self._overlay_master_updating = True
        try:
            self.act_overlay_detections.setChecked(any_on)
        finally:
            self._overlay_master_updating = False

    def _on_overlay_master_toggled(self, checked: bool) -> None:
        """
        Master "Detections (boxes + labels)" switch.
        When off: force YOLO/Faces/Pets overlays off too.
        """
        if self._overlay_master_updating:
            return

        self._overlay_master_updating = True
        try:
            enabled = bool(checked)
            self._overlays.yolo = enabled
            self._overlays.faces = enabled
            self._overlays.pets = enabled

            # Keep AI menu items in sync
            self.act_ai_yolo.setChecked(enabled)
            self.act_ai_faces.setChecked(enabled)
            self.act_ai_pets.setChecked(enabled)
        finally:
            self._overlay_master_updating = False

    def _on_overlay_hud_toggled(self, checked: bool) -> None:
        """Toggle HUD (camera name + date/timestamp)."""
        self._overlays.hud = bool(checked)

    # Bind helpers
    cls._on_ai_toggled = _on_ai_toggled
    cls._on_ai_yolo_toggled = _on_ai_yolo_toggled
    cls._on_ai_faces_toggled = _on_ai_faces_toggled
    cls._on_ai_pets_toggled = _on_ai_pets_toggled
    cls._sync_overlay_master = _sync_overlay_master
    cls._on_overlay_master_toggled = _on_overlay_master_toggled
    cls._on_overlay_hud_toggled = _on_overlay_hud_toggled
