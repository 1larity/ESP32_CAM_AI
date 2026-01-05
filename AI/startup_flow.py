from __future__ import annotations

from typing import Callable


class StartupFlow:
    def __init__(
        self,
        app_cfg: object,
        mqtt: object,
        pump_events: Callable[[int], None],
        init_mqtt: Callable[[object, object, object, Callable[[object, str], None]], None],
        init_cuda: Callable[[object, Callable[[object, str], None]], None],
        init_models: Callable[[object, object, Callable[[object, str], None]], None],
    ) -> None:
        self._app_cfg = app_cfg
        self._mqtt = mqtt
        self._pump_events = pump_events
        self._init_mqtt = init_mqtt
        self._init_cuda = init_cuda
        self._init_models = init_models
        self._win = None

    def ensure_main_window(self):
        if self._win is None:
            from UI.main_window import MainWindow  # deferred to avoid heavy imports before loader

            self._win = MainWindow(
                self._app_cfg, load_on_init=False, mqtt_service=self._mqtt
            )
        return self._win

    def safe_update(self, dlg: object, text: str) -> None:
        try:
            if getattr(dlg, "isVisible")():
                getattr(dlg, "update_status")(text)
        except Exception:
            pass

    def preflight(self, dlg: object) -> None:
        self.safe_update(dlg, "Starting...")
        self._pump_events(2000)  # brief pause after showing the loader

        self._init_mqtt(self._app_cfg, self._mqtt, dlg, self.safe_update)

        # CUDA + model check
        self._init_cuda(dlg, self.safe_update)
        self._init_models(self._app_cfg, dlg, self.safe_update)

    def load_camera(self, cam_obj: object) -> None:
        win = self.ensure_main_window()
        win._add_camera_window(cam_obj)
