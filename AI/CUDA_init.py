from __future__ import annotations

from typing import Callable


def init_cuda(dlg: object, safe_update: Callable[[object, str], None]) -> None:
    try:
        safe_update(dlg, "Checking CUDA...")
        import cv2_dll_fix

        cv2_dll_fix.enable_opencv_cuda_dll_search()
        import cv2

        cuda_ok = False
        detail = ""
        if hasattr(cv2, "cuda"):
            try:
                cnt = cv2.cuda.getCudaEnabledDeviceCount()
                cuda_ok = bool(cnt and cnt > 0)
                detail = f"devices={cnt}"
            except Exception as e:
                detail = str(e)
        else:
            detail = "cv2.cuda missing"

        msg = "CUDA detected; GPU acceleration enabled"
        if not cuda_ok:
            msg = "CUDA not available; using CPU"
            if detail:
                msg += f" ({detail})"
        safe_update(dlg, msg)
    except Exception as e:
        safe_update(dlg, f"CUDA check failed; using CPU ({e})")
