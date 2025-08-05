import numpy as np
import cv2

def render_overlay_canvas(base_size, labels, trails, font_scale=0.5):
    overlay = np.zeros((base_size[1], base_size[0], 4), dtype=np.uint8)
    for text, (x, y) in labels:
        cv2.putText(overlay, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (255, 255, 255, 255), 1, cv2.LINE_AA)
    for pts in trails:
        for i in range(1, len(pts)):
            cv2.line(overlay, pts[i - 1], pts[i], (255, 255, 0, 255), 1)
    return overlay
