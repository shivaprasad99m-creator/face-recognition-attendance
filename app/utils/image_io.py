from __future__ import annotations

import cv2
import numpy as np


def decode_image(image_bytes: bytes) -> np.ndarray:
    array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Uploaded file is not a valid image")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def draw_label(image_bgr: np.ndarray, box: tuple[int, int, int, int], label: str) -> None:
    x1, y1, x2, y2 = box
    cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (36, 130, 50), 2)
    cv2.putText(
        image_bgr,
        label,
        (x1, max(y1 - 8, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (36, 130, 50),
        2,
        cv2.LINE_AA,
    )
