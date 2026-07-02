from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DetectedFace:
    box: tuple[int, int, int, int]
    confidence: float
    crop: np.ndarray


class FaceEncoder:
    """MTCNN detection plus FaceNet embeddings via facenet-pytorch.

    The pretrained model is loaded lazily so API imports and unit tests do not
    require torch until a real recognition request is made.
    """

    model_name = "facenet-pytorch/inception-resnet-v1-vggface2"

    def __init__(self, image_size: int = 160, device: str = "cpu"):
        self.image_size = image_size
        self.device = device
        self._mtcnn = None
        self._model = None

    def _load(self) -> None:
        if self._mtcnn is not None and self._model is not None:
            return

        try:
            import torch
            from facenet_pytorch import InceptionResnetV1, MTCNN
        except ImportError as exc:
            raise RuntimeError(
                "Install the CV dependencies with `pip install -r requirements.txt` "
                "before registering or recognizing faces."
            ) from exc

        self._torch = torch
        self._mtcnn = MTCNN(
            image_size=self.image_size,
            margin=20,
            keep_all=True,
            post_process=True,
            device=self.device,
        )
        self._model = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)

    def detect(self, rgb_image: np.ndarray) -> list[DetectedFace]:
        self._load()
        boxes, probabilities = self._mtcnn.detect(rgb_image)
        if boxes is None or probabilities is None:
            return []

        faces = []
        height, width = rgb_image.shape[:2]
        for box, probability in zip(boxes, probabilities):
            if probability is None:
                continue
            x1, y1, x2, y2 = [int(round(value)) for value in box]
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(x2, width), min(y2, height)
            if x2 <= x1 or y2 <= y1:
                continue
            faces.append(
                DetectedFace(
                    box=(x1, y1, x2, y2),
                    confidence=float(probability),
                    crop=rgb_image[y1:y2, x1:x2],
                )
            )
        return faces

    def encode(self, rgb_image: np.ndarray) -> np.ndarray:
        self._load()
        aligned = self._mtcnn(rgb_image)
        if aligned is None:
            raise ValueError("No face detected in the image")

        if aligned.ndim == 4:
            aligned = aligned[0]

        with self._torch.no_grad():
            embedding = self._model(aligned.unsqueeze(0).to(self.device))
        vector = embedding.cpu().numpy()[0].astype(np.float32)
        norm = np.linalg.norm(vector)
        if norm == 0:
            raise ValueError("Model produced an empty embedding")
        return vector / norm
