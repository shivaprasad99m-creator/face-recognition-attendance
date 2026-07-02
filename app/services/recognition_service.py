from __future__ import annotations

import numpy as np

from app.services.attendance_service import AttendanceService
from app.services.embedding_store import EmbeddingStore
from app.services.face_encoder import FaceEncoder


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = np.linalg.norm(left)
    right_norm = np.linalg.norm(right)
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return float(np.dot(left, right) / (left_norm * right_norm))


class RecognitionService:
    def __init__(
        self,
        encoder: FaceEncoder,
        store: EmbeddingStore,
        attendance: AttendanceService,
        threshold: float,
    ):
        self.encoder = encoder
        self.store = store
        self.attendance = attendance
        self.threshold = threshold

    def register_face(self, external_id: str, name: str, rgb_image: np.ndarray) -> dict:
        embedding = self.encoder.encode(rgb_image)
        user = self.store.upsert_user(external_id=external_id, name=name)
        embedding_id = self.store.add_embedding(
            user_id=user["id"],
            embedding=embedding,
            model_name=self.encoder.model_name,
        )
        return {
            "user": user,
            "embedding_id": embedding_id,
            "model_name": self.encoder.model_name,
        }

    def recognize(self, rgb_image: np.ndarray, mark_attendance: bool = True) -> dict:
        faces = self.encoder.detect(rgb_image)
        known_embeddings = self.store.load_embeddings()
        matches = []
        unknown_faces = 0

        for face in faces:
            query_embedding = self.encoder.encode(face.crop)
            best = self._best_match(query_embedding, known_embeddings)
            if best is None or best["score"] < self.threshold:
                unknown_faces += 1
                continue

            attendance_log_id = None
            marked = False
            if mark_attendance:
                marked, attendance_log_id = self.attendance.mark(
                    user_id=best["user_id"],
                    confidence=best["score"],
                    source="api",
                )

            matches.append(
                {
                    "user_id": best["user_id"],
                    "external_id": best["external_id"],
                    "name": best["name"],
                    "score": best["score"],
                    "marked_attendance": marked,
                    "attendance_log_id": attendance_log_id,
                }
            )

        return {"matches": matches, "unknown_faces": unknown_faces}

    def _best_match(self, query: np.ndarray, known_embeddings: list[dict]) -> dict | None:
        best: dict | None = None
        for candidate in known_embeddings:
            score = cosine_similarity(query, candidate["embedding"])
            if best is None or score > best["score"]:
                best = {
                    "user_id": candidate["user_id"],
                    "external_id": candidate["external_id"],
                    "name": candidate["name"],
                    "score": score,
                }
        return best
