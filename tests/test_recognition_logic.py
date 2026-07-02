from pathlib import Path

import numpy as np

from app.db.database import Database
from app.services.attendance_service import AttendanceService
from app.services.embedding_store import EmbeddingStore
from app.services.recognition_service import RecognitionService, cosine_similarity


class FakeEncoder:
    model_name = "fake-face-encoder"

    def __init__(self, embedding: np.ndarray):
        self.embedding = embedding.astype(np.float32)

    def encode(self, _image: np.ndarray) -> np.ndarray:
        return self.embedding

    def detect(self, image: np.ndarray) -> list:
        return [type("Face", (), {"crop": image})()]


def test_cosine_similarity_prefers_same_direction() -> None:
    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    same = np.array([0.9, 0.0, 0.0], dtype=np.float32)
    different = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    assert cosine_similarity(query, same) > cosine_similarity(query, different)


def test_register_and_recognize_known_face(tmp_path: Path) -> None:
    database = Database(tmp_path / "attendance.db")
    store = EmbeddingStore(database)
    attendance = AttendanceService(database, duplicate_window_minutes=480)
    service = RecognitionService(
        encoder=FakeEncoder(np.array([1.0, 0.0, 0.0])),
        store=store,
        attendance=attendance,
        threshold=0.8,
    )

    image = np.zeros((10, 10, 3), dtype=np.uint8)
    service.register_face("EMP001", "Asha Kumar", image)
    result = service.recognize(image, mark_attendance=True)

    assert result["unknown_faces"] == 0
    assert result["matches"][0]["external_id"] == "EMP001"
    assert result["matches"][0]["marked_attendance"] is True


def test_attendance_duplicate_window_returns_existing_log(tmp_path: Path) -> None:
    database = Database(tmp_path / "attendance.db")
    store = EmbeddingStore(database)
    user = store.upsert_user("EMP002", "Ravi Shah")
    attendance = AttendanceService(database, duplicate_window_minutes=480)

    first_marked, first_id = attendance.mark(user["id"], confidence=0.91)
    second_marked, second_id = attendance.mark(user["id"], confidence=0.93)

    assert first_marked is True
    assert second_marked is False
    assert second_id == first_id
    assert len(attendance.list_logs()) == 1
