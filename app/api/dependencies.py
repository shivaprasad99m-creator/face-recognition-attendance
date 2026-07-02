from functools import lru_cache

from app.core.config import get_settings
from app.db.database import Database
from app.services.attendance_service import AttendanceService
from app.services.embedding_store import EmbeddingStore
from app.services.face_encoder import FaceEncoder
from app.services.recognition_service import RecognitionService


@lru_cache
def get_database() -> Database:
    return Database(get_settings().database_path)


@lru_cache
def get_recognition_service() -> RecognitionService:
    settings = get_settings()
    database = get_database()
    attendance = AttendanceService(
        database=database,
        duplicate_window_minutes=settings.duplicate_window_minutes,
    )
    return RecognitionService(
        encoder=FaceEncoder(image_size=settings.image_size, device=settings.device),
        store=EmbeddingStore(database),
        attendance=attendance,
        threshold=settings.recognition_threshold,
    )


@lru_cache
def get_attendance_service() -> AttendanceService:
    settings = get_settings()
    return AttendanceService(
        database=get_database(),
        duplicate_window_minutes=settings.duplicate_window_minutes,
    )


@lru_cache
def get_embedding_store() -> EmbeddingStore:
    return EmbeddingStore(get_database())
