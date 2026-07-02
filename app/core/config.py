from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Face Recognition Attendance API"
    database_path: Path = Path("data/attendance.db")
    recognition_threshold: float = 0.62
    duplicate_window_minutes: int = 480
    image_size: int = 160
    device: str = "cpu"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="FACE_ATTENDANCE_",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
