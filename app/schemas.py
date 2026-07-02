from pydantic import BaseModel, Field


class UserOut(BaseModel):
    id: int
    external_id: str
    name: str
    created_at: str


class RecognitionMatch(BaseModel):
    user_id: int
    external_id: str
    name: str
    score: float = Field(..., description="Cosine similarity score")
    marked_attendance: bool = False
    attendance_log_id: int | None = None


class RecognitionResponse(BaseModel):
    matches: list[RecognitionMatch]
    unknown_faces: int


class AttendanceLog(BaseModel):
    id: int
    user_id: int
    external_id: str
    name: str
    recognized_at: str
    confidence: float
    source: str


class RegisterFaceResponse(BaseModel):
    user: UserOut
    embedding_id: int
    model_name: str
