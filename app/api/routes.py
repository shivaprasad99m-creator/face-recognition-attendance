from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile

from app.api.dependencies import (
    get_attendance_service,
    get_embedding_store,
    get_recognition_service,
)
from app.schemas import (
    AttendanceLog,
    RecognitionResponse,
    RegisterFaceResponse,
    UserOut,
)
from app.services.attendance_service import AttendanceService
from app.services.embedding_store import EmbeddingStore
from app.services.recognition_service import RecognitionService
from app.utils.image_io import decode_image

router = APIRouter()


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


@router.post("/register_face", response_model=RegisterFaceResponse)
async def register_face(
    external_id: str = Form(...),
    name: str = Form(...),
    image: UploadFile = File(...),
    service: RecognitionService = Depends(get_recognition_service),
) -> dict:
    try:
        rgb_image = decode_image(await image.read())
        return service.register_face(external_id=external_id, name=name, rgb_image=rgb_image)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/recognize", response_model=RecognitionResponse)
async def recognize(
    image: UploadFile = File(...),
    mark_attendance: bool = Query(True),
    service: RecognitionService = Depends(get_recognition_service),
) -> dict:
    try:
        rgb_image = decode_image(await image.read())
        return service.recognize(rgb_image=rgb_image, mark_attendance=mark_attendance)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/attendance", response_model=list[AttendanceLog])
def attendance(
    limit: int = Query(100, ge=1, le=500),
    service: AttendanceService = Depends(get_attendance_service),
) -> list[dict]:
    return service.list_logs(limit=limit)


@router.get("/users", response_model=list[UserOut])
def users(store: EmbeddingStore = Depends(get_embedding_store)) -> list[dict]:
    return store.list_users()
