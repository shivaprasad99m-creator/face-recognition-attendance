from fastapi import FastAPI

from app.api.routes import router
from app.core.config import get_settings


def create_app() -> FastAPI:
    settings = get_settings()
    application = FastAPI(
        title=settings.app_name,
        version="1.0.0",
        description="Real-time face recognition attendance system with FastAPI and SQLite.",
    )
    application.include_router(router)
    return application


app = create_app()
