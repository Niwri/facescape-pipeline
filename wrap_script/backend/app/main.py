"""FastAPI application entrypoint."""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .api.router import api_router
from .api.routes import ws
from .core.config import get_settings


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.project_name, openapi_url=f"{settings.api_v1_prefix}/openapi.json")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health", tags=["health"])
    def healthcheck() -> dict:
        return {"status": "ok"}

    app.mount(
        "/processed",
        StaticFiles(directory=settings.processed_root, check_dir=False),
        name="processed",
    )

    app.include_router(api_router, prefix=settings.api_v1_prefix)
    app.include_router(ws.router)
    return app


app = create_app()
