"""Register all API routes."""
from __future__ import annotations

from fastapi import APIRouter

from .routes import ava256, datasets, facescape, jobs, labels, landmarks, sequences

api_router = APIRouter()
api_router.include_router(datasets.router)
api_router.include_router(labels.router)
api_router.include_router(landmarks.router)
api_router.include_router(jobs.router)
api_router.include_router(sequences.router)
api_router.include_router(facescape.router)
api_router.include_router(ava256.router)
