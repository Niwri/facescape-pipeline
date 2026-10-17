"""Register all API routes."""
from __future__ import annotations

from fastapi import APIRouter

from .routes import datasets, jobs, landmarks, sequences

api_router = APIRouter()
api_router.include_router(datasets.router)
api_router.include_router(landmarks.router)
api_router.include_router(jobs.router)
api_router.include_router(sequences.router)
