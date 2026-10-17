"""Sequence management endpoints."""
from __future__ import annotations

from fastapi import APIRouter

from ...schemas.sequences import PreprocessRequest, PreprocessResponse
from ...services.preprocess_service import enqueue_preprocess

router = APIRouter(prefix="/sequences", tags=["sequences"])


@router.post("/{actor_id}/{sequence_id}/preprocess", response_model=PreprocessResponse)
def preprocess_sequence(actor_id: str, sequence_id: str, request: PreprocessRequest | None = None) -> PreprocessResponse:
    request = request or PreprocessRequest()
    job_id = enqueue_preprocess(actor_id, sequence_id, request.cameras)
    return PreprocessResponse(job_id=job_id, status="accepted")
