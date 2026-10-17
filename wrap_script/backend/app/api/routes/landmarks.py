"""Landmark editing routes."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from ...schemas.landmarks import LandmarkBatchResponse, LandmarkPayload, LandmarkUpsertResponse, LandmarkChange
from ...services.landmark_store import landmark_store
from ...services.loop_service import enqueue_propagation
from typing import List

router = APIRouter(prefix="/landmarks", tags=["landmarks"])


@router.get("", response_model=LandmarkBatchResponse)
def get_landmarks(
    actor_id: str,
    sequence_id: str,
    camera_id: str,
    frame_indices: list[int] = Query(default_factory=list, description="List of frame indices to load"),
    load_all: bool = Query(False, description="Return every frame for this camera"),
) -> LandmarkBatchResponse:
    if load_all:
        frames = landmark_store.load_all_frames(actor_id, sequence_id, camera_id)
    elif frame_indices:
        frames = landmark_store.load_frames(actor_id, sequence_id, camera_id, frame_indices)
    else:
        raise HTTPException(status_code=400, detail="Provide frame_indices or set load_all=true")
    return LandmarkBatchResponse(frames=frames)


@router.post("", response_model=LandmarkUpsertResponse)
async def upsert_landmarks(request: LandmarkChange) -> LandmarkUpsertResponse:
    landmark_store.save_frame(request.payload)
    job_id = enqueue_propagation(request.payload, request.points)
    return LandmarkUpsertResponse(job_id=job_id, status="accepted")
