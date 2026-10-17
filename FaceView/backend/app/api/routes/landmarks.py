"""Landmark editing routes."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from ...schemas.landmarks import (
    LandmarkBatchResponse,
    LandmarkBatchChange,
    LandmarkPayload,
    LandmarkUpsertResponse,
    NeutralFramePayload,
    NeutralFrameResponse,
    NeutralFrameQueryResponse,
)
from ...services.landmark_store import landmark_store
from ...services.loop_service import enqueue_save
from ...core.config import get_settings

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

    available = landmark_store.list_available_frame_indices(actor_id, sequence_id, camera_id)
    return LandmarkBatchResponse(
        frames=frames,
        frame_indices=available,
        frame_stride=get_settings().landmark_frame_stride,
    )


@router.get("/skin", response_model=LandmarkBatchResponse)
def get_skin_landmarks(actor_id: str, sequence_id: str, camera_id: str) -> LandmarkBatchResponse:
    """The optional "skin" landmarks (index >= 74) run_neutral_cotracker_propagation.py
    writes alongside the standard ones. Always bulk-loads every frame for this camera —
    there's no per-frame variant, since this data is small and the frontend caches it
    the same way it does the standard landmarks. Returns an empty frame list (not an
    error) for any actor/sequence that hasn't had skin points propagated yet."""
    frames = landmark_store.load_all_skin_frames(actor_id, sequence_id, camera_id)
    return LandmarkBatchResponse(frames=frames, frame_indices=[frame.frame_idx for frame in frames], frame_stride=get_settings().landmark_frame_stride)


@router.post("", response_model=LandmarkUpsertResponse)
async def upsert_landmarks(request: LandmarkBatchChange) -> LandmarkUpsertResponse:
    if not request.payloads:
        raise HTTPException(status_code=400, detail="payloads must not be empty")

    first = request.payloads[0]
    for payload in request.payloads:
        if payload.actor_id != first.actor_id or payload.sequence_id != first.sequence_id or payload.frame_idx != first.frame_idx:
            raise HTTPException(status_code=400, detail="All payloads must share actor_id, sequence_id, and frame_idx")

    for payload in request.payloads:
        landmark_store.save_frame(payload)

    job_id = enqueue_save(
        request.payloads,
        request.points,
        window=request.window,
        changed_by_camera=request.changed_by_camera,
    )
    return LandmarkUpsertResponse(job_id=job_id, status="accepted")


@router.post("/neutral_frame", response_model=NeutralFrameResponse)
def set_neutral_frame(request: NeutralFramePayload) -> NeutralFrameResponse:
    try:
        raw_frame_idx = landmark_store.save_neutral_frame(request.actor_id, request.sequence_id, request.frame_idx)
    except OSError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return NeutralFrameResponse(frame_idx=raw_frame_idx, status="confirmed")


@router.get("/neutral_frame", response_model=NeutralFrameQueryResponse)
def get_neutral_frame(actor_id: str, sequence_id: str) -> NeutralFrameQueryResponse:
    """`sequence_id` is accepted for URL-shape compatibility but no longer
    affects the result -- the neutral frame is actor-scoped now (one per
    actor, not one per actor/sequence pair), so the same (sequence_id,
    frame_idx) is returned regardless of which sequence the caller is
    currently viewing."""
    neutral = landmark_store.load_actor_neutral_frame(actor_id)
    if neutral is None:
        return NeutralFrameQueryResponse(sequence_id=None, frame_idx=None)
    neutral_sequence_id, frame_idx = neutral
    return NeutralFrameQueryResponse(sequence_id=neutral_sequence_id, frame_idx=frame_idx)
