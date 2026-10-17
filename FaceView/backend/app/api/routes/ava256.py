"""Ava-256 capture review endpoints: one keyline video per capture
(wrap_script_ava256/render_capture_line_video.py), per-frame Valid / Invalid /
Rerun labels in its CSV, and Done (export + archive). See
services/ava256_labels.py."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from ...schemas.ava256 import (
    Ava256DatasetResponse,
    Ava256DoneResponse,
    Ava256FrameRow,
    Ava256FramesResponse,
    Ava256LabelRangesRequest,
)
from ...services import ava256_labels
from ...services.ava256_labels import Ava256Error

router = APIRouter(prefix="/ava256", tags=["ava256"])


def _frames_response(capture_id: str, rows, warnings=None) -> Ava256FramesResponse:
    skin_errors = ava256_labels.read_skin_errors(capture_id)
    return Ava256FramesResponse(
        capture_id=capture_id,
        frames=[Ava256FrameRow(frame_id=f, status=s, skin_error_px=skin_errors.get(f)) for f, s in rows],
        is_done=ava256_labels.is_done(capture_id),
        warnings=warnings or [],
    )


@router.get("/captures", response_model=Ava256DatasetResponse)
def list_captures() -> Ava256DatasetResponse:
    return Ava256DatasetResponse(captures=ava256_labels.list_captures())


@router.get("/{capture_id}/video")
def get_video(capture_id: str) -> FileResponse:
    path = ava256_labels.video_path(capture_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"No line render video at {path}")
    return FileResponse(path, media_type="video/mp4")


@router.get("/{capture_id}/frames", response_model=Ava256FramesResponse)
def get_frames(capture_id: str) -> Ava256FramesResponse:
    return _frames_response(capture_id, ava256_labels.read_frames(capture_id))


@router.post("/{capture_id}/frames", response_model=Ava256FramesResponse)
def label_frames(capture_id: str, request: Ava256LabelRangesRequest) -> Ava256FramesResponse:
    try:
        rows, warnings = ava256_labels.apply_ranges(capture_id, request.ranges)
    except Ava256Error as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
    return _frames_response(capture_id, rows, warnings)


@router.post("/{capture_id}/done", response_model=Ava256DoneResponse)
def done(capture_id: str) -> Ava256DoneResponse:
    try:
        result = ava256_labels.finish_capture(capture_id)
    except Ava256Error as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
    return Ava256DoneResponse(capture_id=capture_id, **result)
