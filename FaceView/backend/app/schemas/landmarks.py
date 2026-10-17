"""Landmark-specific schemas."""
from __future__ import annotations

from typing import Dict, List, Optional
from pydantic import BaseModel, Field

from .common import LandmarkFrame, LandmarkPoint


class LandmarkPayload(BaseModel):
    frame_idx: int = Field(ge=0)
    camera_id: str
    sequence_id: str
    actor_id: str
    points: List[LandmarkPoint]


class LandmarkUpsertResponse(BaseModel):
    job_id: str
    status: str

class LandmarkChange(BaseModel):
    payload: LandmarkPayload
    points: List[int]


class LandmarkBatchChange(BaseModel):
    payloads: List[LandmarkPayload]
    points: List[int]
    # camera_id -> indices the user actually moved in that camera. Only these
    # cameras feed the triangulation for a given index. Empty falls back to
    # using every camera (legacy behavior).
    changed_by_camera: Dict[str, List[int]] = Field(default_factory=dict)
    # CoTracker chunk size (frames processed per call before re-centering/rebuilding).
    # Propagation always runs to the last frame of the video regardless of this value;
    # None uses the server default (Settings.propagate_window).
    window: Optional[int] = None

class LandmarkBatchResponse(BaseModel):
    frames: List[LandmarkFrame]
    frame_indices: List[int] = Field(
        default_factory=list,
        description="Sorted list of every frame index that actually has landmark data on "
        "disk for this camera, derived from file names — the frontend should use this "
        "(not frame_stride math) to decide which frames are navigable.",
    )
    frame_stride: int = Field(
        default=1,
        description="Settings.landmark_frame_stride — a UI hint for expected spacing, not authoritative.",
    )


class NeutralFramePayload(BaseModel):
    actor_id: str
    sequence_id: str
    frame_idx: int = Field(ge=0, description="Displayed (delegation-video) frame index")


class NeutralFrameResponse(BaseModel):
    frame_idx: int = Field(description="Raw frame index (displayed index * RAW_FRAME_STRIDE) saved to neutral_frame.json")
    status: str = Field(default="confirmed", description="\"confirmed\" once neutral_frame.json has been written and read back successfully")


class NeutralFrameQueryResponse(BaseModel):
    sequence_id: str | None = Field(default=None, description="Which sequence this actor's neutral frame was picked from, or null if none has been saved yet")
    frame_idx: int | None = Field(default=None, description="Displayed (delegation-video) frame index of the actor's neutral frame (within sequence_id), or null if none has been saved yet")
