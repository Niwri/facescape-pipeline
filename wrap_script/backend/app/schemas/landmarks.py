"""Landmark-specific schemas."""
from __future__ import annotations

from typing import List
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

class LandmarkBatchResponse(BaseModel):
    frames: List[LandmarkFrame]
