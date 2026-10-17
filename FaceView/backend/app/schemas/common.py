"""Shared Pydantic models."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field


class FrameRange(BaseModel):
    start: int = Field(ge=0)
    end: int = Field(ge=0)

    def to_slice(self) -> slice:
        return slice(self.start, self.end + 1)


class LandmarkPoint(BaseModel):
    index: int
    x: float
    y: float
    z: Optional[float] = None
    visibility: Optional[float] = None


class LandmarkFrame(BaseModel):
    frame_idx: int = Field(ge=0)
    points: List[LandmarkPoint]


class MediaPath(BaseModel):
    actor_id: str
    sequence_id: str
    camera_id: str
    video_path: Path
    fps: Optional[float] = None
    num_frames: Optional[int] = None
