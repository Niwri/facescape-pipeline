"""Dataset-facing schemas."""
from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel
from .common import MediaPath
from .labels import LabelStatus


class CameraDescriptor(BaseModel):
    camera_id: str
    video: MediaPath
    status: str = "unknown"


class SequenceDescriptor(BaseModel):
    sequence_id: str
    cameras: List[CameraDescriptor]
    has_landmarks: bool = False
    has_delegation_video: bool = False
    label_status: LabelStatus = "no_neutral_label"


class ActorDescriptor(BaseModel):
    actor_id: str
    sequences: List[SequenceDescriptor]


class DatasetResponse(BaseModel):
    actors: List[ActorDescriptor]
    face_indices: Optional[dict] = None
    default_cameras: List[str] = []
