"""Job API schemas."""
from __future__ import annotations

from typing import Optional
from pydantic import BaseModel


class JobResponse(BaseModel):
    job_id: str
    job_type: str
    sequence_id: str
    camera_id: str
    status: str
    message: Optional[str] = None


class JobListResponse(BaseModel):
    jobs: list[JobResponse]
