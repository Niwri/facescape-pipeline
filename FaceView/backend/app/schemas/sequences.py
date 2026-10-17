"""Schemas for sequence operations."""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel


class PreprocessRequest(BaseModel):
    cameras: Optional[List[str]] = None


class PreprocessResponse(BaseModel):
    job_id: str
    status: str
