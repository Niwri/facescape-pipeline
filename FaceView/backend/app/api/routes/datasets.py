"""Dataset endpoints."""
from __future__ import annotations

from fastapi import APIRouter

from ...schemas.datasets import DatasetResponse
from ...services.dataset_index import list_dataset

router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.get("", response_model=DatasetResponse)
def get_dataset() -> DatasetResponse:
    return list_dataset()
