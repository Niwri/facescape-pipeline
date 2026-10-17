"""Job inspection endpoints."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ...core.jobs import job_registry
from ...schemas.jobs import JobListResponse, JobResponse

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get("", response_model=JobListResponse)
def list_jobs() -> JobListResponse:
    jobs = [
        JobResponse(
            job_id=record.job_id,
            job_type=record.job_type,
            sequence_id=record.sequence_id,
            camera_id=record.camera_id,
            status=record.status.value,
            message=record.message,
        )
        for record in job_registry.list_jobs().values()
    ]
    return JobListResponse(jobs=jobs)


@router.get("/{job_id}", response_model=JobResponse)
def get_job(job_id: str) -> JobResponse:
    record = job_registry.get_job(job_id)
    if not record:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobResponse(
        job_id=record.job_id,
        job_type=record.job_type,
        sequence_id=record.sequence_id,
        camera_id=record.camera_id,
        status=record.status.value,
        message=record.message,
    )
