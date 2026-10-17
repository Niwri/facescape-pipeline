"""In-memory job registry used until a real worker queue is wired up."""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class JobRecord:
    job_id: str
    job_type: str
    sequence_id: str
    camera_id: str
    created_at: float = field(default_factory=time.time)
    status: JobStatus = JobStatus.PENDING
    message: Optional[str] = None
    payload: Optional[dict] = None


class JobRegistry:
    """Thread-safe registry for lightweight background tasks."""

    def __init__(self) -> None:
        self._jobs: Dict[str, JobRecord] = {}
        self._lock = threading.Lock()

    def create_job(self, job_type: str, sequence_id: str, camera_id: str, payload: Optional[dict] = None) -> JobRecord:
        job = JobRecord(job_id=str(uuid.uuid4()), job_type=job_type, sequence_id=sequence_id, camera_id=camera_id, payload=payload)
        with self._lock:
            self._jobs[job.job_id] = job
        return job

    def update_job(self, job_id: str, status: JobStatus, message: Optional[str] = None) -> None:
        with self._lock:
            record = self._jobs.get(job_id)
            if not record:
                return
            record.status = status
            record.message = message

    def get_job(self, job_id: str) -> Optional[JobRecord]:
        with self._lock:
            return self._jobs.get(job_id)

    def list_jobs(self) -> Dict[str, JobRecord]:
        with self._lock:
            return dict(self._jobs)


job_registry = JobRegistry()
