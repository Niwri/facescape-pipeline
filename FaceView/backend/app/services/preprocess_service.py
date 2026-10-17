"""Background orchestration for preprocessing sequences (ffmpeg + MediaPipe)."""
from __future__ import annotations

import logging
import subprocess
import threading
from pathlib import Path
from typing import Iterable, Sequence

from ..core.config import get_settings
from ..core.jobs import JobStatus, job_registry

log = logging.getLogger(__name__)

SCRIPT_PATH = Path(__file__).resolve().parents[3] / "scripts" / "preprocess" / "process_sequence.py"


def _build_command(actor_id: str, sequence_id: str, cameras: Sequence[str] | None = None) -> list[str]:
    settings = get_settings()
    cmd: list[str] = [str(settings.tracker_python_bin), str(SCRIPT_PATH), actor_id, sequence_id]
    if cameras:
        cmd += ["--cameras", *cameras]
    return cmd


def enqueue_preprocess(actor_id: str, sequence_id: str, cameras: Iterable[str] | None = None) -> str:
    cameras_list = list(cameras or [])
    job = job_registry.create_job(
        job_type="preprocess",
        sequence_id=sequence_id,
        camera_id=",".join(cameras_list) if cameras_list else "*",
        payload={"cameras": cameras_list},
    )

    cmd = _build_command(actor_id, sequence_id, cameras_list)

    def worker() -> None:
        job_registry.update_job(job.job_id, JobStatus.RUNNING)
        try:
            subprocess.run(cmd, check=True)
            job_registry.update_job(job.job_id, JobStatus.COMPLETED)
        except subprocess.CalledProcessError as exc:  # pragma: no cover
            log.exception("Preprocess job failed for sequence %s", sequence_id)
            job_registry.update_job(job.job_id, JobStatus.FAILED, message=str(exc))
        except FileNotFoundError as exc:  # pragma: no cover
            log.exception("Preprocess script missing")
            job_registry.update_job(job.job_id, JobStatus.FAILED, message=str(exc))

    thread = threading.Thread(target=worker, name=f"preprocess-{job.job_id}", daemon=True)
    thread.start()
    return job.job_id
