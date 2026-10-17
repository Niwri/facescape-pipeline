"""High-level orchestration for the human-in-the-loop update cycle."""
from __future__ import annotations

import asyncio
import logging
import threading
import sys
from pathlib import Path

import torch
import numpy as np

# Add co-tracker to path
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "co-tracker") not in sys.path:
    sys.path.append(str(ROOT / "co-tracker"))

from cotracker.predictor import CoTrackerPredictor
from cotracker.utils.visualizer import read_video_from_path

from ..core.config import get_settings
from ..core.jobs import JobStatus, job_registry
from ..schemas.landmarks import LandmarkPayload
from ..schemas.common import LandmarkPoint
from .landmark_store import landmark_store
from ..api.routes import ws
import torch
from typing import List
torch.autograd.set_grad_enabled(False)

log = logging.getLogger(__name__)

_model = None
DELEGATION_SIZE = (550, 802) # width, height

device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint_path = ROOT / "co-tracker/checkpoints/scaled_offline.pth"

_model = CoTrackerPredictor(checkpoint=str(checkpoint_path))
_model = _model.to(device)


def _run_propagation(payload: LandmarkPayload, window: int, changed_points_indices: List[int]) -> int:
    """Run CoTracker to propagate landmarks. Returns number of frames propagated."""
    settings = get_settings()
    video_path = settings.processed_root / payload.actor_id / payload.sequence_id / f"{payload.camera_id}_delegation.mp4"
    
    if not video_path.exists():
        raise FileNotFoundError(f"Delegation video not found: {video_path}")

    # Load video
    video = read_video_from_path(str(video_path))
    if video is None:
        raise ValueError(f"Failed to read video: {video_path}")
        
    video = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    video = video.to(device)

    # Prepare queries: [1, N, 3] -> (t, x, y)
    queries = []
    query_map = {}
    q_idx = 0
    for pt in payload.points:
        if(pt.index in changed_points_indices):
            query_map[pt.index] = q_idx 
            q_idx += 1
            queries.append([float(payload.frame_idx), pt.x * DELEGATION_SIZE[0], pt.y * DELEGATION_SIZE[1]])
    
    if not queries:
        return 0

    queries = torch.tensor(queries).float().to(device)[None] # [1, N, 3]

    model = _model
    pred_tracks, pred_visibility = model(video, queries=queries)
    
    # pred_tracks: [1, T, N, 2]
    T = pred_tracks.shape[1]
    start_frame = payload.frame_idx + 1
    
    if window < 0:
        end_frame = T
    else:
        end_frame = min(payload.frame_idx + window + 1, T)

    orig_frames = landmark_store.load_all_frames(payload.actor_id, payload.sequence_id, payload.camera_id)
    
    for t in range(start_frame, end_frame):
        point_idx_map = {pt.index: j for j, pt in enumerate(orig_frames[t].points)}
        
        for i, idx in enumerate(changed_points_indices):
            # Get tracked position
            x, y = pred_tracks[0, t, query_map[idx]].tolist()
            vis = pred_visibility[0, t, query_map[idx]].item()

            norm_x = x / DELEGATION_SIZE[0]
            norm_y = y / DELEGATION_SIZE[1]

            j = point_idx_map.get(idx)
            if j is not None:
                pt = orig_frames[t].points[j]
                orig_frames[t].points[j] = LandmarkPoint(
                    index=pt.index,
                    x=norm_x,
                    y=norm_y,
                    z=pt.z,
                    visibility=vis,
                )
            
        
        updated_payload = LandmarkPayload(
            frame_idx=t,
            camera_id=payload.camera_id,
            sequence_id=payload.sequence_id,
            actor_id=payload.actor_id,
            points=orig_frames[t].points
        )
        landmark_store.save_frame(updated_payload)
    
    return end_frame - start_frame


def enqueue_propagation(payload: LandmarkPayload, changed_points_indices: List[int], window: int | None = None) -> str:
    # Use -1 to indicate "all future frames" if window is not provided
    target_window = window if window is not None else -1

    job = job_registry.create_job(
        job_type="propagate",
        sequence_id=payload.sequence_id,
        camera_id=payload.camera_id,
        payload={"frame_idx": payload.frame_idx, "window": target_window},
    )

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    def worker() -> None:
        job_registry.update_job(job.job_id, JobStatus.RUNNING)
        try:
            actual_window = _run_propagation(payload, target_window, changed_points_indices)
            job_registry.update_job(job.job_id, JobStatus.COMPLETED)
            
            msg = {
                "event": "propagation_complete",
                "job_id": job.job_id,
                "camera_id": payload.camera_id,
                "frame_idx": payload.frame_idx,
                "window": actual_window,
            }
            if loop and loop.is_running():
                asyncio.run_coroutine_threadsafe(ws.broadcast(payload.sequence_id, msg), loop)
            else:
                asyncio.run(ws.broadcast(payload.sequence_id, msg))

        except Exception as exc:  # pragma: no cover
            log.exception("Propagation job failed")
            job_registry.update_job(job.job_id, JobStatus.FAILED, message=str(exc))
            
            msg = {
                "event": "propagation_failed",
                "job_id": job.job_id,
                "camera_id": payload.camera_id,
                "frame_idx": payload.frame_idx,
                "window": target_window,
                "error": str(exc),
            }
            if loop and loop.is_running():
                asyncio.run_coroutine_threadsafe(ws.broadcast(payload.sequence_id, msg), loop)
            else:
                asyncio.run(ws.broadcast(payload.sequence_id, msg))

    thread = threading.Thread(target=worker, name=f"propagate-{job.job_id}", daemon=True)
    thread.start()
    return job.job_id
