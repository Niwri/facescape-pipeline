"""Dataset discovery helpers."""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import List

from ..core.config import get_settings
from ..core.nersemble_sequence_blacklist import BLACKLISTED_SEQUENCES
from ..schemas.datasets import ActorDescriptor, CameraDescriptor, DatasetResponse, SequenceDescriptor
from ..schemas.common import MediaPath
from .label_tracker import label_tracker

# list_dataset() walks every actor/sequence/camera and does two filesystem exists()
# checks per camera against data_root/processed_root/landmarks_root -- tens of
# thousands of NFS round-trips at this dataset's current size (421 actors). This
# used to take ~100s end to end, but the underlying NFS mount (processed_root/
# landmarks_root live on the flaky /scratch/ondemand32 export) has been observed
# to degrade badly enough to push a single scan past 11 minutes. A plain
# synchronous TTL cache still blocks whichever request lands right after
# expiry behind that full cost, which from the frontend looks identical to the
# backend being unresponsive. So: once we have any prior result, an expired
# cache triggers a background refresh and immediately serves the stale response
# rather than making the caller wait -- only the very first call after backend
# startup (no prior result to fall back on) blocks synchronously. The frontend
# already applies label_status changes in-place rather than refetching, so
# serving a-few-minutes-stale data in the meantime is a non-issue -- but a scan
# that outlives a write is: see _refresh_in_background's own comment for why the
# scan result is discarded rather than committed in that case.
_CACHE_TTL_SECONDS = 30
_cache_lock = threading.Lock()
_cached_response: DatasetResponse | None = None
_cached_at: float = 0.0
_refresh_in_progress = False


def _label_tracker_mtime() -> float:
    try:
        return label_tracker.path.stat().st_mtime
    except OSError:
        return 0.0


def _refresh_in_background() -> None:
    global _cached_response, _cached_at, _refresh_in_progress
    # A scan can take minutes (see module docstring). If a confirm/unconfirm
    # write -- from this backend (a user's click) or external wrap_script
    # tooling writing the same label_tracker.json -- lands while this scan is
    # still in flight, the scan may have read that pair's status either just
    # before or just after the write, and there's no way to tell which after
    # the fact. Comparing the tracker file's mtime before/after catches either
    # case conservatively: if it changed at all during the scan, discard the
    # result instead of committing it, so a write can never get silently
    # clobbered by an in-flight scan that started before it. The existing cache
    # (still marked stale) is left in place, so the next request just retries
    # with a clean scan rather than being served nothing.
    mtime_before = _label_tracker_mtime()
    try:
        response = _scan_dataset()
        if _label_tracker_mtime() != mtime_before:
            return
        with _cache_lock:
            _cached_response = response
            _cached_at = time.monotonic()
    finally:
        with _cache_lock:
            _refresh_in_progress = False


def _load_face_indices(configs_root: Path) -> dict:
    face_indices_path = configs_root / "face_indices.json"
    if face_indices_path.exists():
        with face_indices_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    return {}


def _load_default_cameras(default_cameras_path: Path) -> List[str]:
    if default_cameras_path.exists():
        with default_cameras_path.open("r", encoding="utf-8") as handle:
            camera_ids = json.load(handle)
        return [f"cam_{camera_id}" for camera_id in camera_ids]
    return []


def list_dataset() -> DatasetResponse:
    global _cached_response, _cached_at, _refresh_in_progress
    with _cache_lock:
        is_stale = _cached_response is None or (time.monotonic() - _cached_at) >= _CACHE_TTL_SECONDS
        if not is_stale:
            return _cached_response

        if _cached_response is not None:
            # Have something to serve already -- kick off a background refresh
            # (deduped via _refresh_in_progress) and return the stale response
            # immediately instead of blocking this request behind the rescan.
            if not _refresh_in_progress:
                _refresh_in_progress = True
                threading.Thread(target=_refresh_in_background, daemon=True).start()
            return _cached_response

    # No prior result at all (first call since backend startup) -- nothing to
    # fall back on, so this one has to block.
    response = _scan_dataset()
    with _cache_lock:
        _cached_response = response
        _cached_at = time.monotonic()
    return response


def _scan_dataset() -> DatasetResponse:
    settings = get_settings()
    label_statuses = label_tracker.get_all(dataset="Nersemble")
    actors: List[ActorDescriptor] = []
    for actor_dir in sorted(settings.data_root.glob("*/")):
        actor_id = actor_dir.name
        sequences: List[SequenceDescriptor] = []
        for sequence_dir in sorted((actor_dir / "sequences").glob("*/")):
            sequence_id = sequence_dir.name
            if sequence_id in BLACKLISTED_SEQUENCES:
                continue
            cameras: List[CameraDescriptor] = []
            images_dir = sequence_dir / "images"
            if not images_dir.exists():
                continue
            for video_path in sorted(images_dir.glob("*.mp4")):
                camera_id = video_path.stem
                processed_video = settings.processed_root / actor_id / sequence_id / f"{camera_id}_delegation.mp4"
                landmarks_dir = settings.landmarks_root / actor_id / sequence_id / camera_id
                descriptor = CameraDescriptor(
                    camera_id=camera_id,
                    video=MediaPath(actor_id=actor_id, sequence_id=sequence_id, camera_id=camera_id, video_path=video_path),
                    status="ready" if processed_video.exists() and landmarks_dir.exists() else "pending",
                )
                cameras.append(descriptor)
            sequences.append(
                SequenceDescriptor(
                    sequence_id=sequence_id,
                    cameras=cameras,
                    has_landmarks=any(settings.landmarks_root.joinpath(actor_id, sequence_id, cam.camera_id).exists() for cam in cameras),
                    has_delegation_video=any(
                        settings.processed_root.joinpath(actor_id, sequence_id, f"{cam.camera_id}_delegation.mp4").exists()
                        for cam in cameras
                    ),
                    label_status=label_tracker.get_status(actor_id, sequence_id, dataset="Nersemble", all_statuses=label_statuses),
                )
            )
        actors.append(ActorDescriptor(actor_id=actor_id, sequences=sequences))
    return DatasetResponse(
        actors=actors,
        face_indices=_load_face_indices(settings.configs_root),
        default_cameras=_load_default_cameras(settings.default_cameras_path),
    )
