"""FaceScape dataset discovery. Mirrors dataset_index.py's shape, including its
stale-while-revalidate cache: this was originally assumed to be cheap enough
(no video/delegation-video checks, just directory listing + params.json +
per-camera landmark-file existence) to get away with a plain TTL cache, but
measured in practice at 5+ minutes for a single scan under NFS load -- same
failure mode as Nersemble's pre-fix cache (see dataset_index.py's docstring),
just with a smaller actor count than the 421-actor video-backed Nersemble scan
masking it for longer. Same mechanism here: once any prior result exists, an
expired cache triggers a background refresh and immediately serves the stale
response rather than blocking the caller behind the rescan."""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import List, Set, Tuple

from . import facescape_invalid
from ..core.config import get_settings
from ..schemas.facescape import (
    FaceScapeActorDescriptor,
    FaceScapeCameraDescriptor,
    FaceScapeDatasetResponse,
    FaceScapeExpressionDescriptor,
)
from .label_tracker import label_tracker

_CACHE_TTL_SECONDS = 30
_cache_lock = threading.Lock()
_cached_response: FaceScapeDatasetResponse | None = None
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
    # clobbered by an in-flight scan that started before it (this is exactly
    # what was causing FaceScape's frontend to sometimes show a stale
    # label_status after a confirm/unconfirm). The existing cache (still marked
    # stale) is left in place, so the next request just retries with a clean
    # scan rather than being served nothing.
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


def _load_invalid_pairs(path: Path) -> Set[Tuple[str, str]]:
    if not path.exists():
        return set()
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return set()
    return {(p["actor"], p["expression"]) for p in data.get("pairs", [])}


def _load_cameras_json(path: Path) -> dict:
    empty = {"front": [], "left_front": [], "right_front": [], "all_culled": []}
    if not path.exists():
        return empty
    try:
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
    except (json.JSONDecodeError, OSError):
        return empty
    # Camera ids are stored as ints in facescape_cameras.json (see
    # compute_facescape_cameras.py) but every other camera-id-shaped value in
    # this codebase (params.json keys, jpg filenames, cam_<id> directories) is
    # a string -- normalize once here so downstream code never has to care.
    return {key: [str(c) for c in raw.get(key, [])] for key in empty}


def list_facescape_dataset() -> FaceScapeDatasetResponse:
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


def _scan_dataset() -> FaceScapeDatasetResponse:
    settings = get_settings()
    invalid_pairs = _load_invalid_pairs(settings.facescape_invalid_pairs_path)
    cameras_json = _load_cameras_json(settings.facescape_cameras_path)
    all_cameras: List[str] = cameras_json.get("all_culled", [])
    label_statuses = label_tracker.get_all(dataset="FaceScape")

    actors: List[FaceScapeActorDescriptor] = []
    if not settings.facescape_images_root.is_dir():
        return FaceScapeDatasetResponse(
            actors=[], front_cameras=cameras_json.get("front", []),
            left_front_cameras=cameras_json.get("left_front", []),
            right_front_cameras=cameras_json.get("right_front", []),
        )

    actor_dirs = sorted(
        (p for p in settings.facescape_images_root.iterdir() if p.is_dir() and p.name.isdigit()),
        key=lambda p: int(p.name),
    )
    for actor_dir in actor_dirs:
        actor_id = actor_dir.name
        expressions: List[FaceScapeExpressionDescriptor] = []
        for expr_dir in sorted(actor_dir.iterdir()):
            if not expr_dir.is_dir():
                continue
            expression_id = expr_dir.name
            pair_invalid = (actor_id, expression_id) in invalid_pairs
            if pair_invalid:
                # No origin-space version: raw images/params, own camera ids.
                expr_dir = settings.facescape_raw_images_root / actor_id / expression_id
            if not (expr_dir / "params.json").exists():
                continue
            pair_cameras = facescape_invalid.pair_camera_ids(actor_id, expression_id) if pair_invalid else all_cameras

            landmarks_dir = settings.facescape_landmarks_root / actor_id / expression_id
            cameras: List[FaceScapeCameraDescriptor] = []
            any_landmarks = False
            for camera_id in pair_cameras:
                has_image = (expr_dir / f"{camera_id}.jpg").exists()
                has_landmarks = (landmarks_dir / f"cam_{camera_id}" / "frame_000000.json").exists()
                any_landmarks = any_landmarks or has_landmarks
                cameras.append(FaceScapeCameraDescriptor(
                    camera_id=camera_id, has_image=has_image, has_landmarks=has_landmarks,
                ))

            expressions.append(FaceScapeExpressionDescriptor(
                expression_id=expression_id,
                cameras=cameras,
                has_landmarks=any_landmarks,
                label_status=label_tracker.get_status(
                    actor_id, expression_id, dataset="FaceScape", all_statuses=label_statuses,
                ),
                is_invalid=pair_invalid,
                overlay_camera_ids=facescape_invalid.overlay_camera_ids(actor_id, expression_id) if pair_invalid else [],
            ))
        if expressions:
            actors.append(FaceScapeActorDescriptor(actor_id=actor_id, expressions=expressions))

    return FaceScapeDatasetResponse(
        actors=actors,
        front_cameras=cameras_json.get("front", []),
        left_front_cameras=cameras_json.get("left_front", []),
        right_front_cameras=cameras_json.get("right_front", []),
    )
