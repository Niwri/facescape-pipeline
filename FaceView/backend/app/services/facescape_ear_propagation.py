"""NOT CURRENTLY CALLED -- api/routes/facescape.py's save_facescape_ear_point
no longer invokes propagate_to_other_expressions() (per explicit instruction:
adding an ear landmark should only triangulate the current expression, never
propagate to others). Left in place rather than deleted in case cross-
expression ear propagation is wanted again later; kept up to date with its
own module-level docstring below for that reason.

Real-time propagation of a just-saved ear-top point (index 2 = left, 9 =
right) to every OTHER expression for the same actor, previously triggered
synchronously as part of POST /facescape/landmarks/ear (see
api/routes/facescape.py and facescape_ear_points.save_ear_point). NOT gated to
the neutral expression --
whichever expression the point was saved on is the propagation seed, and
every other expression with a params.json is a target. Deliberately separate
from run_neutral_skin_propagation.py, which stays skin-only and always
neutral-seeded -- ear propagation is its own independent flow, per explicit
instruction.

Reuses loop_service's already-loaded CoTracker model/device/wait_for_gpu_memory
rather than loading a second copy of the checkpoint into GPU memory (this
backend process already pays that cost for Nersemble's own live propagation).

Concurrency: acquire_actor_lock()/release_actor_lock() serialize a single
actor's own ear-save-and-propagate requests (so a double-click can't run two
CoTracker passes for the same actor at once) without affecting any other
actor -- FastAPI already dispatches this (sync) route handler to a thread
pool, so two different actors' requests run on separate threads and neither
waits on the other.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from ..core.config import get_settings
from .facescape_ear_points import EAR_INDEX_BY_SIDE, triangulate_point
from .loop_service import COTRACKER_REQUIRED_MB, _model as _cotracker_model, device, wait_for_gpu_memory

# ear side -> region camera set to triangulate/track with -- reuses
# run_neutral_skin_propagation.py's existing FACESCAPE_SKIN_REGION_CAMERAS
# left/right entries (see that file), per explicit choice not to define a
# separate ear-specific camera list.
EAR_SIDE_CAMERAS: Dict[str, List[str]] = {
    "left": ["4", "5", "7", "16", "20", "22", "31", "44", "45", "48"],
    "right": ["8", "9", "17", "21", "23", "25", "35", "36", "39"],
}

# CoTracker's own confidence that the tracked point is actually visible in the
# target frame (pred_visibility, a [0,1] probability). A camera's track below
# this is dropped rather than counted toward the >=2-observations minimum or
# fed into triangulation -- unlike loop_service.py's live Nersemble propagation
# (which only filters `vis <= 0`), ear propagation runs unattended with no
# reviewer watching the result, so it uses a real confidence bar.
MIN_TRACK_VISIBILITY = 0.5

_actor_lock = threading.Lock()
_actors_in_progress: set[str] = set()


class PropagationInProgress(Exception):
    """Raised when an actor already has an ear-save-and-propagate request in
    flight -- the route maps this to HTTP 409."""


def _load_invalid_pairs(path: Path) -> set[Tuple[str, str]]:
    """Same schema/loader as run_neutral_skin_propagation.py's own copy:
    {"pairs": [{"actor": ..., "expression": ...}, ...]} -- Phase 1's
    coordinate-transform alignment failed for these pairs, so there's no
    usable transformed images/params.json to propagate to (or seed from)."""
    if not path.exists():
        return set()
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return set()
    return {(p["actor"], p["expression"]) for p in data.get("pairs", [])}


def acquire_actor_lock(actor_id: str) -> None:
    with _actor_lock:
        if actor_id in _actors_in_progress:
            raise PropagationInProgress(
                f"Actor {actor_id} already has an ear point save/propagation in progress -- try again shortly."
            )
        _actors_in_progress.add(actor_id)


def release_actor_lock(actor_id: str) -> None:
    with _actor_lock:
        _actors_in_progress.discard(actor_id)


def _load_camera_params(actor_id: str, expression_id: str, settings) -> dict:
    params_path = settings.facescape_images_root / actor_id / expression_id / "params.json"
    with open(params_path) as f:
        params = json.load(f)
    camera_ids = sorted({k.rsplit("_", 1)[0] for k in params if k.endswith("_Rt")}, key=int)
    cams = {}
    for cam_id in camera_ids:
        k_key, rt_key = f"{cam_id}_K", f"{cam_id}_Rt"
        if k_key not in params:
            continue
        ext = np.eye(4, dtype=np.float64)
        ext[:3, :4] = np.array(params[rt_key], dtype=np.float64)
        cams[cam_id] = {
            "intrinsic": np.array(params[k_key], dtype=np.float64),
            "extrinsic": ext,
            "width": int(params.get(f"{cam_id}_width", 0) or 0),
            "height": int(params.get(f"{cam_id}_height", 0) or 0),
        }
    return cams


def _project_world_to_pixel(world_xyz: np.ndarray, intrinsic: np.ndarray, extrinsic: np.ndarray):
    r, t = extrinsic[:3, :3], extrinsic[:3, 3]
    cam_xyz = r @ world_xyz + t
    if cam_xyz[2] <= 1e-8:
        return None
    img = intrinsic @ cam_xyz
    return float(img[0] / img[2]), float(img[1] / img[2])


def _load_image(actor_id: str, expression_id: str, cam_id: str, settings):
    path = settings.facescape_images_root / actor_id / expression_id / f"{cam_id}.jpg"
    if not path.exists():
        return None
    return np.array(Image.open(path).convert("RGB"))


def _track_point_one_camera(
    source_img: np.ndarray, target_img: np.ndarray, index: int, seed_px: Tuple[float, float]
) -> Optional[Tuple[float, float]]:
    """Returns the tracked pixel position, or None if CoTracker's own visibility
    confidence for it falls below MIN_TRACK_VISIBILITY -- the caller drops these
    rather than counting them toward the >=2-observations minimum or feeding
    them into triangulation."""
    if source_img.shape != target_img.shape:
        h = min(source_img.shape[0], target_img.shape[0])
        w = min(source_img.shape[1], target_img.shape[1])
        source_img = source_img[:h, :w]
        target_img = target_img[:h, :w]
    frames = np.stack([source_img, target_img], axis=0)
    video_tensor = torch.from_numpy(frames).permute(0, 3, 1, 2)[None].float().to(device)
    query_tensor = torch.tensor([[0.0, seed_px[0], seed_px[1]]], dtype=torch.float32, device=device)[None]

    wait_for_gpu_memory(COTRACKER_REQUIRED_MB, label="FaceScape ear propagation")
    with torch.no_grad():
        pred_tracks, pred_visibility = _cotracker_model(video_tensor, queries=query_tensor)
    if device == "cuda":
        torch.cuda.synchronize()
    visibility = float(pred_visibility[0].detach().cpu().numpy()[1, 0])
    if visibility < MIN_TRACK_VISIBILITY:
        return None
    tracked_px = pred_tracks[0].detach().cpu().numpy()[1, 0]
    return float(tracked_px[0]), float(tracked_px[1])


def _write_ear_frame_point(actor_id: str, expression_id: str, cam_id: str, index: int, x_norm: float, y_norm: float, settings) -> None:
    cam_dir = settings.facescape_landmarks_root / actor_id / expression_id / f"cam_{cam_id}"
    cam_dir.mkdir(parents=True, exist_ok=True)
    ear_frame_path = cam_dir / "ear_frame_000000.json"
    if ear_frame_path.exists():
        with ear_frame_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    else:
        payload = {"frame_idx": 0, "points": []}
    payload["points"] = [p for p in payload["points"] if p["index"] != index]
    payload["points"].append({"index": index, "x": x_norm, "y": y_norm, "z": 0.0, "visibility": 1.0})
    tmp_path = ear_frame_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(payload), encoding="utf-8")
    tmp_path.replace(ear_frame_path)


def propagate_to_other_expressions(actor_id: str, source_expression_id: str, ear: str, world_xyz: np.ndarray) -> int:
    """Propagates one ear point's just-triangulated world position from
    source_expression_id to every OTHER expression this actor has (every
    expression dir with a params.json, excluding the source), via the same
    CoTracker-seed-track-triangulate-reproject flow
    run_neutral_skin_propagation.py uses for skin points -- just seeded from
    whichever expression was just saved, not always the neutral one. Returns
    the number of target expressions successfully propagated to.

    Caller (save_facescape_ear_point) is responsible for
    acquire_actor_lock()/release_actor_lock() around this call."""
    settings = get_settings()
    actor_dir = settings.facescape_images_root / actor_id
    invalid_pairs = _load_invalid_pairs(settings.facescape_invalid_pairs_path)
    target_expressions = sorted(
        p.name for p in actor_dir.iterdir()
        if p.is_dir() and p.name != source_expression_id and (p / "params.json").exists()
        and (actor_id, p.name) not in invalid_pairs
    )
    if not target_expressions:
        return 0

    cams = _load_camera_params(actor_id, source_expression_id, settings)
    index = EAR_INDEX_BY_SIDE[ear]
    region_cams = [c for c in EAR_SIDE_CAMERAS[ear] if c in cams]
    if len(region_cams) < 2:
        return 0

    if not settings.facescape_cameras_path.exists():
        return 0
    with open(settings.facescape_cameras_path) as f:
        facescape_cameras = json.load(f)
    output_camera_ids = sorted({str(c) for c in facescape_cameras.get("all_culled", [])}, key=int)

    source_seed_by_cam: Dict[str, Tuple[float, float]] = {}
    for cam_id in region_cams:
        cam = cams[cam_id]
        px = _project_world_to_pixel(world_xyz, cam["intrinsic"], cam["extrinsic"])
        if px is not None:
            source_seed_by_cam[cam_id] = px

    source_img_cache: Dict[str, np.ndarray] = {}
    propagated = 0
    for expression_id in target_expressions:
        tracked_by_cam: Dict[str, Tuple[float, float]] = {}
        for cam_id, seed_px in source_seed_by_cam.items():
            source_img = source_img_cache.get(cam_id)
            if source_img is None:
                source_img = _load_image(actor_id, source_expression_id, cam_id, settings)
                if source_img is None:
                    continue
                source_img_cache[cam_id] = source_img
            target_img = _load_image(actor_id, expression_id, cam_id, settings)
            if target_img is None:
                continue
            tracked_px = _track_point_one_camera(source_img, target_img, index, seed_px)
            if tracked_px is None:
                continue
            tracked_by_cam[cam_id] = tracked_px

        observations = list(tracked_by_cam.items())
        if len(observations) < 2:
            continue
        intrinsic_arr = np.stack([cams[c]["intrinsic"] for c in region_cams])
        extrinsic_list = [cams[c]["extrinsic"] for c in region_cams]
        width = cams[region_cams[0]]["width"]
        height = cams[region_cams[0]]["height"]
        point_list = [(region_cams.index(cam_id), np.array(pixel_xy), True) for cam_id, pixel_xy in observations]

        fused = triangulate_point(point_list, intrinsic_arr, extrinsic_list, width, height)
        if fused is None:
            continue

        for cam_id in output_camera_ids:
            cam = cams.get(cam_id)
            if cam is None or not cam["width"] or not cam["height"]:
                continue
            px = _project_world_to_pixel(fused, cam["intrinsic"], cam["extrinsic"])
            if px is None:
                continue
            x_norm, y_norm = px[0] / cam["width"], px[1] / cam["height"]
            _write_ear_frame_point(actor_id, expression_id, cam_id, index, x_norm, y_norm, settings)
        propagated += 1

    return propagated
