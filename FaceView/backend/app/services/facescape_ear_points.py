"""Triangulate-and-project service for FaceScape's two manually-labeled
ear-top landmarks (index 2 = left ear top, index 9 = right ear top) -- the
only two standard FLAME indices with no sapiens2 detection source (see
process_expression_sapiens.py's SAPIENS_LANDMARK_MAP, where both map to
None). A reviewer clicks these in >=2 camera views via the frontend's "Add
Left/Right Ear Top" mode; this module triangulates one 3D point from those
clicks and projects it into every camera present for the pair, writing
cam_<id>/ear_frame_000000.json -- a file process_expression_sapiens.py never
writes to, so a rerun of that script can never clobber a saved ear point.
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Literal, Tuple

import numpy as np

from ..core.config import get_settings
from . import facescape_invalid

# triangulate_point()/is_in_bounds() live in wrap_script/, not this backend's
# own tree -- imported directly (rather than duplicating the least-squares +
# outlier-rejection math here) the same way other cross-tree pipeline pieces
# in this codebase reach into sibling script directories.
_WRAP_SCRIPT_ROOT = get_settings().wrap_script_root  # WRAP_SCRIPT_ROOT in FaceView/backend/.env
if str(_WRAP_SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(_WRAP_SCRIPT_ROOT))
from triangulate import triangulate_point  # noqa: E402

LEFT_EAR_INDEX = 2
RIGHT_EAR_INDEX = 9
EAR_INDEX_BY_SIDE: Dict[str, int] = {"left": LEFT_EAR_INDEX, "right": RIGHT_EAR_INDEX}

Side = Literal["left", "right"]

# save_ear_point's per-camera projection+write is pure I/O (NFS mkdir/stat/
# read/write/rename round-trips, one set per camera in the pair -- up to ~54
# for FaceScape) with no shared state across cameras, so it parallelizes
# cleanly. 32 is comfortably above any actor/expression pair's camera count,
# so every camera gets its own worker rather than queueing behind others.
_MAX_WRITE_WORKERS = 32


def _load_pair_cameras(actor_id: str, expression_id: str):
    """Same params.json loading as wrap_script_facescape/run_pipeline.py's
    run_pipeline() -- per-camera 3x3 intrinsic, 4x4 extrinsic, and pixel
    width/height, keyed by camera_id string. FaceScape cameras each have
    their own distinct focal length/resolution (unlike Nersemble's single
    shared rig intrinsic), so this is per-camera throughout."""
    params_path = facescape_invalid.images_root_for(actor_id, expression_id) / actor_id / expression_id / "params.json"
    if not params_path.exists():
        raise FileNotFoundError(f"params.json not found: {params_path}")
    with params_path.open() as f:
        params = json.load(f)

    camera_ids = sorted({k.rsplit("_", 1)[0] for k in params if k.endswith("_Rt")}, key=int)
    intrinsics: Dict[str, np.ndarray] = {}
    extrinsics: Dict[str, np.ndarray] = {}
    widths: Dict[str, int] = {}
    heights: Dict[str, int] = {}
    for cam_id in camera_ids:
        k_key, rt_key = f"{cam_id}_K", f"{cam_id}_Rt"
        if k_key not in params:
            continue
        intrinsics[cam_id] = np.array(params[k_key], dtype=np.float64)
        rt = np.array(params[rt_key], dtype=np.float64)
        ext = np.eye(4, dtype=np.float64)
        ext[:3, :4] = rt
        extrinsics[cam_id] = ext
        widths[cam_id] = int(params.get(f"{cam_id}_width", 0) or 0)
        heights[cam_id] = int(params.get(f"{cam_id}_height", 0) or 0)
    return intrinsics, extrinsics, widths, heights


def save_ear_point(
    actor_id: str, expression_id: str, ear: Side, clicks: Dict[str, Tuple[float, float]],
) -> Tuple[int, np.ndarray]:
    """Triangulates one 3D point from `clicks` (camera_id -> normalized (x, y)
    in [0,1], as clicked in the frontend) and projects it into every camera
    present in this pair's params.json, merging {"index": 2 or 9, ...} into
    each camera's ear_frame_000000.json (creating it if absent; replacing
    only this one index's entry if the file already exists, e.g. because the
    other ear was saved earlier). Returns (cameras_written, world_xyz) -- the
    route currently ignores world_xyz (cross-expression propagation is
    disabled, see api/routes/facescape.py's save_facescape_ear_point), kept
    in the return signature only so facescape_ear_propagation.py's own
    (currently uncalled) propagate_to_other_expressions() doesn't need to
    re-triangulate from scratch if it's ever wired back in.
    Raises ValueError if fewer than 2 clicks are given (triangulate_point's
    own minimum) or triangulation otherwise fails.
    """
    if len(clicks) < 2:
        raise ValueError(f"Need at least 2 camera clicks to triangulate, got {len(clicks)}")

    settings = get_settings()
    intrinsics, extrinsics, widths, heights = _load_pair_cameras(actor_id, expression_id)

    missing = [cam_id for cam_id in clicks if cam_id not in intrinsics]
    if missing:
        raise ValueError(f"Camera(s) not found in params.json for {actor_id}/{expression_id}: {missing}")

    all_camera_ids = list(intrinsics.keys())
    cam_index_by_id = {cam_id: i for i, cam_id in enumerate(all_camera_ids)}
    intrinsic_array = np.array([intrinsics[c] for c in all_camera_ids])
    extrinsic_list = [extrinsics[c] for c in all_camera_ids]
    width_list = [widths[c] for c in all_camera_ids]
    height_list = [heights[c] for c in all_camera_ids]

    point_list = []
    for cam_id, (x_norm, y_norm) in clicks.items():
        cam_index = cam_index_by_id[cam_id]
        pixel = np.array([x_norm * widths[cam_id], y_norm * heights[cam_id]])
        in_bounds = 0.0 < x_norm < 1.0 and 0.0 < y_norm < 1.0
        point_list.append((cam_index, pixel, in_bounds))

    world_xyz = triangulate_point(point_list, intrinsic_array, extrinsic_list, width_list, height_list)
    if world_xyz is None:
        raise ValueError("Triangulation failed -- no in-bounds click among the given cameras")

    index = EAR_INDEX_BY_SIDE[ear]
    landmarks_dir = settings.facescape_landmarks_root / actor_id / expression_id

    def _project_and_write_one_camera(cam_id: str) -> bool:
        """Projects world_xyz into cam_id and merges it into that camera's
        ear_frame_000000.json. Runs on a worker thread -- see
        _MAX_WRITE_WORKERS -- so must not touch anything another camera's
        call also touches (it doesn't: each camera has its own file)."""
        width, height = widths[cam_id], heights[cam_id]
        if not width or not height:
            return False
        ext = extrinsics[cam_id]
        R, t = ext[:3, :3], ext[:3, 3]
        cam_xyz = R @ world_xyz + t
        if cam_xyz[2] <= 1e-8:
            return False  # point is behind this camera -- nothing sane to project
        img = intrinsics[cam_id] @ cam_xyz
        x_pixel, y_pixel = img[0] / img[2], img[1] / img[2]
        x_norm, y_norm = x_pixel / width, y_pixel / height

        cam_dir = landmarks_dir / f"cam_{cam_id}"
        cam_dir.mkdir(parents=True, exist_ok=True)
        ear_frame_path = cam_dir / "ear_frame_000000.json"
        if ear_frame_path.exists():
            with ear_frame_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
        else:
            payload = {"frame_idx": 0, "points": []}
        payload["points"] = [p for p in payload["points"] if p["index"] != index]
        payload["points"].append({
            "index": index, "x": float(x_norm), "y": float(y_norm), "z": float(cam_xyz[2]),
            "visibility": 1.0,
        })
        tmp_path = ear_frame_path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(payload), encoding="utf-8")
        tmp_path.replace(ear_frame_path)
        return True

    # Every camera's read-merge-write is an independent NFS round-trip with no
    # shared state, so fanning them out across threads (I/O-bound -- the GIL
    # is released during file I/O) turns ~54 sequential round-trips into ~54
    # concurrent ones -- serial writes put a single ear-point save in the
    # 10-20s range on this NFS mount, measured directly against real data.
    with ThreadPoolExecutor(max_workers=_MAX_WRITE_WORKERS) as pool:
        results = pool.map(_project_and_write_one_camera, all_camera_ids)
    written = sum(1 for wrote in results if wrote)
    return written, world_xyz


def load_ear_points(actor_id: str, expression_id: str, camera_id: str) -> List[dict]:
    """Points saved for this camera, or [] if ear_frame_000000.json doesn't
    exist yet (the normal state before either ear has been labeled)."""
    settings = get_settings()
    ear_frame_path = (
        settings.facescape_landmarks_root / actor_id / expression_id / f"cam_{camera_id}" / "ear_frame_000000.json"
    )
    if not ear_frame_path.exists():
        return []
    with ear_frame_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("points", [])


def ear_points_exist(actor_id: str, expression_id: str) -> bool:
    """True only if BOTH ear indices (2 and 9) have a saved point in at least
    one camera's ear_frame_000000.json for this pair -- used by propagation
    tooling to decide whether an actor's neutral expression has ear
    landmarks to propagate at all."""
    settings = get_settings()
    landmarks_dir = settings.facescape_landmarks_root / actor_id / expression_id
    if not landmarks_dir.is_dir():
        return False
    found: set[int] = set()
    for cam_dir in landmarks_dir.iterdir():
        if not cam_dir.is_dir() or not cam_dir.name.startswith("cam_"):
            continue
        ear_frame_path = cam_dir / "ear_frame_000000.json"
        if not ear_frame_path.exists():
            continue
        with ear_frame_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        for p in payload.get("points", []):
            found.add(p["index"])
        if LEFT_EAR_INDEX in found and RIGHT_EAR_INDEX in found:
            return True
    return False
