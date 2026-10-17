"""Camera loading/projection helpers for the Ava-256 pipeline. Wraps
render_overlay.py's own load_camera/load_image/project_points (reused via
explicit-file-path import, same trick mesh_utils.py uses for extract_obj.py)
rather than duplicating the column-major K/T-transpose and AVIF-scale logic
already established/verified there.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_render_overlay = _load_module("wrap_script_ava256_render_overlay", SCRIPT_DIR / "render_overlay.py")

load_camera = _render_overlay.load_camera            # (actor_dir, camera_id) -> (K, Rt), K already AVIF-scaled
load_image = _render_overlay.load_image              # (actor_dir, camera_id, frame_id) -> BGR np.ndarray
project_points = _render_overlay.project_points       # (verts_world, K, Rt) -> (N,2) pixel coords


def load_all_camera_ids(actor_dir: Path) -> list[str]:
    calib_path = Path(actor_dir) / "decoder" / "camera_calibration.json"
    with calib_path.open("r", encoding="utf-8") as f:
        calib = json.load(f)
    return [c["cameraId"] for c in calib["KRT"]]


def camera_forward_world(Rt: np.ndarray) -> np.ndarray:
    """Unit forward vector (camera's own +Z axis) expressed in world space.
    world_2_cam: point_cam = R @ point_world + t, so the world-space
    direction the camera faces is R^T @ [0,0,1], which equals the 3rd ROW of
    R (as a column vector) -- Rt[2, :3]."""
    v = Rt[2, :3]
    return v / np.linalg.norm(v)


def camera_center_world(Rt: np.ndarray) -> np.ndarray:
    """World-space camera center: C = -R^T @ t."""
    R, t = Rt[:3, :3], Rt[:3, 3]
    return -R.T @ t
