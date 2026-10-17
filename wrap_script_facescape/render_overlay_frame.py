#!/usr/bin/env python3
"""FaceScape counterpart to wrap_script/test_render_gt_frame.py -- draws the
wrapped mesh's keylines over one camera's actual photo.

Deliberately simpler than the Nersemble version: no raw/delegation-video
resolution rescaling (FaceScape's params.json intrinsics already match its own
JPG's resolution directly, no separate "raw capture" vs "delegation video"
distinction exists here), and no torch/pytorch3d -- the projection is a plain
pinhole camera model, cheap enough in numpy alone, keeping this script's
dependencies to exactly what run_pipeline.py already uses (numpy/cv2/trimesh).

Usage:
    python3 render_overlay_frame.py <ACTOR> <EXPRESSION> <CAMERA>
"""
from __future__ import annotations

import argparse
import json
from functools import lru_cache
import sys
from pathlib import Path

import cv2
import numpy as np
import trimesh
import facescape_paths as fp  # all paths: wrap_script_facescape/.env

fp.add_faceview_to_sys_path()
from backend.app.core.config import get_settings  # noqa: E402


@lru_cache(maxsize=1)
def _invalid_pairs(path: str) -> frozenset:
    try:
        with open(path) as f:
            return frozenset((str(p["actor"]), str(p["expression"])) for p in json.load(f).get("pairs", []))
    except (OSError, ValueError):
        return frozenset()


def _images_root_for(settings, actor_id: str, expression_id: str) -> Path:
    """Invalid pairs (no origin-space version) are rendered from their raw images/params."""
    if (actor_id, expression_id) in _invalid_pairs(str(settings.facescape_invalid_pairs_path)):
        return settings.facescape_raw_images_root
    return settings.facescape_images_root

LANDMARK_LINES_PATH = fp.LANDMARK_LINES_PATH
# wrapped_mesh.obj here is FaceScape's neutral.obj (built with eyeballs already
# BFS-removed, see render_flame_neutral.py) deformed onto the scan -- the same
# 3931-vertex eye-removed FLAME topology as Nersemble's own truncated
# wrapped_mesh.obj (see test_render_gt_frame.py's WRAPPED_MESH_EYE_CUTOFF), so
# the shared landmark_lines.npy applies here unchanged.
WRAPPED_MESH_EYE_CUTOFF = 3931

# FaceScape's photos are full native camera resolution -- much larger than
# Nersemble's fixed 550x802 delegation video, so the same thin 1px line
# render (test_render_gt_frame.py's own default) reads as barely visible
# here. Cameras also aren't all the same native resolution (e.g. cameras
# 1/4 are 2896x4344, cameras 14/17 are 1728x2592 -- confirmed via
# params.json's <cam>_width/_height) -- scaling every camera to the same
# TARGET_WIDTH (rather than a fixed divide-by-N factor) keeps the final
# render size, and therefore how thick a fixed-pixel line reads, consistent
# across cameras regardless of native resolution.
TARGET_WIDTH = 724
LINE_THICKNESS = 2


def render_overlay_frame(actor_id: str, expression_id: str, camera_id: str) -> Path:
    settings = get_settings()
    wrapped_mesh_path = settings.facescape_output_root / actor_id / expression_id / "wrapped_mesh.obj"
    if not wrapped_mesh_path.exists():
        raise FileNotFoundError(f"No wrapped_mesh.obj at {wrapped_mesh_path}")

    params_path = _images_root_for(settings, actor_id, expression_id) / actor_id / expression_id / "params.json"
    image_path = _images_root_for(settings, actor_id, expression_id) / actor_id / expression_id / f"{camera_id}.jpg"
    if not params_path.exists():
        raise FileNotFoundError(f"No params.json at {params_path}")
    if not image_path.exists():
        raise FileNotFoundError(f"No image at {image_path}")

    with open(params_path) as f:
        params = json.load(f)
    k_key, rt_key = f"{camera_id}_K", f"{camera_id}_Rt"
    if k_key not in params or rt_key not in params:
        raise KeyError(f"Camera {camera_id} not found in {params_path}")
    K = np.array(params[k_key], dtype=np.float64)
    Rt = np.array(params[rt_key], dtype=np.float64)

    landmark_lines = np.load(LANDMARK_LINES_PATH)
    landmark_lines = landmark_lines[(landmark_lines < WRAPPED_MESH_EYE_CUTOFF).all(axis=1)]

    mesh = trimesh.load(str(wrapped_mesh_path), process=False)
    verts = np.asarray(mesh.vertices, dtype=np.float64)  # (V, 3)

    verts_cam = verts @ Rt[:3, :3].T + Rt[:3, 3]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x = verts_cam[:, 0] / verts_cam[:, 2] * fx + cx
    y = verts_cam[:, 1] / verts_cam[:, 2] * fy + cy
    verts2d = np.stack([x, y], axis=-1)

    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Failed to read image {image_path}")

    scale = TARGET_WIDTH / image.shape[1]
    new_width = TARGET_WIDTH
    new_height = round(image.shape[0] * scale)
    image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    verts2d = verts2d * scale

    for line in landmark_lines:
        p1 = (int(verts2d[line[0], 0]), int(verts2d[line[0], 1]))
        p2 = (int(verts2d[line[1], 0]), int(verts2d[line[1], 1]))
        cv2.line(image, p1, p2, (20, 255, 20), LINE_THICKNESS, cv2.LINE_AA)

    output_dir = settings.line_renders_facescape_root
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{actor_id}_{expression_id}_{camera_id}.jpg"
    cv2.imwrite(str(output_path), image)
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("actor_id")
    parser.add_argument("expression_id")
    parser.add_argument("camera_id")
    args = parser.parse_args()
    output_path = render_overlay_frame(args.actor_id, args.expression_id, args.camera_id)
    print(f"Saved overlay frame render to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
