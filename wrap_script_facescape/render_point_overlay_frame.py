#!/usr/bin/env python3
"""FaceScape counterpart to wrap_script/render_point_overlay.py -- draws the
annotated (propagated) landmarks vs. the wrapped mesh's own landmarks over
one camera's photo. Image, camera and resize handling mirror
render_overlay_frame.py (the keyline Overlay Frame render), so the two line
up; point selection and styling come from wrap_script/point_overlay.py.

Output: /scratch/ondemand32/irwinngo/point_overlay_renders_facescape/<actor>_<expression>_<camera>.jpg,
served to FaceView's "Overlay Landmarks" tab by /facescape/point-overlay-frame.

Usage:
    python3 render_point_overlay_frame.py <ACTOR> <EXPRESSION> [CAMERA ...]   # default cameras 1 4 17
    python3 render_point_overlay_frame.py --backfill [--parallel N]
        # every "unconfirmed"/"confirmed" pair with a wrapped_mesh.obj and no
        # point overlay render yet
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from functools import lru_cache
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
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

_point_overlay_spec = importlib.util.spec_from_file_location(
    "wrap_script_point_overlay", str(fp.WRAP_SCRIPT_ROOT / "point_overlay.py")
)
po = importlib.util.module_from_spec(_point_overlay_spec)
_point_overlay_spec.loader.exec_module(po)

POINT_OVERLAY_RENDERS_ROOT = fp.POINT_OVERLAY_RENDERS_FACESCAPE_ROOT
# FaceScape's own point-on-triangle file -- the one its wrap uses
# (run_wrap_script.py DEFAULT_POT_PATH). Its ear rows (1/4/6/14) point at
# different FLAME vertices than Nersemble's.
FACESCAPE_POT_PATH = fp.FACESCAPE_PIPELINE_DIR / "neutral_correspondence_without_eyes_POT.json"
# Same cameras as run_pipeline.py's OVERLAY_FRAME_CAMERA_IDS (the keyline
# Overlay Frame tab) and render_overlay_frame.py's TARGET_WIDTH.
DEFAULT_CAMERA_IDS = ["1", "4", "17"]
TARGET_WIDTH = 724
# FaceView's FaceScape ImageTile shows these on top of whitelist.json
# (FACESCAPE_EXTRA_VISIBLE_INDICES).
FACESCAPE_EXTRA_VISIBLE_INDICES = {69, 72}


def output_path_for(actor_id: str, expression_id: str, camera_id: str) -> Path:
    return POINT_OVERLAY_RENDERS_ROOT / f"{actor_id}_{expression_id}_{camera_id}.jpg"


def render_point_overlay_frame(
    actor_id: str, expression_id: str, camera_id: str,
) -> tuple[Path, float | None, float | None]:
    """Writes the image; returns (path, mean standard-landmark pixel error,
    mean skin-landmark pixel error), at TARGET_WIDTH."""
    settings = get_settings()
    wrapped_mesh_path = settings.facescape_output_root / actor_id / expression_id / "wrapped_mesh.obj"
    params_path = _images_root_for(settings, actor_id, expression_id) / actor_id / expression_id / "params.json"
    image_path = _images_root_for(settings, actor_id, expression_id) / actor_id / expression_id / f"{camera_id}.jpg"
    for path in (wrapped_mesh_path, params_path, image_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}")

    with open(params_path) as f:
        params = json.load(f)
    K = np.array(params[f"{camera_id}_K"], dtype=np.float64)
    Rt = np.array(params[f"{camera_id}_Rt"], dtype=np.float64)

    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Failed to read image {image_path}")
    scale = TARGET_WIDTH / image.shape[1]
    width, height = TARGET_WIDTH, round(image.shape[0] * scale)
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    K = K.copy()
    K[:2] *= scale

    mesh = trimesh.load(str(wrapped_mesh_path), process=False)
    pot_rows, skin_vertex_by_index = po.load_mesh_landmark_spec(FACESCAPE_POT_PATH)
    mesh_world = po.mesh_landmarks_world(
        np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces), pot_rows, skin_vertex_by_index,
    )
    mesh_px = po.project_mesh_landmarks(mesh_world, K, Rt)

    cam_dir = settings.facescape_landmarks_root / actor_id / expression_id / f"cam_{camera_id}"
    visible = po.load_whitelist() | FACESCAPE_EXTRA_VISIBLE_INDICES
    annotated = po.annotated_points_px(cam_dir, "000000", width, height, visible)

    rendered, errors = po.draw_point_overlay(image, annotated, mesh_px)
    output_path = output_path_for(actor_id, expression_id, camera_id)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), rendered)
    return (output_path, *po.split_errors(errors))


def backfill_pairs(camera_ids: list[str], force: bool = False) -> list[tuple[str, str]]:
    settings = get_settings()
    with open(settings.label_tracker_path) as f:
        data = json.load(f).get("FaceScape", {})
    pairs = []
    for actor_id, expressions in sorted(data.items()):
        for expression_id, status in sorted(expressions.items()):
            if status not in ("unconfirmed", "confirmed"):
                continue
            if not force and all(output_path_for(actor_id, expression_id, cam).exists() for cam in camera_ids):
                continue
            if (settings.facescape_output_root / actor_id / expression_id / "wrapped_mesh.obj").exists():
                pairs.append((actor_id, expression_id))
    return pairs


def _render_pair(actor_id: str, expression_id: str, camera_ids: list[str]) -> list[str]:
    """One line per camera. A camera that can't be rendered (e.g. missing from
    params.json, or no photo) is reported as FAIL without skipping the rest."""
    lines = []
    fmt = lambda e: "-" if e is None else f"{e:.2f}"  # noqa: E731
    for camera_id in camera_ids:
        try:
            path, std_err, skin_err = render_point_overlay_frame(actor_id, expression_id, camera_id)
        except Exception as exc:  # per-camera, like run_pipeline.py's overlay loop
            lines.append(f"FAIL {actor_id}/{expression_id} cam {camera_id}: {exc!r}")
            continue
        lines.append(f"OK   {actor_id}/{expression_id} cam {camera_id} -> {path} std_err={fmt(std_err)} skin_err={fmt(skin_err)}")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("actor_id", nargs="?")
    parser.add_argument("expression_id", nargs="?")
    parser.add_argument("camera_ids", nargs="*", default=DEFAULT_CAMERA_IDS)
    parser.add_argument("--backfill", action="store_true",
                        help="Render every unconfirmed/confirmed pair that has no point overlay render yet")
    parser.add_argument("--parallel", type=int, default=8, help="Worker processes for --backfill (default 8)")
    parser.add_argument("--force", action="store_true",
                        help="--backfill only: re-render pairs that already have point overlay renders")
    args = parser.parse_args()
    camera_ids = args.camera_ids or DEFAULT_CAMERA_IDS

    if args.backfill:
        pairs = backfill_pairs(camera_ids, force=args.force)
        print(f"--backfill: {len(pairs)} pair(s) to render")
        failed = 0
        with ProcessPoolExecutor(max_workers=max(1, args.parallel)) as pool:
            futures = {pool.submit(_render_pair, a, e, camera_ids): (a, e) for a, e in pairs}
            for future in as_completed(futures):
                a, e = futures[future]
                try:
                    lines = future.result()
                except Exception as exc:  # keep going; report at the end
                    lines = [f"FAIL {a}/{e}: {exc!r}"]
                failed += sum(line.startswith("FAIL") for line in lines)
                print("\n".join(lines), flush=True)
        print(f"Done. pairs={len(pairs)} failed_cameras={failed}")
        return 0 if failed == 0 else 2

    if not (args.actor_id and args.expression_id):
        parser.error("actor_id and expression_id are required unless --backfill is given")
    print("\n".join(_render_pair(args.actor_id, args.expression_id, camera_ids)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
