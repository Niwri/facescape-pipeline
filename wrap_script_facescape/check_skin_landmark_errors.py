#!/usr/bin/env python3
"""Computes the mean skin-landmark pixel error (annotated points vs. the
wrapped mesh's own landmark projection) across every available camera, for
every "confirmed" FaceScape actor/expression pair, and flags pairs whose
mean exceeds a threshold -- print-only, for manual review on the frontend.

Reuses render_point_overlay_frame.py's own error math
(point_overlay.mesh_landmarks_world/project_mesh_landmarks/
annotated_points_px) but skips the actual image read/resize/draw/write --
this only needs the numeric errors, not a rendered .jpg, so it's much
cheaper to run than render_point_overlay_frame.py --backfill would be. Only
needs each camera image's dimensions (via PIL, which doesn't decode pixel
data), not its pixels.

Works for both valid and invalid pairs: any "confirmed" (actor, expression)
with a wrapped_mesh.obj under facescape_output_root qualifies -- valid pairs
got theirs from Phase 4 (run_pipeline_batch.py), invalid ones from
run_pipeline_invalid.py, but both write to the same
facescape_output_root/<actor>/<expression>/wrapped_mesh.obj path, and
render_point_overlay_frame.py's own _images_root_for() (reused here) already
picks the right images root (raw vs. origin-space) per pair depending on
invalid_pairs.json membership. Skin points aren't whitelist-filtered in
annotated_points_px() (only standard 0-73 points are), so this doesn't need
FaceView/whitelist.json at all.

Usage:
    python3 check_skin_landmark_errors.py [--threshold 4.0] [--parallel 8]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image
import facescape_paths as fp  # all paths: wrap_script_facescape/.env

fp.add_faceview_to_sys_path()
from backend.app.core.config import get_settings  # noqa: E402

_rpo_spec = importlib.util.spec_from_file_location(
    "wrap_script_facescape_render_point_overlay_frame",
    str(fp.FACESCAPE_PIPELINE_DIR / "render_point_overlay_frame.py"),
)
rpo = importlib.util.module_from_spec(_rpo_spec)
_rpo_spec.loader.exec_module(rpo)
po = rpo.po  # point_overlay module, already loaded as a side effect above

DEFAULT_THRESHOLD_PX = 4.0


def discover_pairs(settings) -> list[tuple[str, str]]:
    """Every (actor, expression) currently "confirmed" in label_tracker.json
    with a wrapped_mesh.obj on disk -- covers valid and invalid pairs alike
    (both write to the same facescape_output_root/<actor>/<expression>/
    wrapped_mesh.obj path, see module docstring), but only ones a reviewer
    has actually signed off on."""
    with open(settings.label_tracker_path) as f:
        tracker = json.load(f).get("FaceScape", {})
    pairs = []
    for actor_id, expressions in sorted(tracker.items()):
        for expression_id, status in sorted(expressions.items()):
            if status != "confirmed":
                continue
            if (settings.facescape_output_root / actor_id / expression_id / "wrapped_mesh.obj").exists():
                pairs.append((actor_id, expression_id))
    return pairs


def discover_cameras(settings, actor_id: str, expression_id: str) -> list[str]:
    """Camera ids with skin landmark data for this pair: cam_<id> dirs under
    facescape_landmarks_root with a skin_frame_000000.json -- no skin points,
    nothing this script can measure for that camera."""
    pair_landmarks_dir = settings.facescape_landmarks_root / actor_id / expression_id
    if not pair_landmarks_dir.is_dir():
        return []
    cameras = []
    for cam_dir in sorted(pair_landmarks_dir.iterdir()):
        if (cam_dir.is_dir() and cam_dir.name.startswith("cam_")
                and (cam_dir / "skin_frame_000000.json").exists()):
            cameras.append(cam_dir.name[len("cam_"):])
    return cameras


def mean_skin_error_for_camera(
    settings, actor_id: str, expression_id: str, camera_id: str, mesh_world: dict[int, np.ndarray],
) -> float | None:
    """Same math as render_point_overlay_frame.render_point_overlay_frame(),
    minus the actual image read/resize/draw/write -- only needs the image's
    dimensions (for the same TARGET_WIDTH pixel-space normalization it
    uses), not its pixel data."""
    images_root = rpo._images_root_for(settings, actor_id, expression_id)
    params_path = images_root / actor_id / expression_id / "params.json"
    image_path = images_root / actor_id / expression_id / f"{camera_id}.jpg"
    if not params_path.exists() or not image_path.exists():
        return None

    with open(params_path) as f:
        params = json.load(f)
    k_key, rt_key = f"{camera_id}_K", f"{camera_id}_Rt"
    if k_key not in params or rt_key not in params:
        return None
    K = np.array(params[k_key], dtype=np.float64)
    Rt = np.array(params[rt_key], dtype=np.float64)

    with Image.open(image_path) as img:
        orig_width, orig_height = img.size
    scale = rpo.TARGET_WIDTH / orig_width
    width, height = rpo.TARGET_WIDTH, round(orig_height * scale)
    K = K.copy()
    K[:2] *= scale

    mesh_px = po.project_mesh_landmarks(mesh_world, K, Rt)

    cam_dir = settings.facescape_landmarks_root / actor_id / expression_id / f"cam_{camera_id}"
    # Empty visible_indices: skin points (>=74) aren't gated by it anyway
    # (see annotated_points_px()'s own docstring/code), and standard points
    # (<74) are irrelevant here -- we only want skin_indices below.
    annotated = po.annotated_points_px(cam_dir, "000000", width, height, set())

    skin_indices = sorted(
        i for i in annotated
        if i >= po.NUM_STANDARD_LANDMARKS and i in mesh_px and np.all(np.isfinite(mesh_px[i]))
    )
    if not skin_indices:
        return None
    errors = [float(np.linalg.norm(annotated[i] - mesh_px[i])) for i in skin_indices]
    return float(np.mean(errors))


def process_pair(
    actor_id: str, expression_id: str, pot_rows: list[list[float]], skin_vertex_by_index: dict[int, int],
) -> tuple[str, str, float | None, int]:
    """Returns (actor_id, expression_id, mean skin error across this pair's
    available cameras, or None if no camera had any skin points to compare,
    number of cameras that contributed).

    pot_rows/skin_vertex_by_index are pair-independent (the same POT file and
    skin region assignment for every pair) -- computed ONCE by the caller
    and passed in, not recomputed per call. That's not just an efficiency
    fix: po.load_mesh_landmark_spec() internally does a global
    contextlib.redirect_stdout(...) to suppress a debug print, which is a
    process-wide mutation of sys.stdout -- calling it concurrently from many
    threads (as process_pair() used to, when it called this itself) races on
    that global state and can leave sys.stdout silently pointed at a dead
    buffer, swallowing ALL output process-wide with no error raised."""
    settings = get_settings()
    wrapped_mesh_path = settings.facescape_output_root / actor_id / expression_id / "wrapped_mesh.obj"
    try:
        mesh = trimesh.load(str(wrapped_mesh_path), process=False)
    except Exception:
        return (actor_id, expression_id, None, 0)

    mesh_world = po.mesh_landmarks_world(
        np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces), pot_rows, skin_vertex_by_index,
    )

    per_camera_errors = []
    for camera_id in discover_cameras(settings, actor_id, expression_id):
        try:
            err = mean_skin_error_for_camera(settings, actor_id, expression_id, camera_id, mesh_world)
        except Exception:
            continue
        if err is not None:
            per_camera_errors.append(err)

    if not per_camera_errors:
        return (actor_id, expression_id, None, 0)
    return (actor_id, expression_id, float(np.mean(per_camera_errors)), len(per_camera_errors))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD_PX,
                        help=f"Mean skin landmark pixel error above which a pair is flagged (default: {DEFAULT_THRESHOLD_PX})")
    parser.add_argument("--parallel", type=int, default=64,
                        help="Worker threads (default: 64 -- this is I/O-bound on NFS reads, not CPU-bound, "
                             "so high thread concurrency is appropriate; see main()'s own note)")
    parser.add_argument("--actor-min", type=int, default=None, help="Optional lowest actor id (inclusive)")
    parser.add_argument("--actor-max", type=int, default=None, help="Optional highest actor id (inclusive)")
    args = parser.parse_args()

    settings = get_settings()
    pairs = discover_pairs(settings)
    if args.actor_min is not None or args.actor_max is not None:
        lo = args.actor_min if args.actor_min is not None else -1
        hi = args.actor_max if args.actor_max is not None else float("inf")
        pairs = [(a, e) for a, e in pairs if lo <= int(a) <= hi]
    print(f"Found {len(pairs)} pair(s) with a wrapped_mesh.obj to check")

    # Computed once, up front -- pair-independent (same POT file/skin region
    # assignment every time), and load_mesh_landmark_spec() internally does a
    # global contextlib.redirect_stdout(...) that is NOT safe to call
    # concurrently from multiple threads (see process_pair()'s own
    # docstring). Calling it here, once, before the pool starts, avoids that
    # race entirely and saves 6000+ redundant recomputations.
    pot_rows, skin_vertex_by_index = po.load_mesh_landmark_spec(rpo.FACESCAPE_POT_PATH)

    flagged: list[tuple[str, str, float, int]] = []
    no_data = 0
    failed = 0
    # Threads, not processes: each pair's work here is almost entirely NFS
    # I/O wait (params.json/image/landmark-file reads) with a tiny amount of
    # numpy math, not CPU-bound -- confirmed by observing ~2.6% CPU per
    # worker under an earlier ProcessPoolExecutor(mp_context="spawn")
    # version. Threads release the GIL during blocking file I/O, so many
    # concurrent threads can actually overlap that I/O wait, and unlike
    # processes they need no per-worker interpreter-startup/reimport cost
    # (spawn was paying ~2.4s per worker just to start) -- both matter a lot
    # under a tight time budget. Safe now that the one genuinely shared,
    # non-thread-safe piece of state (see above) is computed once outside
    # the pool: trimesh.load()/PIL.Image.open() read independent files per
    # call, and point_overlay.py's only OTHER module-level mutable-state
    # manipulation (_load_render_landmarks_colors()'s sys.modules stubbing)
    # already ran once at import time, before any thread starts.
    done = 0
    with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as pool:
        futures = {
            pool.submit(process_pair, a, e, pot_rows, skin_vertex_by_index): (a, e)
            for a, e in pairs
        }
        for future in as_completed(futures):
            a, e = futures[future]
            done += 1
            if done % 100 == 0 or done == len(pairs):
                print(f"... {done}/{len(pairs)} pairs processed", flush=True)
            try:
                actor_id, expression_id, mean_err, n_cams = future.result()
            except Exception as exc:
                print(f"FAIL {a}/{e}: {exc!r}", flush=True)
                failed += 1
                continue
            if mean_err is None:
                no_data += 1
                continue
            if mean_err > args.threshold:
                flagged.append((actor_id, expression_id, mean_err, n_cams))
                # Printed the moment this pair crosses the threshold, not
                # batched to the end -- lets you start reviewing on the
                # frontend while the sweep is still running.
                print(f"FLAG {actor_id}/{expression_id}: {mean_err:.2f}px (across {n_cams} camera(s))", flush=True)

    flagged.sort(key=lambda row: row[2], reverse=True)
    print(f"\n{len(flagged)} pair(s) exceed {args.threshold}px mean skin landmark error "
          f"({no_data} pair(s) had no skin data to measure, {failed} failed outright):\n")
    for actor_id, expression_id, mean_err, n_cams in flagged:
        print(f"{actor_id}/{expression_id}: {mean_err:.2f}px (across {n_cams} camera(s))")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
