#!/usr/bin/env python3
"""Render the landmark "point overlay" video for one Nersemble actor/sequence:
the annotated (propagated) landmarks vs. the wrapped mesh's own landmarks,
drawn over the delegation video -- see point_overlay.py for what each point
is and how it's styled. Frame walk, calibration scaling and missing-frame
placeholders mirror test_render_gt.py (the keyline Overlay Video render), so
the two videos line up frame for frame. Landmarks the scan-mask filter
dropped for a frame (scan_face_masking.py; not used by that frame's wrap) are
left out of it, and out of the error means. Frames whose wrap recorded
wrap_landmark_indices.json (run_pipeline_all.run_pipeline) show exactly the
landmarks that wrap was fit to -- whitelist minus untriangulated, FLAME-mask
and scan-mask exclusions.

Output: /scratch/ondemand32/irwinngo/point_overlay_renders/<actor>_<sequence>_<camera>.mp4,
served to FaceView's "Overlay Landmarks" tab by /labels/point-overlay-video.

Usage (from wrap_script/):
    python render_point_overlay.py <actor_id> <sequence_id> [camera_id]
    python render_point_overlay.py --backfill [--parallel N]
        # every "unconfirmed"/"confirmed" pair in label_tracker.json that has
        # no point overlay video yet
"""
from __future__ import annotations

import nersemble_paths
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import imageio
import numpy as np
import trimesh

import point_overlay as po
from scan_face_masking import load_scan_dropped_indices
from test_render_gt import FRAME_STEP, PANEL_CAMERAS, label_panel, make_missing_frame

OUTPUT_ROOT = nersemble_paths.OUTPUT_ROOT
PROCESSED_ROOT = nersemble_paths.PROCESSED_ROOT
DATA_ROOT = nersemble_paths.DATA_ROOT
LANDMARKS_ROOT = nersemble_paths.LANDMARKS_ROOT
LABEL_TRACKER_PATH = nersemble_paths.LABEL_TRACKER_PATH
POINT_OVERLAY_RENDERS_ROOT = nersemble_paths.POINT_OVERLAY_RENDERS_ROOT

# Camera for the Overlay Landmarks video (same value as
# run_pipeline_batch_all.py's POINT_OVERLAY_VIDEO_CAMERA_ID and FaceView's
# labels.py POINT_OVERLAY_VIDEO_CAMERA_ID). Differs from the keyline Overlay
# Video's camera (222200046) by request.
DEFAULT_CAMERA_ID = "222200037"
DELEGATION_WIDTH, DELEGATION_HEIGHT = 550, 802


def output_path_for(actor_id: str, sequence_id: str, camera_id: str) -> Path:
    return POINT_OVERLAY_RENDERS_ROOT / f"{actor_id}_{sequence_id}_{camera_id}.mp4"


def _panel_camera(actor_id: str, sequence_id: str, camera_id: str, param: dict) -> dict | None:
    """Delegation-scaled K / Rt, delegation video reader and landmark dir for
    one panel camera, or None when its videos or calibration are missing."""
    raw_cap = cv2.VideoCapture(str(DATA_ROOT / actor_id / "sequences" / sequence_id / "images" / f"cam_{camera_id}.mp4"))
    cap = cv2.VideoCapture(str(PROCESSED_ROOT / actor_id / sequence_id / f"cam_{camera_id}_delegation.mp4"))
    if not raw_cap.isOpened() or not cap.isOpened() or camera_id not in param["world_2_cam"]:
        print(f"NOTE: camera {camera_id} has no usable raw/delegation video or calibration -- placeholder panels")
        raw_cap.release(); cap.release()
        return None
    orig_width = int(raw_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(raw_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    raw_cap.release()
    # Intrinsics are calibrated for the raw video; the delegation video is a
    # fixed 550x802 resize -- scale x and y independently (test_render_gt.py).
    K = np.array(param["intrinsics"], dtype=np.float64).copy()
    K[0, 0] *= DELEGATION_WIDTH / orig_width
    K[0, 2] *= DELEGATION_WIDTH / orig_width
    K[1, 1] *= DELEGATION_HEIGHT / orig_height
    K[1, 2] *= DELEGATION_HEIGHT / orig_height
    return {"id": camera_id, "K": K, "Rt": np.array(param["world_2_cam"][camera_id], dtype=np.float64), "cap": cap,
            "delegation_idx": -1, "bg": None, "cam_dir": LANDMARKS_ROOT / actor_id / sequence_id / f"cam_{camera_id}"}


def _panel_frame(cam: dict, target_idx: int):
    """Delegation frame target_idx, read sequentially (seeking is unreliable)."""
    while cam["delegation_idx"] < target_idx:
        ret, cam["bg"] = cam["cap"].read()
        cam["delegation_idx"] += 1
        if not ret:
            cam["bg"] = None
            break
    return cam["bg"]


def render_point_overlay_video(
    actor_id: str, sequence_id: str, camera_id: str = DEFAULT_CAMERA_ID, single: bool = False,
) -> tuple[Path, float | None, float | None]:
    """Writes the video -- front | left | right panels (test_render_gt.PANEL_CAMERAS),
    or just camera_id with single=True; the output file is named by camera_id
    either way (the path FaceView serves). Returns (path, mean standard-landmark
    pixel error, mean skin-landmark pixel error), over every frame and panel at 550x802."""
    with open(DATA_ROOT / actor_id / "calibration" / "camera_params.json") as f:
        param = json.load(f)
    sequence_output_dir = OUTPUT_ROOT / actor_id / sequence_id
    panels = [(camera_id, "")] if single else list(PANEL_CAMERAS)
    cameras = [(_panel_camera(actor_id, sequence_id, cam, param), cam, view) for cam, view in panels]
    if all(c is None for c, _, _ in cameras):
        raise RuntimeError(f"No usable camera for {actor_id}/{sequence_id}: {[cam for cam, _ in panels]}")

    # Only frames on this actor's wrap grid (every 25 raw frames for actors
    # 102+, see wrap_frame_step.py) -- stray folders off the grid are skipped.
    from wrap_frame_step import wrap_frame_step
    wrap_step = wrap_frame_step(actor_id)
    frame_dirs = sorted(
        (int(p.name), p) for p in sequence_output_dir.iterdir()
        if p.is_dir() and p.name.isdigit() and int(p.name) % wrap_step == 0
    )
    if not frame_dirs:
        raise RuntimeError(f"No frame directories found under {sequence_output_dir}")

    pot_rows, skin_vertex_by_index = po.load_mesh_landmark_spec()
    visible = po.load_whitelist()

    # Delegation fps is raw/FRAME_STEP; one output frame per wrap_step raw frames.
    delegation_fps = next((c["cap"].get(cv2.CAP_PROP_FPS) for c, _, _ in cameras if c is not None), 0.0)
    fps = (delegation_fps or (73.0 / FRAME_STEP)) * FRAME_STEP / wrap_step

    output_path = output_path_for(actor_id, sequence_id, camera_id)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(output_path.stem + ".tmp.mp4")
    writer = imageio.get_writer(str(tmp_path), fps=fps, quality=10, macro_block_size=None)

    errors: dict[int, list[float]] = {}  # index -> per-frame, per-panel errors
    try:
        for frame_num, frame_dir in frame_dirs:
            target_idx = frame_num // FRAME_STEP
            wrapped_mesh_path = frame_dir / "wrapped_mesh.obj"
            mesh_world = used = None
            if wrapped_mesh_path.exists():
                mesh = trimesh.load(str(wrapped_mesh_path), process=False)
                mesh_world = po.mesh_landmarks_world(
                    np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces), pot_rows, skin_vertex_by_index,
                )
                used_path = frame_dir / "wrap_landmark_indices.json"
                used = set(json.loads(used_path.read_text())) if used_path.exists() else None
                scan_dropped = set(load_scan_dropped_indices(LANDMARKS_ROOT, actor_id, sequence_id, frame_num) or []) if used is None else set()
            tiles = []
            for cam, cam_id, view in cameras:
                bg_frame = _panel_frame(cam, target_idx) if cam is not None else None
                if mesh_world is None or bg_frame is None:
                    tile = make_missing_frame(DELEGATION_WIDTH, DELEGATION_HEIGHT, frame_num)
                else:
                    mesh_px = po.project_mesh_landmarks(mesh_world, cam["K"], cam["Rt"])
                    annotated = po.annotated_points_px(cam["cam_dir"], f"{target_idx:06d}", DELEGATION_WIDTH, DELEGATION_HEIGHT, visible)
                    if used is not None:  # exactly the landmarks that frame's wrap was fit to
                        annotated = {i: p for i, p in annotated.items() if i in used}
                    else:  # wrapped before the index record existed: drop scan-filtered ones
                        annotated = {i: p for i, p in annotated.items() if i not in scan_dropped}
                    tile, frame_errors = po.draw_point_overlay(bg_frame, annotated, mesh_px)
                    for idx, err in frame_errors.items():
                        errors.setdefault(idx, []).append(err)
                if view:
                    label_panel(tile, f"cam {cam_id} ({view}) | frame {frame_num}")
                else:
                    cv2.putText(tile, str(frame_num), (DELEGATION_WIDTH - 60, DELEGATION_HEIGHT - 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                tiles.append(tile)
            writer.append_data(cv2.cvtColor(np.hstack(tiles), cv2.COLOR_BGR2RGB))
    finally:
        for cam, _, _ in cameras:
            if cam is not None:
                cam["cap"].release()
        writer.close()
    tmp_path.replace(output_path)
    std = [e for i, errs in errors.items() if i < po.NUM_STANDARD_LANDMARKS for e in errs]
    skin = [e for i, errs in errors.items() if i >= po.NUM_STANDARD_LANDMARKS for e in errs]
    return output_path, (float(np.mean(std)) if std else None), (float(np.mean(skin)) if skin else None)


def backfill_pairs(camera_id: str) -> list[tuple[str, str]]:
    with open(LABEL_TRACKER_PATH) as f:
        data = json.load(f).get("Nersemble", {})
    pairs = []
    for actor_id, sequences in sorted(data.items()):
        for sequence_id, status in sorted(sequences.items()):
            if status not in ("unconfirmed", "confirmed"):
                continue
            if output_path_for(actor_id, sequence_id, camera_id).exists():
                continue
            if (OUTPUT_ROOT / actor_id / sequence_id).is_dir():
                pairs.append((actor_id, sequence_id))
    return pairs


def _render_one(actor_id: str, sequence_id: str, camera_id: str) -> str:
    path, std_err, skin_err = render_point_overlay_video(actor_id, sequence_id, camera_id)
    return f"OK   {actor_id}/{sequence_id} -> {path} {_fmt_errors(std_err, skin_err)}"


def _fmt_errors(std_err: float | None, skin_err: float | None) -> str:
    fmt = lambda e: "-" if e is None else f"{e:.2f}"  # noqa: E731
    return f"std_err={fmt(std_err)} skin_err={fmt(skin_err)}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("actor_id", nargs="?")
    parser.add_argument("sequence_id", nargs="?")
    parser.add_argument("camera_id", nargs="?", default=DEFAULT_CAMERA_ID)
    parser.add_argument("--backfill", action="store_true",
                        help="Render every unconfirmed/confirmed pair that has no point overlay video yet")
    parser.add_argument("--parallel", type=int, default=4, help="Worker processes for --backfill (default 4)")
    args = parser.parse_args()

    if args.backfill:
        pairs = backfill_pairs(args.camera_id)
        print(f"--backfill: {len(pairs)} pair(s) to render")
        failed = 0
        with ProcessPoolExecutor(max_workers=max(1, args.parallel)) as pool:
            futures = {pool.submit(_render_one, a, s, args.camera_id): (a, s) for a, s in pairs}
            for future in as_completed(futures):
                a, s = futures[future]
                try:
                    print(future.result(), flush=True)
                except Exception as exc:  # keep going; report at the end
                    failed += 1
                    print(f"FAIL {a}/{s}: {exc}", flush=True)
        print(f"Done. rendered={len(pairs) - failed} failed={failed}")
        return 0 if failed == 0 else 2

    if not (args.actor_id and args.sequence_id):
        parser.error("actor_id and sequence_id are required unless --backfill is given")
    print(_render_one(args.actor_id, args.sequence_id, args.camera_id))
    return 0


if __name__ == "__main__":
    sys.exit(main())
