#!/usr/bin/env python3
"""FaceScape counterpart to wrap_script/render_landmarks.py -- draws landmark
points (not the mesh-keyline overlay render_overlay_frame.py/
render_overlay_comparison.py already do) on FACESCAPE_OVERLAY_FRAME_CAMERA_IDS'
own three cameras (the same ones the frontend's Overlay Frame tab uses --
see frontend/src/components/MultiviewPlayer.tsx's own constant of that name),
for every expression of one actor, one panel per expression, all written as
consecutive frames of a single mp4 (there's no time axis in FaceScape's
static-multiview data, so "expression" stands in for render_landmarks.py's
own "frame" axis here).

Reuses wrap_script/render_landmarks.py's own color constants/draw helpers
directly (INDEX_TO_COLOR, SKIN_COLOR, STANDARD_GROUP_COLORS,
PROPAGATED_STANDARD_INDICES, draw_point, make_row, ...) via an explicit-path
import (this directory's own established trick for importing wrap_script/
modules, since a plain sys.path import would risk picking up a shadowed
copy) -- same "don't duplicate colors/whitelist logic a second time" reasoning
as run_neutral_skin_propagation*.py's own re-use of wrap_script/skin_landmarks.py
and wrap_script/triangulate.py.

Usage:
    python3 render_landmarks_facescape.py <ACTOR> [--show-skin-landmarks] [--show-standard-groups]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import facescape_paths as fp  # all paths: wrap_script_facescape/.env

fp.add_faceview_to_sys_path()
from backend.app.core.config import get_settings  # noqa: E402


def _load_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_rl = _load_module("wrap_script_render_landmarks", str(fp.WRAP_SCRIPT_ROOT / "render_landmarks.py"))
# draw_point() reads POINT_RADIUS_PX as a module-level global at call time
# (not a parameter), so overwriting it on the loaded module object after
# import affects every draw_point() call made through _rl from here on --
# sized down 4px from Nersemble's own default (per request: FaceScape's
# points read too big), without touching render_landmarks.py itself/its own
# Nersemble renders.
_rl.POINT_RADIUS_PX = _rl.POINT_RADIUS_PX - 4

# Cameras wrap_script_facescape/run_pipeline.py always renders the Overlay
# Frame keyline renders from -- matches the frontend's own
# FACESCAPE_OVERLAY_FRAME_CAMERA_IDS constant exactly, per request.
CAMERA_IDS = ["1", "4", "17"]

# FaceScape photos are much larger than, and not all the same size as (see
# render_overlay_frame.py's own comment on cameras 1/4 vs 14/17), Nersemble's
# fixed delegation-video frame -- resized to a common width for a
# consistent, reasonably-sized panel, same TARGET_WIDTH render_overlay_frame.py
# already settled on for the same reason.
TARGET_WIDTH = 724

OUTPUT_ROOT = fp.FACESCAPE_OUTPUT_ROOT


def load_photo(images_root: Path, actor_id: str, expression_id: str, camera_id: str) -> np.ndarray | None:
    path = images_root / actor_id / expression_id / f"{camera_id}.jpg"
    if not path.exists():
        return None
    image = cv2.imread(str(path))
    if image is None:
        return None
    h, w = image.shape[:2]
    if w != TARGET_WIDTH:
        new_h = int(round(h * (TARGET_WIDTH / w)))
        image = cv2.resize(image, (TARGET_WIDTH, new_h), interpolation=cv2.INTER_AREA)
    return image


def add_expression_label(image: np.ndarray, expression_id: str) -> np.ndarray:
    """Same styling as render_landmarks.py's own add_frame_index_label, just
    labeling the expression name instead of a frame index."""
    label = expression_id
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.7, image.shape[1] / 2400.0)
    thickness = max(2, int(round(font_scale * 2)))
    margin = max(12, int(round(image.shape[0] * 0.01)))

    (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    x = max(0, (image.shape[1] - text_width) // 2)
    y = max(text_height + baseline + 1, image.shape[0] - margin)

    cv2.putText(image, label, (x, y), font, font_scale, (0, 0, 0), thickness + 2, lineType=cv2.LINE_AA)
    cv2.putText(image, label, (x, y), font, font_scale, (255, 255, 255), thickness, lineType=cv2.LINE_AA)
    return image


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("actor_id", help="FaceScape actor identifier")
    parser.add_argument(
        "--show-skin-landmarks", action="store_true",
        help="Also overlay the propagated skin landmarks (index >= 74, skin_frame_000000.json).",
    )
    parser.add_argument(
        "--show-standard-groups", action="store_true",
        help=(
            "Overlay the propagated standard-index groups (ear/nosebridge/undernose/eyebrow/chin -- "
            "<group>_frame_000000.json) IN PLACE OF those same indices' entries in the base whitelist "
            "overlay -- same behavior as wrap_script/render_landmarks.py's own flag of this name."
        ),
    )
    parser.add_argument("--fps", type=float, default=1.0, help="Frames (expressions) per second in the output video (default: 1.0)")
    args = parser.parse_args()

    settings = get_settings()
    actor_dir = settings.facescape_images_root / args.actor_id
    if not actor_dir.is_dir():
        print(f"No such actor under {settings.facescape_images_root}: {args.actor_id}")
        return 1

    expressions = sorted(
        p.name for p in actor_dir.iterdir()
        if p.is_dir() and (p / "params.json").exists()
    )
    if not expressions:
        print(f"No expressions with params.json found for actor {args.actor_id}")
        return 1
    print(f"Rendering {len(expressions)} expression(s) for actor {args.actor_id}: {expressions}")

    whitelist = _rl.load_whitelist()

    output_dir = OUTPUT_ROOT / args.actor_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "landmarks_render.mp4"
    temp_output_path = output_path.with_name(output_path.stem + "_tmp.mp4")

    writer: cv2.VideoWriter | None = None
    written_frames = 0
    try:
        for expression_id in expressions:
            panels: list[np.ndarray] = []
            frame_ok = True
            for cam_id in CAMERA_IDS:
                image = load_photo(settings.facescape_images_root, args.actor_id, expression_id, cam_id)
                if image is None:
                    print(f"SKIP expression={expression_id}: missing photo for camera {cam_id}")
                    frame_ok = False
                    break
                camera_dir = settings.facescape_landmarks_root / args.actor_id / expression_id / f"cam_{cam_id}"

                for point in _rl.load_points(camera_dir / "frame_000000.json"):
                    index = point.get("index")
                    color = _rl.INDEX_TO_COLOR.get(index)
                    if index not in whitelist or color is None:
                        continue
                    if args.show_standard_groups and index in _rl.PROPAGATED_STANDARD_INDICES:
                        continue
                    _rl.draw_point(image, point["x"], point["y"], color, _rl.LANDMARK_ALPHA)

                if args.show_skin_landmarks:
                    for point in _rl.load_points(camera_dir / "skin_frame_000000.json"):
                        _rl.draw_point(image, point["x"], point["y"], _rl.SKIN_COLOR, _rl.SKIN_ALPHA)

                if args.show_standard_groups:
                    for group_name, color in _rl.STANDARD_GROUP_COLORS.items():
                        for point in _rl.load_points(camera_dir / f"{group_name}_frame_000000.json"):
                            _rl.draw_point(image, point["x"], point["y"], color, _rl.STANDARD_GROUP_ALPHA)

                _rl.add_camera_label(image, cam_id)
                panels.append(image)

            if not frame_ok:
                continue

            row = _rl.make_row(panels)
            add_expression_label(row, expression_id)

            if writer is None:
                h, w = row.shape[:2]
                writer = cv2.VideoWriter(str(temp_output_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
            writer.write(row)
            written_frames += 1
    finally:
        if writer is not None:
            writer.release()

    if written_frames == 0:
        print("No frames were rendered.")
        temp_output_path.unlink(missing_ok=True)
        return 1

    _rl.finalize_windows_compatible_mp4(temp_output_path, output_path)
    print(f"Wrote {written_frames} expression panel(s) to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
