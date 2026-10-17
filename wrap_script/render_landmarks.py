#!/usr/bin/env python3
"""Render Nersemble landmarks for a given actor/sequence, using the same three
cameras as render_mesh_compare.py (CAMERA_IDS below) laid out horizontally in
one row instead of that script's 3x2 mesh/video grid. Landmark selection
(whitelist.json) and per-region colors mirror
FaceView/frontend/src/components/LandmarkOverlay.tsx exactly, so this is a
visual match for what a reviewer sees in the app.

Run from this directory (wrap_script/), same convention as render_mesh_compare.py:
    python render_landmarks.py <actor_id> <sequence_id> [--show-skin-landmarks] [--show-standard-groups]
"""
from __future__ import annotations

import sys
from pathlib import Path

# Also loaded by file path from wrap_script_facescape/ (render_point_overlay_frame.py,
# render_landmarks_facescape.py), where this folder isn't on sys.path: make nersemble_paths,
# which sits next to this file, importable either way.
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
import nersemble_paths  # noqa: E402
import argparse
import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from backend.app.core.config import get_settings

CAMERA_IDS = ["222200037", "221501007", "222200040"]

# wrap_script/backend is a separate, older copy of FaceView/backend (confirmed
# via diff -- it predates RAW_FRAME_STRIDE being added to config.py), so that
# constant isn't importable from here. Its value (5) is stable and documented
# in FaceView/backend/app/core/config.py: raw video frame r == displayed
# (landmark) frame r * RAW_FRAME_STRIDE.
RAW_FRAME_STRIDE = 5

WHITELIST_PATH = Path(__file__).resolve().parent.parent / "FaceView" / "whitelist.json"

# Mirrors LandmarkOverlay.tsx's region groupings and *_COLOR constants exactly
# (same index lists, same (r, g, b) triples), so region colors match the
# frontend's landmark display.
JAW = [31, 28, 12, 53, 48, 49, 52, 51, 50, 66, 67, 64, 63, 68, 3, 7, 10]
LEFT_EAR = [2, 4, 1]
RIGHT_EAR = [9, 14, 6]
LEFT_BROW = [8, 30, 0, 73, 72]
RIGHT_BROW = [29, 47, 5, 70, 69]
NOSE = [60, 62, 54, 57]
LEFT_EYE_UP = [15, 17]
LEFT_EYE = [71, 16, 13, 11]
RIGHT_EYE_UP = [34, 36]
RIGHT_EYE = [35, 33, 32, 65]
UNDER_NOSE = [38, 41, 61, 19, 22]
RIGHT_MOUTH_CORNER = [37, 43]
LEFT_MOUTH_CORNER = [18, 24]
OUTER_MOUTH_RIGHT = [42, 44]
OUTER_MOUTH_LEFT = [23, 25]
OUTER_MOUTH_MIDDLE_OUTER = [39, 20, 46, 27]
INNER_MOUTH_MIDDLE_OUTER = [40, 21, 45, 26]
OUTER_MOUTH_MIDDLE_MIDDLE = [59, 55]
INNER_MOUTH_MIDDLE_MIDDLE = [56, 58]

REGION_COLORS: list[tuple[list[int], tuple[int, int, int]]] = [
    (JAW, (255, 0, 0)),
    (LEFT_EAR, (0, 255, 0)),
    (RIGHT_EAR, (0, 0, 255)),
    (LEFT_BROW, (255, 165, 0)),
    (RIGHT_BROW, (128, 0, 128)),
    (NOSE, (255, 255, 0)),
    (LEFT_EYE, (0, 255, 255)),
    (RIGHT_EYE, (255, 0, 255)),
    (RIGHT_EYE_UP, (200, 0, 200)),
    (LEFT_EYE_UP, (0, 200, 200)),
    (UNDER_NOSE, (0, 128, 0)),
    (RIGHT_MOUTH_CORNER, (128, 0, 0)),
    (LEFT_MOUTH_CORNER, (0, 0, 128)),
    (OUTER_MOUTH_RIGHT, (255, 105, 180)),
    (OUTER_MOUTH_LEFT, (0, 128, 128)),
    (OUTER_MOUTH_MIDDLE_OUTER, (210, 105, 30)),
    (INNER_MOUTH_MIDDLE_OUTER, (75, 0, 130)),
    (OUTER_MOUTH_MIDDLE_MIDDLE, (154, 205, 50)),
    (INNER_MOUTH_MIDDLE_MIDDLE, (70, 130, 180)),
]
INDEX_TO_COLOR: dict[int, tuple[int, int, int]] = {}
for _indices, _color in REGION_COLORS:
    for _idx in _indices:
        INDEX_TO_COLOR[_idx] = _color

LANDMARK_ALPHA = 0.7
SKIN_COLOR = (148, 163, 184)  # slate -- LandmarkOverlay.tsx's SKIN_POINT_COLOR
SKIN_ALPHA = 0.85
POINT_RADIUS_PX = 8

# The standard-index groups run_neutral_cotracker_propagation(_raft).py
# propagates (see STANDARD_INDEX_REGION_GROUPS there) -- each written to its
# own <prefix>_frame_<idx>.json, never merged with skin_frame or each other.
# Colors are arbitrary but distinct from both REGION_COLORS and SKIN_COLOR so
# a propagated point never gets visually confused with a standard whitelist
# one drawn at the same spot.
STANDARD_GROUP_COLORS: dict[str, tuple[int, int, int]] = {
    "ear": (57, 255, 20),         # neon green
    "nosebridge": (255, 215, 0),  # gold
    "undernose": (255, 69, 0),    # orange-red
    "eyebrow": (30, 144, 255),    # dodger blue
    "chin": (238, 130, 238),      # violet
}
STANDARD_GROUP_ALPHA = 0.85

# Standard 0-73 indices that ALSO have a propagated variant (written by
# STANDARD_INDEX_REGION_GROUPS in run_neutral_cotracker_propagation_raft.py --
# EAR_REGIONS/NOSEBRIDGE_REGIONS/UNDERNOSE_REGIONS/EYEBROW_REGIONS/
# CHIN_REGIONS' own "indices" lists, kept in sync here by hand rather than
# imported -- importing that module would pull in torch/RAFT/CoTracker for a
# handful of constants). When --show-standard-groups is on, these are skipped
# in the base whitelist draw below so each point shows only its propagated
# (tracked) position, not both that and its original per-frame detection at
# the same spot.
PROPAGATED_STANDARD_INDICES: set[int] = {
    2, 4, 1, 9, 14, 6,       # ear (left, right)
    54, 57,                  # nosebridge
    38, 61, 19,              # undernose (right_nostril, center, left_nostril)
    8, 30, 73, 29, 47, 70,   # eyebrow (left, right)
    50,                      # chin
}

OUTPUT_ROOT = nersemble_paths.OUTPUT_ROOT


def load_whitelist() -> set[int]:
    with WHITELIST_PATH.open("r", encoding="utf-8") as f:
        return set(json.load(f))


def load_points(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except (json.JSONDecodeError, OSError):
        return []
    return payload.get("points", [])


def list_available_frame_indices(camera_dir: Path) -> list[int]:
    if not camera_dir.exists():
        return []
    indices = []
    for path in camera_dir.glob("frame_*.json"):
        try:
            indices.append(int(path.stem.removeprefix("frame_")))
        except ValueError:
            continue
    return sorted(indices)


def draw_point(image: np.ndarray, x_norm: float, y_norm: float, rgb: tuple[int, int, int], alpha: float) -> None:
    h, w = image.shape[:2]
    x, y = int(round(x_norm * w)), int(round(y_norm * h))
    if x < 0 or y < 0 or x >= w or y >= h:
        return
    bgr = (rgb[2], rgb[1], rgb[0])
    overlay = image.copy()
    cv2.circle(overlay, (x, y), POINT_RADIUS_PX, bgr, thickness=-1, lineType=cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, dst=image)


def add_camera_label(image: np.ndarray, camera_id: str) -> np.ndarray:
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.6, image.shape[1] / 1600.0)
    thickness = max(1, int(round(font_scale * 2)))
    padding = max(8, int(image.shape[1] * 0.01))
    label = f"cam {camera_id}"
    x, y = padding, padding + int(font_scale * 22)
    cv2.putText(image, label, (x, y), font, font_scale, (0, 0, 0), thickness + 2, lineType=cv2.LINE_AA)
    cv2.putText(image, label, (x, y), font, font_scale, (255, 255, 255), thickness, lineType=cv2.LINE_AA)
    return image


def add_frame_index_label(image: np.ndarray, frame_idx: int) -> np.ndarray:
    label = f"Frame {frame_idx}"
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


def make_row(panels: list[np.ndarray]) -> np.ndarray:
    target_h = panels[0].shape[0]
    resized = []
    for panel in panels:
        h, w = panel.shape[:2]
        if h != target_h:
            new_w = int(round(w * (target_h / h)))
            panel = cv2.resize(panel, (new_w, target_h), interpolation=cv2.INTER_AREA)
        resized.append(panel)
    row = np.concatenate(resized, axis=1)
    if row.shape[0] % 2 != 0:
        row = row[: row.shape[0] - 1]
    if row.shape[1] % 2 != 0:
        row = row[:, : row.shape[1] - 1]
    return row


def finalize_windows_compatible_mp4(temp_output_path: Path, output_path: Path) -> None:
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        if output_path.exists():
            output_path.unlink()
        temp_output_path.rename(output_path)
        print("ffmpeg not found; kept OpenCV mp4v output.")
        return

    cmd = [
        ffmpeg_bin, "-y", "-i", str(temp_output_path),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(output_path),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as exc:
        if output_path.exists():
            output_path.unlink()
        temp_output_path.rename(output_path)
        print("ffmpeg conversion failed; kept OpenCV mp4v output.")
        stderr_text = exc.stderr.decode("utf-8", errors="ignore")
        if stderr_text:
            print(stderr_text)
        return
    temp_output_path.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Render a horizontal 1x3 landmark overlay video for the given actor/sequence, "
            "using the same three cameras as render_mesh_compare.py."
        )
    )
    parser.add_argument("actor_id", help="Actor identifier")
    parser.add_argument("sequence_id", help="Sequence identifier")
    parser.add_argument(
        "--show-skin-landmarks",
        action="store_true",
        help="Also overlay the propagated skin landmarks (index >= 74, skin_frame_*.json) alongside the standard whitelist landmarks.",
    )
    parser.add_argument(
        "--show-standard-groups",
        action="store_true",
        help=(
            "Overlay the propagated standard-index groups (ear/nosebridge/undernose/eyebrow/chin -- "
            "<group>_frame_*.json), each in its own distinct color (see STANDARD_GROUP_COLORS), IN PLACE "
            "OF those same indices' entries in the base whitelist overlay (per request -- see "
            "PROPAGATED_STANDARD_INDICES) -- each point shows only its propagated (tracked) position, "
            "not both that and its original per-frame sapiens2 detection at the same spot."
        ),
    )
    args = parser.parse_args()

    actor_id = args.actor_id
    sequence_id = args.sequence_id

    settings = get_settings()
    whitelist = load_whitelist()

    video_paths: dict[str, Path] = {}
    for cam_id in CAMERA_IDS:
        vp = settings.data_root / actor_id / "sequences" / sequence_id / "images" / f"cam_{cam_id}.mp4"
        if not vp.exists():
            print(f"Camera video not found: {vp}")
            return 1
        video_paths[cam_id] = vp

    camera_dirs = {
        cam_id: settings.landmarks_root / actor_id / sequence_id / f"cam_{cam_id}" for cam_id in CAMERA_IDS
    }

    # A frame can only appear in the row once every one of the three cameras
    # has a landmark file for it -- landmarks are written per camera.
    frame_indices: set[int] | None = None
    for cam_id in CAMERA_IDS:
        available = set(list_available_frame_indices(camera_dirs[cam_id]))
        frame_indices = available if frame_indices is None else (frame_indices & available)
    sorted_frame_indices = sorted(frame_indices or [])
    if not sorted_frame_indices:
        print(f"No frame indices with landmarks available across all three cameras for {actor_id}/{sequence_id}")
        return 1

    caps: dict[str, cv2.VideoCapture] = {}
    for cam_id in CAMERA_IDS:
        cap = cv2.VideoCapture(str(video_paths[cam_id]))
        if not cap.isOpened():
            print(f"Failed to open video: {video_paths[cam_id]}")
            return 1
        caps[cam_id] = cap

    raw_fps = caps[CAMERA_IDS[0]].get(cv2.CAP_PROP_FPS) or 73.0
    displayed_fps = (raw_fps / RAW_FRAME_STRIDE) if raw_fps > 0 else (73.0 / RAW_FRAME_STRIDE)

    output_path = OUTPUT_ROOT / actor_id / sequence_id / "landmarks_render.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_output_path = output_path.with_name(f"{output_path.stem}_tmp.mp4")

    writer: cv2.VideoWriter | None = None
    written_frames = 0

    try:
        for frame_idx in tqdm(sorted_frame_indices, desc="Rendering landmarks"):
            raw_frame_idx = frame_idx * RAW_FRAME_STRIDE
            panels: list[np.ndarray] = []
            frame_ok = True

            for cam_id in CAMERA_IDS:
                cap = caps[cam_id]
                cap.set(cv2.CAP_PROP_POS_FRAMES, raw_frame_idx)
                success, image = cap.read()
                if not success or image is None:
                    print(f"SKIP frame={frame_idx}: missing camera frame for cam {cam_id}")
                    frame_ok = False
                    break
                image = image.copy()

                for point in load_points(camera_dirs[cam_id] / f"frame_{frame_idx:06d}.json"):
                    index = point.get("index")
                    color = INDEX_TO_COLOR.get(index)
                    if index not in whitelist or color is None:
                        continue
                    if args.show_standard_groups and index in PROPAGATED_STANDARD_INDICES:
                        continue
                    draw_point(image, point["x"], point["y"], color, LANDMARK_ALPHA)

                if args.show_skin_landmarks:
                    for point in load_points(camera_dirs[cam_id] / f"skin_frame_{frame_idx:06d}.json"):
                        draw_point(image, point["x"], point["y"], SKIN_COLOR, SKIN_ALPHA)

                if args.show_standard_groups:
                    for group_name, color in STANDARD_GROUP_COLORS.items():
                        for point in load_points(camera_dirs[cam_id] / f"{group_name}_frame_{frame_idx:06d}.json"):
                            draw_point(image, point["x"], point["y"], color, STANDARD_GROUP_ALPHA)

                add_camera_label(image, cam_id)
                panels.append(image)

            if not frame_ok:
                continue

            row = make_row(panels)
            add_frame_index_label(row, frame_idx)

            if writer is None:
                h, w = row.shape[:2]
                writer = cv2.VideoWriter(str(temp_output_path), cv2.VideoWriter_fourcc(*"mp4v"), displayed_fps, (w, h))

            writer.write(row)
            written_frames += 1
    finally:
        for cap in caps.values():
            cap.release()
        if writer is not None:
            writer.release()

    if written_frames == 0:
        print("No frames were rendered.")
        temp_output_path.unlink(missing_ok=True)
        return 1

    finalize_windows_compatible_mp4(temp_output_path, output_path)
    print(f"Wrote {written_frames} frames to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
