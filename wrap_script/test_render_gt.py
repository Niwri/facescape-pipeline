import nersemble_paths
from flame import *
import argparse
import trimesh
from tqdm import tqdm
from pathlib import Path
import os

import matplotlib.pyplot as plt
import numpy as np

from uv_sampler import UVSampler
import cv2

from typing import Callable, Optional, TYPE_CHECKING, Union
from torch import _VF, sym_int as _sym_int, Tensor
import math
import json

# Delegation videos keep every FRAME_STEPth raw frame, so raw/output frame number
# f corresponds to delegation-video frame f // FRAME_STEP (matches process_sequence*.py).
FRAME_STEP = 5

# wrapped_mesh.obj is a FLAME mesh with the eyeballs (a trailing block of vertex
# indices, per the split in landmark_lines.npy) removed via truncation -- indices
# below this bound map 1:1 onto the full FLAME mesh; anything at or above it no
# longer exists in wrapped_mesh.obj.
WRAPPED_MESH_EYE_CUTOFF = 3931


def make_missing_frame(width: int, height: int, frame_num: int) -> np.ndarray:
    """Black placeholder frame (BGR) with the frame number centered in white --
    written to the overlay video wherever a frame has no wrapped_mesh.obj (or
    the delegation video itself runs out), so a missing/broken frame is visible
    when scrubbing instead of being silently dropped from the video."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    text = str(frame_num)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2.0
    thickness = 3
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x = (width - text_width) // 2
    y = (height + text_height) // 2
    cv2.putText(frame, text, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return frame


# Cameras shown side by side in the overlay video: front | left | right.
# The output file keeps the camera_id argument's name (the path FaceView's
# /labels/overlay-video serves), whatever cameras it shows.
PANEL_CAMERAS = (("222200037", "front"), ("222200040", "left"), ("222200049", "right"))
DELEGATION_WIDTH, DELEGATION_HEIGHT = 550, 802


def label_panel(image: np.ndarray, text: str) -> None:
    cv2.rectangle(image, (0, 0), (image.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(image, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


class DelegationCamera:
    """One panel's camera: delegation-scaled calibration plus a sequential
    reader of its delegation video (compressed-video seeking via
    cv2.CAP_PROP_POS_FRAMES is unreliable, so frames are read in order in
    lockstep with the monotonically increasing frame numbers). A camera
    whose videos can't be opened gives placeholder panels."""

    def __init__(self, data_dir: Path, processed_dir: Path, actor: str, sequence: str, cam_id: str, param: dict):
        self.cam_id = cam_id
        self.cap = None
        self.delegation_idx = -1
        self.bg_frame = None
        raw_cap = cv2.VideoCapture(str(data_dir / actor / "sequences" / sequence / "images" / f"cam_{cam_id}.mp4"))
        cap = cv2.VideoCapture(str(processed_dir / actor / sequence / f"cam_{cam_id}_delegation.mp4"))
        if not raw_cap.isOpened() or not cap.isOpened() or cam_id not in param["world_2_cam"]:
            print(f"NOTE: camera {cam_id} has no usable raw/delegation video or calibration -- placeholder panels")
            raw_cap.release(); cap.release()
            return
        orig_width = int(raw_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_height = int(raw_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        raw_cap.release()
        # camera_params.json's intrinsics are calibrated for the raw video; the
        # delegation video is a fixed 550x802 resize (not aspect-ratio
        # preserving in general), so scale fx/cx and fy/cy independently.
        intri = torch.tensor(param["intrinsics"], dtype=torch.float32).clone()
        intri[0, 0] *= DELEGATION_WIDTH / orig_width
        intri[0, 2] *= DELEGATION_WIDTH / orig_width
        intri[1, 1] *= DELEGATION_HEIGHT / orig_height
        intri[1, 2] *= DELEGATION_HEIGHT / orig_height
        self.intri = intri
        # wrapped_mesh.obj vertices are already in world_2_cam's world space.
        self.extri = torch.tensor(param["world_2_cam"][cam_id], dtype=torch.float32)
        self.cap = cap

    @property
    def fps(self):
        return self.cap.get(cv2.CAP_PROP_FPS) if self.cap is not None else 0.0

    def frame(self, frame_num: int):
        """Delegation-video frame for raw frame frame_num, or None."""
        if self.cap is None:
            return None
        target_idx = frame_num // FRAME_STEP
        while self.delegation_idx < target_idx:
            ret, self.bg_frame = self.cap.read()
            self.delegation_idx += 1
            if not ret:
                self.bg_frame = None
                break
        return self.bg_frame

    def release(self):
        if self.cap is not None:
            self.cap.release()


def main():
    parser = argparse.ArgumentParser(
        description="Render the wrapped-mesh keyline overlay onto the delegation videos for one actor/sequence: "
                    "front | left | right panels (PANEL_CAMERAS), written to line_renders/<actor>_<sequence>_<camera_id>.mp4."
    )
    parser.add_argument("actor_id", help="Actor identifier, e.g. 043")
    parser.add_argument("sequence_id", help="Sequence identifier, e.g. EMO-1-shout+laugh")
    parser.add_argument("camera_id", help="Camera id naming the output file (the path FaceView serves), e.g. 222200046")
    parser.add_argument("--single", action="store_true", help="Old behaviour: one panel, from camera_id itself")
    args = parser.parse_args()

    outputDir = nersemble_paths.OUTPUT_ROOT
    processedDir = nersemble_paths.PROCESSED_ROOT
    dataDir = nersemble_paths.DATA_ROOT
    lineRendersDir = nersemble_paths.LINE_RENDERS_ROOT
    landmark_lines = np.load(str(nersemble_paths.LANDMARK_LINES_PATH))
    # Drop any line segment that references a now-removed eyeball vertex instead
    # of trying to remap it.
    landmark_lines = landmark_lines[(landmark_lines < WRAPPED_MESH_EYE_CUTOFF).all(axis=1)]

    part = args.actor_id
    cat = args.sequence_id
    sequence_output_dir = outputDir / part / cat
    with open(dataDir / part / "calibration" / "camera_params.json", "r") as f:
        param = json.load(f)

    panels = [(args.camera_id, "")] if args.single else list(PANEL_CAMERAS)
    cameras = [(DelegationCamera(dataDir, processedDir, part, cat, cam, param), view) for cam, view in panels]
    if all(c.cap is None for c, _ in cameras):
        raise SystemExit(f"No usable camera for {part}/{cat}: {[cam for cam, _ in panels]}")

    # Output frame directories are named by raw/output frame number. Only
    # frames on this actor's wrap grid (every 25 raw frames for actors 102+,
    # see wrap_frame_step.py) -- stray folders off the grid are skipped.
    from wrap_frame_step import wrap_frame_step
    wrap_step = wrap_frame_step(part)
    frame_dirs = sorted(
        (int(p.name), p) for p in sequence_output_dir.iterdir()
        if p.is_dir() and p.name.isdigit() and int(p.name) % wrap_step == 0
    )
    if not frame_dirs:
        raise SystemExit(f"No frame directories found under {sequence_output_dir}")

    # Delegation fps is raw/FRAME_STEP; one output frame per wrap_step raw frames.
    delegation_fps = next((c.fps for c, _ in cameras if c.fps), 0.0) or (73.0 / FRAME_STEP)
    fps = delegation_fps * FRAME_STEP / wrap_step

    output_path = lineRendersDir / f"{part}_{cat}_{args.camera_id}.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(output_path.stem + ".tmp.mp4")
    writer = imageio.get_writer(str(tmp_path), fps=fps, quality=10, macro_block_size=None)
    try:
        for frame_num, frame_dir in tqdm(frame_dirs, desc=f"{part}/{cat}"):
            wrapped_mesh_path = frame_dir / "wrapped_mesh.obj"
            verts = None
            if wrapped_mesh_path.exists():
                mesh = trimesh.load(str(wrapped_mesh_path), process=False)
                verts = torch.as_tensor(np.asarray(mesh.vertices), dtype=torch.float32)[None]  # (1, V, 3)
            else:
                print(f"Frame {frame_num}: no wrapped_mesh.obj -- placeholder panels")
            tiles = []
            for cam, view in cameras:
                bg_frame = cam.frame(frame_num)
                if verts is None or bg_frame is None:
                    tile = make_missing_frame(DELEGATION_WIDTH, DELEGATION_HEIGHT, frame_num)
                else:
                    tile = bg_frame.copy()
                    verts2d = project_vertices_paired(verts, cam.extri[None], cam.intri[None])[0].numpy()
                    draw_lines_on_image(tile, verts2d, landmark_lines)
                if view:
                    label_panel(tile, f"cam {cam.cam_id} ({view}) | frame {frame_num}")
                tiles.append(tile)
            writer.append_data(cv2.cvtColor(np.hstack(tiles), cv2.COLOR_BGR2RGB))
    finally:
        writer.close()
        for cam, _ in cameras:
            cam.release()
    tmp_path.replace(output_path)
    print(f"Saved GT render to {output_path}")


def project_vertices_paired(
    verts_3d: torch.Tensor,
    extri: torch.Tensor,
    intri: torch.Tensor,
) -> torch.Tensor:
    """
    Project 3D vertices with 1-to-1 camera-frame correspondence

    Args:
        verts_3d: (N_t, V, 3) - vertices for each frame
        extri: (N_t, 4, 4) - extrinsics for each frame (N_t == N_c)
        intri: (N_t, 3, 3) - intrinsics for each frame

    Returns:
        verts_2d: (N_t, V, 3) - projected vertices (no N_c dimension)
    """
    # Transform to camera space - now it's element-wise per frame
    # extri[:, :3, :3] is (N_t, 3, 3), verts_3d is (N_t, V, 3)
    verts_3d_cam = einops.einsum(
        extri[:, :3, :3],
        verts_3d,
        "N_t i j, N_t V j -> N_t V i"
    )
    verts_3d_cam = verts_3d_cam + extri[:, None, :3, 3]  # (N_t, V, 3)

    # Extract intrinsics - now per-frame
    fx = intri[:, 0, 0][:, None]  # (N_t, 1)
    fy = intri[:, 1, 1][:, None]
    cx = intri[:, 0, 2][:, None]
    cy = intri[:, 1, 2][:, None]

    # Perspective projection
    verts_2d = torch.stack(
        [
            verts_3d_cam[..., 0] / verts_3d_cam[..., 2] * fx + cx,
            verts_3d_cam[..., 1] / verts_3d_cam[..., 2] * fy + cy,
            verts_3d_cam[..., 2] / verts_3d_cam[..., 2].mean(dim=-1, keepdim=True) * (fx + fy) / 2,
        ],
        dim=-1,
    )

    return verts_2d  # (N_t, V, 3)


def composite_with_background(mesh_video, background_video_path, target_size, save_path, translate_x=0, translate_y=0, fps=30, mesh_opacity=0.5, white_threshold=0.98):
    """
    Args:
        mesh_video: np.ndarray (B, H, W, 3) - rendered mesh video, values in [0, 255]
        background_video_path: str - path to background video
        target_size: tuple (H, W) - target resolution
        save_path: str - path to save composited video (e.g., "output.mp4")
        translate_x: int - pixels to translate mesh horizontally (positive=right, negative=left)
        translate_y: int - pixels to translate mesh vertically (positive=down, negative=up)
        fps: int - frames per second for output video
        mesh_opacity: float - opacity of mesh layer (0.0=fully transparent, 1.0=fully opaque)
        white_threshold: float - threshold for detecting white pixels (default 0.95 means pixels with all RGB > 0.95 are considered white)
    Returns:
        composited: np.ndarray (B, H, W, 3) - final composited video in [0, 255]
    """
    # Open background video
    cap = cv2.VideoCapture(background_video_path)

    B, H, W, C = mesh_video.shape
    composited_frames = []

    # Normalize mesh video to [0, 1] if needed
    if mesh_video.max() > 1.0:
        mesh_video = mesh_video.astype(np.float32) / 255.0

    for i in range(B):
        ret, bg_frame = cap.read()
        if not ret:
            # Loop video if it's shorter than mesh video
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, bg_frame = cap.read()

        # Resize background to match target size
        bg_frame = cv2.resize(bg_frame, (W, H))  # cv2 uses (width, height)
        bg_frame = cv2.cvtColor(bg_frame, cv2.COLOR_BGR2RGB)  # BGR to RGB
        bg_frame = bg_frame.astype(np.float32) / 255.0  # Normalize to [0, 1]

        # Get mesh frame
        mesh_frame = mesh_video[i]  # (H, W, 3) in [0, 1]

        # Apply translation to mesh (not background)
        if translate_x != 0 or translate_y != 0:
            M = np.float32([[1, 0, translate_x], [0, 1, translate_y]])
            mesh_frame = cv2.warpAffine(mesh_frame, M, (W, H), borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        # Create alpha mask: detect white pixels (all RGB channels > threshold)
        # White pixels: alpha = 0 (fully transparent)
        # Non-white pixels: alpha = mesh_opacity
        white_mask = np.all(mesh_frame > white_threshold, axis=-1)  # (H, W) boolean mask
        alpha = np.ones((H, W), dtype=np.float32) * mesh_opacity
        alpha[white_mask] = 0.0  # Set white pixels to transparent

        # Add alpha channel dimension for broadcasting
        alpha = alpha[..., np.newaxis]  # (H, W, 1)

        # Alpha blending with per-pixel alpha
        composited = alpha * mesh_frame + (1 - alpha) * bg_frame
        composited_frames.append(composited)

    cap.release()

    # Convert to uint8 for saving
    composited_array = np.stack(composited_frames)  # (B, H, W, 3) in [0, 1]
    composited_uint8 = (composited_array * 255).astype(np.uint8)

    # Save using imageio
    imageio.mimsave(save_path, composited_uint8, fps=fps)

    print(f"Saved composited video to {save_path}")

    return composited_uint8  # Return uint8 version in [0, 255]

if __name__ == "__main__":
    main()
