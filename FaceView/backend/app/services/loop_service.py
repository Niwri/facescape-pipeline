"""High-level orchestration for the human-in-the-loop update cycle."""
from __future__ import annotations

import asyncio
import logging
import subprocess
import threading
import sys
import time
from functools import lru_cache
from pathlib import Path
import json

import torch
import numpy as np

# Add co-tracker to path
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "co-tracker") not in sys.path:
    sys.path.append(str(ROOT / "co-tracker"))

from cotracker.predictor import CoTrackerPredictor

from ..core.config import get_settings, RAW_FRAME_STRIDE
from ..core.jobs import JobStatus, job_registry
from ..schemas.landmarks import LandmarkPayload
from ..schemas.common import LandmarkPoint
from .landmark_store import landmark_store
from ..api.routes import ws
from typing import List
torch.autograd.set_grad_enabled(False)

log = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    # No logging configured anywhere else in the app (main.py doesn't set any up),
    # so INFO-level records below would otherwise go nowhere. No-ops if something
    # else already configured the root logger.
    logging.basicConfig(level=logging.INFO)
log.setLevel(logging.INFO)

_model = None
CROP_SIZE = (550, 802)  # width, height — raw-resolution crop window fed to CoTracker
CROP_EXIT_MARGIN = 0.15  # fraction of crop size that triggers a re-center
SEGMENT_CHUNK_SIZE = 150  # fallback chunk size when no window/Settings.propagate_window is available

# Delegation videos are always generated at exactly this size (see
# scripts/preprocess/process_sequence_sapiens.py's `scale=550:802`), so it doubles
# as the batch-tracking frame size — no need to probe it per camera.
DELEGATION_SIZE = CROP_SIZE  # width, height

# Eyes/lips: tracked individually on raw-resolution crops (see _build_segment) for
# maximum precision. Everything else is batched per-camera on the delegation video
# (see _build_batch_segment) — cheaper, and precise enough for those landmarks.
EYE_LANDMARK_INDICES = {15, 17, 71, 16, 13, 11, 34, 36, 35, 33, 32, 65}
LIP_LANDMARK_INDICES = {37, 43, 18, 24, 42, 44, 23, 25, 39, 20, 46, 27, 40, 21, 45, 26, 59, 55, 56, 58}
PRECISE_LANDMARK_INDICES = EYE_LANDMARK_INDICES | LIP_LANDMARK_INDICES

# Required-free-memory threshold checked via wait_for_gpu_memory() before every
# CoTracker call — see wait_for_gpu_memory's own docstring. This is a starting
# estimate, not an isolated measurement of CoTracker's real footprint (unlike
# wrap_script/run_pipeline.py's MESHROOM_REQUIRED_MB, which came from actual
# profiling) — tune it based on what you observe via `nvidia-smi` under real load,
# especially since this backend process stays resident and was itself the ~8.2GB
# baseline usage observed while profiling the wrap_script pipeline.
COTRACKER_REQUIRED_MB = 3000

# wait_for_gpu_memory() only checks free memory at a point in time -- it doesn't
# reserve it. Two CoTracker callers (e.g. concurrent FaceScape ear-point
# propagation requests, each running in their own background thread) can each
# pass the check
# before either has actually allocated, then both proceed and together exceed
# what was really free, since neither call claims the memory atomically. This
# lock serializes each check-then-allocate-then-infer critical section so only
# one CoTracker call across all concurrent propagation jobs is ever doing that at
# once, closing the race without limiting how many propagation jobs can run
# concurrently otherwise (frame extraction, file I/O, etc. still overlap freely).
_gpu_inference_lock = threading.Lock()

device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint_path = ROOT / "co-tracker/checkpoints/scaled_offline.pth"


def wait_for_gpu_memory(
    required_mb: float,
    label: str = "",
    poll_interval: float = 5.0,
    timeout: float | None = 300.0,
) -> None:
    """Block until at least `required_mb` MiB of GPU memory is free system-wide.

    Uses `torch.cuda.mem_get_info()`, which reports true free memory on the device
    across every process sharing it, not just this one — relevant since this
    backend process shares its GPU with wrap_script's batch pipeline jobs. No-op if
    CUDA isn't available. Raises TimeoutError if `timeout` seconds pass without
    enough memory freeing up (pass None to wait indefinitely). Timeout defaults
    much shorter than wrap_script's equivalent (5 min vs 30 min), since this runs
    inline in the human-in-the-loop edit path — a labeler waiting on a stuck
    propagation should get a clear error, not an indefinite hang.
    """
    if not torch.cuda.is_available():
        return

    start = time.time()
    while True:
        free_bytes, _total_bytes = torch.cuda.mem_get_info()
        free_mb = free_bytes / (1024 ** 2)
        if free_mb >= required_mb:
            return

        elapsed = time.time() - start
        if timeout is not None and elapsed > timeout:
            raise TimeoutError(
                f"Timed out after {timeout:.0f}s waiting for {required_mb:.0f} MiB free GPU memory"
                f"{f' before {label}' if label else ''} — only {free_mb:.0f} MiB free."
            )

        log.info("Waiting for GPU memory%s: %.0f/%.0f MiB free (waited %.0fs), retrying in %.0fs...",
                  f" before {label}" if label else "", free_mb, required_mb, elapsed, poll_interval)
        time.sleep(poll_interval)

_model = CoTrackerPredictor(checkpoint=str(checkpoint_path))
_model = _model.to(device)


def _to_calib_camera_id(camera_id: str) -> str:
    return camera_id[4:] if camera_id.startswith("cam_") else camera_id


def _probe_video_size(video_path: Path) -> tuple[int, int]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "csv=s=x:p=0",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    width_str, height_str = result.stdout.strip().split("x")
    return int(width_str), int(height_str)


def _probe_video_frame_count(video_path: Path) -> int:
    result = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-count_frames",
            "-show_entries", "stream=nb_read_frames",
            "-of", "csv=p=0",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return int(result.stdout.strip())


@lru_cache(maxsize=32)
def _load_calibration(actor_id: str, sequence_id: str, sample_camera_id: str) -> tuple[np.ndarray, dict[str, np.ndarray], int, int]:
    settings = get_settings()
    camera_path = settings.data_root / actor_id / "calibration" / "camera_params.json"
    with camera_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    intrinsics = np.asarray(data["intrinsics"], dtype=np.float64)
    world_2_cam = {k: np.asarray(v, dtype=np.float64) for k, v in data["world_2_cam"].items()}

    # `intrinsics` is calibrated against the raw camera resolution. Tracking now
    # happens on raw-resolution crops, so ray-casting/reprojection stay in raw
    # pixel space too — no rescale needed, we just need the raw dimensions for
    # normalization elsewhere.
    raw_video_path = settings.data_root / actor_id / "sequences" / sequence_id / "images" / f"{sample_camera_id}.mp4"
    raw_width, raw_height = _probe_video_size(raw_video_path)

    return intrinsics, world_2_cam, raw_width, raw_height


def _camera_center_and_direction(pixel_xy: np.ndarray, intrinsic: np.ndarray, world_2_cam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r = world_2_cam[:3, :3]
    t = world_2_cam[:3, 3]
    center = -r.T @ t
    ray_cam = np.linalg.inv(intrinsic) @ np.array([pixel_xy[0], pixel_xy[1], 1.0], dtype=np.float64)
    ray_world = r.T @ ray_cam
    ray_world /= (np.linalg.norm(ray_world) + 1e-8)
    return center, ray_world


def _triangulate_from_observations(observations: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray | None:
    if len(observations) < 2:
        return None
    inv_sum = np.zeros((3, 3), dtype=np.float64)
    sum_vec = np.zeros((3,), dtype=np.float64)
    identity = np.eye(3, dtype=np.float64)
    for center, direction in observations:
        proj = identity - np.outer(direction, direction)
        inv_sum += proj
        sum_vec += proj @ center
    try:
        return np.linalg.solve(inv_sum, sum_vec)
    except np.linalg.LinAlgError:
        return None


def _project_world_to_norm(world_xyz: np.ndarray, intrinsic: np.ndarray, world_2_cam: np.ndarray, raw_width: int, raw_height: int) -> tuple[float, float] | None:
    r = world_2_cam[:3, :3]
    t = world_2_cam[:3, 3]
    cam_xyz = r @ world_xyz + t
    if cam_xyz[2] <= 1e-8:
        return None
    img = intrinsic @ cam_xyz
    x = img[0] / img[2]
    y = img[1] / img[2]
    return float(x / raw_width), float(y / raw_height)


def _clamp_crop_origin(center: float, crop_extent: int, raw_extent: int) -> int:
    if raw_extent <= crop_extent:
        return 0
    return int(np.clip(round(center - crop_extent / 2), 0, raw_extent - crop_extent))


def _extract_crop_frames(video_path: Path, start_frame: int, end_frame: int, crop_x0: int, crop_y0: int, crop_w: int, crop_h: int) -> np.ndarray:
    """`start_frame`/`end_frame` are delegation-frame indices; internally sampled from
    the raw video at RAW_FRAME_STRIDE so the output aligns 1:1 with delegation frames."""
    num_frames = end_frame - start_frame
    if num_frames <= 0:
        return np.zeros((0, crop_h, crop_w, 3), dtype=np.uint8)

    raw_start = start_frame * RAW_FRAME_STRIDE
    raw_end = end_frame * RAW_FRAME_STRIDE
    filter_chain = (
        f"trim=start_frame={raw_start}:end_frame={raw_end},"
        f"setpts=PTS-STARTPTS,"
        f"select='not(mod(n\\,{RAW_FRAME_STRIDE}))',"
        f"setpts=N/FRAME_RATE/TB,"
        f"crop={crop_w}:{crop_h}:{crop_x0}:{crop_y0}"
    )
    result = subprocess.run(
        [
            "ffmpeg",
            "-v", "error",
            "-i", str(video_path),
            "-vf", filter_chain,
            "-pix_fmt", "rgb24",
            "-f", "rawvideo",
            "-",
        ],
        capture_output=True,
        check=True,
    )
    frame_bytes = crop_w * crop_h * 3
    actual_frames = len(result.stdout) // frame_bytes
    frames = np.frombuffer(result.stdout, dtype=np.uint8, count=actual_frames * frame_bytes)
    frames = frames.reshape(actual_frames, crop_h, crop_w, 3)
    if actual_frames < num_frames:
        # Video ended before end_frame (e.g. clamped window) — pad by holding the
        # last decoded frame so callers can still index [0, num_frames).
        pad = np.repeat(frames[-1:], num_frames - actual_frames, axis=0) if actual_frames > 0 else np.zeros((num_frames, crop_h, crop_w, 3), dtype=np.uint8)
        frames = np.concatenate([frames, pad], axis=0)
    return frames


class _TrackSegment:
    __slots__ = ("start_frame", "raw_tracks", "visibility", "exit_frame", "crop_x0", "crop_y0", "crop_w", "crop_h")

    def __init__(self, start_frame: int, raw_tracks: np.ndarray, visibility: np.ndarray, exit_frame: int | None, crop_x0: int, crop_y0: int, crop_w: int, crop_h: int):
        self.start_frame = start_frame
        self.raw_tracks = raw_tracks  # (T, 2) raw pixel coords, indexed by t - start_frame
        self.visibility = visibility  # (T,)
        self.exit_frame = exit_frame  # first frame index where the point neared a movable crop edge, or None
        self.crop_x0 = crop_x0
        self.crop_y0 = crop_y0
        self.crop_w = crop_w
        self.crop_h = crop_h

    def local_index(self, t: int) -> int | None:
        i = t - self.start_frame
        if 0 <= i < self.raw_tracks.shape[0]:
            return i
        return None


def _build_segment(video_path: Path, raw_width: int, raw_height: int, start_frame: int, end_frame: int, center_xy: tuple[float, float]) -> _TrackSegment:
    crop_w = min(CROP_SIZE[0], raw_width)
    crop_h = min(CROP_SIZE[1], raw_height)
    cx, cy = center_xy
    crop_x0 = _clamp_crop_origin(cx, crop_w, raw_width)
    crop_y0 = _clamp_crop_origin(cy, crop_h, raw_height)

    t_extract_start = time.time()
    frames = _extract_crop_frames(video_path, start_frame, end_frame, crop_x0, crop_y0, crop_w, crop_h)
    t_extract = time.time() - t_extract_start

    t_infer_start = time.time()
    with _gpu_inference_lock:
        wait_for_gpu_memory(COTRACKER_REQUIRED_MB, label="CoTracker (single-point segment)")
        video_tensor = torch.from_numpy(frames).permute(0, 3, 1, 2)[None].float().to(device)

        query_x = float(np.clip(cx - crop_x0, 0, crop_w - 1))
        query_y = float(np.clip(cy - crop_y0, 0, crop_h - 1))
        query_tensor = torch.tensor([[0.0, query_x, query_y]], dtype=torch.float32, device=device)[None]

        pred_tracks, pred_visibility = _model(video_tensor, queries=query_tensor)
        if device == "cuda":
            torch.cuda.synchronize()
    t_infer = time.time() - t_infer_start
    del video_tensor
    log.info(
        "_build_segment cam=%s frames=%d [%d,%d) extract=%.2fs infer=%.2fs",
        video_path.stem, frames.shape[0], start_frame, end_frame, t_extract, t_infer,
    )

    crop_tracks = pred_tracks[0, :, 0, :].detach().cpu().numpy()
    visibility = pred_visibility[0, :, 0].detach().cpu().numpy()
    raw_tracks = crop_tracks + np.array([crop_x0, crop_y0], dtype=np.float64)

    left_clamped = crop_x0 == 0
    right_clamped = crop_x0 == raw_width - crop_w
    top_clamped = crop_y0 == 0
    bottom_clamped = crop_y0 == raw_height - crop_h
    margin_x = CROP_EXIT_MARGIN * crop_w
    margin_y = CROP_EXIT_MARGIN * crop_h

    exit_frame: int | None = None
    for i in range(crop_tracks.shape[0]):
        px, py = crop_tracks[i]
        near_edge = (
            (not left_clamped and px < margin_x)
            or (not right_clamped and px > crop_w - margin_x)
            or (not top_clamped and py < margin_y)
            or (not bottom_clamped and py > crop_h - margin_y)
        )
        if near_edge:
            exit_frame = start_frame + i
            break

    return _TrackSegment(start_frame, raw_tracks, visibility, exit_frame, crop_x0, crop_y0, crop_w, crop_h)


def _extract_delegation_frames(video_path: Path, start_frame: int, end_frame: int, width: int, height: int) -> np.ndarray:
    """`start_frame`/`end_frame` are delegation-frame indices, used directly against the
    delegation video — unlike `_extract_crop_frames`, no RAW_FRAME_STRIDE resampling is
    needed since the delegation video was already generated at that sampling rate."""
    num_frames = end_frame - start_frame
    if num_frames <= 0:
        return np.zeros((0, height, width, 3), dtype=np.uint8)

    filter_chain = f"trim=start_frame={start_frame}:end_frame={end_frame},setpts=PTS-STARTPTS"
    result = subprocess.run(
        [
            "ffmpeg",
            "-v", "error",
            "-i", str(video_path),
            "-vf", filter_chain,
            "-pix_fmt", "rgb24",
            "-f", "rawvideo",
            "-",
        ],
        capture_output=True,
        check=True,
    )
    frame_bytes = width * height * 3
    actual_frames = len(result.stdout) // frame_bytes
    frames = np.frombuffer(result.stdout, dtype=np.uint8, count=actual_frames * frame_bytes)
    frames = frames.reshape(actual_frames, height, width, 3)
    if actual_frames < num_frames:
        pad = np.repeat(frames[-1:], num_frames - actual_frames, axis=0) if actual_frames > 0 else np.zeros((num_frames, height, width, 3), dtype=np.uint8)
        frames = np.concatenate([frames, pad], axis=0)
    return frames


def _build_batch_segment(
    video_path: Path,
    raw_width: int,
    raw_height: int,
    start_frame: int,
    end_frame: int,
    centers_by_idx: dict[int, tuple[float, float]],
) -> dict[int, _TrackSegment]:
    """Track multiple points at once in a single CoTracker call, on the (already
    small, uncropped) delegation video rather than a raw-resolution crop. Returns one
    _TrackSegment per point, in the same raw-pixel-space `raw_tracks` convention as
    `_build_segment`, so callers can treat precise and batched segments identically."""
    deleg_w, deleg_h = DELEGATION_SIZE
    idx_list = list(centers_by_idx.keys())

    t_extract_start = time.time()
    frames = _extract_delegation_frames(video_path, start_frame, end_frame, deleg_w, deleg_h)
    t_extract = time.time() - t_extract_start

    t_infer_start = time.time()
    with _gpu_inference_lock:
        wait_for_gpu_memory(COTRACKER_REQUIRED_MB, label=f"CoTracker (batch of {len(idx_list)} points)")
        video_tensor = torch.from_numpy(frames).permute(0, 3, 1, 2)[None].float().to(device)

        scale_x = deleg_w / raw_width
        scale_y = deleg_h / raw_height
        queries = [
            [
                0.0,
                float(np.clip(centers_by_idx[idx][0] * scale_x, 0, deleg_w - 1)),
                float(np.clip(centers_by_idx[idx][1] * scale_y, 0, deleg_h - 1)),
            ]
            for idx in idx_list
        ]
        query_tensor = torch.tensor(queries, dtype=torch.float32, device=device)[None]

        pred_tracks, pred_visibility = _model(video_tensor, queries=query_tensor)
        if device == "cuda":
            torch.cuda.synchronize()
    t_infer = time.time() - t_infer_start
    del video_tensor
    log.info(
        "_build_batch_segment cam=%s points=%d frames=%d [%d,%d) extract=%.2fs infer=%.2fs",
        video_path.stem, len(idx_list), frames.shape[0], start_frame, end_frame, t_extract, t_infer,
    )

    inv_scale = np.array([1.0 / scale_x, 1.0 / scale_y], dtype=np.float64)
    segments: dict[int, _TrackSegment] = {}
    for i, idx in enumerate(idx_list):
        deleg_tracks = pred_tracks[0, :, i, :].detach().cpu().numpy()
        visibility = pred_visibility[0, :, i].detach().cpu().numpy()
        raw_tracks = deleg_tracks * inv_scale
        # No crop to exit — the delegation frame already covers the whole image, so
        # rebuilds are driven purely by chunk_size (at_chunk_end), not exit_frame.
        segments[idx] = _TrackSegment(start_frame, raw_tracks, visibility, None, 0, 0, deleg_w, deleg_h)
    return segments


def _save_edit_frame_triangulated(payloads: List[LandmarkPayload], changed_points_indices: List[int], changed_by_camera: dict[str, List[int]] | None = None) -> list[str]:
    """Save-only path (per request): triangulate the user's raw edited pixel
    positions at the edit frame ONLY (no CoTracker, no propagation to any other
    frame) and reproject the fused point onto every selected camera at that
    same frame -- including the camera(s) the user actually edited, so a point
    moved in one view gets corrected toward multi-view consensus everywhere,
    itself included, rather than keeping its own edited view as unconditional
    ground truth. This is the same triangulation _run_propagation() applies to
    its own edit frame, just without everything else that function also does
    (video decode, CoTracker segments, propagation to frames after this one)."""
    if not payloads or not changed_points_indices:
        return []

    payload_by_camera = {payload.camera_id: payload for payload in payloads}
    sample_payload = payloads[0]
    edit_frame_idx = sample_payload.frame_idx
    intrinsics, world_2_cam_map, raw_width, raw_height = _load_calibration(sample_payload.actor_id, sample_payload.sequence_id, sample_payload.camera_id)

    # Per-index set of cameras the user actually moved this point in -- only
    # those feed the triangulation; when absent/empty, fuse every camera.
    changed_camera_sets: dict[int, set[str]] = {}
    if changed_by_camera:
        for camera_id, indices in changed_by_camera.items():
            for idx in indices:
                changed_camera_sets.setdefault(idx, set()).add(camera_id)

    points_by_camera: dict[str, dict[int, LandmarkPoint]] = {}
    for camera_id, payload in payload_by_camera.items():
        points_by_index = {pt.index: pt for pt in payload.points}
        for idx in changed_points_indices:
            if idx not in points_by_index:
                raise ValueError(f"Camera {camera_id} is missing one or more changed points")
        points_by_camera[camera_id] = points_by_index

    edit_frame_updates: dict[str, dict[int, LandmarkPoint]] = {camera_id: {} for camera_id in payload_by_camera}
    for idx in changed_points_indices:
        track_cameras = changed_camera_sets.get(idx) or set(payload_by_camera.keys())

        observations: list[tuple[np.ndarray, np.ndarray]] = []
        for camera_id in track_cameras:
            pt = points_by_camera[camera_id][idx]
            calib_id = _to_calib_camera_id(camera_id)
            extrinsic = world_2_cam_map.get(calib_id)
            if extrinsic is None:
                continue
            x_px, y_px = pt.x * raw_width, pt.y * raw_height
            center, direction = _camera_center_and_direction(np.array([x_px, y_px], dtype=np.float64), intrinsics, extrinsic)
            observations.append((center, direction))

        fused_point = _triangulate_from_observations(observations)
        if fused_point is None:
            continue

        for camera_id in payload_by_camera:
            calib_id = _to_calib_camera_id(camera_id)
            extrinsic = world_2_cam_map.get(calib_id)
            if extrinsic is None:
                continue
            proj = _project_world_to_norm(fused_point, intrinsics, extrinsic, raw_width, raw_height)
            if proj is None:
                continue
            norm_x, norm_y = proj
            edit_frame_updates[camera_id][idx] = LandmarkPoint(index=idx, x=norm_x, y=norm_y, visibility=1.0)

    touched_cameras: list[str] = []
    for camera_id, updates in edit_frame_updates.items():
        if not updates:
            continue
        loaded = landmark_store.load_frames(sample_payload.actor_id, sample_payload.sequence_id, camera_id, [edit_frame_idx])
        frame = loaded[0] if loaded else None
        if frame is None:
            continue
        point_idx_map = {pt.index: j for j, pt in enumerate(frame.points)}
        for idx, updated in updates.items():
            j = point_idx_map.get(idx)
            if j is None:
                continue
            existing = frame.points[j]
            frame.points[j] = LandmarkPoint(
                index=existing.index,
                x=updated.x,
                y=updated.y,
                z=existing.z,
                visibility=updated.visibility,
            )
        camera_payload = payload_by_camera[camera_id]
        updated_payload = LandmarkPayload(
            frame_idx=edit_frame_idx,
            camera_id=camera_id,
            sequence_id=camera_payload.sequence_id,
            actor_id=camera_payload.actor_id,
            points=frame.points,
        )
        landmark_store.save_frame(updated_payload)
        touched_cameras.append(camera_id)
        log.info("save (no propagate): triangulated frame %d, camera=%s points=%s", edit_frame_idx, camera_id, list(updates.keys()))

    return touched_cameras


def _run_propagation(payloads: List[LandmarkPayload], window: int, changed_points_indices: List[int], changed_by_camera: dict[str, List[int]] | None = None) -> tuple[int, list[str]]:
    """Run CoTracker on raw-resolution crops per (point, camera-that-changed-it), fuse
    visible points via triangulation, and reproject onto every selected camera.
    Always propagates all the way to the last frame of the video — `window` no
    longer bounds how far propagation reaches, it only sets the chunk size CoTracker
    processes at a time. Crops (and chunks) are re-centered/rebuilt on the fused
    position whenever a camera's own track nears a movable crop edge or runs past
    its chunk, so reaching the end costs roughly one pass over the remaining
    frames rather than one giant call or repeated full-tail reprocessing."""
    if not payloads:
        return 0, []

    payload_by_camera = {payload.camera_id: payload for payload in payloads}

    # Per-index set of cameras the user actually moved this point in. Only those
    # cameras feed the triangulation for that index; the rest still receive the
    # reprojected fused position. When the map is absent/empty we fall back to
    # fusing every camera (legacy behavior).
    changed_camera_sets: dict[int, set[str]] = {}
    if changed_by_camera:
        for camera_id, indices in changed_by_camera.items():
            for idx in indices:
                changed_camera_sets.setdefault(idx, set()).add(camera_id)
    sample_payload = payloads[0]
    settings = get_settings()
    intrinsics, world_2_cam_map, raw_width, raw_height = _load_calibration(sample_payload.actor_id, sample_payload.sequence_id, sample_payload.camera_id)

    if not changed_points_indices:
        return 0, list(payload_by_camera.keys())

    raw_video_paths: dict[str, Path] = {}
    delegation_video_paths: dict[str, Path] = {}
    orig_frames_by_camera: dict[str, dict[int, object]] = {}
    points_by_camera: dict[str, dict[int, LandmarkPoint]] = {}
    for camera_id, payload in payload_by_camera.items():
        video_path = settings.data_root / payload.actor_id / "sequences" / payload.sequence_id / "images" / f"{camera_id}.mp4"
        if not video_path.exists():
            raise FileNotFoundError(f"Raw video not found: {video_path}")
        raw_video_paths[camera_id] = video_path
        delegation_video_paths[camera_id] = settings.processed_root / payload.actor_id / payload.sequence_id / f"{camera_id}_delegation.mp4"

        points_by_index = {pt.index: pt for pt in payload.points}
        for idx in changed_points_indices:
            if idx not in points_by_index:
                raise ValueError(f"Camera {camera_id} is missing one or more changed points")
        points_by_camera[camera_id] = points_by_index

        # landmark_store only has a JSON file for frames where the initial extraction
        # (mediapipe/FLAME) actually detected a face — it's sparse, not a dense
        # 0..N-1 array. Key by frame_idx so lookups below are correct even with gaps.
        orig_frames_by_camera[camera_id] = {
            frame.frame_idx: frame
            for frame in landmark_store.load_all_frames(payload.actor_id, payload.sequence_id, camera_id)
        }

    # total_frames is in delegation-frame space (same space as frame_idx/landmark
    # files) — derive it from the delegation video's own frame count, not the raw
    # video's, and not from however many sparse landmark files happen to exist.
    total_frames = _probe_video_frame_count(delegation_video_paths[sample_payload.camera_id])
    start_frame = sample_payload.frame_idx + 1
    # Propagation always reaches the true end of the sequence now — see chunk_size
    # below for what actually bounds a single CoTracker call.
    end_frame = total_frames

    if end_frame <= start_frame:
        return 0, list(payload_by_camera.keys())

    # `window` (None/non-positive falls back to SEGMENT_CHUNK_SIZE) sets how many
    # frames a single CoTracker call/segment covers, not how far propagation goes.
    chunk_size = window if window and window > 0 else SEGMENT_CHUNK_SIZE

    # Only cameras where the user actually changed a given point get a CoTracker
    # track for it. Cameras not in this set never run CoTracker for that point —
    # they only ever receive the reprojected fused position below. When no
    # per-camera edit info is given at all, fall back to tracking/fusing every
    # camera (legacy behavior).
    track_cameras_by_idx: dict[int, set[str]] = {}
    for idx in changed_points_indices:
        fuse_cameras = changed_camera_sets.get(idx) if changed_by_camera else None
        track_cameras_by_idx[idx] = set(fuse_cameras) if fuse_cameras else set(payload_by_camera.keys())

    # Eyes/lips get individual raw-crop CoTracker segments (maximum precision).
    # Everything else is tracked in a batch per camera on the delegation video —
    # one CoTracker call covering every batched point that camera changed, instead
    # of one call per point. Triangulation eligibility (track_cameras_by_idx) is
    # unaffected either way: a camera only ever contributes an observation for a
    # point it actually changed.
    precise_point_indices = [idx for idx in changed_points_indices if idx in PRECISE_LANDMARK_INDICES]
    batch_point_indices = [idx for idx in changed_points_indices if idx not in PRECISE_LANDMARK_INDICES]

    propagation_t_start = time.time()
    rebuild_counts: dict[tuple[str, int], int] = {(cam, idx): 0 for idx in changed_points_indices for cam in track_cameras_by_idx[idx]}
    log.info(
        "_run_propagation start points=%s window=[%d,%d) (%d frames) track_cameras=%s",
        changed_points_indices, start_frame, end_frame, end_frame - start_frame,
        {idx: sorted(cams) for idx, cams in track_cameras_by_idx.items()},
    )

    # One CoTracker segment per (point, camera-that-changed-it), anchored on the
    # point's raw pixel position at the edit frame. Segments are capped to
    # chunk_size frames and rebuilt either when the tracked point nears a movable
    # crop edge or when a chunk runs out, always re-anchored on the
    # fused/triangulated position rather than the camera's own drifting track, and
    # always capped at the next chunk boundary rather than the full remaining
    # video — so reaching the end costs one pass over the frames, not a full-tail
    # reprocess on every re-center, however often that happens to trigger.
    segments: dict[tuple[str, int], _TrackSegment] = {}
    initial_chunk_end = min(sample_payload.frame_idx + chunk_size, end_frame)

    for idx in precise_point_indices:
        for camera_id in track_cameras_by_idx[idx]:
            pt = points_by_camera[camera_id][idx]
            center = (pt.x * raw_width, pt.y * raw_height)
            segments[(camera_id, idx)] = _build_segment(raw_video_paths[camera_id], raw_width, raw_height, sample_payload.frame_idx, initial_chunk_end, center)

    # Group batched points by the camera that changed them so each camera gets one
    # CoTracker call covering all of its batched points at once.
    initial_batch_groups: dict[str, dict[int, tuple[float, float]]] = {}
    for idx in batch_point_indices:
        for camera_id in track_cameras_by_idx[idx]:
            pt = points_by_camera[camera_id][idx]
            initial_batch_groups.setdefault(camera_id, {})[idx] = (pt.x * raw_width, pt.y * raw_height)

    for camera_id, centers_by_idx in initial_batch_groups.items():
        built = _build_batch_segment(delegation_video_paths[camera_id], raw_width, raw_height, sample_payload.frame_idx, initial_chunk_end, centers_by_idx)
        for idx, seg in built.items():
            segments[(camera_id, idx)] = seg

    # The loop below starts at start_frame = frame_idx + 1, so it never touches the
    # edited frame itself. Fix that once here: triangulate from the user's raw
    # edited pixel positions (no CoTracker involved, since it's a single already-known
    # frame) and reproject the fused point onto *every* selected camera at the edit
    # frame, including the one(s) the user actually edited — so a point moved in one
    # view gets corrected toward multi-view consensus everywhere, itself included,
    # rather than having its own edited view kept as unconditional ground truth.
    edit_frame_idx = sample_payload.frame_idx
    edit_frame_updates: dict[str, dict[int, LandmarkPoint]] = {camera_id: {} for camera_id in payload_by_camera}
    for idx in changed_points_indices:
        track_cameras = track_cameras_by_idx[idx]

        observations: list[tuple[np.ndarray, np.ndarray]] = []
        for camera_id in track_cameras:
            pt = points_by_camera[camera_id][idx]
            calib_id = _to_calib_camera_id(camera_id)
            extrinsic = world_2_cam_map.get(calib_id)
            if extrinsic is None:
                continue
            x_px, y_px = pt.x * raw_width, pt.y * raw_height
            center, direction = _camera_center_and_direction(np.array([x_px, y_px], dtype=np.float64), intrinsics, extrinsic)
            observations.append((center, direction))

        fused_point = _triangulate_from_observations(observations)
        if fused_point is None:
            continue

        for camera_id in payload_by_camera:
            calib_id = _to_calib_camera_id(camera_id)
            extrinsic = world_2_cam_map.get(calib_id)
            if extrinsic is None:
                continue
            proj = _project_world_to_norm(fused_point, intrinsics, extrinsic, raw_width, raw_height)
            if proj is None:
                continue
            norm_x, norm_y = proj
            edit_frame_updates[camera_id][idx] = LandmarkPoint(index=idx, x=norm_x, y=norm_y, visibility=1.0)

    for camera_id, updates in edit_frame_updates.items():
        if not updates:
            continue
        frame = orig_frames_by_camera[camera_id].get(edit_frame_idx)
        if frame is None:
            continue
        point_idx_map = {pt.index: j for j, pt in enumerate(frame.points)}
        for idx, updated in updates.items():
            j = point_idx_map.get(idx)
            if j is None:
                continue
            existing = frame.points[j]
            frame.points[j] = LandmarkPoint(
                index=existing.index,
                x=updated.x,
                y=updated.y,
                z=existing.z,
                visibility=updated.visibility,
            )
        camera_payload = payload_by_camera[camera_id]
        updated_payload = LandmarkPayload(
            frame_idx=edit_frame_idx,
            camera_id=camera_id,
            sequence_id=camera_payload.sequence_id,
            actor_id=camera_payload.actor_id,
            points=frame.points,
        )
        landmark_store.save_frame(updated_payload)
        log.info("edit frame %d: reprojected triangulated position onto camera=%s points=%s", edit_frame_idx, camera_id, list(updates.keys()))

    for t in range(start_frame, end_frame):
        frame_updates: dict[str, dict[int, LandmarkPoint]] = {camera_id: {} for camera_id in payload_by_camera.keys()}
        # Batched-point rebuilds are collected here and issued as one CoTracker call
        # per camera after the idx loop below, instead of rebuilding each point's
        # segment individually — same coalescing as the initial segment build.
        pending_batch_rebuilds: dict[str, dict[int, tuple[float, float]]] = {}

        for idx in changed_points_indices:
            track_cameras = track_cameras_by_idx[idx]

            observations: list[tuple[np.ndarray, np.ndarray]] = []
            for camera_id in track_cameras:
                seg = segments[(camera_id, idx)]
                if seg.exit_frame is not None and t > seg.exit_frame:
                    continue  # drifted near its crop edge; excluded from fusion until restarted
                local_i = seg.local_index(t)
                if local_i is None:
                    continue
                vis = float(seg.visibility[local_i])
                if vis <= 0:
                    continue
                calib_id = _to_calib_camera_id(camera_id)
                extrinsic = world_2_cam_map.get(calib_id)
                if extrinsic is None:
                    continue
                x_px, y_px = seg.raw_tracks[local_i]
                center, direction = _camera_center_and_direction(np.array([x_px, y_px], dtype=np.float64), intrinsics, extrinsic)
                observations.append((center, direction))

            fused_point = _triangulate_from_observations(observations)

            for camera_id in payload_by_camera:
                # Cameras outside track_cameras never had a segment built for this
                # point — they rely entirely on the reprojected fused position.
                seg = segments.get((camera_id, idx))
                local_i = seg.local_index(t) if seg is not None else None
                calib_id = _to_calib_camera_id(camera_id)
                extrinsic = world_2_cam_map.get(calib_id)

                if fused_point is not None and extrinsic is not None:
                    proj = _project_world_to_norm(fused_point, intrinsics, extrinsic, raw_width, raw_height)
                else:
                    proj = None

                # Prefer the triangulated/fused position (agreed on by every camera
                # that saw the point) over this camera's own independent CoTracker
                # track, which drifts uncorrected whenever it happens to stay
                # "visible". Only fall back to the raw per-camera track when fusion
                # itself wasn't possible (fewer than 2 cameras had an observation)
                # and this camera actually has its own track.
                if proj is not None:
                    norm_x, norm_y = proj
                    vis = float(seg.visibility[local_i]) if local_i is not None else 1.0
                elif seg is not None and local_i is not None:
                    x_px, y_px = seg.raw_tracks[local_i]
                    norm_x = x_px / raw_width
                    norm_y = y_px / raw_height
                    vis = float(seg.visibility[local_i])
                else:
                    continue

                frame_updates[camera_id][idx] = LandmarkPoint(
                    index=idx,
                    x=norm_x,
                    y=norm_y,
                    visibility=vis,
                )

                # Only cameras with their own track can need a rebuild. Trigger on
                # either the crop-edge margin or simply running out of the current
                # chunk — either way the new segment is capped at the next chunk
                # boundary, not end_frame, so cost never balloons with re-centers.
                if seg is not None and local_i is not None:
                    at_chunk_end = local_i == seg.raw_tracks.shape[0] - 1 and t < end_frame - 1
                    if seg.exit_frame == t or at_chunk_end:
                        if proj is not None:
                            new_center = (proj[0] * raw_width, proj[1] * raw_height)
                        else:
                            new_center = tuple(seg.raw_tracks[local_i])
                        reason = "exit_margin" if seg.exit_frame == t else "chunk_end"
                        rebuild_counts[(camera_id, idx)] = rebuild_counts.get((camera_id, idx), 0) + 1

                        if idx in PRECISE_LANDMARK_INDICES:
                            chunk_end = min(t + chunk_size, end_frame)
                            log.info("rebuilding segment cam=%s idx=%d at t=%d reason=%s (rebuild #%d)",
                                     camera_id, idx, t, reason, rebuild_counts[(camera_id, idx)])
                            segments[(camera_id, idx)] = _build_segment(raw_video_paths[camera_id], raw_width, raw_height, t, chunk_end, new_center)
                        else:
                            # Batched point: defer to the coalesced rebuild below instead
                            # of a one-point-at-a-time CoTracker call.
                            pending_batch_rebuilds.setdefault(camera_id, {})[idx] = new_center

        for camera_id, centers_by_idx in pending_batch_rebuilds.items():
            chunk_end = min(t + chunk_size, end_frame)
            log.info("rebuilding batch segment cam=%s idxs=%s at t=%d (chunk_end=%d)",
                      camera_id, sorted(centers_by_idx.keys()), t, chunk_end)
            built = _build_batch_segment(delegation_video_paths[camera_id], raw_width, raw_height, t, chunk_end, centers_by_idx)
            for idx, seg in built.items():
                segments[(camera_id, idx)] = seg

        for camera_id, payload in payload_by_camera.items():
            frame = orig_frames_by_camera[camera_id].get(t)
            if frame is None:
                continue
            point_idx_map = {pt.index: j for j, pt in enumerate(frame.points)}
            for idx, updated in frame_updates[camera_id].items():
                j = point_idx_map.get(idx)
                if j is None:
                    continue
                existing = frame.points[j]
                frame.points[j] = LandmarkPoint(
                    index=existing.index,
                    x=updated.x,
                    y=updated.y,
                    z=existing.z,
                    visibility=updated.visibility,
                )

            updated_payload = LandmarkPayload(
                frame_idx=t,
                camera_id=camera_id,
                sequence_id=payload.sequence_id,
                actor_id=payload.actor_id,
                points=frame.points,
            )
            landmark_store.save_frame(updated_payload)

    total_rebuilds = sum(rebuild_counts.values())
    log.info(
        "_run_propagation done in %.2fs frames=%d total_rebuilds=%d per_segment=%s",
        time.time() - propagation_t_start, end_frame - start_frame, total_rebuilds, rebuild_counts,
    )
    return end_frame - start_frame, list(payload_by_camera.keys())


def enqueue_save(payloads: List[LandmarkPayload], changed_points_indices: List[int], window: int | None = None, changed_by_camera: dict[str, List[int]] | None = None) -> str:
    """"Save" (per request): triangulate-and-write the edit frame only (see
    _save_edit_frame_triangulated) -- no CoTracker, no propagation to any
    other frame. `window` is accepted only for API-compatibility with the
    existing request schema/frontend call site; it no longer means anything
    (there's no CoTracker chunk to size) and is otherwise unused.

    Still runs through the same job-registry/websocket-job plumbing as the
    old CoTracker-propagation path (job_type="propagate", event names
    "propagation_complete"/"propagation_failed") so the frontend's existing
    "Propagating..." status handling keeps working unchanged -- this is a
    same-frame triangulation, not propagation, but renaming the wire protocol
    would require frontend changes beyond what was asked here."""
    if not payloads:
        raise ValueError("payloads must not be empty")
    first = payloads[0]
    camera_ids = [payload.camera_id for payload in payloads]

    job = job_registry.create_job(
        job_type="propagate",
        sequence_id=first.sequence_id,
        camera_id=",".join(camera_ids),
        payload={"frame_idx": first.frame_idx, "camera_ids": camera_ids},
    )

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    def worker() -> None:
        job_registry.update_job(job.job_id, JobStatus.RUNNING)
        try:
            _save_edit_frame_triangulated(payloads, changed_points_indices, changed_by_camera)
            job_registry.update_job(job.job_id, JobStatus.COMPLETED)

            for camera_id in camera_ids:
                msg = {
                    "event": "propagation_complete",
                    "job_id": job.job_id,
                    "actor_id": first.actor_id,
                    "sequence_id": first.sequence_id,
                    "camera_id": camera_id,
                    "frame_idx": first.frame_idx,
                    "window": 0,
                }
                if loop and loop.is_running():
                    asyncio.run_coroutine_threadsafe(ws.broadcast(first.sequence_id, msg), loop)
                else:
                    asyncio.run(ws.broadcast(first.sequence_id, msg))

        except Exception as exc:  # pragma: no cover
            log.exception("Save job failed")
            job_registry.update_job(job.job_id, JobStatus.FAILED, message=str(exc))

            for camera_id in camera_ids:
                msg = {
                    "event": "propagation_failed",
                    "job_id": job.job_id,
                    "actor_id": first.actor_id,
                    "sequence_id": first.sequence_id,
                    "camera_id": camera_id,
                    "frame_idx": first.frame_idx,
                    "window": 0,
                    "error": str(exc),
                }
                if loop and loop.is_running():
                    asyncio.run_coroutine_threadsafe(ws.broadcast(first.sequence_id, msg), loop)
                else:
                    asyncio.run(ws.broadcast(first.sequence_id, msg))

    thread = threading.Thread(target=worker, name=f"save-{job.job_id}", daemon=True)
    thread.start()
    return job.job_id
