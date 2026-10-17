"""Generate delegation videos and initial landmark dumps for a sequence.

Same as process_sequence_flame.py, but eyelid/lip landmarks are replaced using
sapiens2 keypoints308 pose estimation instead of mediapipe.

--parallel processes multiple actor/sequence pairs concurrently via a thread pool
that shares the one warmed-up DETR detector + sapiens2_1b pose model across all
worker threads, instead of the (now removed) process_sequence_sapiens_batch.py's
approach of spawning a separate subprocess — and therefore a separate full copy of
both models on the GPU — per pair. Model inference calls issued from different
threads still queue onto the same CUDA device rather than truly running at once,
but the CPU/I/O-bound stages per pair (ffmpeg transcode, video decode) genuinely
overlap, without duplicating GPU memory or model load time.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Iterator, List, Tuple
from tqdm import tqdm

import math
import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# Same file script_process_sequence_sapiens.sh tracks via jq, and the same layout
# (actor_id -> [sequence_id, ...]) as wrap_script/run_pipeline_batch.py's DONE_JSON.
DONE_JSON = ROOT / "completed_sequences_sapiens.json"

from backend.app.core.config import get_settings  # noqa: E402
from backend.app.flame.blendshape import *
from backend.app.flame.flame_utils import batch_rodrigues
from backend.app.services.label_tracker import label_tracker

# Only pairs at this stage of the review pipeline are eligible for sapiens2
# extraction at all -- anything further along (unreviewed, unconfirmed,
# confirmed) has already had its landmarks reviewed/built on by a later
# stage, and re-running detection would silently rewrite that out from under
# it. Covers "no label_tracker.json entry at all" too: get_status()'s own
# disk-fallback for an absent entry always resolves to one of these two
# statuses, never anything further along.
SAPIENS_ELIGIBLE_STATUSES = {"no_neutral_label", "unlabeled"}

from transformers import DetrForObjectDetection, DetrImageProcessor
from sapiens.pose.models import init_model
from sapiens.pose.datasets import parse_pose_metainfo, UDPHeatmap
from sapiens.pose.evaluators import nms

MAX_FRAMES = 400
FRAME_STEP = 5
# --v2: tracking fits (fit.npz) sampled every V2_FIT_STEP-th raw frame --
# n_timesteps counts raw frames, while expr/rot/tra/eye_rot hold one row per
# fitted frame. Raw frame f uses the fit of the nearest multiple of
# V2_FIT_STEP, i.e. row round(f / V2_FIT_STEP).
V2_FIT_STEP = 3


def v2_fit_index(raw_frame_idx: int, num_fit_frames: int) -> int:
    """Fit row for a raw frame under --v2: the nearest multiple of V2_FIT_STEP,
    divided by V2_FIT_STEP (clamped to the last fitted frame)."""
    return min(int(round(raw_frame_idx / V2_FIT_STEP)), num_fit_frames - 1)

# Mapping between FLAME landmark indices (same index space as flame_model/landmarks.npy)
# and sapiens2 keypoints308 indices, covering the eyelid, lip, and ear regions.
SAPIENS_LANDMARK_MAP = {
    "11": 158, "13": 156, "15": 108, "16": 105, "17": 110, "18": 189, "20": 198, "21": 214,
    "23": 193, "25": 195, "26": 218, "27": 202, "32": 175, "34": 132, "35": 129, "36": 134,
    "37": 188, "39": 197, "40": 213, "42": 192, "44": 194, "45": 217, "46": 201, "55": 191,
    "56": 207, "58": 206, "59": 190, "65": 120, "71": 96, "33": 173,
    # Chin -- confirmed geometrically (lowest-Y, on-midline point among the 74
    # standard landmarks, via neutral_correspondence_without_eyes_POT.json) and
    # matched to sapiens2 keypoints308's own "tip_of_chin" point. Same addition
    # as FaceScape's process_expression_sapiens.py.
    "50": 77,
    # Ears: left (l_point_1/5_of_inner_helix, l_tip_of_ear_lobe) and the
    # symmetric right-side points.
    "1": 241,
    "6": 267,
    # Nose (root/tip/lower-bridge, undernose center, nostril corners).
    "60": 71,   # Root of nose
    "57": 72,   # Tip of nose
    "54": 75,   # Lower nosebridge
    "61": 179,  # Middle of undernose
    "38": 183,  # Right nostril corner
    "19": 186,  # Left nostril corner
    # Eyebrows.
    "69": 78,
    "47": 83,
  #  "29": None,  # no sapiens2 equivalent
    "72": 87,
    "30": 93,
  #  "8": None,  # no sapiens2 equivalent
    # Additional ear points, beyond the tip-of-lobe pair above.
    "14": 256,
  #  "9": None,  # no sapiens2 equivalent
    "4": 230,
  #  "2": None,  # no sapiens2 equivalent
}

SAPIENS2_ROOT = ROOT / "sapiens2"
SAPIENS_POSE_ROOT = SAPIENS2_ROOT / "sapiens" / "pose"
def _env_path(key: str) -> Path:
    """Required location from the environment (script_process_sequence_sapiens.sh exports backend/.env)."""
    value = os.environ.get(key)
    if not value:
        raise KeyError(f"{key} is not set -- add it to FaceView/backend/.env (see .env.example)")
    return Path(value)


SAPIENS_CHECKPOINT_ROOT = _env_path("SAPIENS_CHECKPOINT_ROOT")
NERSEMBLE_TRACKING_ROOT = _env_path("NERSEMBLE_TRACKING_ROOT")
NERSEMBLE_TRACKING_LOCAL_ROOT = _env_path("NERSEMBLE_TRACKING_LOCAL_ROOT")
SAPIENS_POSE_CONFIG = (
    SAPIENS_POSE_ROOT
    / "configs/keypoints308/shutterstock_goliath_3po/sapiens2_1b_keypoints308_shutterstock_goliath_3po-1024x768.py"
)
SAPIENS_KEYPOINTS_META_FILE = SAPIENS_POSE_ROOT / "configs/_base_/keypoints308.py"
SAPIENS_POSE_CHECKPOINT = SAPIENS_CHECKPOINT_ROOT / "pose" / "sapiens2_1b_pose.safetensors"
SAPIENS_DETECTOR_CHECKPOINT = SAPIENS_CHECKPOINT_ROOT / "detector" / "detr-resnet-101-dc5"

BBOX_THRESHOLD = 0.3
NMS_THRESHOLD = 0.3
# Below this UDP heatmap confidence (see UDPHeatmap.decode), a sapiens2 keypoint
# is treated as obscured/unreliable rather than trusted -- its landmark is
# written as (-1, -1) so it falls outside triangulate.py's is_in_bounds check
# and gets excluded from triangulation instead of corrupting it. Reuses the
# same kpt_thr the sapiens2 pose_visualizer already uses to decide whether a
# keypoint is visible enough to draw.
SAPIENS_SCORE_THRESHOLD = 0.3

# Run the pose/detector forward passes in half precision (large speedup on the
# GPU with negligible keypoint accuracy loss).
POSE_AUTOCAST_DTYPE = torch.float16
# Re-run the person detector every N processed frames. The studio camera is
# static and the actor stays within the same (whole-person) box for a whole
# sequence, so the bbox is essentially constant; <= 0 means detect once per
# camera and reuse it for every frame.
BBOX_REDETECT_EVERY = 0

torch.backends.cudnn.benchmark = True

_sapiens_state: dict = {}


def _get_device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _autocast(device: str):
    return torch.autocast(device_type="cuda", dtype=POSE_AUTOCAST_DTYPE, enabled=device.startswith("cuda"))


def _run_ffmpeg(input_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostats",
        "-y",
        "-i",
        str(input_path),
        "-vf",
        "select=not(mod(n\\,5)),scale=550:802",
        "-vsync",
        "vfr",
        "-frames:v",
        str(MAX_FRAMES / FRAME_STEP),
        "-crf",
        "23",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)


def _load_face_indices(config_path: Path) -> np.array:
    return np.load(config_path)


def _write_frame(payload: dict, output_path: Path) -> None:
    # Write to a sibling temp file and rename into place -- rename is atomic, so a
    # kill/crash/NFS hiccup mid-write can't leave a truncated/empty frame_*.json
    # (see landmark_store.py's save_frame, which was fixed the same way).
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(payload), encoding="utf-8")
    tmp_path.replace(output_path)


def _load_completed(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _is_completed(completed: dict, actor_id: str, sequence_id: str) -> bool:
    return sequence_id in completed.get(actor_id, [])


def _record_completed(path: Path, actor_id: str, sequence_id: str) -> None:
    """Add actor/sequence to the completed-sequences JSON (deduped) and persist it.

    Takes an flock on the file for the read-modify-write, since --parallel > 1 has
    multiple worker threads calling this concurrently, each opening its own file
    descriptor — a plain in-process threading.Lock wouldn't cover a second run of
    this script started at the same time, but flock does.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.seek(0)
            raw = f.read()
            try:
                completed = json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                completed = {}
            sequences = set(completed.get(actor_id, []))
            sequences.add(sequence_id)
            completed[actor_id] = sorted(sequences)
            f.seek(0)
            f.truncate()
            json.dump(completed, f, indent=2)
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _get_flame_model(actor_id: str, sequence_id: str, v2: bool = False):
    """(N, V, 3) FLAME vertices. Default: one row per raw frame, up to
    MAX_FRAMES. v2: one row per FITTED frame (every V2_FIT_STEP-th raw frame),
    enough of them to cover MAX_FRAMES raw frames -- index with v2_fit_index()."""

    flame_params_path = f"{NERSEMBLE_TRACKING_ROOT}/{actor_id}/sequences/{sequence_id}/fit.npz"
    if(os.path.exists(flame_params_path) == False):
        flame_params_path = f"{NERSEMBLE_TRACKING_LOCAL_ROOT}/{actor_id}/sequences/{sequence_id}/fit.npz"

    if(os.path.exists(flame_params_path) == False):
        return None

    data = np.load(flame_params_path, allow_pickle=True)
    num_frames = data['n_timesteps'].item()
    if v2:
        num_fitted = data['expr'].shape[0]
        num_frames = min(num_fitted, v2_fit_index(MAX_FRAMES - 1, num_fitted) + 1)

    flameBS = FLAMEBlendshapes(data['shape'].shape[0], data['expr'].shape[1])
    flameBS.shape_params = torch.as_tensor(data['shape'], dtype=torch.float32).unsqueeze(0).to(flameBS.flame.device)
    os.makedirs(f"output/{actor_id}/{sequence_id}", exist_ok=True)

    vertices_sequence = []
    for i in tqdm(range(0, num_frames if v2 else min(MAX_FRAMES, num_frames)), desc="Processing frames"):
        flameBS.eye_pose_params = torch.as_tensor(
            np.concatenate([data['eye_rot'][i], data['eye_rot'][i]]) if data['eye_rot'].shape[1] == 3 else data['eye_rot'][i]
            , dtype=torch.float32
        ).to(flameBS.flame.device)

        translation = -np.array(data['tra'][i], copy=True) # Translation
        translation[0] *= -1

        # Same rot reordering as wrap_script/flame.py's Flame.loadSequence, so
        # the rotation matches exactly (Rodrigues, not a yaw/pitch/roll Euler
        # composition -- see run_pipeline_batch_temp.py).
        temp_rotation = np.array(data['rot'][i], copy=True)
        rotation = np.zeros_like(temp_rotation)
        rotation[0] = temp_rotation[0] - math.pi
        rotation[1] = temp_rotation[2]
        rotation[2] = -temp_rotation[1]

        V = flameBS.eval(weights=np.concatenate([data['expr'][i], [0, 0, 0]]))

        rotation_t = torch.as_tensor(rotation, dtype=torch.float32).unsqueeze(0)
        R = batch_rodrigues(rotation_t)[0].numpy()
        V = V @ R.T
        V += translation
        # No Z-flip here: Flame.LSB's own flip and the extra one
        # run_pipeline_batch_temp.py/render_flame_exp_without_eyes.py apply on
        # top of it cancel out net, so this already matches their current
        # (test_render_gt_raw.py-matching) convention unflipped.

        vertices_sequence.append(V)

    vertices_sequence = np.array(vertices_sequence)
    return vertices_sequence


def _get_detector(device: str):
    if "det_model" not in _sapiens_state:
        _sapiens_state["det_proc"] = DetrImageProcessor.from_pretrained(str(SAPIENS_DETECTOR_CHECKPOINT))
        _sapiens_state["det_model"] = (
            DetrForObjectDetection.from_pretrained(str(SAPIENS_DETECTOR_CHECKPOINT)).eval().to(device)
        )
    return _sapiens_state["det_proc"], _sapiens_state["det_model"]


def _get_pose_model(device: str):
    if "pose_model" not in _sapiens_state:
        model = init_model(str(SAPIENS_POSE_CONFIG), str(SAPIENS_POSE_CHECKPOINT), device=device)
        model.pose_metainfo = parse_pose_metainfo(dict(from_file=str(SAPIENS_KEYPOINTS_META_FILE)))
        codec_type = model.cfg.codec.pop("type")
        assert codec_type == "UDPHeatmap", "Only support UDPHeatmap"
        model.codec = UDPHeatmap(**model.cfg.codec)
        _sapiens_state["pose_model"] = model
    return _sapiens_state["pose_model"]


def _detect_person_bbox(image_bgr: np.ndarray, device: str) -> np.ndarray:
    proc, model = _get_detector(device)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    inputs = proc(images=Image.fromarray(image_rgb), return_tensors="pt").to(device)
    with torch.no_grad(), _autocast(device):
        outputs = model(**inputs)
    target_sizes = torch.tensor([image_rgb.shape[:2]], device=device)
    results = proc.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=BBOX_THRESHOLD)[0]

    person_mask = results["labels"] == 1
    # Upcast to fp32 before NMS: under fp16 autocast the boxes come back as
    # float16, and a person-sized box area (w*h) overflows float16 (max ~65504).
    boxes = results["boxes"][person_mask].float().cpu().numpy()
    scores = results["scores"][person_mask].float().cpu().numpy()
    if len(boxes) == 0:
        h, w = image_rgb.shape[:2]
        return np.array([0, 0, w - 1, h - 1], dtype=np.float32)

    bboxes = np.concatenate([boxes, scores.reshape(-1, 1)], axis=1)
    keep = nms(bboxes, NMS_THRESHOLD)
    kept = bboxes[keep]
    best = kept[np.argmax(kept[:, 4])]
    return best[:4].astype(np.float32)


def _detect_sapiens_keypoints(image_bgr: np.ndarray, bbox: np.ndarray, device: str) -> Tuple[np.ndarray, np.ndarray]:
    model = _get_pose_model(device)

    data_info = dict(img=image_bgr, bbox=bbox[None], bbox_score=np.ones(1, dtype=np.float32))
    data = model.data_preprocessor(model.pipeline(data_info))

    with torch.no_grad(), _autocast(device):
        pred = model(data["inputs"]).float().cpu().numpy()

    keypoints, keypoint_scores = model.codec.decode(pred[0])
    meta = data["data_samples"]["meta"]
    keypoints = keypoints / meta["input_size"] * meta["bbox_scale"] + meta["bbox_center"] - 0.5 * meta["bbox_scale"]
    return keypoints[0], keypoint_scores[0]  # (308, 2), (308,)


def _project_points_3d_to_2d(points, intrinsic, extrinsic):
    # points: (N, 3)
    # intrinsic: (3, 3)
    # extrinsic: (4, 4) or (3, 4)

    # Convert to homogeneous
    ones = np.ones((points.shape[0], 1))
    pts_h = np.hstack([points, ones])  # (N, 4)

    # Apply extrinsic
    cam = pts_h @ extrinsic.T          # (N, 4) or (N, 3)
    cam = cam[:, :3]

    # Apply intrinsic
    proj = cam @ intrinsic.T           # (N, 3)

    # Normalize
    proj[:, :2] /= proj[:, 2:3]
    return proj


def _extract_landmarks(
    video_path: Path, actor: str, sequence: str, camera: str, indices: np.array,
    vertices: np.array | None, orig_width: int, orig_height: int,
    only_raw_frame: int | None = None,
    mode: str = "both",
    v2: bool = False,
) -> None:
    """mode controls which half of each frame's landmarks get (re)computed:
      - "both" (default): the normal full extraction (FLAME projection for the
        base points, sapiens2 override for SAPIENS_LANDMARK_MAP's indices).
      - "flowface_only": only the FLAME-projected (non-sapiens2) points are
        recomputed; SAPIENS_LANDMARK_MAP indices are left as whatever's already
        in the existing frame_<idx>.json (the sapiens2 detector/pose model are
        never invoked, and `vertices` must be given).
      - "sapiens_only": only SAPIENS_LANDMARK_MAP indices are recomputed via the
        sapiens2 detector/pose model; every other point is copied verbatim from
        the existing frame_<idx>.json (FLAME projection is skipped entirely, so
        `vertices` may be None).
    "flowface_only"/"sapiens_only" only ever patch frames that already have a
    frame_<idx>.json -- they don't create new ones -- since both are meant to
    redo part of an existing extraction, not run it from scratch."""
    assert mode in ("both", "flowface_only", "sapiens_only")

    settings = get_settings()
    device = _get_device()
    cap = cv2.VideoCapture(video_path)

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    camera_dir = settings.landmarks_root / actor / sequence / ("cam_" + camera)

    existing_frame_indices: set[int] | None = None
    if mode in ("flowface_only", "sapiens_only"):
        existing_frame_indices = {
            int(p.stem.split("_")[1]) for p in camera_dir.glob("frame_*.json")
        }
        if not existing_frame_indices:
            print(f"  Cannot run --{mode.replace('_', '-')} for {camera}: no existing frame_<idx>.json found")
            cap.release()
            return
        # Exclusive upper bound on the raw-frame loop below, same convention as
        # the "both" branch: the largest existing delegation frame's raw index,
        # plus one step so range() actually reaches it.
        raw_frame_upper_bound = max(existing_frame_indices) * FRAME_STEP + FRAME_STEP
    else:
        # v2: vertices holds one row per fitted frame (every V2_FIT_STEP-th raw frame).
        raw_frame_upper_bound = min(MAX_FRAMES, vertices.shape[0] * (V2_FIT_STEP if v2 else 1))

    if mode != "sapiens_only":
        # Get Camera intrinsic and extrinsic -- only needed for FLAME projection.
        camera_path = settings.data_root / actor / "calibration" / "camera_params.json"
        assert os.path.exists(camera_path) == True, f"camera_render.py: File path {camera_path} does not exist"

        with open(camera_path, "r") as f:
            cam_params = json.load(f)

        intrinsic = np.array(cam_params["intrinsics"], dtype=np.float64)
        extrinsic = np.array(cam_params["world_2_cam"][camera], dtype=np.float64)

    cached_bbox = None
    frames_since_detect = 0

    for frame_idx in tqdm(range(0, raw_frame_upper_bound, FRAME_STEP), desc="Rendering frames"):

        success, frame = cap.read()
        if(not success):
            print(f"Issue at frame {frame_idx}, {actor} {sequence} {camera}")
            break

        # The delegation video is already subsampled to line up 1:1 with this
        # stepped loop, and reading it is strictly sequential (no seeking) -- so
        # even when only_raw_frame narrows this down to a single frame, every
        # earlier frame still has to be read() (to keep the video position
        # advancing correctly) before it can be skipped here.
        if only_raw_frame is not None and frame_idx != only_raw_frame:
            continue

        delegation_idx = frame_idx // FRAME_STEP
        if existing_frame_indices is not None and delegation_idx not in existing_frame_indices:
            continue

        frame_path = camera_dir / f"frame_{delegation_idx:06d}.json"

        existing_points = None
        if mode in ("flowface_only", "sapiens_only"):
            try:
                with frame_path.open("r", encoding="utf-8") as f:
                    existing_points = json.load(f).get("points")
            except (json.JSONDecodeError, OSError) as exc:
                print(f"  Skipping frame {delegation_idx} for {camera}: couldn't read existing {frame_path.name} ({exc})")
                continue

        if mode == "sapiens_only":
            # Everything except the sapiens2-controlled indices is copied
            # verbatim from what's already on disk -- FLAME is never touched.
            filtered_points = existing_points
        else:
            filtered_points = []
            vertex_row = v2_fit_index(frame_idx, vertices.shape[0]) if v2 else frame_idx
            filtered_vertices = vertices[vertex_row][indices]

            projected_vertices = _project_points_3d_to_2d(filtered_vertices, intrinsic, extrinsic)

            for landmark_idx in range(projected_vertices.shape[0]):

                true_x_pos = float(projected_vertices[landmark_idx][0]) / orig_width
                true_y_pos = float(projected_vertices[landmark_idx][1]) / orig_height

                z_pos = projected_vertices[landmark_idx][2]

                filtered_points.append(
                    {
                        "index": landmark_idx,
                        "x": true_x_pos,
                        "y": true_y_pos,
                        "z": float(z_pos),
                        "visiblility": 0.0,
                        "region": "None",
                    }
                )

            if mode == "flowface_only":
                # Restore whatever sapiens2 already produced for its indices --
                # this mode doesn't touch them at all.
                existing_by_index = {p["index"]: p for p in (existing_points or [])}
                for idx in range(len(filtered_points)):
                    idx_str = str(filtered_points[idx]["index"])
                    if idx_str in SAPIENS_LANDMARK_MAP and filtered_points[idx]["index"] in existing_by_index:
                        filtered_points[idx] = existing_by_index[filtered_points[idx]["index"]]

        if mode != "flowface_only":
            # The camera is static, so detect the person bbox once (or every
            # BBOX_REDETECT_EVERY frames) and reuse it instead of running the
            # detector on every frame.
            if cached_bbox is None or (BBOX_REDETECT_EVERY > 0 and frames_since_detect >= BBOX_REDETECT_EVERY):
                cached_bbox = _detect_person_bbox(frame, device)
                frames_since_detect = 0
            frames_since_detect += 1

            sapiens_keypoints, sapiens_scores = _detect_sapiens_keypoints(frame, cached_bbox, device)

            for idx in range(len(filtered_points)):
                idx_str = str(filtered_points[idx]["index"])
                if idx_str in SAPIENS_LANDMARK_MAP:
                    sapiens_idx = SAPIENS_LANDMARK_MAP[idx_str]
                    if sapiens_idx is None:
                        # No sapiens2 keypoint exists for this FLAME index at all
                        # (see SAPIENS_LANDMARK_MAP) -- always (-1, -1, -1), not
                        # just this frame's low-confidence sentinel below (which
                        # leaves z as FLAME's projected depth). z=-1 here lets
                        # triangulate.py's is_in_bounds()/triangulate_point()
                        # treat this exactly like every view being invalid, so
                        # it gets dropped from the correspondence map instead of
                        # triangulating a meaningless point.
                        filtered_points[idx]["x"] = -1.0
                        filtered_points[idx]["y"] = -1.0
                        filtered_points[idx]["z"] = -1.0
                    elif float(sapiens_scores[sapiens_idx]) < SAPIENS_SCORE_THRESHOLD:
                        # Low-confidence prediction (e.g. an eye/lip/ear obscured by
                        # the frame edge) -- mark out of bounds instead of trusting it.
                        filtered_points[idx]["x"] = -1.0
                        filtered_points[idx]["y"] = -1.0
                    else:
                        kp_x, kp_y = sapiens_keypoints[sapiens_idx]
                        filtered_points[idx]["x"] = float(kp_x) / width
                        filtered_points[idx]["y"] = float(kp_y) / height

        payload = {
            "frame_idx": delegation_idx,
            "actor_id": actor,
            "sequence_id": sequence,
            "camera_id": "cam_" + camera,
            "points": filtered_points,
        }
        _write_frame(payload, frame_path)

        if only_raw_frame is not None:
            break

    cap.release()


def _camera_landmarks_exist(settings, actor_id: str, sequence_id: str, camera_id: str) -> bool:
    """Whether this camera already has at least one extracted frame_<idx>.json --
    used by --skip-completed to skip re-processing a camera view that's already
    done, as opposed to --skip-completed-sequence's coarser (actor, sequence)
    check, which can't tell "done" from "done for a different camera subset"."""
    camera_dir = settings.landmarks_root / actor_id / sequence_id / camera_id
    if not camera_dir.is_dir():
        return False
    return any(camera_dir.glob("frame_*.json"))


def _pair_landmarks_exist(settings, actor_id: str, sequence_id: str) -> bool:
    """Whether this actor/sequence pair has any existing frame_<idx>.json for
    any camera at all -- used to gate --sapiens-only/--flowface-only, which
    patch part of an existing extraction rather than create one from scratch."""
    sequence_landmarks_dir = settings.landmarks_root / actor_id / sequence_id
    if not sequence_landmarks_dir.is_dir():
        return False
    return any(sequence_landmarks_dir.glob("cam_*/frame_*.json"))


def _iter_cameras(sequence_dir: Path, cameras: Iterable[str] | None = None) -> List[tuple[str, Path]]:
    images_dir = sequence_dir / "images"
    if not images_dir.exists():
        return []
    selected = set(cameras or [])
    result = []
    for video_file in sorted(images_dir.glob("*.mp4")):
        cam_id = video_file.stem
        if selected and cam_id not in selected:
            continue
        result.append((cam_id, video_file))
    return result


def _iter_targets(
    settings,
    actor_filter: str | None,
    sequence_filter: str | list[str] | None,
    actor_min: str | None = None,
    actor_max: str | None = None,
) -> Iterator[Tuple[str, str, Path]]:
    if actor_filter:
        actor_dirs = [settings.data_root / actor_filter]
    else:
        actor_dirs = [p for p in sorted(settings.data_root.glob("*/")) if p.is_dir()]
        if actor_min is not None or actor_max is not None:
            # String comparison, same as script_process_sequence_sapiens.sh's bash
            # comparison — correct as long as actor ids are fixed-width (e.g. "071"),
            # same assumption that script already made.
            actor_dirs = [
                p for p in actor_dirs
                if (actor_min is None or p.name >= actor_min) and (actor_max is None or p.name <= actor_max)
            ]
    if isinstance(sequence_filter, str):
        sequence_filter = [sequence_filter]
    for actor_dir in actor_dirs:
        if not actor_dir.exists():
            continue
        sequences_root = actor_dir / "sequences"
        if not sequences_root.exists():
            continue
        if sequence_filter:
            candidate_dirs = [sequences_root / name for name in sequence_filter]
        else:
            candidate_dirs = [p for p in sorted(sequences_root.glob("*/")) if p.is_dir()]
        for seq_dir in candidate_dirs:
            if not seq_dir.exists():
                continue
            yield actor_dir.name, seq_dir.name, seq_dir


def _process_pair(
    actor_id: str,
    sequence_id: str,
    sequence_dir: Path,
    settings,
    cameras: list[str] | None,
    landmarks_indices: np.array,
    raw_frame: int | None = None,
    skip_completed_camera: bool = False,
    mode: str = "both",
    v2: bool = False,
) -> bool:
    """Process every requested camera for one actor/sequence pair. Returns whether
    it should be recorded as completed (i.e. at least one camera ran without
    raising). Safe to call concurrently for different pairs from a thread pool —
    the only shared mutable state touched is the pre-warmed _sapiens_state dict
    (read-only from here on) and DONE_JSON (guarded by flock in _record_completed).
    Also takes an exclusive, non-blocking per-pair flock
    (landmarks_root/actor_id/sequence_id/.sapiens_pair.lock) up front so that a
    second cluster node running this same pair concurrently (e.g. two
    run_process_sequence_sapiens_batch.py workers racing on the same actor) skips
    it instead of duplicating work -- same pattern as
    run_pipeline_batch_all.py's .pipeline_pair.lock.

    When raw_frame is given, this only (re)writes that one frame's landmark JSON
    per camera -- e.g. to patch a single corrupt/empty frame_<idx>.json without
    touching every other already-processed frame. It also skips regenerating the
    delegation video (_run_ffmpeg is lossy and unconditional -- re-running it for
    a one-frame fix would needlessly re-encode footage everything else already
    depends on), so the delegation video must already exist.

    mode ("both" / "flowface_only" / "sapiens_only", see _extract_landmarks) —
    like raw_frame, the two partial modes patch an existing extraction rather
    than starting fresh, so the delegation video is required to already exist
    and is never regenerated here. "sapiens_only" also skips loading the FLAME
    model entirely, since it never touches FLAME-projected points.

    skip_completed_camera (--skip-completed) skips a camera view that already has
    extracted landmarks -- unlike --skip-completed-sequence, which only sees
    (actor, sequence) and can't tell "this sequence is done" from "this sequence
    is done for a different --cameras subset", this catches the latter case (e.g.
    adding a new camera to an already-processed sequence). Not applied under
    --raw-frame, which is specifically for patching a frame within an
    already-processed camera and would otherwise skip the very camera it's
    meant to fix."""
    # Exclusive per-pair lock (non-blocking): claims this actor/sequence pair
    # before doing any work, so at most one cluster node ever processes it at
    # once -- same pattern as run_pipeline_batch_all.py's .pipeline_pair.lock.
    # Skip immediately (rather than block) if another run already holds it,
    # since a full pair can take a while and this worker should move on to a
    # different pair instead of waiting.
    lock_dir = settings.landmarks_root / actor_id / sequence_id
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_file = open(lock_dir / ".sapiens_pair.lock", "w")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print(f"SKIP actor={actor_id} sequence={sequence_id}: pair is locked by another run already processing it")
        lock_file.close()
        return False

    try:
        print(f"Processing actor={actor_id} sequence={sequence_id}")
        flame_vertices = None
        if mode != "sapiens_only":
            flame_vertices = _get_flame_model(actor_id, sequence_id, v2=v2)
            if flame_vertices is None:
                print(f"Cannot find FLAME model for {actor_id} {sequence_id}")
                return False

        regenerate_delegation_video = raw_frame is None and mode == "both"

        cameras_processed = 0
        cameras_skipped = 0
        for camera_id, video_path in _iter_cameras(sequence_dir, cameras):
            if raw_frame is None and mode == "both" and skip_completed_camera and _camera_landmarks_exist(settings, actor_id, sequence_id, camera_id):
                print(f"  SKIP {camera_id}: landmarks already exist for this camera view")
                cameras_skipped += 1
                continue

            delegation_video = settings.processed_root / actor_id / sequence_id / f"{camera_id}_delegation.mp4"
            if regenerate_delegation_video:
                _run_ffmpeg(video_path, delegation_video)
            elif not delegation_video.exists():
                print(f"  Cannot process {camera_id}: delegation video missing "
                      f"({delegation_video}) — run without --raw-frame/--sapiens-only/--flowface-only first to generate it.")
                continue

            cap = cv2.VideoCapture(video_path)
            success, frame = cap.read()

            orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            cap.release()
            _extract_landmarks(delegation_video, actor_id, sequence_id, camera_id.replace('cam_', ''), landmarks_indices, flame_vertices, orig_width, orig_height, only_raw_frame=raw_frame, mode=mode, v2=v2)
            print(f"  Completed camera {camera_id} -> {delegation_video}")
            cameras_processed += 1

        # A pair where every requested camera was already done (all skipped, none
        # newly processed) is still a success -- only report failure when nothing
        # was touched at all (e.g. no cameras matched --cameras, or the pair
        # genuinely produced nothing).
        return cameras_processed > 0 or cameras_skipped > 0
    finally:
        fcntl.flock(lock_file, fcntl.LOCK_UN)
        lock_file.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("actor_id", nargs="?", default=None, help="Actor identifier; omit to process all actors")
    parser.add_argument("sequence_id", nargs="?", default=None, help="Sequence identifier; omit to process all sequences")
    parser.add_argument(
        "--sequences",
        nargs="*",
        default=None,
        help=(
            "Optional list of specific sequence ids to process for the given actor(s), "
            "e.g. for running a curated set concurrently with --parallel. Takes "
            "precedence over the sequence_id positional if both are given."
        ),
    )
    parser.add_argument("--cameras", nargs="*", default=None, help="Optional list of camera ids to process")
    parser.add_argument(
        "--raw-frame",
        type=int,
        default=None,
        help=(
            "Only (re)write this one raw frame index's landmark JSON per camera, "
            "instead of the whole sequence -- e.g. to patch a single corrupt/empty "
            "frame_<idx>.json. Requires actor_id/sequence_id (and normally --cameras) "
            "to already narrow this down to a specific pair, and the delegation "
            "video to already exist (this skips regenerating it)."
        ),
    )
    parser.add_argument(
        "--actor-min",
        default=None,
        help="Lowest actor id to include, inclusive (only applies when actor_id is omitted).",
    )
    parser.add_argument(
        "--actor-max",
        default=None,
        help="Highest actor id to include, inclusive (only applies when actor_id is omitted).",
    )
    parser.add_argument(
        "--skip-completed-sequence",
        action="store_true",
        help=(
            f"Skip actor/sequence pairs already recorded as done in {DONE_JSON}. "
            "This is a coarser, (actor, sequence)-only check -- it can't tell "
            "'this sequence is fully done' from 'this sequence was already done "
            "for a different --cameras subset', so a pair already marked done "
            "gets skipped entirely even when targeting cameras that were never "
            "actually processed. Use --skip-completed instead when adding a new "
            "camera to already-processed sequences."
        ),
    )
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        help=(
            "Skip individual camera views that already have extracted landmarks "
            "(a frame_<idx>.json present under their landmarks directory), "
            "checked per camera rather than per (actor, sequence) -- unlike "
            "--skip-completed-sequence, this correctly still processes a camera "
            "that's genuinely new (e.g. added to --cameras after the rest of the "
            "sequence was already done). Ignored under --raw-frame."
        ),
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help=(
            "Number of actor/sequence pairs to process concurrently (default: 1). "
            "Only helps when this invocation covers multiple pairs (i.e. actor_id "
            "omitted, and/or multiple sequences via --sequences or sequence_id "
            "omitted) — all workers share the one warmed-up detector + pose model, "
            "so this doesn't multiply GPU memory usage the way running separate "
            "`process_sequence_sapiens.py` invocations at once would."
        ),
    )
    parser.add_argument(
        "--v2",
        action="store_true",
        help=(
            f"The tracking fit (fit.npz) has one row per {V2_FIT_STEP}rd raw frame (n_timesteps "
            f"counts raw frames). Each FRAME_STEP={FRAME_STEP} raw frame uses the fit of the nearest "
            f"multiple of {V2_FIT_STEP}, i.e. fit row round(frame / {V2_FIT_STEP})."
        ),
    )
    args = parser.parse_args()

    settings = get_settings()
    sequence_filter = args.sequences if args.sequences else args.sequence_id
    targets = list(_iter_targets(settings, args.actor_id, sequence_filter, args.actor_min, args.actor_max))
    if not targets:
        raise SystemExit("No sequences found matching the provided filters")

    # Always gated to SAPIENS_ELIGIBLE_STATUSES -- pairs already reviewed/
    # wrapped/confirmed are left alone rather than having sapiens2 silently
    # overwrite landmarks a later stage has already built on.
    all_statuses = label_tracker.get_all(dataset="Nersemble")
    before = len(targets)
    targets = [
        (actor_id, sequence_id, sequence_dir)
        for actor_id, sequence_id, sequence_dir in targets
        if label_tracker.get_status(actor_id, sequence_id, dataset="Nersemble", all_statuses=all_statuses) in SAPIENS_ELIGIBLE_STATUSES
    ]
    print(f"Gated to label_tracker.json status in {sorted(SAPIENS_ELIGIBLE_STATUSES)}: "
          f"{before - len(targets)} pair(s) excluded, {len(targets)} remaining")
    if not targets:
        print("Nothing left to process.")
        return

    if args.skip_completed_sequence:
        completed = _load_completed(DONE_JSON)
        before = len(targets)
        targets = [
            (actor_id, sequence_id, sequence_dir)
            for actor_id, sequence_id, sequence_dir in targets
            if not _is_completed(completed, actor_id, sequence_id)
        ]
        print(f"--skip-completed-sequence: {before - len(targets)} pair(s) already in "
              f"{DONE_JSON.name}, {len(targets)} remaining")
        if not targets:
            print("Nothing left to process.")
            return

    landmarks_indices = _load_face_indices("flame_model/landmarks.npy")

    cameras = None
    if(args.cameras):
        cameras = ["cam_" + cam_id for cam_id in args.cameras]

    # Warm up sapiens2 detector + pose model once before processing any sequences.
    device = _get_device()
    _get_detector(device)
    _get_pose_model(device)

    def run_pair(actor_id: str, sequence_id: str, sequence_dir: Path) -> bool:
        try:
            completed_pair = _process_pair(actor_id, sequence_id, sequence_dir, settings, cameras, landmarks_indices, raw_frame=args.raw_frame, skip_completed_camera=args.skip_completed, v2=args.v2)
        except Exception as exc:
            print(f"FAIL actor={actor_id} sequence={sequence_id}: {exc}")
            return False
        # A --raw-frame patch only touches one frame -- don't mark the whole
        # pair "completed" in DONE_JSON, that tracking means the full sequence ran.
        if completed_pair and args.raw_frame is None:
            _record_completed(DONE_JSON, actor_id, sequence_id)
        return completed_pair

    passed = 0
    failed = 0
    print_lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as executor:
        futures = {
            executor.submit(run_pair, actor_id, sequence_id, sequence_dir): (actor_id, sequence_id)
            for actor_id, sequence_id, sequence_dir in targets
        }
        for future in as_completed(futures):
            with print_lock:
                if future.result():
                    passed += 1
                else:
                    failed += 1

    print(f"Done. pairs={len(targets)} passed={passed} failed_or_skipped={failed}")
    if failed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
