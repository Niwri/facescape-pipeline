"""Generate sapiens2 eyelid/lip landmark dumps for FaceScape actor/expressions.

FaceScape counterpart to process_sequence_sapiens.py -- same sapiens2
keypoints308 detector/pose model, same SAPIENS_LANDMARK_MAP, same shared-model
ThreadPoolExecutor pattern, but adapted for FaceScape's data shape: one static
JPG per (actor, expression, camera) instead of a per-camera video, no FLAME
model/projection at all (FaceScape has no wrap pipeline yet -- only sapiens2
landmarks are written), and only the eyelid/lip subset of SAPIENS_LANDMARK_MAP
is actually processed this pass (ears/nosebridge/eyebrow/undernose are a
planned future addition, left inert -- see the module-level constant below).

Reads actor/expression pairs from settings.facescape_images_root (the
Phase-1-transformed, origin-space copy under
/scratch/thirty/irwinngo1/facescape/..., NOT the pristine
/scratch/thirty/irwinngo/facescape source) -- landmarks only make sense once a
pair's cameras are in the shared origin-space canonical numbering. Pairs
recorded in settings.facescape_invalid_pairs_path (Phase 1's failures) are
skipped automatically.

--parallel shares one warmed-up DETR detector + sapiens2_1b pose model across
worker threads, same reasoning as process_sequence_sapiens.py's docstring.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterator, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
import sys
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
# SAPIENS_CHECKPOINT_ROOT / SANITY_RENDER_ROOT: FaceView/backend/.env (a value already in the environment wins,
# e.g. from wrap_script_facescape/.env when launched by run_process_expression_sapiens_batch.py).
from dotenv import load_dotenv  # noqa: E402
load_dotenv(ROOT / "backend" / ".env", override=False)

from backend.app.core.config import get_settings  # noqa: E402
from backend.app.services.label_tracker import label_tracker  # noqa: E402

DONE_JSON = ROOT / "completed_expressions_sapiens.json"
SANITY_RENDER_ROOT = Path(__import__("os").environ.get("SANITY_RENDER_ROOT", Path(__file__).resolve().parents[2] / "temp_renders"))  # set by wrap_script_facescape/.env

from transformers import DetrForObjectDetection, DetrImageProcessor  # noqa: E402
from sapiens.pose.models import init_model  # noqa: E402
from sapiens.pose.datasets import parse_pose_metainfo, UDPHeatmap  # noqa: E402
from sapiens.pose.evaluators import nms  # noqa: E402

# Mapping between FLAME landmark indices (same index space
# process_sequence_sapiens.py/SAPIENS_LANDMARK_MAP uses) and sapiens2
# keypoints308 indices, covering eyelid, lip, and ear regions -- kept as the
# exact same dict/numbering as Nersemble's so a future pass adding the
# remaining 0-73 standard landmarks (and the ear entries already present here)
# doesn't need any index translation.
SAPIENS_LANDMARK_MAP = {
    "11": 158, "13": 156, "15": 108, "16": 105, "17": 110, "18": 189, "20": 198, "21": 214,
    "23": 193, "25": 195, "26": 218, "27": 202, "32": 175, "34": 132, "35": 129, "36": 134,
    "37": 188, "39": 197, "40": 213, "42": 192, "44": 194, "45": 217, "46": 201, "55": 191,
    "56": 207, "58": 206, "59": 190, "65": 120, "71": 96, "33": 173,
    # Chin -- confirmed geometrically (lowest-Y, on-midline point among the 74
    # standard landmarks, via neutral_correspondence_without_eyes_POT.json).
    # Not to be confused with "8", which sits up near the eyebrows despite the
    # generic-sounding index number -- see that entry's own comment below.
    "50": 77,  # Tip of chin
    # Ears -- present for future reuse, deliberately NOT processed this pass
    # (see EYELID_LIP_INDICES below).
    "1": 241,
    "6": 267,
    # Nose (root/tip/lower-bridge, undernose center, nostril corners) --
    # present for future reuse, not processed this pass either (see
    # EYELID_LIP_LANDMARK_MAP below). Mirrors process_sequence_sapiens.py's
    # SAPIENS_LANDMARK_MAP exactly, so indices stay consistent between the two
    # scripts once this is turned on for FaceScape too.
    "60": 71,   # Root of nose
    "57": 72,   # Tip of nose
    "54": 75,   # Lower nosebridge
    "61": 179,  # Middle of undernose
    "38": 183,  # Right nostril corner
    "19": 186,  # Left nostril corner
    # Eyebrows -- present for future reuse, not processed this pass.
    "69": 78,
    "47": 83,
    "29": None,  # no sapiens2 equivalent
    "72": 87,
    "30": 93,
    "8": None,  # no sapiens2 equivalent
    # Additional ear points, beyond the tip-of-lobe pair above -- present for
    # future reuse, not processed this pass.
    "14": 256,
    "9": None,  # no sapiens2 equivalent
    "4": 230,
    "2": None,  # no sapiens2 equivalent
}

# Only worry about eyelids and lips for now -- excludes SAPIENS_LANDMARK_MAP's
# ear/nose/eyebrow entries. Nosebridge/undernose/eyebrows/ears are a planned
# future addition (see module docstring), not handled by this script yet.
_NON_EYELID_LIP_INDICES = {
  #  "1", "6",                                    # ears (tip of lobe)
  #  "60", "57", "54", "61", "38", "19",          # nose
  #  "69", "47", "29", "72", "30", "8",           # eyebrows
  #  "14", "9", "4", "2",                         # ears (additional points)
}
EYELID_LIP_LANDMARK_MAP = {k: v for k, v in SAPIENS_LANDMARK_MAP.items() if k not in _NON_EYELID_LIP_INDICES}

SAPIENS2_ROOT = ROOT / "sapiens2"
SAPIENS_POSE_ROOT = SAPIENS2_ROOT / "sapiens" / "pose"
SAPIENS_CHECKPOINT_ROOT = Path(
    __import__("os").environ["SAPIENS_CHECKPOINT_ROOT"]  # set by wrap_script_facescape/.env or FaceView/backend/.env
)
SAPIENS_POSE_CONFIG = (
    SAPIENS_POSE_ROOT
    / "configs/keypoints308/shutterstock_goliath_3po/sapiens2_1b_keypoints308_shutterstock_goliath_3po-1024x768.py"
)
SAPIENS_KEYPOINTS_META_FILE = SAPIENS_POSE_ROOT / "configs/_base_/keypoints308.py"
SAPIENS_POSE_CHECKPOINT = SAPIENS_CHECKPOINT_ROOT / "pose" / "sapiens2_1b_pose.safetensors"
SAPIENS_DETECTOR_CHECKPOINT = SAPIENS_CHECKPOINT_ROOT / "detector" / "detr-resnet-101-dc5"

BBOX_THRESHOLD = 0.3
NMS_THRESHOLD = 0.3
# Same reasoning/value as process_sequence_sapiens.py: below this UDP heatmap
# confidence, a keypoint is written as (-1, -1) so it falls outside
# triangulate.py's is_in_bounds check instead of corrupting anything downstream.
SAPIENS_SCORE_THRESHOLD = 0.3
POSE_AUTOCAST_DTYPE = torch.float16

torch.backends.cudnn.benchmark = True

_sapiens_state: dict = {}


def _get_device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _autocast(device: str):
    return torch.autocast(device_type="cuda", dtype=POSE_AUTOCAST_DTYPE, enabled=device.startswith("cuda"))


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


def _write_frame(payload: dict, output_path: Path) -> None:
    # Atomic write (temp + rename), same pattern as process_sequence_sapiens.py's
    # _write_frame / landmark_store.py's save_frame.
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


def _is_completed(completed: dict, actor_id: str, expression_id: str) -> bool:
    return expression_id in completed.get(actor_id, [])


def _record_completed(path: Path, actor_id: str, expression_id: str) -> None:
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
            expressions = set(completed.get(actor_id, []))
            expressions.add(expression_id)
            completed[actor_id] = sorted(expressions)
            f.seek(0)
            f.truncate()
            json.dump(completed, f, indent=2)
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def load_invalid_pairs(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return set()
    return {(p["actor"], p["expression"]) for p in data.get("pairs", [])}


def load_facescape_cameras(path: Path) -> list[str]:
    if not path.exists():
        raise SystemExit(
            f"Culled camera list not found: {path} -- run "
            "wrap_script_facescape/labeling/compute_facescape_cameras.py first"
        )
    data = json.loads(path.read_text())
    return [str(c) for c in data["all_culled"]]


def _camera_landmarks_exist(settings, actor_id: str, expression_id: str, camera_id: str) -> bool:
    frame_path = settings.facescape_landmarks_root / actor_id / expression_id / f"cam_{camera_id}" / "frame_000000.json"
    return frame_path.exists()


def _iter_targets(
    images_root: Path,
    invalid_pairs: set[tuple[str, str]],
    actor_filter: str | None,
    expression_filter: str | None,
    actor_min: str | None,
    actor_max: str | None,
    exclude_expressions: set[str],
) -> Iterator[Tuple[str, str, Path]]:
    if actor_filter:
        actor_dirs = [images_root / actor_filter]
    else:
        # Integer comparison, not string -- unlike Nersemble's zero-padded actor
        # ids ("018"), FaceScape actor directories are plain unpadded integers
        # ("1", "2", ..., "15", ...), so a naive string comparison would put "2"
        # after "15" lexicographically.
        actor_dirs = [p for p in images_root.glob("*/") if p.is_dir() and p.name.isdigit()]
        actor_dirs.sort(key=lambda p: int(p.name))
        if actor_min is not None or actor_max is not None:
            actor_min_n = int(actor_min) if actor_min is not None else None
            actor_max_n = int(actor_max) if actor_max is not None else None
            actor_dirs = [
                p for p in actor_dirs
                if (actor_min_n is None or int(p.name) >= actor_min_n) and (actor_max_n is None or int(p.name) <= actor_max_n)
            ]
    for actor_dir in actor_dirs:
        if not actor_dir.exists():
            continue
        if expression_filter:
            candidate_dirs = [actor_dir / expression_filter]
        else:
            candidate_dirs = [p for p in sorted(actor_dir.glob("*/")) if p.is_dir()]
        for expr_dir in candidate_dirs:
            if not expr_dir.exists():
                continue
            expression_id = expr_dir.name
            if expression_id in exclude_expressions:
                continue
            if not (expr_dir / "params.json").exists():
                continue
            if (actor_dir.name, expression_id) in invalid_pairs:
                continue
            yield actor_dir.name, expression_id, expr_dir


def _extract_landmarks_for_image(image_path: Path, device: str) -> list[dict]:
    """Single static image -> eyelid/lip point dicts. Collapses
    process_sequence_sapiens.py's video-frame loop to one cv2.imread + one DETR
    bbox + one sapiens2 pose forward pass -- FaceScape has exactly one image per
    camera per actor/expression, no time dimension, so there's no "previous
    frame" to warm a cached bbox from either way."""
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise ValueError(f"Could not read image: {image_path}")
    height, width = image_bgr.shape[:2]

    bbox = _detect_person_bbox(image_bgr, device)
    sapiens_keypoints, sapiens_scores = _detect_sapiens_keypoints(image_bgr, bbox, device)

    points = []
    for idx_str, sapiens_idx in EYELID_LIP_LANDMARK_MAP.items():
        if sapiens_idx is None:
            # No sapiens2 keypoint exists for this FLAME index at all (see
            # SAPIENS_LANDMARK_MAP) -- always (-1, -1, -1), matching
            # process_sequence_sapiens.py's handling of the same sentinel.
            x, y, z, score = -1.0, -1.0, -1.0, 0.0
        else:
            score = float(sapiens_scores[sapiens_idx])
            z = 0.0
            if score < SAPIENS_SCORE_THRESHOLD:
                x, y = -1.0, -1.0
            else:
                kp_x, kp_y = sapiens_keypoints[sapiens_idx]
                x, y = float(kp_x) / width, float(kp_y) / height
        points.append({
            "index": int(idx_str),
            "x": x,
            "y": y,
            "z": z,
            "visiblility": score,
            "region": "None",
        })
    return points


def _save_sanity_render(image_path: Path, points: list[dict], out_path: Path) -> None:
    """Draws detected points on a copy of the source image for manual visual
    confirmation -- see module docstring / the original request this was
    built for ("I can manually confirm them at the end")."""
    image = cv2.imread(str(image_path))
    if image is None:
        return
    height, width = image.shape[:2]
    for point in points:
        if not (0.0 < point["x"] < 1.0 and 0.0 < point["y"] < 1.0):
            continue  # (-1, -1) sentinel for a low-confidence/undetected point
        px, py = int(point["x"] * width), int(point["y"] * height)
        cv2.circle(image, (px, py), radius=4, color=(0, 0, 255), thickness=-1)
        cv2.putText(image, str(point["index"]), (px + 5, py - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), image)


def _process_pair(
    actor_id: str,
    expression_id: str,
    expr_dir: Path,
    settings,
    cameras: list[str],
    skip_completed_camera: bool,
    sanity_render_root: Path | None = None,
) -> bool:
    """Process every requested camera for one actor/expression pair. Returns
    whether at least one camera was newly processed or already-done (skipped).
    Safe to call concurrently for different pairs -- the only shared mutable
    state is the pre-warmed _sapiens_state dict (read-only from here on)."""
    print(f"Processing actor={actor_id} expression={expression_id}")
    device = _get_device()

    cameras_processed = 0
    cameras_skipped = 0
    for camera_id in cameras:
        image_path = expr_dir / f"{camera_id}.jpg"
        if not image_path.exists():
            continue  # this camera wasn't usable for this specific pair (per-pair calibration gap)

        if skip_completed_camera and _camera_landmarks_exist(settings, actor_id, expression_id, camera_id):
            cameras_skipped += 1
            continue

        try:
            points = _extract_landmarks_for_image(image_path, device)
        except ValueError as exc:
            print(f"  SKIP cam_{camera_id}: {exc}")
            continue

        frame_path = settings.facescape_landmarks_root / actor_id / expression_id / f"cam_{camera_id}" / "frame_000000.json"
        payload = {
            "frame_idx": 0,
            "actor_id": actor_id,
            "sequence_id": expression_id,
            "camera_id": "cam_" + camera_id,
            "points": points,
        }
        _write_frame(payload, frame_path)
        cameras_processed += 1

        if sanity_render_root is not None:
            render_path = sanity_render_root / f"{actor_id}_{expression_id}" / f"{camera_id}.jpg"
            _save_sanity_render(image_path, points, render_path)

    print(f"  actor={actor_id} expression={expression_id}: processed={cameras_processed} skipped={cameras_skipped}")
    return cameras_processed > 0 or cameras_skipped > 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("actor_id", nargs="?", default=None, help="Actor identifier; omit to process all actors")
    parser.add_argument("expression_id", nargs="?", default=None, help="Expression identifier; omit to process all expressions")
    parser.add_argument("--actor-min", default=None, help="Lowest actor id to include, inclusive (only applies when actor_id is omitted)")
    parser.add_argument("--actor-max", default=None, help="Highest actor id to include, inclusive (only applies when actor_id is omitted)")
    parser.add_argument(
        "--exclude-expressions", nargs="*", default=[],
        help="Expression ids to blacklist (skip), e.g. --exclude-expressions 1_neutral. Default: process every expression found.",
    )
    parser.add_argument("--cameras", nargs="*", default=None, help="Override the culled camera list from facescape_cameras.json")
    parser.add_argument("--skip-completed", action="store_true", help="Skip camera views that already have an extracted frame_000000.json")
    parser.add_argument("--skip-completed-pair", action="store_true", help=f"Skip actor/expression pairs already recorded as done in {DONE_JSON}")
    parser.add_argument("--parallel", type=int, default=1, help="Actor/expression pairs to process concurrently, sharing one warmed-up model (default: 1)")
    parser.add_argument(
        "--sanity-render", action="store_true",
        help="Draw detected points on a copy of each processed image and write it to "
             f"{SANITY_RENDER_ROOT}/<actor>_<expression>/<camera_id>.jpg for manual visual confirmation. "
             "Combine with --cameras to limit this to a handful of views (no need to render every camera).",
    )
    args = parser.parse_args()

    settings = get_settings()
    invalid_pairs = load_invalid_pairs(settings.facescape_invalid_pairs_path)
    print(f"Loaded {len(invalid_pairs)} invalid pair(s) from {settings.facescape_invalid_pairs_path} (will be skipped)")

    cameras = args.cameras if args.cameras else load_facescape_cameras(settings.facescape_cameras_path)
    print(f"Using {len(cameras)} camera(s): {cameras}")

    exclude = set(args.exclude_expressions)
    targets = list(_iter_targets(
        settings.facescape_images_root, invalid_pairs, args.actor_id, args.expression_id,
        args.actor_min, args.actor_max, exclude,
    ))
    if not targets:
        raise SystemExit("No actor/expression pairs found matching the provided filters")

    if args.skip_completed_pair:
        completed = _load_completed(DONE_JSON)
        before = len(targets)
        targets = [t for t in targets if not _is_completed(completed, t[0], t[1])]
        print(f"--skip-completed-pair: {before - len(targets)} pair(s) already in {DONE_JSON.name}, {len(targets)} remaining")
        if not targets:
            print("Nothing left to process.")
            return 0

    # Warm up sapiens2 detector + pose model once before processing any pairs.
    device = _get_device()
    _get_detector(device)
    _get_pose_model(device)

    sanity_render_root = SANITY_RENDER_ROOT if args.sanity_render else None

    def run_pair(actor_id: str, expression_id: str, expr_dir: Path) -> bool:
        try:
            completed_pair = _process_pair(
                actor_id, expression_id, expr_dir, settings, cameras, args.skip_completed,
                sanity_render_root=sanity_render_root,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL actor={actor_id} expression={expression_id}: {exc}")
            return False
        if completed_pair:
            _record_completed(DONE_JSON, actor_id, expression_id)
            # Explicit "unlabeled" write (Phase 2's real output signal) rather
            # than relying only on LabelTracker.get_status()'s disk-existence
            # fallback -- but don't downgrade a pair that's already further
            # along (unreviewed/unconfirmed/confirmed) just because sapiens2
            # was rerun on it, e.g. to pick up an improved model.
            current_status = label_tracker.get_status(actor_id, expression_id, dataset="FaceScape")
            if current_status in ("unreviewed", "unconfirmed", "confirmed"):
                print(f"  NOTE: not resetting {actor_id}/{expression_id} to 'unlabeled' -- "
                      f"current status is {current_status!r}")
            else:
                label_tracker.mark_unlabeled(actor_id, expression_id, dataset="FaceScape")
        return completed_pair

    passed = 0
    failed = 0
    print_lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as executor:
        futures = {
            executor.submit(run_pair, actor_id, expression_id, expr_dir): (actor_id, expression_id)
            for actor_id, expression_id, expr_dir in targets
        }
        for future in as_completed(futures):
            with print_lock:
                if future.result():
                    passed += 1
                else:
                    failed += 1

    print(f"Done. pairs={len(targets)} passed={passed} failed_or_skipped={failed}")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
