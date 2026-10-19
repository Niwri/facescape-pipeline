#!/usr/bin/env python3
"""FaceScape skin (74+) + standard-index-group (ear/nosebridge/undernose/
eyebrow/chin) propagation,
but per-vertex/per-group model routing between CoTracker and RAFT (dense
two-frame optical flow, torchvision's port of https://github.com/princeton-vl/raft),
mirroring wrap_script/run_neutral_cotracker_propagation.py's own
RAFT_SKIN_VERTEX_INDICES/RAFT_STANDARD_GROUPS split (same underlying vertex
indices/group names -- see those two constants below).

Unlike Nersemble, there is no frame-chaining question here at all (per
request): FaceScape has no video, so RAFT only ever sees ONE 2-image pair per
target expression -- [neutral_expression's photo, target expression's photo]
-- exactly the same 2-image "clip" CoTracker already used
(track_points_one_camera()'s existing docstring). RAFT's flow for that single
pair is computed once and sampled at each seed's neutral position -- no
chaining, no adaptive re-anchoring (nothing to re-anchor FROM with only one
hop). See _track_points_one_camera_raft()'s own docstring.

Everything else -- camera sets, triangulation, output writing, label_tracker
transitions -- follows the same pattern as wrap_script/
run_neutral_cotracker_propagation.py's own Nersemble equivalent.

Writes facescape_landmarks_root/<actor>/<expression>/cam_<camera>/<prefix>_frame_000000.json
per camera per group (ear_frame_000000.json, nosebridge_frame_000000.json,
undernose_frame_000000.json, eyebrow_frame_000000.json, chin_frame_000000.json)
-- never skin_frame_000000.json, never merged across groups, same "each group
owns its own file" policy as Nersemble's.

Usage: identical to run_neutral_skin_propagation.py
    python3 run_neutral_skin_propagation_raft.py <ACTOR_ID> [--expressions ...] [--dry-run]
    python3 run_neutral_skin_propagation_raft.py [--dry-run]
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import importlib.util
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import facescape_paths as fp  # all paths: wrap_script_facescape/.env

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
fp.add_faceview_to_sys_path()


def _load_module(name: str, path: str):
    """Same explicit-file-path import trick as this directory's own
    run_pipeline.py, and for the same reason: wrap_script/ has its own
    drifted mirror of backend/ (missing FaceScape settings), so a plain
    sys.path-based `import` would silently shadow the real one."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_triangulate_module = _load_module("wrap_script_triangulate", str(fp.WRAP_SCRIPT_ROOT / "triangulate.py"))
triangulate_point = _triangulate_module.triangulate_point

_skin_landmarks_module = _load_module("wrap_script_skin_landmarks", str(fp.WRAP_SCRIPT_ROOT / "skin_landmarks.py"))
SKIN_REGIONS = _skin_landmarks_module.SKIN_REGIONS
assign_global_indices = _skin_landmarks_module.assign_global_indices

from run_pipeline import run_pipeline as wrap_pipeline  # noqa: E402
import forehead_shift  # noqa: E402  (forehead skin target shift)
from backend.app.core.config import get_settings  # noqa: E402
from backend.app.services.label_tracker import label_tracker  # noqa: E402
from backend.app.services import facescape_neutral  # noqa: E402

CO_TRACKER_ROOT = fp.FACEVIEW_ROOT / "co-tracker"
if str(CO_TRACKER_ROOT) not in sys.path:
    sys.path.append(str(CO_TRACKER_ROOT))
from cotracker.predictor import CoTrackerPredictor  # noqa: E402

# torchvision's own RAFT port, same choice/reasoning as
# wrap_script/run_neutral_cotracker_propagation.py's own RAFT import.
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights, RAFT as RaftModel  # noqa: E402

COTRACKER_CHECKPOINT_PATH = CO_TRACKER_ROOT / "checkpoints" / "scaled_offline.pth"
# Same local checkpoint file Nersemble's RAFT integration downloaded and
# copied here for a durable, explicit path -- reused as-is, not re-downloaded.
RAFT_CHECKPOINT_PATH = fp.RAFT_CHECKPOINT_PATH
RAFT_WEIGHTS = Raft_Large_Weights.DEFAULT

# RAFT's all-pairs correlation volume is O((H*W)^2) at its internal 1/8-scale
# feature map -- fine at Nersemble's 550x802 delegation-frame size (that's
# what RAFT_REQUIRED_MB=1500 there reflects), but FaceScape's raw photos are
# ~2896x4344 (verified: a real run at full res tried to allocate 144 GiB and
# OOM'd). Downscaled by this factor before every RAFT call (both images,
# preserving aspect ratio) -- chosen to land RAFT's own working resolution
# close to Nersemble's own delegation-frame scale. Seed positions/resulting
# displacement are converted to/from this scale around the call, same
# raw-pixel <-> working-resolution conversion Nersemble's own
# track_and_fuse() does for its delegation video space.
RAFT_DOWNSCALE = 0.25
device = "cuda" if torch.cuda.is_available() else "cpu"

# Same raw vertex indices (from keypoint_left.txt/keypoint_right.txt) and
# same standard-index group names Nersemble's own RAFT_SKIN_VERTEX_INDICES/
# RAFT_STANDARD_GROUPS settled on, per request ("the same propagation
# implementation") -- everything else stays on CoTracker, same as there.
RAFT_SKIN_VERTEX_INDICES: set[int] = {
    2080, 1575, 3737, 3670, 3662, 3726, 3588, 3582,
    2711, 3115, 3484, 3471, 3438, 3432, 3386, 3390,
    3490, 3801, 3762,
}
RAFT_STANDARD_GROUPS: set[str] = {"eyebrow", "undernose"}

NEUTRAL_EXPRESSION = "1_neutral"
# Tried in this order when 1_neutral itself is in invalid_pairs.json (Phase
# 1's coordinate-transform alignment failed for it, same as any other
# expression can) -- there's no usable transformed images/params.json to seed
# anything from in that case, so a substitute is needed. Chosen for being
# close to a neutral pose already (a slight smile, or eyes/brows held in one
# position), making them reasonable stand-ins as a skin-propagation seed.
NEUTRAL_EXPRESSION_FALLBACKS = ["2_smile", "18_eye_closed", "19_brow_raiser", "20_brow_lower"]

# Per-region camera sets used to TRIANGULATE each region's skin points --
# distinct from facescape_cameras.json's culled labeling set (used only for
# the final reprojection/output step below). Given directly, not computed.
#
# "under" -- keypoint_under.txt (wrap_script/skin_landmarks.py's SKIN_REGIONS,
# shared with Nersemble) is currently empty, but SKIN_REGIONS/indices_by_region
# below still iterate every declared region including "under" -- without an
# entry here, propagate_expression()'s FACESCAPE_SKIN_REGION_CAMERAS[region]
# lookup would KeyError the moment that file gains a vertex, even with 0
# indices today. Empty camera list, matching the file's current empty state.
FACESCAPE_SKIN_REGION_CAMERAS: dict[str, list[str]] = {
    "front": ["1", "10", "16", "20", "22", "23", "24", "33", "43", "52"],
    "left": ["4", "5", "7", "16", "20", "22", "31", "44", "45", "48"],
    "right": ["8", "9", "17", "21", "23", "25", "35", "36", "39"],
    "under": [],
}

# Standard 0-73 sapiens2 index groups, mirroring wrap_script/
# run_neutral_cotracker_propagation.py's own STANDARD_INDEX_REGION_GROUPS
# (same index sets: LandmarkOverlay.tsx's LEFT_EAR/RIGHT_EAR, NOSE minus
# undernose's own center point, LEFT_BROW/RIGHT_BROW minus 2 dropped indices
# each). Camera sets reuse FACESCAPE_SKIN_REGION_CAMERAS' front/left/right --
# same simplification Nersemble's own nosebridge/undernose/eyebrow groups
# already use (only Nersemble's ear cameras were bespoke; here, ear reuses
# skin left/right too, per this test's own scope decision).
#
# "chin" (index 50) is new -- not part of Nersemble's own groups -- added per
# request, using the front camera set (chin is a front-facing point, same
# reasoning as nosebridge/undernose-center/eyebrow reusing front already).
FACESCAPE_STANDARD_INDEX_REGION_GROUPS: list[tuple[str, dict[str, dict[str, object]], str]] = [
    ("ear", {
        "left": {"indices": [2, 4, 1], "cameras": FACESCAPE_SKIN_REGION_CAMERAS["left"]},
        "right": {"indices": [9, 14, 6], "cameras": FACESCAPE_SKIN_REGION_CAMERAS["right"]},
    }, "ear"),
    ("nosebridge", {
        "nosebridge": {"indices": [54, 57], "cameras": FACESCAPE_SKIN_REGION_CAMERAS["front"]},
    }, "nosebridge"),
    # 37/60/18 here (this test's original values) are actually RIGHT_MOUTH_CORNER/
    # NOSE/LEFT_MOUTH_CORNER per render_landmarks.py's own REGION_COLORS
    # (mirroring the frontend's LandmarkOverlay.tsx exactly), not nostrils --
    # corrected to 38/61/19 (render_landmarks.py's own UNDER_NOSE=[38,41,61,19,22]
    # whitelist subset), same fix and same reasoning as Nersemble's own
    # UNDERNOSE_REGIONS correction this session. "center"'s camera set also
    # unioned to front+left+right, matching Nersemble's own widening.
    ("undernose", {
        "right_nostril": {"indices": [38], "cameras": FACESCAPE_SKIN_REGION_CAMERAS["right"]},
        "center": {"indices": [61], "cameras": FACESCAPE_SKIN_REGION_CAMERAS["front"] + FACESCAPE_SKIN_REGION_CAMERAS["left"] + FACESCAPE_SKIN_REGION_CAMERAS["right"]},
        "left_nostril": {"indices": [19], "cameras": FACESCAPE_SKIN_REGION_CAMERAS["left"]},
    }, "undernose"),
    # Front UNIONed into both sides' camera sets, matching Nersemble's own
    # EYEBROW_REGIONS widening this session (eyebrows are also visible
    # front-on). Differs from Nersemble's own EYEBROW_REGIONS by request:
    # FaceScape keeps 30/72 (left) and 47/69 (right) -- the four indices
    # WITH a real sapiens2 mapping (72->87, 69->78, 47->83, 30->93 in
    # SAPIENS_LANDMARK_MAP, process_expression_sapiens.py) -- and drops
    # 8/29 (explicit None in that map -- no sapiens2 source at all) and
    # 73/70 (not even keys in it). Per request.
    ("eyebrow", {
        "left": {"indices": [30, 72], "cameras": FACESCAPE_SKIN_REGION_CAMERAS["front"] + FACESCAPE_SKIN_REGION_CAMERAS["left"]},
        "right": {"indices": [47, 69], "cameras": FACESCAPE_SKIN_REGION_CAMERAS["front"] + FACESCAPE_SKIN_REGION_CAMERAS["right"]},
    }, "eyebrow"),
    ("chin", {
        "chin": {"indices": [50], "cameras": FACESCAPE_SKIN_REGION_CAMERAS["front"]},
    }, "chin"),
]


def load_cotracker_model() -> CoTrackerPredictor:
    return CoTrackerPredictor(checkpoint=str(COTRACKER_CHECKPOINT_PATH)).to(device)


def load_raft_model() -> RaftModel:
    """Same local-checkpoint load as Nersemble's own load_model() (RAFT
    variant): raft_large(weights=None) + manual state_dict load, rather than
    torchvision's own weight-fetching path, so this stays reproducible from
    the durable local copy."""
    model = raft_large(weights=None)
    state_dict = torch.load(str(RAFT_CHECKPOINT_PATH), map_location=device)
    model.load_state_dict(state_dict)
    return model.to(device).eval()


def load_invalid_pairs(path: Path) -> set[tuple[str, str]]:
    """Same schema/loader as run_pipeline_batch.py's own copy:
    {"pairs": [{"actor": ..., "expression": ...}, ...]}."""
    if not path.exists():
        return set()
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return set()
    return {(p["actor"], p["expression"]) for p in data.get("pairs", [])}


def resolve_neutral_expression(actor_id: str, invalid_pairs: set[tuple[str, str]]) -> str | None:
    """Which expression to treat as this actor's "neutral" reference for skin
    propagation. Tries the reviewer-saved neutral expression first (see
    facescape_neutral.load_actor_neutral_expression / the frontend's "Save as
    Neutral" button) -- if that one is itself in invalid_pairs.json, or none
    was ever saved, falls back to NEUTRAL_EXPRESSION (1_neutral), and then
    NEUTRAL_EXPRESSION_FALLBACKS in that exact order (Phase 1's alignment fit
    can fail for any of these too, leaving no usable transformed
    images/params.json to seed anything from). Returns the first candidate
    NOT itself in invalid_pairs for this actor, or None if every candidate is
    invalid (nothing usable to seed from at all)."""
    saved = facescape_neutral.load_actor_neutral_expression(actor_id)
    candidates = ((saved,) if saved else ()) + (NEUTRAL_EXPRESSION, *NEUTRAL_EXPRESSION_FALLBACKS)
    for candidate in candidates:
        if (actor_id, candidate) not in invalid_pairs:
            return candidate
    return None


def load_camera_params(actor_id: str, neutral_expression_id: str, settings) -> dict[str, dict]:
    """cam_id -> {"intrinsic": (3,3), "extrinsic": (4,4) world_2_cam, "width": int, "height": int}.
    Reads this actor's neutral-reference expression's own params.json --
    Phase 1's coordinate-transform already reindexed every actor's cameras
    onto a shared canonical numbering, so any expression's params.json has
    the same camera set/ids for this actor (which one is used here doesn't
    matter for that reason -- it's the neutral_expression_id purely for
    consistency with the rest of this actor's neutral-reference handling)."""
    params_path = settings.facescape_images_root / actor_id / neutral_expression_id / "params.json"
    with open(params_path) as f:
        params = json.load(f)
    camera_ids = sorted({k.rsplit("_", 1)[0] for k in params if k.endswith("_Rt")}, key=int)
    cams = {}
    for cam_id in camera_ids:
        k_key, rt_key = f"{cam_id}_K", f"{cam_id}_Rt"
        if k_key not in params:
            continue
        ext = np.eye(4, dtype=np.float64)
        ext[:3, :4] = np.array(params[rt_key], dtype=np.float64)
        cams[cam_id] = {
            "intrinsic": np.array(params[k_key], dtype=np.float64),
            "extrinsic": ext,
            "width": int(params.get(f"{cam_id}_width", 0) or 0),
            "height": int(params.get(f"{cam_id}_height", 0) or 0),
        }
    return cams


def project_world_to_pixel(world_xyz: np.ndarray, intrinsic: np.ndarray, extrinsic: np.ndarray) -> tuple[float, float] | None:
    r, t = extrinsic[:3, :3], extrinsic[:3, 3]
    cam_xyz = r @ world_xyz + t
    if cam_xyz[2] <= 1e-8:
        return None
    img = intrinsic @ cam_xyz
    return float(img[0] / img[2]), float(img[1] / img[2])


def load_image(actor_id: str, expression_id: str, cam_id: str, settings) -> np.ndarray | None:
    path = settings.facescape_images_root / actor_id / expression_id / f"{cam_id}.jpg"
    if not path.exists():
        return None
    return np.array(Image.open(path).convert("RGB"))


def load_standard_frame_points(
    actor_id: str, expression_id: str, cam_id: str, width: float, height: float, settings,
) -> dict[int, tuple[float, float]]:
    """Reads this camera's EXISTING cam_<camera>/frame_000000.json (Phase 2's
    sapiens2 detections) and returns every in-bounds point as raw-pixel (x,y),
    keyed by its standard 0-73 index -- same in-bounds filter and
    normalized-to-raw-pixel conversion as Nersemble's load_ear_seeds()
    (frame_000000.json stores normalized [0,1] x/y, not raw pixels or a
    width/height of its own -- width/height come from this camera's own
    params.json entry, i.e. cams[cam_id]), but for FaceScape's single static
    frame (always frame_000000.json, no delegation-frame index) and returning
    every point at once rather than one index at a time, since callers here
    often need several indices from the same camera/expression.

    Also merges in cam_<camera>/ear_frame_000000.json's own points (the
    manually-clicked ear-top positions -- indices 2/9, the only standard
    indices with NO sapiens2 source at all, see SAPIENS_LANDMARK_MAP in
    process_expression_sapiens.py) where present, WITHOUT overriding a real
    frame_000000.json entry (there normally isn't one for 2/9 to collide
    with anyway). Without this, a reviewer's manually-saved ear-top position
    can never become a propagation seed -- frame_000000.json alone never has
    a valid entry for 2/9, sapiens2 or not."""
    frame_path = settings.facescape_landmarks_root / actor_id / expression_id / f"cam_{cam_id}" / "frame_000000.json"
    points: dict[int, tuple[float, float]] = {}
    if frame_path.exists():
        with frame_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        for point in data.get("points", []):
            if point.get("x") is None or point.get("y") is None:
                continue
            x, y = point["x"], point["y"]
            if not (0.0 < x < 1.0 and 0.0 < y < 1.0):
                continue
            points[point["index"]] = (x * width, y * height)

    ear_frame_path = settings.facescape_landmarks_root / actor_id / expression_id / f"cam_{cam_id}" / "ear_frame_000000.json"
    if ear_frame_path.exists():
        with ear_frame_path.open("r", encoding="utf-8") as f:
            ear_data = json.load(f)
        for point in ear_data.get("points", []):
            if point.get("x") is None or point.get("y") is None:
                continue
            x, y = point["x"], point["y"]
            if not (0.0 < x < 1.0 and 0.0 < y < 1.0):
                continue
            points.setdefault(point["index"], (x * width, y * height))

    return points


def track_points_one_camera(
    model: CoTrackerPredictor | RaftModel,
    neutral_img: np.ndarray,
    target_img: np.ndarray,
    seeds: dict[int, tuple[float, float]],
) -> dict[int, tuple[np.ndarray, float]]:
    """Dispatches to whichever model this call was given -- callers stay
    generic over "some model with a (neutral_img, target_img, seeds) ->
    {global_idx: (tracked_xy, visibility)} contract", unaware of which one
    it actually got. propagate_expression() decides, per point, which model
    to route to (see RAFT_SKIN_VERTEX_INDICES/RAFT_STANDARD_GROUPS)."""
    if isinstance(model, RaftModel):
        return _track_points_one_camera_raft(model, neutral_img, target_img, seeds)
    return _track_points_one_camera_cotracker(model, neutral_img, target_img, seeds)


def _track_points_one_camera_cotracker(
    model: CoTrackerPredictor,
    neutral_img: np.ndarray,
    target_img: np.ndarray,
    seeds: dict[int, tuple[float, float]],
) -> dict[int, tuple[np.ndarray, float]]:
    """Tracks every seeded point (global_idx -> pixel xy in neutral_img) from
    neutral_img (t=0) to target_img (t=1) in ONE CoTracker call -- a 2-frame
    "clip", not a real video. Returns global_idx -> (tracked pixel xy in
    target_img, visibility confidence)."""
    if not seeds:
        return {}
    if neutral_img.shape != target_img.shape:
        # Different physical cameras can report slightly different raw
        # resolutions between captures in rare cases; CoTracker needs a
        # single consistent frame size for the whole clip.
        h = min(neutral_img.shape[0], target_img.shape[0])
        w = min(neutral_img.shape[1], target_img.shape[1])
        neutral_img = neutral_img[:h, :w]
        target_img = target_img[:h, :w]

    frames = np.stack([neutral_img, target_img], axis=0)  # (2, H, W, 3)
    video_tensor = torch.from_numpy(frames).permute(0, 3, 1, 2)[None].float().to(device)
    global_indices = list(seeds.keys())
    query_tensor = torch.tensor(
        [[0.0, seeds[idx][0], seeds[idx][1]] for idx in global_indices],
        dtype=torch.float32, device=device,
    )[None]

    with torch.no_grad():
        pred_tracks, pred_visibility = model(video_tensor, queries=query_tensor)
    if device == "cuda":
        torch.cuda.synchronize()
    tracks = pred_tracks[0].detach().cpu().numpy()  # (2, N, 2)
    visibility = pred_visibility[0].detach().cpu().numpy()  # (2, N)

    return {
        global_idx: (tracks[1, i], float(visibility[1, i]))
        for i, global_idx in enumerate(global_indices)
    }


def _raft_flow(model: RaftModel, img0: np.ndarray, img1: np.ndarray) -> np.ndarray:
    """One RAFT call between exactly two images -- same padding-to-multiple-
    of-8 + crop-back trick as Nersemble's own _raft_flow. Returns (2, H, W)
    flow (dx, dy) in img0's own ORIGINAL (unpadded) pixel grid."""
    h, w = img0.shape[:2]
    t0 = torch.from_numpy(img0).permute(2, 0, 1)[None].to(device)  # uint8 (1,3,H,W)
    t1 = torch.from_numpy(img1).permute(2, 0, 1)[None].to(device)
    t0, t1 = RAFT_WEIGHTS.transforms()(t0, t1)  # -> float32, rescaled to [-1, 1]

    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8
    if pad_h or pad_w:
        t0 = torch.nn.functional.pad(t0, (0, pad_w, 0, pad_h), mode="replicate")
        t1 = torch.nn.functional.pad(t1, (0, pad_w, 0, pad_h), mode="replicate")

    with torch.no_grad():
        flow_predictions = model(t0, t1)
    flow = flow_predictions[-1][0, :, :h, :w]  # last refinement iteration, cropped back to (2, H, W)
    if device == "cuda":
        torch.cuda.synchronize()
    return flow.detach().cpu().numpy()


def _sample_flow_bilinear(flow: np.ndarray, x: float, y: float) -> tuple[float, float]:
    """Same bilinear sub-pixel sample as Nersemble's own _sample_flow_bilinear,
    clamped to the image bounds."""
    h, w = flow.shape[1], flow.shape[2]
    x = min(max(x, 0.0), w - 1.0)
    y = min(max(y, 0.0), h - 1.0)
    x0, y0 = int(x), int(y)
    x1, y1 = min(x0 + 1, w - 1), min(y0 + 1, h - 1)
    wx, wy = x - x0, y - y0
    top = flow[:, y0, x0] * (1 - wx) + flow[:, y0, x1] * wx
    bot = flow[:, y1, x0] * (1 - wx) + flow[:, y1, x1] * wx
    dx, dy = top * (1 - wy) + bot * wy
    return float(dx), float(dy)


def _track_points_one_camera_raft(
    model: RaftModel,
    neutral_img: np.ndarray,
    target_img: np.ndarray,
    seeds: dict[int, tuple[float, float]],
) -> dict[int, tuple[np.ndarray, float]]:
    """RAFT counterpart of _track_points_one_camera_cotracker() -- same
    (neutral_img, target_img, seeds) -> {global_idx: (tracked_xy, visibility)}
    contract. Per request, this is neutral-frame-to-target ONLY: one dense
    flow(neutral_img, target_img) call, sampled at each seed's own neutral
    position -- no chaining (there's only ever one hop here to begin with,
    unlike Nersemble's own multi-frame video case) and no adaptive
    re-anchoring (nothing to re-anchor FROM with a single hop). Visibility is
    always 1.0, same reasoning as Nersemble's own RAFT integration -- RAFT has
    no occlusion/confidence signal of its own."""
    if not seeds:
        return {}
    if neutral_img.shape != target_img.shape:
        h = min(neutral_img.shape[0], target_img.shape[0])
        w = min(neutral_img.shape[1], target_img.shape[1])
        neutral_img = neutral_img[:h, :w]
        target_img = target_img[:h, :w]

    orig_h, orig_w = neutral_img.shape[:2]
    small_w = max(8, round(orig_w * RAFT_DOWNSCALE))
    small_h = max(8, round(orig_h * RAFT_DOWNSCALE))
    scale_x = small_w / orig_w
    scale_y = small_h / orig_h
    neutral_small = np.array(Image.fromarray(neutral_img).resize((small_w, small_h), Image.BILINEAR))
    target_small = np.array(Image.fromarray(target_img).resize((small_w, small_h), Image.BILINEAR))

    flow = _raft_flow(model, neutral_small, target_small)  # (2, small_h, small_w), in small-image pixel units
    result: dict[int, tuple[np.ndarray, float]] = {}
    for global_idx, (x, y) in seeds.items():
        dx_small, dy_small = _sample_flow_bilinear(flow, x * scale_x, y * scale_y)
        dx, dy = dx_small / scale_x, dy_small / scale_y  # back to orig_img pixel units
        result[global_idx] = (np.array([x + dx, y + dy], dtype=np.float64), 1.0)
    return result


def write_points_frames(
    actor_id: str,
    expression_id: str,
    cams: dict[str, dict],
    world_xyz_by_idx: dict[int, np.ndarray],
    output_camera_ids: list[str],
    settings,
    region_label: str = "skin",
    filename: str = "skin_frame_000000.json",
) -> int:
    """Reprojects already-known 3D point positions onto every
    output_camera_ids camera and writes `filename` per camera -- generalized
    from the original write_skin_frames() so the standard-index groups below
    (ear/nosebridge/undernose/eyebrow/chin) can reuse the exact same
    reprojection+write logic under their own file/region name instead of
    duplicating it. Returns the number of cameras written."""
    updates_by_cam: dict[str, dict[int, tuple[float, float]]] = {}
    for cam_id in output_camera_ids:
        cam = cams.get(cam_id)
        if cam is None:
            continue
        for idx, world_xyz in world_xyz_by_idx.items():
            px = project_world_to_pixel(world_xyz, cam["intrinsic"], cam["extrinsic"])
            if px is None:
                continue
            updates_by_cam.setdefault(cam_id, {})[idx] = (px[0] / cam["width"], px[1] / cam["height"])

    for cam_id, updates in updates_by_cam.items():
        points = [
            {"index": idx, "x": x, "y": y, "z": 0.0, "visibility": 1.0, "region": region_label}
            for idx, (x, y) in sorted(updates.items())
        ]
        out_path = settings.facescape_landmarks_root / actor_id / expression_id / f"cam_{cam_id}" / filename
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"frame_idx": 0, "actor_id": actor_id, "expression_id": expression_id, "camera_id": f"cam_{cam_id}", "points": points}
        tmp_path = out_path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(payload), encoding="utf-8")
        tmp_path.replace(out_path)

    return len(updates_by_cam)


def write_neutral_skin_frames(
    actor_id: str,
    neutral_expression_id: str,
    cams: dict[str, dict],
    skin_world_xyz: dict[int, np.ndarray],
    output_camera_ids: list[str],
    settings,
    dry_run: bool,
) -> bool:
    """Writes the neutral-reference expression's own skin_frame_000000.json
    directly from skin_world_xyz -- its skin positions ARE those vertices'
    positions on its own already-wrapped mesh (ground truth, not tracked),
    same reasoning as wrap_script/backfill_neutral_skin_landmarks.py's
    Nersemble equivalent. The standard-index groups (ear/nosebridge/
    undernose/eyebrow/chin) have their OWN equivalent step now --
    write_neutral_standard_frames(), called alongside this one -- since their
    neutral-frame group files turned out to need writing too (see that
    function's own docstring for why "nothing new to write" was wrong).

    Unconditional on every run (not skip-if-exists), matching this script's
    existing overwrite behavior for every other expression's skin_frame files."""
    tag = f"[actor={actor_id} expression={neutral_expression_id}]"
    if dry_run:
        print(f"  {tag}: would write {len(skin_world_xyz)} ground-truth skin point(s) [dry-run]")
        return True
    written = write_points_frames(actor_id, neutral_expression_id, cams, skin_world_xyz, output_camera_ids, settings)
    print(f"  DONE {tag}: wrote ground-truth skin landmarks for {written} camera(s)")
    return written > 0


def write_neutral_standard_frames(
    actor_id: str,
    neutral_expression_id: str,
    cams: dict[str, dict],
    output_camera_ids: list[str],
    settings,
    dry_run: bool,
) -> bool:
    """Writes the neutral expression's own <group>_frame_000000.json for
    every group in FACESCAPE_STANDARD_INDEX_REGION_GROUPS (ear/nosebridge/
    undernose/eyebrow/chin) -- triangulated straight from its own per-camera
    seed positions (load_standard_frame_points, which now also merges in
    ear_frame_000000.json for indices 2/9). There's no OTHER frame to track
    from/to for the neutral itself, unlike propagate_expression()'s
    per-target-expression tracking, so this is a one-shot triangulation, not
    a RAFT/CoTracker call.

    write_neutral_skin_frames' own docstring used to claim the standard-index
    groups need no such step, since their neutral-frame positions "already
    exist in the neutral's own frame_000000.json" -- true for the indices
    with a real sapiens2 mapping, but false for 2/9 (no sapiens2 source at
    all) and, more importantly, irrelevant either way: propagate_expression()
    only ever writes group files for TARGET expressions, never the neutral
    source itself, so without this the neutral's own group files stay
    permanently empty. That then reads as missing dots in any render that
    suppresses PROPAGATED_STANDARD_INDICES' base whitelist entries in favor
    of the (permanently absent) group file for that one expression -- see
    render_landmarks_facescape.py's --show-standard-groups."""
    tag = f"[actor={actor_id} expression={neutral_expression_id}]"
    frame_points_cache: dict[str, dict[int, tuple[float, float]]] = {}

    def triangulate_indices(indices: list[int], region_cams: list[str]) -> dict[int, np.ndarray]:
        result: dict[int, np.ndarray] = {}
        cam_index_order = [c for c in region_cams if c in cams]
        if not cam_index_order:
            return result
        intrinsic_arr = np.stack([cams[c]["intrinsic"] for c in cam_index_order])
        extrinsic_list = [cams[c]["extrinsic"] for c in cam_index_order]
        width = cams[cam_index_order[0]]["width"]
        height = cams[cam_index_order[0]]["height"]
        for idx in indices:
            observations = []
            for cam_id in cam_index_order:
                points = frame_points_cache.get(cam_id)
                if points is None:
                    points = load_standard_frame_points(
                        actor_id, neutral_expression_id, cam_id, cams[cam_id]["width"], cams[cam_id]["height"], settings,
                    )
                    frame_points_cache[cam_id] = points
                pixel_xy = points.get(idx)
                if pixel_xy is None:
                    continue
                observations.append((cam_index_order.index(cam_id), np.array(pixel_xy), True))
            if len(observations) < 2:
                continue
            fused = triangulate_point(observations, intrinsic_arr, extrinsic_list, width, height)
            if fused is not None:
                result[idx] = fused
        return result

    any_written = False
    for group_name, region_config, file_prefix in FACESCAPE_STANDARD_INDEX_REGION_GROUPS:
        group_world_xyz: dict[int, np.ndarray] = {}
        total_indices = 0
        for region in region_config.values():
            total_indices += len(region["indices"])
            group_world_xyz.update(triangulate_indices(region["indices"], region["cameras"]))
        print(f"  {tag}: triangulated {len(group_world_xyz)}/{total_indices} neutral {group_name} point(s)")
        if dry_run or not group_world_xyz:
            continue
        written = write_points_frames(
            actor_id, neutral_expression_id, cams, group_world_xyz, output_camera_ids, settings,
            region_label=group_name, filename=f"{file_prefix}_frame_000000.json",
        )
        any_written = any_written or written > 0
    return any_written


def propagate_expression(
    actor_id: str,
    expression_id: str,
    neutral_expression_id: str,
    raft_model: RaftModel,
    cotracker_model: CoTrackerPredictor,
    raft_global_indices: set[int],
    settings,
    cams: dict[str, dict],
    skin_world_xyz: dict[int, np.ndarray],
    indices_by_region: dict[str, list[int]],
    output_camera_ids: list[str],
    dry_run: bool,
) -> bool:
    tag = f"[actor={actor_id} expression={expression_id}]"

    # --- Build every camera's neutral-frame seeds: skin (reprojected from
    # the wrapped mesh) + standard-index groups (read straight off the
    # neutral's own frame_000000.json, per-camera, no 3D point needed). ---
    neutral_seed_by_cam: dict[str, dict[int, tuple[float, float]]] = {}

    all_skin_cameras = sorted({c for cams_list in FACESCAPE_SKIN_REGION_CAMERAS.values() for c in cams_list}, key=int)
    for cam_id in all_skin_cameras:
        cam = cams.get(cam_id)
        if cam is None:
            continue
        seeds: dict[int, tuple[float, float]] = {}
        for region, region_cams in FACESCAPE_SKIN_REGION_CAMERAS.items():
            if cam_id not in region_cams:
                continue
            for idx in indices_by_region[region]:
                px = project_world_to_pixel(skin_world_xyz[idx], cam["intrinsic"], cam["extrinsic"])
                if px is not None:
                    seeds[idx] = px
        if seeds:
            neutral_seed_by_cam.setdefault(cam_id, {}).update(seeds)

    all_standard_cameras: set[str] = set()
    for _group_name, region_config, _file_prefix in FACESCAPE_STANDARD_INDEX_REGION_GROUPS:
        for region in region_config.values():
            all_standard_cameras.update(region["cameras"])
    standard_frame_cache: dict[str, dict[int, tuple[float, float]]] = {}
    for cam_id in sorted(all_standard_cameras, key=int):
        if cam_id not in cams:
            continue
        frame_points = standard_frame_cache.setdefault(
            cam_id, load_standard_frame_points(
                actor_id, neutral_expression_id, cam_id, cams[cam_id]["width"], cams[cam_id]["height"], settings,
            ),
        )
        seeds = {}
        for _group_name, region_config, _file_prefix in FACESCAPE_STANDARD_INDEX_REGION_GROUPS:
            for region in region_config.values():
                if cam_id not in region["cameras"]:
                    continue
                for idx in region["indices"]:
                    if idx in frame_points:
                        seeds[idx] = frame_points[idx]
        if seeds:
            neutral_seed_by_cam.setdefault(cam_id, {}).update(seeds)

    # --- Up to two calls per camera (RAFT subset + CoTracker subset),
    # covering every index that camera needs (skin + every standard-index
    # group together), split per point per RAFT_SKIN_VERTEX_INDICES/
    # RAFT_STANDARD_GROUPS -- same per-vertex/per-group routing as
    # Nersemble's own process_target_sequence. ---
    tracked_by_cam: dict[str, dict[int, tuple[np.ndarray, float]]] = {}
    neutral_img_cache: dict[str, np.ndarray] = {}
    for cam_id, seeds in neutral_seed_by_cam.items():
        neutral_img = neutral_img_cache.get(cam_id)
        if neutral_img is None:
            neutral_img = load_image(actor_id, neutral_expression_id, cam_id, settings)
            if neutral_img is None:
                continue
            neutral_img_cache[cam_id] = neutral_img
        target_img = load_image(actor_id, expression_id, cam_id, settings)
        if target_img is None:
            print(f"  NOTE {tag}: missing image for camera {cam_id}, skipping that camera")
            continue

        raft_seeds = {idx: xy for idx, xy in seeds.items() if idx in raft_global_indices}
        cotracker_seeds = {idx: xy for idx, xy in seeds.items() if idx not in raft_global_indices}
        cam_tracked: dict[int, tuple[np.ndarray, float]] = {}
        if raft_seeds:
            cam_tracked.update(track_points_one_camera(raft_model, neutral_img, target_img, raft_seeds))
        if cotracker_seeds:
            cam_tracked.update(track_points_one_camera(cotracker_model, neutral_img, target_img, cotracker_seeds))
        tracked_by_cam[cam_id] = cam_tracked

    def triangulate_indices(indices: list[int], region_cams: list[str]) -> dict[int, np.ndarray]:
        result: dict[int, np.ndarray] = {}
        cam_index_order = [c for c in region_cams if c in cams]
        if not cam_index_order:
            return result
        intrinsic_arr = np.stack([cams[c]["intrinsic"] for c in cam_index_order])
        extrinsic_list = [cams[c]["extrinsic"] for c in cam_index_order]
        width = cams[cam_index_order[0]]["width"]
        height = cams[cam_index_order[0]]["height"]
        for idx in indices:
            observations = []
            for cam_id in region_cams:
                result_tracked = tracked_by_cam.get(cam_id, {}).get(idx)
                if cam_id not in cams or result_tracked is None:
                    continue
                pixel_xy, _visibility = result_tracked
                observations.append((cam_index_order.index(cam_id), pixel_xy, True))
            if len(observations) < 2:
                print(f"  NOTE {tag}: idx={idx} only has {len(observations)} usable "
                      f"observation(s), skipping this point for this expression")
                continue
            fused = triangulate_point(observations, intrinsic_arr, extrinsic_list, width, height)
            if fused is not None:
                result[idx] = fused
        return result

    # --- Skin: triangulate + write (unchanged behavior/filename). ---
    skin_world_xyz_target: dict[int, np.ndarray] = {}
    for region, indices in indices_by_region.items():
        skin_world_xyz_target.update(triangulate_indices(indices, FACESCAPE_SKIN_REGION_CAMERAS[region]))
    print(f"  {tag}: triangulated {len(skin_world_xyz_target)}/{sum(len(v) for v in indices_by_region.values())} skin point(s)")

    # --- Each standard-index group: triangulate + write its own dedicated file. ---
    any_standard_written = False
    for group_name, region_config, file_prefix in FACESCAPE_STANDARD_INDEX_REGION_GROUPS:
        group_world_xyz: dict[int, np.ndarray] = {}
        total_indices = 0
        for region in region_config.values():
            total_indices += len(region["indices"])
            group_world_xyz.update(triangulate_indices(region["indices"], region["cameras"]))
        print(f"  {tag}: triangulated {len(group_world_xyz)}/{total_indices} {group_name} point(s)")
        if dry_run or not group_world_xyz:
            continue
        written = write_points_frames(
            actor_id, expression_id, cams, group_world_xyz, output_camera_ids, settings,
            region_label=group_name, filename=f"{file_prefix}_frame_000000.json",
        )
        any_standard_written = any_standard_written or written > 0

    if dry_run:
        return len(skin_world_xyz_target) > 0

    written = write_points_frames(actor_id, expression_id, cams, skin_world_xyz_target, output_camera_ids, settings)
    print(f"  DONE {tag}: wrote skin landmarks for {written} camera(s)")
    return written > 0 or any_standard_written


def discover_unlabeled_actors(settings) -> list[str]:
    """Every FaceScape actor_id with at least one expression currently
    "unlabeled" in label_tracker.json -- the no-actor-arg sweep's own
    per-actor work list, same idea as
    wrap_script/run_neutral_cotracker_propagation.py's discover_unlabeled_pairs()."""
    all_statuses = label_tracker.get_all(dataset="FaceScape")
    return sorted(
        (actor_id for actor_id, expressions in all_statuses.items()
         if any(status == "unlabeled" for status in expressions.values())),
        key=int,
    )


@contextlib.contextmanager
def _expression_pair_lock(landmarks_root: Path, actor_id: str, expression_id: str):
    """Exclusive, non-blocking per-(actor,expression) flock --
    landmarks_root/<actor>/<expression>/.skin_pair.lock -- so multiple
    concurrent runs (different cluster nodes, or a full sweep racing a
    targeted --expressions run) never propagate into the same expression at
    once. Same pattern as wrap_script/run_neutral_cotracker_propagation.py's
    _sequence_pair_lock. Yields True if the lock was acquired (caller should
    proceed), False if another run already holds it (caller should skip this
    expression)."""
    lock_dir = landmarks_root / actor_id / expression_id
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_file = open(lock_dir / ".skin_pair.lock", "w")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        lock_file.close()
        yield False
        return
    try:
        yield True
    finally:
        fcntl.flock(lock_file, fcntl.LOCK_UN)
        lock_file.close()


def ensure_forehead_shift(actor_id: str, neutral_expression_id: str, mesh, assigned: list, skin_world_xyz: dict,
                          settings) -> dict:
    """The actor's forehead shift (forehead_shift.py), searched once and kept in
    <output>/<actor>/<neutral>/forehead_shift.json: the forehead skin points of the neutral wrap moved down over
    it by d, d found by secant search on trial neutral wraps (run_pipeline with skin_override = every skin point
    of this wrap, forehead ones shifted; no label change, no renders) until the forehead ratio is within
    forehead_shift.TOLERANCE of forehead_shift.TARGET_RATIO (at most MAX_STEPS wraps; the closest is kept). The
    neutral wrap is restored after every trial. Reused when done for the same target, so a rerun never shifts
    an already shifted wrap again."""
    import trimesh
    out_dir = settings.facescape_output_root / actor_id / neutral_expression_id
    existing = forehead_shift.load(out_dir)
    if (existing and existing.get("status") == "done"
            and abs(existing.get("target_ratio", -1) - forehead_shift.TARGET_RATIO) < 1e-9):
        print(f"[actor={actor_id}] forehead shift d={existing['d']:.4f} (ratio {existing['ratio']:.3f}) from {forehead_shift.SHIFT_NAME}")
        return existing
    target = forehead_shift.TARGET_RATIO
    verts = np.asarray(mesh.vertices, dtype=np.float64); faces = np.asarray(mesh.faces, dtype=np.int64)
    rows = [g for g, _region, v in assigned if v in forehead_shift.FOREHEAD_VERTICES]
    base_pts = np.stack([skin_world_xyz[g] for g in rows])
    length = forehead_shift.brow_chin_length(verts); r0 = forehead_shift.forehead_ratio(verts)[0]
    if r0 <= target:  # down only: a forehead already at or below the target is left as is
        payload = forehead_shift.no_shift_payload(r0, target); forehead_shift.save(out_dir, payload)
        print(f"[actor={actor_id}] forehead ratio {r0:.3f} <= target {target:.3f} -- no forehead shift (down only)")
        return payload
    wrapped = out_dir / "wrapped_mesh.obj"; keep = out_dir / "forehead_shift_base_wrapped_mesh.obj"
    shutil.copy2(wrapped, keep)
    history = [(0.0, r0)]; best = None
    d = forehead_shift.next_distance(history, target, length)
    print(f"[actor={actor_id}] forehead shift search -- neutral ratio {r0:.3f}, target {target:.3f}, rows {rows}")
    try:
        for step in range(1, forehead_shift.MAX_STEPS + 1):
            pts = forehead_shift.shift_down(base_pts, verts, faces, d)
            override = dict(skin_world_xyz); override.update(dict(zip(rows, pts)))
            wrap_pipeline(actor_id, neutral_expression_id, skin_override=override, mark=False, render=False)
            r = forehead_shift.forehead_ratio(np.asarray(trimesh.load(str(wrapped), process=False).vertices))[0]
            shutil.copy2(keep, wrapped)
            history.append((d, r))
            print(f"[actor={actor_id}] forehead shift step {step}: d={d:.4f} ({d / length:.3f} brow-chin) -> ratio {r:.3f}")
            if best is None or abs(r - target) < abs(best[1] - target):
                best = (d, r, pts)
            if abs(r - target) < forehead_shift.TOLERANCE:
                break
            d = forehead_shift.next_distance(history, target, length)
    finally:
        shutil.copy2(keep, wrapped)
    payload = {"status": "done", "target_ratio": target, "d": best[0], "ratio": best[1], "d_over_brow_chin": best[0] / length,
               "rows": rows, "points_world": {str(i): p.tolist() for i, p in zip(rows, best[2])},
               "base_ratio": r0, "history": history, "computed": time.strftime("%Y-%m-%d %H:%M:%S")}
    forehead_shift.save(out_dir, payload)
    print(f"[actor={actor_id}] forehead shift d={best[0]:.4f} -> ratio {best[1]:.3f} (target {target:.3f})")
    return payload


def process_actor(
    actor_id: str,
    settings,
    raft_model: RaftModel,
    cotracker_model: CoTrackerPredictor,
    expressions_override: list[str] | None,
    dry_run: bool,
    invalid_pairs: set[tuple[str, str]],
) -> tuple[int, int, int]:
    """Runs the full skin-propagation pass for one actor: wraps this actor's
    neutral-reference expression if needed, writes its own ground-truth skin
    frames, then propagates to every target expression (either
    expressions_override, or every "unlabeled" expression for this actor).
    Returns (passed, failed, skipped) expression counts -- skipped counts
    expressions another concurrent run already held the per-expression lock
    for (see _expression_pair_lock), not failures. Models are shared across
    actors by the caller (main()'s sweep), matching
    script_run_neutral_cotracker_propagation.sh's own reasoning for sharing
    one warmed-up CoTracker across a whole run."""
    tag = f"[actor={actor_id}]"
    settings_images_root = settings.facescape_images_root
    actor_dir = settings_images_root / actor_id
    if not actor_dir.is_dir():
        print(f"No such actor under {settings_images_root}: {actor_id}")
        return 0, 0, 0

    # Normally NEUTRAL_EXPRESSION (1_neutral), but that pair's own Phase 1
    # alignment can fail like any other expression's -- see
    # resolve_neutral_expression()'s own docstring for the fallback order.
    # If NONE of 1_neutral or its fallbacks are usable for this actor, there
    # is nothing to seed skin propagation from at all: skip this actor
    # entirely without touching any of its labels (not even a failed attempt
    # to wrap/confirm anything), rather than guessing.
    neutral_expression_id = resolve_neutral_expression(actor_id, invalid_pairs)
    if neutral_expression_id is None:
        print(f"SKIP {tag}: {NEUTRAL_EXPRESSION} and every fallback "
              f"({', '.join(NEUTRAL_EXPRESSION_FALLBACKS)}) are all in invalid_pairs.json -- "
              "nothing usable to seed skin propagation from, leaving this actor's labels untouched")
        return 0, 0, 0
    if neutral_expression_id != NEUTRAL_EXPRESSION:
        print(f"NOTE {tag}: {NEUTRAL_EXPRESSION} is in invalid_pairs.json -- "
              f"using {neutral_expression_id} as this actor's neutral reference instead")

    neutral_wrapped = settings.facescape_output_root / actor_id / neutral_expression_id / "wrapped_mesh.obj"
    if not neutral_wrapped.exists():
        print(f"{tag} Wrapping {neutral_expression_id} first (missing {neutral_wrapped})...")
        wrap_pipeline(actor_id, neutral_expression_id)
        if not neutral_wrapped.exists():
            print(f"FAIL {tag}: {neutral_wrapped} still missing after wrapping {neutral_expression_id}")
            return 0, 0, 0

    if expressions_override:
        expressions = expressions_override
    else:
        candidate_expressions = sorted(
            p.name for p in actor_dir.iterdir()
            if p.is_dir() and p.name != neutral_expression_id and (p / "params.json").exists()
        )
        # Same automatic status gate as run_pipeline_batch.py's "unreviewed"
        # gate -- only propagate to pairs Phase 2 has actually finished
        # labeling ("unlabeled") and that haven't already been propagated
        # ("unreviewed") or moved further along. Only applies to this default
        # (no --expressions given) discovery -- an explicit --expressions
        # list is a deliberate target, not a batch sweep, and
        # propagate_expression()'s own confirm_labeled() call already
        # no-ops safely if a named expression isn't "unlabeled".
        all_statuses = label_tracker.get_all(dataset="FaceScape")
        expressions = [
            e for e in candidate_expressions
            if label_tracker.get_status(actor_id, e, dataset="FaceScape", all_statuses=all_statuses) == "unlabeled"
        ]
        print(f"{tag} Gated to label_tracker.json status 'unlabeled': "
              f"{len(candidate_expressions) - len(expressions)} expression(s) excluded, {len(expressions)} remaining")
    # Invalid pairs (no origin-space alignment) are propagated by
    # run_neutral_skin_propagation_invalid.py instead -- their raw camera
    # numbering doesn't match the canonical camera sets used here.
    invalid_targets = [e for e in expressions if (actor_id, e) in invalid_pairs]
    if invalid_targets:
        print(f"{tag} Skipping {len(invalid_targets)} invalid pair(s) (use run_neutral_skin_propagation_invalid.py): "
              f"{invalid_targets}")
        expressions = [e for e in expressions if (actor_id, e) not in invalid_pairs]
    print(f"{tag} Propagating skin landmarks: {len(expressions)} expression(s)")

    import trimesh
    mesh = trimesh.load(str(neutral_wrapped), process=False)
    regions = _skin_landmarks_module.load_skin_regions()
    assigned = assign_global_indices(regions)
    # forehead_shift.EXTRA_SKIN_VERTICES (e.g. vertex 668): extra skin rows after the shared ones, tracked with the
    # front region's cameras (run_pipeline.py appends the same rows).
    assigned = list(assigned) + [(_skin_landmarks_module.NUM_STANDARD_LANDMARKS + len(assigned) + k, "front", int(v))
                                 for k, v in enumerate(forehead_shift.EXTRA_SKIN_VERTICES)]
    skin_world_xyz = {global_idx: np.asarray(mesh.vertices[vertex_idx], dtype=np.float64) for global_idx, _region, vertex_idx in assigned}
    # Forehead shift (forehead_shift.py): forehead skin points moved down over the neutral wrap until a neutral
    # wrap's forehead ratio reaches forehead_shift.TARGET_RATIO; they seed the neutral's skin_frame files and the
    # propagation to every other expression.
    if not dry_run:
        forehead = ensure_forehead_shift(actor_id, neutral_expression_id, mesh, assigned, skin_world_xyz, settings)
        for idx, p in forehead_shift.shifted_points(forehead).items():
            if idx in skin_world_xyz:
                skin_world_xyz[idx] = p
    indices_by_region: dict[str, list[int]] = {region: [] for region in SKIN_REGIONS}
    for global_idx, region, _vertex_idx in assigned:
        indices_by_region[region].append(global_idx)
    print(f"{tag} Loaded {sum(len(v) for v in indices_by_region.values())} skin vertex indices "
          f"({', '.join(f'{r}={len(v)}' for r, v in indices_by_region.items())})")

    # Which global indices (skin's own 74+ AND any standard 0-73 index whose
    # STANDARD_INDEX_REGION_GROUPS group is in RAFT_STANDARD_GROUPS) route to
    # RAFT instead of CoTracker -- same split Nersemble's own
    # compute_raft_skin_global_indices()/RAFT_STANDARD_GROUPS make, just
    # combined into one set here since propagate_expression() only needs to
    # ask "is this global_idx RAFT or CoTracker", not which mechanism said so.
    raft_global_indices: set[int] = {
        global_idx for global_idx, _region, vertex_idx in assigned if vertex_idx in RAFT_SKIN_VERTEX_INDICES
    }
    for group_name, region_config, _file_prefix in FACESCAPE_STANDARD_INDEX_REGION_GROUPS:
        if group_name not in RAFT_STANDARD_GROUPS:
            continue
        for region in region_config.values():
            raft_global_indices.update(region["indices"])
    print(f"{tag} RAFT-routed global indices: {len(raft_global_indices)} "
          f"(everything else uses CoTracker)")

    cams = load_camera_params(actor_id, neutral_expression_id, settings)
    print(f"{tag} Loaded camera params for {len(cams)} camera(s)")

    if not settings.facescape_cameras_path.exists():
        print(f"FAIL {tag}: {settings.facescape_cameras_path} not found -- run "
              "wrap_script_facescape/labeling/compute_facescape_cameras.py first")
        return 0, 0, 0
    with open(settings.facescape_cameras_path) as f:
        facescape_cameras = json.load(f)
    # all_culled entries are ints in facescape_cameras.json, but cams (and
    # FACESCAPE_SKIN_REGION_CAMERAS) are keyed by str camera ids -- without
    # this, cams.get(cam_id) below always misses and nothing gets written.
    output_camera_ids = sorted({str(c) for c in facescape_cameras.get("all_culled", [])}, key=int)
    print(f"{tag} Reprojecting onto {len(output_camera_ids)} output camera(s) from {settings.facescape_cameras_path.name}")

    # The neutral-reference expression's own skin_frame_000000.json --
    # ground-truth reprojection, no CoTracker needed, so this runs regardless
    # of what expressions_override was given (it's a separate operation from
    # the propagation loop below, not one more expression in it). Its own
    # standard-index-group files (ear/nosebridge/undernose/eyebrow/chin) get
    # the same treatment right after -- see write_neutral_standard_frames'
    # own docstring for why they need it too.
    neutral_ok = write_neutral_skin_frames(actor_id, neutral_expression_id, cams, skin_world_xyz, output_camera_ids, settings, dry_run)
    write_neutral_standard_frames(actor_id, neutral_expression_id, cams, output_camera_ids, settings, dry_run)
    if neutral_ok and not dry_run:
        # The neutral-reference expression is never one of the
        # propagate_expression() targets below (it's the source, not a
        # target), so without this it would never reach "unreviewed" and
        # Phase 4's automatic "unreviewed"-only gate would never wrap it,
        # even though it's one of the normally-valid expressions for an actor.
        try:
            label_tracker.confirm_labeled(actor_id, neutral_expression_id, dataset="FaceScape")
        except ValueError as exc:
            print(f"  NOTE: not moving {actor_id}/{neutral_expression_id} to 'unreviewed' -- {exc}")

    passed = 0
    failed = 0
    skipped = 0
    for expression_id in expressions:
        with _expression_pair_lock(settings.facescape_landmarks_root, actor_id, expression_id) as acquired:
            if not acquired:
                print(f"SKIP [actor={actor_id} expression={expression_id}]: expression is locked by "
                      "another run already processing it")
                skipped += 1
                continue
            ok = propagate_expression(
                actor_id, expression_id, neutral_expression_id, raft_model, cotracker_model, raft_global_indices,
                settings, cams, skin_world_xyz, indices_by_region, output_camera_ids, dry_run,
            )
        if ok:
            passed += 1
            # Phase 3's real output signal: "unlabeled" (Phase 2's sapiens2
            # landmarks) -> "unreviewed" (ready to wrap), same "unlabeled" ->
            # "unreviewed" transition as Nersemble's own propagation script.
            # Only fires from "unlabeled" -- confirm_labeled() raises
            # otherwise (e.g. a rerun on an already-"unreviewed"/"unconfirmed"/
            # "confirmed" pair), which is fine here, not a bug: this pair's
            # skin points just got refreshed, its status doesn't need to move.
            if not dry_run:
                try:
                    label_tracker.confirm_labeled(actor_id, expression_id, dataset="FaceScape")
                except ValueError as exc:
                    print(f"  NOTE: not moving {actor_id}/{expression_id} to 'unreviewed' -- {exc}")
        else:
            failed += 1

    print(f"{tag} Done. expressions={len(expressions)} passed={passed} failed={failed} skipped={skipped}")
    return passed, failed, skipped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "actor_id", nargs="?", default=None,
        help="Actor identifier; omit to sweep every actor with at least one 'unlabeled' expression "
             "in label_tracker.json (mirrors wrap_script/run_neutral_cotracker_propagation.py's own "
             "no-actor-arg sweep)",
    )
    parser.add_argument(
        "--expressions", nargs="*", default=None,
        help="Expressions to propagate to (default: every 'unlabeled' expression found for this "
             "actor). Requires a specific actor_id -- not valid together with a full sweep.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Triangulate but don't write any *_frame_000000.json files")
    args = parser.parse_args()

    if args.expressions and args.actor_id is None:
        print("--expressions requires a specific actor_id (not a full sweep)")
        return 1

    settings = get_settings()

    if args.actor_id is not None:
        actor_ids = [args.actor_id]
    else:
        actor_ids = discover_unlabeled_actors(settings)
        print(f"No actor given -- sweeping {len(actor_ids)} actor(s) with at least one 'unlabeled' expression")
        if not actor_ids:
            print("Nothing to do.")
            return 0

    # Shared across every actor in the sweep -- same reasoning as
    # script_run_neutral_cotracker_propagation.sh's own comment on why
    # Nersemble's equivalent shares one warmed-up CoTracker across its whole run.
    raft_model = load_raft_model()
    cotracker_model = load_cotracker_model()
    torch.autograd.set_grad_enabled(False)

    # Loaded once for the whole run (rather than per actor) -- same reasoning
    # as everywhere else this file loads something shared across the sweep.
    invalid_pairs = load_invalid_pairs(settings.facescape_invalid_pairs_path)
    print(f"Loaded {len(invalid_pairs)} invalid pair(s) from {settings.facescape_invalid_pairs_path}")

    total_passed = 0
    total_failed = 0
    total_skipped = 0
    for actor_id in actor_ids:
        passed, failed, skipped = process_actor(
            actor_id, settings, raft_model, cotracker_model, args.expressions, args.dry_run, invalid_pairs,
        )
        total_passed += passed
        total_failed += failed
        total_skipped += skipped

    print(f"Done. actors={len(actor_ids)} expressions_passed={total_passed} expressions_failed={total_failed} "
          f"expressions_skipped={total_skipped}")
    return 0 if total_failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
