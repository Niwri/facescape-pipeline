#!/usr/bin/env python3
"""For every actor that has a neutral frame saved (landmarks_root/<actor>/
neutral_frame.json, {"sequence_id": ..., "frame_idx": ...} -- ONE neutral frame
per actor, not one per actor/sequence pair), wrap that neutral frame only,
sample extra "skin" landmark positions from that wrapped mesh, and propagate
those skin points (plus several standard-index landmark groups -- ear,
nosebridge, undernose, eyebrow, chin -- see STANDARD_INDEX_REGION_GROUPS)
into every "unlabeled" sequence that actor has (or, with --all-sequences,
every sequence regardless of label status -- see that flag's own help text).

Point tracking uses BOTH CoTracker (joint multi-frame point tracking) and
RAFT (dense two-frame optical flow, from https://github.com/princeton-vl/raft,
via torchvision's built-in port + pretrained weights), split per point/group:
RAFT_SKIN_VERTEX_INDICES and RAFT_STANDARD_GROUPS say exactly which
vertices/groups use RAFT -- everything else uses CoTracker. This split, and
RAFT's own anchor-frame-then-bidirectional-chaining propagation strategy
(find the frame closest to neutral by direct forward-backward flow, then
chain outward from THAT frame in both directions with per-hop re-anchoring
against the true neutral seed), were worked out through direct comparison
against CoTracker-only propagation and two earlier RAFT strategies -- naive
frame-0-anchored chaining (locked onto the wrong region whenever a
sequence's first frame was a very different expression from neutral) and
single-hop-always (fixed that, but left visible, bounded sub-threshold
jitter with no temporal coupling between frames) -- before landing here --
see run_model_batch()'s own dispatch-by-model-type docstring, and
_run_model_batch_raft's docstring for exactly how RAFT is
adapted to the (frames, queries) -> (tracks, visibility) contract every
other mechanism in this file (forward/backward propagation within the
neutral sequence, cross-sequence propagation, skin/standard-index region
handling, triangulation/fusion, label_tracker.json transitions) talks to a
model through, indifferent to which one it actually got.

For the neutral frame's own sequence, propagation runs forward and backward
from the neutral frame across that sequence's own video (track_and_fuse(),
unchanged from the original one-neutral-per-pair design). For every OTHER
sequence, there's no local neutral frame to start from -- instead, the
neutral sequence's own single delegation frame is prepended as frame 0 of a
fresh CoTracker clip, followed by that OTHER sequence's own frames [0, N) in
order, then tracked forward only (track_and_fuse_from_other_sequence()) --
there's no "before" direction to propagate across two different sequences'
own timelines. See process_target_sequence()'s same_sequence branch.

The existing 0-73 sapiens2 landmarks (eyelid/lip and everything else) are
left untouched — this script only ever writes new "skin" and standard-index
landmark files:
  - landmarks_root/<actor>/<sequence>/cam_<camera>/skin_frame_<idx:06d>.json
    (points from skin_landmarks.py's three region files)
  - landmarks_root/<actor>/<sequence>/cam_<camera>/ear_frame_<idx:06d>.json
  - landmarks_root/<actor>/<sequence>/cam_<camera>/nosebridge_frame_<idx:06d>.json
  - landmarks_root/<actor>/<sequence>/cam_<camera>/undernose_frame_<idx:06d>.json
  - landmarks_root/<actor>/<sequence>/cam_<camera>/eyebrow_frame_<idx:06d>.json
    (all four use EXISTING standard 0-73 indices, see
    STANDARD_INDEX_REGION_GROUPS -- never overwrites the sapiens2-detected
    frame_<idx>.json for these same indices)

This does NOT wrap any non-neutral frame — that's run_pipeline_batch_all.py's
job, once this script has written the skin/standard-index landmark files it
depends on.

Once per actor, alongside its FLAME face mask, the actor's NEUTRAL frame gets
a scan face mask and a filtered landmark JSON (ensure_neutral_landmark_filter;
scan_face_masking.py: SAM prompted at the nosebridge on the front cameras;
every standard and skin landmark is checked in 2D against those masks, with a
30 px boundary margin, in the cameras where the scan doesn't occlude it --
dropped when outside in most of them, or when occluded in all of them;
eyelids, lips, nosebridge and undernose are never dropped).
run_pipeline_all.py leaves those indices out of EVERY frame's correspondence
map, like the neutral FLAME mask. --mask-only runs just the masking (FLAME
mask if missing, neutral-frame filter recomputed), with no propagation and no
label change; --all-frames / --frame additionally write per-frame filtered
JSONs for inspection.

Once a sequence's propagation completes, its label_tracker.json status is
moved from "unlabeled" to "unreviewed" (equivalent to clicking "Confirm
Labeled" in the frontend) — skipped entirely in --dry-run.
"""
from __future__ import annotations

import nersemble_paths
import argparse
import contextlib
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import trimesh

from backend.app.core.config import get_settings
from ignore_lists import IGNORE_ACTOR_SEQUENCE_PAIRS, IGNORE_ACTORS
from sequence_blacklist import BLACKLISTED_SEQUENCES
from skin_landmarks import SKIN_REGIONS, NUM_STANDARD_LANDMARKS, assign_global_indices, load_skin_regions
import scan_face_masking

FACEVIEW_ROOT = nersemble_paths.FACEVIEW_ROOT

# torchvision's own RAFT port (https://github.com/princeton-vl/raft), not a
# clone of the original repo -- same architecture/weights lineage, but
# avoids that repo's Google-Drive-gated checkpoint download and lets this
# script stay dependency-free beyond torchvision, which faceview_venv
# already has. RaftModel is just a type-hint alias (torchvision doesn't
# export a friendlier public class name for it).
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights, RAFT as RaftModel  # noqa: E402

# Both models coexist now: ear/nosebridge/undernose/eyebrow/chin, and MOST of
# skin, still use CoTracker ("usual" propagation, unchanged) -- only the
# specific cheek vertices in RAFT_SKIN_VERTEX_INDICES below use RAFT. See
# run_model_batch()'s own dispatch-by-model-type docstring.
if str(FACEVIEW_ROOT / "co-tracker") not in sys.path:
    sys.path.append(str(FACEVIEW_ROOT / "co-tracker"))
from cotracker.predictor import CoTrackerPredictor  # noqa: E402

OUTPUT_ROOT = nersemble_paths.OUTPUT_ROOT

# Face masks (four files each, <base> + FACE_MASK_SUFFIXES). Same paths as
# run_pipeline_all.py's STATIC_FACE_MASK_PATH / actor_face_mask_path() --
# duplicated rather than imported, since importing run_pipeline_all hides the
# GPU for the whole process.
FACE_MASK_SUFFIXES = (".txt", "_inverse.txt", "_wrapping_node.txt", "_wrapping_node_inverse.txt")
STATIC_FACE_MASK_PATH = Path(__file__).with_name("static_face_mask") / "face_list.txt"
FACE_RAY_MASKING_REQUIRED_MB = 6000
DEFAULT_CAMERAS_PATH = FACEVIEW_ROOT / "default_cameras.json"
COTRACKER_CHECKPOINT_PATH = FACEVIEW_ROOT / "co-tracker" / "checkpoints" / "scaled_offline.pth"
# Downloaded once via torchvision's Raft_Large_Weights.DEFAULT (which caches
# to ~/.cache/torch/hub/checkpoints/ on first use) and copied here so the
# checkpoint has a durable, explicit path like every other model this repo
# uses (see /scratch/ondemand32/irwinngo/models/).
CHECKPOINT_PATH = nersemble_paths.MODELS_ROOT / "raft_large_C_T_SKHT_V2-ff5fadd5.pth"

# Specific wrapped-mesh vertex indices (from keypoint_left.txt/keypoint_right.txt,
# NOT global landmark indices -- converted to global indices at runtime via
# assign_global_indices(), since that mapping depends on file contents/order)
# to track with RAFT instead of CoTracker. Everything else in "left"/"right",
# and all of "front", stays on CoTracker. Given directly, per request.
RAFT_SKIN_VERTEX_INDICES: set[int] = {
    # Left cheek
    2080, 1575, 3737, 3670, 3662, 3726, 3588, 3582,
    # Right cheek
    2711, 3115, 3484, 3471, 3438, 3432, 3386, 3390,
    # Below undernose -- the three keypoint_front.txt vertices sitting in a
    # row directly beneath the standard undernose landmarks (38/61/19), found
    # by projecting each front skin vertex into a front camera and matching
    # x-position to each nostril/center point with the smallest downward y
    # offset: 3490 (below left nostril), 3801 (below center), 3762 (below
    # right nostril).
    3490, 3801, 3762,
    # keypoint_under.txt's 3445/3678 were tried here (and in "under" itself)
    # then removed entirely, per request -- neither is tracked anywhere now.
}
RAFT_WEIGHTS = Raft_Large_Weights.DEFAULT

# IGNORE_ACTOR_SEQUENCE_PAIRS/IGNORE_ACTORS (see ignore_lists.py) are backed
# by ignore_actor_sequence_pairs.json/ignore_actors.json -- the same lists
# run_pipeline_batch_all.py reads, so the two scripts can't drift apart the
# way separate Python-constant copies could.

# Delegation frame index == raw frame index // RAW_FRAME_STRIDE, matching
# /scratch/ondemand32/irwinngo/FaceView/backend/app/core/config.py's RAW_FRAME_STRIDE and this repo's own
# run_pipeline_batch.py FRAME_STEP. Landmark JSON files, skin landmark JSON
# files, and CoTracker frame indices are all in this same (delegation) space.
RAW_FRAME_STRIDE = 5

# Ear landmarks -- EXISTING standard 0-73 indices (LandmarkOverlay.tsx's own
# LEFT_EAR=[2,4,1]/RIGHT_EAR=[9,14,6] groupings), propagated the same way as
# skin_landmarks.py's SKIN_REGIONS (own camera set in, every camera out) --
# given directly by the user, not derived from anything else in this repo.
EAR_REGIONS: dict[str, dict[str, object]] = {
    "left": {
        "indices": [2, 4, 1],
        "cameras": ["222200042", "222200044", "222200046", "222200040", "222200036"],
    },
    "right": {
        "indices": [9, 14, 6],
        "cameras": ["221501007", "222200039", "222200049", "222200043"],
    },
}

# Derived from SKIN_REGIONS' own per-region triangulation camera triples (see
# skin_landmarks.py) -- used by the standard-index region groups below
# (NOSEBRIDGE_REGIONS/EYEBROW_REGIONS, and UNDERNOSE_REGIONS' "center"
# sub-region). EAR_REGIONS above keeps its own bespoke camera lists,
# unchanged -- UNDERNOSE_REGIONS' nostril sub-regions deliberately reuse
# those same ear camera lists directly instead of these.
LEFT_CAMERAS = SKIN_REGIONS["left"]["cameras"]
FRONT_CAMERAS = SKIN_REGIONS["front"]["cameras"]
FRONT_CAMERAS_SET = set(FRONT_CAMERAS)
RIGHT_CAMERAS = SKIN_REGIONS["right"]["cameras"]
UNDER_CAMERAS = SKIN_REGIONS["under"]["cameras"]

# Nosebridge -- EXISTING standard 0-73 indices, whitelist.json's subset of
# LandmarkOverlay.tsx's NOSE=[60,62,54,57] minus 60 (now Undernose's
# "center" sub-region below) and 62 (not in whitelist.json). Single region
# (no left/right split), propagated the same way as EAR_REGIONS (own camera
# set in, every camera out).
NOSEBRIDGE_REGIONS: dict[str, dict[str, object]] = {
    "nosebridge": {
        "indices": [54, 57],
        "cameras": FRONT_CAMERAS,
    },
}

# Undernose -- EXISTING standard 0-73 indices, given directly by the user:
# right nostril (38), center (61), left nostril (19) -- each its own
# sub-region with its own single index, since each uses a different camera
# set (the nostrils track best on the same cameras as the ear on their own
# side; the center point is front-facing).
#
# NOTE: originally 37/18/60 here -- 37/18 are actually RIGHT_MOUTH_CORNER/
# LEFT_MOUTH_CORNER in render_landmarks.py's own REGION_COLORS (mirroring the
# frontend's LandmarkOverlay.tsx exactly), not nostrils, and 60 is NOSE's
# own index rather than UNDER_NOSE's -- corrected to 38/19/61, all three of
# render_landmarks.py's own UNDER_NOSE=[38,41,61,19,22] whitelist subset, per
# request, after the mislabeling surfaced as "undernose"-colored dots
# landing on the lip corners in a render.
UNDERNOSE_REGIONS: dict[str, dict[str, object]] = {
    "right_nostril": {
        "indices": [38],
        "cameras": EAR_REGIONS["right"]["cameras"],
    },
    "center": {
        "indices": [61],
        # Left+front+right UNIONed in per request (originally front-only).
        "cameras": LEFT_CAMERAS + FRONT_CAMERAS + RIGHT_CAMERAS,
    },
    "left_nostril": {
        "indices": [19],
        "cameras": EAR_REGIONS["left"]["cameras"],
    },
}

# Eyebrows -- EXISTING standard 0-73 indices, whitelist.json's subset of
# LandmarkOverlay.tsx's LEFT_BROW=[8,30,0,73,72] (0, 72 dropped) /
# RIGHT_BROW=[29,47,5,70,69] (5, 69 dropped). 0/5 are eye-corner points, not
# eyebrow at all. 72/69 ARE real eyebrow points with their own direct
# per-frame sapiens2 detection already (SAPIENS_LANDMARK_MAP: 72->87,
# 69->78, in process_sequence_sapiens.py, "both"/"sapiens_only" mode,
# recomputed every frame) -- confirmed purposeful: they don't need this
# RAFT/CoTracker propagation on top of an already-reliable direct detection.
# 8/30/73/29/47/70 are the ones actually propagated here.
EYEBROW_REGIONS: dict[str, dict[str, object]] = {
    "left": {
        "indices": [8, 30, 73],
        # Front UNIONed in per request -- eyebrows are also visible
        # front-on, and track_and_fuse's tracking_cameras list is both the
        # seed/track set AND the triangulation source (there's no separate
        # split), so this both propagates from front and triangulates with
        # it alongside the left cameras.
        "cameras": FRONT_CAMERAS + LEFT_CAMERAS,
    },
    "right": {
        "indices": [29, 47, 70],
        "cameras": FRONT_CAMERAS + RIGHT_CAMERAS,
    },
}

# Every standard-index region group propagated via load_ear_seeds()'s exact
# mechanism (own camera set in, every camera out, seeded from the neutral
# frame's own existing frame_<idx>.json sapiens2 detections) -- each writes
# to its own dedicated <file_prefix>_frame_<idx:06d>.json, never skin_frame
# and never merged with another group's file. Order here only affects print
# order, not written data.
# Chin -- new (not part of the original run_neutral_cotracker_propagation.py).
# No bespoke camera list exists for it (unlike EAR_REGIONS' hand-picked one),
# so it reuses FRONT_CAMERAS -- same simplification NOSEBRIDGE_REGIONS/
# UNDERNOSE_REGIONS' "center"/EYEBROW_REGIONS already use, and the same
# choice made for FaceScape's own chin group.
CHIN_REGIONS: dict[str, dict[str, object]] = {
    "chin": {
        "indices": [50],
        "cameras": FRONT_CAMERAS,
    },
}

STANDARD_INDEX_REGION_GROUPS: list[tuple[str, dict[str, dict[str, object]], str]] = [
    ("ear", EAR_REGIONS, "ear"),
    ("nosebridge", NOSEBRIDGE_REGIONS, "nosebridge"),
    ("undernose", UNDERNOSE_REGIONS, "undernose"),
    ("eyebrow", EYEBROW_REGIONS, "eyebrow"),
    ("chin", CHIN_REGIONS, "chin"),
]

# Which STANDARD_INDEX_REGION_GROUPS group_labels use RAFT instead of
# CoTracker (per request) -- everything else in the list above stays on
# CoTracker. Lip/eyelid groups were tried with RAFT and removed entirely
# (didn't track well) -- see git history if revisiting. Chin moved to RAFT
# per request after CoTracker's own frame-0 chin-tip tracking still looked
# wrong on inspection.
RAFT_STANDARD_GROUPS: set[str] = {"eyebrow", "undernose", "chin"}

# Delegation videos are always generated at exactly this size (see
# /scratch/ondemand32/irwinngo/FaceView/backend/app/services/loop_service.py's DELEGATION_SIZE and
# scripts/preprocess/process_sequence_sapiens.py's scale=550:802) -- width, height.
DELEGATION_SIZE = (550, 802)

# Rough required-free-memory threshold before a RAFT call, same idea as
# loop_service.wait_for_gpu_memory / run_pipeline.py's wait_for_gpu_memory.
# RAFT-large's own weights are ~20MB (vs CoTracker's much heavier multi-frame
# attention over a whole clip) and it only ever sees TWO frames at a time
# here, so this is a smaller, deliberately conservative threshold, not a
# measured one.
RAFT_REQUIRED_MB = 1500

# Original CoTracker threshold, unchanged from run_neutral_cotracker_propagation.py.
COTRACKER_REQUIRED_MB = 3000

device = "cuda" if torch.cuda.is_available() else "cpu"


def wait_for_gpu_memory(required_mb: float, label: str = "", poll_interval: float = 5.0, timeout: float | None = 600.0) -> None:
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
        print(f"Waiting for GPU memory{f' before {label}' if label else ''}: "
              f"{free_mb:.0f}/{required_mb:.0f} MiB free (waited {elapsed:.0f}s), retrying in {poll_interval:.0f}s...")
        time.sleep(poll_interval)


def load_default_cameras() -> list[str]:
    with DEFAULT_CAMERAS_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_model(neutral_only: bool) -> RaftModel | None:
    """Loads the RAFT checkpoint once, for reuse across every pair a caller
    processes -- pulled out of main() so other entry points (e.g.
    propagate_unreviewed_missing_skin.py) can load it once themselves instead of
    reloading it per pair by shelling out to this script as a subprocess.
    --neutral-only never calls track_and_fuse, so there's nothing for RAFT
    to do; returns None in that case.

    Loads torchvision's raft_large architecture, then overwrites its weights
    from the local checkpoint (rather than raft_large(weights=RAFT_WEIGHTS)'s
    own on-the-fly download) so this always uses the copy under
    /scratch/ondemand32/irwinngo/models/ -- the same one every other model in
    this repo is loaded from -- regardless of whichever machine's torch hub
    cache this happens to run on."""
    if neutral_only:
        return None
    model = raft_large(weights=None)
    state_dict = torch.load(str(CHECKPOINT_PATH), map_location=device)
    model.load_state_dict(state_dict)
    return model.to(device).eval()


def load_cotracker_model(neutral_only: bool) -> CoTrackerPredictor | None:
    """CoTracker counterpart of load_model() above -- ear/nosebridge/undernose/
    chin (STANDARD_INDEX_REGION_GROUPS minus RAFT_STANDARD_GROUPS) and most of
    skin (everything NOT in RAFT_SKIN_VERTEX_INDICES) are tracked with this
    model instead of RAFT's. Loaded once and shared the same way load_model()'s
    RAFT model is."""
    if neutral_only:
        return None
    return CoTrackerPredictor(checkpoint=str(COTRACKER_CHECKPOINT_PATH)).to(device)


def load_label_data(label_tracker_path: Path) -> dict:
    if not label_tracker_path.exists():
        return {}
    try:
        with label_tracker_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def get_pair_status(landmarks_root: Path, label_data: dict, actor_id: str, sequence_id: str) -> str:
    """Mirrors /scratch/ondemand32/irwinngo/FaceView/backend/app/services/label_tracker.py's LabelTracker.get_status:
    an explicit entry in label_tracker.json wins, otherwise fall back to whether
    this ACTOR has a neutral frame on disk (one per actor now, not one per
    actor/sequence pair — see load_actor_neutral_frame)."""
    status = label_data.get("Nersemble", {}).get(actor_id, {}).get(sequence_id)
    if status is not None:
        return status
    neutral_frame_path = landmarks_root / actor_id / "neutral_frame.json"
    return "unlabeled" if neutral_frame_path.exists() else "no_neutral_label"


def discover_unlabeled_actors(landmarks_root: Path, label_tracker_path: Path) -> list[str]:
    """Actors with a neutral_frame.json AND at least one non-blacklisted
    sequence whose status is currently "unlabeled" -- process_actor() itself
    re-derives the exact per-sequence target list (this is just the outer
    sweep's filter, same role discover_unlabeled_pairs played for the old
    per-pair design)."""
    label_data = load_label_data(label_tracker_path)
    actors: list[str] = []
    if not landmarks_root.is_dir():
        return actors
    for actor_dir in sorted(landmarks_root.iterdir()):
        if not actor_dir.is_dir() or not (actor_dir / "neutral_frame.json").exists():
            continue
        for sequence_dir in sorted(actor_dir.iterdir()):
            if not sequence_dir.is_dir() or sequence_dir.name in BLACKLISTED_SEQUENCES:
                continue
            if get_pair_status(landmarks_root, label_data, actor_dir.name, sequence_dir.name) == "unlabeled":
                actors.append(actor_dir.name)
                break
    return actors


def discover_actors_with_neutral_frame(landmarks_root: Path) -> list[str]:
    """--all-sequences' own sweep-mode discovery (per request): every actor
    with a neutral_frame.json AND at least one non-blacklisted sequence
    directory, regardless of that sequence's label_tracker.json status --
    unlike discover_unlabeled_actors(), does not filter by status at all.
    Still requires a neutral frame -- there is nothing to propagate from
    without one, bypass flag or not."""
    actors: list[str] = []
    if not landmarks_root.is_dir():
        return actors
    for actor_dir in sorted(landmarks_root.iterdir()):
        if not actor_dir.is_dir() or not (actor_dir / "neutral_frame.json").exists():
            continue
        for sequence_dir in sorted(actor_dir.iterdir()):
            if sequence_dir.is_dir() and sequence_dir.name not in BLACKLISTED_SEQUENCES:
                actors.append(actor_dir.name)
                break
    return actors


def confirm_labeled(label_tracker_path: Path, actor_id: str, sequence_id: str) -> None:
    """Moves this pair's label_tracker.json status from "unlabeled" to
    "unreviewed" -- the same lock-guarded, atomic-replace read-modify-write
    as /scratch/ondemand32/irwinngo/FaceView/backend/app/services/label_tracker.py's
    LabelTracker.confirm_labeled/_write, reimplemented here since that module
    lives in a separate backend package. A no-op (with a warning) if the
    pair isn't currently "unlabeled" -- e.g. it was already confirmed by
    someone else, or reset by a fresh "Save as Neutral" while this script
    was running.

    Locks a dedicated sibling .lock file rather than label_tracker.json
    itself, and writes via temp-file-then-replace rather than in-place
    truncate+dump -- see _write's docstring in label_tracker.py for why
    (the old truncate-then-dump-in-place pattern left the file empty/corrupt
    if the process died in between, e.g. an OOM kill, which is what actually
    caused label_tracker.json to appear "wiped" once)."""
    label_tracker_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = label_tracker_path.with_name(label_tracker_path.name + ".lock")
    with lock_path.open("w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            if label_tracker_path.exists():
                try:
                    data = json.loads(label_tracker_path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    data = {}
            else:
                data = {}
            current = data.get("Nersemble", {}).get(actor_id, {}).get(sequence_id, "unlabeled")
            if current != "unlabeled":
                print(f"NOTE: not moving {actor_id}/{sequence_id} to 'unreviewed' — "
                      f"current status is {current!r}, not 'unlabeled'")
                return
            data.setdefault("Nersemble", {}).setdefault(actor_id, {})[sequence_id] = "unreviewed"
            tmp_path = label_tracker_path.with_name(label_tracker_path.name + ".tmp")
            tmp_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            tmp_path.replace(label_tracker_path)
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)
    print(f"Moved {actor_id}/{sequence_id} to 'unreviewed' in {label_tracker_path}")


def load_actor_neutral_frame(landmarks_root: Path, actor_id: str) -> tuple[str, int] | None:
    """landmarks_root/<actor>/neutral_frame.json -- ONE neutral frame for the
    whole actor (see FaceView/backend/app/services/landmark_store.py's
    save_neutral_frame), not one per actor/sequence pair."""
    neutral_frame_path = landmarks_root / actor_id / "neutral_frame.json"
    if not neutral_frame_path.exists():
        return None
    with neutral_frame_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data["sequence_id"], data["frame_idx"]


def wrap_neutral_frame(actor_id: str, sequence_id: str, neutral_frame_id: int, resolution: str, pipeline_script: Path,
                       face_mask_path: Path = STATIC_FACE_MASK_PATH) -> bool:
    wrapped_mesh_path = OUTPUT_ROOT / actor_id / sequence_id / str(neutral_frame_id) / "wrapped_mesh.obj"
    cmd = [sys.executable, str(pipeline_script), actor_id, sequence_id, str(neutral_frame_id), resolution,
           "--face-mask-path", str(face_mask_path)]
    print("RUN  " + " ".join(cmd) + "  (neutral frame)")
    returncode = subprocess.run(cmd).returncode
    if returncode != 0:
        print(f"FAIL wrapping neutral frame {neutral_frame_id} for {actor_id}/{sequence_id}: exit_code={returncode}")
        return False
    if not wrapped_mesh_path.exists():
        print(f"FAIL wrapped_mesh.obj still missing after a successful run: {wrapped_mesh_path}")
        return False
    return True


def load_skin_world_xyz(wrapped_mesh_path: Path) -> tuple[dict[int, np.ndarray], dict[str, list[int]]]:
    mesh = trimesh.load(str(wrapped_mesh_path), process=False)
    regions = load_skin_regions()
    assigned = assign_global_indices(regions)  # [(global_idx, region, vertex_idx), ...]
    skin_world_xyz = {global_idx: np.asarray(mesh.vertices[vertex_idx], dtype=np.float64) for global_idx, _region, vertex_idx in assigned}
    indices_by_region: dict[str, list[int]] = {region: [] for region in SKIN_REGIONS}
    for global_idx, region, _vertex_idx in assigned:
        indices_by_region[region].append(global_idx)
    return skin_world_xyz, indices_by_region


def compute_raft_skin_global_indices() -> set[int]:
    """Converts RAFT_SKIN_VERTEX_INDICES (raw wrapped-mesh vertex indices, from
    keypoint_left.txt/keypoint_right.txt) into the GLOBAL landmark indices
    (74+) process_target_sequence's skin block actually keys everything by --
    that mapping depends on assign_global_indices()'s file-order-dependent
    assignment, so it's recomputed here rather than hardcoded. Independent of
    load_skin_world_xyz() (which needs a specific actor's wrapped mesh);
    this only needs the keypoint files themselves, so it's the same set for
    every actor/pair and only needs computing once per run."""
    assigned = assign_global_indices(load_skin_regions())  # [(global_idx, region, vertex_idx), ...]
    return {global_idx for global_idx, _region, vertex_idx in assigned if vertex_idx in RAFT_SKIN_VERTEX_INDICES}


def compute_skin_cameras_by_index() -> dict[int, list[str]]:
    """Per skin global index, the FULL UNION of camera sets from every region
    file its own vertex appears in -- NOT just the single region
    assign_global_indices() happened to keep (that function drops a vertex's
    duplicate appearance in a second/third region file, keeping only the
    first, since a global index needs to be unique; that's still correct for
    the index assignment itself, but was silently also deciding tracking
    cameras, which isn't the same question). A vertex in both
    keypoint_front.txt and keypoint_right.txt (e.g. vertex 3432, per request)
    should be tracked AND triangulated using BOTH regions' cameras together,
    not whichever one happened to be declared first in SKIN_REGIONS.

    Returns {global_idx: [camera_id, ...]} -- camera order follows
    SKIN_REGIONS' own declared front/right/left order. Used to build
    process_target_sequence's own camera-major seed dicts (see that
    function's own docstring for why grouping by CAMERA rather than by
    landmark group/region matters -- mirrors FaceScape's
    run_neutral_skin_propagation.py propagate_expression())."""
    regions = load_skin_regions()  # {region: [vertex_idx, ...]}
    assigned = assign_global_indices(regions)  # [(global_idx, region, vertex_idx), ...]
    vertex_to_global = {vertex_idx: global_idx for global_idx, _region, vertex_idx in assigned}

    cameras_by_global_idx: dict[int, list[str]] = {}
    for region in SKIN_REGIONS:  # front, right, left -- declared order
        region_cameras = SKIN_REGIONS[region]["cameras"]
        for vertex_idx in regions.get(region, []):
            global_idx = vertex_to_global.get(vertex_idx)
            if global_idx is None:
                continue
            existing = cameras_by_global_idx.setdefault(global_idx, [])
            for cam in region_cameras:
                if cam not in existing:
                    existing.append(cam)
    return cameras_by_global_idx


def load_camera_params(actor_id: str, settings) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    cam_data_path = settings.data_root / actor_id / "calibration" / "camera_params.json"
    with cam_data_path.open("r", encoding="utf-8") as f:
        cam_data = json.load(f)
    intrinsic = np.asarray(cam_data["intrinsics"], dtype=np.float64)
    world_2_cam = {cam: np.asarray(mat, dtype=np.float64) for cam, mat in cam_data["world_2_cam"].items()}
    return intrinsic, world_2_cam


def probe_video_size(video_path: Path) -> tuple[int, int]:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=s=x:p=0", str(video_path)],
        capture_output=True, text=True, check=True,
    )
    w, h = result.stdout.strip().split("x")
    return int(w), int(h)


def probe_video_frame_count(video_path: Path) -> int:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(video_path)],
        capture_output=True, text=True, check=True,
    )
    return int(result.stdout.strip())


def extract_delegation_frames(video_path: Path, start_frame: int, end_frame: int, width: int, height: int) -> np.ndarray:
    """[start_frame, end_frame) in delegation-frame space, no cropping."""
    num_frames = end_frame - start_frame
    if num_frames <= 0:
        return np.zeros((0, height, width, 3), dtype=np.uint8)
    filter_chain = f"trim=start_frame={start_frame}:end_frame={end_frame},setpts=PTS-STARTPTS"
    result = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(video_path), "-vf", filter_chain, "-pix_fmt", "rgb24", "-f", "rawvideo", "-"],
        capture_output=True, check=True,
    )
    frame_bytes = width * height * 3
    actual_frames = len(result.stdout) // frame_bytes
    frames = np.frombuffer(result.stdout, dtype=np.uint8, count=actual_frames * frame_bytes)
    frames = frames.reshape(actual_frames, height, width, 3)
    if actual_frames < num_frames:
        pad = np.repeat(frames[-1:], num_frames - actual_frames, axis=0) if actual_frames > 0 else np.zeros((num_frames, height, width, 3), dtype=np.uint8)
        frames = np.concatenate([frames, pad], axis=0)
    return frames


def _extract_delegation_frames_cached(
    cache: dict[tuple, np.ndarray], video_path: Path, start_frame: int, end_frame: int, width: int, height: int,
) -> np.ndarray:
    """Memoized extract_delegation_frames() -- ffmpeg decode is CPU-bound and
    genuinely expensive (a subprocess spawn + full video decode per call),
    and the exact same (video_path, frame range) gets requested repeatedly
    within one process_target_sequence() call: every STANDARD_INDEX_REGION_GROUPS
    sub-region and every skin camera group that happens to share a camera
    re-requests the SAME delegation-video range for that camera, since the
    "before"/"after" ranges only depend on neutral_delegation_idx/
    total_delegation_frames (fixed for the whole target sequence), not on
    which points are being tracked. Callers never mutate the returned array
    in place (see track_and_fuse()/track_and_fuse_from_other_sequence()'s own
    read-only usage), so sharing it by reference across callers is safe.

    `cache` is a plain dict the caller creates fresh per process_target_sequence()
    call (see that function's own frame_cache local) -- scoping it there means
    it never accumulates across sequences/actors in a long-running actor-sweep
    process; each sequence's decoded frames are released once that sequence's
    processing returns."""
    key = (str(video_path), start_frame, end_frame, width, height)
    cached = cache.get(key)
    if cached is None:
        cached = extract_delegation_frames(video_path, start_frame, end_frame, width, height)
        cache[key] = cached
    return cached


def camera_center_and_direction(pixel_xy: np.ndarray, intrinsic: np.ndarray, world_2_cam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r = world_2_cam[:3, :3]
    t = world_2_cam[:3, 3]
    center = -r.T @ t
    ray_cam = np.linalg.inv(intrinsic) @ np.array([pixel_xy[0], pixel_xy[1], 1.0], dtype=np.float64)
    ray_world = r.T @ ray_cam
    ray_world /= (np.linalg.norm(ray_world) + 1e-8)
    return center, ray_world


def triangulate_from_observations(observations: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray | None:
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


# Max allowed reprojection error (raw pixels) for a camera's own observation
# against the consensus fused point before that camera is treated as an
# outlier for this frame/index -- see _robust_triangulate.
REPROJECTION_ERROR_THRESHOLD_PX = 15.0


def _robust_triangulate(
    obs_list: list[tuple[str, np.ndarray, np.ndarray, float, float, np.ndarray]],
    intrinsic: np.ndarray,
    raw_width: int,
    raw_height: int,
    front_cameras: set[str],
) -> np.ndarray | None:
    """Per-frame/per-index outlier-rejecting triangulation, addressing the
    observed cross-camera trajectory disagreement: cameras track independently
    (see run_model_batch), so tracking error in one view has no way to get
    corrected by the others until fusion runs here.

    obs_list entries are (cam, center, direction, x_px, y_px, extrinsic) for
    every camera with a valid observation at this (frame, index).

    1) Triangulate from ALL observations (the original, un-robust behavior).
    2) Reproject that consensus point into each camera and measure pixel
       distance to that camera's own raw observation -- cameras within
       REPROJECTION_ERROR_THRESHOLD_PX are inliers.
    3) If 2+ inliers survive, re-triangulate from just those (dropping the
       disagreeing camera(s)).
    4) If fewer than 2 inliers survive -- too much mutual disagreement for a
       consensus vote to resolve -- default to trusting the front-camera
       observations specifically (per explicit instruction), re-triangulating
       from those alone when there are 2+ of them.
    5) Any case that still can't produce 2+ agreeing observations falls back
       to the plain all-observations fused point.
    """
    if len(obs_list) < 2:
        return None
    fused_all = triangulate_from_observations([(c, d) for _cam, c, d, _x, _y, _ext in obs_list])
    if fused_all is None:
        return None

    def reprojection_error_px(cam_extrinsic: np.ndarray, x_px: float, y_px: float) -> float:
        proj = project_world_to_norm(fused_all, intrinsic, cam_extrinsic, raw_width, raw_height)
        if proj is None:
            return float("inf")
        return float(np.hypot(proj[0] * raw_width - x_px, proj[1] * raw_height - y_px))

    inliers = [o for o in obs_list if reprojection_error_px(o[5], o[3], o[4]) <= REPROJECTION_ERROR_THRESHOLD_PX]
    if len(inliers) == len(obs_list):
        return fused_all  # full consensus, nothing disagreed

    if len(inliers) >= 2:
        refined = triangulate_from_observations([(o[1], o[2]) for o in inliers])
        if refined is not None:
            return refined

    front_obs = [o for o in obs_list if o[0] in front_cameras]
    if len(front_obs) >= 2:
        refined = triangulate_from_observations([(o[1], o[2]) for o in front_obs])
        if refined is not None:
            return refined

    return fused_all


def project_world_to_norm(world_xyz: np.ndarray, intrinsic: np.ndarray, world_2_cam: np.ndarray, width: int, height: int) -> tuple[float, float] | None:
    r = world_2_cam[:3, :3]
    t = world_2_cam[:3, 3]
    cam_xyz = r @ world_xyz + t
    if cam_xyz[2] <= 1e-8:
        return None
    img = intrinsic @ cam_xyz
    x = img[0] / img[2]
    y = img[1] / img[2]
    return float(x / width), float(y / height)


def _raft_flow(model: RaftModel, img0: np.ndarray, img1: np.ndarray) -> np.ndarray:
    """One RAFT call between exactly two frames -- RAFT has no concept of a
    multi-frame clip or a query point, only dense flow between a pair of
    images. Returns (2, H, W) flow (dx, dy) in the ORIGINAL (unpadded) pixel
    grid -- RAFT internally requires both dimensions to be multiples of 8, so
    inputs are edge-padded up to that before the call and the output flow is
    cropped back down afterward."""
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
    """Bilinearly samples a (2, H, W) flow field at a sub-pixel (x, y),
    clamped to the image bounds -- points can legitimately drift right up to
    (and slightly past, from RAFT's own error) the frame edge over many
    propagation steps."""
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


def run_model_batch(
    model: RaftModel | CoTrackerPredictor, frames: np.ndarray, queries: list[tuple[float, float]],
    raft_model_for_anchor: RaftModel | None = None, known_anchor_t: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Dispatches to whichever model this particular call was given -- every
    caller (track_and_fuse/track_and_fuse_from_other_sequence) stays generic
    over "some model with a (frames, queries) -> (tracks, visibility)
    contract", unaware of which one it actually got. process_target_sequence
    is what decides, per region/vertex, which model to pass in (see
    RAFT_SKIN_VERTEX_INDICES and STANDARD_INDEX_REGION_GROUPS' own call sites).

    ``raft_model_for_anchor``, when given AND `model` is CoTracker, picks a
    query frame for CoTracker the same way RAFT picks its own chain anchor
    (per request: "start from the same Frame A selection as RAFT"), AND
    queries CoTracker at that frame's own RAFT-flowed position -- NOT the
    original neutral seed position. A first attempt at this queried the
    right frame but the WRONG position (still the neutral's own (x,y), which
    is generally not where the point actually is at a non-neutral anchor
    frame -- CoTracker's query format means "this point IS at (x,y) at frame
    t", so that was telling it something false about its own query frame and
    visibly regressed tracking). Fixed by reusing _find_anchor_frame's own
    already-computed direct-from-neutral position AT the anchor frame as the
    query's (x,y) instead. Ignored when `model` is already RAFT (RAFT finds
    its own anchor internally, see _run_model_batch_raft).

    ``known_anchor_t``, when given, is forwarded to whichever path actually
    picks an anchor frame (both do, RAFT internally and CoTracker via
    `raft_model_for_anchor`) -- see process_target_sequence's own
    once-per-sequence global anchor search, and _run_model_batch_raft's/
    _run_model_batch_cotracker's own docstrings for what it skips."""
    if isinstance(model, RaftModel):
        return _run_model_batch_raft(model, frames, queries, known_anchor_t=known_anchor_t)
    return _run_model_batch_cotracker(model, frames, queries, raft_model_for_anchor=raft_model_for_anchor, known_anchor_t=known_anchor_t)


def _run_model_batch_cotracker(
    model: CoTrackerPredictor, frames: np.ndarray, queries: list[tuple[float, float]],
    raft_model_for_anchor: RaftModel | None = None, known_anchor_t: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """One CoTracker call over the whole (T, H, W, 3) clip, querying every
    point at a single query frame. Returns (tracks (T, N, 2), visibility
    (T, N)).

    Query frame/position defaults to (0, neutral seed) -- CoTracker's
    original behavior. When `raft_model_for_anchor` is given, both the query
    FRAME and the query POSITION change together: frame becomes
    _find_anchor_frame's pick (or the precomputed `known_anchor_t`, when
    valid -- skipping that scan same as _run_model_batch_raft), and position
    becomes that SAME already-computed RAFT-flowed position at that frame
    (see run_model_batch's own docstring for why both must change together,
    not just the frame). CoTracker's own joint multi-frame model natively
    tracks bidirectionally from wherever it's queried in one pass, so this
    is the entire change needed -- no chaining loop to add."""
    query_frame_idx = 0
    query_positions = queries
    if raft_model_for_anchor is not None and frames.shape[0] > 1:
        query_arr = np.asarray(queries, dtype=np.float64)
        if known_anchor_t is not None and 1 <= known_anchor_t <= frames.shape[0] - 1:
            query_frame_idx = known_anchor_t
            direct_candidate = _direct_candidate_at(raft_model_for_anchor, frames, query_arr, known_anchor_t)
            query_positions = [(float(x), float(y)) for x, y in direct_candidate]
        else:
            query_frame_idx, direct_candidates = _find_anchor_frame(raft_model_for_anchor, frames, query_arr)
            query_positions = [(float(x), float(y)) for x, y in direct_candidates[query_frame_idx]]

    wait_for_gpu_memory(COTRACKER_REQUIRED_MB, label=f"CoTracker (batch of {len(queries)} points, {frames.shape[0]} frames)")
    video_tensor = torch.from_numpy(frames).permute(0, 3, 1, 2)[None].float().to(device)
    query_tensor = torch.tensor([[float(query_frame_idx), x, y] for x, y in query_positions], dtype=torch.float32, device=device)[None]
    pred_tracks, pred_visibility = model(video_tensor, queries=query_tensor)
    if device == "cuda":
        torch.cuda.synchronize()
    del video_tensor
    tracks = pred_tracks[0].detach().cpu().numpy()  # (T, N, 2)
    visibility = pred_visibility[0].detach().cpu().numpy()  # (T, N)
    return tracks, visibility


# Forward-backward (cycle) consistency threshold (pixels, in the delegation
# frame's own pixel space -- see DELEGATION_SIZE) for _run_model_batch_raft's
# per-hop chain validation during Phase 2 (see that function's docstring):
# round-trip a chained hop's candidate position through the REVERSE flow
# to its own IMMEDIATE chain neighbor (frame_t -> frame_{t-step}, NOT all
# the way back to neutral -- a neutral-referenced check was tried and
# reverted, see chain_direction's own comment for why) and compare to that
# neighbor's own already-accepted position. A large round-trip error means
# this hop's flow can't be trusted (large non-rigid motion, occlusion,
# texture loss, RAFT locking onto the wrong region, ...), so the chain
# re-anchors at this frame instead of carrying the error forward. NOT used
# for Phase 1's anchor-frame selection, which uses argmin instead of a
# threshold (see _find_anchor_frame's own docstring).
RAFT_FB_CONSISTENCY_THRESHOLD_PX = 3.0


def _sample_all(flow: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Bilinearly samples `flow` at every row of `positions` ((N, 2) raw
    pixel coords) and adds it, i.e. flows every position forward by one
    RAFT flow field at once. Shared by _find_anchor_frame(), the known-anchor
    fast path (_direct_candidate_at()), and both _run_model_batch_raft()'s
    Phase 1/2."""
    return np.array(
        [np.add(positions[i], _sample_flow_bilinear(flow, positions[i][0], positions[i][1])) for i in range(positions.shape[0])],
        dtype=np.float64,
    )


def _direct_candidate_at(model: RaftModel, frames: np.ndarray, query_arr: np.ndarray, t: int) -> np.ndarray:
    """Forward-only RAFT flow from neutral (frames[0]) to frames[t], sampled
    at query_arr -- the same 'direct candidate' _find_anchor_frame computes
    per candidate frame during its scan, but for exactly one KNOWN frame
    instead of scanning every candidate. No backward/round-trip check here
    (that's only needed to SELECT a trustworthy anchor in the first place;
    once a frame is already known-good -- either the precomputed global
    anchor, see process_target_sequence, or a per-hop Phase 2 re-anchor
    fallback -- it's used as-is). Used to make Phase 1 cheap when the caller
    already knows which frame to anchor at (see _run_model_batch_raft's/
    _run_model_batch_cotracker's own known_anchor_t handling)."""
    flow_fwd = _raft_flow(model, frames[0], frames[t])
    return _sample_all(flow_fwd, query_arr)


def _find_anchor_frame(
    model: RaftModel, frames: np.ndarray, query_arr: np.ndarray,
) -> tuple[int, dict[int, np.ndarray]]:
    """Finds the frame closest to neutral (frames[0]) by direct forward-backward
    RAFT flow, for however many points `query_arr` holds -- used by
    _run_model_batch_raft's own Phase 1 whenever no known_anchor_t is given
    (see that function's docstring -- the common case now is a precomputed
    GLOBAL anchor, shared across every camera/group in one
    process_target_sequence() call, computed by calling this function only
    ONCE against a single reference camera; this per-call scan remains as
    the fallback for whenever that global anchor isn't available, e.g. the
    reference camera's own delegation video is missing). (Was also reused to
    pick CoTracker's own query frame for a while, per an earlier request --
    reverted, see run_model_batch's own docstring for why; CoTracker always
    queries frame 0 again now.) RAFT's dense flow field can be sampled at any
    pixel coordinate, so this works the same regardless of which model
    ultimately does the tracking.

    Flows EVERY candidate frame directly from neutral (frames[0] -> frames[t])
    and back (frames[t] -> frames[0]), records each frame's max-over-points
    round-trip error, and returns the frame with the LOWEST max error -- no
    threshold, no "does a qualifying frame exist" uncertainty, always
    well-defined. Also returns the direct-from-neutral candidate position for
    EVERY candidate frame (not just the chosen anchor), since callers that
    chain from the anchor (Phase 2) reuse these as their own re-anchor
    fallback rather than recomputing them."""
    T = frames.shape[0]
    neutral_frame = frames[0]

    direct_candidates: dict[int, np.ndarray] = {}
    direct_max_error: dict[int, float] = {}
    for t in range(1, T):
        flow_fwd = _raft_flow(model, neutral_frame, frames[t])
        candidate = _sample_all(flow_fwd, query_arr)
        flow_bwd = _raft_flow(model, frames[t], neutral_frame)
        roundtrip = _sample_all(flow_bwd, candidate)
        per_point_error = np.linalg.norm(roundtrip - query_arr, axis=1)
        direct_candidates[t] = candidate
        direct_max_error[t] = float(per_point_error.max())

    anchor_t = min(direct_max_error, key=direct_max_error.get)
    return anchor_t, direct_candidates


def _run_model_batch_raft(
    model: RaftModel, frames: np.ndarray, queries: list[tuple[float, float]], known_anchor_t: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """RAFT counterpart of CoTracker's own run_model_batch -- same (frames,
    queries) -> (tracks (T, N, 2), visibility (T, N)) contract, so every
    caller (track_and_fuse/track_and_fuse_from_other_sequence) is unchanged.

    Two phases, per request (this replaces both the original pure-chaining
    strategy AND the single-hop-from-neutral-always strategy that preceded
    this one -- see git history/prior docstrings here for why each was
    replaced in turn):

    Phase 1 -- find the best ANCHOR frame. Naive chaining assumed frame 1
    (frames[0]'s immediate neighbor) was a safe, close-to-neutral starting
    point for the chain; single-hop-always gave up on chaining entirely
    because that assumption broke whenever frame 0's own expression was
    already far from neutral. Instead: flow EVERY candidate frame directly
    from neutral (frames[0] -> frames[t]) and back (frames[t] -> frames[0]),
    record each frame's max-over-points round-trip error, and pick the frame
    with the LOWEST max error as the anchor -- no threshold, no "does a
    qualifying frame exist" uncertainty, always well-defined. This costs
    2 RAFT calls per candidate frame (same as the single-hop approach's own
    full-sequence cost), accepted per request since single-hop's own
    jitter -- confirmed via render to be BOUNDED, sub-RAFT_FB_CONSISTENCY_
    THRESHOLD_PX noise, not gross tracking failures, since nothing was ever
    invalidated -- was not acceptable on its own.

    Phase 2 -- chain outward from the anchor in both directions (anchor+1
    up to T-1, and anchor-1 down to 1; frame 0 is the neutral frame itself,
    already known exactly), frame-to-frame (frames[t-step] -> frames[t]),
    getting chaining's smoother trajectory back. Each hop is validated
    LOCALLY, against its own immediate chain neighbor (frames[t] ->
    frames[t-step], round-tripped and compared to `current`, that
    neighbor's own already-accepted position) -- NOT against the original
    neutral seed. An earlier version of this function checked every hop
    against true neutral instead, on the theory that local-only checks could
    miss drift a global one would catch; in practice this backfired: for
    hops far from the anchor, the frame_t->neutral flow needed for that
    check is itself exactly the large-displacement RAFT estimate this whole
    design exists to avoid trusting, so the check misfired constantly and
    replaced good chained positions with worse ones. A local check has a
    small, reliable displacement to flow across (like any single hop), so it
    rarely misfires -- exactly the property that let plain chaining stay
    smooth in the first place. A failing hop re-anchors at that one frame
    using Phase 1's own already-computed direct-from-neutral estimate for it
    (reused, not recomputed) rather than carrying a bad chained position
    into every subsequent frame.

    Visibility is always 1.0 -- unlike the single-hop-always predecessor,
    there is no invalidation case here: a failing hop always has Phase 1's
    direct-neutral estimate to fall back on, so every frame always gets
    SOME position.

    ``known_anchor_t``, when given (and a valid in-range frame index),
    SKIPS Phase 1's scan entirely: instead of flowing every candidate frame
    to find the best one, it trusts the caller's already-known-good anchor
    (see process_target_sequence's own once-per-sequence global anchor
    search) and computes just that ONE frame's direct-from-neutral position
    -- one RAFT call instead of 2*(T-1). Phase 2's re-anchor fallback still
    works exactly the same either way: direct_candidates is now populated
    lazily (only the frames a chain hop actually re-anchors at ever get
    computed), rather than eagerly for every candidate frame up front."""
    wait_for_gpu_memory(RAFT_REQUIRED_MB, label=f"RAFT (batch of {len(queries)} points, {frames.shape[0]} frames)")
    T = frames.shape[0]
    N = len(queries)
    tracks = np.empty((T, N, 2), dtype=np.float32)
    visibility = np.ones((T, N), dtype=np.float32)
    query_arr = np.asarray(queries, dtype=np.float64)  # (N, 2) -- the one and only neutral seed each point is ever flowed from
    tracks[0] = query_arr.astype(np.float32)
    if T <= 1:
        return tracks, visibility

    # Phase 1 (see _find_anchor_frame's own docstring -- shared with
    # run_model_batch's CoTracker anchor-selection path). Uses the
    # precomputed global anchor when available/valid, falling back to a
    # full per-call scan otherwise (e.g. no global anchor could be computed
    # for this target sequence at all).
    if known_anchor_t is not None and 1 <= known_anchor_t <= T - 1:
        anchor_t = known_anchor_t
        direct_candidates: dict[int, np.ndarray] = {anchor_t: _direct_candidate_at(model, frames, query_arr, anchor_t)}
    else:
        anchor_t, direct_candidates = _find_anchor_frame(model, frames, query_arr)
    tracks[anchor_t] = direct_candidates[anchor_t].astype(np.float32)

    # Phase 2: chain outward from anchor_t in both directions. `step` is +1
    # walking up to T-1, or -1 walking down to 1 (frame 0 is neutral itself,
    # already filled in above).
    def chain_direction(step: int) -> None:
        current = direct_candidates[anchor_t]
        t = anchor_t + step
        while 1 <= t <= T - 1:
            prev_frame = frames[t - step]
            flow_fwd = _raft_flow(model, prev_frame, frames[t])
            candidate = _sample_all(flow_fwd, current)

            # LOCAL validation only -- frame_t back to its immediate chain
            # neighbor (small, reliable displacement), compared against
            # `current` (that neighbor's own already-accepted position), NOT
            # against neutral. See this function's own docstring for why a
            # neutral-referenced check was tried and reverted.
            flow_bwd = _raft_flow(model, frames[t], prev_frame)
            roundtrip = _sample_all(flow_bwd, candidate)
            fb_error = np.linalg.norm(roundtrip - current, axis=1)  # (N,)
            needs_reanchor = fb_error > RAFT_FB_CONSISTENCY_THRESHOLD_PX
            if needs_reanchor.any():
                if t not in direct_candidates:
                    direct_candidates[t] = _direct_candidate_at(model, frames, query_arr, t)
                candidate = np.where(needs_reanchor[:, None], direct_candidates[t], candidate)

            tracks[t] = candidate.astype(np.float32)
            current = candidate
            t += step

    chain_direction(1)
    chain_direction(-1)
    return tracks, visibility


def track_and_fuse(
    *,
    label: str,
    propagate_indices: list[int],
    tracking_cameras: list[str],
    output_cameras: list[str],
    seeds_by_camera: dict[str, dict[int, tuple[float, float]]],
    last_fused_init: dict[int, np.ndarray],
    settings,
    actor_id: str,
    sequence_id: str,
    neutral_delegation_idx: int,
    total_delegation_frames: int,
    raw_width: int,
    raw_height: int,
    intrinsic: np.ndarray,
    world_2_cam_map: dict[str, np.ndarray],
    model: RaftModel,
    updates_sink: dict[int, dict[str, dict[int, tuple[float, float, float]]]],
    triangulate: bool = True,
    raft_model_for_anchor: RaftModel | None = None,
    frame_cache: dict[tuple, np.ndarray] | None = None,
    known_anchor_after: int | None = None,
    known_anchor_before: int | None = None,
) -> None:
    """Tracks `propagate_indices` on `tracking_cameras`' delegation videos only
    (forward + backward from the neutral frame, one CoTracker call each way),
    fuses per-frame via triangulation across just those cameras, then reprojects
    the fused position onto every camera in `output_cameras` and records it into
    `updates_sink[frame][camera_id][point_index] = (norm_x, norm_y, visibility)`.
    `last_fused_init` seeds the carry-forward fallback used whenever fusion fails
    at a given frame (fewer than 2 tracking cameras had a visible observation) --
    pass real neutral-frame 3D positions (skin points have no other ground truth).

    `frame_cache`, when given, memoizes extract_delegation_frames() calls
    (see _extract_delegation_frames_cached()) -- pass the same dict across
    every track_and_fuse(_from_other_sequence) call within one
    process_target_sequence() run so cameras shared by multiple point groups
    only get ffmpeg-decoded once. Falls back to an unshared, call-local dict
    (no caching benefit, but still correct) if omitted.

    `known_anchor_after`/`known_anchor_before`, when given, are forwarded to
    the "after"/"before" run_model_batch() calls respectively (per request:
    every camera and every landmark group shares ONE anchor frame per
    direction, computed once for the whole target sequence -- see
    process_target_sequence's own global anchor search -- instead of each
    camera/group independently re-running Phase 1's scan). Each camera here
    still computes its OWN direct-from-neutral position at that shared frame
    (that part is genuinely camera/point-specific), just without re-picking
    WHICH frame to use.
    """
    if not propagate_indices:
        return
    if frame_cache is None:
        frame_cache = {}

    # Delegation videos are always generated at this fixed size (see
    # /scratch/ondemand32/irwinngo/FaceView/backend/app/services/loop_service.py's DELEGATION_SIZE /
    # process_sequence_sapiens.py's scale=550:802) -- NOT raw_width/raw_height,
    # which is the raw (2200x3208) camera resolution. Frames must be extracted at
    # the delegation video's own size, and query/track pixel coordinates scaled
    # between raw-pixel space (what seeds/camera intrinsics use) and
    # delegation-pixel space (what the video frames and CoTracker use).
    deleg_w, deleg_h = DELEGATION_SIZE
    scale_x = deleg_w / raw_width
    scale_y = deleg_h / raw_height

    # (delegation_frame_idx -> {camera_id -> {point_idx -> (x_px, y_px, visibility)}})
    # x_px/y_px here are always in raw-pixel space, matching camera intrinsics.
    raw_tracks: dict[int, dict[str, dict[int, tuple[float, float, float]]]] = {}

    for cam in tracking_cameras:
        camera_id = f"cam_{cam}"
        video_path = settings.processed_root / actor_id / sequence_id / f"{camera_id}_delegation.mp4"
        seeds = seeds_by_camera.get(camera_id, {})
        ordered_idx = [idx for idx in propagate_indices if idx in seeds]
        if not ordered_idx:
            continue
        queries = [
            (float(np.clip(seeds[idx][0] * scale_x, 0, deleg_w - 1)), float(np.clip(seeds[idx][1] * scale_y, 0, deleg_h - 1)))
            for idx in ordered_idx
        ]

        # The neutral frame itself: its landmarks are the seeds (no tracking
        # needed). Recorded like any tracked frame so it goes through the same
        # fuse/reproject/write path and gets its own skin_frame_/<group>_frame_
        # JSONs -- also when the neutral is the last frame (no "after" clip).
        for i, idx in enumerate(ordered_idx):
            raw_tracks.setdefault(neutral_delegation_idx, {}).setdefault(camera_id, {})[idx] = (
                queries[i][0] / scale_x, queries[i][1] / scale_y, 1.0,
            )

        # "After": [neutral, total_delegation_frames), forward, seeded at t=0 == neutral frame.
        after_frames = _extract_delegation_frames_cached(frame_cache, video_path, neutral_delegation_idx, total_delegation_frames, deleg_w, deleg_h)
        if after_frames.shape[0] > 1:
            tracks, vis = run_model_batch(model, after_frames, queries, raft_model_for_anchor=raft_model_for_anchor, known_anchor_t=known_anchor_after)
            for t_local in range(1, tracks.shape[0]):  # skip t_local=0: the neutral frame itself
                t = neutral_delegation_idx + t_local
                for i, idx in enumerate(ordered_idx):
                    x_px, y_px = tracks[t_local, i]
                    raw_tracks.setdefault(t, {}).setdefault(camera_id, {})[idx] = (float(x_px) / scale_x, float(y_px) / scale_y, float(vis[t_local, i]))

        # "Before": [0, neutral], reversed so t_local=0 == neutral frame.
        before_frames = _extract_delegation_frames_cached(frame_cache, video_path, 0, neutral_delegation_idx + 1, deleg_w, deleg_h)
        if before_frames.shape[0] > 1:
            before_frames = np.ascontiguousarray(before_frames[::-1])
            tracks, vis = run_model_batch(model, before_frames, queries, raft_model_for_anchor=raft_model_for_anchor, known_anchor_t=known_anchor_before)
            for t_local in range(1, tracks.shape[0]):
                t = neutral_delegation_idx - t_local
                for i, idx in enumerate(ordered_idx):
                    x_px, y_px = tracks[t_local, i]
                    raw_tracks.setdefault(t, {}).setdefault(camera_id, {})[idx] = (float(x_px) / scale_x, float(y_px) / scale_y, float(vis[t_local, i]))

        print(f"  {label} cam={camera_id}: tracked {len(ordered_idx)} points across "
              f"[0,{neutral_delegation_idx}] and [{neutral_delegation_idx},{total_delegation_frames})")

    _fuse_and_reproject(
        raw_tracks, propagate_indices, last_fused_init, output_cameras,
        intrinsic, world_2_cam_map, raw_width, raw_height, updates_sink,
        triangulate=triangulate,
    )


def _fuse_and_reproject(
    raw_tracks: dict[int, dict[str, dict[int, tuple[float, float, float]]]],
    propagate_indices: list[int],
    last_fused_init: dict[int, np.ndarray],
    output_cameras: list[str],
    intrinsic: np.ndarray,
    world_2_cam_map: dict[str, np.ndarray],
    raw_width: int,
    raw_height: int,
    updates_sink: dict[int, dict[str, dict[int, tuple[float, float, float]]]],
    triangulate: bool = True,
) -> None:
    """Shared fusion/reprojection loop, used by both track_and_fuse() and
    track_and_fuse_from_other_sequence() below -- only how raw_tracks gets
    built differs between the two.

    triangulate=False (per request) skips fusion/reprojection entirely: each
    tracked camera's OWN raw 2D track is written straight into updates_sink
    for that SAME camera, normalized but otherwise untouched -- no
    cross-camera fusion into one 3D point, no reprojection onto
    output_cameras. Only cameras actually tracked end up with any output
    (not every output_camera), and cameras can (and are expected to)
    disagree with each other, since nothing reconciles them. This is a
    diagnostic mode -- see the earlier question about per-camera trajectory
    divergence -- not the normal write path."""
    if not triangulate:
        for t, per_camera in raw_tracks.items():
            for camera_id, points in per_camera.items():
                for idx in propagate_indices:
                    obs = points.get(idx)
                    if obs is None:
                        continue
                    x_px, y_px, vis = obs
                    updates_sink.setdefault(t, {}).setdefault(camera_id, {})[idx] = (x_px / raw_width, y_px / raw_height, vis)
        return

    last_fused: dict[int, np.ndarray] = dict(last_fused_init)
    for t in sorted(raw_tracks.keys()):
        for idx in propagate_indices:
            observations: list[tuple[str, np.ndarray, np.ndarray, float, float, np.ndarray]] = []
            for camera_id, points in raw_tracks[t].items():
                obs = points.get(idx)
                if obs is None or obs[2] <= 0:
                    continue
                cam = camera_id[4:]
                extrinsic = world_2_cam_map.get(cam)
                if extrinsic is None:
                    continue
                x_px, y_px, _vis = obs
                center, direction = camera_center_and_direction(np.array([x_px, y_px], dtype=np.float64), intrinsic, extrinsic)
                observations.append((cam, center, direction, x_px, y_px, extrinsic))

            fused = _robust_triangulate(observations, intrinsic, raw_width, raw_height, FRONT_CAMERAS_SET)
            if fused is None:
                fused = last_fused.get(idx)
                if fused is None:
                    continue
            else:
                last_fused[idx] = fused

            for cam in output_cameras:
                camera_id = f"cam_{cam}"
                extrinsic = world_2_cam_map.get(cam)
                if extrinsic is None:
                    continue
                proj = project_world_to_norm(fused, intrinsic, extrinsic, raw_width, raw_height)
                if proj is None:
                    continue
                vis = 1.0
                obs = raw_tracks[t].get(camera_id, {}).get(idx)
                if obs is not None:
                    vis = obs[2]
                updates_sink.setdefault(t, {}).setdefault(camera_id, {})[idx] = (proj[0], proj[1], vis)


def track_and_fuse_from_other_sequence(
    *,
    label: str,
    propagate_indices: list[int],
    tracking_cameras: list[str],
    output_cameras: list[str],
    seeds_by_camera: dict[str, dict[int, tuple[float, float]]],
    last_fused_init: dict[int, np.ndarray],
    settings,
    actor_id: str,
    neutral_sequence_id: str,
    neutral_delegation_idx: int,
    target_sequence_id: str,
    target_total_delegation_frames: int,
    raw_width: int,
    raw_height: int,
    intrinsic: np.ndarray,
    world_2_cam_map: dict[str, np.ndarray],
    model: RaftModel,
    updates_sink: dict[int, dict[str, dict[int, tuple[float, float, float]]]],
    triangulate: bool = True,
    raft_model_for_anchor: RaftModel | None = None,
    frame_cache: dict[tuple, np.ndarray] | None = None,
    known_anchor: int | None = None,
) -> None:
    """Cross-sequence variant of track_and_fuse(): seeds are still projected
    from the neutral mesh (unchanged), but the CoTracker clip is built by
    prepending the NEUTRAL sequence's own single delegation frame
    (neutral_sequence_id @ neutral_delegation_idx) onto the TARGET sequence's
    entire own delegation video, then tracked FORWARD ONLY through the target
    -- there's no "before" direction to propagate across two different
    sequences' own timelines, unlike track_and_fuse()'s within-sequence
    before+after split.

    `known_anchor`, when given, is forwarded to run_model_batch() the same
    way track_and_fuse()'s known_anchor_after/before are -- one shared anchor
    frame for this whole (single, forward-only) direction, computed once for
    the target sequence rather than per camera/group.

    Writes into target_sequence_id's own landmark files
    (via updates_sink, same downstream _write_skin_frame call site as
    track_and_fuse()) -- the neutral sequence's own directory is never
    touched by this function.

    `frame_cache` -- same memoization role as track_and_fuse()'s own
    parameter, see its docstring."""
    if not propagate_indices:
        return
    if frame_cache is None:
        frame_cache = {}

    deleg_w, deleg_h = DELEGATION_SIZE
    scale_x = deleg_w / raw_width
    scale_y = deleg_h / raw_height

    # (target-sequence delegation_frame_idx -> {camera_id -> {point_idx -> (x_px, y_px, visibility)}})
    raw_tracks: dict[int, dict[str, dict[int, tuple[float, float, float]]]] = {}

    for cam in tracking_cameras:
        camera_id = f"cam_{cam}"
        neutral_video_path = settings.processed_root / actor_id / neutral_sequence_id / f"{camera_id}_delegation.mp4"
        target_video_path = settings.processed_root / actor_id / target_sequence_id / f"{camera_id}_delegation.mp4"
        seeds = seeds_by_camera.get(camera_id, {})
        ordered_idx = [idx for idx in propagate_indices if idx in seeds]
        if not ordered_idx:
            continue

        # Unlike the same-sequence case, this needs a delegation video for
        # `cam` on BOTH the neutral sequence and the target sequence -- and
        # SKIN_REGIONS/EAR_REGIONS' own triangulation cameras (a broader set
        # than default_cameras.json's cultivated subset) are only guaranteed
        # for whichever sequence holds the actor's own neutral frame, so most
        # OTHER sequences may never have gotten delegation videos generated
        # for them. Skip (not crash) when either is missing for this camera --
        # triangulation below just falls back to fewer views, same as any
        # other missing observation.
        if not neutral_video_path.exists() or not target_video_path.exists():
            missing = neutral_video_path if not neutral_video_path.exists() else target_video_path
            print(f"  {label} cam={camera_id}: missing delegation video {missing} -- skipping this camera")
            continue

        queries = [
            (float(np.clip(seeds[idx][0] * scale_x, 0, deleg_w - 1)), float(np.clip(seeds[idx][1] * scale_y, 0, deleg_h - 1)))
            for idx in ordered_idx
        ]

        neutral_frame_img = _extract_delegation_frames_cached(frame_cache, neutral_video_path, neutral_delegation_idx, neutral_delegation_idx + 1, deleg_w, deleg_h)
        target_frames = _extract_delegation_frames_cached(frame_cache, target_video_path, 0, target_total_delegation_frames, deleg_w, deleg_h)
        if neutral_frame_img.shape[0] == 0 or target_frames.shape[0] == 0:
            print(f"  {label} cam={camera_id}: missing neutral or target frames, skipping this camera")
            continue
        clip = np.concatenate([neutral_frame_img, target_frames], axis=0)  # (1 + N, H, W, 3)

        tracks, vis = run_model_batch(model, clip, queries, raft_model_for_anchor=raft_model_for_anchor, known_anchor_t=known_anchor)
        for t_local in range(1, tracks.shape[0]):  # skip t_local=0: the prepended neutral frame
            t = t_local - 1  # target sequence's own delegation-frame index
            for i, idx in enumerate(ordered_idx):
                x_px, y_px = tracks[t_local, i]
                raw_tracks.setdefault(t, {}).setdefault(camera_id, {})[idx] = (float(x_px) / scale_x, float(y_px) / scale_y, float(vis[t_local, i]))

        print(f"  {label} cam={camera_id}: tracked {len(ordered_idx)} points across target "
              f"sequence {target_sequence_id} [0,{target_total_delegation_frames}), seeded from "
              f"{neutral_sequence_id}@{neutral_delegation_idx}")

    _fuse_and_reproject(
        raw_tracks, propagate_indices, last_fused_init, output_cameras,
        intrinsic, world_2_cam_map, raw_width, raw_height, updates_sink,
        triangulate=triangulate,
    )


def load_ear_seeds(
    landmarks_root: Path, actor_id: str, neutral_sequence_id: str, neutral_delegation_idx: int,
    cam: str, ear_indices: set[int], raw_width: int, raw_height: int,
) -> dict[int, tuple[float, float]]:
    """Reads this camera's EXISTING standard frame_<idx>.json at the neutral
    frame and extracts raw-pixel seed positions for the given (already
    standard-indexed) ear_indices -- unlike skin points, ear landmarks are
    part of the existing 0-73 sapiens2 set, so their neutral-frame position
    already exists on disk (no wrapped-mesh vertex lookup needed). Applies the
    same in-bounds filter as triangulate.py's is_in_bounds()."""
    frame_path = landmarks_root / actor_id / neutral_sequence_id / f"cam_{cam}" / f"frame_{neutral_delegation_idx:06d}.json"
    if not frame_path.exists():
        return {}
    with frame_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    seeds: dict[int, tuple[float, float]] = {}
    for point in data["points"]:
        idx = point["index"]
        if idx not in ear_indices:
            continue
        if not (0.0 < point["x"] < 1.0 and 0.0 < point["y"] < 1.0):
            continue
        seeds[idx] = (point["x"] * raw_width, point["y"] * raw_height)
    return seeds


def process_target_sequence(
    *,
    actor_id: str,
    neutral_sequence_id: str,
    neutral_frame_id: int,
    neutral_delegation_idx: int,
    target_sequence_id: str,
    settings,
    default_cameras: list[str],
    model: RaftModel | None,
    cotracker_model: CoTrackerPredictor | None,
    skin_world_xyz: dict[int, np.ndarray],
    indices_by_region: dict[str, list[int]],
    intrinsic: np.ndarray,
    world_2_cam_map: dict[str, np.ndarray],
    output_camera_ids: list[str],
    raw_width: int,
    raw_height: int,
    args: argparse.Namespace,
) -> bool:
    """Propagates into one target sequence. When target_sequence_id ==
    neutral_sequence_id, this is the original per-pair script's own
    before/after track_and_fuse() logic, unchanged. Otherwise it's the
    cross-sequence path: track_and_fuse_from_other_sequence(), forward only,
    seeded from the neutral sequence's own single frame."""
    tag = f"[actor={actor_id} neutral={neutral_sequence_id}@{neutral_frame_id} target={target_sequence_id}]"

    # --no-triangulate (per request): skip fusion/reprojection entirely, see
    # _fuse_and_reproject()'s own docstring for exactly what that changes.
    triangulate = not getattr(args, "no_triangulate", False)

    # Shared across every track_and_fuse(_from_other_sequence) call below --
    # memoizes extract_delegation_frames()'s ffmpeg decode (CPU-bound, one
    # subprocess spawn + full video decode per call) so a camera shared by
    # multiple skin groups / STANDARD_INDEX_REGION_GROUPS sub-regions only
    # gets decoded once for this whole target sequence, not once per group.
    # Scoped to this one process_target_sequence() call (see
    # _extract_delegation_frames_cached()'s own docstring) so it's released
    # once this sequence is done, not held for the rest of an actor-sweep.
    frame_cache: dict[tuple, np.ndarray] = {}

    sample_camera = default_cameras[0]
    delegation_video_sample = settings.processed_root / actor_id / target_sequence_id / f"cam_{sample_camera}_delegation.mp4"
    total_delegation_frames = probe_video_frame_count(delegation_video_sample)

    def project_seed(world_xyz: np.ndarray, cam: str) -> tuple[float, float] | None:
        extrinsic = world_2_cam_map.get(cam)
        if extrinsic is None:
            return None
        proj = project_world_to_norm(world_xyz, intrinsic, extrinsic, raw_width, raw_height)
        if proj is None:
            return None
        return proj[0] * raw_width, proj[1] * raw_height

    same_sequence = target_sequence_id == neutral_sequence_id
    if same_sequence and neutral_delegation_idx >= total_delegation_frames:
        print(f"SKIP {tag}: neutral delegation frame {neutral_delegation_idx} is past this "
              f"sequence's own {total_delegation_frames} frames")
        return False

    # --- Global anchor frame(s): computed ONCE for this whole target
    # sequence, using sample_camera's own neutral-frame standard (0-73)
    # points as a broadly-representative, always-available query set, then
    # reused by EVERY camera and EVERY landmark group below (see
    # known_anchor_after/known_anchor_before/known_anchor on the
    # track_and_fuse(_from_other_sequence) call sites) -- per request, since
    # which frame is "closest to neutral" is a property of TIME (every
    # camera is synchronized), not of which camera or which specific points
    # happen to be checked. Falls back to None (each camera/group
    # independently re-running its own Phase 1 scan, same as before this
    # change) if sample_camera's own data isn't usable for some reason. ---
    known_anchor_after: int | None = None
    known_anchor_before: int | None = None
    known_anchor_cross: int | None = None
    if model is not None:
        deleg_w, deleg_h = DELEGATION_SIZE
        anchor_scale_x = deleg_w / raw_width
        anchor_scale_y = deleg_h / raw_height
        anchor_seeds = load_ear_seeds(
            settings.landmarks_root, actor_id, neutral_sequence_id, neutral_delegation_idx,
            sample_camera, set(range(NUM_STANDARD_LANDMARKS)), raw_width, raw_height,
        )
        if anchor_seeds:
            anchor_query_arr = np.array(
                [
                    (np.clip(x * anchor_scale_x, 0, deleg_w - 1), np.clip(y * anchor_scale_y, 0, deleg_h - 1))
                    for x, y in anchor_seeds.values()
                ],
                dtype=np.float64,
            )
            if same_sequence:
                after_ref = _extract_delegation_frames_cached(
                    frame_cache, delegation_video_sample, neutral_delegation_idx, total_delegation_frames, deleg_w, deleg_h,
                )
                if after_ref.shape[0] > 1:
                    known_anchor_after, _ = _find_anchor_frame(model, after_ref, anchor_query_arr)
                before_ref = _extract_delegation_frames_cached(
                    frame_cache, delegation_video_sample, 0, neutral_delegation_idx + 1, deleg_w, deleg_h,
                )
                if before_ref.shape[0] > 1:
                    before_ref_rev = np.ascontiguousarray(before_ref[::-1])
                    known_anchor_before, _ = _find_anchor_frame(model, before_ref_rev, anchor_query_arr)
            else:
                neutral_video_path = settings.processed_root / actor_id / neutral_sequence_id / f"cam_{sample_camera}_delegation.mp4"
                if neutral_video_path.exists():
                    neutral_ref = _extract_delegation_frames_cached(
                        frame_cache, neutral_video_path, neutral_delegation_idx, neutral_delegation_idx + 1, deleg_w, deleg_h,
                    )
                    target_ref = _extract_delegation_frames_cached(
                        frame_cache, delegation_video_sample, 0, total_delegation_frames, deleg_w, deleg_h,
                    )
                    if neutral_ref.shape[0] > 0 and target_ref.shape[0] > 0:
                        clip_ref = np.concatenate([neutral_ref, target_ref], axis=0)
                        if clip_ref.shape[0] > 1:
                            known_anchor_cross, _ = _find_anchor_frame(model, clip_ref, anchor_query_arr)

    skin_updates: dict[int, dict[str, dict[int, tuple[float, float, float]]]] = {}
    region_group_updates: dict[str, dict[int, dict[str, dict[int, tuple[float, float, float]]]]] = {
        file_prefix: {} for _group_label, _regions_dict, file_prefix in STANDARD_INDEX_REGION_GROUPS
    }

    # --- Build every camera's seed set ONCE, merging skin (reprojected from
    # the wrapped mesh) AND every STANDARD_INDEX_REGION_GROUPS sub-region's
    # own neutral-frame sapiens2 detections into the SAME per-camera dict --
    # mirrors FaceScape's run_neutral_skin_propagation.py propagate_expression()
    # (neutral_seed_by_cam), grouping by CAMERA rather than by landmark
    # group/region. Before this change, a camera shared by several groups
    # (e.g. a front camera used by skin/nosebridge/eyebrow-left/chin/
    # undernose-center all at once) got a separate RAFT or CoTracker forward
    # pass over its OWN full delegation video once per group that happened to
    # use it -- same video decoded/re-run redundantly for each group. Grouping
    # by camera first means each camera gets AT MOST one RAFT call and one
    # CoTracker call per direction (before/after, or the single forward pass
    # for cross-sequence), covering every point from every group at once,
    # split only by RAFT_SKIN_VERTEX_INDICES/RAFT_STANDARD_GROUPS routing.
    # Per-point/per-group triangulation camera lists (region_cameras below)
    # are unaffected -- fusion still only pools observations from whichever
    # cameras that SPECIFIC point's own group declared, same as before; only
    # the tracking CALLS are batched together. ---
    raft_skin_global_indices = compute_raft_skin_global_indices()
    skin_cameras_by_index = compute_skin_cameras_by_index()  # {global_idx: [cam, ...]}

    seeds_by_camera_raft: dict[str, dict[int, tuple[float, float]]] = {}
    seeds_by_camera_cotracker: dict[str, dict[int, tuple[float, float]]] = {}
    last_fused_init_raft: dict[int, np.ndarray] = {}
    last_fused_init_cotracker: dict[int, np.ndarray] = {}
    idx_to_file_prefix: dict[int, str] = {}
    propagate_indices_raft: list[int] = []
    propagate_indices_cotracker: list[int] = []

    def _add_seed(idx: int, cam: str, seed: tuple[float, float], is_raft: bool) -> None:
        target = seeds_by_camera_raft if is_raft else seeds_by_camera_cotracker
        target.setdefault(f"cam_{cam}", {})[idx] = seed

    # Skin points -- every global index always has a real 3D position (the
    # wrapped mesh), so there's no "not enough cameras" fusion-failure case
    # to guard against here (unlike the standard-index groups below).
    for global_idx, cam_list in skin_cameras_by_index.items():
        is_raft = global_idx in raft_skin_global_indices
        (last_fused_init_raft if is_raft else last_fused_init_cotracker)[global_idx] = skin_world_xyz[global_idx]
        (propagate_indices_raft if is_raft else propagate_indices_cotracker).append(global_idx)
        idx_to_file_prefix[global_idx] = "skin"
        for cam in cam_list:
            seed = project_seed(skin_world_xyz[global_idx], cam)
            if seed is not None:
                _add_seed(global_idx, cam, seed, is_raft)

    # Standard-index region groups (ear/nosebridge/undernose/eyebrow/chin) --
    # same per-region camera list and same "skip if fewer than 2 cameras ever
    # triangulate" guard as before, just contributing into the shared
    # per-camera seed dicts above instead of making its own track_and_fuse
    # call per (group, sub-region).
    for group_label, regions_dict, file_prefix in STANDARD_INDEX_REGION_GROUPS:
        # Ear/nosebridge/undernose/chin use CoTracker ("usual" propagation,
        # unchanged). Eyebrow/undernose/chin use RAFT (per request) -- see
        # RAFT_STANDARD_GROUPS. RAFT is otherwise only used for the specific
        # skin cheek vertices above.
        is_raft = group_label in RAFT_STANDARD_GROUPS
        for region, config in regions_dict.items():
            region_indices: list[int] = config["indices"]
            region_cameras: list[str] = config["cameras"]
            region_seeds_by_cam: dict[str, dict[int, tuple[float, float]]] = {
                cam: load_ear_seeds(
                    settings.landmarks_root, actor_id, neutral_sequence_id, neutral_delegation_idx,
                    cam, set(region_indices), raw_width, raw_height,
                )
                for cam in region_cameras
            }

            # Triangulate the neutral frame's own multi-camera detections once,
            # for track_and_fuse's per-frame fusion-failure fallback -- same role
            # skin_world_xyz plays for skin points, just derived from real
            # sapiens2 detections instead of a wrapped-mesh vertex (these
            # points have no dedicated mesh vertex the way skin points do).
            region_last_fused: dict[int, np.ndarray] = {}
            for idx in region_indices:
                observations: list[tuple[np.ndarray, np.ndarray]] = []
                for cam in region_cameras:
                    seed = region_seeds_by_cam.get(cam, {}).get(idx)
                    extrinsic = world_2_cam_map.get(cam)
                    if seed is None or extrinsic is None:
                        continue
                    center, direction = camera_center_and_direction(np.array(seed, dtype=np.float64), intrinsic, extrinsic)
                    observations.append((center, direction))
                fused = triangulate_from_observations(observations)
                if fused is not None:
                    region_last_fused[idx] = fused

            if not region_last_fused:
                print(f"NOTE {tag}: no usable neutral-frame {group_label} detections for region '{region}' "
                      "(fewer than 2 in-bounds cameras) — skipping")
                continue

            (last_fused_init_raft if is_raft else last_fused_init_cotracker).update(region_last_fused)
            for idx in region_indices:
                idx_to_file_prefix[idx] = file_prefix
                (propagate_indices_raft if is_raft else propagate_indices_cotracker).append(idx)
                for cam, seeds in region_seeds_by_cam.items():
                    seed = seeds.get(idx)
                    if seed is not None:
                        _add_seed(idx, cam, seed, is_raft)

    merged_updates: dict[int, dict[str, dict[int, tuple[float, float, float]]]] = {}
    for subset_model, seeds_by_camera, last_fused_init, propagate_indices, subset_label in (
        (model, seeds_by_camera_raft, last_fused_init_raft, propagate_indices_raft, "RAFT"),
        (cotracker_model, seeds_by_camera_cotracker, last_fused_init_cotracker, propagate_indices_cotracker, "CoTracker"),
    ):
        if not propagate_indices:
            continue
        tracking_cameras = sorted({camera_id[4:] for camera_id in seeds_by_camera})  # strip "cam_" prefix
        call_label = f"{tag} merged skin+ear+nosebridge+undernose+eyebrow+chin [{subset_label}] ({len(propagate_indices)} pts)"
        if same_sequence:
            track_and_fuse(
                label=call_label, propagate_indices=propagate_indices,
                tracking_cameras=tracking_cameras, output_cameras=output_camera_ids,
                seeds_by_camera=seeds_by_camera, last_fused_init=last_fused_init,
                settings=settings, actor_id=actor_id, sequence_id=target_sequence_id,
                neutral_delegation_idx=neutral_delegation_idx, total_delegation_frames=total_delegation_frames,
                raw_width=raw_width, raw_height=raw_height, intrinsic=intrinsic, world_2_cam_map=world_2_cam_map,
                model=subset_model, updates_sink=merged_updates, triangulate=triangulate,
                raft_model_for_anchor=model, frame_cache=frame_cache,
                known_anchor_after=known_anchor_after, known_anchor_before=known_anchor_before,
            )
        else:
            track_and_fuse_from_other_sequence(
                label=call_label, propagate_indices=propagate_indices,
                tracking_cameras=tracking_cameras, output_cameras=output_camera_ids,
                seeds_by_camera=seeds_by_camera, last_fused_init=last_fused_init,
                settings=settings, actor_id=actor_id,
                neutral_sequence_id=neutral_sequence_id, neutral_delegation_idx=neutral_delegation_idx,
                target_sequence_id=target_sequence_id, target_total_delegation_frames=total_delegation_frames,
                raw_width=raw_width, raw_height=raw_height, intrinsic=intrinsic, world_2_cam_map=world_2_cam_map,
                model=subset_model, updates_sink=merged_updates, triangulate=triangulate,
                raft_model_for_anchor=model, frame_cache=frame_cache,
                known_anchor=known_anchor_cross,
            )

    # Split the merged tracking output back into skin_updates/region_group_updates
    # by each index's own destination file (idx_to_file_prefix) -- downstream
    # writing (_write_skin_frame/_write_region_frame) and the summary print
    # below are otherwise unchanged from before this grouping change.
    for t, per_camera in merged_updates.items():
        for camera_id, points in per_camera.items():
            for idx, value in points.items():
                prefix = idx_to_file_prefix.get(idx)
                if prefix is None:
                    continue
                sink = skin_updates if prefix == "skin" else region_group_updates[prefix]
                sink.setdefault(t, {}).setdefault(camera_id, {})[idx] = value

    skin_frames_written = len(skin_updates.keys())
    if not args.dry_run:
        for t, per_camera in skin_updates.items():
            for camera_id, updates in per_camera.items():
                _write_skin_frame(settings.landmarks_root, actor_id, target_sequence_id, camera_id, t, updates)
        for file_prefix, updates_by_frame in region_group_updates.items():
            for t, per_camera in updates_by_frame.items():
                for camera_id, updates in per_camera.items():
                    _write_region_frame(settings.landmarks_root, actor_id, target_sequence_id, camera_id, t, updates, file_prefix=file_prefix)
        # This sequence's own face mask + landmark filter (needs the landmarks
        # just written), rebuilt with them, before the pair can be wrapped. On
        # failure its frames fall back to the actor's mask.
        if not same_sequence:
            try:
                ok_mask = ensure_expression_mask(actor_id, neutral_sequence_id, neutral_frame_id, target_sequence_id,
                                                 settings, default_cameras, model, args, force=True)
            except Exception as exc:  # noqa: BLE001
                print(f"FAIL {tag}: expression mask: {exc}")
                ok_mask = False
            if not ok_mask:
                print(f"NOTE {tag}: no expression mask -- this sequence wraps with the actor's mask")
        confirm_labeled(settings.label_tracker_path, actor_id, target_sequence_id)

    total_skin = sum(len(v) for v in indices_by_region.values())
    summary_parts = [f"{total_skin} skin points across {skin_frames_written} frames"]
    for group_label, regions_dict, file_prefix in STANDARD_INDEX_REGION_GROUPS:
        total_points = sum(len(config["indices"]) for config in regions_dict.values())
        frames_written = len(region_group_updates[file_prefix])
        summary_parts.append(f"{total_points} {group_label} points across {frames_written} frames")
    print(f"DONE {tag}: propagated " + ", ".join(summary_parts) + f", of {target_sequence_id}")
    return True


def write_neutral_skin_frames(
    actor_id: str, neutral_sequence_id: str, neutral_delegation_idx: int, settings,
    skin_world_xyz: dict[int, np.ndarray], intrinsic: np.ndarray, world_2_cam_map: dict[str, np.ndarray],
    output_camera_ids: list[str], raw_width: int, raw_height: int, dry_run: bool = False,
) -> int:
    """Writes the neutral frame's own skin_frame_<neutral_idx>.json for every
    output camera, reprojected straight from skin_world_xyz (the neutral
    wrap's skin vertices -- no tracking needed at the neutral frame itself).
    Returns the number of cameras written. No label change."""
    neutral_updates: dict[str, dict[int, tuple[float, float, float]]] = {}
    for idx, world_xyz in skin_world_xyz.items():
        for cam in output_camera_ids:
            extrinsic = world_2_cam_map.get(cam)
            if extrinsic is None:
                continue
            proj = project_world_to_norm(world_xyz, intrinsic, extrinsic, raw_width, raw_height)
            if proj is None:
                continue
            neutral_updates.setdefault(f"cam_{cam}", {})[idx] = (proj[0], proj[1], 1.0)
    if not dry_run:
        for camera_id, updates in neutral_updates.items():
            _write_skin_frame(settings.landmarks_root, actor_id, neutral_sequence_id, camera_id, neutral_delegation_idx, updates)
    return len(neutral_updates)


@contextlib.contextmanager
def _actor_mask_lock(actor_id: str, neutral_sequence_id: str, shared: bool = False):
    """Blocking per-actor lock around the neutral wrap/mask files. Exclusive
    while ensure_actor_mask() rewrites them; shared for readers of the
    neutral wrapped_mesh.obj, so no reader sees it mid-rewrite. Blocks rather
    than skips: a pair must not go ahead before the actor mask exists."""
    lock_path = OUTPUT_ROOT / actor_id / neutral_sequence_id / ".actor_mask.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a+") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_SH if shared else fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def _load_neutral_camera_images(actor_id: str, sequence_id: str, frame_id: int, settings) -> dict:
    """{cam_id: BGR frame} at raw frame `frame_id` from every camera video."""
    import cv2  # local: only the masking step needs it

    images = {}
    images_dir = settings.data_root / actor_id / "sequences" / sequence_id / "images"
    for mp4 in sorted(images_dir.glob("cam_*.mp4")):
        cap = cv2.VideoCapture(str(mp4), cv2.CAP_FFMPEG)
        if frame_id > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ok, frame = cap.read()
        cap.release()
        if ok:
            images[mp4.stem.replace("cam_", "")] = frame
        else:
            print(f"NOTE: could not read frame {frame_id} from {mp4}")
    return images


def compute_actor_face_mask(
    actor_id: str, neutral_sequence_id: str, neutral_frame_id: int,
    wrapped_mesh_path: Path, mask_path: Path, settings,
) -> bool:
    """Runs face_ray_masking (SAM/U2Net + pytorch3d rasterizer, GPU) on the
    neutral frame's wrapped mesh with every camera's raw frame at the neutral
    frame, SAM prompted at the frame's nosebridge, eyebrow-end, lip-corner and
    ear landmarks (face_ray_masking_wrapping_test.SAM_PROMPT_INDICES, each
    where that camera has it), and writes the four mask
    files at mask_path (+ FACE_MASK_SUFFIXES). Must run with cwd = wrap_script/
    (face_ray_masking reads wrap_mask.txt / wrapping_node_mask.txt from it)."""
    tag = f"[actor={actor_id}]"
    from gpu_utils import wait_for_gpu_memory as reserve_gpu_memory
    from pytorch3d.io import load_objs_as_meshes
    import face_ray_masking_wrapping_test as frm

    intrinsic, world_2_cam_map = load_camera_params(actor_id, settings)
    cam_images = _load_neutral_camera_images(actor_id, neutral_sequence_id, neutral_frame_id, settings)
    # Eyebrow / ear prompts only where not occluded: positions on this wrap,
    # occluder = the neutral frame's scan (hair included), falling back to
    # the wrap itself when there is no scan.
    wrap_tm = trimesh.load(str(wrapped_mesh_path), process=False)
    wrap_v, wrap_f = np.asarray(wrap_tm.vertices, dtype=np.float64), np.asarray(wrap_tm.faces, dtype=np.int64)
    with open(Path(__file__).with_name("neutral_correspondence_without_eyes_POT_all.json")) as f:
        pot_rows = json.load(f)
    prompt = {i: wrap_v[scan_face_masking.pot_row_vertex(pot_rows[i], wrap_f)] for i in frm.SAM_PROMPT_INDICES}
    try:
        occluder = scan_face_masking.load_scan(actor_id, neutral_sequence_id, neutral_frame_id)
    except FileNotFoundError:
        print(f"NOTE {tag}: no Meshroom scan for the neutral frame -- SAM prompt occlusion-tested against the wrap")
        occluder = (wrap_v, wrap_f)
    visible = scan_face_masking.prompt_visibility(
        prompt, intrinsic, world_2_cam_map, {c: im.shape[:2] for c, im in cam_images.items()}, *occluder)
    hints = frm.load_sam_prompt_hints(
        settings.landmarks_root / actor_id / neutral_sequence_id, neutral_frame_id, cam_images.keys(), visible=visible,
    )
    for cam in sorted(visible):
        hidden = sorted(set(frm.SAM_PROMPT_INDICES) - set(visible[cam]))
        if hidden:
            print(f"{tag} SAM prompt cam {cam}: occluded, left out: {hidden}")
    print(f"{tag} SAM prompt: landmarks for {len(hints)}/{len(cam_images)} camera(s), "
          f"{sum(map(len, hints.values()))} points")
    full_extrinsics = {cam: mat.tolist() for cam, mat in world_2_cam_map.items()}
    with reserve_gpu_memory(FACE_RAY_MASKING_REQUIRED_MB, label="face ray masking (SAM)"):
        mesh = load_objs_as_meshes([str(wrapped_mesh_path)], device=torch.device(device))
        masks = frm.face_ray_masking(
            mesh, frm.NERSEMBLE_FRONT_CAMERAS, frm.NERSEMBLE_SIDE_RIGHT_CAMERAS, frm.NERSEMBLE_SIDE_LEFT_CAMERAS,
            cam_images, full_extrinsics, intrinsic.tolist(), sam_point_hints=hints,
        )
        frm.release_segmenters()
        del mesh
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if masks is None:
        print(f"FAIL {tag}: face_ray_masking found no usable camera")
        return False
    Path(mask_path).parent.mkdir(parents=True, exist_ok=True)
    frm.write_face_mask_files(str(mask_path), masks)
    return True


def ensure_actor_mask(
    actor_id: str, neutral_sequence_id: str, neutral_frame_id: int, settings,
    default_cameras: list[str], args: argparse.Namespace,
) -> bool:
    """Once per actor, before anything propagates from its neutral frame:
    make sure OUTPUT_ROOT/<actor>/<neutral_seq>/face_list.txt* (the actor's
    face mask, which run_pipeline_all.py requires for every wrap) exists.
    If it doesn't:
      1. bootstrap-wrap the neutral frame (static mask) if not wrapped yet,
      2. write the neutral frame's own skin landmarks from that wrap,
      3. wrap the neutral frame again with those skin landmarks (static mask)
         for a tighter fit,
      4. run face_ray_masking (SAM/U2Net, GPU) on that second wrap and write
         the actor's four mask files.
    If the mask exists but the neutral wrapped_mesh.obj doesn't (e.g. the
    neutral sequence was sent back to unreviewed, which renames every frame's
    wrap -- the neutral frame's too -- to old_wrapped_mesh.obj), the neutral
    frame is re-wrapped with that face mask instead of every sequence of the
    actor being skipped.
    Returns whether the neutral frame is wrapped (and, unless --dry-run, the
    mask exists) afterwards."""
    tag = f"[actor={actor_id}]"
    mask_path = OUTPUT_ROOT / actor_id / neutral_sequence_id / "face_list.txt"
    wrapped_mesh_path = OUTPUT_ROOT / actor_id / neutral_sequence_id / str(neutral_frame_id) / "wrapped_mesh.obj"

    def mask_exists() -> bool:
        return all((mask_path.parent / (mask_path.name + suffix)).exists() for suffix in FACE_MASK_SUFFIXES)

    if mask_exists():
        if wrapped_mesh_path.exists():
            return True
        if args.dry_run:
            print(f"[dry-run] {tag}: face mask exists but no neutral wrap at {wrapped_mesh_path} -- would re-wrap "
                  f"the neutral frame with the actor's face mask")
            return False
        with _actor_mask_lock(actor_id, neutral_sequence_id):
            if not wrapped_mesh_path.exists():  # not re-wrapped by another run while we waited for the lock
                print(f"{tag} neutral {neutral_sequence_id}@{neutral_frame_id} has a face mask but no wrapped_mesh.obj "
                      f"-- re-wrapping it with the actor's face mask")
                if not wrap_neutral_frame(actor_id, neutral_sequence_id, neutral_frame_id, args.resolution,
                                          Path(args.pipeline_script), face_mask_path=mask_path):
                    return False
        return True
    if args.dry_run:
        print(f"[dry-run] {tag}: no face mask at {mask_path}* -- would bootstrap-wrap, write neutral skin, "
              "re-wrap and run face_ray_masking")
        return wrapped_mesh_path.exists()

    with _actor_mask_lock(actor_id, neutral_sequence_id):
        if mask_exists():  # built by another run while we waited for the lock
            return True
        pipeline_script = Path(args.pipeline_script)

        # 1) Bootstrap wrap (static mask), unless the neutral is already wrapped.
        if not wrapped_mesh_path.exists():
            print(f"{tag} Mask step 1/4: bootstrap-wrapping neutral {neutral_sequence_id}@{neutral_frame_id}")
            if not wrap_neutral_frame(actor_id, neutral_sequence_id, neutral_frame_id, args.resolution, pipeline_script):
                return False

        # 2) The neutral frame's own skin landmarks, from that wrap.
        skin_world_xyz, _indices_by_region = load_skin_world_xyz(wrapped_mesh_path)
        intrinsic, world_2_cam_map = load_camera_params(actor_id, settings)
        output_camera_ids = sorted(cam for cam in default_cameras if cam in world_2_cam_map)
        raw_video_sample = settings.data_root / actor_id / "sequences" / neutral_sequence_id / "images" / f"cam_{default_cameras[0]}.mp4"
        raw_width, raw_height = probe_video_size(raw_video_sample)
        neutral_delegation_idx = neutral_frame_id // RAW_FRAME_STRIDE
        n_cams = write_neutral_skin_frames(
            actor_id, neutral_sequence_id, neutral_delegation_idx, settings, skin_world_xyz,
            intrinsic, world_2_cam_map, output_camera_ids, raw_width, raw_height,
        )
        print(f"{tag} Mask step 2/4: wrote neutral skin landmarks ({len(skin_world_xyz)} points) for {n_cams} camera(s)")

        # 3) Second wrap, now with the neutral frame's own skin landmarks.
        print(f"{tag} Mask step 3/4: re-wrapping neutral with its skin landmarks")
        if not wrap_neutral_frame(actor_id, neutral_sequence_id, neutral_frame_id, args.resolution, pipeline_script):
            return False

        # 4) face_ray_masking on the second wrap (the only GPU step of wrapping).
        print(f"{tag} Mask step 4/4: face_ray_masking on {wrapped_mesh_path}")
        if not compute_actor_face_mask(actor_id, neutral_sequence_id, neutral_frame_id, wrapped_mesh_path, mask_path, settings):
            return False
        print(f"DONE {tag}: actor face mask written to {mask_path}*")
    return True


_raft_for_expression_masks: RaftModel | None = None


def closest_scan_frame_to_neutral(
    actor_id: str, neutral_sequence_id: str, neutral_frame_id: int, sequence_id: str, settings,
    default_cameras: list[str], model: RaftModel | None,
) -> tuple[int | None, int]:
    """(raw frame, number of candidates): the frame of `sequence_id` closest to
    the actor's neutral frame, picked by _find_anchor_frame() -- the same RAFT
    forward/backward test, camera (default_cameras[0]) and query points (the
    neutral frame's standard landmarks) as process_target_sequence()'s global
    anchor -- but only among frames that have a Meshroom scan and landmarks
    (frames_with_scans_and_landmarks), since the expression mask needs a wrap
    of that frame. (None, n) when no frame qualifies."""
    global _raft_for_expression_masks
    cam = default_cameras[0]
    video = settings.processed_root / actor_id / sequence_id / f"cam_{cam}_delegation.mp4"
    neutral_video = settings.processed_root / actor_id / neutral_sequence_id / f"cam_{cam}_delegation.mp4"
    if not video.exists() or not neutral_video.exists():
        return None, 0
    total = probe_video_frame_count(video)
    candidates = [raw for raw in scan_face_masking.frames_with_scans_and_landmarks(
        settings.landmarks_root, actor_id, sequence_id, default_cameras) if raw // RAW_FRAME_STRIDE < total]
    if len(candidates) <= 1:
        return (candidates[0] if candidates else None), len(candidates)
    raw_width, raw_height = probe_video_size(
        settings.data_root / actor_id / "sequences" / neutral_sequence_id / "images" / f"cam_{cam}.mp4")
    deleg_w, deleg_h = DELEGATION_SIZE
    neutral_delegation_idx = neutral_frame_id // RAW_FRAME_STRIDE
    seeds = load_ear_seeds(settings.landmarks_root, actor_id, neutral_sequence_id, neutral_delegation_idx, cam,
                           set(range(NUM_STANDARD_LANDMARKS)), raw_width, raw_height)
    if not seeds:
        return None, len(candidates)
    query = np.array([(np.clip(x * deleg_w / raw_width, 0, deleg_w - 1), np.clip(y * deleg_h / raw_height, 0, deleg_h - 1))
                      for x, y in seeds.values()], dtype=np.float64)
    if model is None:  # --mask-only runs don't load RAFT up front
        if _raft_for_expression_masks is None:
            _raft_for_expression_masks = load_model(False)
        model = _raft_for_expression_masks
    neutral_img = extract_delegation_frames(neutral_video, neutral_delegation_idx, neutral_delegation_idx + 1, deleg_w, deleg_h)
    frames = extract_delegation_frames(video, 0, total, deleg_w, deleg_h)
    clip = np.concatenate([neutral_img, frames[[raw // RAW_FRAME_STRIDE for raw in candidates]]], axis=0)
    anchor_t, _ = _find_anchor_frame(model, clip, query)
    return candidates[anchor_t - 1], len(candidates)


def ensure_expression_mask(
    actor_id: str, neutral_sequence_id: str, neutral_frame_id: int, sequence_id: str, settings,
    default_cameras: list[str], model: RaftModel | None, args: argparse.Namespace, force: bool = False,
) -> bool:
    """The per-expression face mask + landmark filter of a NON-neutral sequence
    (scan_face_masking.expression_mask_dir()), which run_pipeline_all.py uses for
    that sequence's frames instead of the actor's -- the actor mask comes from
    the neutral frame, so an ear visible there but under hair in this sequence
    (another hairstyle) would otherwise be wrapped onto the hair. Built like the
    actor mask, at this sequence's frame closest to the neutral
    (closest_scan_frame_to_neutral):
      1. wrap that frame with the static mask, into expression_mask/work/ (its
         own OUTPUT_ROOT, the neutral frame's folder linked in as the source
         mesh) -- the sequence's real frame folders are not touched;
      2. landmark filter (filter_frame_landmarks, dry run) at that frame, skin
         points from that wrap -> landmark_filter.json;
      3. face mask (compute_actor_face_mask) on that wrap -> face_list.txt*,
         written last: run_pipeline_all.py switches to the expression mask once
         its files exist.
    No-op (True) for the neutral sequence and when both already exist unless
    `force`. Returns whether they exist afterwards (unless --dry-run). Callers
    hold the sequence's _sequence_pair_lock."""
    if sequence_id == neutral_sequence_id:
        return True
    tag = f"[actor={actor_id} sequence={sequence_id}]"
    out_dir = scan_face_masking.expression_mask_dir(actor_id, sequence_id)
    mask_path = out_dir / "face_list.txt"
    filter_path = out_dir / scan_face_masking.EXPRESSION_FILTER_NAME

    def done() -> bool:
        return filter_path.exists() and all((out_dir / ("face_list.txt" + s)).exists() for s in FACE_MASK_SUFFIXES)

    if done() and not force:
        return True
    raw_frame, n_candidates = closest_scan_frame_to_neutral(
        actor_id, neutral_sequence_id, neutral_frame_id, sequence_id, settings, default_cameras, model)
    if raw_frame is None:
        print(f"NOTE {tag}: no frame with a Meshroom scan and landmarks -- no expression mask, "
              "its frames use the actor's mask")
        return False
    print(f"{tag} expression mask: closest frame to the neutral among {n_candidates} scanned frame(s) is {raw_frame}")
    if args.dry_run:
        print(f"[dry-run] {tag}: would wrap frame {raw_frame} with the static mask and build its mask + landmark filter")
        return done()

    # 1) Static-mask wrap of that frame, in its own OUTPUT_ROOT.
    work_root = out_dir / "work"
    neutral_dir = OUTPUT_ROOT / actor_id / neutral_sequence_id / str(neutral_frame_id)
    neutral_link = work_root / actor_id / neutral_sequence_id / str(neutral_frame_id)
    if not neutral_link.exists():
        neutral_link.parent.mkdir(parents=True, exist_ok=True)
        neutral_link.symlink_to(neutral_dir, target_is_directory=True)
    wrapped = work_root / actor_id / sequence_id / str(raw_frame) / "wrapped_mesh.obj"
    wrapped.unlink(missing_ok=True)  # never reuse a wrap of another (or an older) frame choice
    cmd = [sys.executable, str(args.pipeline_script), actor_id, sequence_id, str(raw_frame), str(args.resolution),
           "--face-mask-path", str(STATIC_FACE_MASK_PATH)]
    print(f"{tag} expression mask 1/3: static-mask wrap of frame {raw_frame} into {work_root}")
    returncode = subprocess.run(cmd, env=dict(os.environ, OUTPUT_ROOT=str(work_root))).returncode
    if returncode != 0 or not wrapped.exists():
        print(f"FAIL {tag}: expression-mask wrap of frame {raw_frame} (exit_code={returncode})")
        return False
    (wrapped.parent / "mesh.obj").unlink(missing_ok=True)  # the unzipped scan, ~13 MB

    # 2) Landmark filter at that frame (dry run: writes nothing under landmarks/ or the frame's output folder).
    from gpu_utils import wait_for_gpu_memory as reserve_gpu_memory
    import face_ray_masking_wrapping_test as frm

    print(f"{tag} expression mask 2/3: landmark filter at frame {raw_frame}")
    with reserve_gpu_memory(FACE_RAY_MASKING_REQUIRED_MB, label="expression landmark filter (SAM)"):
        payload = scan_face_masking.filter_frame_landmarks(
            actor_id, sequence_id, raw_frame, default_cameras, len(assign_global_indices()), settings,
            dry_run=True, skin_xyz_override=load_skin_world_xyz(wrapped)[0])
        frm.release_segmenters()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    tmp = filter_path.with_name(f"{filter_path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload))
    os.replace(tmp, filter_path)

    # 3) Face mask on that wrap.
    print(f"{tag} expression mask 3/3: face_ray_masking on {wrapped}")
    if not compute_actor_face_mask(actor_id, sequence_id, raw_frame, wrapped, mask_path, settings):
        return False
    (out_dir / scan_face_masking.EXPRESSION_INFO_NAME).write_text(json.dumps({
        "raw_frame": raw_frame, "candidates": n_candidates, "neutral": [neutral_sequence_id, neutral_frame_id],
        "wrap": str(wrapped), "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, indent=1))
    dropped = sorted(int(i) for i in payload["dropped"])
    print(f"DONE {tag}: expression mask + landmark filter from frame {raw_frame} in {out_dir} "
          f"(filter drops {len(dropped)}: {dropped})")
    return True


def _write_neutral_only(
    tag: str, actor_id: str, neutral_sequence_id: str, neutral_delegation_idx: int, settings,
    skin_world_xyz: dict[int, np.ndarray], indices_by_region: dict[str, list[int]],
    intrinsic: np.ndarray, world_2_cam_map: dict[str, np.ndarray], output_camera_ids: list[str],
    raw_width: int, raw_height: int, args: argparse.Namespace,
) -> bool:
    """The neutral frame's own skin positions are exactly skin_world_xyz (no
    tracking/fusion needed -- that *is* the ground truth there), so just
    reproject onto every camera and write skin_frame_<neutral_idx>.json
    directly, into the NEUTRAL sequence's own directory. Backfills the gap
    left by track_and_fuse's t_local=0 skip (see its "skip t_local=0: the
    neutral frame itself" comments) without redoing the (already-done)
    CoTracker propagation for every other frame. Skin-only, same as the
    original per-pair --neutral-only (the standard-index region groups --
    ear/nosebridge/undernose/eyebrow, see STANDARD_INDEX_REGION_GROUPS -- have
    no equivalent gap: load_ear_seeds already reads the neutral frame's own
    sapiens2 detections directly whenever they're needed as a fusion
    fallback)."""
    n_cams = write_neutral_skin_frames(
        actor_id, neutral_sequence_id, neutral_delegation_idx, settings, skin_world_xyz,
        intrinsic, world_2_cam_map, output_camera_ids, raw_width, raw_height, dry_run=args.dry_run,
    )
    if not args.dry_run:
        confirm_labeled(settings.label_tracker_path, actor_id, neutral_sequence_id)

    total_skin = sum(len(v) for v in indices_by_region.values())
    print(f"DONE {tag}: wrote neutral-frame ({neutral_delegation_idx}) skin landmarks for "
          f"{total_skin} points across {n_cams} camera(s)")
    return True


@contextlib.contextmanager
def _sequence_pair_lock(landmarks_root: Path, actor_id: str, sequence_id: str):
    """Exclusive, non-blocking per-(actor,sequence) flock --
    landmarks_root/<actor>/<sequence>/.cotracker_pair.lock -- so multiple
    concurrent runs (different cluster nodes, or an actor-level sweep racing
    a targeted single-pair run) never propagate into the same sequence at
    once. Same lock file run_neutral_cotracker_propagation_batch.py's own
    process_pair() used to hold around the whole subprocess -- moved here
    (the actual per-sequence work, shared by both process_pair()'s single-pair
    path and process_actor()'s per-sequence sweep loop) so it applies
    uniformly regardless of dispatch granularity, and the batch wrapper no
    longer needs (or holds) a lock of its own.

    Yields True if the lock was acquired (caller should proceed), False if
    another run already holds it (caller should skip this sequence)."""
    lock_dir = landmarks_root / actor_id / sequence_id
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_file = open(lock_dir / ".cotracker_pair.lock", "w")
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


def filter_sequence_by_scan(
    actor_id: str, sequence_id: str, raw_frames: list[int], settings, default_cameras: list[str],
    dry_run: bool = False, skin_xyz_override: dict[int, np.ndarray] | None = None,
) -> tuple[int, int]:
    """Scan face mask + scan-filtered landmark JSON (scan_face_masking.
    filter_frame_landmarks) for each raw frame. A frame that fails is reported
    and skipped -- run_pipeline_all.py then wraps it without the filter.
    Returns (frames done, frames failed)."""
    from gpu_utils import wait_for_gpu_memory as reserve_gpu_memory
    import face_ray_masking_wrapping_test as frm

    tag = f"[actor={actor_id} sequence={sequence_id}]"
    num_skin = len(assign_global_indices())
    done = failed = 0
    with reserve_gpu_memory(FACE_RAY_MASKING_REQUIRED_MB, label="scan face masking (SAM)"):
        for raw_frame in raw_frames:
            try:
                scan_face_masking.filter_frame_landmarks(
                    actor_id, sequence_id, raw_frame, default_cameras, num_skin, settings, dry_run=dry_run,
                    skin_xyz_override=skin_xyz_override)
                done += 1
            except Exception as exc:  # one bad frame must not stop the rest
                print(f"FAIL {tag} frame={raw_frame}: scan mask / landmark filter: {exc}")
                failed += 1
        frm.release_segmenters()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print(f"{tag} scan filter: {done} frame(s) done, {failed} failed")
    return done, failed


def ensure_neutral_landmark_filter(actor_id: str, settings, default_cameras: list[str], args: argparse.Namespace,
                                   force: bool = False) -> bool:
    """The actor's neutral-frame scan mask + filtered landmark JSON (the
    landmark indices run_pipeline_all.py drops from every frame), computed
    once per actor -- skipped when it already exists unless `force`. Returns
    whether it exists afterwards (always True under --dry-run)."""
    neutral = load_actor_neutral_frame(settings.landmarks_root, actor_id)
    if neutral is None:
        return False
    neutral_sequence_id, neutral_frame_id = neutral
    tag = f"[actor={actor_id} neutral={neutral_sequence_id}@{neutral_frame_id}]"
    out = scan_face_masking.filtered_landmarks_path(settings.landmarks_root, actor_id, neutral_sequence_id, neutral_frame_id)
    if out.exists() and not force:
        return True
    if not scan_face_masking.scan_gz_path(actor_id, neutral_sequence_id, neutral_frame_id).exists():
        print(f"NOTE {tag}: no Meshroom scan for the neutral frame -- no landmark filter; its frames wrap unfiltered")
        return False
    with _actor_mask_lock(actor_id, neutral_sequence_id):
        if out.exists() and not force:  # built by another run while we waited
            return True
        print(f"{tag} neutral-frame landmark filter (scan mask + SAM)")
        # Skin points straight from the neutral wrap (their ground truth there),
        # not triangulated from possibly-stale skin_frame files.
        wrapped = OUTPUT_ROOT / actor_id / neutral_sequence_id / str(neutral_frame_id) / "wrapped_mesh.obj"
        skin_xyz = load_skin_world_xyz(wrapped)[0] if wrapped.exists() else None
        _done, failed = filter_sequence_by_scan(actor_id, neutral_sequence_id, [neutral_frame_id], settings,
                                                default_cameras, args.dry_run, skin_xyz_override=skin_xyz)
    return failed == 0


def run_mask_only(actor_id: str, sequence_id: str, settings, default_cameras: list[str], args: argparse.Namespace) -> bool:
    """--mask-only, after ensure_actor_mask() (FLAME mask built if missing):
    --force-mask recomputes the FLAME mask; then the neutral-frame landmark
    filter is recomputed (once per actor -- what every frame's wrap uses).
    --frame N / --all-frames additionally write per-frame filtered JSONs for
    `sequence_id` (inspection only; run_pipeline_all.py doesn't read them).
    No propagation, no label change."""
    tag = f"[actor={actor_id} sequence={sequence_id}]"
    if getattr(args, "force_mask", False):
        neutral_sequence_id, neutral_frame_id = load_actor_neutral_frame(settings.landmarks_root, actor_id)
        wrapped = OUTPUT_ROOT / actor_id / neutral_sequence_id / str(neutral_frame_id) / "wrapped_mesh.obj"
        print(f"{tag} --force-mask: recomputing the FLAME face mask on {wrapped}")
        if not args.dry_run:
            with _actor_mask_lock(actor_id, neutral_sequence_id):
                if not compute_actor_face_mask(actor_id, neutral_sequence_id, neutral_frame_id, wrapped,
                                               OUTPUT_ROOT / actor_id / neutral_sequence_id / "face_list.txt", settings):
                    return False

    ok = ensure_neutral_landmark_filter(actor_id, settings, default_cameras, args, force=True)

    # A non-neutral sequence's own face mask + landmark filter (built if missing;
    # --force-expression-mask rebuilds it) -- e.g. to backfill sequences
    # propagated before per-expression masks existed.
    neutral_sequence_id, neutral_frame_id = load_actor_neutral_frame(settings.landmarks_root, actor_id)
    if sequence_id != neutral_sequence_id:
        with _sequence_pair_lock(settings.landmarks_root, actor_id, sequence_id) as acquired:
            if not acquired:
                print(f"SKIP {tag}: sequence is locked by another run already processing it")
                return False
            ok = ensure_expression_mask(actor_id, neutral_sequence_id, neutral_frame_id, sequence_id, settings,
                                        default_cameras, None, args,
                                        force=getattr(args, "force_expression_mask", False)) and ok

    if args.frame is None and not getattr(args, "all_frames", False):
        return ok
    if args.frame is not None:
        if args.frame % RAW_FRAME_STRIDE or not scan_face_masking.scan_gz_path(actor_id, sequence_id, args.frame).exists():
            print(f"FAIL {tag}: --frame {args.frame} has no Meshroom scan (frames are multiples of {RAW_FRAME_STRIDE})")
            return False
        raw_frames = [args.frame]
    else:
        raw_frames = scan_face_masking.frames_with_scans_and_landmarks(settings.landmarks_root, actor_id, sequence_id, default_cameras)
    if not raw_frames:
        print(f"SKIP {tag}: no frame with both a Meshroom scan and landmarks")
        return ok
    with _sequence_pair_lock(settings.landmarks_root, actor_id, sequence_id) as acquired:
        if not acquired:
            print(f"SKIP {tag}: sequence is locked by another run already processing it")
            return False
        _done, failed = filter_sequence_by_scan(actor_id, sequence_id, raw_frames, settings, default_cameras, args.dry_run)
    return ok and failed == 0


def process_pair(
    actor_id: str,
    sequence_id: str,
    settings,
    default_cameras: list[str],
    model: RaftModel | None,
    args: argparse.Namespace,
    cotracker_model: CoTrackerPredictor | None = None,
) -> bool:
    """Back-compat entry point for propagate_unreviewed_missing_skin.py/
    propagate_batch.py, preserved with its original signature and return type
    from the one-neutral-per-pair design. Internally: loads actor_id's ONE
    neutral frame (see load_actor_neutral_frame), wraps it if needed, then
    propagates into just this one target sequence via process_target_sequence()
    -- same call process_actor() makes per sequence, just without the "every
    unlabeled sequence" iteration.

    Safe to call concurrently (e.g. from a ThreadPoolExecutor) for different
    actor/sequence pairs sharing one `model` instance -- same "one warmed-up
    model, no lock around the forward call" pattern as
    process_sequence_sapiens.py's shared sapiens2 pose model. Each pair only
    touches its own landmarks_root/<actor>/<sequence>/... files, and
    confirm_labeled()'s write to the shared label_tracker.json is itself
    flock-guarded, so no additional locking is needed here.

    `args` only needs `.resolution`, `.pipeline_script`, `.skip_neutral_wrap`,
    `.neutral_only`, and `.dry_run` -- any object with those attributes works,
    it doesn't have to be this module's own parsed argparse.Namespace."""
    tag = f"[actor={actor_id} sequence={sequence_id}]"
    neutral = load_actor_neutral_frame(settings.landmarks_root, actor_id)
    if neutral is None:
        print(f"SKIP {tag}: no landmarks_root/{actor_id}/neutral_frame.json")
        return False
    neutral_sequence_id, neutral_frame_id = neutral
    neutral_delegation_idx = neutral_frame_id // RAW_FRAME_STRIDE

    # Neutral wrap + actor face mask, once per actor (see ensure_actor_mask).
    if not ensure_actor_mask(actor_id, neutral_sequence_id, neutral_frame_id, settings, default_cameras, args):
        return False
    wrapped_mesh_path = OUTPUT_ROOT / actor_id / neutral_sequence_id / str(neutral_frame_id) / "wrapped_mesh.obj"
    if not wrapped_mesh_path.exists():
        print(f"SKIP {tag}: {wrapped_mesh_path} does not exist")
        return False
    if getattr(args, "mask_only", False):
        return run_mask_only(actor_id, sequence_id, settings, default_cameras, args)
    # Neutral-frame landmark filter, once per actor (no-op when it exists).
    if not args.dry_run:
        ensure_neutral_landmark_filter(actor_id, settings, default_cameras, args)

    with _actor_mask_lock(actor_id, neutral_sequence_id, shared=True):
        skin_world_xyz, indices_by_region = load_skin_world_xyz(wrapped_mesh_path)
    intrinsic, world_2_cam_map = load_camera_params(actor_id, settings)
    # Reproject/write skin landmarks onto default_cameras.json's cameras only --
    # not every camera in the actor's calibration. Cameras outside that set
    # (e.g. 222200038/041/045/048) never get a landmarks/<actor>/<sequence>/cam_<id>/
    # directory from the regular per-frame extraction step, so writing
    # skin_frame_*.json there would be for a camera nothing else ever reads
    # landmarks from anyway.
    output_camera_ids = sorted(cam for cam in default_cameras if cam in world_2_cam_map)

    sample_camera = default_cameras[0]
    raw_video_sample = settings.data_root / actor_id / "sequences" / neutral_sequence_id / "images" / f"cam_{sample_camera}.mp4"
    raw_width, raw_height = probe_video_size(raw_video_sample)

    if getattr(args, "neutral_only", False):
        return _write_neutral_only(
            tag, actor_id, neutral_sequence_id, neutral_delegation_idx, settings,
            skin_world_xyz, indices_by_region, intrinsic, world_2_cam_map,
            output_camera_ids, raw_width, raw_height, args,
        )

    with _sequence_pair_lock(settings.landmarks_root, actor_id, sequence_id) as acquired:
        if not acquired:
            print(f"SKIP {tag}: sequence is locked by another run already processing it")
            return False
        return process_target_sequence(
            actor_id=actor_id, neutral_sequence_id=neutral_sequence_id, neutral_frame_id=neutral_frame_id,
            neutral_delegation_idx=neutral_delegation_idx, target_sequence_id=sequence_id, settings=settings,
            default_cameras=default_cameras, model=model, cotracker_model=cotracker_model, skin_world_xyz=skin_world_xyz,
            indices_by_region=indices_by_region, intrinsic=intrinsic, world_2_cam_map=world_2_cam_map,
            output_camera_ids=output_camera_ids, raw_width=raw_width, raw_height=raw_height, args=args,
        )


def process_actor(
    actor_id: str,
    settings,
    default_cameras: list[str],
    model: RaftModel | None,
    args: argparse.Namespace,
    label_data: dict | None = None,
    cotracker_model: CoTrackerPredictor | None = None,
) -> tuple[int, int, int]:
    """Wraps the actor's one neutral frame (if needed), then propagates into
    every "unlabeled" sequence that actor has -- process_target_sequence()
    branches internally on same-sequence vs cross-sequence. Returns
    (passed, failed, skipped) sequence counts -- skipped counts sequences
    another concurrent run already held the per-sequence lock for (see
    _sequence_pair_lock), not failures."""
    tag = f"[actor={actor_id}]"
    neutral = load_actor_neutral_frame(settings.landmarks_root, actor_id)
    if neutral is None:
        print(f"SKIP {tag}: no landmarks_root/{actor_id}/neutral_frame.json")
        return 0, 0, 0
    neutral_sequence_id, neutral_frame_id = neutral
    neutral_delegation_idx = neutral_frame_id // RAW_FRAME_STRIDE
    print(f"{tag} Using neutral frame {neutral_sequence_id}@{neutral_frame_id} "
          f"(delegation idx {neutral_delegation_idx})")

    # Neutral wrap + actor face mask, once per actor (see ensure_actor_mask).
    if not ensure_actor_mask(actor_id, neutral_sequence_id, neutral_frame_id, settings, default_cameras, args):
        return 0, 1, 0
    wrapped_mesh_path = OUTPUT_ROOT / actor_id / neutral_sequence_id / str(neutral_frame_id) / "wrapped_mesh.obj"
    if not wrapped_mesh_path.exists():
        print(f"FAIL {tag}: {wrapped_mesh_path} does not exist")
        return 0, 0, 0

    with _actor_mask_lock(actor_id, neutral_sequence_id, shared=True):
        skin_world_xyz, indices_by_region = load_skin_world_xyz(wrapped_mesh_path)
    intrinsic, world_2_cam_map = load_camera_params(actor_id, settings)
    output_camera_ids = sorted(cam for cam in default_cameras if cam in world_2_cam_map)

    sample_camera = default_cameras[0]
    raw_video_sample = settings.data_root / actor_id / "sequences" / neutral_sequence_id / "images" / f"cam_{sample_camera}.mp4"
    raw_width, raw_height = probe_video_size(raw_video_sample)

    if getattr(args, "neutral_only", False):
        ok = _write_neutral_only(
            tag, actor_id, neutral_sequence_id, neutral_delegation_idx, settings,
            skin_world_xyz, indices_by_region, intrinsic, world_2_cam_map,
            output_camera_ids, raw_width, raw_height, args,
        )
        return (1, 0, 0) if ok else (0, 1, 0)

    if label_data is None:
        label_data = load_label_data(settings.label_tracker_path)

    all_sequences = bool(getattr(args, "all_sequences", False))
    actor_dir = settings.landmarks_root / actor_id
    target_sequence_ids = [
        seq_dir.name for seq_dir in sorted(actor_dir.iterdir())
        if seq_dir.is_dir() and seq_dir.name not in BLACKLISTED_SEQUENCES
        and (all_sequences or get_pair_status(settings.landmarks_root, label_data, actor_id, seq_dir.name) == "unlabeled")
    ]

    if args.ignore_list:
        before = len(target_sequence_ids)
        ignore_set = set(IGNORE_ACTOR_SEQUENCE_PAIRS)
        target_sequence_ids = [seq for seq in target_sequence_ids if (actor_id, seq) not in ignore_set]
        if before != len(target_sequence_ids):
            print(f"{tag} --ignore-list: {before - len(target_sequence_ids)} sequence(s) ignored, "
                  f"{len(target_sequence_ids)} remaining")

    status_desc = "every" if all_sequences else "'unlabeled'"
    print(f"{tag} Propagating into {len(target_sequence_ids)} {status_desc} sequence(s): {target_sequence_ids}")

    passed = 0
    failed = 0
    skipped = 0
    for target_sequence_id in target_sequence_ids:
        with _sequence_pair_lock(settings.landmarks_root, actor_id, target_sequence_id) as acquired:
            if not acquired:
                print(f"SKIP [actor={actor_id} target={target_sequence_id}]: sequence is locked by "
                      "another run already processing it")
                skipped += 1
                continue
            try:
                ok = process_target_sequence(
                    actor_id=actor_id, neutral_sequence_id=neutral_sequence_id,
                    neutral_frame_id=neutral_frame_id, neutral_delegation_idx=neutral_delegation_idx,
                    target_sequence_id=target_sequence_id, settings=settings, default_cameras=default_cameras,
                    model=model, cotracker_model=cotracker_model, skin_world_xyz=skin_world_xyz, indices_by_region=indices_by_region,
                    intrinsic=intrinsic, world_2_cam_map=world_2_cam_map, output_camera_ids=output_camera_ids,
                    raw_width=raw_width, raw_height=raw_height, args=args,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"FAIL [actor={actor_id} target={target_sequence_id}]: {exc}")
                ok = False
        if ok:
            passed += 1
        else:
            failed += 1

        # Release PyTorch's cached (but unused) CUDA allocations back to the
        # driver's free pool after every sequence. Without this, GPU memory
        # only grows across this whole actor-level run (many sequences per
        # process, unlike the old per-pair design where each invocation was
        # its own short-lived process that released everything on exit) --
        # wait_for_gpu_memory()'s torch.cuda.mem_get_info() check sees
        # globally free memory, not "free from this process's own cache", so
        # an unbounded cache growth eventually self-deadlocks: the process
        # can't reach its own required-free-memory threshold because the
        # memory it needs is sitting unused in its own cache. Confirmed for
        # real: a run got stuck here with 19.6 GiB cached by this process
        # alone, 22.9/24.6 GiB total used, deadlocked waiting for 3 GiB free.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return passed, failed, skipped


def _write_skin_frame(landmarks_root: Path, actor_id: str, sequence_id: str, camera_id: str, delegation_idx: int, updates: dict[int, tuple[float, float, float]]) -> None:
    path = landmarks_root / actor_id / sequence_id / camera_id / f"skin_frame_{delegation_idx:06d}.json"
    points = [
        {"index": idx, "x": x, "y": y, "z": 0.0, "visibility": vis, "region": "skin"}
        for idx, (x, y, vis) in sorted(updates.items())
    ]
    data = {
        "frame_idx": delegation_idx,
        "actor_id": actor_id,
        "sequence_id": sequence_id,
        "camera_id": camera_id,
        "points": points,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f)


def _write_region_frame(landmarks_root: Path, actor_id: str, sequence_id: str, camera_id: str, delegation_idx: int, updates: dict[int, tuple[float, float, float]], file_prefix: str) -> None:
    """Same shape as _write_skin_frame(), separate file (<file_prefix>_frame_<idx>.json)
    and "region" tag -- see module docstring for why this doesn't overwrite
    frame_<idx>.json's own sapiens2-detected values for these same indices.
    Used by every group in STANDARD_INDEX_REGION_GROUPS (ear/nosebridge/
    undernose/eyebrow), one dedicated file per group -- file_prefix="ear"
    reproduces the original _write_ear_frame()'s output exactly."""
    path = landmarks_root / actor_id / sequence_id / camera_id / f"{file_prefix}_frame_{delegation_idx:06d}.json"
    points = [
        {"index": idx, "x": x, "y": y, "z": 0.0, "visibility": vis, "region": file_prefix}
        for idx, (x, y, vis) in sorted(updates.items())
    ]
    data = {
        "frame_idx": delegation_idx,
        "actor_id": actor_id,
        "sequence_id": sequence_id,
        "camera_id": camera_id,
        "points": points,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("actor_id", nargs="?", default=None, help="Restrict to one actor (omit to sweep every actor with an 'unlabeled' sequence)")
    parser.add_argument("sequence_id", nargs="?", default=None, help="Restrict to one target sequence within actor_id (requires actor_id)")
    parser.add_argument("--resolution", default="3208", help="Resolution arg passed through to the neutral frame's run_pipeline.py call")
    parser.add_argument("--pipeline-script", default=str(Path(__file__).with_name("run_pipeline_all.py")), help="Path to the pipeline script used to wrap the neutral frame (default: run_pipeline_all.py, the \"Wrapping\"-node graph)")
    parser.add_argument("--skip-neutral-wrap", action="store_true", help="No-op, kept for existing callers: the neutral frame is wrapped (twice) only when the actor has no face mask yet -- see ensure_actor_mask()")
    parser.add_argument(
        "--neutral-only",
        action="store_true",
        help=(
            "Only write the neutral frame's own skin_frame_<idx>.json (reprojected directly "
            "from the wrapped mesh, no CoTracker) instead of the full forward/backward "
            "propagation across every other frame. For backfilling pairs that were already "
            "propagated before track_and_fuse's neutral-frame skip was noticed."
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="Run CoTracker and triangulation but don't write any landmark files or change label_tracker.json")
    parser.add_argument(
        "--no-triangulate", action="store_true",
        help=(
            "Diagnostic mode: skip fusion/reprojection entirely -- each tracked camera's own raw 2D "
            "track is written straight to that same camera's landmark file, un-fused with any other "
            "camera and never reprojected onto output_cameras. Only cameras actually tracked get any "
            "output. Useful for seeing per-camera tracking disagreement that triangulation normally "
            "smooths over."
        ),
    )
    parser.add_argument(
        "--ignore-list",
        action="store_true",
        help=(
            f"When sweeping (i.e. actor_id omitted, or given without sequence_id): skip "
            f"pairs listed in IGNORE_ACTOR_SEQUENCE_PAIRS ({len(IGNORE_ACTOR_SEQUENCE_PAIRS)} "
            f"entries) and every actor listed in IGNORE_ACTORS ({len(IGNORE_ACTORS)} entries) "
            "-- same lists as run_pipeline_batch_all.py. Has no effect when actor_id is given "
            "explicitly (a deliberate override shouldn't be silently blocked)."
        ),
    )
    parser.add_argument(
        "--all-sequences",
        action="store_true",
        help=(
            "Bypass the 'unlabeled' status filter entirely (per request): process EVERY "
            "non-blacklisted sequence for every actor that has a neutral_frame.json, "
            "regardless of its current label_tracker.json status (confirmed/unreviewed/"
            "unlabeled/etc all included). Still never runs for an actor with no neutral "
            "frame at all -- there's nothing to propagate from without one -- and "
            "confirm_labeled() still only moves a pair from 'unlabeled' to 'unreviewed' "
            "(a no-op, with a NOTE, for any pair that wasn't already 'unlabeled'), so this "
            "does not silently change any pair's review status. When sweeping (actor_id "
            "omitted), also switches actor discovery to discover_actors_with_neutral_frame() "
            "instead of discover_unlabeled_actors(). Has no effect when both actor_id and "
            "sequence_id are given explicitly (that path never filtered by status anyway)."
        ),
    )
    parser.add_argument(
        "--mask-only", action="store_true",
        help=(
            "No propagation, no label change, once per actor: make sure the actor's FLAME neutral-frame face "
            "mask exists (built if missing, see ensure_actor_mask; --force-mask recomputes it), then recompute "
            "the neutral frame's scan mask + filtered landmark JSON -- the landmark indices run_pipeline_all.py "
            "drops from every frame of the actor. Needs actor_id. With a non-neutral sequence_id, also builds "
            "that sequence's own expression mask + landmark filter if missing (see ensure_expression_mask; "
            "--force-expression-mask rebuilds it)."
        ),
    )
    parser.add_argument("--frame", type=int, default=None,
                        help="--mask-only: also write a filtered JSON for this raw frame of sequence_id (inspection)")
    parser.add_argument("--all-frames", action="store_true",
                        help="--mask-only: also write filtered JSONs for every frame of sequence_id that has a "
                             "Meshroom scan (inspection only -- wraps use the neutral frame's; ~40 s per frame)")
    parser.add_argument("--force-mask", action="store_true",
                        help="--mask-only: recompute the actor's FLAME neutral-frame face mask even if it exists")
    parser.add_argument("--force-expression-mask", action="store_true",
                        help="--mask-only with a non-neutral sequence_id: rebuild that sequence's expression mask + "
                             "landmark filter even if they exist")
    args = parser.parse_args()
    if (args.frame is not None or args.force_mask or args.force_expression_mask or args.all_frames) and not args.mask_only:
        parser.error("--frame / --all-frames / --force-mask / --force-expression-mask only apply with --mask-only")
    if args.mask_only and not args.actor_id:
        parser.error("--mask-only needs actor_id")
    if (args.frame is not None or args.all_frames) and not args.sequence_id:
        parser.error("--frame / --all-frames need sequence_id")

    settings = get_settings()
    default_cameras = load_default_cameras()
    regions = load_skin_regions()
    for region, indices in regions.items():
        cameras = SKIN_REGIONS[region]["cameras"]
        if indices:
            print(f"Loaded {len(indices)} skin vertex indices for region '{region}' from "
                  f"{SKIN_REGIONS[region]['path']} (triangulated via cameras {cameras})")
        else:
            print(f"NOTE: {SKIN_REGIONS[region]['path']} missing or empty — region '{region}' will be skipped")

    # --mask-only tracks nothing, so neither tracker is loaded.
    model = load_model(args.neutral_only or args.mask_only)
    cotracker_model = load_cotracker_model(args.neutral_only or args.mask_only)
    torch.autograd.set_grad_enabled(False)

    if args.mask_only and not args.sequence_id:
        neutral = load_actor_neutral_frame(settings.landmarks_root, args.actor_id)
        if neutral is None:
            print(f"SKIP [actor={args.actor_id}]: no landmarks_root/{args.actor_id}/neutral_frame.json")
            return 1
        args.sequence_id = neutral[0]  # once-per-actor work runs under the neutral sequence

    if args.actor_id and args.sequence_id:
        if args.sequence_id in BLACKLISTED_SEQUENCES:
            print(f"'{args.sequence_id}' is blacklisted (see sequence_blacklist.py) -- refusing to process.")
            return 1
        skipped_total = 0
        passed_total, failed_total = (1, 0) if process_pair(
            args.actor_id, args.sequence_id, settings, default_cameras, model, args, cotracker_model=cotracker_model,
        ) else (0, 1)
        actor_count = 1
    else:
        if args.actor_id:
            actor_ids = [args.actor_id]
        else:
            actor_ids = (
                discover_actors_with_neutral_frame(settings.landmarks_root) if args.all_sequences
                else discover_unlabeled_actors(settings.landmarks_root, settings.label_tracker_path)
            )
            if args.ignore_list:
                before = len(actor_ids)
                actor_ids = [a for a in actor_ids if a not in IGNORE_ACTORS]
                print(f"--ignore-list: {before - len(actor_ids)} actor(s) ignored, {len(actor_ids)} remaining")

        if not actor_ids:
            reason = "a neutral_frame.json" if args.all_sequences else "an 'unlabeled' sequence (and a neutral_frame.json)"
            print(f"No actors with {reason} found.")
            return 0
        print(f"Processing {len(actor_ids)} actor(s): {actor_ids}")

        label_data = load_label_data(settings.label_tracker_path)
        passed_total = 0
        failed_total = 0
        skipped_total = 0
        for actor_id in actor_ids:
            passed, failed, skipped = process_actor(actor_id, settings, default_cameras, model, args, label_data=label_data, cotracker_model=cotracker_model)
            passed_total += passed
            failed_total += failed
            skipped_total += skipped
        actor_count = len(actor_ids)

    print(f"Done. actors={actor_count} sequences_passed={passed_total} sequences_failed={failed_total} sequences_skipped={skipped_total}")
    return 0 if failed_total == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
