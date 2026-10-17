#!/usr/bin/env python3
"""Same as run_pipeline_batch.py, but runs run_pipeline_all.py
(which also folds in skin landmark points from
run_neutral_cotracker_propagation_wrapping_test.py) instead of run_pipeline.py.

The neutral frame is NOT skipped here (unlike the production
run_pipeline_batch_all.py): run_neutral_cotracker_propagation_wrapping_test.py
wraps it first (and now also writes its own skin_frame_<idx>.json, see that
script), but that initial wrap has no skin points baked in yet -- this script
re-runs process_frame() on the neutral frame too, same as every other frame,
which upgrades its wrapped_mesh.obj to include skin points now that they're
available. A pair is still skipped if its neutral frame hasn't been wrapped at
all yet (that first wrap remains run_neutral_cotracker_propagation_wrapping_test.py's
job, not this script's).

By default (no --target-only/--run-completed given), this POLLS FOREVER: each
round discovers every actor/sequence pair currently "unreviewed" in
label_tracker.json, dispatches that round's pairs, then sleeps
POLL_INTERVAL_SECONDS and rescans -- so newly-"unreviewed" pairs get picked up
without re-invoking this script. Re-scanning a pair still mid-flight from a
previous round is safe: process_pair()'s own per-pair .pipeline_pair.lock
makes it a no-op skip, not duplicated work -- see process_pair()'s docstring.
--target-only and --run-completed both remain ONE-SHOT instead, for
curated/manual/testing runs.

--confirmed-only (one-shot) RE-WRAPS every "confirmed" pair instead, e.g. after
a pipeline fix: every frame is re-run through process_frame() -- which
re-triangulates the landmarks and re-applies the face-mask filter from
scratch, nothing from the old wrap is reused -- then the pair's overlay and
point-overlay videos are re-rendered and its final_wrapped_data .pt is
re-exported (the old .pt is first copied to FINAL_WRAPPED_BACKUP_ROOT). Each
selected actor's neutral frame is re-wrapped first (pass 1), since every other
frame wraps against it. Finished pairs are recorded in REWRAP_DONE_JSON so a
rerun resumes (--no-resume redoes them). Labels are left "confirmed".

--unconfirmed-only (one-shot) does the same re-wrap for every "unconfirmed"
pair (neutral frames first, then every frame, then both videos re-rendered),
but exports no .pt -- that only happens when the pair is Confirmed in
FaceView, so there's nothing to back up or re-export yet. Finished pairs are
recorded in REWRAP_UNCONFIRMED_DONE_JSON. Labels are left "unconfirmed".

CPU-only: run_pipeline_all.py hides the GPU on import and needs none (Wrap
is CPU-only, scans come precomputed from MESHROOM_DATA_ROOT, and each actor's
face mask is made by run_neutral_cotracker_propagation.py). A pair whose actor
has no face mask yet is skipped until propagation has produced one.
"""
import nersemble_paths
import argparse
import contextlib
import fcntl
import json
import multiprocessing
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from backend.app.core.config import get_settings
from run_pipeline_all import (
    MESHROOM_DATA_ROOT,
    actor_face_mask_path,
    face_mask_files_exist,
    process_frame,
    load_default_cameras,
    load_skin_vertex_indices,
)
from ignore_lists import IGNORE_ACTOR_SEQUENCE_PAIRS, IGNORE_ACTORS
from sequence_blacklist import BLACKLISTED_SEQUENCES


FRAME_STEP = 5  # landmark/delegation frame step; the WRAP step is wrap_frame_step(actor)
from wrap_frame_step import wrap_frame_step  # noqa: E402
POLL_INTERVAL_SECONDS = 60
OUTPUT_ROOT = nersemble_paths.OUTPUT_ROOT
DATA_ROOT = nersemble_paths.DATA_ROOT
DONE_JSON = Path(__file__).with_name("completed_sequences_all_wrapping_test.json")
# --confirmed-only: pairs already re-wrapped + re-exported (resume record), and
# where each pair's pre-rewrap .pt is kept (first copy only, never overwritten).
REWRAP_DONE_JSON = Path(__file__).with_name("completed_sequences_rewrap_confirmed.json")
# --unconfirmed-only's own resume record (no .pt export/backup for those).
REWRAP_UNCONFIRMED_DONE_JSON = Path(__file__).with_name("completed_sequences_rewrap_unconfirmed.json")
FINAL_WRAPPED_BACKUP_ROOT = nersemble_paths.WORK_ROOT / "final_wrapped_data_backup_pre_rewrap"
# FaceView's own settings.final_wrapped_data_root (wrap_script/backend/'s small
# config copy doesn't have it) and repo root, for the .pt re-export below.
FINAL_WRAPPED_DATA_ROOT = nersemble_paths.FINAL_WRAPPED_DATA_ROOT
FACEVIEW_ROOT = nersemble_paths.FACEVIEW_ROOT


# Camera the "done" overlay video is rendered from once a pair completes -- see
# mark_unconfirmed/render_overlay_video below. Matches
# FaceView/backend/app/api/routes/labels.py's OVERLAY_VIDEO_CAMERA_ID, which
# serves this same file to the frontend's Overlay Video tab.
OVERLAY_VIDEO_CAMERA_ID = "222200046"
# Camera for the Overlay Landmarks point-overlay video (render_point_overlay.py);
# must match FaceView's labels.py POINT_OVERLAY_VIDEO_CAMERA_ID.
POINT_OVERLAY_VIDEO_CAMERA_ID = "222200037"
TEST_RENDER_GT_SCRIPT = Path(__file__).with_name("test_render_gt.py")
RENDER_POINT_OVERLAY_SCRIPT = Path(__file__).with_name("render_point_overlay.py")

# Allowlist consulted when --run-completed is given: actor_id -> [sequence_id, ...].
# Distinct from DONE_JSON above, which this script writes to as it finishes sequences.
RUN_COMPLETED_JSON = Path.home() / "FaceView" / "completed_sequences.json"

# Only these expressions are processed, for every actor found under DATA_ROOT.
TARGET_SEQUENCES = [
    "EMO-1-shout+laugh",
    "EXP-2-eyes",
    "EXP-3-cheeks+nose",
    "EXP-4-lips",
    "EXP-8-jaw-1",
    "EMO-2-surprise+fear",
    "EMO-3-angry+sad",
    "EMO-4-disgust+happy",
    "EXP-1-head",
    "EXP-5-mouth",
    "EXP-9-jaw-2",
    "FREE",
    "SEN-01-cramp_small_danger",
    "SEN-02-same_phrase_thirty_times",
    "SEN-03-pluck_bright_rose",
    "SEN-04-two_plus_seven",
    "SEN-05-glow_eyes_sweet_girl",
    "SEN-06-problems_wise_chief",
    "SEN-07-fond_note_friend",
    "SEN-08-clothes_and_lodging",
    "SEN-09-frown_events_bad",
    "SEN-10-port_strong_smokey"
]
# Belt-and-suspenders: TARGET_SEQUENCES above never included either blacklisted
# sequence, but filtering it too (see sequence_blacklist.py) means a future
# edit to that list can't silently reintroduce them.
TARGET_SEQUENCES = [s for s in TARGET_SEQUENCES if s not in BLACKLISTED_SEQUENCES]

# Per-actor sequences to exclude even when present in TARGET_SEQUENCES.
EXCLUDED_ACTOR_SEQUENCES = {
# "017","018","024"
        }

# --ignore-list's two lists -- (actor_id, sequence_id) pairs, and whole
# actors (every sequence skipped) -- live in ignore_actor_sequence_pairs.json /
# ignore_actors.json, shared with run_neutral_cotracker_propagation.py via
# ignore_lists.py. Unlike EXCLUDED_ACTOR_SEQUENCES (always applied), both are
# opt-in.


def is_ignored(actor_id: str, sequence_id: str) -> bool:
    """--ignore-list's test: the actor is in IGNORE_ACTORS, or the exact pair
    is in IGNORE_ACTOR_SEQUENCE_PAIRS."""
    return actor_id in IGNORE_ACTORS or (actor_id, sequence_id) in IGNORE_ACTOR_SEQUENCE_PAIRS


TARGET_ACTORS = [
#	"017"
#  "017", "018", "024", "030", "031", "032", "033", "035"
"042"
]


def discover_actor_sequences(data_root: Path, target_only: bool = False):
    """Return (actor_id, sequence_id) for the target expressions of every actor.

    Iterates every actor folder under ``data_root`` (actors without a folder are
    simply skipped), unless ``target_only`` is set, in which case only actors in
    ``TARGET_ACTORS`` are considered. Keeps only the sequences in
    ``TARGET_SEQUENCES`` that exist, and drops any per-actor exclusions.
    """
    pairs = []
    if not data_root.is_dir():
        return pairs
    for actor_dir in sorted(data_root.iterdir()):
        if not actor_dir.is_dir():
            continue
        actor_id = actor_dir.name
        if target_only and actor_id not in TARGET_ACTORS:
            continue
        sequences_root = actor_dir / "sequences"
        if not sequences_root.is_dir():
            continue
        excluded = EXCLUDED_ACTOR_SEQUENCES.get(actor_id, set())
        for sequence_id in TARGET_SEQUENCES:
            if sequence_id in excluded:
                continue
            if (sequences_root / sequence_id).is_dir():
                pairs.append((actor_id, sequence_id))
    return pairs


def load_run_completed_pairs(path: Path) -> list[tuple[str, str]]:
    """Return (actor_id, sequence_id) pairs from a completed_sequences.json-style
    allowlist, in the file's own actor/sequence order."""
    with open(path) as f:
        completed = json.load(f)
    return [
        (actor_id, sequence_id)
        for actor_id in sorted(completed)
        for sequence_id in completed[actor_id]
    ]


def get_pair_status(landmarks_root: Path, label_data: dict, actor_id: str, sequence_id: str) -> str:
    """Mirrors run_neutral_cotracker_propagation.py's function of the same name
    (itself mirroring FaceView/backend/app/services/label_tracker.py's
    LabelTracker.get_status): an explicit entry in label_tracker.json wins,
    otherwise fall back to whether neutral_frame.json exists on disk.
    Reimplemented locally rather than imported -- run_neutral_cotracker_propagation.py
    pulls in CoTracker/torch at module import time just for this one small
    lookup, which this script has no other reason to depend on."""
    status = label_data.get("Nersemble", {}).get(actor_id, {}).get(sequence_id)
    if status is not None:
        return status
    neutral_frame_path = landmarks_root / actor_id / sequence_id / "neutral_frame.json"
    return "unlabeled" if neutral_frame_path.exists() else "no_neutral_label"


def load_completed(path: Path) -> dict:
    if path.exists():
        try:
            with open(path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def record_completed(path: Path, actor_id: str, sequence_id: str) -> dict:
    """Add actor/sequence to the completed map (deduped, sorted) and persist it.
    Lock-guarded, atomic-replace read-modify-write against ``path`` itself --
    same idiom as mark_unconfirmed()/label_tracker.py's _write below, reading
    the current on-disk state fresh under the lock rather than trusting an
    in-memory copy. dispatch() now runs each pair in its own OS process (see
    its own note on why), so there is no single shared ``completed`` dict left
    to protect with an in-memory threading.Lock the way there used to be --
    every writer must go through disk, guarded by this file lock, instead."""
    lock_path = path.with_name(path.name + ".lock")
    with lock_path.open("w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            completed = load_completed(path)
            sequences = set(completed.get(actor_id, []))
            sequences.add(sequence_id)
            completed[actor_id] = sorted(sequences)
            tmp = path.with_name(path.name + ".tmp")
            tmp.write_text(json.dumps(completed, indent=2), encoding="utf-8")
            tmp.replace(path)
            return completed
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def mark_unconfirmed(label_tracker_path: Path, actor_id: str, sequence_id: str) -> bool:
    """Moves this pair's label_tracker.json status from "unreviewed" to
    "unconfirmed" -- the same lock-guarded, atomic-replace read-modify-write
    as FaceView/backend/app/services/label_tracker.py's
    LabelTracker.mark_unconfirmed/_write, reimplemented here since that
    module lives in a separate backend package (same reasoning as
    run_neutral_cotracker_propagation.py's confirm_labeled). A no-op (with a
    note) if the pair isn't currently "unreviewed" -- e.g. a rerun on a pair
    someone already unconfirmed or confirmed, which this must not silently
    reset.

    Locks a dedicated sibling .lock file rather than label_tracker.json
    itself, and writes via temp-file-then-replace rather than in-place
    truncate+dump -- see _write's docstring in label_tracker.py for why
    (the old truncate-then-dump-in-place pattern left the file empty/corrupt
    if the process died in between, e.g. an OOM kill, which is what actually
    caused label_tracker.json to appear "wiped" once).

    Returns whether THIS call actually performed the transition (True) or was
    a no-op because another call already had (False) -- since nothing stops
    two cluster nodes from both picking up the same "unreviewed" pair and
    racing through the exact same frame loop, this flock-guarded read is the
    only reliable "did I win" signal available. process_pair() uses it to
    gate render_overlay_video(), which -- unlike this function -- has no lock
    of its own and would otherwise have two processes writing the same output
    .mp4 concurrently via imageio if both reached the end of the pair."""
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
            current = data.get("Nersemble", {}).get(actor_id, {}).get(sequence_id, "unreviewed")
            if current != "unreviewed":
                print(f"NOTE: not moving {actor_id}/{sequence_id} to 'unconfirmed' — "
                      f"current status is {current!r}, not 'unreviewed'")
                return False
            data.setdefault("Nersemble", {}).setdefault(actor_id, {})[sequence_id] = "unconfirmed"
            tmp_path = label_tracker_path.with_name(label_tracker_path.name + ".tmp")
            tmp_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            tmp_path.replace(label_tracker_path)
            print(f"Moved {actor_id}/{sequence_id} to 'unconfirmed' in {label_tracker_path}")
            return True
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def render_overlay_video(actor_id: str, sequence_id: str) -> None:
    """Renders the "done" overlay video via test_render_gt.py for the frontend's
    Overlay Video tab (see FaceView/backend/app/api/routes/labels.py's
    /labels/overlay-video endpoint). Failure here is logged but never fails the
    pair overall -- the frames are already wrapped and status already moved to
    "unconfirmed" by this point, so a rendering hiccup shouldn't undo that."""
    cmd = [sys.executable, str(TEST_RENDER_GT_SCRIPT), actor_id, sequence_id, OVERLAY_VIDEO_CAMERA_ID]
    print("RENDER  " + " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"WARNING: overlay video render failed for {actor_id}/{sequence_id} "
              f"(exit_code={result.returncode})")


def render_point_overlay_video(actor_id: str, sequence_id: str) -> None:
    """Renders the landmark point-overlay video (annotated vs. wrapped-mesh
    landmarks) via render_point_overlay.py for the frontend's Overlay
    Landmarks tab (/labels/point-overlay-video), on POINT_OVERLAY_VIDEO_CAMERA_ID.
    Same failure policy as render_overlay_video(): logged, never fails the pair."""
    cmd = [sys.executable, str(RENDER_POINT_OVERLAY_SCRIPT), actor_id, sequence_id, POINT_OVERLAY_VIDEO_CAMERA_ID]
    print("RENDER  " + " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"WARNING: point overlay video render failed for {actor_id}/{sequence_id} "
              f"(exit_code={result.returncode})")


def sequence_has_frame(sequence_dir: Path, frame_id: int) -> bool:
    """Support both sequence/frame and sequence/cam_*/frame_XXXXXX.json layouts."""
    direct_candidates = [
        sequence_dir / str(frame_id),
        sequence_dir / f"{frame_id:03d}",
        sequence_dir / f"{frame_id:06d}",
        sequence_dir / f"frame_{frame_id:03d}",
        sequence_dir / f"frame_{frame_id:06d}",
        sequence_dir / f"frame_{frame_id:06d}.json",
    ]
    if any(path.exists() for path in direct_candidates):
        return True

    for cam_dir in sequence_dir.glob("cam_*"):
        if cam_dir.is_dir() and (cam_dir / f"frame_{frame_id:06d}.json").exists():
            return True

    return False


def skin_landmarks_complete(
    sequence_dir: Path,
    default_cameras: list[str],
    max_frame: int | None,
    neutral_frame_id: int | None,
    step: int = FRAME_STEP,
) -> tuple[bool, int | None]:
    """Whether every frame this pair would process already has a
    skin_frame_<idx>.json for every camera in default_cameras -- the same
    per-camera lookup run_pipeline_all.process_frame() does per
    frame (see its skin_path check). Used as a pair-level pre-flight gate:
    process_frame() itself only degrades a single frame to standard-only
    landmarks when skin data is missing for it, which would otherwise let this
    script silently wrap a partially-propagated sequence with some frames
    missing skin points. Walks frame ids exactly like process_pair()'s own
    loop (same FRAME_STEP/max_frame/sequence_has_frame stopping condition),
    so it checks precisely the frames that would actually run -- INCLUDING the
    neutral frame, now that process_pair() no longer skips it: its own
    skin_frame_<idx>.json is written automatically by
    run_neutral_cotracker_propagation_wrapping_test.py (every run, not just
    --neutral-only), so it's expected to already be present by the time this
    runs, same as every other frame.

    ``neutral_frame_id`` is accepted for signature/call-site compatibility but
    no longer treated specially.

    Returns (True, None) if complete, or (False, first_missing_frame_id) for the
    first (in ascending order) frame id missing skin data for at least one
    default camera.
    """
    del neutral_frame_id  # no longer special-cased, see docstring
    frame_id = 0
    while True:
        if max_frame is not None and frame_id > max_frame:
            break
        if not sequence_has_frame(sequence_dir, int(frame_id / FRAME_STEP)):
            break

        delegation_idx = int(frame_id / FRAME_STEP)
        for cam in default_cameras:
            skin_path = sequence_dir / ("cam_" + cam) / f"skin_frame_{delegation_idx:06d}.json"
            if not skin_path.exists():
                return False, frame_id

        frame_id += step

    return True, None


def iter_actor_sequence_dirs(landmarks_root: Path):
    for actor_dir in sorted(landmarks_root.iterdir()):
        if not actor_dir.is_dir():
            continue
        for sequence_dir in sorted(actor_dir.iterdir()):
            if sequence_dir.is_dir():
                yield actor_dir.name, sequence_dir.name, sequence_dir


def wrapped_mesh_exists(actor_id: str, sequence_id: str, frame_id: int) -> bool:
    wrapped_mesh = OUTPUT_ROOT / actor_id / sequence_id / str(frame_id) / "wrapped_mesh.obj"
    return wrapped_mesh.exists()


def mesh_obj_exists(actor_id: str, sequence_id: str, frame_id: int) -> bool:
    mesh_obj = OUTPUT_ROOT / actor_id / sequence_id / str(frame_id) / "mesh.obj"
    return mesh_obj.exists()


def load_actor_neutral_frame(landmarks_root: Path, actor_id: str) -> tuple[str, int] | None:
    """Return (sequence_id, raw frame index) of this ACTOR's one neutral/
    reference frame (via the frontend's "Save as Neutral" button; see
    FaceView/backend/app/services/landmark_store.py's save_neutral_frame()),
    or None if it hasn't been set yet. One neutral per actor now, not one per
    actor/sequence pair -- landmarks_root/<actor>/neutral_frame.json holds
    {"sequence_id": ..., "frame_idx": ...} identifying which sequence's video
    that frame actually belongs to, since a non-origin sequence has no
    neutral frame of its own to fall back on (see process_pair()'s own use of
    this, and run_pipeline_all.process_frame()'s matching resolution)."""
    neutral_frame_path = landmarks_root / actor_id / "neutral_frame.json"
    if not neutral_frame_path.exists():
        return None
    with open(neutral_frame_path) as f:
        data = json.load(f)
    return data["sequence_id"], data["frame_idx"]


def pairs_with_status(label_data: dict, wanted_status: str) -> list[tuple[str, str]]:
    """--confirmed-only/--unconfirmed-only's source list: every Nersemble pair
    at wanted_status in label_tracker.json (minus blacklisted sequences),
    straight from the tracker rather than discover_actor_sequences()'s
    TARGET_SEQUENCES scan, so no such pair is missed."""
    return sorted(
        (actor_id, sequence_id)
        for actor_id, sequences in label_data.get("Nersemble", {}).items()
        for sequence_id, status in sequences.items()
        if status == wanted_status and sequence_id not in BLACKLISTED_SEQUENCES
    )


def rewrap_status(args: argparse.Namespace) -> str | None:
    """The label status a one-shot re-wrap run targets, or None for a normal run."""
    if getattr(args, "confirmed_only", False):
        return "confirmed"
    if getattr(args, "unconfirmed_only", False):
        return "unconfirmed"
    return None


def rewrap_done_json(status: str) -> Path:
    return REWRAP_DONE_JSON if status == "confirmed" else REWRAP_UNCONFIRMED_DONE_JSON


def rewrap_neutral_frame(
    actor_id: str,
    landmarks_root: Path,
    args: argparse.Namespace,
    log_dir: Path | None,
    default_cameras,
    skin_vertex_indices: list[str],
) -> tuple[str, bool]:
    """--confirmed-only/--unconfirmed-only pass 1: re-wrap this actor's neutral frame with the
    current pipeline before any of its pairs, since every other frame of the
    actor wraps against the neutral frame's wrapped_mesh.obj. Returns
    (actor_id, success)."""
    neutral = load_actor_neutral_frame(landmarks_root, actor_id)
    if neutral is None:
        print(f"SKIP [actor={actor_id}] neutral re-wrap: no neutral_frame.json")
        return actor_id, False
    neutral_sequence_id, neutral_frame_id = neutral
    tag = f"[actor={actor_id} neutral={neutral_sequence_id}/{neutral_frame_id}]"
    print(f"RUN  neutral re-wrap process_frame({actor_id}, {neutral_sequence_id}, {neutral_frame_id})  {tag}")
    if args.dry_run:
        return actor_id, True
    try:
        if log_dir is None:
            process_frame(actor_id, neutral_sequence_id, int(neutral_frame_id), args.resolution,
                          default_cameras, skin_vertex_indices, segmentation_lock=None)
        else:
            log_path = log_dir / f"{actor_id}_{neutral_sequence_id}_{neutral_frame_id}_neutral.log"
            with open(log_path, "w") as log_file, contextlib.redirect_stdout(log_file):
                process_frame(actor_id, neutral_sequence_id, int(neutral_frame_id), args.resolution,
                              default_cameras, skin_vertex_indices, segmentation_lock=None)
        return actor_id, True
    except Exception as exc:
        print(f"FAIL {tag} neutral re-wrap: {exc}")
        return actor_id, False


def export_pt(actor_id: str, sequence_id: str) -> str | None:
    """final_wrapped_data/<actor>/<sequence>.pt from the pair's current
    wrapped_mesh.obj files, via FaceView's export_wrapped_sequence (what its
    Confirm uses), run inside the FaceView repo: wrap_script/ has its own small
    backend/ package that shadows FaceView's, so it can't be imported from
    here. Returns the written path, or None on failure."""
    code = ("import sys; from backend.app.services.wrapped_mesh_export import export_wrapped_sequence; "
            f"p = export_wrapped_sequence({actor_id!r}, {sequence_id!r}); print(p or ''); sys.exit(0 if p else 1)")
    proc = subprocess.run([sys.executable, "-c", code], cwd=str(FACEVIEW_ROOT), capture_output=True, text=True)
    exported = proc.stdout.strip().splitlines()[-1] if proc.returncode == 0 and proc.stdout.strip() else None
    if exported is None:
        print(f"  export stderr ({actor_id}/{sequence_id}): {proc.stderr.strip()[-500:]}")
    return exported


def pt_is_stale(actor_id: str, sequence_id: str) -> bool:
    """The pair's .pt is missing or older than its newest wrapped_mesh.obj."""
    pt_path = FINAL_WRAPPED_DATA_ROOT / actor_id / f"{sequence_id}.pt"
    wraps = list((OUTPUT_ROOT / actor_id / sequence_id).glob("*/wrapped_mesh.obj"))
    if not wraps:
        return False
    newest = max(p.stat().st_mtime for p in wraps)
    return not pt_path.exists() or pt_path.stat().st_mtime < newest


def reexport_stale_confirmed(actor_ids, label_tracker_path: Path) -> None:
    """--confirmed-only sweep at the end of a run: every still-confirmed pair of
    these actors whose .pt is older than its wraps is re-exported -- e.g. the
    neutral pair after pass 1 re-wrapped its neutral frame although the pair
    itself was already done, or a pair with frames that failed to re-wrap."""
    statuses = json.loads(label_tracker_path.read_text()).get("Nersemble", {}) if label_tracker_path.exists() else {}
    stale = [(a, s) for a in sorted(actor_ids) for s, st in sorted(statuses.get(a, {}).items())
             if st == "confirmed" and pt_is_stale(a, s)]
    print(f"Export sweep: {len(stale)} confirmed pair(s) of {len(actor_ids)} actor(s) with a .pt older than their wraps")
    for a, s in stale:
        exported = export_pt(a, s)
        print(f"  {'RE-EXPORTED' if exported else 'FAIL export'} [actor={a} sequence={s}]" + (f" -> {exported}" if exported else ""))


def finish_rewrapped_pair(actor_id: str, sequence_id: str) -> bool:
    """--confirmed-only, after every frame of a pair re-wrapped cleanly: back up
    the old .pt (first time only), re-render both videos, re-export the .pt
    (FaceView's own export_wrapped_sequence, as Confirm uses), record the
    pair in REWRAP_DONE_JSON. Returns whether the export succeeded."""
    import shutil

    tag = f"[actor={actor_id} sequence={sequence_id}]"
    pt_path = FINAL_WRAPPED_DATA_ROOT / actor_id / f"{sequence_id}.pt"
    backup = FINAL_WRAPPED_BACKUP_ROOT / actor_id / f"{sequence_id}.pt"
    if pt_path.exists() and not backup.exists():
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(pt_path, backup)
        print(f"BACKUP {tag}: {pt_path} -> {backup}")
    render_overlay_video(actor_id, sequence_id)
    render_point_overlay_video(actor_id, sequence_id)
    exported = export_pt(actor_id, sequence_id)
    if exported is None:
        print(f"FAIL {tag}: .pt re-export failed -- not recorded in {REWRAP_DONE_JSON.name}")
        return False
    record_completed(REWRAP_DONE_JSON, actor_id, sequence_id)
    print(f"DONE {tag}: re-wrapped, videos re-rendered, .pt re-exported to {exported}")
    return True


def finish_rewrapped_unconfirmed_pair(actor_id: str, sequence_id: str) -> bool:
    """--unconfirmed-only, after every frame of a pair re-wrapped cleanly:
    re-render both videos and record the pair in REWRAP_UNCONFIRMED_DONE_JSON.
    No .pt backup/export -- that happens when the pair is Confirmed."""
    tag = f"[actor={actor_id} sequence={sequence_id}]"
    render_overlay_video(actor_id, sequence_id)
    render_point_overlay_video(actor_id, sequence_id)
    record_completed(REWRAP_UNCONFIRMED_DONE_JSON, actor_id, sequence_id)
    print(f"DONE {tag}: re-wrapped, videos re-rendered (still 'unconfirmed')")
    return True


@dataclass
class PairResult:
    actor_id: str
    sequence_id: str
    total_jobs: int = 0
    passed: int = 0
    failed: int = 0
    skipped_existing: int = 0


def process_pair(
    actor_id: str,
    sequence_id: str,
    landmarks_root: Path,
    args: argparse.Namespace,
    log_dir: Path | None,
    default_cameras,
    skin_vertex_indices: list[str],
) -> PairResult:
    """Run one actor/sequence pair's frames in frame order, INCLUDING the
    neutral frame (later frames wrap against the neutral frame's
    wrapped_mesh.obj output, so that sequential dependency is preserved).
    The neutral frame must already have been wrapped once by
    run_neutral_cotracker_propagation_wrapping_test.py (without skin points --
    this pair is skipped entirely if that hasn't happened yet), but this
    function re-runs process_frame() on it anyway, same as every other frame,
    to upgrade it to include skin points now that skin_frame_<idx>.json exists
    for it too. Safe to call concurrently for different pairs -- dispatch()
    below runs each pair in its own OS process (see its own note on why), so
    there is no in-memory state shared between concurrent calls of this
    function at all; every point of cross-pair/cross-cluster coordination
    goes through a flock-guarded file instead (record_completed(),
    mark_unconfirmed(), and process_frame()'s own per-frame .pipeline.lock).

    This is CPU-only (see run_pipeline_all.py's own top-of-file note --
    CUDA_VISIBLE_DEVICES is cleared before anything imports torch, so FLAME
    is the only torch user left and it always runs on CPU); there used to be
    a shared, GPU-resident SAM/U2Net segmentation step here too, but
    process_frame() dropped it (see its own docstring) -- there is nothing
    GPU-resident left to share across pairs, so nothing about --gpu-ids or a
    single-process-per-GPU model applies anymore either."""
    result = PairResult(actor_id, sequence_id)
    tag = f"[actor={actor_id} sequence={sequence_id}]"

    # Exclusive per-PAIR lock (not per-frame): claims the whole actor/sequence
    # pair atomically before doing any work, so at most one cluster node ever
    # runs this pair's frame loop at a time. Non-blocking -- if another
    # cluster already holds it, skip the whole pair immediately and let the
    # caller move on to a different pair, rather than tie up this worker
    # waiting on a pair that could take many minutes.
    #
    # This replaces per-frame locking as the mechanism for cross-cluster
    # coordination of THIS script specifically. Per-frame locking alone
    # (process_frame()'s own .pipeline.lock) is insufficient here: two
    # clusters could each independently run this pair's full frame loop,
    # each blocking-then-redoing whichever frames the other is mid-flight on
    # -- wasteful, and fragile to reason about together with shortcuts like
    # --skip-meshroom's mesh_obj_exists() check, which only confirms
    # Meshroom's mesh.obj exists, not that the frame's full wrap finished.
    # A third cluster hitting that shortcut mid-way through another cluster's
    # still-in-progress frame could treat it as done, reach the end of the
    # pair early, and call render_overlay_video()/mark_unconfirmed() while
    # frames are still being wrapped elsewhere. Claiming the whole pair up
    # front removes that possibility entirely: only the one owner ever
    # iterates this pair's frames or reaches the end-of-pair tail.
    #
    # process_frame()'s own per-frame lock is left in place regardless --
    # it still protects this pair's frame output_dirs against a *different*
    # entry point touching the same frame concurrently (e.g. someone manually
    # running run_pipeline_wrapping_test.py directly on one frame while this
    # batch script also holds the pair), which a pair-scoped lock here
    # wouldn't cover.
    pair_lock_dir = OUTPUT_ROOT / actor_id / sequence_id
    pair_lock_dir.mkdir(parents=True, exist_ok=True)
    pair_lock_file = open(pair_lock_dir / ".pipeline_pair.lock", "w")
    try:
        fcntl.flock(pair_lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print(f"SKIP {tag}: pair is locked by another run already processing it")
        pair_lock_file.close()
        return result

    try:
        sequence_dir = landmarks_root / actor_id / sequence_id
        if not sequence_dir.is_dir():
            print(f"SKIP {tag}: folder not found at {sequence_dir}")
            return result

        precomputed_gz_path = MESHROOM_DATA_ROOT / actor_id / sequence_id / "frame_0" / "mesh.obj.gz"
        if not precomputed_gz_path.exists():
            print(f"SKIP {tag}: no precomputed scan at {precomputed_gz_path}")
            return result

        neutral = load_actor_neutral_frame(landmarks_root, actor_id)
        if neutral is None:
            print(f"SKIP {tag}: no neutral_frame.json found at {landmarks_root / actor_id / 'neutral_frame.json'}")
            return result
        neutral_sequence_id, neutral_frame_id = neutral
        if not wrapped_mesh_exists(actor_id, neutral_sequence_id, neutral_frame_id):
            print(f"SKIP {tag}: neutral frame {neutral_frame_id} (from {actor_id}/{neutral_sequence_id}) has not "
                  f"been wrapped yet (missing {OUTPUT_ROOT / actor_id / neutral_sequence_id / str(neutral_frame_id) / 'wrapped_mesh.obj'}) "
                  "— run run_neutral_cotracker_propagation.py first.")
            return result

        face_mask_path = actor_face_mask_path(actor_id, neutral_sequence_id)
        if not face_mask_files_exist(face_mask_path):
            print(f"SKIP {tag}: no face mask for {actor_id} yet ({face_mask_path}*) -- "
                  "run run_neutral_cotracker_propagation.py first.")
            return result

        if skin_vertex_indices:
            skin_complete, missing_frame = skin_landmarks_complete(sequence_dir, default_cameras, args.max_frame, neutral_frame_id,
                                                                  step=wrap_frame_step(actor_id))
            if not skin_complete:
                print(f"SKIP {tag}: skin landmarks missing for frame {missing_frame} (and possibly "
                      "later frames too) -- run run_neutral_cotracker_propagation.py for this pair first")
                return result

        sequence_frames_ran = 0
        sequence_failed = False

        def run_job(frame_id: int) -> int:
            """Calls process_frame() directly (in-process) and converts any
            exception into a nonzero returncode, matching the old
            subprocess-returncode convention so the rest of this function (result
            counting, sequence_failed, etc.) didn't need to change."""
            try:
                if log_dir is None:
                    process_frame(
                        actor_id, sequence_id, frame_id, args.resolution,
                        default_cameras, skin_vertex_indices,
                        segmentation_lock=None,
                    )
                else:
                    log_path = log_dir / f"{actor_id}_{sequence_id}_{frame_id}.log"
                    with open(log_path, "w") as log_file, contextlib.redirect_stdout(log_file):
                        process_frame(
                            actor_id, sequence_id, frame_id, args.resolution,
                            default_cameras, skin_vertex_indices,
                            segmentation_lock=None,
                        )
                return 0
            except Exception as exc:
                print(f"FAIL {tag} frame={frame_id}: {exc}")
                return 1

        rewrap_target = rewrap_status(args)
        rewrap = rewrap_target is not None

        wrap_step = wrap_frame_step(actor_id)  # 25 for actors 102+, else 5
        frame_id = 0
        while True:
            if args.max_frame is not None and frame_id > args.max_frame:
                break

            if not sequence_has_frame(sequence_dir, int(frame_id / FRAME_STEP)):
                print(f"STOP {tag} frame={frame_id}: missing frame data")
                if rewrap:
                    if not args.dry_run and sequence_frames_ran > 0 and not sequence_failed:
                        if rewrap_target == "confirmed":
                            finish_rewrapped_pair(actor_id, sequence_id)
                        else:
                            finish_rewrapped_unconfirmed_pair(actor_id, sequence_id)
                    elif sequence_failed and rewrap_target == "confirmed" and not args.dry_run:
                        # Export what did re-wrap (failed frames keep their previous
                        # wrapped_mesh.obj); not recorded as done, so a later run retries.
                        exported = export_pt(actor_id, sequence_id)
                        print(f"PARTIAL {tag}: some frames failed to re-wrap -- .pt "
                              + (f"re-exported anyway to {exported}" if exported else "re-export FAILED")
                              + f"; not recorded in {REWRAP_DONE_JSON.name}, so it is retried next run")
                    elif sequence_failed:
                        print(f"NOT EXPORTED {tag}: some frames failed to re-wrap -- "
                              "videos not re-rendered, not recorded as done")
                    break
                # Record the sequence as completed once all its frames ran cleanly.
                if not args.dry_run and sequence_frames_ran > 0 and not sequence_failed:
                    record_completed(DONE_JSON, actor_id, sequence_id)
                    print(f"DONE {tag}: recorded in {DONE_JSON.name}")
                    moved = mark_unconfirmed(get_settings().label_tracker_path, actor_id, sequence_id)
                    if moved:
                        render_overlay_video(actor_id, sequence_id)
                        render_point_overlay_video(actor_id, sequence_id)
                    else:
                        print(f"SKIP {tag}: overlay video render skipped -- another run already "
                              "finished this pair (see NOTE above)")
                break

            if rewrap and sequence_id == neutral_sequence_id and frame_id == int(neutral_frame_id):
                # Already re-wrapped in pass 1.
                print(f"SKIP {tag} frame={frame_id}: neutral frame, re-wrapped in pass 1")
                sequence_frames_ran += 1
                frame_id += wrap_step
                continue

            if not rewrap and wrapped_mesh_exists(actor_id, sequence_id, frame_id):
                # Unconditional (unlike --skip-meshroom below): a frame whose
                # wrap already ran successfully has nothing left to do, and
                # counts toward sequence_frames_ran (not just
                # skipped_existing) so a sequence made ENTIRELY of
                # already-wrapped frames still reaches the STOP block's
                # "record as completed" / mark_unconfirmed() path below,
                # same as if every frame had just been (re)run here. To force
                # a real re-wrap of a frame (e.g. after a reviewer sends a
                # pair back from "unconfirmed" to "unreviewed"), its
                # wrapped_mesh.obj must be moved/renamed out of the way first
                # -- see FaceView's unconfirmed->unreviewed handler, which
                # renames it to old_wrapped_mesh.obj for exactly this reason.
                print(f"SKIP {tag} frame={frame_id}: wrapped_mesh.obj already exists")
                result.skipped_existing += 1
                sequence_frames_ran += 1
                frame_id += wrap_step
                continue

            if not rewrap and args.skip_meshroom and mesh_obj_exists(actor_id, sequence_id, frame_id):
                print(f"SKIP {tag} frame={frame_id}: mesh.obj exists")
                result.skipped_existing += 1
                frame_id += wrap_step
                continue

            result.total_jobs += 1
            print(f"RUN  process_frame({actor_id}, {sequence_id}, {frame_id}, {args.resolution})  {tag}")

            if args.dry_run:
                result.passed += 1
                frame_id += wrap_step
                continue

            returncode = run_job(frame_id)
            if returncode == 0:
                result.passed += 1
                sequence_frames_ran += 1
            else:
                result.failed += 1
                sequence_failed = True
                print(f"FAIL {tag} frame={frame_id} exit_code={returncode}")

            frame_id += wrap_step

        return result
    finally:
        fcntl.flock(pair_lock_file, fcntl.LOCK_UN)
        pair_lock_file.close()


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run run_pipeline_all.py for all actor/sequence pairs at frame multiples of 5 "
            "until frame data is missing. Each pair's neutral frame must already be wrapped "
            "(via run_neutral_cotracker_propagation.py) before its other frames can run here."
        )
    )
    parser.add_argument(
        "--landmarks-root",
        default=str(nersemble_paths.LANDMARKS_ROOT),
        help="Root landmarks folder with actor/sequence subfolders.",
    )
    parser.add_argument(
        "--resolution",
        default="3208",
        help="Resolution arg passed to run_pipeline_all.process_frame().",
    )
    parser.add_argument(
        "--max-frame",
        type=int,
        default=None,
        help="Optional upper bound for frame ids.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing.",
    )
    parser.add_argument(
        "--target-only",
        action="store_true",
        help="Only iterate actors in TARGET_ACTORS instead of every actor under DATA_ROOT.",
    )
    parser.add_argument(
        "--run-completed",
        action="store_true",
        help=(
            f"Only run actor/sequence pairs listed in {RUN_COMPLETED_JSON}, instead of "
            "sweeping every actor under DATA_ROOT for TARGET_SEQUENCES. Overrides --target-only."
        ),
    )
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        help=(
            f"Skip actor/sequence pairs already recorded as done in {DONE_JSON}."
        ),
    )
    parser.add_argument(
        "--ignore-list",
        action="store_true",
        help=(
            f"Skip actor/sequence pairs in ignore_actor_sequence_pairs.json "
            f"({len(IGNORE_ACTOR_SEQUENCE_PAIRS)} entries) and every sequence of the actors in "
            f"ignore_actors.json ({len(IGNORE_ACTORS)} entries) -- see ignore_lists.py."
        ),
    )
    parser.add_argument(
        "--skip-meshroom",
        action="store_true",
        help=(
            "Skip frames whose mesh.obj already exists in OUTPUT_ROOT (i.e. meshroom "
            "reconstruction already ran for that frame)."
        ),
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help=(
            "Number of actor/sequence pairs to process concurrently (default: 1, i.e. "
            "today's fully-sequential behavior). Each pair's own frames still run in "
            "order within that pair (later frames wrap against the neutral frame's "
            "output), but different pairs have no such dependency, so this launches "
            "that many worker PROCESSES at once (see dispatch()'s own note on why it's "
            "processes, not threads -- GIL contention among the Python-level work "
            "surrounding each WrapCmd subprocess call made threads scale badly). Pair "
            "with --wrap-max-threads so parallel workers x wrap-max-threads roughly "
            "matches your job's actual core allocation."
        ),
    )
    parser.add_argument(
        "--log-dir",
        default=None,
        help=(
            "Directory to write each job's stdout/stderr to (<actor>_<sequence>_<frame>.log) "
            "instead of the shared terminal. Strongly recommended whenever --parallel > 1: "
            "concurrent jobs' output (including tqdm progress bars) otherwise interleaves "
            "illegibly on one terminal."
        ),
    )
    parser.add_argument(
        "--gpu-ids",
        nargs="+",
        default=None,
        help=(
            "Deprecated / no-op now that this script runs every pair in-process instead of "
            "one subprocess per frame: CUDA_VISIBLE_DEVICES-based round-robin distribution "
            "required each pair to be its own subprocess. FLAME and the SAM/U2Net "
            "segmentation models are now each a single process-wide instance on one GPU "
            "(whatever this process itself sees as cuda:0), shared by every pair -- passing "
            "this flag prints a warning and is otherwise ignored. Run separate "
            "CUDA_VISIBLE_DEVICES-pinned invocations of this script (one per GPU, disjoint "
            "--target-only/--run-completed pair sets) if you need multi-GPU spread."
        ),
    )
    parser.add_argument(
        "--wrap-max-threads",
        type=int,
        default=2,
        help=(
            "Caps WrapCmd's own internal thread pool per invocation (passed as WrapCmd "
            "compute's --maxThreads). Without this, each of --parallel's concurrent "
            "wrap_mesh() calls independently sizes its thread pool off every core the "
            "cgroup allocation exposes, so N concurrent workers can end up running N x "
            "that-many threads on a machine with only that-many cores total -- "
            "oversubscription that gets WORSE the more cores a node reports. Size this as "
            "roughly (cpus-per-task / --parallel). Default: 2."
        ),
    )
    parser.add_argument(
        "--confirmed-only", action="store_true",
        help=(
            "One-shot: RE-WRAP every 'confirmed' pair (every frame, landmarks re-triangulated and "
            "re-filtered), re-render its overlay + point-overlay videos, and re-export its "
            "final_wrapped_data .pt (old .pt backed up to "
            f"{FINAL_WRAPPED_BACKUP_ROOT}). Each selected actor's neutral frame is re-wrapped first. "
            f"Resumable via {REWRAP_DONE_JSON.name}. Labels stay 'confirmed'."
        ),
    )
    parser.add_argument(
        "--unconfirmed-only", action="store_true",
        help=(
            "One-shot: RE-WRAP every 'unconfirmed' pair (every frame, landmarks re-triangulated and "
            "re-filtered) and re-render its overlay + point-overlay videos. Each selected actor's "
            "neutral frame is re-wrapped first. No .pt export (that happens on Confirm). "
            f"Resumable via {REWRAP_UNCONFIRMED_DONE_JSON.name}. Labels stay 'unconfirmed'."
        ),
    )
    parser.add_argument(
        "--pairs", nargs="+", default=None,
        help="Only these actor:sequence pairs (e.g. 037:EXP-5-mouth), for --confirmed-only/--unconfirmed-only testing",
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help=(f"--confirmed-only/--unconfirmed-only: redo pairs already recorded in "
              f"{REWRAP_DONE_JSON.name} / {REWRAP_UNCONFIRMED_DONE_JSON.name}"),
    )
    parser.add_argument(
        "--min-actor", type=int, default=None,
        help="Only actors whose numeric id is >= this (e.g. --min-actor 102)",
    )
    args = parser.parse_args()
    if args.confirmed_only and args.unconfirmed_only:
        parser.error("--confirmed-only and --unconfirmed-only are mutually exclusive")

    # Read by run_wrap_script_wrapping_test.wrap_mesh()'s subprocess call to
    # WrapCmd compute -- set here (once, before any worker threads start)
    # rather than threaded through process_pair()/process_frame()/
    # run_pipeline()'s signatures. Safe to set once up front: every worker
    # thread's subprocess.run() inherits this same os.environ at call time.
    os.environ["WRAP_MAX_THREADS"] = str(args.wrap_max_threads)

    landmarks_root = Path(args.landmarks_root)
    log_dir = Path(args.log_dir) if args.log_dir else None
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)

    if args.gpu_ids:
        print("--gpu-ids is deprecated and ignored now that pairs run in-process on one "
              "shared GPU -- see --help.")

    # Load (or create) the completed-sequences record.
    completed = load_completed(DONE_JSON)
    if not args.dry_run and not DONE_JSON.exists():
        with open(DONE_JSON, "w") as f:
            json.dump(completed, f, indent=2)

    if not landmarks_root.exists():
        print(f"Landmarks root does not exist: {landmarks_root}")
        return 1

    # default_cameras/skin_vertex_indices are cheap (JSON reads) but
    # per-actor-independent, so loaded once here rather than reloaded every
    # round of the (possibly polling) discover/dispatch loop below. Each
    # gets pickled to every worker process dispatch() spawns -- cheap, since
    # both are plain JSON-derived data.
    #
    # There used to be a shared, GPU-resident SAM/U2Net segmentation step
    # here too (with its own segmentation_lock), warmed once per batch run
    # instead of reloaded per frame -- process_frame() has since dropped
    # segmentation entirely (see its own docstring), and this whole pipeline
    # is CPU-only by design (CUDA_VISIBLE_DEVICES is cleared before anything
    # imports torch, see run_pipeline_all.py's top-of-file note), so there is
    # nothing GPU-resident left to warm or share across pairs.
    default_cameras = load_default_cameras()
    skin_vertex_indices = load_skin_vertex_indices()

    label_tracker_path = get_settings().label_tracker_path

    def discover_and_gate() -> list[tuple[str, str]]:
        """(Re)builds this round's pair list: source list (--run-completed
        allowlist or a fresh discover_actor_sequences() scan) -> dedup -> only
        pairs currently "unreviewed" in label_tracker.json -> optional
        --skip-completed / --ignore-list filters. Safe to call repeatedly --
        every step re-reads its source from disk, so a later call picks up
        pairs that became "unreviewed" (or newly-unblocked) since the last
        call."""
        if args.run_completed:
            pairs = load_run_completed_pairs(RUN_COMPLETED_JSON)
        else:
            pairs = discover_actor_sequences(DATA_ROOT, target_only=args.target_only)

        # Guard against duplicate (actor_id, sequence_id) pairs in the source list (e.g. a
        # hand-edited RUN_COMPLETED_JSON, or a repeated TARGET_SEQUENCES entry) — with
        # --parallel > 1, a duplicate pair would otherwise get two concurrent workers
        # racing to write into the same output/<actor>/<sequence>/<frame>/ directories.
        before_dedup = len(pairs)
        pairs = list(dict.fromkeys(pairs))
        if len(pairs) != before_dedup:
            print(f"Deduplicated {before_dedup - len(pairs)} repeated actor/sequence pair(s)")

        # Only ever run pairs currently at label_tracker.json status "unreviewed" --
        # this script has no other gate against re-processing a pair a human has
        # since moved on to "unconfirmed"/"confirmed" (mark_unconfirmed() below only
        # guards against overwriting those statuses, it doesn't stop the frame work
        # itself from re-running).
        label_data = {}
        if label_tracker_path.exists():
            try:
                with open(label_tracker_path) as f:
                    label_data = json.load(f)
            except (json.JSONDecodeError, OSError):
                label_data = {}
        before = len(pairs)
        pairs = [
            (actor_id, sequence_id)
            for actor_id, sequence_id in pairs
            if get_pair_status(landmarks_root, label_data, actor_id, sequence_id) == "unreviewed"
        ]
        print(f"Gated to label_tracker.json status 'unreviewed': {before - len(pairs)} "
              f"pair(s) excluded, {len(pairs)} remaining")

        if args.skip_completed:
            before = len(pairs)
            # Reloaded fresh from disk each call, rather than reusing the
            # outer `completed` variable loaded once at startup -- workers
            # now record completions directly to DONE_JSON (see
            # record_completed()), not into any in-memory dict this closure
            # could see updated, so a later round of this polling loop would
            # otherwise never notice pairs other workers already finished.
            current_completed = load_completed(DONE_JSON)
            pairs = [
                (actor_id, sequence_id)
                for actor_id, sequence_id in pairs
                if sequence_id not in current_completed.get(actor_id, [])
            ]
            print(f"--skip-completed: {before - len(pairs)} pair(s) already in "
                  f"{DONE_JSON.name}, {len(pairs)} remaining")

        if args.ignore_list:
            before = len(pairs)
            pairs = [
                (actor_id, sequence_id)
                for actor_id, sequence_id in pairs
                if not is_ignored(actor_id, sequence_id)
            ]
            print(f"--ignore-list: {before - len(pairs)} pair(s) ignored, "
                  f"{len(pairs)} remaining")

        pairs = apply_min_actor(pairs)
        return pairs

    def dispatch(pairs: list[tuple[str, str]]) -> tuple[int, int, int, int]:
        """--parallel concurrent OS PROCESSES, not threads. This used to be a
        ThreadPoolExecutor -- fine for the GPU-sharing/model-warming reasons
        that motivated the original in-process design, but WrapCmd's own
        compute step is an external subprocess (releases the GIL while it
        runs) while everything AROUND it in process_frame()/run_pipeline()
        (triangulation, the FLAME mesh rebuild, gzip/shutil scan copying,
        JSON writes) is regular Python competing for that one GIL across
        every concurrent thread. That competition scales with thread count,
        not core count, so it got WORSE as --parallel and --cpus-per-task
        both grew, even though WrapCmd's own thread math (via
        --wrap-max-threads) was correctly sized -- a real 4.5x per-frame
        slowdown was observed going from 4 cpus/1 worker to 64 cpus/16
        workers despite an unchanged, correctly-sized 1:1 wrap-threads:cores
        ratio. A ProcessPoolExecutor gives every worker its own interpreter
        (and GIL), which is why process_pair()/record_completed() no longer
        share any in-memory state -- see their own docstrings.

        mp_context="spawn" rather than the POSIX default "fork": each
        worker re-imports run_pipeline_all/run_pipeline_batch_all fresh
        rather than inheriting this process's state via fork, which is the
        generally-safe choice whenever torch might be involved anywhere in
        the import graph (it also correctly re-runs run_pipeline_all.py's
        top-of-file CUDA_VISIBLE_DEVICES-clearing in every worker, though
        that's moot today since nothing here touches a GPU at all)."""
        total_jobs = passed = failed = skipped_existing = 0
        mp_context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=max(1, args.parallel), mp_context=mp_context) as executor:
            futures = {
                executor.submit(
                    process_pair, actor_id, sequence_id, landmarks_root,
                    args, log_dir, default_cameras, skin_vertex_indices,
                ): (actor_id, sequence_id)
                for actor_id, sequence_id in pairs
            }
            for future in as_completed(futures):
                result = future.result()
                total_jobs += result.total_jobs
                passed += result.passed
                failed += result.failed
                skipped_existing += result.skipped_existing
        return total_jobs, passed, failed, skipped_existing

    def apply_min_actor(pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
        """--min-actor: keep only actors with a numeric id >= the given value."""
        if args.min_actor is None:
            return pairs
        before = len(pairs)
        pairs = [(a, s) for a, s in pairs if a.isdigit() and int(a) >= args.min_actor]
        print(f"--min-actor {args.min_actor}: {before - len(pairs)} pair(s) excluded, {len(pairs)} remaining")
        return pairs

    target_status = rewrap_status(args)
    if target_status is not None:
        done_json = rewrap_done_json(target_status)
        label_data = json.loads(label_tracker_path.read_text()) if label_tracker_path.exists() else {}
        pairs = pairs_with_status(label_data, target_status)
        print(f"--{target_status}-only: {len(pairs)} {target_status} pair(s) in label_tracker.json")
        if args.pairs:
            wanted = {tuple(p.split(":", 1)) for p in args.pairs}
            pairs = [p for p in pairs if p in wanted]
            missing = sorted(wanted - set(pairs))
            if missing:
                print(f"--pairs: not {target_status} (skipped): {missing}")
        if args.ignore_list:
            before = len(pairs)
            pairs = [(a, s) for a, s in pairs if not is_ignored(a, s)]
            print(f"--ignore-list: {before - len(pairs)} pair(s) ignored, {len(pairs)} remaining")
        pairs = apply_min_actor(pairs)
        if not args.no_resume:
            done = load_completed(done_json)
            before = len(pairs)
            pairs = [(a, s) for a, s in pairs if s not in done.get(a, [])]
            print(f"resume: {before - len(pairs)} pair(s) already re-wrapped ({done_json.name}), {len(pairs)} to do")
        if not pairs:
            print("Nothing to re-wrap.")
            return 0

        # Pass 1: each selected actor's neutral frame, in parallel across actors.
        actors = sorted({a for a, _ in pairs})
        print(f"Pass 1: re-wrapping the neutral frame of {len(actors)} actor(s)")
        ok_actors: set[str] = set()
        with ProcessPoolExecutor(max_workers=max(1, args.parallel), mp_context=multiprocessing.get_context("spawn")) as executor:
            futures = [executor.submit(rewrap_neutral_frame, a, landmarks_root, args, log_dir, default_cameras, skin_vertex_indices)
                       for a in actors]
            for future in as_completed(futures):
                actor_id, ok = future.result()
                if ok:
                    ok_actors.add(actor_id)
        failed_actors = sorted(set(actors) - ok_actors)
        if failed_actors:
            print(f"Neutral re-wrap failed for {failed_actors} -- their pairs are skipped this run")
        pairs = [(a, s) for a, s in pairs if a in ok_actors]

        # Pass 2: the pairs themselves.
        print(f"Pass 2: re-wrapping {len(pairs)} pair(s)")
        total_jobs, passed, failed, skipped_existing = dispatch(pairs)
        done_now = load_completed(done_json)
        not_done = [(a, s) for a, s in pairs if s not in done_now.get(a, [])]
        finished = "re-exported" if target_status == "confirmed" else "re-wrapped"
        print(f"Done. jobs={total_jobs} passed={passed} failed={failed}; "
              f"{len(pairs) - len(not_done)}/{len(pairs)} pair(s) {finished}")
        if not_done and not args.dry_run:
            print(f"Not {finished} (see FAIL/SKIP/NOT EXPORTED lines above): {not_done}")
        if target_status == "confirmed" and not args.dry_run:
            reexport_stale_confirmed(set(actors), label_tracker_path)
        return 0 if (failed == 0 and not failed_actors) else 2

    if args.run_completed or args.target_only:
        # Curated/manual runs: one-shot, same as before.
        if args.run_completed and not RUN_COMPLETED_JSON.exists():
            print(f"--run-completed given but {RUN_COMPLETED_JSON} does not exist")
            return 1
        pairs = discover_and_gate()
        total_jobs, passed, failed, skipped_existing = dispatch(pairs)
        print(f"Done. jobs={total_jobs} passed={passed} failed={failed} skipped_existing={skipped_existing}")
        return 0 if failed == 0 else 2

    # Default full sweep: poll label_tracker.json forever for newly-"unreviewed"
    # pairs instead of a single sweep-and-exit. Re-discovering a pair still
    # mid-flight from a previous round is safe -- process_pair()'s own
    # per-pair flock (.pipeline_pair.lock) makes a redundant discovery a no-op
    # skip rather than duplicated work.
    round_num = 0
    while True:
        round_num += 1
        print(f"[round {round_num}] Discovering pairs...")
        pairs = discover_and_gate()
        if pairs:
            total_jobs, passed, failed, skipped_existing = dispatch(pairs)
            print(f"[round {round_num}] Done. jobs={total_jobs} passed={passed} failed={failed} "
                  f"skipped_existing={skipped_existing}")
        print(f"[round {round_num}] Sleeping {POLL_INTERVAL_SECONDS}s before rescanning label_tracker.json...")
        time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    raise SystemExit(main())
