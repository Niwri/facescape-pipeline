#!/usr/bin/env python3
"""Multi-GPU parallel batch wrapper for run_neutral_cotracker_propagation.py.

Every mode dispatches one subprocess PER ACTOR/SEQUENCE PAIR:
`run_neutral_cotracker_propagation.py <actor_id> <sequence_id>`. The default
sweep and --target-only first collect EVERY non-blacklisted actor/sequence
pair marked "unlabeled" in label_tracker.json (a pair with no entry is
skipped) that also has a
/scratch/ondemand32/irwinngo/meshroom-tracking-full/<actor>/<sequence>/ directory
and delegation videos for every default camera under processed/
(see discover_unlabeled_pairs()), shuffle that list, then
run it in the shuffled order -- so when the dataset can't be finished in
time, the propagated (and, downstream, wrapped -- run_pipeline_batch_all.py
only picks up "unreviewed" pairs) subset is a random sample spread across
all actors instead of a depth-first run through the first few actors.
--seed makes the shuffle reproducible; by default every round reshuffles.

Trade-off vs. the old per-actor dispatch: RAFT/CoTracker are loaded once per
sequence instead of once per actor.

The neutral frame: run_neutral_cotracker_propagation.py's
ensure_actor_mask() wraps an actor's neutral frame (twice) and computes its
face mask the first time any pair of that actor runs, under its own
cross-process lock. process_pair() additionally holds a per-actor
threading.Lock while the actor's mask doesn't exist yet, so other pairs of
that actor wait here -- instead of each starting a child that loads
RAFT/CoTracker onto the GPU and then just blocks on that lock.

No file lock is held at this level --
run_neutral_cotracker_propagation.py takes its own exclusive, non-blocking
per-sequence flock internally (_sequence_pair_lock,
landmarks_root/actor_id/sequence_id/.cotracker_pair.lock) around the actual
per-sequence work. That's what protects against two concurrent runs (e.g.
two cluster nodes) touching the same sequence at once -- a flock held here
too would self-conflict with that inner lock for no benefit.

By default (no --pairs/--target-only given), this POLLS FOREVER: each round
collects and shuffles every "unlabeled" pair, dispatches them, then sleeps
POLL_INTERVAL_SECONDS and rescans -- so newly "unlabeled" sequences get
picked up without re-invoking this script. --pairs and --target-only are
ONE-SHOT, for curated/manual/testing runs (--pairs keeps the given order).

Usage:
    python3 run_neutral_cotracker_propagation_batch.py \\
        --pairs 017:EMO-1-shout+laugh 037:EXP-3-cheeks+nose 018:EMO-1-shout+laugh \\
        --skip-neutral-wrap --parallel 3 --gpu-ids 0 1 2
        # one-shot, explicit pairs, in the given order

    python3 run_neutral_cotracker_propagation_batch.py --parallel 3 --gpu-ids 0 1 2
        # polls forever, every "unlabeled" pair each round, in random order

    python3 run_neutral_cotracker_propagation_batch.py --target-only --parallel 2 --seed 0
        # one-shot, "unlabeled" pairs of actors in TARGET_LIST, reproducible order

    python3 run_neutral_cotracker_propagation_batch.py --mask-only [--unconfirmed] [--confirmed] --parallel 2
        # one-shot, no propagation, ONE job per actor of the selected pairs
        # ("unreviewed", plus "unconfirmed"/"confirmed" with those flags; or
        # --pairs): FLAME neutral mask (if missing) + the neutral frame's scan
        # mask and landmark filter, which run_pipeline_all.py applies to every
        # frame. Actors whose neutral frame has no Meshroom scan are skipped.
        # --all-frames: one job per pair instead, also writing every frame's
        # filtered JSON (inspection only, ~40 s per frame).
"""
from __future__ import annotations

import nersemble_paths
import argparse
import json
import os
import random
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from backend.app.core.config import get_settings
from sequence_blacklist import BLACKLISTED_SEQUENCES

POLL_INTERVAL_SECONDS = 60

SCRIPT_PATH = Path(__file__).with_name("run_neutral_cotracker_propagation.py")

# Same as run_neutral_cotracker_propagation.py's own OUTPUT_ROOT -- where
# wrap_neutral_frame() writes <actor>/<neutral_seq>/<neutral_frame>/wrapped_mesh.obj.
OUTPUT_ROOT = nersemble_paths.OUTPUT_ROOT
FACE_MASK_SUFFIXES = (".txt", "_inverse.txt", "_wrapping_node.txt", "_wrapping_node_inverse.txt")

# Same root as run_pipeline_all.py's MESHROOM_TRACKING_FULL_ROOT. The swept
# modes only pick pairs that have a <actor>/<sequence> directory here.
MESHROOM_TRACKING_FULL_ROOT = nersemble_paths.MESHROOM_DATA_ROOT

# Cameras run_neutral_cotracker_propagation.py tracks/reads delegation videos
# for (its load_default_cameras()); a pair is only runnable once
# processed/<actor>/<sequence>/cam_<id>_delegation.mp4 exists for all of them.
DEFAULT_CAMERAS_PATH = nersemble_paths.FACEVIEW_ROOT / "default_cameras.json"

# Actors to restrict the default sweep to when --target-only is given --
# same name/role as run_pipeline_batch_all.py's own TARGET_ACTORS, renamed
# per request. Empty by default; fill in as needed.
TARGET_LIST: list[str] = [
]


def get_explicit_status(label_data: dict, actor_id: str, sequence_id: str) -> str | None:
    """The pair's status exactly as written in label_tracker.json, or None
    when it has no entry. Unlike run_neutral_cotracker_propagation.py's
    get_pair_status() (and FaceView's LabelTracker.get_status), a missing
    entry is NOT treated as "unlabeled" here: the sweep only propagates pairs
    explicitly marked "unlabeled", so a stray landmarks/<actor>/<sequence>/
    folder with no tracker entry is never picked up."""
    return label_data.get("Nersemble", {}).get(actor_id, {}).get(sequence_id)


def _load_label_data(label_tracker_path: Path) -> dict:
    if not label_tracker_path.exists():
        return {}
    try:
        with open(label_tracker_path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def neutral_pair_with_scan(landmarks_root: Path, actor_id: str) -> tuple[str, str] | None:
    """(actor, neutral sequence) when the actor's neutral frame
    (neutral_frame.json) has a Meshroom scan, else None -- --mask-only's
    once-per-actor job."""
    try:
        with open(landmarks_root / actor_id / "neutral_frame.json") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    scan = MESHROOM_TRACKING_FULL_ROOT / actor_id / data["sequence_id"] / f"frame_{data['frame_idx']}" / "mesh.obj.gz"
    return (actor_id, data["sequence_id"]) if scan.exists() else None


def to_actor_jobs(pairs: list[tuple[str, str]], landmarks_root: Path) -> list[tuple[str, str]]:
    """--mask-only without --all-frames: one (actor, neutral sequence) job per
    actor in `pairs`, dropping actors whose neutral frame has no scan."""
    actors = list(dict.fromkeys(a for a, _ in pairs))
    jobs = [j for j in (neutral_pair_with_scan(landmarks_root, a) for a in actors) if j]
    skipped = sorted(set(actors) - {a for a, _ in jobs})
    if skipped:
        print(f"--mask-only: skipping {len(skipped)} actor(s) whose neutral frame has no Meshroom scan "
              f"(or no neutral_frame.json): {' '.join(skipped)}")
    return jobs


def has_meshroom_scans(actor_id: str, sequence_id: str) -> bool:
    """At least one precomputed scan (frame_*/mesh.obj.gz) for this pair --
    what --mask-only's scan masks and landmark filter need."""
    return any((MESHROOM_TRACKING_FULL_ROOT / actor_id / sequence_id).glob("frame_*/mesh.obj.gz"))


def _has_delegation_videos(processed_root: Path, actor_id: str, sequence_id: str, camera_ids: list[str]) -> bool:
    sequence_dir = processed_root / actor_id / sequence_id
    return all((sequence_dir / f"cam_{cam}_delegation.mp4").exists() for cam in camera_ids)


def _readable_dir(path: Path) -> bool:
    """is_dir() that treats an unreadable path (e.g. another user's folder still
    being written) as missing instead of raising PermissionError."""
    try:
        return path.is_dir()
    except OSError as exc:
        print(f"NOTE: skipping unreadable {path} ({exc.strerror})")
        return False


def discover_unlabeled_pairs(
    landmarks_root: Path, label_tracker_path: Path, target_only: bool, processed_root: Path,
    status: str | set[str] = "unlabeled", require_scans: bool = False, actor_ok=None,
) -> list[tuple[str, str]]:
    """Every (actor_id, sequence_id) where the actor has a neutral_frame.json,
    the sequence is non-blacklisted and explicitly "unlabeled" in
    label_tracker.json (see get_explicit_status() -- no entry means skip), MESHROOM_TRACKING_FULL_ROOT/<actor>/<sequence>/
    exists, and processed_root holds that sequence's delegation video for
    every DEFAULT_CAMERAS_PATH camera (without them the propagation fails at
    its first ffprobe). Restricted to TARGET_LIST's actors when target_only
    is set. Returned in sorted order -- the caller shuffles. `status` swaps
    "unlabeled" for another status, or a set of them (--mask-only sweeps
    "unreviewed", plus "unconfirmed"/"confirmed" with --unconfirmed/--confirmed);
    require_scans additionally needs at least one frame_*/mesh.obj.gz in that
    MESHROOM_TRACKING_FULL_ROOT folder (has_meshroom_scans). actor_ok(actor_id)
    -> bool (--min-actor/--max-actor) drops actors before any of their folders
    are read; unreadable folders are skipped with a NOTE."""
    statuses = {status} if isinstance(status, str) else set(status)
    label_data = _load_label_data(label_tracker_path)
    with open(DEFAULT_CAMERAS_PATH) as f:
        camera_ids: list[str] = json.load(f)

    pairs: list[tuple[str, str]] = []
    if not landmarks_root.is_dir():
        return pairs
    for actor_dir in sorted(landmarks_root.iterdir()):
        if not actor_dir.is_dir() or not (actor_dir / "neutral_frame.json").exists():
            continue
        actor_id = actor_dir.name
        if target_only and actor_id not in TARGET_LIST:
            continue
        if actor_ok is not None and not actor_ok(actor_id):
            continue
        meshroom_actor_dir = MESHROOM_TRACKING_FULL_ROOT / actor_id
        if not _readable_dir(meshroom_actor_dir):
            continue
        for sequence_dir in sorted(actor_dir.iterdir()):
            if not sequence_dir.is_dir() or sequence_dir.name in BLACKLISTED_SEQUENCES:
                continue
            if not _readable_dir(meshroom_actor_dir / sequence_dir.name):
                continue
            try:
                if require_scans and not has_meshroom_scans(actor_id, sequence_dir.name):
                    continue
            except OSError as exc:
                print(f"NOTE: skipping {actor_id}/{sequence_dir.name}: scans unreadable ({exc.strerror})")
                continue
            if get_explicit_status(label_data, actor_id, sequence_dir.name) not in statuses:
                continue
            if not _has_delegation_videos(processed_root, actor_id, sequence_dir.name, camera_ids):
                continue
            pairs.append((actor_id, sequence_dir.name))
    return pairs


def _neutral_wrapped_mesh_path(landmarks_root: Path, actor_id: str) -> Path | None:
    """OUTPUT_ROOT/<actor>/<neutral_seq>/<neutral_frame>/wrapped_mesh.obj for
    the actor's one neutral frame (landmarks_root/<actor>/neutral_frame.json),
    or None when the actor has no (readable) neutral_frame.json."""
    try:
        with open(landmarks_root / actor_id / "neutral_frame.json") as f:
            data = json.load(f)
        return OUTPUT_ROOT / actor_id / data["sequence_id"] / str(data["frame_idx"]) / "wrapped_mesh.obj"
    except (OSError, json.JSONDecodeError, KeyError):
        return None


# One lock per actor, held by whichever pair is wrapping that actor's neutral
# frame -- see process_pair().
_neutral_wrap_locks: dict[str, threading.Lock] = {}
_neutral_wrap_locks_guard = threading.Lock()

# Set on Ctrl+C (see _dispatch()): queued pairs stop launching, and every
# still-running child in _running_children is terminated.
_stop_event = threading.Event()
_running_children: set[subprocess.Popen] = set()
_running_children_guard = threading.Lock()


def _run_child(cmd: list[str], env: dict | None, log_path: Path | None) -> int:
    """subprocess.run() equivalent that registers the child so a Ctrl+C in
    _dispatch() can terminate it. Returns -2 without launching if a stop was
    already requested."""
    if _stop_event.is_set():
        return -2
    log_file = open(log_path, "w") if log_path is not None else None
    try:
        proc = subprocess.Popen(
            cmd, env=env,
            stdout=log_file, stderr=subprocess.STDOUT if log_file is not None else None,
        )
        with _running_children_guard:
            _running_children.add(proc)
        try:
            return proc.wait()
        finally:
            with _running_children_guard:
                _running_children.discard(proc)
    finally:
        if log_file is not None:
            log_file.close()


def _terminate_children(timeout: float = 20.0) -> None:
    with _running_children_guard:
        children = list(_running_children)
    for proc in children:
        if proc.poll() is None:
            proc.terminate()
    deadline = time.monotonic() + timeout
    for proc in children:
        try:
            proc.wait(timeout=max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            proc.kill()


def _neutral_wrap_lock(actor_id: str) -> threading.Lock:
    with _neutral_wrap_locks_guard:
        return _neutral_wrap_locks.setdefault(actor_id, threading.Lock())


@dataclass
class PairResult:
    actor_id: str
    sequence_id: str
    returncode: int


def process_pair(
    actor_id: str,
    sequence_id: str,
    extra_args: list[str],
    log_dir: Path | None,
    gpu_id: str | None,
    landmarks_root: Path,
    label_tracker_path: Path | None = None,
    expected_status: str | set[str] = "unlabeled",
) -> PairResult:
    """Runs one actor/sequence pair as its own run_neutral_cotracker_propagation.py
    subprocess. ``gpu_id``, when given, restricts this subprocess (via
    CUDA_VISIBLE_DEVICES) to that single GPU. Safe to call concurrently for
    different pairs.

    Runs under that actor's _neutral_wrap_lock until the actor's neutral wrap
    and face mask exist (see the module docstring); after that, freely.

    With ``label_tracker_path`` (the swept modes), the pair is skipped if it
    is no longer "unlabeled" by the time its turn comes -- a shuffled round
    can take long enough for that to change."""
    tag = f"[actor={actor_id} sequence={sequence_id}]" + (f" gpu={gpu_id}" if gpu_id is not None else "")

    if _stop_event.is_set():
        return PairResult(actor_id, sequence_id, -2)

    if label_tracker_path is not None:
        status = get_explicit_status(_load_label_data(label_tracker_path), actor_id, sequence_id)
        if status not in ({expected_status} if isinstance(expected_status, str) else expected_status):
            print(f"SKIP {tag}: status is now {status or 'missing from label_tracker.json'!r}")
            return PairResult(actor_id, sequence_id, 0)

    job_env = None
    if gpu_id is not None:
        job_env = os.environ.copy()
        job_env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    def run(args: list[str]) -> int:
        cmd = [sys.executable, str(SCRIPT_PATH), actor_id, sequence_id, *args]
        if _stop_event.is_set():
            return -2
        print("RUN  " + " ".join(cmd) + f"  {tag}")
        log_path = log_dir / f"{actor_id}_{sequence_id}.log" if log_dir is not None else None
        return _run_child(cmd, job_env, log_path)

    def neutral_ready() -> bool:
        """Neutral wrapped AND the actor's face mask (four files next to the
        neutral sequence's outputs) written by ensure_actor_mask()."""
        mesh_path = _neutral_wrapped_mesh_path(landmarks_root, actor_id)
        if mesh_path is None or not mesh_path.exists():
            return False
        mask_base = mesh_path.parent.parent / "face_list.txt"
        return all(Path(str(mask_base) + suffix).exists() for suffix in FACE_MASK_SUFFIXES)

    if neutral_ready():
        returncode = run(extra_args)
    else:
        with _neutral_wrap_lock(actor_id):
            returncode = run(extra_args)

    status = "PASS" if returncode == 0 else "FAIL"
    print(f"{status} {tag} exit_code={returncode}")
    return PairResult(actor_id, sequence_id, returncode)


def parse_pairs(raw_pairs: list[str]) -> list[tuple[str, str]]:
    pairs = []
    for raw in raw_pairs:
        if ":" not in raw:
            raise ValueError(f"--pairs entries must be actor:sequence, got {raw!r}")
        actor_id, sequence_id = raw.split(":", 1)
        pairs.append((actor_id, sequence_id))
    return pairs


def _dispatch(
    pairs: list[tuple[str, str]],
    extra_args: list[str],
    log_dir: Path | None,
    gpu_ids: list[str] | None,
    parallel: int,
    landmarks_root: Path,
    label_tracker_path: Path | None = None,
    expected_status: str | set[str] = "unlabeled",
) -> tuple[int, int]:
    """Runs one round of pairs through process_pair() across a --parallel
    thread pool, round-robining --gpu-ids the same way for every round.
    Pairs are submitted in list order, so a shuffled list runs (roughly --
    up to --parallel at a time) in that shuffled order."""
    before_dedup = len(pairs)
    pairs = list(dict.fromkeys(pairs))
    if len(pairs) != before_dedup:
        print(f"Deduplicated {before_dedup - len(pairs)} repeated actor/sequence pair(s)")

    if gpu_ids:
        print(f"--gpu-ids: distributing {len(pairs)} pair(s) round-robin across GPUs {gpu_ids}")

    passed = failed = 0
    executor = ThreadPoolExecutor(max_workers=max(1, parallel))
    try:
        futures = {
            executor.submit(
                process_pair, actor_id, sequence_id, extra_args, log_dir,
                gpu_ids[i % len(gpu_ids)] if gpu_ids else None, landmarks_root, label_tracker_path,
                expected_status,
            ): (actor_id, sequence_id)
            for i, (actor_id, sequence_id) in enumerate(pairs)
        }
        for future in as_completed(futures):
            result = future.result()
            if result.returncode == 0:
                passed += 1
            else:
                failed += 1
    except KeyboardInterrupt:
        # Ctrl+C: launch nothing else, stop what's running, then re-raise.
        _stop_event.set()
        executor.shutdown(wait=False, cancel_futures=True)
        print("\nInterrupted -- cancelling queued pairs and stopping running jobs...", flush=True)
        _terminate_children()
        raise
    finally:
        executor.shutdown(wait=True, cancel_futures=True)

    return passed, failed


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--pairs",
        nargs="+",
        default=None,
        help=(
            "actor:sequence pairs to run, one subprocess per pair, e.g. --pairs "
            "017:EMO-1-shout+laugh 037:EXP-3-cheeks+nose, run in the given order. Omit "
            "to sweep every \"unlabeled\" pair instead, in random order (see "
            "--target-only to restrict that sweep to TARGET_LIST's actors)."
        ),
    )
    parser.add_argument(
        "--target-only",
        action="store_true",
        help="Only used when --pairs is omitted: restrict the \"unlabeled\" sweep to actors in TARGET_LIST.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Seed for shuffling the swept pairs (default: none, i.e. a fresh random "
            "order every round). Ignored under --pairs."
        ),
    )
    parser.add_argument(
        "--skip-neutral-wrap", action="store_true",
        help="Passed through to each run_neutral_cotracker_propagation.py subprocess.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Passed through to each run_neutral_cotracker_propagation.py subprocess.",
    )
    parser.add_argument(
        "--no-triangulate", action="store_true",
        help="Passed through to each run_neutral_cotracker_propagation.py subprocess.",
    )
    parser.add_argument(
        "--mask-only", action="store_true",
        help=(
            "Passed through: no propagation, just the FLAME neutral-frame mask (if missing) and every "
            "frame's scan mask + scan-filtered landmark JSON. One-shot; without --pairs it sweeps "
            "\"unreviewed\" pairs (already propagated, not yet wrapped) instead of \"unlabeled\" ones. "
            "Only pairs with at least one scan in meshroom-tracking-full (frame_*/mesh.obj.gz) are run, "
            "with or without --pairs."
        ),
    )
    parser.add_argument(
        "--all-frames", action="store_true",
        help=("With --mask-only: one job per pair instead of per actor, also writing every frame's filtered JSON "
              "(passed through; inspection only -- wraps use the neutral frame's filter)."),
    )
    parser.add_argument(
        "--unconfirmed", action="store_true",
        help="With --mask-only (no --pairs): also sweep \"unconfirmed\" pairs, on top of \"unreviewed\".",
    )
    parser.add_argument(
        "--confirmed", action="store_true",
        help="With --mask-only (no --pairs): also sweep \"confirmed\" pairs, on top of \"unreviewed\".",
    )
    parser.add_argument(
        "--force-mask", action="store_true",
        help="With --mask-only: passed through, recomputes each actor's FLAME neutral-frame mask.",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help=(
            "Number of concurrent subprocess workers (default: 1, i.e. fully "
            "sequential). Each pair gets its own subprocess (and its own "
            "RAFT/CoTracker checkpoint load) -- this is the right knob for keeping "
            "multiple GPUs busy at once, see --gpu-ids."
        ),
    )
    parser.add_argument(
        "--log-dir",
        default=None,
        help=(
            "Directory to write each job's stdout/stderr to (<actor>_<sequence>.log) "
            "instead of the shared terminal. "
            "Strongly recommended whenever --parallel > 1: concurrent jobs' output "
            "(including tqdm progress bars) otherwise interleaves illegibly on one "
            "terminal."
        ),
    )
    parser.add_argument(
        "--gpu-ids",
        nargs="+",
        default=None,
        help=(
            "GPU indices to spread work across (e.g. --gpu-ids 0 1 2 3), as visible to "
            "THIS process -- i.e. matching `nvidia-smi -L` / $CUDA_VISIBLE_DEVICES inside "
            "the current allocation, not necessarily the machine's physical GPU numbering. "
            "Jobs are assigned round-robin, one GPU per job for that job's whole run "
            "(via CUDA_VISIBLE_DEVICES on each subprocess) -- so with --parallel higher "
            "than len(--gpu-ids), multiple concurrent jobs share each GPU, same as "
            "single-GPU contention but split across however many GPUs you give it. Without "
            "this flag, every subprocess defaults to cuda:0 -- with --parallel > 1 and more "
            "than one real GPU visible, that means every worker piles onto just the first "
            "GPU while the rest sit idle."
        ),
    )
    parser.add_argument("--min-actor", type=int, default=None, help="Only actors whose numeric id is >= this")
    parser.add_argument("--max-actor", type=int, default=None, help="Only actors whose numeric id is <= this")
    args = parser.parse_args()
    if args.min_actor is not None and args.max_actor is not None and args.min_actor > args.max_actor:
        parser.error("--min-actor is greater than --max-actor")

    def actor_ok(actor_id: str) -> bool:
        if args.min_actor is None and args.max_actor is None:
            return True
        if not actor_id.isdigit():
            return False
        return ((args.min_actor is None or int(actor_id) >= args.min_actor)
                and (args.max_actor is None or int(actor_id) <= args.max_actor))

    def in_actor_range(pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
        """--min-actor / --max-actor: keep only actors with a numeric id in range."""
        if args.min_actor is None and args.max_actor is None:
            return pairs
        lo = args.min_actor if args.min_actor is not None else float("-inf")
        hi = args.max_actor if args.max_actor is not None else float("inf")
        kept = [(a, s) for a, s in pairs if a.isdigit() and lo <= int(a) <= hi]
        print(f"--min-actor/--max-actor [{args.min_actor}, {args.max_actor}]: {len(pairs) - len(kept)} pair(s) excluded, "
              f"{len(kept)} remaining")
        return kept

    log_dir = Path(args.log_dir) if args.log_dir else None
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)

    extra_args: list[str] = []
    if args.skip_neutral_wrap:
        extra_args.append("--skip-neutral-wrap")
    if args.dry_run:
        extra_args.append("--dry-run")
    if args.no_triangulate:
        extra_args.append("--no-triangulate")
    if (args.force_mask or args.unconfirmed or args.confirmed or args.all_frames) and not args.mask_only:
        parser.error("--force-mask / --unconfirmed / --confirmed / --all-frames only apply with --mask-only")
    if args.mask_only:
        extra_args.append("--mask-only")
        if args.force_mask:
            extra_args.append("--force-mask")
        if args.all_frames:
            extra_args.append("--all-frames")

    settings = get_settings()
    rng = random.Random(args.seed)

    if args.pairs:
        # Explicit pair list: always one-shot, one subprocess per pair.
        pairs = in_actor_range(parse_pairs(args.pairs))
        if args.mask_only and not args.all_frames:
            pairs = to_actor_jobs(pairs, settings.landmarks_root)
            print(f"--mask-only: {len(pairs)} actor job(s) (neutral frame only)")
        elif args.mask_only:
            no_scans = [p for p in pairs if not has_meshroom_scans(*p)]
            if no_scans:
                print(f"--mask-only: skipping {len(no_scans)} pair(s) with no scans in {MESHROOM_TRACKING_FULL_ROOT}: "
                      + " ".join(f"{a}:{s}" for a, s in no_scans))
            pairs = [p for p in pairs if p not in no_scans]
        passed, failed = _dispatch(pairs, extra_args, log_dir, args.gpu_ids, args.parallel, settings.landmarks_root)
        print(f"Done. pairs={len(pairs)} passed={passed} failed={failed}")
        return 0 if failed == 0 else 2

    if args.mask_only:
        # One-shot sweep of propagated-but-unwrapped pairs (--pairs handled above).
        statuses = {"unreviewed"} | ({"unconfirmed"} if args.unconfirmed else set()) | ({"confirmed"} if args.confirmed else set())
        pairs = discover_unlabeled_pairs(settings.landmarks_root, settings.label_tracker_path, args.target_only,
                                         settings.processed_root, status=statuses, require_scans=True,
                                         actor_ok=actor_ok)
        pairs = in_actor_range(pairs)
        print(f"--mask-only: {len(pairs)} pair(s) with status in {sorted(statuses)} and Meshroom scans in "
              f"{MESHROOM_TRACKING_FULL_ROOT}")
        if args.all_frames:
            passed, failed = _dispatch(
                pairs, extra_args, log_dir, args.gpu_ids, args.parallel, settings.landmarks_root,
                label_tracker_path=settings.label_tracker_path, expected_status=statuses,
            )
        else:
            jobs = to_actor_jobs(pairs, settings.landmarks_root)
            print(f"--mask-only: {len(jobs)} actor job(s) (neutral frame only)")
            # Status was checked per pair above; the neutral pair's own status doesn't matter here.
            passed, failed = _dispatch(jobs, extra_args, log_dir, args.gpu_ids, args.parallel, settings.landmarks_root)
            pairs = jobs
        print(f"Done. pairs={len(pairs)} passed={passed} failed={failed}")
        return 0 if failed == 0 else 2

    if args.target_only:
        # Curated TARGET_LIST sweep: also one-shot.
        pairs = in_actor_range(discover_unlabeled_pairs(settings.landmarks_root, settings.label_tracker_path, True, settings.processed_root,
                                                         actor_ok=actor_ok))
        rng.shuffle(pairs)
        print(f"--target-only: {len(pairs)} \"unlabeled\" pair(s) of actors {TARGET_LIST}, in random order")
        passed, failed = _dispatch(
            pairs, extra_args, log_dir, args.gpu_ids, args.parallel, settings.landmarks_root,
            label_tracker_path=settings.label_tracker_path,
        )
        print(f"Done. pairs={len(pairs)} passed={passed} failed={failed}")
        return 0 if failed == 0 else 2

    # Default full sweep, no --pairs/--target-only: poll label_tracker.json
    # forever for newly-"unlabeled" pairs instead of a single sweep-and-exit,
    # reshuffling every round. Rediscovering a pair still mid-flight from a
    # previous round is safe -- run_neutral_cotracker_propagation.py's own
    # per-sequence flock (_sequence_pair_lock) makes a redundant discovery a
    # no-op skip rather than duplicated work.
    round_num = 0
    while True:
        round_num += 1
        pairs = in_actor_range(discover_unlabeled_pairs(settings.landmarks_root, settings.label_tracker_path, False, settings.processed_root,
                                                         actor_ok=actor_ok))
        rng.shuffle(pairs)
        print(f"[round {round_num}] {len(pairs)} \"unlabeled\" pair(s) across "
              f"{len({actor_id for actor_id, _ in pairs})} actor(s), in random order")
        if pairs:
            passed, failed = _dispatch(
                pairs, extra_args, log_dir, args.gpu_ids, args.parallel, settings.landmarks_root,
                label_tracker_path=settings.label_tracker_path,
            )
            print(f"[round {round_num}] Done. pairs={len(pairs)} passed={passed} failed={failed}")
        print(f"[round {round_num}] Sleeping {POLL_INTERVAL_SECONDS}s before rescanning label_tracker.json...")
        time.sleep(POLL_INTERVAL_SECONDS)

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Stopped.")
        raise SystemExit(130)
