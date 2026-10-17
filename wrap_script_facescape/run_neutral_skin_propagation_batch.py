#!/usr/bin/env python3
"""Multi-GPU parallel batch wrapper for run_neutral_skin_propagation.py --
mirrors wrap_script/run_neutral_cotracker_propagation_batch.py's design for
the Nersemble side.

Default (no --pairs given) and --target-only modes dispatch one subprocess
PER ACTOR (not per expression): `run_neutral_skin_propagation.py <actor_id>`
with no --expressions triggers that script's own "propagate into every
unlabeled expression this actor has" sweep (process_actor()), sharing one
RAFT/CoTracker model load across all of that actor's pending expressions
instead of paying it per expression.

--pairs (explicit actor:expression list) dispatches one subprocess PER PAIR
instead (`run_neutral_skin_propagation.py <actor> --expressions <expression>`)
-- for testing/curating specific expressions, not sweeping an actor's whole
backlog.

No lock is held at this level -- run_neutral_skin_propagation.py takes its
own exclusive, non-blocking per-(actor,expression) flock internally
(_expression_pair_lock, facescape_landmarks_root/actor_id/expression_id/
.skin_pair.lock) around the actual per-expression work, shared by both its
--expressions path and its actor-sweep's own per-expression loop. A lock
held here too would self-conflict with that inner lock (a subprocess's own
flock() on a path can't see its own parent process's flock() on the same
path, since they're different open file descriptions) for no benefit -- see
wrap_script/run_neutral_cotracker_propagation_batch.py's own module
docstring for the fuller explanation of why.

By default (no --pairs/--target-only given), this POLLS FOREVER: each round
sweeps every actor with at least one expression currently "unlabeled" in
label_tracker.json (dataset="FaceScape"), dispatches that round's actors,
then sleeps POLL_INTERVAL_SECONDS and rescans. --pairs and --target-only both
remain ONE-SHOT instead, for curated/manual/testing runs.

Invalid pairs (invalid_pairs.json -- no origin-space version) are propagated
by run_neutral_skin_propagation_invalid.py instead (raw cameras, Ava-256-style
per-landmark camera selection); the regular script skips them. Each actor job
runs the regular script if the actor has an "unlabeled" valid expression, then
the _invalid script if it has an "unlabeled" invalid one. --pairs picks the
script per pair.

Usage:
    python3 run_neutral_skin_propagation_batch.py \\
        --pairs 1:2_smile 2:10_dimpler --parallel 2 --gpu-ids 0 1
        # one-shot, explicit pairs, one subprocess per pair

    python3 run_neutral_skin_propagation_batch.py --parallel 3 --gpu-ids 0 1 2
        # polls forever, one subprocess per actor, sweeping every actor
        # currently having an "unlabeled" expression each round

    python3 run_neutral_skin_propagation_batch.py --target-only --parallel 2
        # one-shot, one subprocess per actor in TARGET_LIST below
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
import facescape_paths as fp  # all paths: wrap_script_facescape/.env

sys.path.insert(0, str(Path(__file__).resolve().parent))
fp.add_faceview_to_sys_path()

from backend.app.core.config import get_settings  # noqa: E402
from backend.app.services.label_tracker import label_tracker  # noqa: E402

POLL_INTERVAL_SECONDS = 60

SCRIPT_PATH = Path(__file__).with_name("run_neutral_skin_propagation.py")
INVALID_SCRIPT_PATH = Path(__file__).with_name("run_neutral_skin_propagation_invalid.py")

# Actors to restrict the default sweep to when --target-only is given --
# same name/role as wrap_script/run_neutral_cotracker_propagation_batch.py's
# own TARGET_LIST. Empty by default; fill in as needed.
TARGET_LIST: list[str] = [
       "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
    "11", "12", "13", "14", "15", "16", "17", "18", "19", "20",
    "21", "22", "23", "24", "25", "26", "27", "28", "29", "30",
    "31", "32", "33" 
]


def load_invalid_pairs() -> set[tuple[str, str]]:
    """{"pairs": [{"actor": ..., "expression": ...}, ...]} -- same loader as
    run_pipeline_batch.py's (kept import-light, see discover_unlabeled_actors)."""
    path = get_settings().facescape_invalid_pairs_path
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return set()
    return {(str(p["actor"]), str(p["expression"])) for p in data.get("pairs", [])}


def discover_unlabeled_actors(target_only: bool) -> list[str]:
    """Every FaceScape actor_id with at least one expression currently
    "unlabeled" in label_tracker.json -- the default sweep's unit of
    dispatch (one subprocess per actor). Restricted to TARGET_LIST when
    target_only is set. Mirrors run_neutral_skin_propagation.py's own
    discover_unlabeled_actors(), reimplemented here since importing that
    module would pull in torch/RAFT/CoTracker at import time just for this
    one lookup, which this orchestrator (it never loads a model itself,
    only spawns subprocesses that do) has no other reason to depend on."""
    all_statuses = label_tracker.get_all(dataset="FaceScape")
    actor_ids = sorted(
        (actor_id for actor_id, expressions in all_statuses.items()
         if any(status == "unlabeled" for status in expressions.values())),
        key=int,
    )
    if target_only:
        actor_ids = [a for a in actor_ids if a in TARGET_LIST]
    return actor_ids


def parse_pairs(raw_pairs: list[str]) -> list[tuple[str, str]]:
    pairs = []
    for raw in raw_pairs:
        if ":" not in raw:
            raise ValueError(f"--pairs entries must be actor:expression, got {raw!r}")
        actor_id, expression_id = raw.split(":", 1)
        pairs.append((actor_id, expression_id))
    return pairs


@dataclass
class PairResult:
    actor_id: str
    expression_id: str
    returncode: int


def process_pair(
    actor_id: str,
    expression_id: str,
    extra_args: list[str],
    log_dir: Path | None,
    gpu_id: str | None,
) -> PairResult:
    """Runs one actor/expression pair as its own run_neutral_skin_propagation.py
    (run_neutral_skin_propagation_invalid.py for an invalid pair) subprocess -- used only for --pairs (explicit curated list) mode; the
    default/--target-only sweep modes dispatch per ACTOR instead (see
    process_actor_job()). ``gpu_id``, when given, restricts this subprocess
    (via CUDA_VISIBLE_DEVICES) to that single GPU. No lock held here -- see
    module docstring."""
    tag = f"[actor={actor_id} expression={expression_id}]" + (f" gpu={gpu_id}" if gpu_id is not None else "")
    script = INVALID_SCRIPT_PATH if (actor_id, expression_id) in load_invalid_pairs() else SCRIPT_PATH
    cmd = [sys.executable, str(script), actor_id, "--expressions", expression_id, *extra_args]
    print("RUN  " + " ".join(cmd) + f"  {tag}")

    job_env = None
    if gpu_id is not None:
        job_env = os.environ.copy()
        job_env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    if log_dir is None:
        returncode = subprocess.run(cmd, env=job_env).returncode
    else:
        log_path = log_dir / f"{actor_id}_{expression_id}.log"
        with open(log_path, "w") as log_file:
            returncode = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT, env=job_env).returncode

    status = "PASS" if returncode == 0 else "FAIL"
    print(f"{status} {tag} exit_code={returncode}")
    return PairResult(actor_id, expression_id, returncode)


@dataclass
class ActorResult:
    actor_id: str
    returncode: int


def _actor_scripts(actor_id: str) -> list[Path]:
    """Which propagation script(s) this actor needs this round: the regular one
    for "unlabeled" valid expressions, the _invalid one for "unlabeled"
    invalid expressions (the regular script skips those)."""
    invalid_pairs = load_invalid_pairs()
    unlabeled = [e for e, status in label_tracker.get_all(dataset="FaceScape").get(actor_id, {}).items()
                 if status == "unlabeled"]
    scripts = []
    if any((actor_id, e) not in invalid_pairs for e in unlabeled):
        scripts.append(SCRIPT_PATH)
    if any((actor_id, e) in invalid_pairs for e in unlabeled):
        scripts.append(INVALID_SCRIPT_PATH)
    return scripts


def process_actor_job(
    actor_id: str,
    extra_args: list[str],
    log_dir: Path | None,
    gpu_id: str | None,
) -> ActorResult:
    """Runs one actor through run_neutral_skin_propagation.py and/or
    run_neutral_skin_propagation_invalid.py (see _actor_scripts), each as its
    own subprocess with NO --expressions -- that script's actor-sweep mode,
    which loads RAFT/CoTracker once and propagates into every one of this
    actor's "unlabeled" expressions it handles. ``gpu_id``, when given,
    restricts the subprocesses (via CUDA_VISIBLE_DEVICES) to that single GPU,
    same pattern as process_pair(). No lock held here -- see module docstring.
    Returns the first non-zero exit code, else 0."""
    tag = f"[actor={actor_id}]" + (f" gpu={gpu_id}" if gpu_id is not None else "")

    job_env = None
    if gpu_id is not None:
        job_env = os.environ.copy()
        job_env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    final_returncode = 0
    for script in _actor_scripts(actor_id):
        invalid = script == INVALID_SCRIPT_PATH
        cmd = [sys.executable, str(script), actor_id, *extra_args]
        print("RUN  " + " ".join(cmd) + f"  {tag}")
        if log_dir is None:
            returncode = subprocess.run(cmd, env=job_env).returncode
        else:
            log_path = log_dir / (f"{actor_id}_invalid.log" if invalid else f"{actor_id}.log")
            with open(log_path, "w") as log_file:
                returncode = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT, env=job_env).returncode
        status = "PASS" if returncode == 0 else "FAIL"
        print(f"{status} {tag}{' invalid' if invalid else ''} exit_code={returncode}")
        if returncode != 0 and final_returncode == 0:
            final_returncode = returncode
    return ActorResult(actor_id, final_returncode)


def _dispatch(
    pairs: list[tuple[str, str]],
    extra_args: list[str],
    log_dir: Path | None,
    gpu_ids: list[str] | None,
    parallel: int,
) -> tuple[int, int]:
    before_dedup = len(pairs)
    pairs = list(dict.fromkeys(pairs))
    if len(pairs) != before_dedup:
        print(f"Deduplicated {before_dedup - len(pairs)} repeated actor/expression pair(s)")

    if gpu_ids:
        print(f"--gpu-ids: distributing {len(pairs)} pair(s) round-robin across GPUs {gpu_ids}")

    passed = failed = 0
    with ThreadPoolExecutor(max_workers=max(1, parallel)) as executor:
        futures = {
            executor.submit(
                process_pair, actor_id, expression_id, extra_args, log_dir,
                gpu_ids[i % len(gpu_ids)] if gpu_ids else None,
            ): (actor_id, expression_id)
            for i, (actor_id, expression_id) in enumerate(pairs)
        }
        for future in as_completed(futures):
            result = future.result()
            if result.returncode == 0:
                passed += 1
            else:
                failed += 1

    return passed, failed


def _dispatch_actors(
    actor_ids: list[str],
    extra_args: list[str],
    log_dir: Path | None,
    gpu_ids: list[str] | None,
    parallel: int,
) -> tuple[int, int]:
    before_dedup = len(actor_ids)
    actor_ids = list(dict.fromkeys(actor_ids))
    if len(actor_ids) != before_dedup:
        print(f"Deduplicated {before_dedup - len(actor_ids)} repeated actor id(s)")

    if gpu_ids:
        print(f"--gpu-ids: distributing {len(actor_ids)} actor(s) round-robin across GPUs {gpu_ids}")

    passed = failed = 0
    with ThreadPoolExecutor(max_workers=max(1, parallel)) as executor:
        futures = {
            executor.submit(
                process_actor_job, actor_id, extra_args, log_dir,
                gpu_ids[i % len(gpu_ids)] if gpu_ids else None,
            ): actor_id
            for i, actor_id in enumerate(actor_ids)
        }
        for future in as_completed(futures):
            result = future.result()
            if result.returncode == 0:
                passed += 1
            else:
                failed += 1

    return passed, failed


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--pairs", nargs="+", default=None,
        help=(
            "actor:expression pairs to run, one subprocess per pair, e.g. --pairs "
            "1:2_smile 2:10_dimpler. Omit to sweep every actor currently having an "
            "\"unlabeled\" expression instead, one subprocess per actor (see "
            "--target-only to restrict that sweep to TARGET_LIST)."
        ),
    )
    parser.add_argument(
        "--target-only", action="store_true",
        help="Only used when --pairs is omitted: restrict the \"unlabeled\" sweep to actors in TARGET_LIST.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Passed through to each run_neutral_skin_propagation.py subprocess.",
    )
    parser.add_argument(
        "--parallel", type=int, default=1,
        help=(
            "Number of concurrent subprocess workers (default: 1, i.e. fully "
            "sequential). Under --pairs, each pair gets its own subprocess; "
            "otherwise each ACTOR does (its own RAFT/CoTracker checkpoint load, "
            "reused across all of that actor's pending expressions) -- either way, "
            "this is the right knob for keeping multiple GPUs busy at once, see "
            "--gpu-ids."
        ),
    )
    parser.add_argument(
        "--log-dir", default=None,
        help=(
            "Directory to write each job's stdout/stderr to (<actor>_<expression>.log "
            "under --pairs, <actor>.log otherwise) instead of the shared terminal."
        ),
    )
    parser.add_argument(
        "--gpu-ids", nargs="+", default=None,
        help=(
            "GPU indices to spread work across (e.g. --gpu-ids 0 1 2 3), via "
            "CUDA_VISIBLE_DEVICES per subprocess. Jobs are assigned round-robin, one "
            "GPU per job for that job's whole run. Without this flag, every "
            "subprocess defaults to cuda:0."
        ),
    )
    args = parser.parse_args()

    log_dir = Path(args.log_dir) if args.log_dir else None
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)

    extra_args: list[str] = []
    if args.dry_run:
        extra_args.append("--dry-run")

    if args.pairs:
        pairs = parse_pairs(args.pairs)
        passed, failed = _dispatch(pairs, extra_args, log_dir, args.gpu_ids, args.parallel)
        print(f"Done. pairs={len(pairs)} passed={passed} failed={failed}")
        return 0 if failed == 0 else 2

    if args.target_only:
        actor_ids = discover_unlabeled_actors(target_only=True)
        print(f"--target-only: sweeping {len(actor_ids)} actor(s) restricted to {TARGET_LIST}")
        passed, failed = _dispatch_actors(actor_ids, extra_args, log_dir, args.gpu_ids, args.parallel)
        print(f"Done. actors={len(actor_ids)} passed={passed} failed={failed}")
        return 0 if failed == 0 else 2

    # Default full sweep: poll label_tracker.json forever for actors with
    # newly-"unlabeled" expressions instead of a single sweep-and-exit.
    # Rediscovering an actor still mid-flight from a previous round is safe
    # -- run_neutral_skin_propagation.py's own per-expression flock
    # (_expression_pair_lock) makes a redundant discovery a no-op skip
    # per-expression rather than duplicated work.
    round_num = 0
    while True:
        round_num += 1
        actor_ids = discover_unlabeled_actors(target_only=False)
        print(f"[round {round_num}] Sweeping {len(actor_ids)} actor(s) with an \"unlabeled\" expression")
        if actor_ids:
            passed, failed = _dispatch_actors(actor_ids, extra_args, log_dir, args.gpu_ids, args.parallel)
            print(f"[round {round_num}] Done. actors={len(actor_ids)} passed={passed} failed={failed}")
        print(f"[round {round_num}] Sleeping {POLL_INTERVAL_SECONDS}s before rescanning label_tracker.json...")
        time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    raise SystemExit(main())
