#!/usr/bin/env python3
"""Multi-GPU wrapper for FaceView/scripts/preprocess/process_expression_sapiens.py.
FaceScape counterpart to wrap_script/run_process_sequence_sapiens_batch.py --
same actor-range-sharding design (that script's whole point is reusing ONE
warmed-up DETR detector + sapiens2 pose model across many pairs via its own
internal --parallel thread pool, so a subprocess per pair would reload both
models from scratch for every single pair). Shards the full sorted actor id
list into one contiguous chunk per GPU, each chunk handed to its own
process_expression_sapiens.py subprocess (--actor-min/--actor-max), pinned to
one GPU via CUDA_VISIBLE_DEVICES.

Unlike Nersemble's actor ids ("017", zero-padded, compared as strings),
FaceScape actor ids are plain unpadded integers ("1".."20") --
process_expression_sapiens.py's own _iter_targets compares --actor-min/
--actor-max as ints, so this wrapper sorts/shards by int() too.

Default mode (no --target-only): polls forever -- after every chunk
subprocess finishes a full sweep of its shard, re-globs facescape_images_root
(picks up newly-added actors) and re-shards. Every subprocess call also gets
--skip-completed-pair by default (see --no-skip-completed) -- label_tracker.json
status can't gate re-processing here, since process_expression_sapiens.py
writes "unlabeled" itself once a pair completes but doesn't downgrade a pair
that's already further along, so a completed pair stays eligible-looking
under naive re-scans; --skip-completed-pair's own completed_expressions_sapiens.json
record is what actually prevents re-running it every round.

Usage:
    python3 run_process_expression_sapiens_batch.py --gpu-ids 0 1 2 3 --parallel 3

    python3 run_process_expression_sapiens_batch.py --target-only --gpu-ids 0 1
        # runs TARGET_LIST actors once, round-robined across 2 GPUs, then exits

    python3 run_process_expression_sapiens_batch.py --dry-run --gpu-ids 0 1 2 3
        # prints the planned subprocess commands without launching anything
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
import facescape_paths as fp  # all paths: wrap_script_facescape/.env

FACEVIEW_ROOT = fp.FACEVIEW_ROOT
SCRIPT_PATH = FACEVIEW_ROOT / "scripts" / "preprocess" / "process_expression_sapiens.py"
VENV_PYTHON = fp.SAPIENS_PYTHON
DATA_ROOT = fp.FACESCAPE_IMAGES_ROOT

POLL_INTERVAL_SECONDS = 60

# Actors to restrict to when --target-only is given -- same name/role as
# wrap_script/run_process_sequence_sapiens_batch.py's own TARGET_LIST. Empty
# by default.
TARGET_LIST: list[str] = [
]

# Expressions to skip entirely, even if present -- mirrors
# script_process_expression_sapiens.sh's own EXCLUDE_EXPRESSIONS blacklist.
EXCLUDE_EXPRESSIONS: list[str] = [
]


def discover_actor_ids(data_root: Path) -> list[str]:
    if not data_root.is_dir():
        return []
    return sorted((p.name for p in data_root.iterdir() if p.is_dir() and p.name.isdigit()), key=int)


def shard_contiguous(items: list[str], n_shards: int) -> list[list[str]]:
    """Split items (already sorted by int()) into up to n_shards contiguous,
    roughly-even chunks. Fewer (non-empty) chunks than n_shards are returned
    if there aren't enough items to give every shard at least one."""
    if not items or n_shards <= 1:
        return [items] if items else []
    n_shards = min(n_shards, len(items))
    base, remainder = divmod(len(items), n_shards)
    shards = []
    start = 0
    for i in range(n_shards):
        size = base + (1 if i < remainder else 0)
        shards.append(items[start:start + size])
        start += size
    return [s for s in shards if s]


@dataclass
class ShardResult:
    label: str
    returncode: int


def _run_subprocess(cmd: list[str], gpu_id: str | None, log_dir: Path | None, log_name: str, dry_run: bool) -> int:
    print("RUN  " + " ".join(cmd) + (f"  gpu={gpu_id}" if gpu_id is not None else ""))
    if dry_run:
        return 0

    job_env = os.environ.copy()
    if gpu_id is not None:
        job_env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    if log_dir is None:
        return subprocess.run(cmd, cwd=FACEVIEW_ROOT, env=job_env).returncode
    log_path = log_dir / log_name
    with open(log_path, "w") as log_file:
        return subprocess.run(cmd, cwd=FACEVIEW_ROOT, env=job_env, stdout=log_file, stderr=subprocess.STDOUT).returncode


def run_shard(actor_ids: list[str], parallel: int, gpu_id: str | None, log_dir: Path | None, dry_run: bool, skip_completed: bool) -> ShardResult:
    actor_min, actor_max = actor_ids[0], actor_ids[-1]
    label = f"actors[{actor_min}..{actor_max}]" + (f" gpu={gpu_id}" if gpu_id is not None else "")
    cmd = [
        str(VENV_PYTHON), str(SCRIPT_PATH),
        "--actor-min", actor_min, "--actor-max", actor_max,
        "--exclude-expressions", *EXCLUDE_EXPRESSIONS,
        "--parallel", str(parallel),
    ]
    if skip_completed:
        cmd.append("--skip-completed-pair")
    returncode = _run_subprocess(cmd, gpu_id, log_dir, f"shard_{actor_min}_{actor_max}.log", dry_run)
    status = "PASS" if returncode == 0 else "FAIL"
    print(f"{status} {label} exit_code={returncode}")
    return ShardResult(label, returncode)


def run_actor(actor_id: str, gpu_id: str | None, log_dir: Path | None, dry_run: bool, skip_completed: bool) -> ShardResult:
    label = f"actor={actor_id}" + (f" gpu={gpu_id}" if gpu_id is not None else "")
    cmd = [
        str(VENV_PYTHON), str(SCRIPT_PATH), actor_id,
        "--exclude-expressions", *EXCLUDE_EXPRESSIONS,
    ]
    if skip_completed:
        cmd.append("--skip-completed-pair")
    returncode = _run_subprocess(cmd, gpu_id, log_dir, f"actor_{actor_id}.log", dry_run)
    status = "PASS" if returncode == 0 else "FAIL"
    print(f"{status} {label} exit_code={returncode}")
    return ShardResult(label, returncode)


def dispatch_shards(shards: list[list[str]], gpu_ids: list[str] | None, parallel: int, log_dir: Path | None, dry_run: bool, skip_completed: bool) -> tuple[int, int]:
    passed = failed = 0
    workers = len(gpu_ids) if gpu_ids else 1
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = [
            executor.submit(run_shard, shard, parallel, gpu_ids[i % len(gpu_ids)] if gpu_ids else None, log_dir, dry_run, skip_completed)
            for i, shard in enumerate(shards)
        ]
        for future in as_completed(futures):
            result = future.result()
            if result.returncode == 0:
                passed += 1
            else:
                failed += 1
    return passed, failed


def dispatch_actors(actor_ids: list[str], gpu_ids: list[str] | None, log_dir: Path | None, dry_run: bool, skip_completed: bool) -> tuple[int, int]:
    passed = failed = 0
    workers = len(gpu_ids) if gpu_ids else 1
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = [
            executor.submit(run_actor, actor_id, gpu_ids[i % len(gpu_ids)] if gpu_ids else None, log_dir, dry_run, skip_completed)
            for i, actor_id in enumerate(actor_ids)
        ]
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
        "--target-only", action="store_true",
        help="Run only TARGET_LIST actors, one subprocess per actor, round-robined across --gpu-ids. Runs once and exits (no polling).",
    )
    parser.add_argument(
        "--gpu-ids", nargs="+", default=None,
        help="GPU indices to spread shards/actors across (e.g. --gpu-ids 0 1 2 3), via CUDA_VISIBLE_DEVICES per subprocess. Without this, runs as a single unpinned shard/batch.",
    )
    parser.add_argument(
        "--parallel", type=int, default=3,
        help="Forwarded as each shard subprocess's own --parallel (its internal thread pool sharing one model). Only used in default (non --target-only) mode.",
    )
    parser.add_argument(
        "--log-dir", default=None,
        help="Directory to write each shard/actor's stdout/stderr to instead of the shared terminal.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print planned subprocess commands without launching anything.",
    )
    parser.add_argument(
        "--no-skip-completed", action="store_true",
        help=(
            "By default every subprocess call gets --skip-completed-pair, so repeated "
            "polling rounds don't re-run sapiens2 detection on pairs already recorded in "
            "completed_expressions_sapiens.json. Pass --no-skip-completed to force a full "
            "re-detection anyway (e.g. after adding a new sapiens2 landmark mapping)."
        ),
    )
    args = parser.parse_args()

    log_dir = Path(args.log_dir) if args.log_dir else None
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)

    skip_completed = not args.no_skip_completed

    if args.target_only:
        if not TARGET_LIST:
            print("--target-only given but TARGET_LIST is empty -- edit this script and add some actor ids first.")
            return 1
        passed, failed = dispatch_actors(TARGET_LIST, args.gpu_ids, log_dir, args.dry_run, skip_completed)
        print(f"Done. actors={len(TARGET_LIST)} passed={passed} failed={failed}")
        return 0 if failed == 0 else 2

    round_num = 0
    while True:
        round_num += 1
        actor_ids = discover_actor_ids(DATA_ROOT)
        if not actor_ids:
            print(f"[round {round_num}] No actors found under {DATA_ROOT}")
        else:
            shards = shard_contiguous(actor_ids, len(args.gpu_ids) if args.gpu_ids else 1)
            print(f"[round {round_num}] {len(actor_ids)} actor(s) under {DATA_ROOT}, split into {len(shards)} shard(s)")
            passed, failed = dispatch_shards(shards, args.gpu_ids, args.parallel, log_dir, args.dry_run, skip_completed)
            print(f"[round {round_num}] Done. shards={len(shards)} passed={passed} failed={failed}")

        if args.dry_run:
            return 0

        print(f"[round {round_num}] Sleeping {POLL_INTERVAL_SECONDS}s before rescanning {DATA_ROOT}...")
        time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    raise SystemExit(main())
