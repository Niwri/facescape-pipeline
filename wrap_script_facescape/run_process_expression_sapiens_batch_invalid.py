#!/usr/bin/env python3
"""Multi-GPU wrapper for FaceView/scripts/preprocess/process_expression_sapiens_invalid.py --
run_process_expression_sapiens_batch.py's counterpart for FaceScape's INVALID
pairs (invalid_pairs.json: pairs with no origin-space version, whose images
and params are read from the raw root).

Same design as run_process_expression_sapiens_batch.py: the actors that have
invalid pairs are sharded into one contiguous chunk per GPU, each chunk run
as one process_expression_sapiens_invalid.py subprocess (--actor-min/
--actor-max, its own --parallel thread pool sharing one warmed-up model),
pinned via CUDA_VISIBLE_DEVICES. Actors come from invalid_pairs.json, not a
directory glob.

Default mode polls forever (re-reads invalid_pairs.json each round), with
--skip-completed-pair on every subprocess (see --no-skip-completed), so
pairs recorded in completed_expressions_sapiens_invalid.json aren't redone.

Usage:
    python3 run_process_expression_sapiens_batch_invalid.py --gpu-ids 0 1 2 3 --parallel 3

    python3 run_process_expression_sapiens_batch_invalid.py --target-only --gpu-ids 0 1
        # runs TARGET_LIST actors once, round-robined across 2 GPUs, then exits

    python3 run_process_expression_sapiens_batch_invalid.py --dry-run --gpu-ids 0 1 2 3
        # prints the planned subprocess commands without launching anything
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import run_process_expression_sapiens_batch as base  # sharding + subprocess helpers
import facescape_paths as fp  # all paths: wrap_script_facescape/.env

SCRIPT_PATH = base.FACEVIEW_ROOT / "scripts" / "preprocess" / "process_expression_sapiens_invalid.py"
# settings.facescape_invalid_pairs_path -- read directly so this orchestrator
# doesn't import the FaceView backend.
INVALID_PAIRS_PATH = fp.INVALID_PAIRS_PATH

POLL_INTERVAL_SECONDS = 60

# Actors to restrict to when --target-only is given. Empty by default.
TARGET_LIST: list[str] = [
]

# Expressions to skip entirely, even if present.
EXCLUDE_EXPRESSIONS: list[str] = [
]


def discover_actor_ids() -> list[str]:
    """Every actor with at least one pair in invalid_pairs.json."""
    try:
        data = json.loads(INVALID_PAIRS_PATH.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Could not read {INVALID_PAIRS_PATH}: {exc}")
        return []
    return sorted({str(p["actor"]) for p in data.get("pairs", [])}, key=int)


def _command(extra: list[str], skip_completed: bool) -> list[str]:
    cmd = [str(base.VENV_PYTHON), str(SCRIPT_PATH), *extra]
    if EXCLUDE_EXPRESSIONS:
        cmd += ["--exclude-expressions", *EXCLUDE_EXPRESSIONS]
    if skip_completed:
        cmd.append("--skip-completed-pair")
    return cmd


def run_shard(actor_ids, parallel, gpu_id, log_dir, dry_run, skip_completed) -> base.ShardResult:
    actor_min, actor_max = actor_ids[0], actor_ids[-1]
    label = f"actors[{actor_min}..{actor_max}]" + (f" gpu={gpu_id}" if gpu_id is not None else "")
    cmd = _command(["--actor-min", actor_min, "--actor-max", actor_max, "--parallel", str(parallel)], skip_completed)
    returncode = base._run_subprocess(cmd, gpu_id, log_dir, f"shard_invalid_{actor_min}_{actor_max}.log", dry_run)
    print(f"{'PASS' if returncode == 0 else 'FAIL'} {label} exit_code={returncode}")
    return base.ShardResult(label, returncode)


def run_actor(actor_id, gpu_id, log_dir, dry_run, skip_completed) -> base.ShardResult:
    label = f"actor={actor_id}" + (f" gpu={gpu_id}" if gpu_id is not None else "")
    returncode = base._run_subprocess(_command([actor_id], skip_completed), gpu_id, log_dir,
                                      f"actor_invalid_{actor_id}.log", dry_run)
    print(f"{'PASS' if returncode == 0 else 'FAIL'} {label} exit_code={returncode}")
    return base.ShardResult(label, returncode)


def _dispatch(fn, jobs: list, gpu_ids: list[str] | None) -> tuple[int, int]:
    """Runs fn(job, gpu_id) for every job, one worker per GPU."""
    passed = failed = 0
    with ThreadPoolExecutor(max_workers=max(1, len(gpu_ids) if gpu_ids else 1)) as executor:
        futures = [executor.submit(fn, job, gpu_ids[i % len(gpu_ids)] if gpu_ids else None)
                   for i, job in enumerate(jobs)]
        for future in as_completed(futures):
            if future.result().returncode == 0:
                passed += 1
            else:
                failed += 1
    return passed, failed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target-only", action="store_true",
                        help="Run only TARGET_LIST actors, one subprocess per actor, round-robined across --gpu-ids. Runs once and exits.")
    parser.add_argument("--gpu-ids", nargs="+", default=None,
                        help="GPU indices to spread shards/actors across, via CUDA_VISIBLE_DEVICES per subprocess.")
    parser.add_argument("--parallel", type=int, default=3,
                        help="Forwarded as each shard subprocess's own --parallel. Only used in default (non --target-only) mode.")
    parser.add_argument("--log-dir", default=None, help="Directory for each shard/actor's stdout/stderr.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned subprocess commands without launching anything.")
    parser.add_argument("--no-skip-completed", action="store_true",
                        help="Don't pass --skip-completed-pair (re-run pairs already in completed_expressions_sapiens_invalid.json).")
    args = parser.parse_args()

    log_dir = Path(args.log_dir) if args.log_dir else None
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
    skip_completed = not args.no_skip_completed

    if args.target_only:
        if not TARGET_LIST:
            print("--target-only given but TARGET_LIST is empty -- edit this script and add some actor ids first.")
            return 1
        passed, failed = _dispatch(
            lambda actor_id, gpu_id: run_actor(actor_id, gpu_id, log_dir, args.dry_run, skip_completed),
            TARGET_LIST, args.gpu_ids)
        print(f"Done. actors={len(TARGET_LIST)} passed={passed} failed={failed}")
        return 0 if failed == 0 else 2

    round_num = 0
    while True:
        round_num += 1
        actor_ids = discover_actor_ids()
        if not actor_ids:
            print(f"[round {round_num}] No actors with invalid pairs in {INVALID_PAIRS_PATH}")
        else:
            shards = base.shard_contiguous(actor_ids, len(args.gpu_ids) if args.gpu_ids else 1)
            print(f"[round {round_num}] {len(actor_ids)} actor(s) with invalid pairs, split into {len(shards)} shard(s)")
            passed, failed = _dispatch(
                lambda shard, gpu_id: run_shard(shard, args.parallel, gpu_id, log_dir, args.dry_run, skip_completed),
                shards, args.gpu_ids)
            print(f"[round {round_num}] Done. shards={len(shards)} passed={passed} failed={failed}")

        if args.dry_run:
            return 0

        print(f"[round {round_num}] Sleeping {POLL_INTERVAL_SECONDS}s before re-reading {INVALID_PAIRS_PATH}...")
        time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    raise SystemExit(main())
