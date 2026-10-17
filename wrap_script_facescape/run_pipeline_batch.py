#!/usr/bin/env python3
"""Runs wrap_script_facescape/run_pipeline.py for every FaceScape actor/
expression pair that has passed Phase 1 (coordinate_transforms), one
subprocess per pair -- FaceScape's counterpart to wrap_script/run_pipeline_batch.py,
simpler in that there's no per-frame loop and no meshroom (run_pipeline.py
already does everything for one pair in a single call: triangulate + wrap +
mark_unconfirmed + overlay render), but with the same neutral-chaining
ordering requirement as Nersemble's own batch script: every actor's
neutral pair (per-actor, read from neutral_frame.json -- see
facescape_neutral.load_actor_neutral_expression) is run to completion before
any of that actor's other expressions, so they can chain off its
wrapped_mesh.obj instead of falling back to the generic template (see
run_pipeline.py's own neutral_mesh_path selection, and main()'s two-pass
split below). Actors with no neutral_frame.json yet are skipped entirely for
this round -- there's no per-actor neutral to chain off of, and guessing
"1_neutral" would silently process the wrong reference expression for an
actor whose reviewer picked something else.

By default, this POLLS FOREVER: each round discovers every actor/expression
pair currently "unreviewed" in label_tracker.json, dispatches that round's
pairs, then sleeps POLL_INTERVAL_SECONDS and rescans -- so newly-"unreviewed"
pairs get picked up without re-invoking this script. Re-scanning a pair still
mid-flight from a previous round is safe: process_pair()'s own per-pair
.pipeline_pair.lock makes it a no-op skip, not duplicated work. --dry-run
remains one-shot (a single preview, not printed forever).

Pairs listed in invalid_pairs.json (settings.facescape_invalid_pairs_path --
whole actor/expression pairs that failed Phase 1's alignment fit, so they have
no origin-space version) are discovered from the RAW root instead
(facescape_raw_images_root/facescape_raw_shape_root) and wrapped with
run_pipeline_invalid.py rather than run_pipeline.py -- same status gate, same
neutral-first ordering, same per-pair lock. An invalid non-neutral pair
additionally waits for its actor's neutral to be wrapped (it has no generic
template fallback -- see run_pipeline_invalid.neutral_template_mesh).

--confirmed-only (one-shot) RE-WRAPS every "confirmed" pair instead, e.g. after
a pipeline/template fix -- same logic as wrap_script/run_pipeline_batch_all.py's
--confirmed-only: each selected actor's neutral pair is re-wrapped first (pass
1, every other expression chains off it), then every selected pair (pass 2),
each through run_pipeline.py / run_pipeline_invalid.py as usual (which also
re-renders its overlay frames; its mark_unconfirmed() is a no-op on a
confirmed pair). A pair that re-wrapped cleanly then has its
final_wrapped_data .pt re-exported with FaceView's
export_wrapped_facescape_pair() -- the same export Confirm uses, which maps the
wrap from Phase 1's origin space back to the pair's original scan space
(invalid pairs are already in raw space) -- after the old .pt is copied to
FINAL_WRAPPED_BACKUP_ROOT (first copy only). Finished pairs are recorded in
REWRAP_CONFIRMED_DONE_JSON so a rerun resumes (--no-resume redoes them).
Labels stay "confirmed".

--unconfirmed-only (one-shot) does the same re-wrap for every "unconfirmed"
pair, without any .pt export (that happens on Confirm). Finished pairs are
recorded in REWRAP_UNCONFIRMED_DONE_JSON. Labels stay "unconfirmed".
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
import facescape_paths as fp  # all paths: wrap_script_facescape/.env

POLL_INTERVAL_SECONDS = 60

SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_SCRIPT = SCRIPT_DIR / "run_pipeline.py"
INVALID_PIPELINE_SCRIPT = SCRIPT_DIR / "run_pipeline_invalid.py"
# run_pipeline_invalid.WRAPPED_TEMPLATE_NAME -- not imported, so this
# orchestrator doesn't pull in the wrap/camera modules just for a filename.
INVALID_WRAPPED_TEMPLATE_NAME = "wrapped_mesh_template.obj"

# --confirmed-only / --unconfirmed-only resume records, and where a confirmed
# pair's pre-rewrap .pt is kept (first copy only, never overwritten) -- same
# backup root as wrap_script/run_pipeline_batch_all.py; FaceScape exports live
# under final_wrapped_data/facescape_<actor>/, so their backups do too.
REWRAP_CONFIRMED_DONE_JSON = SCRIPT_DIR / "completed_rewrap_confirmed.json"
REWRAP_UNCONFIRMED_DONE_JSON = SCRIPT_DIR / "completed_rewrap_unconfirmed.json"
FINAL_WRAPPED_BACKUP_ROOT = fp.FINAL_WRAPPED_BACKUP_ROOT

# Self-contained like run_pipeline.py's own sys.path handling -- so this runs
# with a plain `python run_pipeline_batch.py`, no PYTHONPATH setup needed.
fp.add_faceview_to_sys_path()
from backend.app.core.config import get_settings  # noqa: E402
from backend.app.services import facescape_neutral  # noqa: E402
from backend.app.services.label_tracker import label_tracker  # noqa: E402


def load_invalid_pairs(path: Path) -> set[tuple[str, str]]:
    """Same schema/loader as facescape_dataset_index.py's/process_expression_sapiens.py's
    own copies: {"pairs": [{"actor": ..., "expression": ...}, ...]}."""
    if not path.exists():
        return set()
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return set()
    return {(p["actor"], p["expression"]) for p in data.get("pairs", [])}


def discover_pairs(
    images_root: Path,
    invalid_pairs: set[tuple[str, str]],
    actor_min: str | None,
    actor_max: str | None,
    exclude_expressions: set[str],
) -> list[tuple[str, str]]:
    """Every actor/expression under facescape_images_root (the Phase 1 output
    tree, irwinngo1 -- not the pristine source) that has a params.json and is
    not in invalid_pairs, plus every invalid pair that has a raw params.json
    and raw scan. Integer actor comparison, not string -- FaceScape actor
    directories are plain unpadded integers ("1".."20", not "018"), so a naive
    string compare would misorder them (see coordinate_transforms/run_batch_transform.py's
    same note)."""
    settings = get_settings()
    actor_min_n = int(actor_min) if actor_min is not None else None
    actor_max_n = int(actor_max) if actor_max is not None else None

    def in_range(actor_id: str) -> bool:
        return ((actor_min_n is None or int(actor_id) >= actor_min_n)
                and (actor_max_n is None or int(actor_id) <= actor_max_n))

    pairs = []
    if images_root.is_dir():
        actor_dirs = sorted(
            (p for p in images_root.iterdir() if p.is_dir() and p.name.isdigit()),
            key=lambda p: int(p.name),
        )
        for actor_dir in actor_dirs:
            actor_id = actor_dir.name
            if not in_range(actor_id):
                continue
            for expr_dir in sorted(actor_dir.iterdir()):
                if not expr_dir.is_dir():
                    continue
                expression_id = expr_dir.name
                if expression_id in exclude_expressions:
                    continue
                if not (expr_dir / "params.json").exists():
                    continue
                if (actor_id, expression_id) in invalid_pairs:
                    continue
                pairs.append((actor_id, expression_id))

    for actor_id, expression_id in invalid_pairs:
        if not in_range(actor_id) or expression_id in exclude_expressions:
            continue
        if not (settings.facescape_raw_images_root / actor_id / expression_id / "params.json").exists():
            continue
        if not (settings.facescape_raw_shape_root / actor_id / f"{expression_id}.ply").exists():
            continue
        pairs.append((actor_id, expression_id))
    return sorted(pairs, key=lambda t: (int(t[0]), t[1]))


def neutral_wrap_ready(actor_id: str, neutral_expression_id: str, invalid_pairs: set[tuple[str, str]]) -> bool:
    """Whether an invalid pair's source mesh exists: the actor's neutral wrap in
    the template frame (run_pipeline_invalid.neutral_template_mesh)."""
    name = INVALID_WRAPPED_TEMPLATE_NAME if (actor_id, neutral_expression_id) in invalid_pairs else "wrapped_mesh.obj"
    return (get_settings().facescape_output_root / actor_id / neutral_expression_id / name).exists()


@dataclass
class PairResult:
    actor_id: str
    expression_id: str
    passed: bool = False
    skipped: bool = False


def process_pair(
    actor_id: str,
    expression_id: str,
    log_dir: Path | None,
    gpu_id: str | None = None,
    invalid: bool = False,
) -> PairResult:
    """Runs run_pipeline.py (run_pipeline_invalid.py when ``invalid``) as a subprocess -- safe to call concurrently for
    different pairs, no shared mutable state between them (unlike Nersemble's
    run_pipeline_batch_all.py, there's no warmed-up SAM/U2Net/FLAME model to
    share in-process here, so a subprocess per pair, same as Nersemble's own
    run_pipeline_batch.py, costs nothing extra).

    ``gpu_id``, when given, restricts this pair's whole subprocess (via
    CUDA_VISIBLE_DEVICES) to that single GPU -- same round-robin --gpu-ids
    design as wrap_script/run_pipeline_batch.py.

    Takes an exclusive, non-blocking per-pair flock
    (facescape_output_root/actor_id/expression_id/.pipeline_pair.lock) before
    doing any work, same pattern as wrap_script/run_pipeline_batch_all.py's
    .pipeline_pair.lock -- lets multiple cluster nodes run this script (or
    its polling default mode, see main()) against the same label_tracker.json
    without two of them ever processing the same pair at once. Skips (rather
    than blocks) on contention."""
    tag = (f"[actor={actor_id} expression={expression_id}]" + (" invalid" if invalid else "")
           + (f" gpu={gpu_id}" if gpu_id is not None else ""))

    lock_dir = get_settings().facescape_output_root / actor_id / expression_id
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_file = open(lock_dir / ".pipeline_pair.lock", "w")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print(f"SKIP {tag}: pair is locked by another run already processing it")
        lock_file.close()
        return PairResult(actor_id, expression_id, passed=True, skipped=True)

    try:
        script = INVALID_PIPELINE_SCRIPT if invalid else PIPELINE_SCRIPT
        cmd = [sys.executable, str(script), actor_id, expression_id]
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

        if returncode != 0:
            print(f"FAIL {tag}: exit_code={returncode}")
            return PairResult(actor_id, expression_id, passed=False)
        return PairResult(actor_id, expression_id, passed=True)
    finally:
        fcntl.flock(lock_file, fcntl.LOCK_UN)
        lock_file.close()


def load_completed(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def record_completed(path: Path, actor_id: str, expression_id: str) -> None:
    """Add actor/expression to the resume record -- lock-guarded, atomic
    read-modify-write (several cluster nodes may run this script at once),
    same as wrap_script/run_pipeline_batch_all.record_completed()."""
    lock_path = path.with_name(path.name + ".lock")
    with lock_path.open("w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            completed = load_completed(path)
            completed[actor_id] = sorted(set(completed.get(actor_id, [])) | {expression_id})
            tmp = path.with_name(path.name + ".tmp")
            tmp.write_text(json.dumps(completed, indent=2), encoding="utf-8")
            os.replace(tmp, path)
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def finish_rewrapped_pair(actor_id: str, expression_id: str, target_status: str) -> bool:
    """After a pair re-wrapped cleanly. "confirmed": back up the old .pt (first
    time only), re-export it with export_wrapped_facescape_pair() -- origin
    space back to the pair's original scan space, as Confirm does -- and
    record it; "unconfirmed": just record it (its overlay frames were already
    re-rendered by run_pipeline.py). Returns whether the pair is done."""
    import shutil

    tag = f"[actor={actor_id} expression={expression_id}]"
    if target_status == "confirmed":
        from backend.app.services.wrapped_mesh_export import export_wrapped_facescape_pair

        pt_path = get_settings().final_wrapped_data_root / f"facescape_{actor_id}" / f"{expression_id}.pt"
        backup = FINAL_WRAPPED_BACKUP_ROOT / f"facescape_{actor_id}" / f"{expression_id}.pt"
        if pt_path.exists() and not backup.exists():
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(pt_path, backup)
            print(f"BACKUP {tag}: {pt_path} -> {backup}")
        try:
            exported = export_wrapped_facescape_pair(actor_id, expression_id)
        except Exception as exc:  # a failed export must not stop the other pairs
            print(f"FAIL {tag}: .pt re-export raised {exc!r}")
            exported = None
        if exported is None:
            print(f"NOT EXPORTED {tag}: .pt re-export failed -- the old .pt is left in place, not recorded")
            return False
        record_completed(REWRAP_CONFIRMED_DONE_JSON, actor_id, expression_id)
        print(f"DONE {tag}: re-wrapped, .pt re-exported to {exported}")
        return True
    record_completed(REWRAP_UNCONFIRMED_DONE_JSON, actor_id, expression_id)
    print(f"DONE {tag}: re-wrapped (still 'unconfirmed')")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--actor-min", default=None, help="Lowest actor id to include, inclusive")
    parser.add_argument("--actor-max", default=None, help="Highest actor id to include, inclusive")
    parser.add_argument("--exclude-expressions", nargs="*", default=[], help="Expression ids to skip (blacklist)")
    parser.add_argument("--parallel", type=int, default=1, help="Number of pairs to wrap concurrently (default: 1)")
    parser.add_argument("--dry-run", action="store_true", help="Print pairs that would run, don't execute")
    parser.add_argument(
        "--log-dir", default=None,
        help=(
            "Directory to write each pair's stdout/stderr to (<actor>_<expression>.log) "
            "instead of the shared terminal. Recommended whenever --parallel > 1."
        ),
    )
    parser.add_argument(
        "--gpu-ids", nargs="+", default=None,
        help=(
            "GPU indices to spread work across (e.g. --gpu-ids 0 1 2 3), as visible to "
            "THIS process -- i.e. matching `nvidia-smi -L` / $CUDA_VISIBLE_DEVICES inside "
            "the current allocation, not necessarily the machine's physical GPU numbering "
            "(SLURM's --gres=gpu:N already restricts/remaps what's visible to a job). "
            "Pairs are assigned round-robin, one GPU per pair for that pair's whole run "
            "(via CUDA_VISIBLE_DEVICES on each subprocess) -- so with --parallel higher "
            "than len(--gpu-ids), multiple concurrent pairs share each GPU, same as today's "
            "single-GPU contention but split across however many GPUs you give it. Without "
            "this flag, every subprocess defaults to cuda:0 (today's behavior) -- with "
            "--parallel > 1 and more than one real GPU visible, that means every worker "
            "piles onto just the first GPU while the rest sit idle."
        ),
    )
    parser.add_argument(
        "--confirmed-only", action="store_true",
        help=("One-shot: RE-WRAP every 'confirmed' pair (actors' neutral pairs first) and re-export its "
              "final_wrapped_data .pt in the original scan space (old .pt backed up to "
              f"{FINAL_WRAPPED_BACKUP_ROOT}). Resumable via {REWRAP_CONFIRMED_DONE_JSON.name}. Labels stay 'confirmed'."),
    )
    parser.add_argument(
        "--unconfirmed-only", action="store_true",
        help=("One-shot: RE-WRAP every 'unconfirmed' pair (actors' neutral pairs first), no .pt export. "
              f"Resumable via {REWRAP_UNCONFIRMED_DONE_JSON.name}. Labels stay 'unconfirmed'."),
    )
    parser.add_argument("--pairs", nargs="+", default=None,
                        help="With --confirmed-only/--unconfirmed-only: only these actor:expression pairs (e.g. 101:1_neutral)")
    parser.add_argument("--no-resume", action="store_true",
                        help="With --confirmed-only/--unconfirmed-only: redo pairs already in the resume record")
    args = parser.parse_args()
    if args.confirmed_only and args.unconfirmed_only:
        parser.error("--confirmed-only and --unconfirmed-only are mutually exclusive")
    if (args.pairs or args.no_resume) and not (args.confirmed_only or args.unconfirmed_only):
        parser.error("--pairs / --no-resume only apply with --confirmed-only or --unconfirmed-only")

    settings = get_settings()
    log_dir = Path(args.log_dir) if args.log_dir else None
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)

    invalid_pairs = load_invalid_pairs(settings.facescape_invalid_pairs_path)
    print(f"Loaded {len(invalid_pairs)} invalid pair(s) from {settings.facescape_invalid_pairs_path} "
          "(wrapped from the raw root with run_pipeline_invalid.py)")

    exclude_expressions = set(args.exclude_expressions)

    def discover_and_gate() -> list[tuple[str, str]]:
        pairs = discover_pairs(
            settings.facescape_images_root, invalid_pairs,
            args.actor_min, args.actor_max, exclude_expressions,
        )
        # Always gated to label_tracker.json status "unreviewed" -- same automatic
        # gate as Nersemble's run_pipeline_batch_all.py, so this never re-wraps a
        # pair a human (or Phase 4) has already moved on to
        # "unconfirmed"/"confirmed", nor one that hasn't had its skin landmarks
        # propagated yet (still "unlabeled").
        all_statuses = label_tracker.get_all(dataset="FaceScape")
        before = len(pairs)
        pairs = [
            (actor_id, expression_id)
            for actor_id, expression_id in pairs
            if label_tracker.get_status(actor_id, expression_id, dataset="FaceScape", all_statuses=all_statuses) == "unreviewed"
        ]
        print(f"Gated to label_tracker.json status 'unreviewed': {before - len(pairs)} pair(s) excluded, "
              f"{len(pairs)} remaining")
        return pairs

    def run_rewrap(target_status: str) -> int:
        """--confirmed-only / --unconfirmed-only (see the module docstring)."""
        done_json = REWRAP_CONFIRMED_DONE_JSON if target_status == "confirmed" else REWRAP_UNCONFIRMED_DONE_JSON
        pairs = discover_pairs(settings.facescape_images_root, invalid_pairs, args.actor_min, args.actor_max, exclude_expressions)
        all_statuses = label_tracker.get_all(dataset="FaceScape")
        pairs = [(a, e) for a, e in pairs
                 if label_tracker.get_status(a, e, dataset="FaceScape", all_statuses=all_statuses) == target_status]
        print(f"--{target_status}-only: {len(pairs)} {target_status} pair(s)")
        if args.pairs:
            wanted = {tuple(p.split(":", 1)) for p in args.pairs}
            missing = sorted(wanted - set(pairs))
            if missing:
                print(f"--pairs: not {target_status} (skipped): {missing}")
            pairs = [p for p in pairs if p in wanted]
        if not args.no_resume:
            done = load_completed(done_json)
            before = len(pairs)
            pairs = [(a, e) for a, e in pairs if e not in done.get(a, [])]
            print(f"resume: {before - len(pairs)} pair(s) already re-wrapped ({done_json.name}), {len(pairs)} to do")
        neutral_by_actor = {a: facescape_neutral.load_actor_neutral_expression(a) for a in {a for a, _ in pairs}}
        no_neutral = sorted(a for a, n in neutral_by_actor.items() if n is None)
        if no_neutral:
            print(f"SKIP {len(no_neutral)} actor(s) with no saved neutral_frame.json: {', '.join(no_neutral)}")
            pairs = [(a, e) for a, e in pairs if neutral_by_actor[a] is not None]
        if args.dry_run:
            for a, e in pairs:
                print(f"  {a}/{e}" + ("  [neutral]" if e == neutral_by_actor[a] else "")
                      + ("  [invalid]" if (a, e) in invalid_pairs else ""))
            return 0
        if not pairs:
            print("Nothing to re-wrap.")
            return 0

        selected = set(pairs)
        finished, failed = [], []

        def finish(results: list[PairResult]) -> None:
            for r in results:
                key = (r.actor_id, r.expression_id)
                if key not in selected or r.skipped:
                    continue
                if r.passed and finish_rewrapped_pair(r.actor_id, r.expression_id, target_status):
                    finished.append(key)
                else:
                    failed.append(key)

        # Pass 1: every selected actor's neutral pair (whatever its own status),
        # since every other expression of the actor chains off its wrap.
        neutral_pairs = sorted({(a, neutral_by_actor[a]) for a, _ in pairs}, key=lambda t: (int(t[0]), t[1]))
        print(f"Pass 1/2: re-wrapping the neutral pair of {len(neutral_pairs)} actor(s)")
        results = run_pairs_results(neutral_pairs)
        finish(results)
        bad_actors = {r.actor_id for r in results if not r.passed or r.skipped}
        if bad_actors:
            print(f"Neutral re-wrap failed or locked for actor(s) {sorted(bad_actors)} -- their other pairs are skipped this run")

        # Pass 2: the remaining selected pairs.
        other = [(a, e) for a, e in pairs if e != neutral_by_actor[a] and a not in bad_actors]
        print(f"Pass 2/2: re-wrapping {len(other)} pair(s)")
        finish(run_pairs_results(other))

        word = "re-exported" if target_status == "confirmed" else "re-wrapped"
        not_done = sorted(selected - set(finished))
        print(f"Done. {len(finished)}/{len(selected)} pair(s) {word}; failed {len(failed)}")
        if not_done:
            print(f"Not {word} (see FAIL/SKIP/NOT EXPORTED lines above): {not_done}")
        return 0 if not failed and not bad_actors else 2

    if args.dry_run and not (args.confirmed_only or args.unconfirmed_only):
        pairs = discover_and_gate()
        print(f"Processing {len(pairs)} actor/expression pair(s) [DRY RUN]")
        for actor_id, expression_id in pairs:
            print(f"  {actor_id}/{expression_id}" + ("  [invalid]" if (actor_id, expression_id) in invalid_pairs else ""))
        return 0

    if args.gpu_ids:
        print(f"--gpu-ids: round-robin across GPUs {args.gpu_ids}")

    def run_pairs_results(batch: list[tuple[str, str]]) -> list[PairResult]:
        results = []
        with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as executor:
            futures = {
                executor.submit(
                    process_pair, actor_id, expression_id, log_dir,
                    args.gpu_ids[i % len(args.gpu_ids)] if args.gpu_ids else None,
                    (actor_id, expression_id) in invalid_pairs,
                ): (actor_id, expression_id)
                for i, (actor_id, expression_id) in enumerate(batch)
            }
            for future in as_completed(futures):
                results.append(future.result())
        return results

    def run_pairs(batch: list[tuple[str, str]]) -> tuple[int, int]:
        passed = failed = 0
        with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as executor:
            futures = {
                executor.submit(
                    process_pair, actor_id, expression_id, log_dir,
                    args.gpu_ids[i % len(args.gpu_ids)] if args.gpu_ids else None,
                    (actor_id, expression_id) in invalid_pairs,
                ): (actor_id, expression_id)
                for i, (actor_id, expression_id) in enumerate(batch)
            }
            for future in as_completed(futures):
                result = future.result()
                if result.passed:
                    passed += 1
                else:
                    failed += 1
        return passed, failed

    def run_round(pairs: list[tuple[str, str]]) -> tuple[int, int]:
        # run_pipeline.py chains every non-neutral expression off this actor's
        # own neutral wrapped_mesh.obj when it exists, falling back to the
        # generic template otherwise -- so the neutral pair needs to actually
        # finish wrapping before its sibling expressions run, or they'd all
        # silently fall back to the generic template even though a much
        # better per-actor starting point was about to become available. A
        # single flat ThreadPoolExecutor.submit() over every pair can't
        # guarantee that (a non-neutral pair could easily start, or even
        # finish, before its own actor's neutral does) -- so this runs as two
        # full passes instead: every actor's neutral pair first (this pass
        # has no fallback dependency between its own pairs, so it's safe to
        # run at full --parallel), then everything else once every neutral
        # pair that was going to run this invocation already has. This is
        # coarser than "per-actor" ordering (no non-neutral pair from any
        # actor starts until ALL actors' neutral pairs in this round are
        # done, not just its own actor's), but needs no per-actor dependency
        # tracking to get right.
        #
        # Which expression counts as "neutral" is per-actor, read from
        # facescape_landmarks_root/<actor_id>/neutral_frame.json (see
        # facescape_neutral.load_actor_neutral_expression -- the same
        # reviewer-saved file run_neutral_skin_propagation.py's own
        # resolve_neutral_expression() consumes), NOT a hardcoded
        # "1_neutral" -- so an actor whose reviewer picked a different
        # expression as neutral chains its siblings off THAT wrapped mesh
        # instead. An actor with no neutral_frame.json yet has no known
        # neutral to chain off of, so every one of its pairs is SKIPPED this
        # round rather than guessing "1_neutral" -- it'll be picked up once
        # a neutral has been saved for it.
        actor_ids = {actor_id for actor_id, _ in pairs}
        neutral_by_actor = {
            actor_id: facescape_neutral.load_actor_neutral_expression(actor_id)
            for actor_id in actor_ids
        }
        skipped_actors = sorted(a for a, n in neutral_by_actor.items() if n is None)
        if skipped_actors:
            print(f"SKIP {len(skipped_actors)} actor(s) with no saved neutral_frame.json yet "
                  f"(nothing to chain sibling expressions off of): {', '.join(skipped_actors)}")

        neutral_pairs = [(a, e) for a, e in pairs if neutral_by_actor[a] is not None and e == neutral_by_actor[a]]
        other_pairs = [(a, e) for a, e in pairs if neutral_by_actor[a] is not None and e != neutral_by_actor[a]]
        passed = failed = 0
        if neutral_pairs:
            print(f"Pass 1/2: {len(neutral_pairs)} neutral pair(s) (run first so sibling "
                  "expressions can chain off them)")
            p, f = run_pairs(neutral_pairs)
            passed += p
            failed += f
        # An invalid pair has no generic-template fallback: it waits until its
        # actor's neutral is wrapped (checked now, after pass 1 has run).
        waiting = [(a, e) for a, e in other_pairs
                   if (a, e) in invalid_pairs and not neutral_wrap_ready(a, neutral_by_actor[a], invalid_pairs)]
        if waiting:
            print(f"SKIP {len(waiting)} invalid pair(s) whose actor's neutral isn't wrapped yet")
            other_pairs = [p for p in other_pairs if p not in waiting]
        if other_pairs:
            print(f"Pass 2/2: {len(other_pairs)} remaining pair(s)")
            p, f = run_pairs(other_pairs)
            passed += p
            failed += f
        return passed, failed

    if args.confirmed_only or args.unconfirmed_only:
        return run_rewrap("confirmed" if args.confirmed_only else "unconfirmed")

    # Polls forever, discovering newly-"unreviewed" pairs each round --
    # re-scanning a pair still mid-flight from a previous round is safe,
    # process_pair()'s own per-pair .pipeline_pair.lock makes it a no-op skip.
    round_num = 0
    while True:
        round_num += 1
        pairs = discover_and_gate()
        print(f"[round {round_num}] Processing {len(pairs)} actor/expression pair(s)")
        if pairs:
            passed, failed = run_round(pairs)
            print(f"[round {round_num}] Done. pairs={len(pairs)} passed={passed} failed={failed}")
        print(f"[round {round_num}] Sleeping {POLL_INTERVAL_SECONDS}s before rescanning label_tracker.json...")
        time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    raise SystemExit(main())
