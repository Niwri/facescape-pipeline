#!/usr/bin/env python3
"""Aligns every FaceScape actor/expression's params.json + scanned mesh into
the origin space defined by run_origin_transform.py (1_1_neutral, transformed
via 1_1_neutral_to_origin.json), and reindexes every camera by serial number
to match 1_1_neutral's own canonical numbering (see params_io.py's docstring
for why that reindex is necessary).

Run run_origin_transform.py first -- this script requires
irwinngo1/.../1/1_neutral/params.json to already exist as the alignment target.

ProcessPoolExecutor (not threads): this is pure-CPU numpy/SVD work across many
independent small jobs, not GPU-bound model sharing like process_sequence_sapiens.py,
so separate processes parallelize better here than threads would.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from params_io import build_canonical_index, remap_and_transform
from solve_alignment import MIN_SHARED_CAMERAS, align_params_to_target
from transform_mesh import transform_mesh_file
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # wrap_script_facescape/, for facescape_paths
import facescape_paths as fp  # all paths: wrap_script_facescape/.env

SOURCE_IMAGES_ROOT = fp.FACESCAPE_RAW_IMAGES_ROOT
SOURCE_SHAPE_ROOT = fp.FACESCAPE_RAW_SHAPE_ROOT
DEST_IMAGES_ROOT = fp.FACESCAPE_IMAGES_ROOT
DEST_SHAPE_ROOT = fp.FACESCAPE_SHAPE_ROOT

ORIGIN_ACTOR = "1"
ORIGIN_EXPRESSION = "1_neutral"
TARGET_PARAMS_PATH = DEST_IMAGES_ROOT / ORIGIN_ACTOR / ORIGIN_EXPRESSION / "params.json"

ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = ROOT / "manifest.jsonl"
FAILURES_PATH = ROOT / "failures.jsonl"


def discover_pairs(actor_min: str | None, actor_max: str | None, exclude_expressions: set[str]) -> list[tuple[str, str]]:
    pairs = []
    if not SOURCE_IMAGES_ROOT.is_dir():
        return pairs
    # Integer comparison, not string -- FaceScape actor directories are plain
    # unpadded integers ("1", "2", ..., "15", ...), not zero-padded like
    # Nersemble's ("018"), so a naive string comparison puts "2" after "15"
    # lexicographically and silently pulls in actors outside the intended range.
    actor_min_n = int(actor_min) if actor_min is not None else None
    actor_max_n = int(actor_max) if actor_max is not None else None
    actor_dirs = sorted(
        (p for p in SOURCE_IMAGES_ROOT.iterdir() if p.is_dir() and p.name.isdigit()),
        key=lambda p: int(p.name),
    )
    for actor_dir in actor_dirs:
        actor_id = actor_dir.name
        if actor_min_n is not None and int(actor_id) < actor_min_n:
            continue
        if actor_max_n is not None and int(actor_id) > actor_max_n:
            continue
        for expr_dir in sorted(actor_dir.iterdir()):
            if not expr_dir.is_dir():
                continue
            expr_id = expr_dir.name
            if expr_id in exclude_expressions:
                continue
            if not (expr_dir / "params.json").exists():
                continue
            if actor_id == ORIGIN_ACTOR and expr_id == ORIGIN_EXPRESSION:
                continue  # already the origin target itself
            pairs.append((actor_id, expr_id))
    return pairs


def process_pair(
    actor_id: str,
    expression_id: str,
    pos_threshold: float,
    rot_threshold: float,
    min_shared_cameras: int,
    force: bool,
) -> dict:
    """Runs inside a worker process -- reads everything fresh from disk, no
    shared state with the parent or other workers."""
    # NOT a check on params.json existing: Phase 0's blind rsync copy already
    # places an untransformed, non-canonical-indexed params.json at this exact
    # path before Phase 1 ever runs, for every pair not yet reached. Checking
    # only file existence would silently skip those, permanently leaving them
    # as raw, wrongly-indexed copies (this bit us on the first full run: 98
    # pairs, including actor 1's own non-neutral expressions, were skipped this
    # way). ".transform_complete" is only ever written at the end of a
    # successful run of this function, so it's the one reliable "already done"
    # signal.
    dest_dir = DEST_IMAGES_ROOT / actor_id / expression_id
    marker_path = dest_dir / ".transform_complete"
    if not force and marker_path.exists():
        return dict(actor=actor_id, expression=expression_id, status="skipped_existing")

    src_params_path = SOURCE_IMAGES_ROOT / actor_id / expression_id / "params.json"
    src_mesh_path = SOURCE_SHAPE_ROOT / actor_id / f"{expression_id}.ply"
    if not src_mesh_path.exists():
        return dict(actor=actor_id, expression=expression_id, status="failed",
                     reason=f"missing mesh: {src_mesh_path}")

    try:
        solved = align_params_to_target(str(src_params_path), str(TARGET_PARAMS_PATH))
    except ValueError as exc:
        return dict(actor=actor_id, expression=expression_id, status="failed", reason=str(exc))

    metrics = solved["metrics"]
    if solved["n_used"] < min_shared_cameras:
        return dict(actor=actor_id, expression=expression_id, status="failed",
                     reason=f"only {solved['n_used']} shared valid cameras", metrics=metrics)
    if metrics["pos_rmse"] > pos_threshold or metrics["rot_rmse"] > rot_threshold:
        return dict(actor=actor_id, expression=expression_id, status="failed",
                     reason=(f"residual too high: pos_rmse={metrics['pos_rmse']:.4f} "
                             f"rot_rmse={metrics['rot_rmse']:.4f}"),
                     metrics=metrics)

    canonical_index = build_canonical_index(str(TARGET_PARAMS_PATH))
    out_params, index_map, dropped = remap_and_transform(
        solved["dict_A"], solved["cams_A"], solved["s"], solved["Rg"], solved["tg"], solved["F"],
        canonical_index,
    )

    dest_dir.mkdir(parents=True, exist_ok=True)
    with open(dest_dir / "params.json", "w") as f:
        json.dump(out_params, f)

    # Wipe whatever Phase 0's blind copy placed here (original-index-named
    # JPGs), then write only the canonically-renamed, verified set fresh --
    # avoids any stale/incorrectly-numbered leftovers.
    for jpg in dest_dir.glob("*.jpg"):
        jpg.unlink()
    src_img_dir = SOURCE_IMAGES_ROOT / actor_id / expression_id
    for orig_i, canon_i in index_map.items():
        src_jpg = src_img_dir / f"{orig_i}.jpg"
        if src_jpg.exists():
            shutil.copy(src_jpg, dest_dir / f"{canon_i}.jpg")

    dest_mesh_path = DEST_SHAPE_ROOT / actor_id / f"{expression_id}.ply"
    transform_mesh_file(src_mesh_path, dest_mesh_path, solved["s"], solved["Rg"], solved["tg"])

    # Written last, only on full success -- this (not params.json existing) is
    # what the skip-check above looks for.
    marker_path.write_text(str(time.time()))

    return dict(
        actor=actor_id, expression=expression_id, status="passed",
        n_shared=solved["n_shared"], n_used=solved["n_used"],
        n_dropped_cameras=len(dropped),
        pos_rmse=metrics["pos_rmse"], rot_rmse=metrics["rot_rmse"], scale=solved["s"],
        timestamp=time.time(),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--actor-min", default=None, help="Lowest actor id to include, inclusive")
    parser.add_argument("--actor-max", default=None, help="Highest actor id to include, inclusive")
    parser.add_argument("--exclude-expressions", nargs="*", default=[], help="Expression ids to skip (blacklist)")
    parser.add_argument("--parallel", type=int, default=4, help="Worker processes (default: 4)")
    parser.add_argument("--dry-run", action="store_true", help="List pairs that would be processed, don't run")
    parser.add_argument("--force", action="store_true", help="Reprocess pairs that already have an output params.json")
    # Defaults set from real measured residuals across actors 2/3/5/7/10 (not a
    # guess): pos_rmse ranged ~0.03-0.074, rot_rmse ~1.9-2.3 deg -- consistent,
    # smoothly-distributed calibration noise across the rig (checked per-camera:
    # no single dominant outlier camera, see coordinate_transforms notes), not a
    # bug in the solver. Thresholds below give margin above that normal range
    # while still catching a genuinely broken/mismatched alignment.
    parser.add_argument("--rmse-threshold-pos", type=float, default=0.10,
                         help="Max acceptable position RMSE (origin-space world units) before flagging failure (default: 0.10)")
    parser.add_argument("--rmse-threshold-rot", type=float, default=3.0,
                         help="Max acceptable rotation RMSE (degrees) before flagging failure (default: 3.0)")
    parser.add_argument("--min-shared-cameras", type=int, default=MIN_SHARED_CAMERAS,
                         help=f"Minimum shared valid cameras required to solve an alignment (default: {MIN_SHARED_CAMERAS})")
    args = parser.parse_args()

    if not TARGET_PARAMS_PATH.exists():
        raise SystemExit(f"Origin target not found: {TARGET_PARAMS_PATH} -- run run_origin_transform.py first")

    exclude = set(args.exclude_expressions)
    pairs = discover_pairs(args.actor_min, args.actor_max, exclude)
    if not pairs:
        print("No actor/expression pairs found matching filters.")
        return 0

    print(f"Processing {len(pairs)} actor/expression pair(s)" + (" [DRY RUN]" if args.dry_run else ""))
    if args.dry_run:
        for actor_id, expr_id in pairs:
            print(f"  {actor_id}/{expr_id}")
        return 0

    passed = 0
    failed = 0
    skipped = 0
    with open(MANIFEST_PATH, "a") as manifest_f, open(FAILURES_PATH, "a") as failures_f:
        with ProcessPoolExecutor(max_workers=max(1, args.parallel)) as executor:
            futures = {
                executor.submit(
                    process_pair, actor_id, expr_id,
                    args.rmse_threshold_pos, args.rmse_threshold_rot,
                    args.min_shared_cameras, args.force,
                ): (actor_id, expr_id)
                for actor_id, expr_id in pairs
            }
            for future in as_completed(futures):
                actor_id, expr_id = futures[future]
                try:
                    result = future.result()
                except Exception as exc:  # noqa: BLE001
                    result = dict(actor=actor_id, expression=expr_id, status="failed", reason=f"exception: {exc}")

                if result["status"] == "passed":
                    passed += 1
                    manifest_f.write(json.dumps(result) + "\n")
                    manifest_f.flush()
                    print(f"DONE  [actor={actor_id} expression={expr_id}]: "
                          f"n_shared={result['n_shared']} n_used={result['n_used']} "
                          f"dropped={result['n_dropped_cameras']} pos_rmse={result['pos_rmse']:.4f} "
                          f"rot_rmse={result['rot_rmse']:.4f}")
                elif result["status"] == "skipped_existing":
                    skipped += 1
                    print(f"SKIP  [actor={actor_id} expression={expr_id}]: output already exists")
                else:
                    failed += 1
                    failures_f.write(json.dumps(result) + "\n")
                    failures_f.flush()
                    print(f"FAIL  [actor={actor_id} expression={expr_id}]: {result.get('reason')}")

    print(f"Done. pairs={len(pairs)} passed={passed} failed={failed} skipped_existing={skipped}")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
