#!/usr/bin/env python3
"""Sapiens2 landmark extraction for FaceScape's INVALID pairs
(settings.facescape_invalid_pairs_path) -- the pairs process_expression_sapiens.py
skips because their cameras could not be aligned to the canonical origin rig.

Same model, inference and output as process_expression_sapiens.py (its
_process_pair() is reused as-is), with three differences:
  - only invalid pairs are processed,
  - images come from the pair's RAW directory (settings.facescape_raw_images_root),
  - the camera list is every valid camera in that pair's own raw params.json
    (raw camera indices -- the canonical facescape_cameras.json list doesn't
    apply to these pairs).
Output: landmarks_facescape/<actor>/<expression>/cam_<raw index>/frame_000000.json,
the same file format; done-list: completed_expressions_sapiens_invalid.json.

Usage:
    python process_expression_sapiens_invalid.py [actor_id [expression_id]] [--actor-min A] [--actor-max B]
        [--exclude-expressions E ...] [--parallel N] [--skip-completed-pair]
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import process_expression_sapiens as pes  # noqa: E402  (valid-pair version: model, inference, writers)

DONE_JSON = pes.ROOT / "completed_expressions_sapiens_invalid.json"


def load_invalid_pairs(path: Path) -> list[tuple[str, str]]:
    with open(path) as f:
        data = json.load(f)
    return sorted({(str(p["actor"]), str(p["expression"])) for p in data.get("pairs", [])},
                  key=lambda t: (int(t[0]), t[1]))


def pair_camera_ids(expr_dir: Path) -> list[str]:
    """Every camera in the pair's own raw params.json that isn't flagged invalid."""
    with open(expr_dir / "params.json") as f:
        params = json.load(f)
    ids = {k.split("_")[0] for k in params if k.endswith("_Rt")}
    return sorted((i for i in ids if params.get(f"{i}_valid", True) is not False), key=int)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("actor_id", nargs="?", default=None, help="Restrict to one actor")
    parser.add_argument("expression_id", nargs="?", default=None, help="Restrict to one expression (requires actor_id)")
    parser.add_argument("--actor-min", default=None, help="Lowest actor id to include, inclusive (integer compare)")
    parser.add_argument("--actor-max", default=None, help="Highest actor id to include, inclusive (integer compare)")
    parser.add_argument("--exclude-expressions", nargs="*", default=[], help="Expression ids to skip")
    parser.add_argument("--skip-completed", action="store_true", help="Skip camera views that already have a frame_000000.json")
    parser.add_argument("--skip-completed-pair", action="store_true", help=f"Skip pairs already recorded in {DONE_JSON.name}")
    parser.add_argument("--parallel", type=int, default=1, help="Pairs processed concurrently, sharing one warmed-up model")
    parser.add_argument("--sanity-render", action="store_true", help="Also write point renders to " + str(pes.SANITY_RENDER_ROOT))
    args = parser.parse_args()

    settings = pes.get_settings()
    targets = load_invalid_pairs(settings.facescape_invalid_pairs_path)
    if args.actor_id:
        targets = [t for t in targets if t[0] == args.actor_id and (args.expression_id is None or t[1] == args.expression_id)]
    if args.actor_min is not None:
        targets = [t for t in targets if int(t[0]) >= int(args.actor_min)]
    if args.actor_max is not None:
        targets = [t for t in targets if int(t[0]) <= int(args.actor_max)]
    targets = [t for t in targets if t[1] not in set(args.exclude_expressions)]
    targets = [t for t in targets if (settings.facescape_raw_images_root / t[0] / t[1] / "params.json").exists()]
    if args.skip_completed_pair:
        completed = pes._load_completed(DONE_JSON)
        targets = [t for t in targets if not pes._is_completed(completed, t[0], t[1])]
    print(f"{len(targets)} invalid pair(s) to process from {settings.facescape_raw_images_root}")
    if not targets:
        return 0

    device = pes._get_device()
    pes._get_detector(device)
    pes._get_pose_model(device)
    sanity_render_root = pes.SANITY_RENDER_ROOT if args.sanity_render else None

    def run_pair(actor_id: str, expression_id: str) -> bool:
        expr_dir = settings.facescape_raw_images_root / actor_id / expression_id
        try:
            ok = pes._process_pair(
                actor_id, expression_id, expr_dir, settings, pair_camera_ids(expr_dir), args.skip_completed,
                sanity_render_root=sanity_render_root,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL actor={actor_id} expression={expression_id}: {exc}")
            return False
        if ok:
            pes._record_completed(DONE_JSON, actor_id, expression_id)
            # Same rule as process_expression_sapiens.py: mark "unlabeled"
            # without downgrading a pair that's already further along.
            status = pes.label_tracker.get_status(actor_id, expression_id, dataset="FaceScape")
            if status in ("unreviewed", "unconfirmed", "confirmed"):
                print(f"  NOTE: not resetting {actor_id}/{expression_id} to 'unlabeled' -- current status is {status!r}")
            else:
                pes.label_tracker.mark_unlabeled(actor_id, expression_id, dataset="FaceScape")
        return ok

    passed = failed = 0
    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as executor:
        futures = [executor.submit(run_pair, a, e) for a, e in targets]
        for future in as_completed(futures):
            with lock:
                passed, failed = (passed + 1, failed) if future.result() else (passed, failed + 1)
    print(f"Done. pairs={len(targets)} passed={passed} failed_or_skipped={failed}")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
