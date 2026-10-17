#!/usr/bin/env python3
"""One-off: culls the candidate front/left-front/right-front camera groups
(defined against 1_1_neutral's own canonical camera numbering -- see
coordinate_transforms/params_io.py for why camera *index* is only meaningful
post-Phase-1-reindex) down to whichever of those camera ids actually exist,
with a valid entry, for every successfully-origin-space-transformed
actor/expression pair. Writes the result to a JSON that
process_expression_sapiens.py loads at startup as its default camera list.

Run once (or re-run whenever the actor range grows) -- not folded into the
labeling script's own per-run logic, so the culled list stays stable/inspectable
between labeling runs instead of silently shrinking/growing run to run.

Usage:
    python compute_facescape_cameras.py
    python compute_facescape_cameras.py --out /custom/path.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # wrap_script_facescape/, for facescape_paths
import facescape_paths as fp  # all paths: wrap_script_facescape/.env

IMAGES_ROOT = fp.FACESCAPE_IMAGES_ROOT
MANIFEST_PATH = fp.FACESCAPE_PIPELINE_DIR / "coordinate_transforms" / "manifest.jsonl"
DEFAULT_OUT_PATH = fp.FACESCAPE_PIPELINE_DIR / "facescape_cameras.json"

# Candidate camera ids, in 1_1_neutral's own canonical numbering (see the
# origin actor/expression's params.json). Front-facing can see both eyelids
# and lips; left/right-front can only see their own side -- excludes
# fully-left/fully-right/rear entirely, not relevant to eyelids/lips (those
# are reserved for the future ear/nosebridge/eyebrow/undernose pass).
CANDIDATE_GROUPS = {
    "front": [1, 4, 10, 19, 20, 22, 23, 24, 30, 31, 32, 33, 34, 43, 52, 54],
    "left_front": [5, 7, 16, 44, 45, 48],
    "right_front": [8, 9, 17, 21, 25, 35, 36, 37, 39],
}

ORIGIN_ACTOR = "1"
ORIGIN_EXPRESSION = "1_neutral"


def successfully_transformed_pairs() -> list[tuple[str, str]]:
    pairs = [(ORIGIN_ACTOR, ORIGIN_EXPRESSION)]
    if MANIFEST_PATH.exists():
        seen = set()
        for line in MANIFEST_PATH.open():
            d = json.loads(line)
            seen.add((d["actor"], d["expression"]))
        pairs.extend(sorted(seen))
    return pairs


def camera_usable(actor: str, expression: str, camera_id: int) -> bool:
    pair_dir = IMAGES_ROOT / actor / expression
    jpg_path = pair_dir / f"{camera_id}.jpg"
    if not jpg_path.exists():
        return False
    params_path = pair_dir / "params.json"
    if not params_path.exists():
        return False
    try:
        params = json.loads(params_path.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    return bool(params.get(f"{camera_id}_valid", True)) and f"{camera_id}_Rt" in params


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(DEFAULT_OUT_PATH), help="Output JSON path")
    parser.add_argument(
        "--min-coverage", type=float, default=0.95,
        help=(
            "Minimum fraction of successfully-transformed pairs a camera must be usable "
            "in to be kept (default: 0.95). Requiring literal 100%% is too strict in "
            "practice -- most candidate cameras sit at 97-100%% coverage (a handful of "
            "pairs drop a *different* camera each due to per-pair calibration noise, see "
            "coordinate_transforms' own README notes), but 1-2 cameras can be genuine "
            "outliers (e.g. one seen at 67%% coverage) worth excluding outright rather "
            "than accepting frequent per-pair gaps."
        ),
    )
    args = parser.parse_args()

    pairs = successfully_transformed_pairs()
    if not pairs:
        raise SystemExit(f"No successfully-transformed pairs found (checked {MANIFEST_PATH})")
    print(f"Culling camera groups against {len(pairs)} successfully-transformed pair(s) "
          f"(min coverage: {args.min_coverage:.0%})")

    culled = {}
    for group, candidates in CANDIDATE_GROUPS.items():
        kept = []
        dropped = []
        for cam in candidates:
            count = sum(1 for actor, expression in pairs if camera_usable(actor, expression, cam))
            coverage = count / len(pairs)
            if coverage >= args.min_coverage:
                kept.append(cam)
            else:
                dropped.append((cam, coverage))
        if dropped:
            print(f"  {group}: dropped " + ", ".join(f"{cam} ({cov:.0%})" for cam, cov in dropped))
        culled[group] = kept

    culled["all_culled"] = sorted({cam for group in culled.values() for cam in group})

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(culled, f, indent=2)

    print(f"front={len(culled['front'])} left_front={len(culled['left_front'])} "
          f"right_front={len(culled['right_front'])} all_culled={len(culled['all_culled'])}")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
