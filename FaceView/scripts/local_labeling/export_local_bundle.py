#!/usr/bin/env python3
"""Host side, step 1: snapshot what the local labeler needs into a small bundle.

Writes to OUT_DIR (default ./local_bundle):
  label_tracker.json        copy of the live tracker -- the one you edit locally
  label_tracker.base.json   identical copy -- the merge base; never edit it
  ava256_frames.json        {capture: {"frames": [[frame_id, status], ...],
                                        "is_done": bool,
                                        "skin_errors": {frame_id: px}}}
  rewrapped.json            {"unconfirmed": {actor: [sequence, ...]}, "confirmed": {...}}:
                            the Nersemble pairs wrap_script/run_pipeline_batch_all.py
                            --unconfirmed-only / --confirmed-only has re-wrapped
                            (its completed_sequences_rewrap_*.json records), for the
                            labeler's "Re-wrapped" filter

Videos are NOT copied; the script prints rsync commands for the three video
folders (Nersemble line_renders + point_overlay_renders, Ava-256
line_renders_ava256). Copy the bundle and the videos to your PC, run
local_labeler.py there, then bring label_tracker.json back and run
merge_local_labels.py.

Usage (from FaceView/):
    python scripts/local_labeling/export_local_bundle.py [--out OUT_DIR]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # FaceView/
sys.path.insert(0, str(ROOT))
# run_pipeline_batch_all.py's re-wrap resume records, next to it in wrap_script/ (a sibling of FaceView/).
REWRAP_RECORDS = {
    "unconfirmed": ROOT.parent / "wrap_script" / "completed_sequences_rewrap_unconfirmed.json",
    "confirmed": ROOT.parent / "wrap_script" / "completed_sequences_rewrap_confirmed.json",
}

from backend.app.core.config import get_settings  # noqa: E402
from backend.app.services import ava256_labels  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("local_bundle"))
    args = parser.parse_args()
    settings = get_settings()
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    # One read of the tracker, written twice, so the two copies are identical.
    tracker_bytes = settings.label_tracker_path.read_bytes()
    json.loads(tracker_bytes)  # refuse to bundle a corrupt tracker
    (out / "label_tracker.json").write_bytes(tracker_bytes)
    (out / "label_tracker.base.json").write_bytes(tracker_bytes)

    frames = {}
    for capture in ava256_labels.list_captures():
        capture_id = capture["capture_id"]
        frames[capture_id] = {
            "frames": [[f, s] for f, s in ava256_labels.read_frames(capture_id)],
            "is_done": ava256_labels.is_done(capture_id),
            "skin_errors": ava256_labels.read_skin_errors(capture_id),
        }
    (out / "ava256_frames.json").write_text(json.dumps(frames), encoding="utf-8")

    rewrapped = {}
    for kind, path in REWRAP_RECORDS.items():
        try:
            rewrapped[kind] = {actor: sorted(seqs) for actor, seqs in json.loads(path.read_text()).items()}
        except (OSError, ValueError) as exc:
            print(f"NOTE: no re-wrap record for {kind} ({path}: {exc}); the labeler shows none as re-wrapped")
            rewrapped[kind] = {}
    (out / "rewrapped.json").write_text(json.dumps(rewrapped), encoding="utf-8")

    nersemble = json.loads(tracker_bytes).get("Nersemble", {})
    n_pairs = sum(len(seqs) for seqs in nersemble.values())
    n_rewrapped = {k: sum(len(v) for v in d.values()) for k, d in rewrapped.items()}
    print(f"Wrote {out.resolve()}: tracker ({n_pairs} Nersemble pairs), "
          f"ava256_frames.json ({len(frames)} captures), rewrapped.json "
          f"({n_rewrapped['unconfirmed']} unconfirmed + {n_rewrapped['confirmed']} confirmed re-wrapped pairs)")
    print("\nCopy to your PC (VIDEOS is any folder you like; pass it as --videos):")
    host = "<user>@<host>"
    print(f"  scp -r {host}:{out.resolve()} ./local_bundle")
    for src in (settings.line_renders_root, settings.point_overlay_renders_root, settings.line_renders_ava256_root):
        print(f"  rsync -av --include='*.mp4' --exclude='*/' --exclude='*' {host}:{src}/ VIDEOS/{src.name}/")
    print("Then: python local_labeler.py --bundle ./local_bundle --videos VIDEOS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
