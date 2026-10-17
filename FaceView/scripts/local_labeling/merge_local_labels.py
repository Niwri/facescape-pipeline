#!/usr/bin/env python3
"""Host side, step 3: merge a locally edited label_tracker.json back into the
real one, replaying each change through FaceView's own service functions so
the side effects match the web app exactly.

3-way merge against the bundle export_local_bundle.py made (BUNDLE/
label_tracker.base.json and BUNDLE/ava256_frames.json):
  * a pair/capture counts as changed locally when local != base;
  * if the host still equals base it is applied; if the host changed too
    (e.g. the batch moved it), it is a CONFLICT: skipped and reported, or
    applied anyway with --prefer-local.

Nersemble: the status is walked base -> local one step at a time with
label_tracker.confirm_labeled / unconfirm_labeled / confirm / unconfirm /
unconfirm_confirmed, running the same side effects as routes/labels.py
(synchronously): export_wrapped_sequence on confirm,
rename_wrapped_to_old_sequence on unconfirm, delete_exported_sequence on
unconfirm-confirmed.

Ava-256: changed frame labels are written into the capture's review CSV with
the backend's own lock + CSV writer; a capture marked Done locally then goes
through ava256_labels.finish_capture (export .pt, archive CSV, tracker ->
confirmed). --no-done writes the frame labels only: captures stay open (no
.pt export, CSV not archived, status unchanged) to be finished later with
Done in the labeler / web app.

FaceScape and anything unchanged locally are never touched. --only FROM:TO
(repeatable, e.g. --only unconfirmed:confirmed) restricts the merge to
Nersemble pairs going from status FROM (base) to TO (local); every other
change -- including all Ava-256 ones -- is listed as skipped. Dry run by
default; --apply performs it, after backing up the tracker and every touched
CSV / .pt to local_merge_backup_<timestamp>/.

Usage (from FaceView/):
    python scripts/local_labeling/merge_local_labels.py /path/to/label_tracker.json --bundle local_bundle [--apply] [--prefer-local]
        [--only unconfirmed:confirmed ...] [--no-done]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # FaceView/
sys.path.insert(0, str(ROOT))

from backend.app.core.config import get_settings  # noqa: E402
from backend.app.services import ava256_labels  # noqa: E402
from backend.app.services.label_tracker import label_tracker  # noqa: E402
from backend.app.services.wrapped_mesh_export import (  # noqa: E402
    delete_exported_sequence,
    export_wrapped_sequence,
    rename_wrapped_to_old_sequence,
)

LOCAL_AVA_KEY = "_local_ava256"
# action -> (from, to), the API's one-step transitions (label_tracker.py).
ACTIONS = {
    "confirm_labeled": ("unlabeled", "unreviewed"),
    "unconfirm_labeled": ("unreviewed", "unlabeled"),
    "confirm": ("unconfirmed", "confirmed"),
    "unconfirm": ("unconfirmed", "unreviewed"),
    "unconfirm_confirmed": ("confirmed", "unconfirmed"),
}
SIDE_EFFECT = {
    "confirm": ("export .pt", export_wrapped_sequence),
    "unconfirm": ("rename wrapped_mesh.obj -> old_wrapped_mesh.obj", rename_wrapped_to_old_sequence),
    "unconfirm_confirmed": ("delete .pt", delete_exported_sequence),
}


def action_path(start: str, goal: str) -> list[str] | None:
    """Shortest list of API actions from start to goal (None if unreachable)."""
    queue, seen = deque([(start, [])]), {start}
    while queue:
        status, path = queue.popleft()
        if status == goal:
            return path
        for action, (src, dst) in ACTIONS.items():
            if src == status and dst not in seen:
                seen.add(dst)
                queue.append((dst, path + [action]))
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("local_tracker", type=Path, help="The label_tracker.json you edited locally")
    parser.add_argument("--bundle", type=Path, required=True, help="The bundle folder export_local_bundle.py wrote")
    parser.add_argument("--apply", action="store_true", help="Perform the changes (default: dry run)")
    parser.add_argument("--prefer-local", action="store_true", help="Apply local labels even where the host changed too")
    parser.add_argument("--no-done", action="store_true",
                        help="Ava-256: write frame labels only -- ignore local Done (no .pt export, no archive, no confirm)")
    parser.add_argument("--only", action="append", default=None, metavar="FROM:TO",
                        help="Only merge Nersemble pairs whose status went FROM (base) -> TO (local); repeatable")
    args = parser.parse_args()
    only = None
    if args.only:
        only = set()
        for item in args.only:
            src, sep, dst = item.partition(":")
            if not sep or not src or not dst:
                parser.error(f"--only expects FROM:TO, got {item!r}")
            only.add((src, dst))

    settings = get_settings()
    local = json.loads(args.local_tracker.read_text(encoding="utf-8"))
    base = json.loads((args.bundle / "label_tracker.base.json").read_text(encoding="utf-8"))
    base_frames = json.loads((args.bundle / "ava256_frames.json").read_text(encoding="utf-8"))
    host = json.loads(settings.label_tracker_path.read_text(encoding="utf-8"))

    plan: list[tuple] = []          # (kind, key, details...)
    skipped_only: list[str] = []    # changes outside --only
    already: list[str] = []         # host already has the local value (e.g. a re-run after --apply)
    conflicts: list[str] = []
    problems: list[str] = []

    # --- Nersemble ---------------------------------------------------------------
    lN, bN, hN = local.get("Nersemble", {}), base.get("Nersemble", {}), host.get("Nersemble", {})
    for actor in sorted(set(lN) | set(bN)):
        for seq in sorted(set(lN.get(actor, {})) | set(bN.get(actor, {}))):
            l, b, h = lN.get(actor, {}).get(seq), bN.get(actor, {}).get(seq), hN.get(actor, {}).get(seq)
            if l == b or l is None:
                continue
            key = f"Nersemble {actor}/{seq}"
            if only is not None and (b, l) not in only:
                skipped_only.append(f"{key}: {b} -> {l}")
                continue
            if h == l:
                already.append(key)
                continue
            start = b
            if h != b:
                if not args.prefer_local:
                    conflicts.append(f"{key}: base {b}, local {l}, but host is now {h}")
                    continue
                start = h if h is not None else label_tracker.get_status(actor, seq, dataset="Nersemble")
            path = action_path(start, l)
            if path is None:
                problems.append(f"{key}: no API transition path {start} -> {l}")
                continue
            if path:
                plan.append(("nersemble", key, actor, seq, start, l, path))

    # --- Ava-256 -----------------------------------------------------------------
    lA, bA, hA = local.get("Ava256", {}), base.get("Ava256", {}), host.get("Ava256", {})
    local_frames = local.get(LOCAL_AVA_KEY, {})
    for cap in sorted(set(lA) | set(bA)):
        if lA.get(cap) != bA.get(cap) and cap not in local_frames:
            problems.append(f"Ava256 {cap}: status changed locally ({bA.get(cap)} -> {lA.get(cap)}) without frame labels; skipped")
    for cap, entry in sorted(local_frames.items()):
        key = f"Ava256 {cap}"
        if only is not None:
            continue  # --only selects Nersemble transitions; Ava-256 is left alone
        lrows = [tuple(r) for r in entry["frames"]]
        brows = [tuple(r) for r in base_frames.get(cap, {}).get("frames", [])]
        local_done = bool(entry.get("done")) and not args.no_done
        if lrows == brows and not local_done:
            continue
        hrows = [tuple(r) for r in ava256_labels.read_frames(cap)]
        host_done = ava256_labels.is_done(cap)
        if hrows == lrows and host_done == local_done and (not local_done or hA.get(cap) == "confirmed"):
            already.append(key)
            continue
        base_done = bool(base_frames.get(cap, {}).get("is_done"))
        if [f for f, _ in lrows] != [f for f, _ in hrows]:
            problems.append(f"{key}: host frame list differs from yours (video re-rendered?) -- skipped")
            continue
        host_changed = hrows != brows or host_done != base_done or hA.get(cap) != bA.get(cap)
        if host_changed and not args.prefer_local:
            conflicts.append(f"{key}: changed on the host since the bundle (done={host_done}, status={hA.get(cap)})")
            continue
        n_changed = sum(1 for a, c in zip(lrows, hrows) if a[1] != c[1])
        plan.append(("ava", key, cap, lrows, n_changed, local_done, host_done))

    # --- Report ------------------------------------------------------------------
    print(f"Host tracker: {settings.label_tracker_path}")
    print(f"{'APPLY' if args.apply else 'DRY RUN'}: {len(plan)} change(s), {len(already)} already applied, "
          f"{len(conflicts)} conflict(s), {len(problems)} problem(s)"
          + (f", {len(skipped_only)} skipped (not in --only {sorted(':'.join(t) for t in only)})" if only is not None else "")
          + "\n")
    for item in plan:
        if item[0] == "nersemble":
            _, key, _a, _s, start, goal, path = item
            effects = [SIDE_EFFECT[a][0] for a in path if a in SIDE_EFFECT]
            print(f"  {key}: {start} -> {goal} via {' -> '.join(path)}" + (f"  [{'; '.join(effects)}]" if effects else ""))
        else:
            _, key, _c, rows, n_changed, local_done, host_done = item
            extra = (" [restore from archive first]" if host_done else "") + (" + Done (export .pt, archive CSV, confirmed)" if local_done else "")
            print(f"  {key}: {n_changed} frame label(s) to write{extra}")
    for c in conflicts:
        print(f"  CONFLICT {c}")
    for p in problems:
        print(f"  PROBLEM  {p}")
    if not args.apply:
        print("\nNothing changed. Re-run with --apply to perform the changes above.")
        return 0

    # --- Apply -------------------------------------------------------------------
    backup = settings.label_tracker_path.parent / f"local_merge_backup_{time.strftime('%Y%m%d_%H%M%S')}"
    backup.mkdir(parents=True)
    shutil.copy2(settings.label_tracker_path, backup / "label_tracker.json")
    print(f"\nBackup: {backup}")
    applied = failed = 0
    for item in plan:
        try:
            if item[0] == "nersemble":
                _, key, actor, seq, _start, _goal, path = item
                for action in path:
                    getattr(label_tracker, action)(actor, seq, dataset="Nersemble")
                    if action in SIDE_EFFECT:
                        SIDE_EFFECT[action][1](actor, seq)
            else:
                _, key, cap, rows, _n, local_done, host_done = item
                review = ava256_labels._review_csv_path(cap)
                archived = ava256_labels._archived_csv_path(cap)
                export = ava256_labels.export_path(cap)
                for path in (review, archived, export):
                    if path.exists():
                        (backup / "ava256").mkdir(exist_ok=True)
                        shutil.copy2(path, backup / "ava256" / path.name)
                with ava256_labels._capture_lock(cap):
                    if host_done:  # --prefer-local on a capture finished on the host: reopen it
                        shutil.move(str(archived), review)
                        if export.exists():
                            export.unlink()
                        ava256_labels.set_capture_status(cap, ava256_labels.UNCONFIRMED)
                    ava256_labels._write_frames(review, [list(r) for r in rows])
                if local_done:
                    ava256_labels.finish_capture(cap)
            applied += 1
            print(f"  OK   {key}")
        except Exception as exc:  # keep going; report at the end
            failed += 1
            print(f"  FAIL {key}: {exc}")
    print(f"\nDone: {applied} applied, {failed} failed, {len(conflicts)} conflict(s) skipped, {len(problems)} problem(s).")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
