"""Per-camera standard (0-73) landmark loading that prefers PROPAGATED points
over the raw sapiens2 detections.

frame_<tag>.json holds sapiens2's own per-frame detections and is never
overwritten by propagation. The propagation scripts
(run_neutral_cotracker_propagation.py for Nersemble,
wrap_script_facescape/run_neutral_skin_propagation.py for FaceScape) write
their tracked versions of some standard indices to separate per-group files
next to it -- <prefix>_frame_<tag>.json for each prefix in
PROPAGATED_GROUP_PREFIXES. load_camera_standard_points() merges them the same
way FaceView's landmark_store.py (STANDARD_PROPAGATED_GROUPS) does for the
Cameras tab: start from frame_<tag>.json, then replace every index that
appears in a group file with the group file's point.

Shared by wrap_script/run_pipeline_all.py (Nersemble, tag = delegation index
as :06d) and wrap_script_facescape/run_pipeline.py (tag = "000000"), and by
the point-overlay renderers.
"""
from __future__ import annotations

import json
from pathlib import Path

PROPAGATED_GROUP_PREFIXES: list[str] = ["ear", "nosebridge", "undernose", "eyebrow", "chin"]


def _load_points(path: Path) -> list[dict] | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f).get("points", [])


def load_camera_standard_points(cam_dir: Path, frame_tag: str) -> list[dict] | None:
    """Merged standard points for one camera/frame: frame_<frame_tag>.json's
    sapiens2 points, with every index found in a
    <prefix>_frame_<frame_tag>.json (PROPAGATED_GROUP_PREFIXES) replaced by
    that propagated point. Returns None only when none of those files exist;
    a camera with no frame_ file but some group files returns just the group
    points."""
    cam_dir = Path(cam_dir)
    base = _load_points(cam_dir / f"frame_{frame_tag}.json")
    overrides: dict[int, dict] = {}
    for prefix in PROPAGATED_GROUP_PREFIXES:
        for point in _load_points(cam_dir / f"{prefix}_frame_{frame_tag}.json") or []:
            overrides[point["index"]] = point
    if base is None and not overrides:
        return None
    merged = [p for p in (base or []) if p["index"] not in overrides]
    merged.extend(overrides.values())
    return merged


def load_camera_skin_points(cam_dir: Path, frame_tag: str) -> list[dict] | None:
    """skin_frame_<frame_tag>.json's points (index >= 74), or None if absent."""
    return _load_points(Path(cam_dir) / f"skin_frame_{frame_tag}.json")
