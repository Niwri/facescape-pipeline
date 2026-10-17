"""Shared "skin" landmark region config, used by both
run_neutral_cotracker_propagation.py (which propagates these points and writes
skin_frame_<idx>.json files) and run_pipeline_all.py (which reads those files
back in and needs the same vertex-index -> global-landmark-index assignment to
line up).

Each region is a face-skin area (front/right/left/under) with its own text file of
wrapped-mesh vertex indices (one per line, in
/scratch/ondemand32/irwinngo/wrap_script/keypoint_<region>.txt) and its own triangulation camera triple,
chosen for best visibility of that region. A region's points are tracked and
triangulated using ONLY its own camera triple, then the fused 3D position is
reprojected onto every camera in the actor's calibration -- not just the
triangulating triple -- so skin landmarks show up consistently everywhere
regardless of which cameras happened to produce them.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent
NUM_STANDARD_LANDMARKS = 74

SKIN_REGIONS: dict[str, dict[str, object]] = {
    "front": {
        "path": ROOT / "keypoint_front.txt",
        "cameras": ["222200037", "222200047", "222200036"],
    },
    "right": {
        "path": ROOT / "keypoint_right.txt",
        "cameras": ["221501007", "222200039", "222200049"],
    },
    "left": {
        "path": ROOT / "keypoint_left.txt",
        "cameras": ["222200042", "222200044", "222200040"],
    },
    # Added last (per request) so front/right/left's existing global-index
    # assignment (assign_global_indices() iterates SKIN_REGIONS in this
    # dict's own declared order) is unaffected -- "under"'s vertices just
    # get new global indices appended after them.
    "under": {
        "path": ROOT / "keypoint_under.txt",
        "cameras": ["222200043", "222200039", "222200040"],
    },
}


def _load_region_file(path: Path) -> list[int]:
    if not path.exists():
        return []
    indices: list[int] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            indices.append(int(line))
    return indices


def load_skin_regions() -> dict[str, list[int]]:
    """Region name -> list of wrapped-mesh vertex indices, straight from each
    keypoints_<region>.txt (empty list if that file doesn't exist yet)."""
    return {region: _load_region_file(config["path"]) for region, config in SKIN_REGIONS.items()}


def assign_global_indices(regions: dict[str, list[int]] | None = None) -> list[tuple[int, str, int]]:
    """Flattens every region's vertex indices into a single, stably-ordered list
    of (global_landmark_index, region, vertex_index), assigning global indices
    74, 75, 76, ... in region order (front, then right, then left, then under -- matching
    SKIN_REGIONS' own declaration order) and then file order within a region. A
    vertex index that appears in more than one region file keeps only its first
    occurrence (by that same order) -- a landmark index must be unique.

    This exact ordering is the single source of truth both scripts rely on to
    agree on which global index means which vertex -- if you change it, both
    scripts need to be regenerated/rerun together, since skin_frame_<idx>.json
    files on disk encode the *global* index, not the region+vertex it came from.
    """
    if regions is None:
        regions = load_skin_regions()
    seen: set[int] = set()
    assigned: list[tuple[int, str, int]] = []
    next_index = NUM_STANDARD_LANDMARKS
    for region in SKIN_REGIONS:  # dict insertion order: front, right, left
        for vertex_idx in regions.get(region, []):
            if vertex_idx in seen:
                print(f"NOTE: skin vertex {vertex_idx} appears in more than one region file; "
                      f"keeping its first assignment, dropping the duplicate in '{region}'")
                continue
            seen.add(vertex_idx)
            assigned.append((next_index, region, vertex_idx))
            next_index += 1
    return assigned


def total_skin_count(regions: dict[str, list[int]] | None = None) -> int:
    return len(assign_global_indices(regions))
