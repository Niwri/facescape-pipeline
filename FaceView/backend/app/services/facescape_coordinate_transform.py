"""Inverts wrap_script_facescape/coordinate_transforms' Phase 1 origin-space
alignment for one FaceScape actor/expression pair, so a confirmed pair's
wrapped mesh (built in shared origin space, since Phase 2/3 sapiens landmarks
and wrapping both operate against facescape_images_root's origin-space
params.json) can be exported back into that pair's own original scan space.

Phase 1 (run_batch_transform.py) solves a fresh (s, Rg, tg) per pair but never
persists it to its own file -- only the *result* (the origin-space params.json
and transformed .ply) survives on disk. Rather than modify Phase 1 to start
persisting a transform file (which wouldn't retroactively help the thousands of
pairs already processed), this re-derives the transform on demand by solving
the alignment a second time in the opposite direction: origin-space params.json
(as the "source" being aligned) onto that same pair's own pristine raw
params.json (as the "target") -- reusing wrap_script_facescape's own tested
solver, so there's no hand-derived matrix-inverse algebra to get subtly wrong.
This is expected to fit near-exactly (not just well), since it's recovering an
already-applied exact transform rather than fitting independent noisy data;
solve_facescape_inverse_transform logs a warning if the residual is larger than
that implies.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

from ..core.config import get_settings

COORDINATE_TRANSFORMS_ROOT = get_settings().facescape_coordinate_transforms_root  # FACESCAPE_COORDINATE_TRANSFORMS_ROOT in FaceView/backend/.env
if str(COORDINATE_TRANSFORMS_ROOT) not in sys.path:
    sys.path.insert(0, str(COORDINATE_TRANSFORMS_ROOT))

from solve_alignment import align_params_to_target  # noqa: E402

log = logging.getLogger(__name__)

# Recovering an already-applied exact transform (not fitting independent noisy
# data) should land near machine precision, modulo float round-trip through
# params.json's default json.dump serialization -- these are generous
# thresholds that would only trip on an actual mismatch (wrong pair, stale
# file, etc.), not ordinary float rounding.
_RESIDUAL_POS_WARN_THRESHOLD = 1e-3
_RESIDUAL_ROT_DEG_WARN_THRESHOLD = 0.1


def solve_facescape_inverse_transform(actor_id: str, expression_id: str) -> tuple[float, np.ndarray, np.ndarray] | None:
    """Returns (s, Rg, tg) mapping this pair's origin-space points back to its
    own original (pristine, pre-Phase-1) scan space, or None (logged) if either
    params.json is missing or too few cameras are shared to solve it -- never
    raises, matching this module's callers' "never raises" contract."""
    settings = get_settings()
    origin_params_path = settings.facescape_images_root / actor_id / expression_id / "params.json"
    raw_params_path = settings.facescape_raw_images_root / actor_id / expression_id / "params.json"

    if not origin_params_path.exists():
        log.warning("solve_facescape_inverse_transform: no origin-space params.json at %s", origin_params_path)
        return None
    if not raw_params_path.exists():
        log.warning("solve_facescape_inverse_transform: no raw params.json at %s", raw_params_path)
        return None

    try:
        # source=origin-space, target=raw -- the exact reverse of Phase 1's own
        # align_params_to_target(raw, origin-target) call, so this solves for
        # the inverse transform directly rather than inverting Phase 1's matrices.
        solved = align_params_to_target(str(origin_params_path), str(raw_params_path))
    except ValueError as exc:
        log.warning("solve_facescape_inverse_transform: cannot solve for %s/%s: %s", actor_id, expression_id, exc)
        return None

    metrics = solved["metrics"]
    if metrics["pos_rmse"] > _RESIDUAL_POS_WARN_THRESHOLD or metrics["rot_rmse"] > _RESIDUAL_ROT_DEG_WARN_THRESHOLD:
        log.warning(
            "solve_facescape_inverse_transform: unexpectedly high residual for %s/%s "
            "(pos_rmse=%.6f rot_rmse=%.6f°) -- expected near-exact recovery of an "
            "already-applied transform; double check origin/raw params.json actually "
            "correspond to the same pair.",
            actor_id, expression_id, metrics["pos_rmse"], metrics["rot_rmse"],
        )

    return solved["s"], solved["Rg"], solved["tg"]


def apply_similarity_to_verts(verts: np.ndarray, s: float, Rg: np.ndarray, tg: np.ndarray) -> np.ndarray:
    """Same point-transform convention as wrap_script_facescape/coordinate_transforms/
    transform_mesh.py's apply_similarity_to_mesh (kept as a local copy rather than
    imported, same reasoning as that codebase's own backfill_camera_landmarks.py:
    a one-line formula isn't worth a cross-repo import for)."""
    return verts @ (s * Rg).T + tg


def write_ply(path: Path, verts: np.ndarray, faces: np.ndarray) -> None:
    """Minimal ASCII PLY writer -- avoids adding trimesh as a new backend
    dependency for this one well-known, simple case (same reasoning as this
    module's sibling wrapped_mesh_export.py's own _load_obj_vertices_faces)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {verts.shape[0]}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        handle.write(f"element face {faces.shape[0]}\n")
        handle.write("property list uchar int vertex_indices\n")
        handle.write("end_header\n")
        for x, y, z in verts:
            handle.write(f"{x} {y} {z}\n")
        for face in faces:
            handle.write(f"3 {face[0]} {face[1]} {face[2]}\n")
