"""Forehead target shift: lowers the forehead skin targets so the neutral wrap's forehead ratio matches Nersemble.

Forehead ratio (on a wrap V, FLAME without eyes, same topology everywhere):
    ((V[3520] - brow) . up) / |brow - chin|
brow = mean of the eyebrow landmark vertices (standard 30/47/69/72), chin = standard 50, up = unit(brow - chin),
vertex 3520 = top of the outer (hairline) keyline on the midline. Nersemble FLAME fits (fit.npz, 223 neutral
frames) average 0.342; TARGET_RATIO = 0.340.

The forehead skin targets (FOREHEAD_VERTICES: skin landmarks 77, 78, 79, 85, 86, 87, 97, 106, 107, 124 plus an
extra point at vertex 668, the mirror of 106) are moved DOWN over the neutral wrap's surface by one distance d,
the same for every point (never up: d >= 0, and a forehead already at or below the target is not moved;
along -up projected onto the surface, re-projected onto the faces every step; points end
up anywhere on the faces). d is found by a secant search on trial neutral wraps (see the dataset's propagation
script), and the shifted world positions are stored in <neutral output dir>/forehead_shift.json. Everything that
uses the neutral frame's skin targets then takes the forehead points from that file: the tracking seeds (so every
expression frame starts from the shifted positions) and the neutral wraps. The eyebrows, the brow-level skin
points (74, 75, 84, 91) and everything else are not moved.
Sandbox test: ava_forehead_slope/shift_0340 (4 Ava-256 captures, 1-2 trial wraps each, standard-landmark error
unchanged).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

TARGET_RATIO = 0.340
TOLERANCE = 0.002
MAX_STEPS = 4
# |d| is kept within MAX_SHIFT_FRACTION of the brow-chin length (every tested capture needed <= 0.04): when the forehead
# barely responds (e.g. its skin targets are later dropped by a landmark filter), the secant step would otherwise jump.
MAX_SHIFT_FRACTION = 0.06
SHIFT_NAME = "forehead_shift.json"
# FLAME vertices of the shifted points: skin landmarks 77, 78, 79, 85, 86, 87, 97, 106, 107, 124 and vertex 668
# (the mirror of 106, which has no shared skin landmark -- EXTRA_SKIN_VERTICES).
FOREHEAD_VERTICES = (691, 705, 1472, 2186, 2196, 2609, 3524, 2166, 2167, 669, 668)
EXTRA_SKIN_VERTICES = (668,)
TOP_VERTEX = 3520
_BAND_PATH = Path(__file__).with_name("forehead_band.json")


def _frame_vertices() -> dict:
    return json.load(open(_BAND_PATH))["frame_vertices"]


def pot_row_vertex(row, faces) -> int:
    """Corner vertex a POT row [face, u, w] pins (u = 1 -> corner 0, w = 1 -> corner 1, else corner 2)."""
    face_index, u, w = row[:3]
    return int(faces[int(face_index)][0 if u >= 0.5 else 1 if w >= 0.5 else 2])


def extra_pot_rows(faces) -> list[list[float]]:
    """Corner-exact POT rows for EXTRA_SKIN_VERTICES, appended after the shared skin rows."""
    faces = np.asarray(faces)
    rows = []
    for v in EXTRA_SKIN_VERTICES:
        f = int(np.nonzero((faces == v).any(1))[0][0]); c = int(np.nonzero(faces[f] == v)[0][0])
        rows.append([float(f), *((1.0, 0.0), (0.0, 1.0), (0.0, 0.0))[c]])
    return rows


def forehead_rows(pot_rows, faces, first_index: int = 74) -> list[int]:
    """Correspondence rows (skin, index >= first_index) of the forehead points."""
    faces = np.asarray(faces)
    return [i for i in range(first_index, len(pot_rows)) if pot_row_vertex(pot_rows[i], faces) in FOREHEAD_VERTICES]


def forehead_ratio(V) -> tuple[float, float]:
    """(ratio, brow -> keyline top height in V's own units)."""
    fv = _frame_vertices(); V = np.asarray(V, dtype=np.float64)
    brow = V[fv["brow"]].mean(0); chin = V[fv["chin"]]
    L = np.linalg.norm(brow - chin); up = (brow - chin) / L
    h = float((V[TOP_VERTEX] - brow) @ up)
    return h / L, h


def brow_chin_length(V) -> float:
    fv = _frame_vertices(); V = np.asarray(V, dtype=np.float64)
    return float(np.linalg.norm(V[fv["brow"]].mean(0) - V[fv["chin"]]))


def shift_down(points, V, faces, d: float, steps: int = 20) -> np.ndarray:
    """Moves each point down over the mesh (V, faces) by d: `steps` steps along -up projected onto the closest
    face's plane, each re-projected onto the surface."""
    import trimesh
    V = np.asarray(V, dtype=np.float64)
    mesh = trimesh.Trimesh(V, np.asarray(faces), process=False)
    fv = _frame_vertices()
    up = V[fv["brow"]].mean(0) - V[fv["chin"]]; up /= np.linalg.norm(up)
    pts = np.asarray(points, dtype=np.float64).copy()
    if abs(d) < 1e-12:
        return pts
    h = d / steps
    for _ in range(steps):
        _, _, tri = trimesh.proximity.closest_point(mesh, pts)
        n = mesh.face_normals[tri]
        t = -up[None] + (up[None] * n).sum(1, keepdims=True) * n
        t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-12
        pts, _, _ = trimesh.proximity.closest_point(mesh, pts + h * t)
    return pts


def next_distance(history: list[tuple[float, float]], target: float, length: float) -> float:
    """Next trial distance from (d, ratio) pairs: first guess (ratio - target) * brow-chin length, then secant;
    clamped to [0, MAX_SHIFT_FRACTION * length] -- the forehead points are only ever moved DOWN."""
    if len(history) < 2:
        d0, r0 = history[-1]
        d = d0 + (r0 - target) * length
    else:
        (d0, r0), (d1, r1) = history[-2], history[-1]
        d = d1 + (r1 - target) * length if abs(r1 - r0) < 1e-6 else d1 + (target - r1) * (d1 - d0) / (r1 - r0)
    limit = MAX_SHIFT_FRACTION * length
    return float(min(max(d, 0.0), limit))


def no_shift_payload(ratio: float, target: float = TARGET_RATIO) -> dict:
    """forehead_shift.json content for a forehead already at or below the target: nothing is moved (the shift is
    down only, never up)."""
    return {"status": "done", "target_ratio": target, "d": 0.0, "ratio": ratio, "d_over_brow_chin": 0.0, "rows": [],
            "points_world": {}, "base_ratio": ratio, "history": [], "note": "ratio already <= target: no shift (down only)"}


def save(output_dir: Path, payload: dict) -> Path:
    path = Path(output_dir) / SHIFT_NAME
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=1))
    os.replace(tmp, path)
    return path


def load(output_dir: Path) -> dict | None:
    path = Path(output_dir) / SHIFT_NAME
    if not path.exists():
        return None
    return json.loads(path.read_text())


def shifted_points(payload: dict | None) -> dict[int, np.ndarray]:
    """row -> shifted world position, from a forehead_shift.json payload (empty when there is none)."""
    if not payload:
        return {}
    return {int(k): np.asarray(v, dtype=np.float64) for k, v in payload["points_world"].items()}


def override_targets(target_points: list, payload: dict | None, to_target_space) -> list:
    """Replaces the forehead rows of a target list ({"x","y","z"} dicts) with the stored shifted positions,
    mapped by to_target_space (world (N,3) -> target space (N,3))."""
    pts = shifted_points(payload)
    rows = [i for i in sorted(pts) if i < len(target_points)]
    if not rows:
        return target_points
    mapped = np.asarray(to_target_space(np.stack([pts[i] for i in rows])), dtype=np.float64)
    out = list(target_points)
    for i, q in zip(rows, mapped):
        out[i] = {"x": float(q[0]), "y": float(q[1]), "z": float(q[2])}
    return out
