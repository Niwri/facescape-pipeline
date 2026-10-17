"""Face-based scaling ("P5") of a FaceScape target before wrapping from the FLAME template, undone on the result.

FaceScape scans are wrapped as they are (metres, near template size), which left the forehead keyline ~3.7 mm
above Nersemble's. Scaling the target so its face matches the template -- along the face's own axes, vertical =
template / pair brow-chin (standard 30/47/69/72 mean -> 50), horizontal = template / pair outer-eye-corner distance
(16 <-> 35), depth = sqrt(vertical * horizontal), about the brow midpoint -- brought it to ~0.6 mm with no
landmark-error change (27-actor sandbox comparison). Same math as wrap_script_ava256/dynamic_transform.py's
compute_face_params.

Only used when wrapping from the template (the neutral expression, or the fallback when an actor's neutral wrap
does not exist yet). Expressions chained from the actor's neutral wrap start at the actor's own scale already.

wrap_with_face_scale() writes the scaled scan / targets next to the pair's output (scan_face_scaled.obj, deleted
after the wrap; target_correspondence_face_scaled.json), wraps those, maps wrapped_mesh.obj back to world space and
records the map in face_scale.json. scan.obj and keypoint_vertices.json stay in world space.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
TEMPLATE_MESH_PATH = SCRIPT_DIR / "neutral.obj"
TEMPLATE_POT_PATH = SCRIPT_DIR / "neutral_correspondence_without_eyes_POT.json"
BROW_INDICES = (30, 47, 69, 72)
CHIN_INDEX = 50
OUTER_EYE_CORNERS = (16, 35)  # left, right
NEEDED = (*BROW_INDICES, CHIN_INDEX, *OUTER_EYE_CORNERS)


def _read_obj_vertices(path: Path) -> tuple[list[str], list[int], np.ndarray]:
    lines = Path(path).read_text().split("\n")
    vi = [i for i, line in enumerate(lines) if line.startswith("v ")]
    return lines, vi, np.array([lines[i].split()[1:4] for i in vi], dtype=np.float64)


def _write_obj_vertices(path: Path, lines: list[str], vi: list[int], verts: np.ndarray) -> None:
    out = list(lines)
    for i, v in zip(vi, verts):
        out[i] = f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}"
    tmp = Path(path).with_name(Path(path).name + ".tmp")
    tmp.write_text("\n".join(out))
    tmp.replace(path)


def _face_frame(points: dict[int, np.ndarray]) -> tuple[np.ndarray, float, float]:
    brow = np.mean([points[i] for i in BROW_INDICES], axis=0)
    up = brow - points[CHIN_INDEX]
    brow_chin = float(np.linalg.norm(up))
    up /= brow_chin
    right = points[OUTER_EYE_CORNERS[0]] - points[OUTER_EYE_CORNERS[1]]
    eye_width = float(np.linalg.norm(right))
    right = right - (right @ up) * up
    right /= np.linalg.norm(right)
    return np.stack([right, up, np.cross(right, up)]), brow_chin, eye_width


def _template_points() -> dict[int, np.ndarray]:
    _, _, tv = _read_obj_vertices(TEMPLATE_MESH_PATH)
    faces = [[int(t.split("/")[0]) - 1 for t in line.split()[1:4]]
             for line in TEMPLATE_MESH_PATH.read_text().split("\n") if line.startswith("f ")]
    pot = json.load(open(TEMPLATE_POT_PATH))
    corner = {(1.0, 0.0): 0, (0.0, 1.0): 1, (0.0, 0.0): 2}  # POT rows sit exactly on a face corner
    return {i: tv[faces[int(pot[i][0])][corner[(float(pot[i][1]), float(pot[i][2]))]]] for i in NEEDED}


def compute_face_map(targets: list[dict], missing: set[int] | None = None) -> tuple[np.ndarray, np.ndarray] | None:
    """(A, c) from the pair's triangulated standard targets (rows of keypoint_vertices.json), or None when one of the
    brow / chin / outer eye corner rows is missing (untriangulated or all-zero)."""
    missing = set(missing or [])
    pts = {}
    for i in NEEDED:
        row = targets[i] if i < len(targets) else None
        if row is None or i in missing or all(float(row[k]) == 0.0 for k in ("x", "y", "z")):
            return None
        pts[i] = np.array([row["x"], row["y"], row["z"]], dtype=np.float64)
    _, t_v, t_h = _face_frame(_template_points())
    R, k_v, k_h = _face_frame(pts)
    s_v, s_h = t_v / k_v, t_h / k_h
    A = R.T @ np.diag([s_h, s_v, np.sqrt(s_h * s_v)]) @ R
    c = np.mean([pts[i] for i in BROW_INDICES], axis=0)
    return A, c


def apply(verts: np.ndarray, A: np.ndarray, c: np.ndarray) -> np.ndarray:
    return (np.asarray(verts) - c) @ A.T + c


def invert(verts: np.ndarray, A: np.ndarray, c: np.ndarray) -> np.ndarray:
    return (np.asarray(verts) - c) @ np.linalg.inv(A).T + c


def wrap_with_face_scale(wrap_mesh, *, target_mesh_path: Path, target_correspondence_path: Path, output_path: Path,
                         extra_indices_to_remove=None, **wrap_kwargs) -> str:
    """wrap_mesh() on the face-scaled target, wrapped mesh mapped back to world space. Returns the method used:
    "face_p5", or "none" (plain wrap) when the targets lack a needed landmark."""
    output_path = Path(output_path)
    out_dir = output_path.parent
    targets = json.load(open(target_correspondence_path))
    fm = compute_face_map(targets, set(extra_indices_to_remove or []))
    if fm is None:
        print("NOTE: face scale skipped -- a brow / chin / outer eye corner target is missing; wrapping unscaled")
        wrap_mesh(target_mesh_path=str(target_mesh_path), target_correspondence_path=str(target_correspondence_path),
                  output_path=str(output_path), extra_indices_to_remove=extra_indices_to_remove, **wrap_kwargs)
        (out_dir / "face_scale.json").write_text(json.dumps({"method": "none"}))
        return "none"
    A, c = fm
    scaled_targets = []
    for row in targets:
        xyz = np.array([row["x"], row["y"], row["z"]], dtype=np.float64)
        if np.allclose(xyz, 0.0):  # placeholder (untriangulated) rows stay recognisable
            scaled_targets.append(dict(row))
        else:
            s = apply(xyz[None], A, c)[0]
            scaled_targets.append({"x": float(s[0]), "y": float(s[1]), "z": float(s[2])})
    scaled_corr = out_dir / "target_correspondence_face_scaled.json"
    scaled_corr.write_text(json.dumps(scaled_targets))
    scaled_scan = out_dir / "scan_face_scaled.obj"
    lines, vi, sv = _read_obj_vertices(target_mesh_path)
    _write_obj_vertices(scaled_scan, lines, vi, apply(sv, A, c))
    try:
        wrap_mesh(target_mesh_path=str(scaled_scan), target_correspondence_path=str(scaled_corr),
                  output_path=str(output_path), extra_indices_to_remove=extra_indices_to_remove, **wrap_kwargs)
    finally:
        scaled_scan.unlink(missing_ok=True)  # ~100 MB per pair
    lines, vi, wv = _read_obj_vertices(output_path)
    _write_obj_vertices(output_path, lines, vi, invert(wv, A, c))
    (out_dir / "face_scale.json").write_text(json.dumps({"method": "face_p5", "A": A.tolist(), "c": c.tolist()}))
    print(f"Face scale (P5): wrapped on the scaled target, result mapped back to world space ({out_dir / 'face_scale.json'})")
    return "face_p5"
