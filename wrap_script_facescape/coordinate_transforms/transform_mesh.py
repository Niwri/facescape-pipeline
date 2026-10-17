"""Apply an already-solved similarity transform to a .ply mesh's vertices.

Pure numpy/trimesh -- no Blender/bpy dependency, unlike
~/test_code/apply_mesh_transform_blender.py, whose math this mirrors exactly:
verts_new = verts @ (s*Rg).T + tg (equivalently, using the 4x4 "matrix" form,
verts @ M[:3,:3].T + M[:3,3]). Meshes have no per-camera index concept, so
unlike params_io.remap_and_transform there's no reindexing step here -- just
the geometric transform.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh


def apply_similarity_to_mesh(verts: np.ndarray, s: float, Rg: np.ndarray, tg: np.ndarray) -> np.ndarray:
    return verts @ (s * Rg).T + tg


def transform_mesh_file(src_path: str | Path, dst_path: str | Path, s: float, Rg: np.ndarray, tg: np.ndarray) -> None:
    mesh = trimesh.load(str(src_path), process=False)
    mesh.vertices = apply_similarity_to_mesh(np.asarray(mesh.vertices), s, Rg, tg)
    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(dst_path))
