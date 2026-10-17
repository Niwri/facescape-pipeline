#!/usr/bin/env python3
"""One-off: composes an additional -90 degree rotation about the X axis onto
1_1_neutral_to_origin.json's transform (origin space is still rotated 90
degrees on X after the existing transform is applied -- confirmed by visual
inspection), and regenerates outputs from the corrected transform.

The correction is a global rotation applied on top of the existing solved
transform, in the OUTPUT (origin) frame: for any point x transformed by the
old (s, Rg, tg) as x' = s*Rg@x + tg, the corrected point is
R_fix @ x' = s*(R_fix@Rg)@x + (R_fix@tg) -- so Rg_new = R_fix@Rg_old and
tg_new = R_fix@tg_old reproduce the desired extra rotation for every
downstream consumer (camera centers, camera rotations via apply_transform's
own F@R@Rg.T composition, and mesh vertices via transform_mesh_file) without
those functions needing any changes -- they already correctly propagate
whatever (s, Rg, tg) they're given.

Staged rollout (per explicit request): this script currently only regenerates
1_1_neutral's own scanned mesh, to a test-only path that does NOT touch the
live irwinngo1 tree or the master transform JSON -- for visual confirmation
before touching params.json, the origin target's live output, or any other
actor/expression.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from params_io import load_transform
from transform_mesh import transform_mesh_file
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # wrap_script_facescape/, for facescape_paths
import facescape_paths as fp  # all paths: wrap_script_facescape/.env

ROOT = Path(__file__).resolve().parent
ORIGIN_TRANSFORM_JSON = Path.home() / "test_code" / "1_1_neutral_to_origin.json"
SOURCE_MESH_PATH = fp.FACESCAPE_RAW_SHAPE_ROOT / "1" / "1_neutral.ply"
TEST_OUTPUT_DIR = ROOT / "test_output"
TEST_MESH_PATH = TEST_OUTPUT_DIR / "1_1_neutral_origin_fixed.ply"

# Rotation by -90 degrees about the X axis (right-hand rule): Y -> -Z, Z -> Y.
R_FIX_X_MINUS_90 = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0],
    [0.0, -1.0, 0.0],
])


def compute_corrected_transform() -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    s, Rg, tg, F, _meta = load_transform(str(ORIGIN_TRANSFORM_JSON))
    Rg_new = R_FIX_X_MINUS_90 @ Rg
    tg_new = R_FIX_X_MINUS_90 @ tg
    return s, Rg_new, tg_new, F


def main() -> int:
    if not ORIGIN_TRANSFORM_JSON.exists():
        raise SystemExit(f"Origin transform not found: {ORIGIN_TRANSFORM_JSON}")
    if not SOURCE_MESH_PATH.exists():
        raise SystemExit(f"Source mesh not found: {SOURCE_MESH_PATH}")

    s, Rg_new, tg_new, _F = compute_corrected_transform()
    print(f"Corrected transform: scale={s:.9f}")
    print(f"Rg_new=\n{Rg_new}")
    print(f"tg_new={tg_new}")

    TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    transform_mesh_file(SOURCE_MESH_PATH, TEST_MESH_PATH, s, Rg_new, tg_new)
    print(f"Saved corrected 1_1_neutral mesh (test only, live tree untouched) -> {TEST_MESH_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
