#!/usr/bin/env python3
"""One-shot: applies 1_1_neutral_to_origin.json to actor 1's 1_neutral
params.json and scanned mesh, writing the origin-space versions into the
irwinngo1 copy. This becomes the fixed alignment target -- and the canonical
camera-index source -- for every other FaceScape actor/expression pair (see
run_batch_transform.py). 1_1_neutral's own camera indexing is left unchanged;
it IS the canonical numbering.

Run this once before run_batch_transform.py.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from params_io import apply_transform, load_cams, load_transform
from transform_mesh import transform_mesh_file
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # wrap_script_facescape/, for facescape_paths
import facescape_paths as fp  # all paths: wrap_script_facescape/.env

ROOT = Path(__file__).resolve().parent

SOURCE_IMAGES_ROOT = fp.FACESCAPE_RAW_IMAGES_ROOT
SOURCE_SHAPE_ROOT = fp.FACESCAPE_RAW_SHAPE_ROOT
DEST_IMAGES_ROOT = fp.FACESCAPE_IMAGES_ROOT
DEST_SHAPE_ROOT = fp.FACESCAPE_SHAPE_ROOT

ORIGIN_TRANSFORM_JSON = Path.home() / "test_code" / "1_1_neutral_to_origin.json"
EXAMPLE_CHAIN_JSON = Path.home() / "test_code" / "2_1_neutral_to_1_1_neutral.json"

ORIGIN_ACTOR = "1"
ORIGIN_EXPRESSION = "1_neutral"


def main() -> int:
    if not ORIGIN_TRANSFORM_JSON.exists():
        raise SystemExit(f"Origin transform not found: {ORIGIN_TRANSFORM_JSON}")

    inputs_dir = ROOT / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(ORIGIN_TRANSFORM_JSON, inputs_dir / ORIGIN_TRANSFORM_JSON.name)
    if EXAMPLE_CHAIN_JSON.exists():
        shutil.copy(EXAMPLE_CHAIN_JSON, inputs_dir / EXAMPLE_CHAIN_JSON.name)
    print(f"Copied transform JSON(s) into {inputs_dir} for provenance")

    s, Rg, tg, F, meta = load_transform(str(ORIGIN_TRANSFORM_JSON))
    print(f"Loaded origin transform: scale={s:.9f} method={meta.get('method')} rmse={meta.get('rmse')}")

    src_params_path = SOURCE_IMAGES_ROOT / ORIGIN_ACTOR / ORIGIN_EXPRESSION / "params.json"
    if not src_params_path.exists():
        raise SystemExit(f"Source params.json not found: {src_params_path}")
    dict_A, cams_A = load_cams(str(src_params_path))
    out_params = apply_transform(dict_A, cams_A, s, Rg, tg, F)

    dest_dir = DEST_IMAGES_ROOT / ORIGIN_ACTOR / ORIGIN_EXPRESSION
    dest_dir.mkdir(parents=True, exist_ok=True)
    with open(dest_dir / "params.json", "w") as f:
        json.dump(out_params, f)
    print(f"Wrote origin-space params.json ({len(cams_A)} cameras) -> {dest_dir / 'params.json'}")

    src_img_dir = SOURCE_IMAGES_ROOT / ORIGIN_ACTOR / ORIGIN_EXPRESSION
    n_copied = 0
    for jpg in src_img_dir.glob("*.jpg"):
        shutil.copy(jpg, dest_dir / jpg.name)
        n_copied += 1
    print(f"Copied {n_copied} images -> {dest_dir} (1_1_neutral's own indexing is canonical, no rename needed)")

    src_mesh_path = SOURCE_SHAPE_ROOT / ORIGIN_ACTOR / f"{ORIGIN_EXPRESSION}.ply"
    if not src_mesh_path.exists():
        raise SystemExit(f"Source mesh not found: {src_mesh_path}")
    dest_mesh_path = DEST_SHAPE_ROOT / ORIGIN_ACTOR / f"{ORIGIN_EXPRESSION}.ply"
    transform_mesh_file(src_mesh_path, dest_mesh_path, s, Rg, tg)
    print(f"Wrote origin-space mesh -> {dest_mesh_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
