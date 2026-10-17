#!/usr/bin/env python3
"""Confirmed fix (see fix_origin_rotation.py's test-only mesh render, visually
approved): permanently composes the -90 degree X-axis correction into the
master 1_1_neutral_to_origin.json, backs up the original, then regenerates
the origin target's real params.json + mesh + images in the live irwinngo1
tree via run_origin_transform.py (unchanged -- it already does exactly the
right thing given a corrected transform JSON).

Only touches 1/1_neutral (the origin target itself). Every other actor/
expression is realigned directly against this corrected target by a separate,
later run_batch_transform.py --force pass -- not run by this script.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from fix_origin_rotation import ORIGIN_TRANSFORM_JSON, R_FIX_X_MINUS_90
import run_origin_transform


def main() -> int:
    if not ORIGIN_TRANSFORM_JSON.exists():
        raise SystemExit(f"Origin transform not found: {ORIGIN_TRANSFORM_JSON}")

    backup_path = ORIGIN_TRANSFORM_JSON.with_name(ORIGIN_TRANSFORM_JSON.stem + ".pre_x_fix_backup.json")
    if not backup_path.exists():
        shutil.copy(ORIGIN_TRANSFORM_JSON, backup_path)
        print(f"Backed up original transform -> {backup_path}")
    else:
        print(f"Backup already exists, leaving it alone -> {backup_path}")

    with open(ORIGIN_TRANSFORM_JSON) as f:
        d = json.load(f)

    Rg_old = np.asarray(d["rotation"], dtype=float)
    tg_old = np.asarray(d["translation"], dtype=float)
    s = float(d["scale"])

    Rg_new = R_FIX_X_MINUS_90 @ Rg_old
    tg_new = R_FIX_X_MINUS_90 @ tg_old

    d["rotation"] = Rg_new.tolist()
    d["translation"] = tg_new.tolist()
    matrix = np.eye(4)
    matrix[:3, :3] = s * Rg_new
    matrix[:3, 3] = tg_new
    d["matrix"] = matrix.tolist()
    d["note"] = "Composed with an additional -90 degree rotation about X on top of the originally solved transform -- origin space was still rotated 90deg on X after the original transform alone. See fix_origin_rotation.py."

    with open(ORIGIN_TRANSFORM_JSON, "w") as f:
        json.dump(d, f, indent=2)
    print(f"Wrote corrected transform -> {ORIGIN_TRANSFORM_JSON}")

    return run_origin_transform.main()


if __name__ == "__main__":
    raise SystemExit(main())
