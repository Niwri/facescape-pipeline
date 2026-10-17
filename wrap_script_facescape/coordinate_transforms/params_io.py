"""IO and transform-application helpers for FaceScape params.json files.

load_cams/apply_transform are ported from ~/test_code/align_camera_params.py;
load_transform is ported from ~/test_code/apply_camera_params_transform.py
(handles both the scale/rotation/translation and 4x4-"matrix" transform-JSON
schemas -- the same schema mesh_transform_blender.py writes, so
1_1_neutral_to_origin.json loads through this unchanged).

build_canonical_index/remap_and_transform are new: FaceScape's per-file camera
index `i` (the `<i>_Rt`/`<i>_sn`/... key prefix) is NOT a stable physical-camera
identity across actors -- only the serial number (`<i>_sn`) is. Verified
empirically: actor 1's index 0 and actor 2's index 0 are different physical
cameras entirely, and camera counts differ per actor (55/58/59/57 seen across
actors 1/2/5/10). Every actor/expression's cameras must be re-keyed by serial
number to match 1_1_neutral's own indexing (the canonical numbering) before
anything downstream (culled camera groups, sapiens2 labeling) can assume
"camera 7" means the same physical camera everywhere.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from geometry import orthonormalize


def cam_indices(d: dict) -> list[str]:
    return sorted(
        {k.split("_", 1)[0] for k in d if k.split("_", 1)[0].lstrip("-").isdigit()},
        key=int,
    )


def load_cams(path: str | Path) -> tuple[dict, dict]:
    """Returns (raw_dict, {serial_number: {key, R, t, C, valid}})."""
    with open(path, "r") as f:
        d = json.load(f)
    cams = {}
    for i in cam_indices(d):
        if f"{i}_Rt" not in d:
            continue
        Rt = np.asarray(d[f"{i}_Rt"], dtype=float)
        R, t = Rt[:, :3].copy(), Rt[:, 3].copy()
        det = np.linalg.det(R)
        if abs(det - 1.0) > 1e-3:
            R = orthonormalize(R)
        sn = str(d.get(f"{i}_sn", f"idx{i}"))
        cams[sn] = dict(key=i, R=R, t=t, C=-R.T @ t,
                        valid=bool(d.get(f"{i}_valid", True)))
    return d, cams


def apply_transform(dict_A: dict, cams_A: dict, s: float, Rg: np.ndarray, tg: np.ndarray, F: np.ndarray) -> dict:
    """Applies the similarity transform to every camera, keeping each
    camera's ORIGINAL index (no reindexing). Used only for the one-shot
    origin transform on 1_1_neutral itself, which IS the canonical index
    source -- its own numbering never changes."""
    out = dict(dict_A)
    for _sn, a in cams_A.items():
        i = a["key"]
        C_new = s * Rg @ a["C"] + tg
        R_new = orthonormalize(F @ a["R"] @ Rg.T)
        t_new = -R_new @ C_new
        out[f"{i}_Rt"] = np.hstack([R_new, t_new[:, None]]).tolist()
    return out


def build_canonical_index(target_params_path: str | Path) -> dict[str, str]:
    """serial -> canonical camera index (as a string, matching the key-prefix
    convention), from the already-origin-space 1_1_neutral params.json."""
    _d, cams = load_cams(target_params_path)
    return {sn: a["key"] for sn, a in cams.items()}


# Per-camera key groups in a FaceScape params.json (confirmed via inspection:
# <i>_K, <i>_Rt, <i>_distortion, <i>_height, <i>_match_num, <i>_ori, <i>_sn,
# <i>_valid, <i>_width). "_Rt" is handled separately since it's transformed,
# not copied through as-is.
CAMERA_KEY_SUFFIXES = ("_K", "_distortion", "_width", "_height", "_sn", "_valid", "_ori", "_match_num")


def remap_and_transform(
    dict_A: dict,
    cams_A: dict,
    s: float,
    Rg: np.ndarray,
    tg: np.ndarray,
    F: np.ndarray,
    canonical_index: dict[str, str],
) -> tuple[dict, dict[str, str], list[tuple[str, str]]]:
    """Applies the similarity transform AND reindexes every camera by serial
    number to match `canonical_index`. Cameras whose serial has no canonical
    match are dropped entirely -- they have no canonical slot and nothing
    downstream would ever reference them by an original, actor-specific index.

    Returns (new_dict, index_map, dropped):
      new_dict: params dict keyed by canonical index
      index_map: {original_index: canonical_index} for kept cameras -- the
                 caller uses this to rename/copy the corresponding <index>.jpg
      dropped: [(original_index, serial), ...] for cameras with no canonical match
    """
    out: dict = {}
    index_map: dict[str, str] = {}
    dropped: list[tuple[str, str]] = []

    for sn, a in cams_A.items():
        orig_i = a["key"]
        if sn not in canonical_index:
            dropped.append((orig_i, sn))
            continue
        canon_i = str(canonical_index[sn])

        C_new = s * Rg @ a["C"] + tg
        R_new = orthonormalize(F @ a["R"] @ Rg.T)
        t_new = -R_new @ C_new
        out[f"{canon_i}_Rt"] = np.hstack([R_new, t_new[:, None]]).tolist()

        for suffix in CAMERA_KEY_SUFFIXES:
            src_key = f"{orig_i}{suffix}"
            if src_key in dict_A:
                out[f"{canon_i}{suffix}"] = dict_A[src_key]

        index_map[orig_i] = canon_i

    return out, index_map, dropped


def load_transform(path: str | Path, inverse: bool = False):
    """Reads a transform JSON (mesh_transform_blender.py / align_camera_params.py
    --matrix-out schema): either scale+rotation+translation keys, or a 4x4
    'matrix'. Returns (s, Rg, tg, F, meta)."""
    with open(path, "r") as f:
        d = json.load(f)

    if all(k in d for k in ("scale", "rotation", "translation")):
        s = float(d["scale"])
        Rg = np.asarray(d["rotation"], float)
        tg = np.asarray(d["translation"], float)
    elif "matrix" in d:
        M = np.asarray(d["matrix"], float)
        if M.shape != (4, 4):
            sys.exit(f"'matrix' in {path} is not 4x4")
        lin = M[:3, :3]
        s = float(np.cbrt(np.linalg.det(lin)))
        if abs(s) < 1e-12:
            s = 1.0
        Rg = lin / s
        tg = M[:3, 3].copy()
    else:
        sys.exit(f"{path}: need either scale/rotation/translation or matrix")

    Rg = orthonormalize(Rg)
    F = np.asarray(d["F"], float) if d.get("F") is not None else np.eye(3)
    F = orthonormalize(F)

    if inverse:
        s, Rg, tg, F = (1.0 / s), Rg.T.copy(), -(1.0 / s) * (Rg.T @ tg), F.T.copy()
    return s, Rg, tg, F, d
