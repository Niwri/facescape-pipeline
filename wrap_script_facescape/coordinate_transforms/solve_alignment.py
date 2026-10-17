"""Solves the similarity transform aligning one FaceScape params.json onto
another (in practice: any actor/expression onto the origin-space 1_1_neutral
target), matched by shared camera serial number. Ported from
~/test_code/align_camera_params.py's solve().

Every (actor, expression) pair is aligned directly and independently against
the one origin-space 1_1_neutral target -- not chained through a per-actor
precomputed transform -- since serial number is the ground-truth camera
correspondence key regardless of each file's own arbitrary index ordering.
This is a one-hop alignment per pair, so no error accumulates from chaining.
"""
from __future__ import annotations

import numpy as np

from geometry import evaluate, fit_scale_trans, solve_Rg_from_rotations
from params_io import load_cams

# A least-squares similarity fit needs at least 3 non-collinear point
# correspondences to be well-posed; matches align_camera_params.py's own floor.
MIN_SHARED_CAMERAS = 3

DEFAULT_CANDIDATES = {
    "I": np.eye(3),
    "flip_YZ": np.diag([1.0, -1.0, -1.0]),
    "flip_XZ": np.diag([-1.0, 1.0, -1.0]),
    "flip_XY": np.diag([-1.0, -1.0, 1.0]),
}


def solve(cams_A: dict, cams_B: dict, sns: list[str], with_scale: bool = True, candidates: dict | None = None) -> dict:
    """Tries each camera-axis correction F, keeps the one with the lowest
    rotation residual (tie-broken by position residual)."""
    if candidates is None:
        candidates = DEFAULT_CANDIDATES
    Rs_A = [cams_A[sn]["R"] for sn in sns]
    Rs_B = [cams_B[sn]["R"] for sn in sns]
    CA = np.array([cams_A[sn]["C"] for sn in sns])
    CB = np.array([cams_B[sn]["C"] for sn in sns])

    best = None
    for name, F in candidates.items():
        Rg = solve_Rg_from_rotations(Rs_A, Rs_B, F)
        s, tg = fit_scale_trans(CA, CB, Rg)
        if not with_scale:
            s = 1.0
            tg = CB.mean(0) - Rg @ CA.mean(0)
        m = evaluate(cams_A, cams_B, sns, s, Rg, tg, F)
        cand = dict(name=name, F=F, s=s, Rg=Rg, tg=tg, metrics=m)
        key = (round(m["rot_rmse"], 6), m["pos_rmse"])
        if best is None or key < (round(best["metrics"]["rot_rmse"], 6), best["metrics"]["pos_rmse"]):
            best = cand
    return best


def align_params_to_target(params_path: str, target_params_path: str, with_scale: bool = True) -> dict:
    """Solves the transform mapping params_path's cameras onto
    target_params_path's, matched by shared serial number.

    Returns dict with dict_A/cams_A (for the caller to pass to
    remap_and_transform), s/Rg/tg/F, metrics (pos_rmse/pos_max/rot_rmse/rot_max),
    n_shared (total shared serials), n_used (valid-in-both, used for the fit).
    Raises ValueError if fewer than MIN_SHARED_CAMERAS shared valid cameras.
    """
    dict_A, cams_A = load_cams(params_path)
    _dict_B, cams_B = load_cams(target_params_path)

    shared = [sn for sn in cams_A if sn in cams_B]
    used = [sn for sn in shared if cams_A[sn]["valid"] and cams_B[sn]["valid"]]
    if len(used) < MIN_SHARED_CAMERAS:
        raise ValueError(
            f"Only {len(used)} shared valid camera(s) between {params_path} and "
            f"{target_params_path} (need >= {MIN_SHARED_CAMERAS})"
        )

    best = solve(cams_A, cams_B, used, with_scale=with_scale)
    return dict(
        dict_A=dict_A, cams_A=cams_A,
        s=best["s"], Rg=best["Rg"], tg=best["tg"], F=best["F"],
        metrics=best["metrics"], n_shared=len(shared), n_used=len(used),
    )
