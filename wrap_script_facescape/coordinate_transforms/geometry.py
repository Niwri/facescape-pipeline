"""Shared geometry helpers, ported from ~/test_code/align_camera_params.py.
Pure numpy, no Blender/bpy dependency -- see that file's own docstring for the
derivation. Kept as a standalone module so params_io.py/solve_alignment.py
don't duplicate this math.
"""
from __future__ import annotations

from itertools import permutations, product

import numpy as np


def orthonormalize(R: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(R)
    Rn = U @ Vt
    if np.linalg.det(Rn) < 0:
        U[:, -1] *= -1.0
        Rn = U @ Vt
    return Rn


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True):
    """Least-squares similarity  dst ~= s R src + t  (src, dst: (N,3))."""
    src = np.asarray(src, float)
    dst = np.asarray(dst, float)
    n = src.shape[0]
    mu_s, mu_d = src.mean(0), dst.mean(0)
    Sc, Dc = src - mu_s, dst - mu_d
    cov = Dc.T @ Sc / n
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt
    var_s = (Sc ** 2).sum() / n
    s = float((D * np.diag(S)).sum() / var_s) if with_scale else 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t


def rot_angle_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    c = (np.trace(Ra.T @ Rb) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def signed_permutations_SO3() -> list[np.ndarray]:
    """The 24 axis-aligned proper rotations (entries in {-1,0,1}, det = +1)."""
    mats = []
    for perm in permutations(range(3)):
        P = np.zeros((3, 3))
        for r, c in enumerate(perm):
            P[r, c] = 1.0
        for signs in product((1.0, -1.0), repeat=3):
            M = P * np.array(signs)
            if abs(np.linalg.det(M) - 1.0) < 1e-9:
                mats.append(M)
    return mats


def solve_Rg_from_rotations(Rs_A: list[np.ndarray], Rs_B: list[np.ndarray], F: np.ndarray) -> np.ndarray:
    """R_B = F R_A Rg^T  =>  Rg = R_B^T F R_A ; average, project to SO(3)."""
    M = np.zeros((3, 3))
    for Ra, Rb in zip(Rs_A, Rs_B):
        M += Rb.T @ F @ Ra
    return orthonormalize(M / len(Rs_A))


def fit_scale_trans(CA: np.ndarray, CB: np.ndarray, Rg: np.ndarray):
    """Given rotation Rg, LS-fit  CB ~= s Rg CA + tg."""
    muA, muB = CA.mean(0), CB.mean(0)
    Ac, Bc = CA - muA, CB - muB
    denom = float((Ac ** 2).sum())
    s = float((Bc * (Ac @ Rg.T)).sum() / denom) if denom > 0 else 1.0
    tg = muB - s * Rg @ muA
    return s, tg


def evaluate(cams_A: dict, cams_B: dict, sns: list[str], s: float, Rg: np.ndarray, tg: np.ndarray, F: np.ndarray) -> dict:
    pos_err, rot_err = [], []
    for sn in sns:
        a, b = cams_A[sn], cams_B[sn]
        C_pred = s * Rg @ a["C"] + tg
        R_pred = F @ a["R"] @ Rg.T
        pos_err.append(np.linalg.norm(C_pred - b["C"]))
        rot_err.append(rot_angle_deg(R_pred, b["R"]))
    pos_err_arr, rot_err_arr = np.array(pos_err), np.array(rot_err)
    return dict(
        pos_rmse=float(np.sqrt((pos_err_arr ** 2).mean())),
        pos_max=float(pos_err_arr.max()),
        rot_rmse=float(np.sqrt((rot_err_arr ** 2).mean())),
        rot_max=float(rot_err_arr.max()),
    )
