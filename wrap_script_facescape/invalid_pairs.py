"""Shared helpers for FaceScape's invalid-pair pipeline (the *_invalid.py scripts).

"Invalid" pairs (settings.facescape_invalid_pairs_path) are actor/expression
pairs whose cameras could not be aligned to the canonical origin rig, so they
have no origin-space params/images/scan and no canonical camera numbering.
Everything here works in each pair's own RAW space instead:

- Cameras come from the pair's own raw params.json, keyed by that pair's raw
  camera index. Raw indices differ between pairs (even two expressions of one
  actor), so cameras are matched across pairs by serial number (`<i>_sn`).
- Front/left/right and per-landmark camera choice are computed from geometry
  (Ava-256's camera_classify / camera_selection) instead of hard-coded,
  canonical-numbered camera sets.
- A raw scan is moved into the FLAME template frame for wrapping by a rotation
  (Kabsch, from triangulated landmarks onto the template's landmark positions)
  followed by Ava-256's bounding-box scaling and center-to-center translation
  (dynamic_transform.py); untransform() is its exact inverse, used to save the
  wrapped mesh back in the pair's raw space.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image
import facescape_paths as fp  # all paths: wrap_script_facescape/.env

SCRIPT_DIR = Path(__file__).resolve().parent
AVA256_DIR = fp.AVA256_SHARED_ROOT
fp.add_faceview_to_sys_path()  # backend.app.core.config

from backend.app.core.config import get_settings  # noqa: E402


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Ava-256's camera_selection / camera_classify import `mesh_utils` and
# `camera_utils` by plain name -- load those from wrap_script_ava256/ by file
# path and register them under those names first (no module of either name
# exists on FaceScape's/FaceView's own paths). Everything else in
# wrap_script_ava256/ stays off sys.path, so its same-named scripts
# (run_pipeline.py, run_wrap_script.py, ...) can never shadow FaceScape's.
if "mesh_utils" not in sys.modules:
    _load_module("mesh_utils", AVA256_DIR / "mesh_utils.py")
if "camera_utils" not in sys.modules:
    _load_module("camera_utils", AVA256_DIR / "camera_utils.py")
mesh_utils = sys.modules["mesh_utils"]
camera_classify = _load_module("ava256_camera_classify", AVA256_DIR / "camera_classify.py")
camera_selection = _load_module("ava256_camera_selection", AVA256_DIR / "camera_selection.py")

TEMPLATE_MESH_PATH = SCRIPT_DIR / "neutral.obj"          # FaceScape's FLAME template (same as run_wrap_script.DEFAULT_NEUTRAL_MESH_PATH)
BASE_POT_PATH = SCRIPT_DIR / "neutral_correspondence_without_eyes_POT.json"  # 74 standard landmark rows
NUM_STANDARD_LANDMARKS = 74


# --- invalid pairs / roots -------------------------------------------------

@lru_cache(maxsize=1)
def load_invalid_pairs() -> frozenset[tuple[str, str]]:
    path = get_settings().facescape_invalid_pairs_path
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return frozenset()
    return frozenset((str(p["actor"]), str(p["expression"])) for p in data.get("pairs", []))


def is_invalid(actor_id: str, expression_id: str) -> bool:
    return (actor_id, expression_id) in load_invalid_pairs()


def images_root_for(actor_id: str, expression_id: str) -> Path:
    """Raw images root for an invalid pair, origin-space root otherwise."""
    s = get_settings()
    return s.facescape_raw_images_root if is_invalid(actor_id, expression_id) else s.facescape_images_root


def scan_path_for(actor_id: str, expression_id: str) -> Path:
    s = get_settings()
    root = s.facescape_raw_shape_root if is_invalid(actor_id, expression_id) else s.facescape_shape_root
    return root / actor_id / f"{expression_id}.ply"


# --- cameras ---------------------------------------------------------------

def load_pair_cameras(actor_id: str, expression_id: str) -> dict[str, dict]:
    """cam_id -> {"intrinsic": (3,3), "extrinsic": (4,4) world->cam, "width",
    "height", "serial"} from the pair's own params.json (raw for an invalid
    pair). Cameras flagged `<i>_valid: false` are skipped. Same dict shape as
    run_neutral_skin_propagation.load_camera_params(), plus "serial"."""
    params_path = images_root_for(actor_id, expression_id) / actor_id / expression_id / "params.json"
    with open(params_path) as f:
        params = json.load(f)
    cam_ids = sorted({k.split("_")[0] for k in params if k.endswith("_Rt")}, key=int)
    cams: dict[str, dict] = {}
    for cam_id in cam_ids:
        if f"{cam_id}_K" not in params or params.get(f"{cam_id}_valid", True) is False:
            continue
        ext = np.eye(4, dtype=np.float64)
        ext[:3, :4] = np.array(params[f"{cam_id}_Rt"], dtype=np.float64)
        cams[cam_id] = {
            "intrinsic": np.array(params[f"{cam_id}_K"], dtype=np.float64),
            "extrinsic": ext,
            "width": int(params.get(f"{cam_id}_width", 0) or 0),
            "height": int(params.get(f"{cam_id}_height", 0) or 0),
            "serial": str(params.get(f"{cam_id}_sn", "")),
        }
    return cams


def cam_id_by_serial(cams: dict[str, dict]) -> dict[str, str]:
    return {cam["serial"]: cam_id for cam_id, cam in cams.items() if cam["serial"]}


def ava_camera_params(cams: dict[str, dict]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """{cam_id: (K, Rt 4x4)} -- the shape Ava-256's camera helpers take."""
    return {cam_id: (cam["intrinsic"], cam["extrinsic"]) for cam_id, cam in cams.items()}


def load_pair_image(actor_id: str, expression_id: str, cam_id: str) -> np.ndarray | None:
    path = images_root_for(actor_id, expression_id) / actor_id / expression_id / f"{cam_id}.jpg"
    if not path.exists():
        return None
    try:
        return np.array(Image.open(path).convert("RGB"))
    except OSError as exc:  # truncated/corrupt jpg -- skip this camera, don't fail the pair
        print(f"NOTE: unreadable image {path}: {exc}")
        return None


def project(world_xyz: np.ndarray, cam: dict) -> tuple[float, float] | None:
    cam_xyz = cam["extrinsic"][:3, :3] @ world_xyz + cam["extrinsic"][:3, 3]
    if cam_xyz[2] <= 1e-8:
        return None
    img = cam["intrinsic"] @ cam_xyz
    return float(img[0] / img[2]), float(img[1] / img[2])


# --- geometric camera choice (Ava-256) --------------------------------------

def pot_rows_by_index(pot_path: Path) -> dict[int, list[float]]:
    with open(pot_path) as f:
        return dict(enumerate(json.load(f)))


FRONT_ANGLE_THRESHOLD_DEG = camera_classify.FRONT_ANGLE_THRESHOLD_DEG  # 35 deg, as Ava-256
EAR_INDICES = (1, 4, 6, 14)
NOSE_TIP_INDEX = 60


def face_front_direction(verts_world, faces, pot_rows) -> tuple[np.ndarray, np.ndarray]:
    """(unit front vector, face center) from the nose: nose tip (60) minus the
    midpoint of the ear points (1, 4, 6, 14). Ava-256's classifier uses the
    surface normal under the nose (index 61) instead; on FaceScape wraps that
    normal points almost straight down (checked on 14/1_neutral: cos -0.04
    against the nose direction), which made nearly every camera "left/right"."""
    ears = np.mean([mesh_utils.pot_row_world_xyz(verts_world, faces, pot_rows[i]) for i in EAR_INDICES], axis=0)
    nose = mesh_utils.pot_row_world_xyz(verts_world, faces, pot_rows[NOSE_TIP_INDEX])
    front = nose - ears
    return front / np.linalg.norm(front), ears


def classify_front_right_left(verts_world, faces, pot_rows, cams) -> tuple[set[str], set[str], set[str]]:
    """Ava-256's front/right/left split (camera within FRONT_ANGLE_THRESHOLD_DEG
    of the face's front direction -> front; else by side of the nostril axis
    38 - 19), with the front direction from face_front_direction()."""
    front_dir, center = face_front_direction(verts_world, faces, pot_rows)
    right_axis = (mesh_utils.pot_row_world_xyz(verts_world, faces, pot_rows[38])
                  - mesh_utils.pot_row_world_xyz(verts_world, faces, pot_rows[19]))
    right_axis = right_axis - np.dot(right_axis, front_dir) * front_dir
    right_axis /= np.linalg.norm(right_axis)
    threshold = np.cos(np.radians(FRONT_ANGLE_THRESHOLD_DEG))
    front, right, left = set(), set(), set()
    for cam_id, cam in cams.items():
        rt = cam["extrinsic"]
        rel = (-rt[:3, :3].T @ rt[:3, 3]) - center
        rel /= np.linalg.norm(rel)
        if float(np.dot(rel, front_dir)) >= threshold:
            front.add(cam_id)
        elif float(np.dot(rel, right_axis)) > 0:
            right.add(cam_id)
        else:
            left.add(cam_id)
    return front, right, left


def select_cameras_for_landmarks(landmark_indices, pot_rows, verts_world, faces, cams,
                                 min_cameras: int = 5, angle_threshold_deg: float = 80.0) -> dict[int, list[str]]:
    """Ava-256's per-landmark camera choice (facing angle + occlusion, then
    greedy coverage) on a FLAME-topology wrap. The occlusion-ray offset is
    scaled to this mesh (0.5% of its bbox diagonal) instead of Ava's mm value."""
    diag = float(np.linalg.norm(verts_world.max(axis=0) - verts_world.min(axis=0)))
    return camera_selection.select_cameras_for_landmarks(
        list(landmark_indices), pot_rows, verts_world, faces, list(cams), ava_camera_params(cams),
        min_cameras=min_cameras, angle_threshold_deg=angle_threshold_deg, normal_nudge=0.005 * diag,
    )


def most_frontal_left_right(verts_world, faces, pot_rows, cams) -> list[str]:
    """Three overlay cameras for an invalid pair: the most frontal camera and
    the most frontal of the left and right groups (by angle to the face's
    front direction). Falls back to whatever exists."""
    front, right, left = classify_front_right_left(verts_world, faces, pot_rows, cams)
    front_dir, centroid = face_front_direction(verts_world, faces, pot_rows)

    def frontalness(cam_id: str) -> float:
        rt = cams[cam_id]["extrinsic"]
        center = -rt[:3, :3].T @ rt[:3, 3]
        rel = center - centroid
        return float(np.dot(rel / np.linalg.norm(rel), front_dir))

    picks = []
    for group in (front, left, right):
        if group:
            picks.append(max(sorted(group), key=frontalness))
    return list(dict.fromkeys(picks))


# --- raw <-> template transform ----------------------------------------------

def template_landmarks() -> dict[int, np.ndarray]:
    """The 74 standard landmark positions on the FLAME template (neutral.obj)."""
    mesh = trimesh.load(str(TEMPLATE_MESH_PATH), process=False)
    verts, faces = np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces)
    rows = pot_rows_by_index(BASE_POT_PATH)
    return {i: mesh_utils.pot_row_world_xyz(verts, faces, rows[i]) for i in range(NUM_STANDARD_LANDMARKS)}


def _bbox(verts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = verts.min(axis=0), verts.max(axis=0)
    return (lo + hi) / 2.0, hi - lo


def compute_template_transform(raw_landmarks: dict[int, np.ndarray], scan_verts: np.ndarray) -> dict:
    """raw -> template frame: R (Kabsch rotation of the raw landmarks onto
    the template's, no reflection), then Ava-256's bbox scaling + center
    translation of the rotated scan onto the template mesh's bbox:
        v_r = v @ R.T
        v'  = (v_r - scan_center) * scale + template_center
    with scan_center/scale from the ROTATED scan's bbox."""
    template = template_landmarks()
    common = sorted(i for i in raw_landmarks if i in template)
    if len(common) < 4:
        raise ValueError(f"need >=4 triangulated standard landmarks to solve the rotation, got {len(common)}")
    a = np.stack([raw_landmarks[i] for i in common])
    b = np.stack([template[i] for i in common])
    a_c, b_c = a - a.mean(axis=0), b - b.mean(axis=0)
    u, _s, vt = np.linalg.svd(a_c.T @ b_c)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ np.diag([1.0, 1.0, d]) @ u.T          # b ~ R a

    template_mesh = trimesh.load(str(TEMPLATE_MESH_PATH), process=False)
    template_center, template_extent = _bbox(np.asarray(template_mesh.vertices, dtype=np.float64))
    scan_center, scan_extent = _bbox(scan_verts @ rotation.T)
    scale = float(np.linalg.norm(template_extent) / np.linalg.norm(scan_extent))

    # Rotation fit quality, after the scale/translation (landmark rms in template units).
    fitted = (a @ rotation.T - scan_center) * scale + template_center
    rms = float(np.sqrt(np.mean(np.sum((fitted - b) ** 2, axis=1))))
    rot_deg = float(np.degrees(np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1, 1))))
    return {"rotation": rotation, "scale": scale, "scan_center": scan_center,
            "template_center": template_center, "landmark_rms": rms, "rotation_deg": rot_deg,
            "landmarks_used": len(common)}


def transform(verts: np.ndarray, p: dict) -> np.ndarray:
    return (np.asarray(verts, dtype=np.float64) @ p["rotation"].T - p["scan_center"]) * p["scale"] + p["template_center"]


def untransform(verts: np.ndarray, p: dict) -> np.ndarray:
    return ((np.asarray(verts, dtype=np.float64) - p["template_center"]) / p["scale"] + p["scan_center"]) @ p["rotation"]


def save_transform(path: Path, p: dict) -> None:
    np.savez(str(path), **{k: np.asarray(v) for k, v in p.items()})


def load_transform(path: Path) -> dict:
    data = np.load(str(path))
    return {k: (float(data[k]) if data[k].ndim == 0 else data[k]) for k in data.files}
