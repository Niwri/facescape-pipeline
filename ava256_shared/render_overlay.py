#!/usr/bin/env python3
"""Renders a wireframe overlay of one ava-256 actor's per-frame registered
mesh (decoder/kinematic_tracking/<frame>.ply) onto one camera's photo for
that frame.

Uses the same core pinhole camera-projection math as
wrap_script/test_render_gt_raw.py's project_vertices_paired() (adapted to
plain numpy for a single frame/camera -- no batching needed here -- same
adaptation style as FaceView/.../render_overlay_frame.py's own numpy
projection), and local copies of wrap_script/flame.py's draw_lines_on_image()
and wrap_script_facescape/test_render_wireframe.py's compute_wireframe_edges()
for the actual line-drawing. flame.py itself is NOT imported: it pulls in
pytorch3d and a full FLAME parametric model loader at import time, neither of
which this script needs -- ava-256's mesh is a real registered scan per
frame, not a FLAME parametric fit requiring UV-resampling/reconstruction like
test_render_gt_raw.py's own subject.

decoder/kinematic_tracking/<frame>.ply is VERTEX-ONLY (confirmed via its own
PLY header: "element vertex 7306", no "element face" at all) -- the actual
face/topology data lives separately, in FaceView/ava-256/assets/face_topology.obj
(also 7306 vertices, confirmed matching; 11432 faces). That file's own vertex
POSITIONS aren't used at all here (only its face index list, i.e. topology,
via compute_wireframe_edges()) -- every frame's own kinematic_tracking/<frame>.ply
already provides this frame's real vertex positions.

decoder/head_pose/<frame>.txt is a row-major 3x4 [R|t] rigid transform
mapping the per-frame registered mesh (in its own local/canonical space) into
this capture's shared world space -- decoder/camera_calibration.json's
extrinsics are expressed directly in that same world space, so the two
compose directly (world_verts = head_pose @ local_verts homogeneous).

decoder/camera_calibration.json's K and T matrices are stored COLUMN-MAJOR
(each inner list is a column, not a row) -- confirmed by inspection: T's
un-transposed last row is [tx, ty, tz, 1.0] (a translation vector, not
[0,0,0,1]), and K's un-transposed last row is [cx, cy, 1.0]. Transposing both
recovers the standard row-major [R|t] / [[fx,0,cx],[0,fy,cy],[0,0,1]] form.

K is calibrated against each camera's raw sensor resolution, not the AVIF
photos actually shipped in decoder/image/cam<id>.zip -- those are a uniform
4x downsample (confirmed: AVIF 667x1024 vs raw ~2668x4096 inferred from
2*cx/2*cy landing almost exactly on that round number), so AVIF_SCALE below
scales fx/fy/cx/cy to match, same idea as test_render_gt_raw.py's own
RESOLUTION_FACTOR for Nersemble's raw-vs-delegation-video mismatch.

Usage:
    python3 render_overlay.py <FRAME_ID> <CAMERA_ID> [--actor-dir ...]
"""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import cv2
import numpy as np
import trimesh
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ACTOR_DIR = Path("/scratch/thirty/irwinngo/ava-256/20210810--1306--FXN596")
OUTPUT_ROOT = SCRIPT_DIR / "renders"
FACE_TOPOLOGY_PATH = SCRIPT_DIR / "face_topology.obj"  # not used by load_camera/load_image/project_points -- camera_utils.py's own only real usage of this module

# See module docstring's last paragraph.
AVIF_SCALE = 0.25
LINE_THICKNESS = 1


def load_head_pose(actor_dir: Path, frame_id: str) -> np.ndarray:
    """Row-major 3x4 [R|t] -- local (registered-mesh) space -> world space."""
    path = actor_dir / "decoder" / "head_pose" / "extracted" / f"{frame_id}.txt"
    if not path.exists():
        raise FileNotFoundError(f"head_pose not found: {path} (extract decoder/head_pose/head_pose.zip first)")
    return np.loadtxt(path).reshape(3, 4)


def load_mesh_vertices(actor_dir: Path, frame_id: str) -> np.ndarray:
    path = actor_dir / "decoder" / "kinematic_tracking" / f"{frame_id}.ply"
    if not path.exists():
        raise FileNotFoundError(f"registration mesh not found: {path}")
    mesh = trimesh.load(str(path), process=False)
    return np.asarray(mesh.vertices, dtype=np.float64)


def compute_wireframe_edges(faces: np.ndarray) -> np.ndarray:
    """faces: (F, 3) triangle vertex indices -> (E, 2) unique undirected edges
    (each edge's two vertex indices sorted ascending, then deduplicated).
    Duplicated from wrap_script_facescape/test_render_wireframe.py's own
    function of the same name -- see module docstring for why this doesn't
    import that module directly (unrelated FaceScape-specific dependencies)."""
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
    edges = np.sort(edges, axis=1)
    return np.unique(edges, axis=0)


def _load_obj_faces(path: Path) -> np.ndarray:
    """Minimal direct parser for this file's own 'f i/j j/k k/l ...' lines --
    deliberately NOT trimesh: this file has fewer 'vt ' lines (5779) than
    'v ' lines (7306, confirmed matching kinematic_tracking/<frame>.ply's own
    vertex count), and trimesh.load() reports only 5779 vertices for it --
    it's silently dropping/renumbering vertices unreferenced by any (v, vt)
    pair, which would desync face indices from kinematic_tracking's own
    per-frame vertex numbering (confirmed: raw 'f' lines directly reference
    vertex indices up to 7306, 1-based, spanning the file's full 'v' list --
    exactly the indexing this script actually needs). Only the leading
    vertex-index token of each 'i[/vt[/vn]]' face corner is used."""
    faces: list[list[int]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.startswith("f "):
                continue
            faces.append([int(tok.split("/")[0]) - 1 for tok in line.split()[1:4]])
    return np.array(faces, dtype=np.int64)


def load_wireframe_edges(num_vertices_expected: int) -> np.ndarray:
    """Loads FACE_TOPOLOGY_PATH's face list (only the topology -- this file's
    own vertex positions are never used, see module docstring) and returns
    its unique wireframe edges."""
    if not FACE_TOPOLOGY_PATH.exists():
        raise FileNotFoundError(f"face topology not found: {FACE_TOPOLOGY_PATH}")
    faces = _load_obj_faces(FACE_TOPOLOGY_PATH)
    max_idx = int(faces.max())
    if max_idx >= num_vertices_expected:
        raise ValueError(
            f"{FACE_TOPOLOGY_PATH} references vertex index {max_idx} (0-based), "
            f"but this frame's mesh only has {num_vertices_expected} vertices"
        )
    return compute_wireframe_edges(faces)


def load_camera(actor_dir: Path, camera_id: str) -> tuple[np.ndarray, np.ndarray]:
    """Returns (K, Rt) as standard row-major matrices -- K already scaled by
    AVIF_SCALE, Rt is the 4x4 world_2_cam extrinsic."""
    calib_path = actor_dir / "decoder" / "camera_calibration.json"
    with open(calib_path) as f:
        calib = json.load(f)
    entry = next((c for c in calib["KRT"] if c["cameraId"] == camera_id), None)
    if entry is None:
        available = [c["cameraId"] for c in calib["KRT"]]
        raise KeyError(f"Camera {camera_id!r} not found in {calib_path}. Available: {available}")

    K = np.array(entry["K"], dtype=np.float64).T  # stored column-major -- see module docstring
    K[0, 0] *= AVIF_SCALE  # fx
    K[1, 1] *= AVIF_SCALE  # fy
    K[0, 2] *= AVIF_SCALE  # cx
    K[1, 2] *= AVIF_SCALE  # cy

    Rt = np.array(entry["T"], dtype=np.float64).T  # stored column-major -- see module docstring
    return K, Rt


def load_image(actor_dir: Path, camera_id: str, frame_id: str) -> np.ndarray:
    """Reads one frame's AVIF photo straight out of its camera's zip (never
    extracted in bulk -- these zips are large and per-camera), as BGR (cv2
    convention)."""
    zip_path = actor_dir / "decoder" / "image" / f"cam{camera_id}.zip"
    if not zip_path.exists():
        raise FileNotFoundError(f"No image zip for camera {camera_id}: {zip_path}")
    with zipfile.ZipFile(zip_path) as zf:
        member = f"cam{camera_id}/{frame_id}.avif"
        with zf.open(member) as f:
            pil_image = Image.open(f).convert("RGB")
            rgb = np.array(pil_image)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def project_points(verts_world: np.ndarray, K: np.ndarray, Rt: np.ndarray) -> np.ndarray:
    """Same pinhole perspective-projection math as
    wrap_script/test_render_gt_raw.py's project_vertices_paired(), adapted to
    plain numpy for a single frame/camera (no batching needed here)."""
    verts_cam = verts_world @ Rt[:3, :3].T + Rt[:3, 3]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x = verts_cam[:, 0] / verts_cam[:, 2] * fx + cx
    y = verts_cam[:, 1] / verts_cam[:, 2] * fy + cy
    return np.stack([x, y], axis=-1)


def draw_lines_on_image(image: np.ndarray, verts: np.ndarray, lines: np.ndarray, thickness: int = LINE_THICKNESS) -> None:
    """Duplicated from wrap_script/flame.py's function of the same name --
    see module docstring for why this script doesn't import flame.py itself."""
    for line in lines:
        x1 = int(verts[line[0], 0])
        y1 = int(verts[line[0], 1])
        x2 = int(verts[line[1], 0])
        y2 = int(verts[line[1], 1])
        cv2.line(image, (x1, y1), (x2, y2), (20, 255, 20), thickness, cv2.LINE_AA)


def render_overlay(actor_dir: Path, frame_id: str, camera_id: str) -> Path:
    head_pose = load_head_pose(actor_dir, frame_id)  # (3,4)
    verts_local = load_mesh_vertices(actor_dir, frame_id)  # (V,3)
    verts_world = verts_local @ head_pose[:3, :3].T + head_pose[:3, 3]

    wireframe_edges = load_wireframe_edges(num_vertices_expected=verts_local.shape[0])

    K, Rt_cam = load_camera(actor_dir, camera_id)
    verts2d = project_points(verts_world, K, Rt_cam)

    image = load_image(actor_dir, camera_id, frame_id)
    draw_lines_on_image(image, verts2d, wireframe_edges)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_ROOT / f"{actor_dir.name}_{frame_id}_{camera_id}.jpg"
    cv2.imwrite(str(output_path), image)
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("frame_id", help="Frame identifier, e.g. 029693 (matches kinematic_tracking/head_pose/image filenames)")
    parser.add_argument("camera_id", help="Camera identifier without the 'cam' prefix, e.g. 400939")
    parser.add_argument("--actor-dir", default=str(DEFAULT_ACTOR_DIR), help="ava-256 actor directory")
    args = parser.parse_args()

    output_path = render_overlay(Path(args.actor_dir), args.frame_id, args.camera_id)
    print(f"Saved overlay render to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
