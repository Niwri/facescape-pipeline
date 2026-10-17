#!/usr/bin/env python3
"""Saves one ava-256 actor's per-frame registered mesh
(decoder/kinematic_tracking/<frame>.ply) as a standalone .obj file, with
faces from FaceView/ava-256/assets/face_topology.obj (the .ply itself is
VERTEX-ONLY -- see render_overlay.py's own module docstring for why the
topology has to come from that separate file, and why it's parsed directly
rather than via trimesh).

By default vertices are written in WORLD space (local registered-mesh space
transformed by decoder/head_pose/<frame>.txt), matching what render_overlay.py
and render_overlay_mesh.py project through the camera -- i.e. this frame's
mesh lines up with decoder/camera_calibration.json's cameras and with every
other frame's own world-space export. Pass --space local to instead write
the raw per-frame registered-mesh space (no head_pose applied).

Usage:
    python3 extract_obj.py <FRAME_ID> [--actor ACTOR] [--actor-dir ...] [--space world|local] [--output PATH]

--actor takes just the 6-character actor code (e.g. FXN596), the last
'--'-separated segment of the actor directory's own name (e.g.
20210810--1306--FXN596) -- resolved against ACTOR_ROOT by globbing for that
suffix, since the leading date/session segments aren't something a caller
would otherwise know. --actor-dir still accepts a full path directly (e.g.
for an actor directory that lives outside ACTOR_ROOT).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ACTOR_ROOT = Path("/scratch/thirty/irwinngo/ava-256")
DEFAULT_ACTOR_DIR = ACTOR_ROOT / "20210810--1306--FXN596"
OUTPUT_ROOT = SCRIPT_DIR / "renders"
FACE_TOPOLOGY_PATH = SCRIPT_DIR / "face_topology.obj"  # overridden per-call by mesh_utils.build_world_mesh()


def resolve_actor_dir(actor: str) -> Path:
    """Resolves a 6-character actor code (e.g. 'FXN596') to its full
    directory under ACTOR_ROOT (e.g. .../20210810--1306--FXN596), by
    globbing for that suffix -- the leading date/session segments aren't
    something a caller would otherwise know."""
    matches = sorted(ACTOR_ROOT.glob(f"*--{actor}"))
    if not matches:
        raise FileNotFoundError(f"No actor directory matching '*--{actor}' found under {ACTOR_ROOT}")
    if len(matches) > 1:
        raise ValueError(f"Multiple actor directories matched '*--{actor}' under {ACTOR_ROOT}: {matches}")
    return matches[0]


def load_head_pose(actor_dir: Path, frame_id: str) -> np.ndarray:
    """Row-major 3x4 [R|t] -- local (registered-mesh) space -> world space."""
    path = actor_dir / "decoder" / "head_pose" / f"{frame_id}.txt"
    if not path.exists():
        raise FileNotFoundError(f"head_pose not found: {path} (extract decoder/head_pose/head_pose.zip first)")
    return np.loadtxt(path).reshape(3, 4)


def load_mesh_vertices(actor_dir: Path, frame_id: str) -> np.ndarray:
    import trimesh

    path = actor_dir / "decoder" / "kinematic_tracking" / f"{frame_id}.ply"
    if not path.exists():
        raise FileNotFoundError(f"registration mesh not found: {path}")
    mesh = trimesh.load(str(path), process=False)
    return np.asarray(mesh.vertices, dtype=np.float64)


def load_faces(num_vertices_expected: int) -> np.ndarray:
    """Minimal direct parser for FACE_TOPOLOGY_PATH's own 'f i/j j/k k/l ...'
    lines -- deliberately NOT trimesh, see render_overlay.py's own
    load_wireframe_edges()/_load_obj_faces() docstring for why (trimesh drops/
    renumbers vertices unreferenced by any (v, vt) pair in this specific
    file, desyncing face indices from kinematic_tracking's own per-frame
    vertex numbering)."""
    if not FACE_TOPOLOGY_PATH.exists():
        raise FileNotFoundError(f"face topology not found: {FACE_TOPOLOGY_PATH}")
    faces: list[list[int]] = []
    with FACE_TOPOLOGY_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.startswith("f "):
                continue
            faces.append([int(tok.split("/")[0]) - 1 for tok in line.split()[1:4]])
    faces_arr = np.array(faces, dtype=np.int64)
    max_idx = int(faces_arr.max())
    if max_idx >= num_vertices_expected:
        raise ValueError(
            f"{FACE_TOPOLOGY_PATH} references vertex index {max_idx} (0-based), "
            f"but this frame's mesh only has {num_vertices_expected} vertices"
        )
    return faces_arr


def write_obj(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as f:
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")


def extract_obj(actor_dir: Path, frame_id: str, space: str = "world", output: Path | None = None) -> Path:
    verts_local = load_mesh_vertices(actor_dir, frame_id)  # (V,3)

    if space == "world":
        head_pose = load_head_pose(actor_dir, frame_id)  # (3,4)
        vertices = verts_local @ head_pose[:3, :3].T + head_pose[:3, 3]
    elif space == "local":
        vertices = verts_local
    else:
        raise ValueError(f"space must be 'world' or 'local', got {space!r}")

    faces = load_faces(num_vertices_expected=verts_local.shape[0])

    if output is None:
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        output = OUTPUT_ROOT / f"{actor_dir.name}_{frame_id}_{space}.obj"
    else:
        output.parent.mkdir(parents=True, exist_ok=True)

    write_obj(output, vertices, faces)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("frame_id", help="Frame identifier, e.g. 029693 (matches kinematic_tracking/head_pose filenames)")
    actor_group = parser.add_mutually_exclusive_group()
    actor_group.add_argument("--actor", default=None,
                              help="6-character actor code, e.g. FXN596 (resolved against ACTOR_ROOT="
                                   f"{ACTOR_ROOT})")
    actor_group.add_argument("--actor-dir", default=None, help=f"Full ava-256 actor directory (default: {DEFAULT_ACTOR_DIR})")
    parser.add_argument("--space", choices=["world", "local"], default="world",
                         help="'world' applies head_pose (aligns with camera_calibration.json's cameras); "
                              "'local' writes the raw per-frame registered-mesh space (default: world)")
    parser.add_argument("--output", default=None, help="Output .obj path (default: renders/<actor>_<frame>_<space>.obj)")
    args = parser.parse_args()

    if args.actor is not None:
        actor_dir = resolve_actor_dir(args.actor)
    elif args.actor_dir is not None:
        actor_dir = Path(args.actor_dir)
    else:
        actor_dir = DEFAULT_ACTOR_DIR

    output_path = extract_obj(
        actor_dir,
        args.frame_id,
        args.space,
        Path(args.output) if args.output else None,
    )
    print(f"Saved mesh to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
