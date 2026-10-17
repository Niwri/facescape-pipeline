"""Mesh/world-space loading and per-capture zip-extraction helpers for the
Ava-256 pipeline. Wraps extract_obj.py's own vertex/head-pose/face loaders
(reused via explicit-file-path import, same trick wrap_script_facescape's
run_pipeline.py already uses for wrap_script/triangulate.py) rather than
duplicating their parsing logic.
"""
from __future__ import annotations

import fcntl
import importlib.util
import json
import zipfile
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_extract_obj = _load_module("wrap_script_ava256_extract_obj", SCRIPT_DIR / "extract_obj.py")


def resolve_capture_dir(ava256_data_root: Path, capture_id: str) -> Path:
    capture_dir = Path(ava256_data_root) / capture_id
    if not capture_dir.is_dir():
        raise FileNotFoundError(f"No capture directory for {capture_id!r} at {capture_dir}")
    return capture_dir


# --- Zip extraction ---------------------------------------------------------

# subfolder -> (zip filename, extracted-file extension). Confirmed on disk
# across all 64 captures under /scratch/thirty/irwinngo/ava-256: each zip's
# members are exactly the flat <frame_id>.<ext> filenames that sit alongside
# it once extracted (kinematic_tracking/head_pose already extracted in every
# capture inspected; keypoints_3d is NOT extracted in 63/64).
_EXTRACTION_TARGETS: dict[str, tuple[str, str]] = {
    "kinematic_tracking": ("registration_vertices.zip", "ply"),
    "head_pose": ("head_pose.zip", "txt"),
    "keypoints_3d": ("keypoints_3d.zip", "npy"),
}

# In-process cache of (capture_dir, subfolder) pairs already confirmed
# extracted this run, so repeated per-frame calls (e.g. keypoints_3d reads
# during propagation) don't re-touch the lock file/re-glob every time.
_EXTRACTED_THIS_PROCESS: set[tuple[Path, str]] = set()


def ensure_extracted(actor_dir: Path, subfolder: str, frame_id: str | None = None) -> Path:
    """Ensures decoder/<subfolder>/ has its flat per-frame files extracted
    from decoder/<subfolder>/<zip_name> alongside the zip. Returns
    decoder/<subfolder> itself.

    With frame_id given, checks for that SPECIFIC file
    (decoder/<subfolder>/<frame_id>.<ext>) -- more robust than a bare glob,
    since keypoints_3d can be fully extracted yet still lack one specific
    frame's .npy (a real per-frame data gap, not an extraction gap); the
    glob-only check would misreport that as "needs extraction" and
    redundantly re-run extractall(). Without frame_id, checks via glob (used
    when scanning many candidate frames broadly, e.g. neutral-frame
    coverage scanning).

    Extraction is a whole-zip zipfile.extractall() in place -- not a
    member-by-member on-the-fly read (unlike camera_utils.py's AVIF image
    zip reads) -- matching the kinematic_tracking/head_pose convention
    already established on disk. Guarded by a BLOCKING fcntl.flock(LOCK_EX)
    on decoder/<subfolder>/.extract.lock: this is a one-time prerequisite
    every concurrent worker on this capture actually needs completed (not
    redundant work to skip), so workers wait rather than skip on contention.
    """
    subfolder_path = Path(actor_dir) / "decoder" / subfolder
    cache_key = (Path(actor_dir), subfolder)
    if cache_key in _EXTRACTED_THIS_PROCESS:
        return subfolder_path

    zip_name, ext = _EXTRACTION_TARGETS[subfolder]
    zip_path = subfolder_path / zip_name

    def _already_extracted() -> bool:
        if frame_id is not None:
            return (subfolder_path / f"{frame_id}.{ext}").exists()
        return any(subfolder_path.glob(f"*.{ext}"))

    if _already_extracted():
        _EXTRACTED_THIS_PROCESS.add(cache_key)
        return subfolder_path

    if not zip_path.exists():
        # Neither the flat file(s) nor the zip exist -- a genuinely missing
        # capture/frame, not an extraction problem. Let the caller's own
        # load_*() raise its normal FileNotFoundError.
        return subfolder_path

    subfolder_path.mkdir(parents=True, exist_ok=True)
    lock_path = subfolder_path / ".extract.lock"
    with lock_path.open("w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            if not _already_extracted():
                print(f"Extracting {zip_path} -> {subfolder_path}/ (first access this capture)")
                with zipfile.ZipFile(zip_path) as zf:
                    zf.extractall(subfolder_path)
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)

    _EXTRACTED_THIS_PROCESS.add(cache_key)
    return subfolder_path


def ensure_capture_extracted(actor_dir: Path) -> None:
    """Called once per capture, before any per-frame read, for the two data
    types whose specific frame isn't known yet at this point in both
    run_pipeline.py's and run_neutral_skin_propagation.py's own startup
    (the neutral frame itself hasn't been resolved yet -- resolving it is
    what needs keypoints_3d readable in the first place)."""
    ensure_extracted(actor_dir, "kinematic_tracking")
    ensure_extracted(actor_dir, "head_pose")
    ensure_extracted(actor_dir, "keypoints_3d")


# --- World-space mesh construction ------------------------------------------

def build_world_mesh(actor_dir: Path, frame_id: str, mesh_topology_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """(verts_world (7306,3), faces (11432,3)) -- verts via extract_obj's own
    load_mesh_vertices/load_head_pose composition; faces via its
    load_faces(), with its module-global FACE_TOPOLOGY_PATH overridden per
    call so --ava256-mesh-topology-path is honored without duplicating
    extract_obj.py's own manual OBJ-face parser (trimesh silently
    drops/renumbers vertices unreferenced by any 'vt' pair in this specific
    file -- see extract_obj.py's own docstring)."""
    ensure_extracted(actor_dir, "kinematic_tracking", frame_id=frame_id)
    ensure_extracted(actor_dir, "head_pose", frame_id=frame_id)

    _extract_obj.FACE_TOPOLOGY_PATH = Path(mesh_topology_path)
    verts_local = _extract_obj.load_mesh_vertices(actor_dir, frame_id)
    head_pose = _extract_obj.load_head_pose(actor_dir, frame_id)
    verts_world = verts_local @ head_pose[:3, :3].T + head_pose[:3, 3]
    faces = _extract_obj.load_faces(num_vertices_expected=verts_local.shape[0])
    return verts_world, faces


def write_obj(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    _extract_obj.write_obj(Path(path), vertices, faces)


def load_keypoints_3d(actor_dir: Path, frame_id: str) -> np.ndarray:
    """(N,6) array, columns [keypoint_id, x, y, z, confidence_sum,
    num_inliers] per the official Ava-256 DATASHEET.md -- N <= 274, some
    keypoint_ids may be absent for a given frame if confidence was too low.
    Returns an empty (0,6) array if this specific frame has no file at all
    (not every frame necessarily has a keypoints_3d entry)."""
    ensure_extracted(actor_dir, "keypoints_3d", frame_id=frame_id)
    path = Path(actor_dir) / "decoder" / "keypoints_3d" / f"{frame_id}.npy"
    if not path.exists():
        return np.empty((0, 6), dtype=np.float64)
    return np.load(path)


def keypoints_3d_by_id(actor_dir: Path, frame_id: str) -> dict[int, np.ndarray]:
    """keypoint_id -> (x,y,z), for whichever ids are present in this frame."""
    kp = load_keypoints_3d(actor_dir, frame_id)
    return {int(row[0]): row[1:4] for row in kp}


# --- POT row -> FLAME vertex/normal -----------------------------------------
#
# Every row of neutral_correspondence_without_eyes_POT_all.json is
# [face_index, u, w] with (u, w) always exactly one of (1,0)/(0,1)/(0,0) --
# i.e. it pins to one EXACT vertex of the FLAME template (local corner 0, 1,
# or 2 of that face), never an interior barycentric point (verified
# numerically against the live file: every row's (u,w) pair is corner-exact).

def pot_row_local_corner(u: float, w: float) -> int:
    if u == 1.0 and w == 0.0:
        return 0
    if u == 0.0 and w == 1.0:
        return 1
    if u == 0.0 and w == 0.0:
        return 2
    raise ValueError(f"POT row (u={u}, w={w}) is not corner-exact -- unexpected non-vertex correspondence entry")


def pot_row_vertex_index(faces: np.ndarray, row: list[float]) -> int:
    """The FLAME template vertex index a POT row pins to."""
    face_index, u, w = row
    corner = pot_row_local_corner(u, w)
    return int(faces[int(face_index)][corner])


def pot_row_world_xyz(verts_world: np.ndarray, faces: np.ndarray, row: list[float]) -> np.ndarray:
    return verts_world[pot_row_vertex_index(faces, row)]


def pot_row_face_normal(verts_world: np.ndarray, faces: np.ndarray, row: list[float]) -> np.ndarray:
    """Outward-facing unit normal of the triangle a POT row's vertex sits
    on, using neutral.obj's own consistent winding order."""
    face_index = int(row[0])
    v0, v1, v2 = verts_world[faces[face_index]]
    normal = np.cross(v1 - v0, v2 - v0)
    norm = np.linalg.norm(normal)
    if norm == 0.0:
        raise ValueError(f"Degenerate triangle at face_index={face_index}")
    return normal / norm


# --- live skin correspondence (parity with FaceScape's own self-healing) ----
#
# wrap_script_facescape/neutral_correspondence_without_eyes_POT_all.json is a
# CACHE, not a fixed reference: FaceScape's own run_pipeline.py rebuilds it
# via ensure_extended_correspondence_files() whenever its cached row count
# doesn't match len(load_skin_vertex_indices()) -- i.e. whenever
# wrap_script/skin_landmarks.py's keypoint_<region>.txt files have grown/
# shrunk since the cache was last built. Confirmed stale right now: the
# cached file has 33 skin rows, but the LIVE region files give 66. Ava-256
# must track the SAME live vertex set FaceScape's own tooling would use
# today, not a frozen snapshot -- so this reads FaceScape's own read-only
# helpers (load_skin_vertex_indices/_compute_skin_pot_entries) directly
# rather than trusting the cached "_all" file, but writes the result to
# Ava-256's OWN cache file (never FaceScape's shared one) to avoid mutating
# a file another pipeline owns as a side effect of running this one.

_LIVE_EXTENDED_POT_CACHE = SCRIPT_DIR / "neutral_correspondence_without_eyes_POT_all_live.json"

# Overridable module-level constant -- portability handoff (same monkeypatch
# pattern as face_ray_masking_ava256.WRAP_SCRIPT_DIR): run_pipeline.py's
# --facescape-run-pipeline-path flag sets this after import. Defaults to the
# local, self-contained copy (stripped of its original FaceView/backend.app.*
# dependency -- see facescape_run_pipeline.py's own module docstring) rather
# than reaching into wrap_script_facescape/.
FACESCAPE_RUN_PIPELINE_PATH = SCRIPT_DIR / "facescape_run_pipeline.py"

_fs_run_pipeline = None


def _facescape_run_pipeline_module():
    global _fs_run_pipeline
    if _fs_run_pipeline is None:
        _fs_run_pipeline = _load_module(
            "wrap_script_facescape_run_pipeline",
            Path(FACESCAPE_RUN_PIPELINE_PATH),
        )
    return _fs_run_pipeline


def build_live_extended_pot(base_pot_path: Path, neutral_mesh_path: Path) -> Path:
    """Returns the path to a 74-standard + LIVE-skin POT file, rebuilding
    Ava-256's own cache (see module note above) whenever its row count
    doesn't match today's live skin vertex count -- same idempotent
    staleness check as FaceScape's own ensure_extended_correspondence_files()."""
    fs = _facescape_run_pipeline_module()
    skin_vertex_indices = fs.load_skin_vertex_indices()  # pure/read-only -- live assign_global_indices()

    need_rebuild = True
    if _LIVE_EXTENDED_POT_CACHE.exists():
        try:
            existing = json.loads(_LIVE_EXTENDED_POT_CACHE.read_text(encoding="utf-8"))
            with Path(base_pot_path).open("r", encoding="utf-8") as f:
                base_len = len(json.load(f))
            need_rebuild = len(existing) != base_len + len(skin_vertex_indices)
        except (json.JSONDecodeError, OSError):
            need_rebuild = True

    if need_rebuild:
        with Path(base_pot_path).open("r", encoding="utf-8") as f:
            base_pot = json.load(f)
        skin_pot = fs._compute_skin_pot_entries(str(neutral_mesh_path), skin_vertex_indices)  # pure/read-only
        combined = base_pot + skin_pot
        lock_path = _LIVE_EXTENDED_POT_CACHE.with_name(_LIVE_EXTENDED_POT_CACHE.name + ".lock")
        with lock_path.open("w") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            try:
                tmp_path = _LIVE_EXTENDED_POT_CACHE.with_name(_LIVE_EXTENDED_POT_CACHE.name + ".tmp")
                tmp_path.write_text(json.dumps(combined), encoding="utf-8")
                tmp_path.replace(_LIVE_EXTENDED_POT_CACHE)
            finally:
                fcntl.flock(lock_file, fcntl.LOCK_UN)
        print(f"Rebuilt {_LIVE_EXTENDED_POT_CACHE.name} ({len(base_pot)} standard + {len(skin_pot)} live skin indices)")

    return _LIVE_EXTENDED_POT_CACHE
