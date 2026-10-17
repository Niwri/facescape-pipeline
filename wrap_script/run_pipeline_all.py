import nersemble_paths
import os as _os

# CPU-only by design: Wrap (WrapCmd) runs on CPU, the scan comes precomputed
# from MESHROOM_DATA_ROOT, and the actor's face mask is computed once by
# run_neutral_cotracker_propagation.py (ensure_actor_mask), not here. The GPU
# is hidden before anything imports torch, so FLAME (render_expression_mesh,
# the only torch user left) always runs on CPU and nothing in this process --
# or run_pipeline_batch_all.py, which imports it -- can touch a GPU.
_os.environ["CUDA_VISIBLE_DEVICES"] = ""

from triangulate import triangulate, is_in_bounds
from backend.app.core.config import get_settings
from render_flame_exp_without_eyes import render_expression_mesh
import argparse
import fcntl
import gzip
import threading
import time
import uuid
import cv2
import json
import numpy as np
import os
import shutil
import sys
import trimesh
from pathlib import Path
from run_wrap_script_wrapping_test import wrap_mesh
from skin_landmarks import assign_global_indices, NUM_STANDARD_LANDMARKS
from propagated_landmarks import load_camera_standard_points, load_camera_skin_points
from scan_face_masking import expression_mask_dir, load_actor_neutral_dropped_indices, load_expression_dropped_indices

DEFAULT_CAMERAS_PATH = str(nersemble_paths.FACEVIEW_ROOT / "default_cameras.json")
OUTPUT_ROOT = nersemble_paths.OUTPUT_ROOT

# Precomputed Meshroom scans, one per frame:
# MESHROOM_DATA_ROOT/<actor>/<sequence>/frame_<frame_id>/mesh.obj.gz. The only
# scan source -- Meshroom itself (GPU) is never run from this pipeline; a frame
# without a precomputed scan fails.
MESHROOM_DATA_ROOT = nersemble_paths.MESHROOM_DATA_ROOT

# Extended (74 standard + N skin) versions of the two files run_wrap_script.py's
# wrap_mesh() otherwise hardcodes (neutral_correspondence_without_eyes_POT.json,
# whitelist.json). Generated once and cached here, never mutating the originals
# -- run_pipeline.py / run_pipeline_batch.py keep using the unmodified 74-point
# versions untouched.
EXTENDED_POT_PATH = Path(__file__).with_name("neutral_correspondence_without_eyes_POT_all.json")
EXTENDED_WHITELIST_PATH = Path(__file__).with_name("whitelist_all.json")

# Face masks: four files, <base> + FACE_MASK_SUFFIXES (same list as
# face_ray_masking_wrapping_test.FACE_MASK_SUFFIXES -- not imported from there,
# since that module pulls in torch/pytorch3d/SAM).
FACE_MASK_SUFFIXES = (".txt", "_inverse.txt", "_wrapping_node.txt", "_wrapping_node_inverse.txt")
# Static default mask (build_static_face_mask.py): used only for an actor's
# first two neutral wraps, before its own mask exists.
STATIC_FACE_MASK_PATH = str(Path(__file__).with_name("static_face_mask") / "face_list.txt")


# Written per frame next to wrapped_mesh.obj: the landmark indices that frame's
# wrap actually used (see run_pipeline()).
WRAP_LANDMARK_INDICES_NAME = "wrap_landmark_indices.json"


def actor_face_mask_path(actor: str, neutral_sequence_id: str) -> str:
    """Base path of the actor's own face mask, written by
    run_neutral_cotracker_propagation.ensure_actor_mask()."""
    return str(OUTPUT_ROOT / actor / neutral_sequence_id / "face_list.txt")


def expression_face_mask_path(actor: str, sequence: str) -> str:
    """Base path of a non-neutral sequence's own face mask, written by
    run_neutral_cotracker_propagation.ensure_expression_mask()."""
    return str(expression_mask_dir(actor, sequence) / "face_list.txt")


def face_mask_files_exist(face_mask_path: str) -> bool:
    return all(os.path.exists(face_mask_path + suffix) for suffix in FACE_MASK_SUFFIXES)


def load_default_cameras():
    with open(DEFAULT_CAMERAS_PATH, "r") as f:
        return json.load(f)


def load_skin_vertex_indices() -> list[int]:
    """Wrapped-mesh vertex index for each global skin landmark index (74, 75,
    ...), in that same global-index order -- see skin_landmarks.assign_global_indices."""
    return [vertex_idx for _global_idx, _region, vertex_idx in assign_global_indices()]


def _compute_skin_pot_entries(topology_mesh_path: str, skin_vertex_indices: list[int]) -> list[list[float]]:
    """Point-on-triangle (face_index, u, v) entries for each skin vertex index, in
    the same format as neutral_correspondence_without_eyes_POT.json's existing 74
    entries. Computed from whatever mesh is passed (any wrapped_mesh.obj works --
    every actor/sequence shares the exact same FLAME-derived topology, only vertex
    *positions* differ), by picking a face containing that vertex and assigning it
    full barycentric weight. Convention inferred from the existing file's samples
    (e.g. [3, 1.0, 0.0], [49, 0.0, 0.0]): a face's 0th vertex is (1,0), 1st is
    (0,1), 2nd is (0,0) -- unverified beyond those samples, so double check the
    first wrap run that actually uses a skin point before trusting this broadly.
    """
    mesh = trimesh.load(topology_mesh_path, process=False)
    vertex_to_face: dict[int, tuple[int, int]] = {}
    for face_idx, face in enumerate(mesh.faces):
        for local_pos, vertex_idx in enumerate(face):
            vertex_idx = int(vertex_idx)
            if vertex_idx not in vertex_to_face:
                vertex_to_face[vertex_idx] = (face_idx, local_pos)

    entries = []
    for v in skin_vertex_indices:
        if v not in vertex_to_face:
            raise ValueError(f"Skin vertex index {v} not found in {topology_mesh_path} (mesh has {len(mesh.vertices)} vertices)")
        face_idx, local_pos = vertex_to_face[v]
        if local_pos == 0:
            u, w = 1.0, 0.0
        elif local_pos == 1:
            u, w = 0.0, 1.0
        else:
            u, w = 0.0, 0.0
        entries.append([face_idx, u, w])
    return entries


def _pot_entry_vertex_index(face_idx: int, u: float, w: float, mesh_faces) -> int:
    """FLAME vertex index for one point-on-triangle [face_idx, u, w] entry.
    Every POT entry in this codebase sits exactly on one vertex of its face
    (u==1 -> vertex 0, w==1 -> vertex 1, else -> vertex 2) -- the same
    convention _compute_skin_pot_entries() above generates skin entries
    with, confirmed to hold for all 74 entries currently in
    neutral_correspondence_without_eyes_POT.json too. Falls back to whichever
    vertex has the largest barycentric weight for an entry that doesn't
    cleanly conform (shouldn't happen given the above, but a genuine
    mid-face point would otherwise silently pick the wrong vertex)."""
    face = mesh_faces[face_idx]
    if u == 1.0:
        return int(face[0])
    if w == 1.0:
        return int(face[1])
    if u == 0.0 and w == 0.0:
        return int(face[2])
    weights = (1.0 - u - w, u, w)
    return int(face[max(range(3), key=lambda i: weights[i])])


def _face_mask_vertex_indices(face_mask_path: str, mesh_faces) -> set[int]:
    """Vertex indices touched by at least one face INSIDE this face mask,
    i.e. its face_mask_path + "_inverse.txt" list (a JSON list of polygon/face
    indices). The base face_mask_path + ".txt" is the complement -- the region
    OUTSIDE the face (scalp, neck, eyes, lips), which is what Wrap's FaceMask
    node is given as its selection -- and the two exactly partition every face
    in the topology (e.g. static_face_mask/face_list.txt*: 2083 + 5717 = 7800,
    zero overlap). Checked on actors 032/037 and the static mask: the skin
    landmarks sit on the _inverse.txt faces (31/33 vs 5/33 on .txt for 032)."""
    with open(face_mask_path + "_inverse.txt", "r") as f:
        included_face_indices = json.load(f)
    vertex_indices: set[int] = set()
    for face_idx in included_face_indices:
        vertex_indices.update(int(v) for v in mesh_faces[face_idx])
    return vertex_indices


def ensure_extended_correspondence_files(topology_mesh_path: str, skin_vertex_indices: list[int]) -> tuple[str, str]:
    """Idempotently builds (if missing) and returns paths to the extended POT
    template and whitelist files that include the skin indices on top of the
    standard 74. Safe to call every run -- only regenerates if either file is
    missing or the cached skin-index count doesn't match keypoint_skin_vertices.json
    anymore."""
    need_rebuild = True
    if EXTENDED_POT_PATH.exists() and EXTENDED_WHITELIST_PATH.exists():
        with EXTENDED_POT_PATH.open("r", encoding="utf-8") as f:
            existing = json.load(f)
        need_rebuild = len(existing) != NUM_STANDARD_LANDMARKS + len(skin_vertex_indices)

    if need_rebuild:
        with open("neutral_correspondence_without_eyes_POT.json", "r") as f:
            base_pot = json.load(f)
        skin_pot = _compute_skin_pot_entries(topology_mesh_path, skin_vertex_indices)
        with EXTENDED_POT_PATH.open("w", encoding="utf-8") as f:
            json.dump(base_pot + skin_pot, f)

        with open("whitelist.json", "r") as f:
            base_whitelist = json.load(f)
        skin_indices = [NUM_STANDARD_LANDMARKS + i for i in range(len(skin_vertex_indices))]
        with EXTENDED_WHITELIST_PATH.open("w", encoding="utf-8") as f:
            json.dump(base_whitelist + skin_indices, f)
        print(f"Rebuilt {EXTENDED_POT_PATH.name} / {EXTENDED_WHITELIST_PATH.name} "
              f"({NUM_STANDARD_LANDMARKS} standard + {len(skin_vertex_indices)} skin indices)")

    return str(EXTENDED_POT_PATH), str(EXTENDED_WHITELIST_PATH)


def run_pipeline(intrinsic, points_list, triangulation_extrinsics, source_mesh_path, output_dir,
                 face_mask_path: str, source_mesh_2_path: str, correspondence_2_path: str, scan_gz_path: Path,
                 pot_template_path: str | None = None, whitelist_path: str | None = None,
                 orig_width: int | None = None, orig_height: int | None = None,
                 scan_dropped_indices: list[int] | None = None) -> dict:
    """CPU-only wrap of one frame: triangulate its landmarks, unzip its
    precomputed scan (MESHROOM_DATA_ROOT) to output_dir/mesh.obj, and run Wrap
    with the given face mask (face_mask_path + FACE_MASK_SUFFIXES).

    Skin landmarks (74+) whose FLAME vertex falls outside that same face mask
    are dropped from the correspondence map too -- via the same
    extra_indices_to_remove mechanism triangulate()'s own untriangulated
    points already use, see below. Every standard landmark (0-73: eyes, nose,
    mouth, jaw, eyebrows, ears) is always kept regardless of the mask -- an
    earlier version of this checked every standard landmark and dropped
    roughly half of them (including jaw/eyebrow/nose/mouth points), which was
    too broad. Skin landmarks in SKIN_REGIONS' "under" region (chin/under-jaw
    -- see skin_landmarks.py's own docstring on its chin-facing camera triple)
    are also always kept, for the same reason: they're jaw, not cheek, skin,
    so the face mask isn't the right thing to gate them on either. (Currently
    a no-op in practice -- keypoint_under.txt is still empty/unpopulated as of
    writing this -- but correct once it isn't.)

    scan_dropped_indices: landmarks (standard AND skin) the SAM landmark
    filter dropped on the actor's NEUTRAL frame -- outside the front cameras'
    SAM masks, or occluded in all of them -- from the filtered landmark JSON
    run_neutral_cotracker_propagation.py writes (scan_face_masking.py). Same
    indices for every frame of the actor, like the neutral FLAME face mask;
    also removed from the correspondence map."""
    times = {}

    # 1) Triangulate. Uses only the default/selected cameras (triangulation_extrinsics).
    start_time = time.time()
    formatted, untriangulated_indices = triangulate(intrinsic, np.array(list(triangulation_extrinsics.values())), points_list, orig_width, orig_height)
    if untriangulated_indices:
        print(f"NOTE: {len(untriangulated_indices)} point(s) had no in-bounds view in any camera this "
              f"frame -- excluded from the correspondence map: {untriangulated_indices}")

    # Same [face_idx, u, w] -> vertex convention as _compute_skin_pot_entries()
    # above, against source_mesh_path's own topology (identical for every
    # wrapped_mesh.obj / the base FLAME template alike, see that function's
    # own docstring). Loads the exact same POT file wrap_mesh() itself loads
    # below (same pot_template_path/fallback), so this always lines up with
    # `formatted`'s actual length even when the caller fell back to the
    # 74-point-only POT file (see process_frame()'s own
    # skin-insufficient-observations note).
    pot_template_file = pot_template_path or "neutral_correspondence_without_eyes_POT.json"
    with open(pot_template_file, "r") as f:
        pot_data = json.load(f)
    topology_mesh = trimesh.load(source_mesh_path, process=False)
    mask_vertex_indices = _face_mask_vertex_indices(face_mask_path, topology_mesh.faces)
    jaw_skin_global_indices = {
        global_idx for global_idx, region, _vertex_idx in assign_global_indices()
        if region == "under"
    }
    maskable_indices = set(range(NUM_STANDARD_LANDMARKS, len(pot_data))) - jaw_skin_global_indices
    outside_mask_indices = [
        i for i in sorted(maskable_indices)
        if _pot_entry_vertex_index(int(pot_data[i][0]), pot_data[i][1], pot_data[i][2], topology_mesh.faces)
        not in mask_vertex_indices
    ]
    if outside_mask_indices:
        print(f"NOTE: {len(outside_mask_indices)} skin/ear correspondence point(s) fall outside the "
              f"neutral frame's face mask ({face_mask_path}_inverse.txt) -- excluded from the correspondence "
              f"map: {outside_mask_indices}")

    scan_dropped = sorted(i for i in (scan_dropped_indices or []) if i < len(formatted))
    if scan_dropped:
        print(f"NOTE: {len(scan_dropped)} landmark(s) dropped by the SAM landmark filter (actor's or this "
              f"sequence's, see process_frame) -- outside the SAM masks, or occluded in every filter camera -- "
              f"excluded from the correspondence map: {scan_dropped}")

    correpsondence_path = os.path.join(output_dir, "keypoint_vertices.json")
    with open(correpsondence_path, 'w') as out:
        json.dump(formatted, out)

    # The landmark indices Wrap actually fits to for this frame -- the
    # whitelist minus every removal above (untriangulated, outside the FLAME
    # face mask, outside the scan mask), same set wrap_mesh() keeps -- so
    # overlays (render_point_overlay.py) show exactly these.
    with open(whitelist_path or "whitelist.json", "r") as f:
        whitelist = set(int(i) for i in json.load(f))
    removed = set(untriangulated_indices) | set(outside_mask_indices) | set(scan_dropped)
    with open(os.path.join(output_dir, WRAP_LANDMARK_INDICES_NAME), "w") as out:
        json.dump(sorted(i for i in range(len(formatted)) if i in whitelist and i not in removed), out)
    times["triangulate_time"] = time.time() - start_time

    # 2) Scan: the precomputed Meshroom mesh, unzipped next to the frame's other
    # outputs for WrapCmd. Skipped when an up-to-date copy is already there.
    start_time = time.time()
    target_path = os.path.join(output_dir, "mesh.obj")
    if not (os.path.exists(target_path) and os.path.getmtime(target_path) >= os.path.getmtime(scan_gz_path)):
        tmp_path = os.path.join(output_dir, f"mesh.obj.{uuid.uuid4().hex}.tmp")
        try:
            with gzip.open(scan_gz_path, "rb") as f_in, open(tmp_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
            os.replace(tmp_path, target_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        print(f"Using precomputed scan: {scan_gz_path} -> {target_path}")
    times["scan_time"] = time.time() - start_time

    # 3) Wrap (CPU).
    print("Face mask is", face_mask_path)
    start_time = time.time()
    output_wrap_path = os.path.join(output_dir, "wrapped_mesh.obj")
    wrap_mesh(face_mask_path, source_mesh_path, target_path, correpsondence_path, output_wrap_path, source_mesh_2_path, correspondence_2_path, ignore_ears=False,
              pot_template_path=pot_template_path, whitelist_path=whitelist_path,
              extra_indices_to_remove=sorted(set(untriangulated_indices) | set(outside_mask_indices) | set(scan_dropped)))
    times["wrap_time"] = time.time() - start_time

    return times

def process_frame(
    actor: str,
    sequence: str,
    frame_id: int,
    resolution,
    default_cameras,
    skin_vertex_indices: list[int],
    face_mask_path: str | None = None,
    segmentation_lock: "threading.Lock | None" = None,
) -> dict:
    """Runs one frame's full CPU-only pipeline (triangulation, FLAME mesh,
    precomputed scan, wrap).

    ``face_mask_path`` overrides which face mask Wrap uses (base path of the
    four FACE_MASK_SUFFIXES files) -- run_neutral_cotracker_propagation.py
    passes STATIC_FACE_MASK_PATH for an actor's first two neutral wraps.
    Otherwise the actor's own mask (actor_face_mask_path()) is required and a
    missing one raises RuntimeError. ``segmentation_lock`` is ignored (kept so
    existing callers don't break; there is no segmentation here anymore).

    FLAME mesh generation goes through render_flame_exp_without_eyes.py's
    render_expression_mesh(), self-contained per call (loads fit.npz and
    rebuilds FLAMEBlendshapes fresh every time) -- same as
    ~/wrap_script/run_pipeline_all.py's original design. This does not share a
    warmed-up Flame instance across frames/pairs the way run_pipeline_batch_temp.py
    does; callers processing many frames pay that reload cost per frame.

    Raises RuntimeError if this frame genuinely cannot proceed (e.g. its
    sequence's neutral frame hasn't been wrapped yet) -- callers processing many
    frames should catch this per frame rather than let it propagate, the same
    way a nonzero subprocess return code was previously treated as a failure.
    Other "nothing to do" conditions (missing landmarks root, no triangulation
    data, no source videos) match the original script's behavior of printing a
    note and returning normally (not a failure)."""
    # Exclusive per-frame lock: running the same actor/sequence/frame_id pipeline
    # concurrently (e.g. from two different cluster nodes pointed at the same
    # NFS-mounted output) races on this frame's output_dir -- same lock file
    # (and same blocking LOCK_EX semantics) as run_pipeline_wrapping_test.py's
    # own process_frame(), deliberately shared so the two implementations
    # serialize against each other too when pointed at the same output_dir.
    # Released in the `finally` below rather than left for process exit --
    # this function is called many times in one long-running process, so
    # leaking the lock past this one frame would block every later frame
    # (including this same frame on a legitimate retry) for the rest of the
    # process's life.
    lock_dir = str(OUTPUT_ROOT / actor / sequence / str(frame_id))
    os.makedirs(lock_dir, exist_ok=True)
   
    scan_gz_path = MESHROOM_DATA_ROOT / actor / sequence / f"frame_{frame_id}" / "mesh.obj.gz"
    if not scan_gz_path.exists():
        raise RuntimeError(f"no precomputed scan at {scan_gz_path}")

    lock_file = open(os.path.join(lock_dir, ".pipeline.lock"), "w")
    fcntl.flock(lock_file, fcntl.LOCK_EX)
    try:
        start_absolute = time.time()
        start = time.time()
        times = {}
        settings = get_settings()
        root_path = settings.landmarks_root / actor / sequence
        if(os.path.exists(root_path) == False):
            print(root_path, "does not exist")
            return times

        num_vertices = NUM_STANDARD_LANDMARKS + len(skin_vertex_indices)

        points_list = [[] for _ in range(num_vertices)]

        # Raw video size (landmarks are normalized to it) -- container metadata
        # only, no frame is decoded.
        video_path = settings.data_root / actor / "sequences" / sequence / "images" / "cam_222200037.mp4"
        cap = cv2.VideoCapture(str(video_path))
        orig_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        cam_data_path = settings.data_root / actor / "calibration" / "camera_params.json"
        cam_data = None
        with open(cam_data_path, "r") as f:
            cam_data = json.load(f)

        intrinsic = cam_data["intrinsics"]

        # Triangulation-only camera set (landmark keypoints), read from the same
        # default_cameras.json the frontend uses to preselect selectedCameras. Standard
        # (0..73) points come from propagated_landmarks.load_camera_standard_points():
        # each camera's sapiens2 frame_<idx>.json, with every index that
        # run_neutral_cotracker_propagation.py propagated (ear/nosebridge/undernose/
        # eyebrow/chin_frame_<idx>.json) replaced by its propagated point -- same
        # merge the FaceView Cameras tab shows. Skin points (74+) come from
        # skin_frame_<idx>.json, when present -- missing/never-generated skin files
        # just contribute nothing for that camera, same as any standard point
        # sapiens2 failed to detect.
        # cam_index is assigned by insertion order into triangulation_extrinsics, not
        # by position in default_cameras -- a camera skipped below (no landmark
        # files at all) must not leave a gap, since points_list stores cam_index
        # expecting it to match triangulation_extrinsics.values()'s eventual array
        # position.
        triangulation_extrinsics = {}
        frame_tag = f"{int(frame_id/5):06d}"
        for cam in default_cameras:
            cam_dir = root_path / ("cam_" + cam)
            standard_points = load_camera_standard_points(cam_dir, frame_tag)
            skin_points = load_camera_skin_points(cam_dir, frame_tag)
            if standard_points is None and skin_points is None:
                print(f"NOTE: no landmark files for frame {frame_tag} in {cam_dir} -- "
                      f"excluding cam={cam} from triangulation for this frame")
                continue

            cam_index = len(triangulation_extrinsics)
            triangulation_extrinsics[cam] = cam_data["world_2_cam"][cam]

            for point_data in standard_points or []:
                point = np.array([round(point_data["x"] * orig_width), round(point_data["y"] * orig_height)])
                in_bounds = is_in_bounds(point_data)
                if not in_bounds:
                    print(f"Excluding out-of-bounds point idx={point_data['index']} cam={cam} "
                          f"x={point_data['x']:.4f} y={point_data['y']:.4f}")
                points_list[point_data['index']].append((cam_index, point, in_bounds))

            if skin_points is not None:
                for point_data in skin_points:
                    point = np.array([round(point_data["x"] * orig_width), round(point_data["y"] * orig_height)])
                    in_bounds = is_in_bounds(point_data)
                    if not in_bounds:
                        print(f"Excluding out-of-bounds skin point idx={point_data['index']} cam={cam} "
                              f"x={point_data['x']:.4f} y={point_data['y']:.4f}")
                    points_list[point_data['index']].append((cam_index, point, in_bounds))
            elif skin_vertex_indices:
                print(f"NOTE: {cam_dir / f'skin_frame_{frame_tag}.json'} missing -- run_neutral_cotracker_propagation.py hasn't "
                      f"propagated skin points for cam={cam} frame={frame_id} yet")

        if not triangulation_extrinsics:
            print(f"No default camera has frame data for frame {frame_id} -- nothing to triangulate")
            return times

        # triangulate_point() (triangulate.py) has no graceful fallback for a point
        # with fewer than 2 camera observations -- 0 observations raises LinAlgError
        # (singular matrix), and 1 observation is under-determined and can silently
        # return a bogus point (e.g. the origin) rather than erroring. That's the
        # normal state for every skin index at once when this actor/sequence simply
        # hasn't been through run_neutral_cotracker_propagation.py yet. Rather than
        # crash (or silently corrupt the wrap) in that case, degrade to exactly
        # run_pipeline.py's standard-only (74-point) behavior for this frame.
        if skin_vertex_indices:
            insufficient = [idx for idx in range(NUM_STANDARD_LANDMARKS, num_vertices) if len(points_list[idx]) < 2]
            if insufficient:
                print(f"NOTE: {len(insufficient)}/{len(skin_vertex_indices)} skin point(s) have fewer than 2 camera "
                      f"observations for frame {frame_id} (probably not propagated yet for {actor}/{sequence}) -- "
                      "falling back to standard-only (no skin points) for this frame.")
                points_list = points_list[:NUM_STANDARD_LANDMARKS]
                skin_vertex_indices = []

        times["camera_params_time"] = time.time() - start
        print("Time Elapsed: ", times["camera_params_time"], ". Rendering source mesh...")


        # Determine source meshes. `obj_path` is the wrap topology target: the
        # neutral/reference frame is rendered directly from the FLAME parameters;
        # every other frame instead wraps against the neutral frame's already-computed
        # wrapped_mesh.obj — UNLESS no neutral frame is set at all for this actor, in
        # which case every frame wraps against its own expression.obj independently.
        # `current_frame_obj_path` is always this frame's own per-frame FLAME expression
        # mesh, regardless of whether it is the neutral frame.
        #
        # One neutral frame per ACTOR now (not one per actor/sequence pair), recorded
        # at landmarks_root/<actor>/neutral_frame.json as
        # {"sequence_id": ..., "frame_idx": ...} -- see
        # FaceView/backend/app/services/landmark_store.py's save_neutral_frame().
        # neutral_sequence_id/neutral_frame_id together identify the ONE frame (within
        # its own sequence's video) that is the actual neutral. Whenever this frame's
        # own sequence isn't that sequence, it still wraps against THAT frame's
        # wrapped_mesh.obj -- which lives under output/<actor>/<neutral_sequence_id>/
        # <neutral_frame_id>/, not under this frame's own sequence -- since
        # run_neutral_cotracker_propagation.py's track_and_fuse_from_other_sequence()
        # propagates cross-sequence from that one frame, not from a same-sequence
        # stand-in; no non-origin sequence has a neutral frame of its own anymore.
        actor_neutral_frame_path = settings.landmarks_root / actor / "neutral_frame.json"
        neutral_frame_id = None
        neutral_sequence_id = None
        if actor_neutral_frame_path.exists():
            with open(actor_neutral_frame_path, "r") as f:
                neutral_data = json.load(f)
            neutral_sequence_id = neutral_data["sequence_id"]
            neutral_frame_id = neutral_data["frame_idx"]
        else:
            print(f"No neutral frame set for actor {actor} (missing {actor_neutral_frame_path}) — "
                  "wrapping this frame directly against its own expression.obj instead of "
                  "chaining off a neutral frame's wrapped_mesh.obj.")

        start = time.time()
        current_frame_obj_path = render_expression_mesh(actor, sequence, frame_id)
        if neutral_frame_id is None:
            obj_path = current_frame_obj_path
        elif sequence == neutral_sequence_id and frame_id == neutral_frame_id:
            # This frame IS the actor's one neutral frame -- render directly from FLAME.
            obj_path = current_frame_obj_path
        else:
            obj_path = str(OUTPUT_ROOT / actor / neutral_sequence_id / str(neutral_frame_id) / "wrapped_mesh.obj")
            if not os.path.exists(obj_path):
                # Was sys.exit(1) when this was always its own subprocess -- that
                # would tear down the whole batch (and every other in-flight pair)
                # now that many frames share one process, so this must be a normal
                # exception the caller can catch per frame instead.
                raise RuntimeError(
                    f"neutral frame {neutral_frame_id} for {actor}/{neutral_sequence_id} has not been "
                    f"wrapped yet (missing {obj_path}). Run run_neutral_cotracker_propagation.py first."
                )
        times["source_mesh_time"] = time.time() - start
        print("Time Elapsed: ", times["source_mesh_time"], ". Running pipeline...")

        # Face mask: an explicit override (the static mask, for propagation's
        # bootstrap/second neutral wraps and expression-mask wraps), else this
        # sequence's own mask (run_neutral_cotracker_propagation.
        # ensure_expression_mask(), non-neutral sequences) when it exists, else
        # the actor's own mask from ensure_actor_mask().
        use_expression_mask = False
        if face_mask_path is None:
            if neutral_sequence_id is None:
                raise RuntimeError(f"no neutral frame for {actor}, so no actor face mask -- run propagation first")
            expression_mask = expression_face_mask_path(actor, sequence)
            use_expression_mask = sequence != neutral_sequence_id and face_mask_files_exist(expression_mask)
            face_mask_path = expression_mask if use_expression_mask else actor_face_mask_path(actor, neutral_sequence_id)
        print(f"Face mask ({'expression' if use_expression_mask else 'actor / override'}): {face_mask_path}")
        if not face_mask_files_exist(face_mask_path):
            raise RuntimeError(f"no face mask at {face_mask_path}* -- run run_neutral_cotracker_propagation.py first")

        pot_template_path = None
        whitelist_path = None
        if skin_vertex_indices:
            pot_template_path, whitelist_path = ensure_extended_correspondence_files(obj_path, skin_vertex_indices)

        output_dir = str(OUTPUT_ROOT / actor / sequence / str(frame_id))
        os.makedirs(output_dir, exist_ok=True)
        source_mesh_2_path = current_frame_obj_path
        correspondence_2_path = os.path.join(os.getcwd(), "target_correspondence_without_mask.json")

        # SAM landmark filter, paired with the face mask above: the sequence's own
        # (made with its expression mask) when that mask is used, else the actor's
        # neutral frame's, applied to every frame (written by
        # run_neutral_cotracker_propagation.py, scan_face_masking.filter_frame_landmarks).
        scan_dropped_indices = None
        if use_expression_mask:
            scan_dropped_indices, expression_filter_path = load_expression_dropped_indices(actor, sequence)
            if scan_dropped_indices is None:
                print(f"NOTE: expression mask without its landmark filter ({expression_filter_path}) -- "
                      "using the actor's neutral-frame filter")
            else:
                print(f"Landmark filter (expression): {expression_filter_path}")
        if scan_dropped_indices is None:
            scan_dropped_indices, neutral_filter_path = load_actor_neutral_dropped_indices(settings.landmarks_root, actor)
        if scan_dropped_indices is None:
            print(f"NOTE: no neutral-frame landmark filter at {neutral_filter_path} -- wrapping without it "
                  f"(run run_neutral_cotracker_propagation.py {actor} --mask-only)")

        start = time.time()
        pipeline_times = run_pipeline(
            intrinsic, points_list, triangulation_extrinsics, obj_path, output_dir,
            face_mask_path, source_mesh_2_path, correspondence_2_path, scan_gz_path,
            pot_template_path=pot_template_path, whitelist_path=whitelist_path,
            orig_width=orig_width, orig_height=orig_height,
            scan_dropped_indices=scan_dropped_indices,
        )
        time_pipeline = time.time() - start
        print("Time Elapsed: ", time_pipeline)

        times = {**times, **pipeline_times}
        times["total_time"] = time.time() - start_absolute

        path = "time_anal.json"
        time_anal = {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                time_anal = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            time_anal = {}

        time_anal.setdefault(actor, {}).setdefault(sequence, {})[str(frame_id)] = times
        with open("time_anal.json", "w", encoding="utf-8") as f:
            json.dump(time_anal, f, indent=2)

        return times
    finally:
        fcntl.flock(lock_file, fcntl.LOCK_UN)
        lock_file.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("actor_id", nargs="?", default=None, help="Actor identifier; omit to process all actors")
    parser.add_argument("sequence_id", nargs="?", default=None, help="Sequence identifier; omit to process all sequences")
    parser.add_argument("frame_id", nargs="?", default=None, help="Sequence identifier; omit to process all sequences")
    parser.add_argument("resolution", nargs="?", default=3208, help="Sequence identifier; omit to process all sequences")
    parser.add_argument("--face-mask-path", default=None,
                        help="Base path of the face mask to wrap with (four files: <path>" + "{" + ",".join(FACE_MASK_SUFFIXES) + "}). "
                             "Default: the actor's own mask from run_neutral_cotracker_propagation.py")
    parser.add_argument("--wrap-max-threads", type=int, default=2,
                        help="Caps WrapCmd's own internal thread pool per invocation "
                             "(passed as WrapCmd compute's --maxThreads) -- otherwise it "
                             "sizes its thread pool off every core the cgroup allocation "
                             "exposes, which oversubscribes badly once multiple wraps run "
                             "concurrently (see run_pipeline_batch_all.py's --parallel). "
                             "Default: 2.")
    args = parser.parse_args()

    actor = args.actor_id
    sequence = args.sequence_id
    frame_id = int(args.frame_id)

    # Read by run_wrap_script_wrapping_test.wrap_mesh()'s subprocess call to
    # WrapCmd compute -- set here (once, before any work starts) rather than
    # threaded through process_frame()/run_pipeline()'s signatures.
    os.environ["WRAP_MAX_THREADS"] = str(args.wrap_max_threads)

    default_cameras = load_default_cameras()
    skin_vertex_indices = load_skin_vertex_indices()

    process_frame(
        actor, sequence, frame_id, args.resolution,
        default_cameras, skin_vertex_indices,
        face_mask_path=args.face_mask_path,
    )


if __name__ == "__main__":
    main()
