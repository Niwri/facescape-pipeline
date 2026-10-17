#!/usr/bin/env python3
"""FaceScape counterpart to wrap_script/run_pipeline.py -- wraps a neutral
FLAME template (see render_flame_neutral.py) onto one FaceScape
actor/expression pair's own scanned mesh, using sapiens2 landmarks
triangulated across every camera view available for that pair.

Deliberately much simpler than the Nersemble pipeline:
  - No per-actor FLAME parameters (no fit.npz) -- there is no per-frame FLAME
    render, so unlike Nersemble's "own render vs. chained wrap" obj_path
    choice, here it's "generic template vs. chained wrap": every expression
    OTHER than this actor's own neutral (per-actor, read from
    neutral_frame.json -- see facescape_neutral.load_actor_neutral_expression)
    chains off that neutral expression's own wrapped_mesh.obj once it exists
    (falling back to the generic DEFAULT_NEUTRAL_MESH_PATH otherwise); the
    neutral expression itself always wraps from the generic template, since
    it has nothing of this actor's own to chain from yet. An actor with no
    neutral_frame.json saved yet has no known neutral to chain off of, or to
    tell apart from a non-neutral expression, so run_pipeline() raises rather
    than guessing "1_neutral". See run_pipeline()'s own neutral_mesh_path
    selection below.
  - No face_ray_masking -- the whole template mesh is deformed, no polygon
    restriction (see template.wrap / run_wrap_script.py's own notes).
  - No extract_frames/generate_sfm/meshroom_compute -- FaceScape already has
    a real scanned mesh per actor/expression
    (facescape_shape_root/<actor>/<expression>.ply), used directly as the
    wrap target instead of a meshroom reconstruction.
  - Single-stage wrap only (no blendshape-retargeting refinement pass -- that
    stage in Nersemble's pipeline refines toward "this frame's own FLAME
    expression", which doesn't exist as a concept here).

So the only two real steps are: (1) triangulate sapiens2 landmarks to 3D
using every camera FaceScape has for this pair, and (2) wrap_mesh().

Usage:
    python3 run_pipeline.py <ACTOR> <EXPRESSION>
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import trimesh
import facescape_paths as fp  # all paths: wrap_script_facescape/.env

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
# FaceView's own root has an unrelated triangulate.py, and wrap_script/ has its
# own drifted mirror of backend/ (missing the FaceScape settings added to the
# real one) -- both would silently shadow what we actually want via a plain
# sys.path-based import, so wrap_script/triangulate.py is loaded by explicit
# file path instead, and only FaceView's root goes on sys.path for
# `backend.app.core.config` (the real one, with facescape_images_root etc.).
fp.add_faceview_to_sys_path()  # backend.app.core.config

_triangulate_spec = importlib.util.spec_from_file_location(
    "wrap_script_triangulate", str(fp.WRAP_SCRIPT_ROOT / "triangulate.py")
)
_triangulate_module = importlib.util.module_from_spec(_triangulate_spec)
_triangulate_spec.loader.exec_module(_triangulate_module)
triangulate = _triangulate_module.triangulate
is_in_bounds = _triangulate_module.is_in_bounds

# Same explicit-file-path import trick as above, and as run_neutral_skin_propagation.py's
# own use of skin_landmarks.py -- reused here (not duplicated) since it's the
# single source of truth for which FLAME vertices count as "skin" landmarks.
_skin_landmarks_spec = importlib.util.spec_from_file_location(
    "wrap_script_skin_landmarks", str(fp.WRAP_SCRIPT_ROOT / "skin_landmarks.py")
)
_skin_landmarks_module = importlib.util.module_from_spec(_skin_landmarks_spec)
_skin_landmarks_spec.loader.exec_module(_skin_landmarks_module)
assign_global_indices = _skin_landmarks_module.assign_global_indices
NUM_STANDARD_LANDMARKS = _skin_landmarks_module.NUM_STANDARD_LANDMARKS

# Same explicit-file-path import trick -- propagated_landmarks.py merges each
# camera's propagated ear/nosebridge/undernose/eyebrow/chin_frame_000000.json
# (run_neutral_skin_propagation.py) over sapiens2's frame_000000.json.
_propagated_landmarks_spec = importlib.util.spec_from_file_location(
    "wrap_script_propagated_landmarks", str(fp.WRAP_SCRIPT_ROOT / "propagated_landmarks.py")
)
_propagated_landmarks_module = importlib.util.module_from_spec(_propagated_landmarks_spec)
_propagated_landmarks_spec.loader.exec_module(_propagated_landmarks_module)
load_camera_standard_points = _propagated_landmarks_module.load_camera_standard_points
load_camera_skin_points = _propagated_landmarks_module.load_camera_skin_points

from render_overlay_frame import render_overlay_frame  # noqa: E402
from run_wrap_script import wrap_mesh, DEFAULT_NEUTRAL_MESH_PATH, DEFAULT_POT_PATH  # noqa: E402
import face_scale  # noqa: E402  (face-based target scaling for template wraps)
from backend.app.core.config import get_settings  # noqa: E402
from backend.app.services import facescape_neutral  # noqa: E402
from backend.app.services.label_tracker import label_tracker  # noqa: E402

NUM_VERTICES = 74  # standard FLAME landmark count -- matches neutral_correspondence_without_eyes_POT.json's own length

# Extended (74 standard + N skin) version of DEFAULT_POT_PATH, same idea as
# wrap_script/run_pipeline_all.py's EXTENDED_POT_PATH (Nersemble) -- generated
# once and cached here, never mutating DEFAULT_POT_PATH itself. FaceScape's
# wrap_mesh() has no whitelist.json-based restriction to extend on top of it
# (see run_wrap_script.py's own docstring for why), so unlike the Nersemble
# version this is the only extended file needed.
EXTENDED_POT_PATH = SCRIPT_DIR / "neutral_correspondence_without_eyes_POT_all.json"


def load_skin_vertex_indices() -> list[int]:
    """Wrapped-mesh vertex index for each global skin landmark index (74, 75,
    ...), in that same global-index order -- see skin_landmarks.assign_global_indices.
    Same helper as wrap_script/run_pipeline_all.py's own (Nersemble) version."""
    return [vertex_idx for _global_idx, _region, vertex_idx in assign_global_indices()]


def _compute_skin_pot_entries(topology_mesh_path: str, skin_vertex_indices: list[int]) -> list[list[float]]:
    """Point-on-triangle (face_index, u, w) entries for each skin vertex index, in
    the same format as DEFAULT_POT_PATH's existing 74 entries. Duplicated from
    wrap_script/run_pipeline_all.py's own version rather than imported -- that
    module pulls in cv2/pytorch3d/meshroom/SAM-U2Net at import time, which would
    defeat this script's own "deliberately much simpler" design (see module
    docstring). Convention: a face's 0th vertex is (1,0), 1st is (0,1), 2nd is
    (0,0) -- see the Nersemble version's own docstring for how this was inferred."""
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


def ensure_extended_correspondence_files(topology_mesh_path: str, skin_vertex_indices: list[int]) -> str:
    """Idempotently builds (if missing or stale) and returns the path to the
    extended POT template file that includes the skin indices on top of the
    standard 74 -- FaceScape counterpart to wrap_script/run_pipeline_all.py's
    ensure_extended_correspondence_files(), minus the whitelist half (not
    needed here, see EXTENDED_POT_PATH's own comment above)."""
    need_rebuild = True
    if EXTENDED_POT_PATH.exists():
        with EXTENDED_POT_PATH.open("r", encoding="utf-8") as f:
            existing = json.load(f)
        need_rebuild = len(existing) != NUM_STANDARD_LANDMARKS + len(skin_vertex_indices)

    if need_rebuild:
        with open(DEFAULT_POT_PATH, "r") as f:
            base_pot = json.load(f)
        skin_pot = _compute_skin_pot_entries(topology_mesh_path, skin_vertex_indices)
        with EXTENDED_POT_PATH.open("w", encoding="utf-8") as f:
            json.dump(base_pot + skin_pot, f)
        print(f"Rebuilt {EXTENDED_POT_PATH.name} ({NUM_STANDARD_LANDMARKS} standard + "
              f"{len(skin_vertex_indices)} skin indices)")

    return str(EXTENDED_POT_PATH)

# Cameras the Overlay Frame tab's keyline renders always come from -- see
# render_overlay_frame() below and FaceView/backend/app/api/routes/facescape.py's
# /overlay-frame endpoint (one render per camera_id, served by its own URL),
# which serves exactly these files. Matches Nersemble's equivalent
# OVERLAY_VIDEO_CAMERA_ID pattern in run_pipeline_batch_all.py/labels.py, just
# with more than one camera.
OVERLAY_FRAME_CAMERA_IDS = ["1", "4", "17"]


def run_pipeline(actor_id: str, expression_id: str, exclude_cameras: set[str] | None = None) -> Path:
    settings = get_settings()
    expr_dir = settings.facescape_images_root / actor_id / expression_id
    params_path = expr_dir / "params.json"
    mesh_path = settings.facescape_shape_root / actor_id / f"{expression_id}.ply"
    landmarks_dir = settings.facescape_landmarks_root / actor_id / expression_id

    if not params_path.exists():
        raise FileNotFoundError(f"params.json not found: {params_path} (run Phase 1's coordinate transform first)")
    if not mesh_path.exists():
        raise FileNotFoundError(f"Scanned mesh not found: {mesh_path}")

    with open(params_path) as f:
        params = json.load(f)

    # Every camera view given for this actor/expression pair -- all cameras
    # present in its own params.json, not just the culled labeling subset
    # (more views only helps triangulation; a camera with no landmark file
    # for this pair just contributes nothing below, harmlessly).
    camera_ids = sorted({k.rsplit("_", 1)[0] for k in params if k.endswith("_Rt")}, key=int)
    if exclude_cameras:
        excluded_present = [c for c in exclude_cameras if c in camera_ids]
        camera_ids = [c for c in camera_ids if c not in exclude_cameras]
        if excluded_present:
            print(f"Excluding camera(s) by request: {excluded_present}")

    intrinsics: list[np.ndarray] = []
    extrinsics: list[np.ndarray] = []
    widths: list[int] = []
    heights: list[int] = []
    cam_index_by_id: dict[str, int] = {}
    for cam_id in camera_ids:
        k_key, rt_key = f"{cam_id}_K", f"{cam_id}_Rt"
        if k_key not in params:
            continue
        cam_index_by_id[cam_id] = len(intrinsics)
        intrinsics.append(np.array(params[k_key], dtype=np.float64))
        rt = np.array(params[rt_key], dtype=np.float64)
        ext = np.eye(4, dtype=np.float64)
        ext[:3, :4] = rt
        extrinsics.append(ext)
        widths.append(int(params.get(f"{cam_id}_width", 0) or 0))
        heights.append(int(params.get(f"{cam_id}_height", 0) or 0))

    if not intrinsics:
        raise RuntimeError(f"No cameras found in {params_path}")

    # Skin points (index >= 74), when run_neutral_skin_propagation.py has
    # already propagated them for this actor/expression -- same idea as
    # wrap_script/run_pipeline_all.py's skin_frame_<idx>.json handling, just
    # one static frame (skin_frame_000000.json) per camera instead of one per
    # video frame. skin_vertex_indices is the fixed, always-present list of
    # which FLAME vertices count as skin landmarks (from skin_landmarks.py) --
    # NOT conditional on propagation having actually happened yet, unlike the
    # per-pair skin_frame_000000.json files themselves.
    skin_vertex_indices = load_skin_vertex_indices()
    num_vertices = NUM_STANDARD_LANDMARKS + len(skin_vertex_indices)

    points_list: list[list[tuple[int, np.ndarray, bool]]] = [[] for _ in range(num_vertices)]
    cameras_used = 0
    for cam_id, cam_index in cam_index_by_id.items():
        width, height = widths[cam_index], heights[cam_index]
        cam_dir = landmarks_dir / f"cam_{cam_id}"
        # Standard points: sapiens2's frame_000000.json with every propagated
        # index (ear/nosebridge/undernose/eyebrow/chin_frame_000000.json)
        # replaced by its propagated point -- see propagated_landmarks.py.
        standard_points = load_camera_standard_points(cam_dir, "000000")
        skin_points = load_camera_skin_points(cam_dir, "000000")
        if not width or not height or (standard_points is None and skin_points is None):
            continue
        cameras_used += 1
        for point_data in standard_points or []:
            idx = point_data["index"]
            if idx >= NUM_VERTICES:
                continue
            pixel = np.array([round(point_data["x"] * width), round(point_data["y"] * height)])
            points_list[idx].append((cam_index, pixel, is_in_bounds(point_data)))

        if skin_points is not None:
            for point_data in skin_points:
                idx = point_data["index"]
                if idx >= num_vertices:
                    continue
                pixel = np.array([round(point_data["x"] * width), round(point_data["y"] * height)])
                points_list[idx].append((cam_index, pixel, is_in_bounds(point_data)))
        elif skin_vertex_indices:
            print(f"NOTE: {cam_dir / 'skin_frame_000000.json'} missing -- run_neutral_skin_propagation.py hasn't "
                  f"propagated skin points for cam={cam_id} yet")

    # Same graceful degrade as wrap_script/run_pipeline_all.py: triangulate.py's
    # triangulate_point() needs >=2 camera observations per point, which is the
    # normal state for every skin index at once when this actor/expression simply
    # hasn't been through run_neutral_skin_propagation.py yet. Rather than error
    # (or silently corrupt the wrap), degrade to exactly this script's prior
    # standard-only (74-point) behavior for this pair.
    if skin_vertex_indices:
        insufficient = [idx for idx in range(NUM_STANDARD_LANDMARKS, num_vertices) if len(points_list[idx]) < 2]
        if insufficient:
            print(f"NOTE: {len(insufficient)}/{len(skin_vertex_indices)} skin point(s) have fewer than 2 camera "
                  f"observations (probably not propagated yet for {actor_id}/{expression_id}) -- "
                  "falling back to standard-only (no skin points).")
            points_list = points_list[:NUM_STANDARD_LANDMARKS]
            skin_vertex_indices = []

    print(f"Triangulating actor={actor_id} expression={expression_id} using {cameras_used} camera(s) with landmark data"
          + (f" ({len(skin_vertex_indices)} skin point(s) included)" if skin_vertex_indices else ""))

    # Per-camera intrinsics AND per-camera resolution (unlike Nersemble's
    # single shared rig intrinsic/resolution -- see triangulate.py's
    # _intrinsic_for/_resolution_for). Passing widths/heights here enables
    # leave-one-out outlier rejection: a camera whose landmark disagrees
    # geometrically with every other camera (e.g. a bystander in frame
    # throwing off sapiens2's detection) gets dropped for that point instead
    # of silently corrupting the triangulated result.
    formatted, untriangulated_indices = triangulate(np.array(intrinsics), extrinsics, points_list, widths, heights)
    if untriangulated_indices:
        print(f"NOTE: {len(untriangulated_indices)} point(s) could not be triangulated (no in-bounds view in any "
              f"camera) -- excluded from the correspondence map: {untriangulated_indices}")

    output_dir = settings.facescape_output_root / actor_id / expression_id
    output_dir.mkdir(parents=True, exist_ok=True)

    correspondence_path = output_dir / "keypoint_vertices.json"
    with open(correspondence_path, "w") as f:
        json.dump(formatted, f)

    # Wrap's LoadGeom is only exercised via .obj elsewhere in this codebase --
    # convert the scanned .ply once per pair rather than relying on untested
    # native .ply support.
    scan_obj_path = output_dir / "scan.obj"
    mesh = trimesh.load(str(mesh_path), process=False)
    mesh.export(str(scan_obj_path))

    # Extended POT template (74 standard + N skin) is always built against
    # DEFAULT_NEUTRAL_MESH_PATH regardless of which neutral mesh actually gets
    # wrapped below -- it only needs face TOPOLOGY (which face a vertex sits
    # on), not vertex positions, and 1_neutral's own wrapped_mesh.obj shares
    # the exact same topology as the raw template (it's a deformation of it,
    # not a re-triangulation), so either source gives identical POT entries.
    pot_template_path = None
    if skin_vertex_indices:
        pot_template_path = ensure_extended_correspondence_files(str(DEFAULT_NEUTRAL_MESH_PATH), skin_vertex_indices)

    # Same neutral-frame chaining as wrap_script/run_pipeline_all.py: every
    # expression OTHER than this actor's own neutral wraps from THIS actor's
    # own neutral wrapped_mesh.obj when it exists (a much better per-actor
    # starting point than the generic template, since it already captures
    # this actor's own identity/shape), falling back to the generic
    # DEFAULT_NEUTRAL_MESH_PATH when it doesn't exist yet -- same fallback
    # semantics as Nersemble's "neutral_frame_id is None or frame_id ==
    # neutral_frame_id" case. The neutral expression itself always wraps from
    # the generic template (it has nothing to chain from -- it *is* the
    # chain's root). run_pipeline_batch.py is responsible for running the
    # neutral pair first so this fallback is the exception rather than the
    # rule in practice.
    #
    # Which expression counts as "neutral" is per-actor, read from
    # facescape_landmarks_root/<actor_id>/neutral_frame.json (see
    # facescape_neutral.load_actor_neutral_expression), NOT a hardcoded
    # "1_neutral" -- an actor whose reviewer picked a different expression as
    # neutral chains off THAT wrapped mesh instead. An actor with no
    # neutral_frame.json saved yet has no known neutral to chain off of, or
    # to tell apart from a non-neutral expression, so this raises rather than
    # silently guessing "1_neutral" (run_pipeline_batch.py already gates on
    # this before ever invoking run_pipeline(), so this should only fire for
    # a direct/manual invocation against such an actor).
    neutral_expression_id = facescape_neutral.load_actor_neutral_expression(actor_id)
    if neutral_expression_id is None:
        raise RuntimeError(
            f"No neutral_frame.json saved for actor {actor_id} yet (see "
            "facescape_neutral.save_actor_neutral_expression / the frontend's \"Save as Neutral\" "
            "button) -- nothing to chain non-neutral expressions off of, or to tell this expression "
            "apart from the neutral one."
        )

    neutral_wrapped_path = settings.facescape_output_root / actor_id / neutral_expression_id / "wrapped_mesh.obj"
    if expression_id != neutral_expression_id and neutral_wrapped_path.exists():
        neutral_mesh_path = neutral_wrapped_path
    else:
        neutral_mesh_path = DEFAULT_NEUTRAL_MESH_PATH

    output_wrap_path = output_dir / "wrapped_mesh.obj"
    wrap_kwargs = dict(
        target_mesh_path=str(scan_obj_path),
        target_correspondence_path=str(correspondence_path),
        output_path=str(output_wrap_path),
        extra_indices_to_remove=untriangulated_indices,
        neutral_mesh_path=str(neutral_mesh_path),
        **({"pot_template_path": pot_template_path} if pot_template_path else {}),
    )
    if Path(neutral_mesh_path) == Path(DEFAULT_NEUTRAL_MESH_PATH):
        # Wrapping from the generic template: scale the target so its face
        # matches the template first ("P5", face_scale.py), result back in world space.
        face_scale.wrap_with_face_scale(wrap_mesh, **wrap_kwargs)
    else:
        (output_dir / "face_scale.json").unlink(missing_ok=True)  # chained from the actor's neutral wrap: no scaling
        wrap_mesh(**wrap_kwargs)
    print(f"Saved wrapped mesh to: {output_wrap_path}")

    # Same "unreviewed -> unconfirmed" automatic transition as Nersemble's
    # run_pipeline_batch_all.py's mark_unconfirmed() -- a no-op if this pair
    # isn't currently "unreviewed" (e.g. a rerun on a pair already unconfirmed
    # or confirmed), so reruns can't silently undo a human's later decision.
    label_tracker.mark_unconfirmed(actor_id, expression_id, dataset="FaceScape")

    # Best-effort, same reasoning as run_pipeline_batch_all.py's
    # render_overlay_video(): the pair is already wrapped and its status
    # already moved to "unconfirmed" by this point, so a rendering hiccup
    # here shouldn't be treated as the whole pipeline failing.
    for camera_id in OVERLAY_FRAME_CAMERA_IDS:
        try:
            render_overlay_frame(actor_id, expression_id, camera_id)
        except Exception as exc:
            print(f"WARNING: overlay frame render failed for {actor_id}/{expression_id} camera {camera_id}: {exc}")

    # Landmark point overlay (annotated vs. wrapped-mesh landmarks) for the
    # frontend's Overlay Landmarks tab -- same cameras, same best-effort policy.
    # Imported here so a problem in the renderer can never block a wrap.
    try:
        from render_point_overlay_frame import render_point_overlay_frame
    except Exception as exc:
        print(f"WARNING: point overlay renderer unavailable: {exc}")
    else:
        for camera_id in OVERLAY_FRAME_CAMERA_IDS:
            try:
                render_point_overlay_frame(actor_id, expression_id, camera_id)
            except Exception as exc:
                print(f"WARNING: point overlay frame render failed for {actor_id}/{expression_id} camera {camera_id}: {exc}")

    return output_wrap_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("actor_id")
    parser.add_argument("expression_id")
    parser.add_argument("--exclude-cameras", nargs="*", default=[], help="Camera ids to leave out of triangulation entirely, e.g. --exclude-cameras 30")
    args = parser.parse_args()
    run_pipeline(args.actor_id, args.expression_id, exclude_cameras=set(args.exclude_cameras))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
