#!/usr/bin/env python3
"""Face mask on a frame's Meshroom SCAN, and the scan-based landmark filter --
Nersemble's counterpart to wrap_script_ava256/face_ray_masking_ava256_mesh.py
(scan mask) + run_pipeline._filter_by_scan_mask (landmark filter).

Scan mask (compute_scan_face_mask):
  - SAM, per FILTER camera (face_ray_masking_wrapping_test.NERSEMBLE_FRONT_CAMERAS
    plus FILTER_SIDE_CAMERAS, the two cameras facing the ears, that have a
    video), on the frame's raw image at half resolution, with the
    frame's triangulated SAM_PROMPT_INDICES landmarks (nosebridge, eyebrow
    ends, lip corners, ears -- whichever were triangulated) reprojected into
    that camera as positive point prompts -- same as Ava-256. Largest connected component
    kept, upsampled back to full resolution.
  - The scan (MESHROOM_DATA_ROOT/<actor>/<sequence>/frame_<raw>/mesh.obj.gz,
    world space) is rasterized into each of those cameras (nvdiffrast). A face
    counts as seen by a camera when it owns at least one pixel of that
    camera's depth-tested raster (i.e. not fully occluded), and as covered
    when one of those pixels is inside the camera's SAM mask. Kept: faces
    covered for a strict majority (> AGREEMENT_FRACTION) of the cameras that
    see them, or by at least MIN_AGREEING_CAMERAS of them; then one boundary
    layer is eroded. Same vote/erosion as Ava-256 (whose angle limit is
    disabled, 180 deg, so occlusion is its only visibility test too -- here
    the z-buffer does that instead of Ava-256's per-face ray casts, which are
    far too slow on a ~240k-face scan without embree).
  - Unlike Ava-256's tracked kinematic mesh, every Meshroom scan has its own
    topology, so the mask is per frame: OUTPUT_ROOT/<actor>/<sequence>/<raw>/
    scan_face_mask_included.json.

Landmark filter (filter_frame_landmarks): every landmark of the frame --
the 74 standard ones AND the skin ones -- is triangulated exactly as
run_pipeline_all.process_frame() does (same default cameras, same
propagated/sapiens2 merge, same triangulate()), then checked in 2D against the
same filter cameras' SAM masks (landmark_sam_votes):
  - a camera only votes when the landmark is on-image and NOT occluded: the
    scan's own depth at the landmark's pixel (depth-tested raster) must not be
    more than OCCLUSION_TOLERANCE_M in front of the landmark;
  - it votes "inside" when the landmark is inside the SAM mask or within
    MASK_MARGIN_PX of it (0: only points on the mask itself count as inside);
  - kept when inside for a strict majority (> AGREEMENT_FRACTION) of the
    voting cameras, or for at least MIN_AGREEING_CAMERAS of them; a landmark
    that is occluded (or off-image) in every camera is dropped -- e.g. ear
    points behind hair.
NEVER_FILTERED_INDICES (eyelids, lips, nosebridge, undernose) are always kept.

Like the FLAME face mask, the filter is computed once per actor, on its
neutral frame, and applied to every frame of the actor:
run_pipeline_all.process_frame() drops load_actor_neutral_dropped_indices()
from every frame's correspondence map (landmark indices are the same in every
frame). Per-frame filtered JSONs (--mask-only --all-frames / --frame) are for
inspection only.
Per-expression masks: every other sequence of the actor can have its own FLAME
face mask + landmark filter, computed at that sequence's frame closest to the
neutral (run_neutral_cotracker_propagation.ensure_expression_mask), stored in
expression_mask_dir(); run_pipeline_all.process_frame() uses them for that
sequence instead of the actor's when they exist (hairstyles change between
sequences, so an ear visible on the neutral frame can be under hair elsewhere).
The scan mask above is still computed and written, but no longer decides
which landmarks are kept (a single nearest scan face at the cheek/jaw
silhouette is seen by only 0-3 cameras and got eroded, dropping points that
are well inside SAM in 2D). The result is written to
filtered_landmarks_path() (landmarks_root/<actor>/<sequence>/scan_filtered/
frame_<idx:06d>.json); run_pipeline_all.run_pipeline() removes its "dropped"
indices from the correspondence map (load_scan_dropped_indices()).

Imports: nothing GPU-related at module level -- run_pipeline_all.py (which
hides the GPU) imports load_scan_dropped_indices() from here.
"""
from __future__ import annotations

import nersemble_paths
import gzip
import json
import os
from pathlib import Path

import numpy as np

MESHROOM_DATA_ROOT = nersemble_paths.MESHROOM_DATA_ROOT
OUTPUT_ROOT = nersemble_paths.OUTPUT_ROOT
SCAN_FACE_MASK_NAME = "scan_face_mask_included.json"
FILTERED_DIR_NAME = "scan_filtered"
RAW_FRAME_STRIDE = 5
NOSEBRIDGE_INDEX = 57  # face_ray_masking_wrapping_test.SAM_PROMPT_LANDMARK_INDEX
# SAM's positive point prompts (same as wrap_script_ava256/face_ray_masking_ava256_mesh.py):
# nosebridge (57), outer eyebrow ends (30 left, 47 right), lip corners (18
# left, 37 right) and ears (1, 4, 6, 14; ear tops 2/9 excluded, as on
# Ava-256). Each is used only when it was triangulated for this frame; the
# nosebridge is required.
SAM_PROMPT_INDICES = (57, 30, 47, 18, 37, 1, 4, 6, 14)
# Eyebrow and ear prompts are occlusion-tested per camera (prompt_visibility):
# a camera only gets them when the occluder surface (the frame's scan, so hair
# counts) at their pixel is not more than OCCLUSION_TOLERANCE_M in front of
# them -- the landmark filter's own rule. A hidden ear's point lands on hair,
# which would make SAM take the hair. Nosebridge / lip corners always go in.
OCCLUSION_GATED_PROMPT_INDICES = frozenset((30, 47, 1, 4, 6, 14))
AGREEMENT_FRACTION = 0.5  # wrap_script_ava256/face_ray_masking_ava256.py
MIN_AGREEING_CAMERAS = 3
MASK_EROSION_ITERATIONS = 1
NUM_STANDARD_LANDMARKS = 74
# 2D landmark test: a landmark within this many full-resolution pixels of a
# camera's SAM mask counts as inside it. Was 30 (~4.5 mm on the face), which
# let ear points on hair just beside the face pass; 0 = on the mask itself.
MASK_MARGIN_PX = 0
# Cameras facing the left / right ear, added to the front cameras for the SAM
# masks of the scan mask and the landmark filter, so ear points get a vote from
# a camera that actually sees the ear.
FILTER_SIDE_CAMERAS = ("222200044", "221501007")
# Per-expression mask + landmark filter of a non-neutral sequence
# (expression_mask_dir()): FLAME mask files face_list.txt + FACE_MASK_SUFFIXES,
# EXPRESSION_FILTER_NAME (filter_frame_landmarks payload) and
# EXPRESSION_INFO_NAME (which frame they were made from).
EXPRESSION_MASK_DIR_NAME = "expression_mask"
EXPRESSION_FILTER_NAME = "landmark_filter.json"
EXPRESSION_INFO_NAME = "info.json"
# A camera sees a landmark unless the scan surface at its pixel is more than
# this far in front of it (landmarks sit ~1-7 mm off the scan).
OCCLUSION_TOLERANCE_M = 0.01
# Never dropped by the filter, wherever they land: eyelids + lips (sapiens2's
# precise indices, reproject_side_cameras.EYE_/LIP_LANDMARK_INDICES) and the
# nosebridge + undernose groups (run_neutral_cotracker_propagation.
# STANDARD_INDEX_REGION_GROUPS). E.g. SAM leaves glasses lenses out of the face
# mask, which would otherwise drop every eyelid point of an actor wearing them.
EYELID_INDICES = frozenset({11, 13, 15, 16, 17, 32, 33, 34, 35, 36, 65, 71})
LIP_INDICES = frozenset({18, 20, 21, 23, 24, 25, 26, 27, 37, 39, 40, 42, 43, 44, 45, 46, 55, 56, 58, 59})
NOSEBRIDGE_INDICES = frozenset({54, 57})
UNDERNOSE_INDICES = frozenset({19, 38, 61})
NEVER_FILTERED_INDICES = EYELID_INDICES | LIP_INDICES | NOSEBRIDGE_INDICES | UNDERNOSE_INDICES

# Meshroom scans are stored upside down relative to camera_params.json's world
# frame: a 180 deg rotation about X (same as wrap_script_ava256/run_pipeline.py's
# FLIP_X) puts them in it (070/EXP-2-eyes/0: landmarks 58.9 mm -> 1.7 mm from
# the scan). Applied by load_scan(), so every scan here is in world space.
SCAN_TO_WORLD = np.diag([1.0, -1.0, -1.0])


# --- paths / readers (torch-free) -------------------------------------------

def scan_gz_path(actor_id: str, sequence_id: str, raw_frame: int) -> Path:
    return MESHROOM_DATA_ROOT / actor_id / sequence_id / f"frame_{raw_frame}" / "mesh.obj.gz"


def scan_mask_path(actor_id: str, sequence_id: str, raw_frame: int) -> Path:
    return OUTPUT_ROOT / actor_id / sequence_id / str(raw_frame) / SCAN_FACE_MASK_NAME


def filtered_landmarks_path(landmarks_root: Path, actor_id: str, sequence_id: str, raw_frame: int) -> Path:
    return Path(landmarks_root) / actor_id / sequence_id / FILTERED_DIR_NAME / f"frame_{raw_frame // RAW_FRAME_STRIDE:06d}.json"


def load_scan_dropped_indices(landmarks_root: Path, actor_id: str, sequence_id: str, raw_frame: int) -> list[int] | None:
    """The filter's dropped landmark indices for this frame, or None when the
    frame has no filtered JSON (not masked yet)."""
    path = filtered_landmarks_path(landmarks_root, actor_id, sequence_id, raw_frame)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return sorted(int(i) for i in json.load(f)["dropped"])


def expression_mask_dir(actor_id: str, sequence_id: str) -> Path:
    """OUTPUT_ROOT/<actor>/<sequence>/expression_mask -- the sequence's own FLAME
    face mask (face_list.txt*) and landmark filter, see ensure_expression_mask."""
    return OUTPUT_ROOT / actor_id / sequence_id / EXPRESSION_MASK_DIR_NAME


def load_expression_dropped_indices(actor_id: str, sequence_id: str) -> tuple[list[int] | None, Path]:
    """(dropped indices, path) of the sequence's own landmark filter; indices are
    None when the sequence has none (its frames then use the actor's)."""
    path = expression_mask_dir(actor_id, sequence_id) / EXPRESSION_FILTER_NAME
    if not path.exists():
        return None, path
    with open(path, "r", encoding="utf-8") as f:
        return sorted(int(i) for i in json.load(f)["dropped"]), path


def load_actor_neutral_dropped_indices(landmarks_root: Path, actor_id: str) -> tuple[list[int] | None, Path]:
    """(dropped indices, filtered-JSON path) of the actor's neutral frame
    (landmarks_root/<actor>/neutral_frame.json) -- what every frame of the
    actor leaves out. Indices are None when the actor has no neutral frame or
    its neutral frame hasn't been filtered yet."""
    neutral_path = Path(landmarks_root) / actor_id / "neutral_frame.json"
    if not neutral_path.exists():
        return None, neutral_path
    with open(neutral_path, "r", encoding="utf-8") as f:
        neutral = json.load(f)
    path = filtered_landmarks_path(landmarks_root, actor_id, neutral["sequence_id"], int(neutral["frame_idx"]))
    return load_scan_dropped_indices(landmarks_root, actor_id, neutral["sequence_id"], int(neutral["frame_idx"])), path


def load_scan(actor_id: str, sequence_id: str, raw_frame: int) -> tuple[np.ndarray, np.ndarray]:
    """(verts, faces) of the frame's Meshroom scan, in world space (SCAN_TO_WORLD
    applied; a proper rotation, so face winding is unchanged). Plain OBJ
    parse (v / f lines only, polygons fanned into triangles) -- much faster
    than trimesh on a 10 MB OBJ and keeps face indices in file order."""
    path = scan_gz_path(actor_id, sequence_id, raw_frame)
    if not path.exists():
        raise FileNotFoundError(f"no Meshroom scan at {path}")
    verts, faces = [], []
    with gzip.open(path, "rt") as f:
        for line in f:
            if line.startswith("v "):
                verts.append(line.split()[1:4])
            elif line.startswith("f "):
                ids = [int(tok.split("/")[0]) - 1 for tok in line.split()[1:]]
                for k in range(1, len(ids) - 1):
                    faces.append((ids[0], ids[k], ids[k + 1]))
    return np.asarray(verts, dtype=np.float64) @ SCAN_TO_WORLD.T, np.asarray(faces, dtype=np.int64)


# --- landmarks (same triangulation as run_pipeline_all.process_frame) --------

def triangulate_frame_landmarks(
    actor_id: str, sequence_id: str, raw_frame: int, default_cameras: list[str], num_skin: int, settings,
) -> tuple[dict[int, np.ndarray], list[int]]:
    """({index: world xyz}, untriangulated indices) for every standard (0-73)
    and skin (74..74+num_skin-1) landmark of the frame. Mirrors
    run_pipeline_all.process_frame()'s camera loop and triangulate() call,
    including its fallback to standard-only when any skin point has fewer
    than 2 observations."""
    import cv2
    from propagated_landmarks import load_camera_skin_points, load_camera_standard_points
    from triangulate import is_in_bounds, triangulate

    root = Path(settings.landmarks_root) / actor_id / sequence_id
    cap = cv2.VideoCapture(str(Path(settings.data_root) / actor_id / "sequences" / sequence_id / "images" / "cam_222200037.mp4"))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    with open(Path(settings.data_root) / actor_id / "calibration" / "camera_params.json", "r") as f:
        cam_data = json.load(f)

    num_points = NUM_STANDARD_LANDMARKS + num_skin
    points_list = [[] for _ in range(num_points)]
    extrinsics = []
    tag = f"{raw_frame // RAW_FRAME_STRIDE:06d}"
    for cam in default_cameras:
        cam_dir = root / f"cam_{cam}"
        standard = load_camera_standard_points(cam_dir, tag)
        skin = load_camera_skin_points(cam_dir, tag)
        if standard is None and skin is None:
            continue
        cam_index = len(extrinsics)
        extrinsics.append(cam_data["world_2_cam"][cam])
        for p in (standard or []) + (skin or []):
            if p["index"] < num_points:
                points_list[p["index"]].append(
                    (cam_index, np.array([round(p["x"] * width), round(p["y"] * height)]), is_in_bounds(p)))
    if not extrinsics:
        raise RuntimeError(f"no landmark files for {actor_id}/{sequence_id} frame {raw_frame}")
    if num_skin and any(len(points_list[i]) < 2 for i in range(NUM_STANDARD_LANDMARKS, num_points)):
        print(f"NOTE: {actor_id}/{sequence_id}/{raw_frame}: skin points under-observed -- standard-only, "
              "same fallback as run_pipeline_all")
        points_list = points_list[:NUM_STANDARD_LANDMARKS]

    formatted, untriangulated = triangulate(cam_data["intrinsics"], np.array(extrinsics), points_list, width, height)
    missing = set(untriangulated)
    xyz = {i: np.array([p["x"], p["y"], p["z"]], dtype=np.float64) for i, p in enumerate(formatted) if i not in missing}
    return xyz, sorted(missing)


# --- scan mask ----------------------------------------------------------------

def load_camera_images(actor_id: str, sequence_id: str, raw_frame: int, camera_ids, settings) -> dict[str, np.ndarray]:
    """{camera: BGR raw frame} for the cameras that have a readable video."""
    import cv2

    images = {}
    images_dir = Path(settings.data_root) / actor_id / "sequences" / sequence_id / "images"
    for cam in sorted(camera_ids):
        mp4 = images_dir / f"cam_{cam}.mp4"
        if not mp4.exists():
            continue
        cap = cv2.VideoCapture(str(mp4), cv2.CAP_FFMPEG)
        if raw_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, raw_frame)
        ok, frame = cap.read()
        cap.release()
        if ok:
            images[cam] = frame
    return images


def project(xyz: np.ndarray, K: np.ndarray, world_2_cam: np.ndarray) -> np.ndarray:
    cam = xyz @ world_2_cam[:3, :3].T + world_2_cam[:3, 3]
    uv = cam @ K.T
    return uv[:, :2] / uv[:, 2:3]


def sam_prompt_indices(xyz: dict) -> list[int]:
    """SAM_PROMPT_INDICES that were triangulated for this frame (keys of xyz)."""
    return [i for i in SAM_PROMPT_INDICES if i in xyz]


def sam_prompt_xyz(xyz: dict) -> np.ndarray:
    """(N, 3) world positions of this frame's SAM prompt landmarks."""
    return np.stack([np.asarray(xyz[i], dtype=np.float64) for i in sam_prompt_indices(xyz)])


def pot_row_vertex(row, faces: np.ndarray) -> int:
    """Vertex index a POT row [face, u, w] pins to (run_pipeline_all._pot_entry_vertex_index)."""
    face_idx, u, w = int(row[0]), float(row[1]), float(row[2])
    face = faces[face_idx]
    if u == 1.0:
        return int(face[0])
    if w == 1.0:
        return int(face[1])
    if u == 0.0 and w == 0.0:
        return int(face[2])
    return int(face[int(np.argmax((1.0 - u - w, u, w)))])


def prompt_visibility(prompt: dict[int, np.ndarray], K: np.ndarray, world_2_cam: dict[str, np.ndarray],
                      image_sizes: dict[str, tuple[int, int]], occ_verts: np.ndarray, occ_faces: np.ndarray,
                      ) -> dict[str, list[int]]:
    """camera -> the prompt indices (keys of prompt, {index: world xyz}) that
    camera may use: OCCLUSION_GATED_PROMPT_INDICES only when not occluded by
    the (occ_verts, occ_faces) surface by more than OCCLUSION_TOLERANCE_M at
    their pixel, every other index always. image_sizes: camera -> (h, w)."""
    out = {}
    for cam, (h, w) in image_sizes.items():
        if cam not in world_2_cam:
            continue
        _faces, depth = rasterize(occ_verts, occ_faces, K, world_2_cam[cam], w, h)
        keep = []
        for i, p in prompt.items():
            if i in OCCLUSION_GATED_PROMPT_INDICES:
                z = (np.asarray(p) @ world_2_cam[cam][:3, :3].T + world_2_cam[cam][:3, 3])[2]
                u, v = project(np.asarray(p, dtype=np.float64)[None], K, world_2_cam[cam])[0]
                ui, vi = int(round(u)), int(round(v))
                if z <= 0 or not (0 <= ui < w and 0 <= vi < h) or depth[vi, ui] < z - OCCLUSION_TOLERANCE_M:
                    continue
            keep.append(i)
        out[cam] = keep
    return out


def compute_sam_masks(images: dict[str, np.ndarray], K: np.ndarray, world_2_cam: dict[str, np.ndarray],
                      prompt_xyz: np.ndarray, prompt_indices: list[int] | None = None,
                      occluder: tuple[np.ndarray, np.ndarray] | None = None) -> dict[str, np.ndarray]:
    """camera -> full-resolution 0/1 SAM face mask, prompted at the reprojected
    prompt landmark(s) -- one (3,) point or an (N, 3) array (sam_prompt_xyz());
    a camera uses the ones that land on its image and is left out of the vote
    when none do (Ava-256's compute_sam_masks, with Nersemble's SAM wrapper).
    With prompt_indices (the rows' landmark indices) and occluder (verts,
    faces), eyebrow / ear rows occluded in a camera are left out of its
    prompt (prompt_visibility)."""
    import cv2
    import face_ray_masking_wrapping_test as frm

    segmenter, _bg = frm._get_segmenters()
    all_anchors = np.asarray(prompt_xyz, dtype=np.float64).reshape(-1, 3)
    visible = None
    if prompt_indices is not None and occluder is not None:
        visible = prompt_visibility(dict(zip(prompt_indices, all_anchors)), K, world_2_cam,
                                    {c: im.shape[:2] for c, im in images.items()}, *occluder)
    masks = {}
    for cam, image in images.items():
        h, w = image.shape[:2]
        seg_w, seg_h = max(1, w // 2), max(1, h // 2)
        seg_input = cv2.resize(image, (seg_w, seg_h), interpolation=cv2.INTER_AREA)
        anchors = all_anchors
        if visible is not None:
            anchors = all_anchors[[k for k, i in enumerate(prompt_indices) if i in visible.get(cam, [])]]
        points = [(int(round(px * seg_w / w)), int(round(py * seg_h / h)))
                  for px, py in project(anchors, K, world_2_cam[cam])]
        points = [p for p in points if 0 <= p[0] < seg_w and 0 <= p[1] < seg_h]
        if not points:
            print(f"NOTE: SAM prompt landmarks off-image in cam {cam} -- camera left out of the scan mask vote")
            continue
        mask = segmenter.get_face_mask(seg_input, points=points).astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if n > 1:
            mask = (labels == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
        masks[cam] = (cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST) > 0).astype(np.uint8)
    return masks


_glctx = None


def rasterize(verts: np.ndarray, faces: np.ndarray, K: np.ndarray, world_2_cam: np.ndarray,
              width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """(face map, depth map), each (height, width), depth-tested, OpenCV pixel
    convention (row 0 = top). face map: visible face per pixel, -1 = none;
    depth map: camera-space z of the visible surface, inf = none."""
    import nvdiffrast.torch as dr
    import torch

    global _glctx
    if _glctx is None:
        _glctx = dr.RasterizeCudaContext()
    cam = verts @ world_2_cam[:3, :3].T + world_2_cam[:3, 3]
    z = cam[:, 2]
    uv = cam @ K.T
    x_ndc = (uv[:, 0] / z) / width * 2.0 - 1.0
    y_ndc = (uv[:, 1] / z) / height * 2.0 - 1.0  # nvdiffrast row 0 is y = -1
    near, far = 0.05, 20.0
    clip = np.stack([x_ndc * z, y_ndc * z, (far + near) / (far - near) * z - 2 * far * near / (far - near), z], axis=1)
    pos = torch.from_numpy(clip).float().cuda()[None].contiguous()
    tri = torch.from_numpy(faces.astype(np.int32)).cuda().contiguous()
    rast, _ = dr.rasterize(_glctx, pos, tri, resolution=[height, width])
    depth, _ = dr.interpolate(torch.from_numpy(z[:, None]).float().cuda()[None].contiguous(), rast, tri)
    face_map = rast[0, ..., 3].round().long().cpu().numpy() - 1  # triangle id + 1, 0 = background
    depth = depth[0, ..., 0].cpu().numpy()
    depth[face_map < 0] = np.inf
    return face_map, depth


def rasterize_face_ids(verts: np.ndarray, faces: np.ndarray, K: np.ndarray, world_2_cam: np.ndarray,
                       width: int, height: int) -> np.ndarray:
    return rasterize(verts, faces, K, world_2_cam, width, height)[0]


def landmark_sam_votes(xyz: dict[int, np.ndarray], masks: dict[str, np.ndarray], K: np.ndarray,
                       world_2_cam: dict[str, np.ndarray], verts: np.ndarray, faces: np.ndarray) -> dict[int, dict]:
    """{index: {"inside": [...], "outside": [...], "occluded": [...], "off_image": [...]}}
    (camera ids) -- the 2D test described in the module docstring."""
    import cv2

    votes = {i: {"inside": [], "outside": [], "occluded": [], "off_image": []} for i in xyz}
    indices = sorted(xyz)
    pts = np.stack([xyz[i] for i in indices])
    for cam, mask in masks.items():
        h, w = mask.shape
        W = world_2_cam[cam]
        _face_map, depth = rasterize(verts, faces, K, W, w, h)
        # Distance (px) from every pixel to the nearest SAM-mask pixel; 0 inside.
        dist_to_mask = cv2.distanceTransform((mask == 0).astype(np.uint8), cv2.DIST_L2, 5)
        z = (pts @ W[:3, :3].T + W[:3, 3])[:, 2]
        for i, (x, y), zi in zip(indices, project(pts, K, W), z):
            xi, yi = int(round(x)), int(round(y))
            if zi <= 0 or not (0 <= xi < w and 0 <= yi < h):
                votes[i]["off_image"].append(cam)
            elif depth[yi, xi] < zi - OCCLUSION_TOLERANCE_M:
                votes[i]["occluded"].append(cam)
            elif dist_to_mask[yi, xi] <= MASK_MARGIN_PX:
                votes[i]["inside"].append(cam)
            else:
                votes[i]["outside"].append(cam)
    return votes


def passes_landmark_vote(v: dict) -> bool:
    n_inside, n_seen = len(v["inside"]), len(v["inside"]) + len(v["outside"])
    if n_seen == 0:
        return False  # occluded / off-image in every camera (e.g. an ear behind hair)
    return n_inside / n_seen > AGREEMENT_FRACTION or n_inside >= MIN_AGREEING_CAMERAS


def vote_faces(verts: np.ndarray, faces: np.ndarray, masks: dict[str, np.ndarray], K: np.ndarray,
               world_2_cam: dict[str, np.ndarray], weights_out: dict | None = None) -> list[int]:
    from face_ray_masking_wrapping_test import erode_hit_faces

    num_faces = len(faces)
    seen = np.zeros(num_faces, dtype=np.int32)
    votes = np.zeros(num_faces, dtype=np.int32)
    for cam, mask in masks.items():
        h, w = mask.shape
        face_map = rasterize_face_ids(verts, faces, K, world_2_cam[cam], w, h)
        visible = np.unique(face_map[face_map >= 0])
        covered = np.unique(face_map[(face_map >= 0) & (mask > 0)])
        seen[visible] += 1
        votes[covered] += 1
    if weights_out is not None:
        weights_out["seen"], weights_out["votes"] = seen, votes
    keep = (seen > 0) & ((votes / np.maximum(seen, 1) > AGREEMENT_FRACTION) | (votes >= MIN_AGREEING_CAMERAS))
    hit = set(np.nonzero(keep)[0].tolist())
    print(f"scan faces inside the SAM mask of > {AGREEMENT_FRACTION:.0%} (or >= {MIN_AGREEING_CAMERAS}) of the "
          f"cameras that see them: {len(hit)} / {num_faces} ({len(masks)} cameras)")
    hit = erode_hit_faces(faces, hit, iterations=MASK_EROSION_ITERATIONS)
    print(f"after eroding the boundary ({MASK_EROSION_ITERATIONS}x): {len(hit)}")
    return sorted(hit)


def compute_scan_face_mask(actor_id: str, sequence_id: str, raw_frame: int, verts: np.ndarray, faces: np.ndarray,
                           prompt_xyz: np.ndarray, settings, *, sam_masks_out: dict | None = None,
                           weights_out: dict | None = None, dry_run: bool = False,
                           prompt_indices: list[int] | None = None) -> list[int]:
    """Included scan face indices (SAM on the front + FILTER_SIDE_CAMERAS cameras
    + vote), written to scan_mask_path() unless dry_run."""
    import face_ray_masking_wrapping_test as frm

    with open(Path(settings.data_root) / actor_id / "calibration" / "camera_params.json", "r") as f:
        cam_data = json.load(f)
    K = np.asarray(cam_data["intrinsics"], dtype=np.float64)
    world_2_cam = {c: np.asarray(m, dtype=np.float64) for c, m in cam_data["world_2_cam"].items()}
    cams = [c for c in sorted(set(frm.NERSEMBLE_FRONT_CAMERAS) | set(FILTER_SIDE_CAMERAS)) if c in world_2_cam]
    images = load_camera_images(actor_id, sequence_id, raw_frame, cams, settings)
    # Eyebrow / ear prompts occlusion-tested against this scan (hair included).
    masks = compute_sam_masks(images, K, world_2_cam, prompt_xyz, prompt_indices, (verts, faces))
    if not masks:
        raise RuntimeError(f"no filter camera usable for the scan mask of {actor_id}/{sequence_id}/{raw_frame}")
    if sam_masks_out is not None:
        sam_masks_out.update(masks)
    included = vote_faces(verts, faces, masks, K, world_2_cam, weights_out=weights_out)
    if dry_run:
        return included

    out = scan_mask_path(actor_id, sequence_id, raw_frame)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f"{out.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(included))
    os.replace(tmp, out)
    return included


# --- filter -------------------------------------------------------------------

def filter_frame_landmarks(
    actor_id: str, sequence_id: str, raw_frame: int, default_cameras: list[str], num_skin: int, settings,
    *, dry_run: bool = False, sam_masks_out: dict | None = None,
    skin_xyz_override: dict[int, np.ndarray] | None = None,
) -> dict:
    """Scan mask + landmark filter for one frame. Returns the filtered-JSON
    payload (also written to filtered_landmarks_path() unless dry_run).
    skin_xyz_override: {skin index: world xyz} used instead of triangulating
    the skin points -- the neutral frame passes its own wrap's skin vertices
    (exact there, and independent of stale skin_frame files)."""
    import trimesh

    xyz, untriangulated = triangulate_frame_landmarks(actor_id, sequence_id, raw_frame, default_cameras, num_skin, settings)
    if skin_xyz_override:
        xyz = {i: p for i, p in xyz.items() if i < NUM_STANDARD_LANDMARKS}
        xyz.update({int(i): np.asarray(p, dtype=np.float64) for i, p in skin_xyz_override.items()})
        untriangulated = [i for i in untriangulated if i < NUM_STANDARD_LANDMARKS]
    if NOSEBRIDGE_INDEX not in xyz:
        raise RuntimeError(f"nosebridge (index {NOSEBRIDGE_INDEX}) not triangulated for {actor_id}/{sequence_id}/{raw_frame} "
                           "-- no SAM prompt")
    verts, faces = load_scan(actor_id, sequence_id, raw_frame)
    masks: dict[str, np.ndarray] = {}
    print(f"{actor_id}/{sequence_id}/{raw_frame}: SAM prompt landmarks {sam_prompt_indices(xyz)}")
    included = set(compute_scan_face_mask(actor_id, sequence_id, raw_frame, verts, faces, sam_prompt_xyz(xyz),
                                          settings, sam_masks_out=masks, dry_run=dry_run,
                                          prompt_indices=sam_prompt_indices(xyz)))
    if sam_masks_out is not None:
        sam_masks_out.update(masks)
    with open(Path(settings.data_root) / actor_id / "calibration" / "camera_params.json", "r") as f:
        cam_data = json.load(f)
    K = np.asarray(cam_data["intrinsics"], dtype=np.float64)
    world_2_cam = {c: np.asarray(m, dtype=np.float64) for c, m in cam_data["world_2_cam"].items()}
    votes = landmark_sam_votes(xyz, masks, K, world_2_cam, verts, faces)

    indices = sorted(xyz)
    scan = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    _closest, dist, face_ids = trimesh.proximity.closest_point(scan, np.stack([xyz[i] for i in indices]))
    kept, dropped = {}, {}
    for i, d, f in zip(indices, dist, face_ids):
        passed = passes_landmark_vote(votes[i])
        entry = {"xyz": xyz[i].tolist(), "sam_votes": votes[i], "passed_sam_vote": passed,
                 "scan_face": int(f), "scan_distance": float(d), "in_scan_mask": int(f) in included}
        (kept if passed or i in NEVER_FILTERED_INDICES else dropped)[str(i)] = entry
    payload = {
        "actor_id": actor_id, "sequence_id": sequence_id, "frame_id": raw_frame,
        "scan": str(scan_gz_path(actor_id, sequence_id, raw_frame)),
        "scan_mask": str(scan_mask_path(actor_id, sequence_id, raw_frame)),
        "scan_mask_faces": len(included), "scan_faces": len(faces),
        "kept": kept, "dropped": dropped, "untriangulated": untriangulated,
        "never_filtered": sorted(NEVER_FILTERED_INDICES),
        "rule": {"mask_margin_px": MASK_MARGIN_PX, "occlusion_tolerance_m": OCCLUSION_TOLERANCE_M,
                 "agreement_fraction": AGREEMENT_FRACTION, "min_agreeing_cameras": MIN_AGREEING_CAMERAS},
    }
    n_std = sum(1 for i in dropped if int(i) < NUM_STANDARD_LANDMARKS)
    print(f"{actor_id}/{sequence_id}/{raw_frame}: scan filter kept {len(kept)}, dropped {len(dropped)} "
          f"({n_std} standard, {len(dropped) - n_std} skin): {sorted(int(i) for i in dropped)}")
    if not dry_run:
        out = filtered_landmarks_path(settings.landmarks_root, actor_id, sequence_id, raw_frame)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(f"{out.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(payload))
        os.replace(tmp, out)
    return payload


def frames_with_scans_and_landmarks(landmarks_root: Path, actor_id: str, sequence_id: str,
                                    default_cameras: list[str]) -> list[int]:
    """Raw frame ids that have both a Meshroom scan and landmark files for at
    least one default camera."""
    frames = []
    for d in (MESHROOM_DATA_ROOT / actor_id / sequence_id).glob("frame_*"):
        raw = int(d.name.split("_", 1)[1])
        if raw % RAW_FRAME_STRIDE or not (d / "mesh.obj.gz").exists():
            continue
        tag = f"{raw // RAW_FRAME_STRIDE:06d}"
        if any((Path(landmarks_root) / actor_id / sequence_id / f"cam_{c}" / f"frame_{tag}.json").exists()
               for c in default_cameras):
            frames.append(raw)
    return sorted(frames)
