"""Automatic per-landmark camera selection for Ava-256 propagation.

No more hardcoded per-group camera lists (FaceScape's FACESCAPE_STANDARD_
INDEX_REGION_GROUPS/FACESCAPE_SKIN_REGION_CAMERAS) -- each landmark gets its
own camera set, computed from the neutral frame's own wrapped mesh:

1. Candidate pool: every camera in camera_calibration.json (~80).
2. Score: angle between the landmark's surface normal and the camera's
   -forward vector (lower = camera looking more directly at the surface).
3. Occlusion: genuine ray-mesh intersection against the wrapped mesh's own
   triangles, not a normal/backface heuristic -- occluded cameras get
   score=inf and are dropped as candidates entirely.
4. Threshold: keep unoccluded candidates under angle_threshold_deg; if fewer
   than min_cameras survive, relax to the best-scoring min_cameras regardless.
5. Clustering: two-stage greedy weighted-max-coverage (minimize distinct
   cameras touched across the whole capture) + a free-reuse pass (give a
   landmark extra already-selected cameras beyond its quota at zero marginal
   camera-selection cost).
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict

import numpy as np
import trimesh

import mesh_utils
import camera_utils

DEFAULT_MIN_CAMERAS = 5
DEFAULT_ANGLE_THRESHOLD_DEG = 80.0
NORMAL_NUDGE = 2.0  # mesh-scale epsilon (FLAME template's median edge length is ~3.4 units;
                     # empirically, nudges below ~0.5 units still produce false self-hits on
                     # neighboring triangles for rays at a shallow/grazing angle to the surface
                     # normal -- verified against this template directly, not a guessed constant)


def score_camera_for_landmark(face_normal: np.ndarray, Rt_cam: np.ndarray) -> float:
    """Radians between the landmark's surface normal and -camera_forward --
    0 = camera looking straight along the outward normal (best possible
    view), pi = camera looking the wrong way entirely. Lower is better."""
    forward = camera_utils.camera_forward_world(Rt_cam)
    cos_angle = float(np.clip(np.dot(face_normal, -forward), -1.0, 1.0))
    return math.acos(cos_angle)


def build_ray_intersector(verts_world: np.ndarray, faces: np.ndarray):
    mesh = trimesh.Trimesh(vertices=verts_world, faces=faces, process=False)
    try:
        import trimesh.ray.ray_pyembree as ray_pyembree
        return ray_pyembree.RayMeshIntersector(mesh)
    except ImportError:
        return trimesh.ray.ray_triangle.RayMeshIntersector(mesh)


def is_occluded(intersector, landmark_xyz: np.ndarray, face_normal: np.ndarray, camera_center: np.ndarray,
                normal_nudge: float = NORMAL_NUDGE) -> bool:
    """Ray from just off the landmark's own surface (nudged along its
    normal to dodge a false self-hit on its own triangle) toward the camera
    center. Any OTHER intersection strictly closer than the camera means
    something on this same mesh blocks the view."""
    origin = landmark_xyz + normal_nudge * face_normal
    to_cam = camera_center - origin
    dist = float(np.linalg.norm(to_cam))
    if dist == 0.0:
        return False
    direction = to_cam / dist
    locations, _, _ = intersector.intersects_location(
        ray_origins=origin[None], ray_directions=direction[None], multiple_hits=True
    )
    if len(locations) == 0:
        return False
    hit_dists = np.linalg.norm(locations - origin, axis=1)
    return bool(np.any(hit_dists < dist - normal_nudge))


def landmark_position_and_normal(
    verts_world: np.ndarray, faces: np.ndarray, pot_row: list[float]
) -> tuple[np.ndarray, np.ndarray]:
    xyz = mesh_utils.pot_row_world_xyz(verts_world, faces, pot_row)
    normal = mesh_utils.pot_row_face_normal(verts_world, faces, pot_row)
    return xyz, normal


def rank_candidates_for_landmark(
    xyz: np.ndarray,
    normal: np.ndarray,
    camera_ids: list[str],
    camera_params: dict[str, tuple[np.ndarray, np.ndarray]],
    intersector,
    normal_nudge: float = NORMAL_NUDGE,
) -> list[tuple[str, float]]:
    """All unoccluded cameras for this landmark, sorted by ascending angle
    score (best first). Occluded cameras are excluded entirely (not scored
    as infinity in the output -- infinity is just how they're conceptually
    treated during filtering)."""
    scored: list[tuple[str, float]] = []
    for cam_id in camera_ids:
        K, Rt = camera_params[cam_id]
        center = camera_utils.camera_center_world(Rt)
        if is_occluded(intersector, xyz, normal, center, normal_nudge):
            continue
        score = score_camera_for_landmark(normal, Rt)
        scored.append((cam_id, score))
    scored.sort(key=lambda pair: pair[1])
    return scored


def select_cameras_for_landmarks(
    landmark_indices: list[int],
    pot_rows_by_index: dict[int, list[float]],
    verts_world: np.ndarray,
    faces: np.ndarray,
    camera_ids: list[str],
    camera_params: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    min_cameras: int = DEFAULT_MIN_CAMERAS,
    angle_threshold_deg: float = DEFAULT_ANGLE_THRESHOLD_DEG,
    normal_nudge: float | None = None,
) -> dict[int, list[str]]:
    """Full pipeline: score+occlusion-filter every (landmark, camera) pair,
    threshold, then two-stage greedy-cluster to bound total distinct
    cameras touched. Returns landmark_index -> assigned camera ids.

    normal_nudge: occlusion-ray offset in the mesh's own world units
    (default NORMAL_NUDGE, tuned for Ava-256's mm-scale world); pass one
    scaled to the mesh for other datasets (e.g. FaceScape raw scans).
    """
    intersector = build_ray_intersector(verts_world, faces)
    angle_threshold = math.radians(angle_threshold_deg)

    full_ranked: dict[int, list[tuple[str, float]]] = {}
    thresholded_pool: dict[int, list[tuple[str, float]]] = {}
    for idx in landmark_indices:
        xyz, normal = landmark_position_and_normal(verts_world, faces, pot_rows_by_index[idx])
        ranked = rank_candidates_for_landmark(xyz, normal, camera_ids, camera_params, intersector,
                                              NORMAL_NUDGE if normal_nudge is None else normal_nudge)
        full_ranked[idx] = ranked
        under_threshold = [pair for pair in ranked if pair[1] <= angle_threshold]
        if len(under_threshold) < min_cameras:
            if len(ranked) < min_cameras:
                print(
                    f"WARNING: landmark {idx} has only {len(ranked)} unoccluded camera(s) "
                    f"available (< min_cameras={min_cameras})"
                )
            thresholded_pool[idx] = ranked[:min_cameras]
        else:
            thresholded_pool[idx] = under_threshold

    # Stage (a): greedy weighted maximum coverage -- minimize distinct
    # cameras while satisfying every landmark's min_cameras quota.
    remaining_need = {idx: min_cameras for idx in landmark_indices}
    candidate_pool = {idx: list(thresholded_pool[idx]) for idx in landmark_indices}
    assigned: dict[int, list[str]] = {idx: [] for idx in landmark_indices}

    while any(remaining_need[idx] > 0 for idx in landmark_indices):
        coverage_count: Counter[str] = Counter()
        scores_by_cam: dict[str, list[float]] = defaultdict(list)
        for idx in landmark_indices:
            if remaining_need[idx] <= 0:
                continue
            for cam_id, score in candidate_pool[idx]:
                coverage_count[cam_id] += 1
                scores_by_cam[cam_id].append(score)

        if not coverage_count:
            break  # some landmarks permanently under quota -- already warned above

        best_cam = max(
            coverage_count,
            key=lambda c: (coverage_count[c], -(sum(scores_by_cam[c]) / len(scores_by_cam[c]))),
        )

        for idx in landmark_indices:
            if remaining_need[idx] <= 0:
                continue
            pool_cams = dict(candidate_pool[idx])
            if best_cam in pool_cams:
                assigned[idx].append(best_cam)
                remaining_need[idx] -= 1
                candidate_pool[idx] = [(c, s) for c, s in candidate_pool[idx] if c != best_cam]

    selected_cameras = {cam for cams in assigned.values() for cam in cams}

    # Stage (b): free-reuse pass -- add any already-selected camera that's
    # in a landmark's full qualifying pool (under threshold), since reusing
    # an already-selected camera costs nothing extra in total distinct
    # cameras touched.
    for idx in landmark_indices:
        already = set(assigned[idx])
        for cam_id, score in full_ranked[idx]:
            if score > angle_threshold:
                continue
            if cam_id in selected_cameras and cam_id not in already:
                assigned[idx].append(cam_id)
                already.add(cam_id)

    return assigned
