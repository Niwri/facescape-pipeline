#!/usr/bin/env python3
"""run_neutral_skin_propagation.py for FaceScape's INVALID pairs
(invalid_pairs.json) -- pairs whose cameras could not be aligned to the
canonical origin rig, so canonical camera numbers and the hard-coded
front/left/right camera sets (FACESCAPE_SKIN_REGION_CAMERAS,
FACESCAPE_STANDARD_INDEX_REGION_GROUPS' "cameras") don't apply.

Same propagation as run_neutral_skin_propagation.py -- per camera, track each
landmark from the actor's neutral-expression photo into the target
expression's photo (RAFT/CoTracker, same per-point routing), triangulate, and
write skin_frame_/<group>_frame_000000.json -- with the camera handling
replaced by Ava-256's geometric method:

  * Which cameras see each landmark is chosen per landmark from the neutral's
    wrapped mesh (facing angle + occlusion, then greedy coverage --
    camera_selection.select_cameras_for_landmarks), with front/left/right
    derived from the nose (invalid_pairs.classify_front_right_left: Ava-256's split, front = nose tip - ear midpoint).
  * Cameras are matched between the neutral and each target by serial number,
    since raw camera indices differ per pair.
  * Each target is triangulated with its OWN raw cameras.

Neutral: the actor's neutral_frame.json expression (required, as for
run_pipeline.py). If it is a valid pair (class B actors: only some expressions
invalid), its existing origin-space wrap and cameras are the source. If it is
itself invalid (class A actors: whole actor on a different rig), it is
bootstrap-wrapped with run_pipeline_invalid.py first if needed, and its own
skin/group landmark files are written from that wrap.

Targets: the actor's invalid expressions currently "unlabeled" (sapiens2 done,
see FaceView/scripts/preprocess/process_expression_sapiens_invalid.py), or
--expressions. Each successful target moves "unlabeled" -> "unreviewed".

Usage:
    python run_neutral_skin_propagation_invalid.py [actor_id] [--expressions E ...] [--dry-run]
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import trimesh

import invalid_pairs as ip
import run_neutral_skin_propagation as nsp  # trackers, triangulation, JSON writers
import run_pipeline as rp
from run_pipeline_invalid import run_pipeline_invalid

from backend.app.services import facescape_neutral  # noqa: E402  (FaceView root is on sys.path via nsp)
from backend.app.services.label_tracker import label_tracker  # noqa: E402

MIN_CAMERAS_PER_LANDMARK = 5
ANGLE_THRESHOLD_DEG = 80.0


def _standard_groups() -> list[tuple[str, str, list[int]]]:
    """(group name, file prefix, indices) for ear/nosebridge/undernose/eyebrow/chin --
    the same groups as run_neutral_skin_propagation.py, cameras ignored."""
    groups = []
    for group_name, region_config, file_prefix in nsp.FACESCAPE_STANDARD_INDEX_REGION_GROUPS:
        indices = sorted({i for region in region_config.values() for i in region["indices"]})
        groups.append((group_name, file_prefix, indices))
    return groups


def _triangulate(observations_by_idx: dict[int, list[tuple[str, np.ndarray]]], cams: dict[str, dict]) -> dict[int, np.ndarray]:
    """{idx: [(cam_id, pixel_xy), ...]} -> {idx: world xyz}, with the given cameras."""
    result = {}
    for idx, observations in observations_by_idx.items():
        if len(observations) < 2:
            continue
        cam_order = [cam_id for cam_id, _xy in observations]
        intrinsic_arr = np.stack([cams[c]["intrinsic"] for c in cam_order])
        extrinsic_list = [cams[c]["extrinsic"] for c in cam_order]
        obs = [(i, np.asarray(xy, dtype=np.float64), True) for i, (_c, xy) in enumerate(observations)]
        fused = nsp.triangulate_point(obs, intrinsic_arr, extrinsic_list, cams[cam_order[0]]["width"], cams[cam_order[0]]["height"])
        if fused is not None:
            result[idx] = fused
    return result


def _write_landmarks(actor_id, expression_id, cams, skin_xyz, group_xyz: dict[str, tuple[str, dict]], dry_run) -> bool:
    output_cams = sorted(cams, key=int)
    if dry_run:
        return bool(skin_xyz)
    written = nsp.write_points_frames(actor_id, expression_id, cams, skin_xyz, output_cams, nsp.get_settings()) if skin_xyz else 0
    for group_name, (file_prefix, xyz) in group_xyz.items():
        if xyz:
            nsp.write_points_frames(actor_id, expression_id, cams, xyz, output_cams, nsp.get_settings(),
                                    region_label=group_name, filename=f"{file_prefix}_frame_000000.json")
    return written > 0


def _mark_labeled(actor_id: str, expression_id: str) -> None:
    try:
        label_tracker.confirm_labeled(actor_id, expression_id, dataset="FaceScape")
    except ValueError as exc:
        print(f"  NOTE: not moving {actor_id}/{expression_id} to 'unreviewed' -- {exc}")


def process_actor(actor_id: str, raft_model, cotracker_model, expressions_override, dry_run: bool) -> tuple[int, int, int]:
    tag = f"[actor={actor_id}]"
    settings = nsp.get_settings()
    neutral_id = facescape_neutral.load_actor_neutral_expression(actor_id)
    if neutral_id is None:
        print(f"SKIP {tag}: no neutral_frame.json (save the actor's neutral expression in FaceView first)")
        return 0, 0, 0
    neutral_invalid = ip.is_invalid(actor_id, neutral_id)
    neutral_dir = settings.facescape_output_root / actor_id / neutral_id
    wrapped_path = neutral_dir / "wrapped_mesh.obj"   # raw space if the neutral is invalid, origin space otherwise
    if not wrapped_path.exists():
        if not neutral_invalid:
            print(f"SKIP {tag}: valid neutral {neutral_id} isn't wrapped yet (run_pipeline_batch.py's job)")
            return 0, 0, 0
        print(f"{tag} Bootstrap-wrapping invalid neutral {neutral_id} (standard landmarks only)...")
        run_pipeline_invalid(actor_id, neutral_id, mark_unconfirmed=False, render=False)

    neutral_cams = ip.load_pair_cameras(actor_id, neutral_id)
    mesh = trimesh.load(str(wrapped_path), process=False)
    verts, faces = np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces)

    # Landmarks: skin (74+, fixed FLAME vertices) + the standard groups.
    assigned = rp.assign_global_indices(nsp._skin_landmarks_module.load_skin_regions())
    skin_xyz = {g: np.asarray(verts[v], dtype=np.float64) for g, _region, v in assigned}
    groups = _standard_groups()
    standard_indices = sorted({i for _g, _p, idxs in groups for i in idxs})
    raft_indices = {g for g, _r, v in assigned if v in nsp.RAFT_SKIN_VERTEX_INDICES}
    raft_indices |= {i for g, _p, idxs in groups if g in nsp.RAFT_STANDARD_GROUPS for i in idxs}

    # Geometric camera choice on the neutral wrap (Ava-256).
    pot_path = rp.ensure_extended_correspondence_files(str(rp.DEFAULT_NEUTRAL_MESH_PATH), rp.load_skin_vertex_indices())
    pot_rows = ip.pot_rows_by_index(pot_path)
    front, right, left = ip.classify_front_right_left(verts, faces, pot_rows, neutral_cams)
    print(f"{tag} neutral={neutral_id} ({'invalid/raw' if neutral_invalid else 'valid/origin'}), {len(neutral_cams)} cameras: "
          f"front={sorted(front, key=int)} left={sorted(left, key=int)} right={sorted(right, key=int)}")
    selected = ip.select_cameras_for_landmarks(
        sorted(skin_xyz) + standard_indices, pot_rows, verts, faces, neutral_cams,
        min_cameras=MIN_CAMERAS_PER_LANDMARK, angle_threshold_deg=ANGLE_THRESHOLD_DEG,
    )

    # Seeds per neutral camera: skin projected from the wrap, standard from the
    # neutral's own sapiens2/ear landmarks.
    seeds_by_cam: dict[str, dict[int, tuple[float, float]]] = {}
    frame_points_cache: dict[str, dict] = {}
    for idx, cam_ids in selected.items():
        for cam_id in cam_ids:
            cam = neutral_cams[cam_id]
            if idx in skin_xyz:
                xy = ip.project(skin_xyz[idx], cam)
            else:
                points = frame_points_cache.setdefault(cam_id, nsp.load_standard_frame_points(
                    actor_id, neutral_id, cam_id, cam["width"], cam["height"], settings))
                xy = points.get(idx)
            if xy is not None:
                seeds_by_cam.setdefault(cam_id, {})[idx] = xy

    # An invalid neutral writes its own landmark files (skin from the wrap,
    # groups triangulated from its own seeds); a valid neutral already has them.
    if neutral_invalid:
        neutral_groups = {}
        for group_name, file_prefix, idxs in groups:
            obs = {i: [(c, s[i]) for c, s in seeds_by_cam.items() if i in s] for i in idxs}
            neutral_groups[group_name] = (file_prefix, _triangulate(obs, neutral_cams))
        if _write_landmarks(actor_id, neutral_id, neutral_cams, skin_xyz, neutral_groups, dry_run) and not dry_run:
            _mark_labeled(actor_id, neutral_id)
        print(f"{tag} wrote the invalid neutral's own skin/group landmarks")

    # Targets.
    if expressions_override:
        targets = expressions_override
    else:
        statuses = label_tracker.get_all(dataset="FaceScape").get(actor_id, {})
        targets = sorted(e for a, e in ip.load_invalid_pairs()
                         if a == actor_id and e != neutral_id and statuses.get(e) == "unlabeled")
    print(f"{tag} {len(targets)} invalid target expression(s): {targets}")

    passed = failed = skipped = 0
    for expression_id in targets:
        with nsp._expression_pair_lock(settings.facescape_landmarks_root, actor_id, expression_id) as acquired:
            if not acquired:
                skipped += 1
                continue
            try:
                ok = propagate_expression(actor_id, neutral_id, expression_id, neutral_cams, seeds_by_cam, raft_indices,
                                          groups, raft_model, cotracker_model, dry_run)
            except Exception as exc:  # noqa: BLE001 -- one bad target must not stop the actor's others
                print(f"FAIL [actor={actor_id} expression={expression_id}]: {exc}")
                ok = False
        if ok:
            passed += 1
            if not dry_run:
                _mark_labeled(actor_id, expression_id)
        else:
            failed += 1
    print(f"{tag} Done. targets={len(targets)} passed={passed} failed={failed} skipped={skipped}")
    return passed, failed, skipped


def propagate_expression(actor_id, neutral_id, expression_id, neutral_cams, seeds_by_cam, raft_indices, groups,
                         raft_model, cotracker_model, dry_run) -> bool:
    tag = f"[actor={actor_id} expression={expression_id}]"
    target_cams = ip.load_pair_cameras(actor_id, expression_id)
    target_by_serial = ip.cam_id_by_serial(target_cams)

    observations: dict[int, list[tuple[str, np.ndarray]]] = {}
    matched = 0
    for neutral_cam_id, seeds in seeds_by_cam.items():
        target_cam_id = target_by_serial.get(neutral_cams[neutral_cam_id]["serial"])
        if target_cam_id is None:
            continue
        neutral_img = ip.load_pair_image(actor_id, neutral_id, neutral_cam_id)
        target_img = ip.load_pair_image(actor_id, expression_id, target_cam_id)
        if neutral_img is None or target_img is None:
            continue
        matched += 1
        tracked = {}
        raft_seeds = {i: xy for i, xy in seeds.items() if i in raft_indices}
        other_seeds = {i: xy for i, xy in seeds.items() if i not in raft_indices}
        if raft_seeds:
            tracked.update(nsp.track_points_one_camera(raft_model, neutral_img, target_img, raft_seeds))
        if other_seeds:
            tracked.update(nsp.track_points_one_camera(cotracker_model, neutral_img, target_img, other_seeds))
        for idx, (xy, _visibility) in tracked.items():
            observations.setdefault(idx, []).append((target_cam_id, xy))

    world = _triangulate(observations, target_cams)
    skin_xyz = {i: p for i, p in world.items() if i >= rp.NUM_STANDARD_LANDMARKS}
    group_xyz = {g: (prefix, {i: world[i] for i in idxs if i in world}) for g, prefix, idxs in groups}
    print(f"  {tag}: {matched} camera(s) matched by serial; triangulated {len(skin_xyz)} skin and "
          f"{sum(len(x) for _p, x in group_xyz.values())} group point(s)")
    return _write_landmarks(actor_id, expression_id, target_cams, skin_xyz, group_xyz, dry_run)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("actor_id", nargs="?", default=None, help="Actor id; omit to sweep every actor with an 'unlabeled' invalid pair")
    parser.add_argument("--expressions", nargs="*", default=None, help="Explicit target expressions (default: the actor's 'unlabeled' invalid ones)")
    parser.add_argument("--dry-run", action="store_true", help="Track and triangulate, but write nothing and change no labels")
    args = parser.parse_args()

    if args.actor_id:
        actors = [args.actor_id]
    else:
        statuses = label_tracker.get_all(dataset="FaceScape")
        actors = sorted({a for a, e in ip.load_invalid_pairs() if statuses.get(a, {}).get(e) == "unlabeled"}, key=int)
    print(f"{len(actors)} actor(s) to process")
    raft_model = nsp.load_raft_model()
    cotracker_model = nsp.load_cotracker_model()
    failed_total = 0
    for actor_id in actors:
        try:
            _passed, failed, _skipped = process_actor(actor_id, raft_model, cotracker_model, args.expressions, args.dry_run)
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL [actor={actor_id}]: {exc}")
            failed = 1
        failed_total += failed
    return 0 if failed_total == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
