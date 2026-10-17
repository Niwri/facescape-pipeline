#!/usr/bin/env python3
"""run_pipeline.py for FaceScape's INVALID pairs -- pairs whose cameras could
not be aligned to the canonical origin rig (invalid_pairs.json), so they only
exist in their own raw space. See invalid_pairs.py for the shared helpers.

Same two steps as run_pipeline.py (triangulate sapiens2 + propagated
landmarks, then wrap_mesh with the static facescape_mask.txt), done in the
pair's raw space with one extra move:

  1. Triangulate the pair's landmarks with its own raw cameras (raw space).
  2. Move the raw scan + landmarks into the FLAME template frame: a Kabsch
     rotation (triangulated standard landmarks -> template landmarks), then
     Ava-256's bounding-box scaling and center-to-center translation
     (invalid_pairs.compute_template_transform). Saved as transform_params.npz.
  3. Wrap in the template frame -> wrapped_mesh_template.obj.
  4. Untransform back to the pair's raw space -> wrapped_mesh.obj. So, unlike
     valid pairs, the saved mesh is already in the scan's original space and
     FaceView's confirm/export must NOT apply the origin->raw transform to it.

Source mesh (as run_pipeline.py): the actor's neutral expression wraps from
the generic template; every other expression chains off the neutral's wrap in
the template frame -- wrapped_mesh_template.obj when the neutral is itself
invalid (class A actors), or the neutral's own wrapped_mesh.obj when it is a
valid, origin-space pair (the origin frame is the template frame).

Overlay renders use three cameras picked geometrically (most frontal / left /
right, see invalid_pairs.most_frontal_left_right), recorded in
overlay_cameras.json next to the outputs for FaceView.

Usage:
    python3 run_pipeline_invalid.py <ACTOR> <EXPRESSION> [--no-mark]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh

import invalid_pairs as ip
import run_pipeline as rp  # triangulation helpers, extended POT, wrap_mesh, neutral lookup, label tracker

WRAPPED_TEMPLATE_NAME = "wrapped_mesh_template.obj"
TRANSFORM_PARAMS_NAME = "transform_params.npz"
OVERLAY_CAMERAS_NAME = "overlay_cameras.json"


def _triangulate_pair(actor_id: str, expression_id: str, cams: dict[str, dict]):
    """Same landmark gathering + triangulation as run_pipeline.run_pipeline(),
    with the pair's own (raw) cameras. Returns (formatted, untriangulated,
    skin_vertex_indices)."""
    settings = rp.get_settings()
    landmarks_dir = settings.facescape_landmarks_root / actor_id / expression_id
    cam_ids = list(cams)
    intrinsics = [cams[c]["intrinsic"] for c in cam_ids]
    extrinsics = [cams[c]["extrinsic"] for c in cam_ids]
    widths = [cams[c]["width"] for c in cam_ids]
    heights = [cams[c]["height"] for c in cam_ids]

    skin_vertex_indices = rp.load_skin_vertex_indices()
    num_vertices = rp.NUM_STANDARD_LANDMARKS + len(skin_vertex_indices)
    points_list: list[list] = [[] for _ in range(num_vertices)]
    for cam_index, cam_id in enumerate(cam_ids):
        width, height = widths[cam_index], heights[cam_index]
        cam_dir = landmarks_dir / f"cam_{cam_id}"
        standard_points = rp.load_camera_standard_points(cam_dir, "000000")
        skin_points = rp.load_camera_skin_points(cam_dir, "000000")
        if not width or not height or (standard_points is None and skin_points is None):
            continue
        for point in standard_points or []:
            if point["index"] < rp.NUM_VERTICES:
                pixel = np.array([round(point["x"] * width), round(point["y"] * height)])
                points_list[point["index"]].append((cam_index, pixel, rp.is_in_bounds(point)))
        for point in skin_points or []:
            if point["index"] < num_vertices:
                pixel = np.array([round(point["x"] * width), round(point["y"] * height)])
                points_list[point["index"]].append((cam_index, pixel, rp.is_in_bounds(point)))

    if skin_vertex_indices:
        insufficient = [i for i in range(rp.NUM_STANDARD_LANDMARKS, num_vertices) if len(points_list[i]) < 2]
        if insufficient:
            print(f"NOTE: {len(insufficient)}/{len(skin_vertex_indices)} skin point(s) have fewer than 2 camera "
                  "observations (not propagated yet?) -- falling back to standard-only (no skin points).")
            points_list = points_list[: rp.NUM_STANDARD_LANDMARKS]
            skin_vertex_indices = []

    formatted, untriangulated = rp.triangulate(np.array(intrinsics), extrinsics, points_list, widths, heights)
    return formatted, untriangulated, skin_vertex_indices


def _write_obj(path: Path, verts: np.ndarray, faces: np.ndarray) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as f:
        for v in verts:
            f.write(f"v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")
        for face in faces:
            f.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")
    tmp.replace(path)


def neutral_template_mesh(actor_id: str, neutral_expression_id: str) -> Path | None:
    """The actor's neutral wrap in the template frame, or None if not wrapped yet."""
    neutral_dir = rp.get_settings().facescape_output_root / actor_id / neutral_expression_id
    path = neutral_dir / (WRAPPED_TEMPLATE_NAME if ip.is_invalid(actor_id, neutral_expression_id) else "wrapped_mesh.obj")
    return path if path.exists() else None


def run_pipeline_invalid(actor_id: str, expression_id: str, mark_unconfirmed: bool = True, render: bool = True) -> Path:
    settings = rp.get_settings()
    if not ip.is_invalid(actor_id, expression_id):
        raise ValueError(f"{actor_id}/{expression_id} is not in invalid_pairs.json -- use run_pipeline.py")
    scan_path = ip.scan_path_for(actor_id, expression_id)
    if not scan_path.exists():
        raise FileNotFoundError(f"raw scan not found: {scan_path}")

    cams = ip.load_pair_cameras(actor_id, expression_id)
    formatted, untriangulated, skin_vertex_indices = _triangulate_pair(actor_id, expression_id, cams)
    print(f"Triangulated {actor_id}/{expression_id} with {len(cams)} raw camera(s); "
          f"{len(untriangulated)} untriangulated; {len(skin_vertex_indices)} skin point(s)")

    output_dir = settings.facescape_output_root / actor_id / expression_id
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "keypoint_vertices.json").write_text(json.dumps(formatted))  # raw space

    # Raw -> template frame.
    scan = trimesh.load(str(scan_path), process=False)
    scan_verts = np.asarray(scan.vertices, dtype=np.float64)
    missing = set(untriangulated)
    raw_landmarks = {i: np.array([p["x"], p["y"], p["z"]]) for i, p in enumerate(formatted[: rp.NUM_STANDARD_LANDMARKS])
                     if i not in missing}
    params = ip.compute_template_transform(raw_landmarks, scan_verts)
    ip.save_transform(output_dir / TRANSFORM_PARAMS_NAME, params)
    print(f"Raw->template: rotation {params['rotation_deg']:.1f} deg, scale {params['scale']:.5f}, "
          f"landmark rms {params['landmark_rms']:.4f} ({params['landmarks_used']} landmarks)")

    scan_obj_path = output_dir / "scan.obj"
    trimesh.Trimesh(ip.transform(scan_verts, params), np.asarray(scan.faces), process=False).export(str(scan_obj_path))
    formatted_template = [
        p if i in missing else dict(zip("xyz", map(float, ip.transform(np.array([[p["x"], p["y"], p["z"]]]), params)[0])))
        for i, p in enumerate(formatted)
    ]
    correspondence_path = output_dir / "keypoint_vertices_template.json"
    correspondence_path.write_text(json.dumps(formatted_template))

    pot_template_path = None
    if skin_vertex_indices:
        pot_template_path = rp.ensure_extended_correspondence_files(str(rp.DEFAULT_NEUTRAL_MESH_PATH), skin_vertex_indices)

    # Source mesh: template for the neutral itself, the neutral's template-frame wrap otherwise.
    neutral_expression_id = rp.facescape_neutral.load_actor_neutral_expression(actor_id)
    if neutral_expression_id is None:
        raise RuntimeError(f"No neutral_frame.json saved for actor {actor_id} yet")
    source_mesh = rp.DEFAULT_NEUTRAL_MESH_PATH
    if expression_id != neutral_expression_id:
        source_mesh = neutral_template_mesh(actor_id, neutral_expression_id) or rp.DEFAULT_NEUTRAL_MESH_PATH
    print(f"Source mesh: {source_mesh}")

    wrapped_template_path = output_dir / WRAPPED_TEMPLATE_NAME
    wrap_kwargs = dict(
        target_mesh_path=str(scan_obj_path),
        target_correspondence_path=str(correspondence_path),
        output_path=str(wrapped_template_path),
        extra_indices_to_remove=untriangulated,
        neutral_mesh_path=str(source_mesh),
        **({"pot_template_path": pot_template_path} if pot_template_path else {}),
    )
    if Path(source_mesh) == Path(rp.DEFAULT_NEUTRAL_MESH_PATH):
        # Same face-based target scaling as run_pipeline.py's template wraps (face_scale.py),
        # applied in this pair's template frame and undone before the untransform below.
        rp.face_scale.wrap_with_face_scale(rp.wrap_mesh, **wrap_kwargs)
    else:
        (output_dir / "face_scale.json").unlink(missing_ok=True)
        rp.wrap_mesh(**wrap_kwargs)

    # Back to the pair's raw space.
    wrapped = trimesh.load(str(wrapped_template_path), process=False)
    faces = np.asarray(wrapped.faces)
    raw_verts = ip.untransform(np.asarray(wrapped.vertices, dtype=np.float64), params)
    wrapped_path = output_dir / "wrapped_mesh.obj"
    _write_obj(wrapped_path, raw_verts, faces)
    print(f"Saved wrapped mesh (raw space) to: {wrapped_path}")

    if mark_unconfirmed:
        rp.label_tracker.mark_unconfirmed(actor_id, expression_id, dataset="FaceScape")

    if render:
        _render_overlays(actor_id, expression_id, raw_verts, faces, cams, output_dir)
    return wrapped_path


def _render_overlays(actor_id, expression_id, raw_verts, faces, cams, output_dir: Path) -> None:
    """Keyline + point overlays on three geometrically chosen cameras. Best-effort."""
    try:
        pot_rows = ip.pot_rows_by_index(ip.BASE_POT_PATH)
        camera_ids = ip.most_frontal_left_right(raw_verts, faces, pot_rows, cams)
        (output_dir / OVERLAY_CAMERAS_NAME).write_text(json.dumps(camera_ids))
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: could not choose overlay cameras for {actor_id}/{expression_id}: {exc}")
        return
    for camera_id in camera_ids:
        try:
            rp.render_overlay_frame(actor_id, expression_id, camera_id)
        except Exception as exc:  # noqa: BLE001
            print(f"WARNING: overlay frame render failed for {actor_id}/{expression_id} camera {camera_id}: {exc}")
        try:
            from render_point_overlay_frame import render_point_overlay_frame
            render_point_overlay_frame(actor_id, expression_id, camera_id)
        except Exception as exc:  # noqa: BLE001
            print(f"WARNING: point overlay render failed for {actor_id}/{expression_id} camera {camera_id}: {exc}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("actor_id")
    parser.add_argument("expression_id")
    parser.add_argument("--no-mark", action="store_true", help="Don't move the pair to 'unconfirmed'")
    parser.add_argument("--no-render", action="store_true", help="Skip the overlay renders")
    args = parser.parse_args()
    run_pipeline_invalid(args.actor_id, args.expression_id, mark_unconfirmed=not args.no_mark, render=not args.no_render)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
