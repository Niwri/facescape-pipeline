import numpy as np
from backend.app.core.config import get_settings
import argparse
import cv2
import json
import time

# A least-squares ray intersection needs at least 2 views to be well-posed at all.
MIN_VIEWS_FOR_TRIANGULATION = 2

# Outlier rejection (see _reject_outlier_views) only runs above this many usable
# views, and never drops a view once usable views would fall to this count or
# below -- leave-one-out needs a reasonably-sized "everyone else" majority to be
# a trustworthy reference, not just one other camera.
MIN_VIEWS_FOR_OUTLIER_REJECTION = 4

# A camera's leave-one-out reprojection residual (normalized by frame size) is
# flagged as an outlier only when BOTH hold: its modified z-score (median/MAD
# across this point's per-camera residuals) clears this, AND the residual
# itself clears OUTLIER_MIN_RESIDUAL. The z-score alone isn't enough -- when
# every camera already agrees closely, MAD is tiny and z-scores get noisy, so
# the absolute floor keeps normal per-camera detection noise from ever
# triggering rejection.
OUTLIER_Z_THRESHOLD = 2.5
OUTLIER_MIN_RESIDUAL = 0.02


def is_in_bounds(point_data: dict) -> bool:
    """A landmark is only a usable triangulation observation if it's actually
    within the visible frame. CoTracker clamps a point that's drifted off-screen
    to the crop edge instead of reporting it missing, so without this check a
    clamped/off-frame point silently feeds a garbage ray into the least-squares
    solve — see e.g. actor 018 EMO-1-shout+laugh frame 355, where cam_222200042
    and cam_222200044 both tracked several lip points past the frame edge during
    a wide mouth-open expression, corrupting the triangulated result."""
    return 0.0 < point_data["x"] < 1.0 and 0.0 < point_data["y"] < 1.0


def _intrinsic_for(intrinsic, cam_index):
    """`intrinsic` is either a single (3,3) matrix shared by every camera
    (Nersemble's single-rig-intrinsic convention -- the original, sole caller
    of this module) or a per-camera (N,3,3) array/list indexed by cam_index
    (FaceScape: each physical camera has its own distinct focal length/
    resolution, so one shared intrinsic would be wrong)."""
    intrinsic = np.asarray(intrinsic)
    return intrinsic[cam_index] if intrinsic.ndim == 3 else intrinsic


def _least_squares_triangulate(observations, intrinsic, extrinsics):
    """The actual closed-form least-squares ray intersection. `observations` is a
    list of (camera_index, pixel_xy) pairs; `extrinsics` is the full per-camera
    list, indexed by camera_index."""
    inv_sum = np.zeros((3, 3))
    sum_vec = np.zeros((3,))

    for cam_index, point in observations:
        R = extrinsics[cam_index][:3, :3]
        t = extrinsics[cam_index][:3, 3]

        cj = -R.T @ t
        vj = R.T @ np.linalg.inv(_intrinsic_for(intrinsic, cam_index)) @ np.append(point, 1)
        vj /= np.linalg.norm(vj)

        temp_matrix = np.identity(3) - np.outer(vj, vj)
        inv_sum += temp_matrix
        sum_vec += temp_matrix @ cj
    return np.linalg.inv(inv_sum) @ sum_vec


def _reprojection_residual_px(world_xyz, cam_index, pixel_xy, intrinsic, extrinsics):
    """Pixel-space (dx, dy) between a 3D point's reprojection into `cam_index`
    and that camera's own observed pixel, or None if the point falls behind it."""
    R = extrinsics[cam_index][:3, :3]
    t = extrinsics[cam_index][:3, 3]
    cam_xyz = R @ world_xyz + t
    if cam_xyz[2] <= 1e-8:
        return None
    img = _intrinsic_for(intrinsic, cam_index) @ cam_xyz
    x, y = img[0] / img[2], img[1] / img[2]
    return np.array([x - pixel_xy[0], y - pixel_xy[1]])


def _resolution_for(orig_width, orig_height, cam_index):
    """Same idea as _intrinsic_for: orig_width/orig_height are each either a
    single value shared by every camera (Nersemble, one rig resolution) or a
    per-camera list/array indexed by cam_index (FaceScape, which mixes camera
    models with different resolutions -- normalizing every camera's residual
    by ONE shared width/height would silently misjudge outliers for whichever
    cameras don't actually match that resolution)."""
    w = np.asarray(orig_width)
    h = np.asarray(orig_height)
    width = w[cam_index] if w.ndim > 0 else orig_width
    height = h[cam_index] if h.ndim > 0 else orig_height
    return width, height


def _reject_outlier_views(usable, intrinsic, extrinsics, orig_width, orig_height):
    """Iteratively drops the single worst-agreeing camera at a time, via
    leave-one-out reprojection error: triangulate from every OTHER usable
    camera, reproject that result into the held-out camera, and measure how far
    it lands from what that camera actually observed. A camera whose own
    detection is wrong (e.g. a squeezed/misdetected landmark) disagrees with the
    consensus of every other camera, not just one -- so unlike comparing raw
    per-camera pixel positions, this isolates the actual geometric
    inconsistency. Stops as soon as the worst remaining residual is no longer a
    clear outlier (see OUTLIER_Z_THRESHOLD/OUTLIER_MIN_RESIDUAL), or once
    MIN_VIEWS_FOR_OUTLIER_REJECTION views remain, whichever comes first."""
    dropped = []
    while len(usable) > MIN_VIEWS_FOR_OUTLIER_REJECTION:
        residuals = {}
        for i in range(len(usable)):
            others = usable[:i] + usable[i + 1:]
            loo_point = _least_squares_triangulate(others, intrinsic, extrinsics)
            cam_index, pixel_xy = usable[i]
            residual = _reprojection_residual_px(loo_point, cam_index, pixel_xy, intrinsic, extrinsics)
            if residual is None:
                continue
            width, height = _resolution_for(orig_width, orig_height, cam_index)
            residuals[i] = float(np.linalg.norm(residual / np.array([width, height])))

        if len(residuals) < 3:
            break  # not enough points landed in front of their camera to judge an outlier

        values = np.array(list(residuals.values()))
        median = np.median(values)
        mad = np.median(np.abs(values - median)) + 1e-9
        worst_i = max(residuals, key=residuals.get)
        worst_residual = residuals[worst_i]
        worst_z = 0.6745 * (worst_residual - median) / mad

        if worst_z <= OUTLIER_Z_THRESHOLD or worst_residual <= OUTLIER_MIN_RESIDUAL:
            break

        dropped.append((usable[worst_i][0], worst_residual, worst_z))
        usable = usable[:worst_i] + usable[worst_i + 1:]

    return usable, dropped


def triangulate_point(point_list, intrinsic, extrinsics, orig_width=None, orig_height=None):
    """`point_list` is a list of (camera_index, 2D pixel point, in_bounds) tuples —
    one entry per camera that has *any* observation for this vertex. `extrinsics`
    is the full, unfiltered per-camera list (indexed by camera_index), so this
    stays correctly aligned even when some cameras get excluded below.

    Only in-bounds observations are used, unless that leaves fewer than
    MIN_VIEWS_FOR_TRIANGULATION, in which case every observation is used anyway —
    a degraded triangulation beats a crash (or a silently-`nan` singular solve).
    The one exception is when literally none of the views are in-bounds (e.g. a
    FLAME index with no sapiens2 equivalent at all, always (-1, -1) -- see
    process_sequence_sapiens.py's SAPIENS_LANDMARK_MAP -- or every camera
    genuinely missed this point this frame): falling back to "every view
    anyway" there would triangulate from N copies of the same out-of-bounds
    sentinel pixel, producing a confident-looking but meaningless 3D point
    instead of an honest "can't triangulate this one" -- so this returns None
    instead, and the caller (triangulate()) drops this point's row entirely.

    When `orig_width`/`orig_height` are given and enough in-bounds views remain,
    also runs leave-one-out outlier rejection (see _reject_outlier_views) before
    the final fit -- an in-bounds-but-wrong observation (e.g. a squeezed lip
    landmark from one camera) otherwise pulls the fit just as hard as a correct
    one, since is_in_bounds alone can't tell the two apart."""
    usable = [(cam_index, point) for cam_index, point, in_bounds in point_list if in_bounds]
    excluded = len(point_list) - len(usable)
    if len(usable) < MIN_VIEWS_FOR_TRIANGULATION:
        if not usable:
            return None
        if excluded:
            print(f"triangulate_point: only {len(usable)}/{len(point_list)} views in bounds, "
                  "falling back to using every view for this point.")
        usable = [(cam_index, point) for cam_index, point, _ in point_list]

    if orig_width and orig_height and len(usable) > MIN_VIEWS_FOR_OUTLIER_REJECTION:
        usable, dropped = _reject_outlier_views(usable, intrinsic, extrinsics, orig_width, orig_height)
        for cam_index, residual, z in dropped:
            print(f"triangulate_point: excluding outlier camera_index={cam_index} "
                  f"(leave-one-out reprojection residual={residual:.4f}, z={z:.2f})")

    return _least_squares_triangulate(usable, intrinsic, extrinsics)

def keypoint_to_3d(actor, sequence, cam_list, frame_id, num_vertices=74):
    settings = get_settings()
    root_path = settings.landmarks_root / actor / sequence
    
    points_list = [[] for _ in range(num_vertices)]
    extrinsics = []

    # Need to get width and height to convert from [0, 1] to [0, width/height]
    video_path = settings.data_root / actor / "sequences" / sequence / "images" / "cam_222200037.mp4"
    cap = cv2.VideoCapture(video_path)
    success, frame = cap.read()
    orig_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    cam_data_path = settings.data_root / actor / "calibration" / "camera_params.json"
    cam_data = None
    with open(cam_data_path, "r") as f:
        cam_data = json.load(f)
        
    intrinsic = cam_data["intrinsics"]

    for cam_index, cam in enumerate(cam_list):
        cam_path = root_path / ("cam_" + cam) / f"frame_{int(frame_id):06d}.json"

        data = None
        with open(cam_path, "r") as f:
            data = json.load(f)


        extrinsics.append(np.array(cam_data["world_2_cam"][cam]))

        for point_data in data["points"]:
            point = np.array([round(point_data["x"] * orig_width), round(point_data["y"] * orig_height)])
            points_list[point_data['index']].append((cam_index, point, is_in_bounds(point_data)))

    data, untriangulated_indices = triangulate(intrinsic, extrinsics, points_list, orig_width, orig_height)
    if untriangulated_indices:
        print(f"keypoint_to_3d: could not triangulate indices {untriangulated_indices} (no in-bounds view)")
    with open("test.json", "w") as f:
        json.dump(data, f)

def triangulate(intrinsic, extrinsics, points_list, orig_width=None, orig_height=None):
    '''
        Triangulates N list of K 2D points into 3D points, given N camera views.

        Args:
            intrinsic: A 3x3 intrinsic matrix shared by every camera, or an
                (N,3,3) array/list of per-camera intrinsics (see
                _intrinsic_for) -- FaceScape's cameras each have their own
                distinct focal length/resolution, unlike Nersemble's single
                shared rig intrinsic.
            extrinsics: An N-list of 4x4 extrinsic matrices (one per camera)
            point_List: An NxKx2 list of K 2D points for each camera view
            orig_width/orig_height: raw frame size in pixels, each a single
                shared value or a per-camera list/array (see _resolution_for)
                -- when given, enables leave-one-out outlier rejection (see
                triangulate_point)

        Returns:
            (formatted, untriangulated_indices):
              - formatted: a list of dict containing {x, y, z}, same length/
                position as points_list -- points at untriangulated_indices get
                a (0, 0, 0) placeholder (never meant to be used as-is; the
                caller is expected to drop those rows from whatever
                correspondence map this feeds, e.g. via run_wrap_script.py's
                wrap_mesh(extra_indices_to_remove=...)).
              - untriangulated_indices: positions where triangulate_point()
                returned None (no view had an in-bounds observation at all --
                see its own docstring).
    '''
    triangulated_points = []
    untriangulated_indices = []
    for idx, point_list in enumerate(points_list):
        result = triangulate_point(point_list, intrinsic, extrinsics, orig_width, orig_height)
        if result is None:
            untriangulated_indices.append(idx)
            result = np.zeros(3)
        triangulated_points.append(result)

    #np.save("triangulate_point.npy", np.array(triangulated_points))

	# For each vertex, create a tiny triangle around it
    # eps = 1e-3
    # all_vertices = []
    # faces = []
    # for i, v in enumerate(triangulated_points):
    #     v1 = v
    #     v2 = [v[0] + eps, v[1], v[2]]
    #     v3 = [v[0], v[1] + eps, v[2]]
    #     idx = len(all_vertices)
    #     all_vertices.extend([v1, v2, v3])
    #     faces.append([idx, idx+1, idx+2])

    # with open('triangulate_point.obj', 'w') as f:
    #     for v in all_vertices:
    #         f.write(f'v {v[0]} {v[1]} {v[2]}\n')
    #     for face in faces:
    #         # OBJ face indices are 1-based
    #         f.write(f'f {face[0]+1} {face[1]+1} {face[2]+1}\n')

    formatted = [{"x": v[0], "y": v[1], "z": v[2]} for v in triangulated_points]
    return formatted, untriangulated_indices


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("actor_id", nargs="?", default=None, help="Actor identifier; omit to process all actors")
    parser.add_argument("sequence_id", nargs="?", default=None, help="Sequence identifier; omit to process all sequences")
    parser.add_argument("frame_id", nargs="?", default=None, help="Sequence identifier; omit to process all sequences")
    parser.add_argument("--cameras", nargs="*", default=None, help="Optional list of camera ids to process")
    args = parser.parse_args()
    cameras = args.cameras
    if(args.cameras is None):
        cameras = [
            "220700191",
            "221501007",
            "222200036",
            "222200037",
            "222200038",
            "222200039",
            "222200040",
            "222200041",
            "222200042",
            "222200043",
            "222200044",
            "222200045"
        ]
    start = time.time()
    keypoint_to_3d(args.actor_id, args.sequence_id, cameras, args.frame_id, 74)
	
    end = time.time()
    print("Estimated time: ", end-start)

if __name__ == "__main__":
    main()
