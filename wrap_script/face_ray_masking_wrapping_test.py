import os
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from pytorch3d.io import load_objs_as_meshes
from tqdm import tqdm
import cv2
from segment import FaceSegmentation, BackgroundSegmentation
import time
from collections import defaultdict

import argparse
from pytorch3d.renderer import MeshRasterizer, RasterizationSettings, PerspectiveCameras, PointLights, MeshRenderer, SoftPhongShader, TexturesVertex

def ray_triangle_intersection(ray_origins, ray_directions, vertices, faces, epsilon=1e-6):
    """
    Moller-Trumbore ray-triangle intersection algorithm.
    Returns only the closest intersection for each ray.
    
    Args:
        ray_origins: (N, 3) tensor of ray origins
        ray_directions: (N, 3) tensor of ray directions (normalized)
        vertices: (V, 3) tensor of mesh vertices
        faces: (F, 3) tensor of face indices
        epsilon: small value to avoid division by zero
    
    Returns:
        hit_face_indices: list of face indices that were hit by the first intersection of any ray
    """
    device = ray_origins.device
    N = ray_origins.shape[0]
    F = faces.shape[0]
    
    print(f"    Processing {N} rays against {F} faces = {N*F} ray-triangle tests")
    
    # Get triangle vertices: (F, 3, 3) - F triangles, each with 3 vertices of 3 coords
    v0 = vertices[faces[:, 0]]  # (F, 3)
    v1 = vertices[faces[:, 1]]  # (F, 3)
    v2 = vertices[faces[:, 2]]  # (F, 3)
    
    # Compute edges
    edge1 = v1 - v0  # (F, 3)
    edge2 = v2 - v0  # (F, 3)
    
    hit_faces = set()
    
    # Process rays in batches to avoid memory issues
    # Larger batch size for better GPU utilization
    batch_size = 5000
    for i in range(0, N, batch_size):
        end_i = min(i + batch_size, N)
        batch_origins = ray_origins[i:end_i]  # (B, 3)
        batch_dirs = ray_directions[i:end_i]  # (B, 3)
        B = batch_origins.shape[0]
        
        # Use broadcasting without explicit expand (more efficient)
        ro = batch_origins.unsqueeze(1)  # (B, 1, 3)
        rd = batch_dirs.unsqueeze(1)  # (B, 1, 3)
        
        e1 = edge1.unsqueeze(0)  # (1, F, 3)
        e2 = edge2.unsqueeze(0)  # (1, F, 3)
        v0_exp = v0.unsqueeze(0)  # (1, F, 3)
        
        # Moller-Trumbore algorithm - use broadcasting instead of expand
        h = torch.cross(rd, e2, dim=2)  # (B, F, 3) via broadcasting
        a = (e1 * h).sum(dim=2)  # (B, F)
        
        # Check if ray is parallel to triangle OR hitting back face (back-face culling)
        # When a < epsilon, the ray either hits the back face or is parallel
        parallel_mask = a < epsilon  # (B, F)
        
        f = 1.0 / (a + epsilon)
        s = ro - v0_exp  # (B, F, 3) via broadcasting
        u = f * (s * h).sum(dim=2)  # (B, F)
        
        # Check u bounds
        u_mask = (u < 0.0) | (u > 1.0)  # (B, F)
        
        q = torch.cross(s, e1, dim=2)  # (B, F, 3)
        v = f * (rd * q).sum(dim=2)  # (B, F)
        
        # Check v bounds and u+v bounds
        v_mask = (v < 0.0) | (u + v > 1.0)  # (B, F)
        
        t = f * (e2 * q).sum(dim=2)  # (B, F)
        
        # Check t > epsilon (intersection in front of ray)
        t_mask = t <= epsilon  # (B, F)
        
        # Combine all masks - valid intersections are where all masks are False
        valid = ~(parallel_mask | u_mask | v_mask | t_mask)  # (B, F)
        
        # Debug: print statistics
        if i == 0:  # Only for first batch
            print(f"    Valid intersections in first batch: {valid.sum().item()} out of {B * F}")
            print(f"    Parallel: {parallel_mask.sum().item()}, u_mask: {u_mask.sum().item()}, v_mask: {v_mask.sum().item()}, t_mask: {t_mask.sum().item()}")
            if valid.any():
                valid_t = t[valid]
                print(f"    Valid t range: min={valid_t.min().item():.6f}, max={valid_t.max().item():.6f}")
        
        # For each ray, find the closest intersection (minimum t)
        # Set invalid intersections to inf so they don't affect the min
        t_filtered = t.clone()
        t_filtered[~valid] = float('inf')
        
        # Get minimum t for each ray
        min_t, min_face_idx = t_filtered.min(dim=1)  # (B,)
        
        # Only keep rays that actually hit something (min_t != inf)
        hit_mask = min_t != float('inf')
        hit_face_idx = min_face_idx[hit_mask]
        
        hit_faces.update(hit_face_idx.cpu().numpy().tolist())
    
    return list(hit_faces)

def find_connected_face_components(faces, face_indices_set):
    """
    Find connected components among a set of face indices.
    Two faces are connected if they share at least one vertex.
    
    Args:
        faces: (F, 3) tensor of face indices (all faces in mesh)
        face_indices_set: set of face indices to consider
    
    Returns:
        list of sets, where each set is a connected component of face indices
    """
    if not face_indices_set:
        return []
    
    # Build adjacency list: face_idx -> set of adjacent face indices
    # Two faces are adjacent if they share at least one vertex
    vertex_to_faces = defaultdict(set)
    for face_idx in face_indices_set:
        v0, v1, v2 = faces[face_idx].tolist()
        vertex_to_faces[v0].add(face_idx)
        vertex_to_faces[v1].add(face_idx)
        vertex_to_faces[v2].add(face_idx)
    
    adjacency = defaultdict(set)
    for face_idx in face_indices_set:
        v0, v1, v2 = faces[face_idx].tolist()
        # All faces sharing any of these vertices are adjacent
        neighbors = vertex_to_faces[v0] | vertex_to_faces[v1] | vertex_to_faces[v2]
        neighbors.discard(face_idx)  # Don't include self
        adjacency[face_idx] = neighbors
    
    # Find connected components using BFS
    visited = set()
    components = []
    
    for start_face in face_indices_set:
        if start_face in visited:
            continue
        
        # BFS from start_face
        component = set()
        queue = [start_face]
        visited.add(start_face)
        
        while queue:
            face = queue.pop(0)
            component.add(face)
            
            for neighbor in adjacency[face]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        components.append(component)
    
    return components

def erode_hit_faces(faces, hit_faces_set, iterations=1):
    """
    Shrinks hit_faces_set by removing its boundary faces, where a face counts
    as boundary if it shares any vertex (point, edge, or face adjacency) with
    a face outside the set. Repeats for `iterations` passes.

    Args:
        faces: (F, 3) tensor/array of face indices (all faces in mesh)
        hit_faces_set: set of face indices considered "hit"
        iterations: number of boundary layers to strip

    Returns:
        set of face indices with `iterations` boundary layers removed
    """
    if not hit_faces_set:
        return set(hit_faces_set)

    faces_np = faces.cpu().numpy() if torch.is_tensor(faces) else np.asarray(faces)

    # Build adjacency list: vertex_idx -> set of faces using that vertex
    vertex_to_faces = defaultdict(set)
    for face_idx, (v0, v1, v2) in enumerate(faces_np):
        vertex_to_faces[v0].add(face_idx)
        vertex_to_faces[v1].add(face_idx)
        vertex_to_faces[v2].add(face_idx)

    eroded = set(hit_faces_set)
    for _ in range(iterations):
        boundary_faces = set()
        for face_idx in eroded:
            v0, v1, v2 = faces_np[face_idx]
            neighbors = vertex_to_faces[v0] | vertex_to_faces[v1] | vertex_to_faces[v2]
            neighbors.discard(face_idx)
            if not neighbors.issubset(eroded):
                boundary_faces.add(face_idx)
        eroded -= boundary_faces

    return eroded

def load_camera_params(json_path):
    with open(json_path, 'r') as f:
        params = json.load(f)
    world_2_cam = params['world_2_cam']
    K = np.array(params['intrinsics'])
    return world_2_cam, K

def get_camera_image_paths(participant_id, sequence_id, frame_id):
    """
    Returns a dict {camera_id: image_path} for the given frame_id.
    Assumes images are in data/{participant_id}/sequences/{sequence_id}/images/cam_*.mp4
    """
    input_dir = os.path.join("/scratch", "local", "2023", "irwinngo", "data", participant_id, "sequences", sequence_id, "images")
    cam_images = {}
    for fname in os.listdir(input_dir):
        if fname.lower().endswith(".mp4"):
            cam_id = fname.replace("cam_", "").replace(".mp4", "")
            mp4_path = os.path.join(input_dir, fname)
            cap = cv2.VideoCapture(mp4_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_id < total_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                ret, frame = cap.read()
                if ret:
                    cam_images[cam_id] = frame
            cap.release()
    return cam_images

# SAM (ViT-H) and U2Net are built on first use instead of at import: in steady
# state the pipeline reuses the neutral frame's mask and never calls
# face_ray_masking(), so loading them eagerly only held GPU memory (and made a
# --cpu-only run touch the GPU at import).
_segmenter = None
_bg_segmenter = None


# Nersemble camera groups used to weight each camera's vote (front 1.0, sides
# 1.5) -- moved here from run_pipeline_all.process_frame() along with the
# masking itself.
NERSEMBLE_FRONT_CAMERAS = {'222200037', '222200038', '220700191', '222200036',
                           '222200041', '222200043', '222200047'}
NERSEMBLE_SIDE_RIGHT_CAMERAS = {'221501007', '222200039', '222200045', '222200049'}
NERSEMBLE_SIDE_LEFT_CAMERAS = {'222200040', '222200042', '222200044', '222200046', '222200048'}

# Landmark index used as SAM's point prompt: nosebridge (upper nose, above
# where a beard/mustache reaches) -- same index as
# wrap_script_ava256/face_ray_masking_ava256.py's prompt anchor.
SAM_PROMPT_LANDMARK_INDEX = 57
# SAM's positive point prompts for the actor face mask (same as
# scan_face_masking.SAM_PROMPT_INDICES and Ava-256): nosebridge (57), outer
# eyebrow ends (30, 47), lip corners (18, 37) and ears (1, 4, 6, 14).
SAM_PROMPT_INDICES = (57, 30, 47, 18, 37, 1, 4, 6, 14)


def load_nosebridge_sam_hints(landmarks_dir, frame_id, cam_ids):
    """{cam_id: (x, y)} normalized nosebridge landmark for this frame, per
    camera: the propagated nosebridge_frame_<idx>.json point when present,
    else sapiens2's frame_<idx>.json point at the same index (merged by
    propagated_landmarks.load_camera_standard_points). Cameras with neither,
    or whose point is off-image, are left out -- face_ray_masking falls back
    to U2Net's centroid for them. frame_id is the raw frame index."""
    from propagated_landmarks import load_camera_standard_points

    hints = {}
    frame_tag = f"{int(frame_id / 5):06d}"
    for cam_id in cam_ids:
        for point in load_camera_standard_points(Path(landmarks_dir) / f"cam_{cam_id}", frame_tag) or []:
            if point["index"] == SAM_PROMPT_LANDMARK_INDEX and 0.0 < point["x"] < 1.0 and 0.0 < point["y"] < 1.0:
                hints[cam_id] = (float(point["x"]), float(point["y"]))
    return hints


def load_sam_prompt_hints(landmarks_dir, frame_id, cam_ids, visible=None):
    """{cam_id: [(x, y), ...]} normalized SAM_PROMPT_INDICES landmarks for this
    frame, per camera, from the same files as load_nosebridge_sam_hints. A
    landmark the camera has no on-image point for is left out of that
    camera's prompt; cameras with none at all are left out -- face_ray_masking
    falls back to U2Net's centroid for them. frame_id is the raw frame index.
    visible: optional {cam_id: indices that camera may use} (scan_face_masking.
    prompt_visibility -- drops occluded eyebrow / ear points)."""
    from propagated_landmarks import load_camera_standard_points

    hints = {}
    frame_tag = f"{int(frame_id / 5):06d}"
    for cam_id in cam_ids:
        by_index = {p["index"]: p for p in load_camera_standard_points(Path(landmarks_dir) / f"cam_{cam_id}", frame_tag) or []}
        allowed = SAM_PROMPT_INDICES if visible is None else [i for i in SAM_PROMPT_INDICES if i in visible.get(cam_id, ())]
        points = [(float(by_index[i]["x"]), float(by_index[i]["y"])) for i in allowed
                  if i in by_index and 0.0 < by_index[i]["x"] < 1.0 and 0.0 < by_index[i]["y"] < 1.0]
        if points:
            hints[cam_id] = points
    return hints


FACE_MASK_SUFFIXES = (".txt", "_inverse.txt", "_wrapping_node.txt", "_wrapping_node_inverse.txt")


def finalize_face_mask(faces, hit_faces, all_faces):
    """The shared end of every face mask: intersect the ray/SAM hit set with
    the reference region (wrap_mask.txt), erode one boundary layer, and derive
    the Wrapping node's own selection (wrapping_node_mask.txt). Reference
    files are opened relative to the cwd (wrap_script/). Returns
    (not_hit, hit, wrapping_node_selection, wrapping_node_invisible) -- the
    four lists write_face_mask_files() stores."""
    all_faces = set(all_faces)
    # Intersect with the reference hit list in wrap_mask.txt, so a frame's segmentation
    # errors can never mark a face outside the established face region as hit.
    wrap_mask_path = "wrap_mask.txt"
    with open(wrap_mask_path, "r") as f:
        reference_not_hit_faces = json.load(f)
        reference_hit_faces = all_faces - set(reference_not_hit_faces)
    hit_faces = set(hit_faces) & reference_hit_faces
    print(f"Faces after intersecting with {wrap_mask_path}: {len(hit_faces)}")

    # Shrink the mask by stripping boundary faces, so triangles right at the
    # edge of the segmentation don't get kept from a marginal/noisy hit.
    MASK_EROSION_ITERATIONS = 1
    hit_faces = erode_hit_faces(faces, hit_faces, iterations=MASK_EROSION_ITERATIONS)
    print(f"Faces after eroding mask boundary ({MASK_EROSION_ITERATIONS} iteration(s)): {len(hit_faces)}")

    not_hit_faces = sorted(all_faces - hit_faces)
    print(f"Final not_hit_faces count: {len(not_hit_faces)}")

    # Third mask, for the Wrapping node's own polygon selection specifically --
    # same "intersect with a maintained reference file" pattern as wrap_mask.txt
    # above, just a separate reference (wrapping_node_mask.txt) and applied to
    # the final hit_faces (post wrap_mask.txt intersection + erosion) rather
    # than the raw ray/SAM-weighted hit set.
    wrapping_node_mask_path = "wrapping_node_mask.txt"
    with open(wrapping_node_mask_path, "r") as f:
        reference_not_wrapping_node_faces = json.load(f)
        reference_wrapping_node_faces = all_faces - set(reference_not_wrapping_node_faces)
    wrapping_node_invisible = hit_faces & reference_wrapping_node_faces
    wrapping_node_selection = sorted(all_faces - wrapping_node_invisible)
    print(f"Faces after intersecting with {wrapping_node_mask_path}: {len(wrapping_node_invisible)} "
          f"(invisible to Wrapping node), {len(wrapping_node_selection)} (selected)")

    return (list(not_hit_faces), sorted(hit_faces), wrapping_node_selection, sorted(wrapping_node_invisible))


def write_face_mask_files(face_ray_path, masks):
    """Writes the four face mask files (face_ray_path + FACE_MASK_SUFFIXES) in
    the formats run_wrap_script_wrapping_test.wrap_mesh() reads, each through
    a temp file + rename so a concurrent reader never sees a partial file."""
    face_list, face_not_list, wrapping_node_face_list, wrapping_node_face_not_list = masks
    contents = {
        ".txt": str(list(face_list)) + "\n",
        "_inverse.txt": json.dumps([int(i) for i in face_not_list]),
        "_wrapping_node.txt": str(list(wrapping_node_face_list)) + "\n",
        "_wrapping_node_inverse.txt": json.dumps([int(i) for i in wrapping_node_face_not_list]),
    }
    for suffix in FACE_MASK_SUFFIXES:
        final_path = face_ray_path + suffix
        tmp_path = f"{final_path}.{os.getpid()}.tmp"
        with open(tmp_path, "w") as f:
            f.write(contents[suffix])
        os.replace(tmp_path, final_path)


def face_mask_files_exist(face_ray_path):
    return all(os.path.exists(face_ray_path + suffix) for suffix in FACE_MASK_SUFFIXES)


def release_segmenters():
    """Drops the SAM/U2Net instances so their GPU memory can be freed (callers
    follow with torch.cuda.empty_cache())."""
    global _segmenter, _bg_segmenter
    _segmenter = None
    _bg_segmenter = None


def _get_segmenters():
    global _segmenter, _bg_segmenter
    if _segmenter is None:
        _segmenter = FaceSegmentation()
        _bg_segmenter = BackgroundSegmentation()
    return _segmenter, _bg_segmenter


def main(ACTOR_ID, SEQUENCE_ID, FRAME_ID, OBJ_PATH, CAMERA_PARAMS_PATH, OUTPUT_PATH):
    
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mesh = load_objs_as_meshes([OBJ_PATH], device=device)
    # Define camera weights: front view = 1.0, side views = 1.5
    FRONT_CAMERAS = {'222200037', '222200038', '220700191', '222200036', 
                     '222200041', '222200043', '222200047'}
    SIDE_RIGHT_CAMERAS = {'221501007', '222200039', '222200045', '222200049'}
    SIDE_LEFT_CAMERAS = {'222200040', '222200042', '222200044', '222200046', '222200048'}
    
    cam_images = get_camera_image_paths(ACTOR_ID, SEQUENCE_ID, FRAME_ID)
    world_2_cam, K = load_camera_params(CAMERA_PARAMS_PATH)

    not_hit_faces = face_ray_masking(mesh, FRONT_CAMERAS, SIDE_RIGHT_CAMERAS, SIDE_LEFT_CAMERAS, cam_images, world_2_cam, K)
    with open(OUTPUT_PATH, 'w') as f:
        f.write(str(list(not_hit_faces)) + "\n")
    
    return OUTPUT_PATH
    
def face_ray_masking(mesh, FRONT_CAMERAS, SIDE_RIGHT_CAMERAS, SIDE_LEFT_CAMERAS, cam_images, extrinsics, intrinsic,
                     sam_point_hints=None):
    """sam_point_hints: optional {cam_id: (x, y)} in normalized [0, 1] image
    coordinates -- the frame's nosebridge landmark per camera (see
    run_pipeline_all.load_nosebridge_sam_hints). When a camera has one, it is
    SAM's point prompt; otherwise that camera falls back to U2Net's
    largest-person-blob centroid (then the image center). Same idea as
    wrap_script_ava256/face_ray_masking_ava256.py: the person centroid sits
    around the chin/neck for a head+shoulders framing, which lands on the
    beard for bearded actors, so SAM segments the beard instead of the face."""
    sam_point_hints = sam_point_hints or {}
    segmenter, bg_segmenter = _get_segmenters()
    vertices = mesh.verts_list()[0]  # (V, 3)
    faces = mesh.faces_list()[0]  # (F, 3)
    num_faces = faces.shape[0]

    camera_weights = {}
    for cam in FRONT_CAMERAS:
        camera_weights[cam] = 1.0
    for cam in SIDE_RIGHT_CAMERAS | SIDE_LEFT_CAMERAS:
        camera_weights[cam] = 1.5
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    K_torch = torch.from_numpy(np.array(intrinsic)).float().to(device)  # Convert K to torch tensor on GPU
    K_inv_torch = torch.linalg.inv(K_torch)  # Precompute inverse on GPU
    
    face_to_weight = defaultdict(float)
    # --- Segmentation step (per camera, as before) ---
    masks = {}
    img_hw = {}
    t_start = time.time()
    for cam_id, image in tqdm(cam_images.items()):
        orig_h, orig_w = image.shape[:2]
        # Run segmentation at half resolution for speed, then upsample mask.
        seg_input = cv2.resize(image, (max(1, orig_w // 2), max(1, orig_h // 2)), interpolation=cv2.INTER_AREA)

        seg_h, seg_w = seg_input.shape[:2]
        hint = sam_point_hints.get(cam_id)
        # One (x, y) or a list of them (load_sam_prompt_hints), normalized.
        hint_points = [] if hint is None else ([hint] if np.isscalar(hint[0]) else list(hint))
        hint_points = [h for h in hint_points if 0.0 < h[0] < 1.0 and 0.0 < h[1] < 1.0]
        sam_points = None
        if hint_points:
            # The frame's own landmarks (normalized, so they map straight
            # onto the half-resolution seg_input) -- skips U2Net.
            sam_points = [(min(seg_w - 1, int(round(x * seg_w))), min(seg_h - 1, int(round(y * seg_h)))) for x, y in hint_points]
            sam_point = sam_points[0]
        else:
            # Fallback: U2Net's whole-person mask centroid, since the face
            # isn't always centered in the raw camera frame.
            person_mask = (bg_segmenter.get_mask(seg_input) > 0).astype(np.uint8)
            num_person_labels, person_labels, person_stats, person_centroids = cv2.connectedComponentsWithStats(person_mask, connectivity=8)
            if num_person_labels > 1:
                largest_person_label = 1 + np.argmax(person_stats[1:, cv2.CC_STAT_AREA])
                cx, cy = person_centroids[largest_person_label]
                sam_point = (int(round(cx)), int(round(cy)))
            else:
                sam_point = (seg_w // 2, seg_h // 2)

        mask = segmenter.get_face_mask(seg_input, point=sam_point, points=sam_points).astype(np.uint8)
        # Keep every connected region that contains one of this camera's
        # prompt points (e.g. an ear that shows up as its own island next to
        # the face); without prompt points, or if none lands inside the mask,
        # keep the largest region as before.
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num_labels > 1:
            prompted = {int(labels[y, x]) for x, y in (sam_points or []) if labels[y, x] > 0}
            if prompted:
                mask = np.isin(labels, sorted(prompted)).astype(np.uint8)
            else:
                largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                mask = (labels == largest_label).astype(np.uint8)

        if mask.shape != (orig_h, orig_w):
            mask = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            mask = (mask > 0).astype(np.uint8)

        # os.makedirs("test_output", exist_ok=True)
        # cv2.imwrite(f"test_output/{cam_id}_segment.png", mask * 255)
        masks[cam_id] = mask
        img_hw[cam_id] = mask.shape  # (h, w)
    # --- Prepare batched camera parameters for Pytorch3D ---
    valid_cam_ids = [cam_id for cam_id in cam_images if cam_id in extrinsics]
    if not valid_cam_ids:
        print("No valid cameras found in camera params.")
        return

    # All images must have the same size for batching in Pytorch3D
    # We'll use the first valid camera's size
    h, w = img_hw[valid_cam_ids[0]]
    img_size = torch.tensor([[w, h]] * len(valid_cam_ids), dtype=torch.int, device=device)
    fl = K_torch[[0,1],[0,1]].to(device)
    pp = K_torch[[0,1],[2,2]].to(device)
    scale = img_size[0].min() / 2.0
    c0 = img_size[0] / 2.0
    focal_pytorch3d = fl / scale
    p0_pytorch3d = -(pp - c0) / scale
    Rs, Ts, focals, pps = [], [], [], []
    for cam_id in valid_cam_ids:
        extr = torch.tensor(extrinsics[cam_id]).to(device)  # 4x4
        R, T = extr[:3, :3], extr[:3, 3]

        R_pytorch3d = R.clone().T
        T_pytorch3d = T.clone()
        R_pytorch3d[:, :2] *= -1
        T_pytorch3d[:2] *= -1
        Rs.append(R_pytorch3d)
        Ts.append(T_pytorch3d)
        focals.append(focal_pytorch3d)
        pps.append(p0_pytorch3d)

    Rs = torch.stack(Rs, dim=0)
    Ts = torch.stack(Ts, dim=0)
    focals = torch.stack(focals, dim=0)
    pps = torch.stack(pps, dim=0)

    cameras = PerspectiveCameras(
        device=device,
        focal_length=focals,
        principal_point=pps,
        R=Rs,
        T=Ts,
        image_size=img_size,
    )

    raster_settings = RasterizationSettings(
        image_size=(h, w),
        blur_radius=0.0,
        faces_per_pixel=1,
    )

    rasterizer = MeshRasterizer(
        cameras=cameras,
        raster_settings=raster_settings
    )
    meshes = mesh.extend(len(valid_cam_ids))
    fragments = rasterizer(meshes)
    num_faces_per_mesh = meshes.num_faces_per_mesh().cpu().numpy()
    face_offsets = np.concatenate(([0], np.cumsum(num_faces_per_mesh[:-1])))

    # --- Process each camera's output ---
    # Per face: how many cameras see it at all, and how many of those have it
    # inside their SAM mask -- for the relative vote below.
    seen_count = defaultdict(int)
    inside_count = defaultdict(int)
    for i, cam_id in enumerate(valid_cam_ids):
        mask = masks[cam_id]
        face_idx_map = fragments.pix_to_face[i, ..., 0].cpu().numpy()
        seen = face_idx_map[face_idx_map >= 0] - face_offsets[i]
        for face_idx in np.unique(seen[(seen >= 0) & (seen < num_faces)]):
            seen_count[face_idx] += 1
        # print(f'test_output/{cam_id}_face_idx_map.png')
        # plt.imsave(f'test_output/{cam_id}_face_idx_map.png', face_idx_map, vmin=-1, vmax=faces.shape[0])
        hit_face_idx = face_idx_map[mask > 0]
        hit_face_idx = hit_face_idx[hit_face_idx >= 0]
        hit_face_idx = hit_face_idx - face_offsets[i]
        hit_face_idx = hit_face_idx[(hit_face_idx >= 0) & (hit_face_idx < num_faces)]
        hit_face_idx_list = np.unique(hit_face_idx)
        cam_weight = camera_weights.get(cam_id, 1.0)
        for face_idx in hit_face_idx_list:
            face_to_weight[face_idx] += cam_weight
            inside_count[face_idx] += 1
        print(f"  Camera weight: {cam_weight}, Total faces hit so far: {len(face_to_weight)}")
    # Only keep faces that have cumulative weight >= MIN_WEIGHT
    # Kept when its weighted votes reach MIN_WEIGHT, OR (relative vote, as on
    # Ava-256) when more than RELATIVE_FRACTION of the cameras that see it at
    # all have it inside their SAM mask, with at least RELATIVE_MIN_CAMERAS
    # of them -- so faces few cameras can see (the side of the head, ears)
    # aren't held to the same absolute count as the front of the face.
    MIN_WEIGHT = 8.0  # Previously MIN_CAMERAS, now weighted threshold
    RELATIVE_FRACTION = 0.5
    RELATIVE_MIN_CAMERAS = 2
    by_weight = {face_idx for face_idx, weight in face_to_weight.items() if weight >= MIN_WEIGHT}
    by_fraction = {face_idx for face_idx, n_in in inside_count.items()
                   if n_in >= RELATIVE_MIN_CAMERAS and n_in > RELATIVE_FRACTION * seen_count[face_idx]}
    hit_faces = by_weight | by_fraction
    print(f"Faces by weight >= {MIN_WEIGHT}: {len(by_weight)}; by > {RELATIVE_FRACTION:.0%} of seeing cameras "
          f"(>= {RELATIVE_MIN_CAMERAS}): {len(by_fraction)}; union: {len(hit_faces)}")

    all_faces = set(range(num_faces))

    print(f"Faces with cumulative weight >= {MIN_WEIGHT}: {len(hit_faces)} out of {len(all_faces)}")

    return finalize_face_mask(faces, hit_faces, all_faces)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Face Ray Masking Script")
    parser.add_argument("actor_id", type=str, help="Actor ID")
    parser.add_argument("sequence_id", type=str, help="Sequence ID")
    parser.add_argument("frame_num", type=int, help="Frame number")
    parser.add_argument("--obj_path", type=str, default="expression.obj", help="Path to OBJ file")
    parser.add_argument("--output_path", type=str, default="face_list.txt", help="Output path for not_hit_faces")
    args = parser.parse_args()

    ACTOR_ID = args.actor_id
    SEQUENCE_ID = args.sequence_id
    FRAME_ID = args.frame_num
    OBJ_PATH = args.obj_path
    CAMERA_PARAMS_PATH = os.path.join("/scratch", "local", "2023", "irwinngo", "data", ACTOR_ID, "calibration", "camera_params.json")
    OUTPUT_PATH = args.output_path
    start = time.time()
    main(ACTOR_ID, SEQUENCE_ID, FRAME_ID, OBJ_PATH, CAMERA_PARAMS_PATH, OUTPUT_PATH)
    end = time.time()
    print("Estimated time: ", end-start)
