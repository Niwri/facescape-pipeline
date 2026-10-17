import nersemble_paths
from blendshape import *
from numpy.typing import NDArray
import numpy as np
import torch
import argparse
from scipy.spatial.transform import Rotation as R
import math
import json
from flame_utils import batch_rodrigues
import time

def write_obj(path, vertices, faces):
    """
    vertices: (N, 3) float array
    faces:    (F, 3) int array, 0-based indexing
    """
    with open(path, "w") as f:
        # Write vertices
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")

        # Write faces (OBJ uses 1-based indexing)
        for tri in faces:
            i, j, k = tri + 1
            f.write(f"f {i} {j} {k}\n")


def rotation_matrix_yaw_pitch_roll(yaw, pitch, roll):
    """
    Create a 3D rotation matrix from yaw, pitch, and roll (in radians).
    Yaw: rotation around y-axis
    Pitch: rotation around x-axis
    Roll: rotation around z-axis
    """
    # Rotation matrices for each axis
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(pitch), -np.sin(pitch)],
        [0, np.sin(pitch), np.cos(pitch)]
    ])
    Ry = np.array([
        [np.cos(yaw), 0, np.sin(yaw)],
        [0, 1, 0],
        [-np.sin(yaw), 0, np.cos(yaw)]
    ])
    Rz = np.array([
        [np.cos(roll), -np.sin(roll), 0],
        [np.sin(roll), np.cos(roll), 0],
        [0, 0, 1]
    ])
    # Combined rotation: R = Rz * Ry * Rx
    R = Rz @ Ry @ Rx
    return R


# --v2: tracking fits (fit.npz) sampled every V2_FIT_STEP-th raw frame --
# n_timesteps counts raw frames, while expr/rot/tra/eye_rot hold one row per
# fitted frame. Raw frame f uses the fit of the nearest multiple of
# V2_FIT_STEP, i.e. row round(f / V2_FIT_STEP). Same as
# FaceView/scripts/preprocess/process_sequence_sapiens.py --v2.
V2_FIT_STEP = 3


def v2_fit_index(raw_frame_idx, num_fit_frames):
    return min(int(round(int(raw_frame_idx) / V2_FIT_STEP)), num_fit_frames - 1)


def render_expression_mesh(actor_id, sequence_id, frame_id, v2=False):
    """Writes output/<actor>/<sequence>/<frame_id>/expression.obj from the FLAME
    fit. frame_id is the raw frame; with v2 the fit row is v2_fit_index(frame_id)."""

    start = time.time()

    flame_params_path = f"{nersemble_paths.NERSEMBLE_TRACKING_ROOT}/{actor_id}/sequences/{sequence_id}/fit.npz"
    if not os.path.exists(flame_params_path):
        # Same fallback as process_sequence_sapiens.py's _get_flame_model.
        flame_params_path = f"{nersemble_paths.NERSEMBLE_TRACKING_LOCAL_ROOT}/{actor_id}/sequences/{sequence_id}/fit.npz"

    assert os.path.exists(flame_params_path) == True, f"flame_render.py: File path {flame_params_path} does not exist"

    data = np.load(flame_params_path, allow_pickle=True)
    num_frames = data['n_timesteps'].item()
    '''
    ['fx', 'fy', 'cx', 'cy', 'extr', 'shape', 'expr', 'rot', 'tra',
    'eye_rot', 'camera_order', 'fps', 'error', 'n_timesteps', 'n_views',
    'valid_mask', 'crop_boxes', 'resolutions']
    '''

    flameBS = FLAMEBlendshapes(data['shape'].shape[0], data['expr'].shape[1])

    flameBS.shape_params = torch.as_tensor(data['shape'], dtype=torch.float32).unsqueeze(0).to(flameBS.flame.device)

    frame_id = int(frame_id)
    output_frame_id = frame_id  # expression.obj stays under the raw frame's folder
    # A v2 fit has fewer rows than raw frames (e.g. 125 rows for n_timesteps=373). Detect it here so
    # callers that don't pass v2 (run_pipeline_all.py's process_frame) don't index it by raw frame.
    if not v2 and data['expr'].shape[0] < num_frames:
        v2 = True
    if v2:
        frame_id = v2_fit_index(frame_id, data['expr'].shape[0])  # fit row from here on
    flameBS.eye_pose_params = torch.as_tensor(
        np.concatenate([data['eye_rot'][frame_id], data['eye_rot'][frame_id]]) if data['eye_rot'].shape[1] == 3 else data['eye_rot'][frame_id]
        , dtype=torch.float32
    ).to(flameBS.flame.device)

    temp_translation = np.array(data['tra'][frame_id], copy=True) # Translation


    translation = np.zeros(data['tra'][frame_id].shape)
    translation[0] = temp_translation[0]
    translation[1] = -temp_translation[1]
    translation[2] = -temp_translation[2]

    rotation = np.zeros(data['rot'][frame_id].shape)
    temp_rotation = np.array(data['rot'][frame_id], copy=True)

    # rotation[0] = temp_rotation[0]
    rotation[0] = 1 * ((temp_rotation[0] - math.pi) if temp_rotation[0] > 0 else (temp_rotation[0] + math.pi))
    rotation[1] = 2/3 * temp_rotation[2]
    rotation[2] = -0.5 * temp_rotation[1]

    V = flameBS.eval(weights=np.concatenate([data['expr'][frame_id], [0, 0, 0]]))
    R = rotation_matrix_yaw_pitch_roll(rotation[1], rotation[0], rotation[2])
    V = V @ R.T
    V += translation

    # Remove all vertices and faces directly or indirectly connected to 4000 and 4600
    remove_indices = [4000, 4600]
    # Build adjacency list for mesh
    adjacency = {i: set() for i in range(V.shape[0])}
    for tri in flameBS.F:
        for i in range(3):
            adjacency[tri[i]].update([tri[(i+1)%3], tri[(i+2)%3]])

    # BFS to find all connected vertices
    to_remove = set(remove_indices)
    queue = list(remove_indices)
    while queue:
        v = queue.pop(0)
        for neighbor in adjacency[v]:
            if neighbor not in to_remove:
                to_remove.add(neighbor)
                queue.append(neighbor)

    # Create mask for vertices to keep
    keep_mask = np.ones(V.shape[0], dtype=bool)
    for idx in to_remove:
        if idx < V.shape[0]:
            keep_mask[idx] = False

    # Map old indices to new indices
    old_to_new = {}
    new_idx = 0
    for i, keep in enumerate(keep_mask):
        if keep:
            old_to_new[i] = new_idx
            new_idx += 1

    V_filtered = V[keep_mask]
    F_filtered = []
    for tri in flameBS.F:
        if all(keep_mask[tri]):
            # Remap indices
            F_filtered.append([old_to_new[tri[0]], old_to_new[tri[1]], old_to_new[tri[2]]])
    F_filtered = np.array(F_filtered)

    os.makedirs(f"{nersemble_paths.OUTPUT_ROOT}/{actor_id}", exist_ok = True)
    os.makedirs(f"{nersemble_paths.OUTPUT_ROOT}/{actor_id}/{sequence_id}", exist_ok = True)
    os.makedirs(f"{nersemble_paths.OUTPUT_ROOT}/{actor_id}/{sequence_id}/{output_frame_id}", exist_ok = True)
    obj_path = f"{nersemble_paths.OUTPUT_ROOT}/{actor_id}/{sequence_id}/{output_frame_id}/expression.obj"
    write_obj(obj_path, V_filtered, F_filtered)
    end = time.time()
    print("Estimated time: ", end-start)
    print("Saved mesh to: ", obj_path)
    return obj_path

def main():
    actor_id = 240
    sequence_id = "EMO-1-shout+laugh"
    frame_id = 185
    render_expression_mesh(actor_id, sequence_id, frame_id)

if __name__ == "__main__":
    main()
