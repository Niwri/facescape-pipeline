"""One-shot script: renders a single, fixed FLAME mesh with zero shape and
zero expression parameters, and saves it as wrap_script_facescape/neutral.obj.

FaceScape counterpart to wrap_script/render_flame_exp_without_eyes.py's
render_expression_mesh() -- but FaceScape has no per-actor FLAME fit (no
fit.npz), so there's no per-actor/per-frame shape/expression/pose to load.
FLAMEBlendshapes already zero-initializes shape_params/exp_params/pose_params/
jaw_params/eye_pose_params in its own __init__, so calling .eval() with no
weights already produces exactly this neutral mesh -- no rotation/translation
handling needed at all (unlike render_expression_mesh, which converts a
per-frame FLAME pose into a yaw/pitch/roll rotation + translation).

The eye-region removal (BFS from seed vertices 4000/4600, same as
render_flame_exp_without_eyes.py) is kept: it's a topological edit independent
of shape/expression parameters, and this script is meant to be a drop-in
structural equivalent for FaceScape's single, actor/expression-independent
wrap template.

Run once:
    python3 render_flame_neutral.py
Output: wrap_script_facescape/neutral.obj (reused, unchanged, for every
FaceScape actor/expression pair's wrap -- see run_pipeline.py).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import facescape_paths as fp  # all paths: wrap_script_facescape/.env

# blendshape.py / flame_utils.py / the FLAME model pickle files all live in
# wrap_script/, not duplicated here -- reuse them directly rather than copying
# ~GB of model data.
WRAP_SCRIPT_ROOT = fp.WRAP_SCRIPT_ROOT
sys.path.insert(0, str(WRAP_SCRIPT_ROOT))

from blendshape import FLAMEBlendshapes  # noqa: E402

OUTPUT_PATH = Path(__file__).resolve().parent / "neutral.obj"

# Purely a blendshape-basis dimensionality choice -- with all-zero weights
# (FLAMEBlendshapes' own default), the resulting vertex positions are
# identical regardless of these values; the mesh topology itself (vertex/face
# count and connectivity) comes from the fixed FLAME model files, not from
# n_shape/n_exp. Matches the common FLAME2020 defaults used elsewhere in this
# codebase's fit.npz files.
N_SHAPE = 300
N_EXP = 100


def write_obj(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    with open(path, "w") as f:
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for tri in faces:
            i, j, k = tri + 1
            f.write(f"f {i} {j} {k}\n")


def main() -> None:
    flameBS = FLAMEBlendshapes(N_SHAPE, N_EXP)
    V = flameBS.eval()  # weights=None -> FLAMEBlendshapes' own zero-initialized shape/exp/pose/jaw params

    # Remove all vertices/faces directly or indirectly connected to the eye
    # region (seed vertices 4000/4600) -- identical logic to
    # render_flame_exp_without_eyes.py, since eyes are handled separately
    # (not part of this wrap pass) for Nersemble too.
    remove_indices = [4000, 4600]
    adjacency = {i: set() for i in range(V.shape[0])}
    for tri in flameBS.F:
        for i in range(3):
            adjacency[tri[i]].update([tri[(i + 1) % 3], tri[(i + 2) % 3]])

    to_remove = set(remove_indices)
    queue = list(remove_indices)
    while queue:
        v = queue.pop(0)
        for neighbor in adjacency[v]:
            if neighbor not in to_remove:
                to_remove.add(neighbor)
                queue.append(neighbor)

    keep_mask = np.ones(V.shape[0], dtype=bool)
    for idx in to_remove:
        if idx < V.shape[0]:
            keep_mask[idx] = False

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
            F_filtered.append([old_to_new[tri[0]], old_to_new[tri[1]], old_to_new[tri[2]]])
    F_filtered = np.array(F_filtered)

    write_obj(OUTPUT_PATH, V_filtered, F_filtered)
    print(f"Saved neutral mesh to: {OUTPUT_PATH} ({V_filtered.shape[0]} vertices, {F_filtered.shape[0]} faces)")


if __name__ == "__main__":
    main()
