"""FaceScape counterpart to wrap_script/run_wrap_script.py's wrap_mesh() --
drives the same "Wrap" tool (WrapCmd) but against the template.wrap built for
FaceScape (BlendWrapping for landmark-guided retargeting, then a Wrapping ICP
refinement pass restricted to FACE_MASK_PATH's curated face-region polygons):
no face-ray-masking (the Nersemble SAM/U2Net image-segmentation step), no
per-frame FLAME "_2" refinement stage, no ignore_ears variant.

Unlike the Nersemble version, there's no whitelist.json-based restriction on
which FLAME indices participate -- FaceScape's own sapiens2 landmark map
already only covers a subset of the 74 standard indices (eyelid/lip, plus
nose/eyebrow/ear once/if that scope is turned on), and triangulate.py's
triangulate() already reports exactly which indices couldn't be triangulated
at all (no camera had an in-bounds observation -- guaranteed for every FLAME
index FaceScape has no landmark for). Dropping those rows via
extra_indices_to_remove is sufficient on its own; no separate allowlist is
needed on top of it.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
import facescape_paths as fp  # all paths: wrap_script_facescape/.env

SCRIPT_DIR = Path(__file__).resolve().parent
TEMPLATE_PATH = SCRIPT_DIR / "template.wrap"
DEFAULT_POT_PATH = SCRIPT_DIR / "neutral_correspondence_without_eyes_POT.json"
DEFAULT_NEUTRAL_MESH_PATH = SCRIPT_DIR / "neutral.obj"
# Curated face-region polygon selection (replaces the old every-face
# all_polygons.txt) -- restricts BlendWrapping/Wrapping to the actual face
# region instead of deforming the whole template mesh, part of the template
# fix that also added the Wrapping (ICP refinement) node below.
FACE_MASK_PATH = SCRIPT_DIR / "facescape_mask.txt"

# Same blendshape basis as the Nersemble pipeline (wrap_script/wrap_template/
# template.wrap's own BlendWrapping.blendshapeFileNames/neutralFileName) --
# same FLAME topology (3931 vertices, verified against our own neutral.obj),
# so directly reusable for retargeting FaceScape's wrap too.
# NERSEMBLE_BLENDSHAPE_ROOT is read FRESH inside wrap_mesh() (not baked into
# a precomputed NEUTRAL_PATH/FILE_NAMES pair here) so a caller on a different
# cluster can monkeypatch just this one constant after import (same pattern
# already used for TEMPLATE_PATH/FACE_MASK_PATH) -- e.g. run_pipeline.py's
# --blendshape-root flag.
NERSEMBLE_BLENDSHAPE_ROOT = fp.WRAP_SCRIPT_ROOT / "blendshape"
NERSEMBLE_BLENDSHAPE_FILE_BASENAMES = [
    "shape_6_neg.obj", "shape_7_neg.obj", "shape_7_pos.obj", "shape_8_neg.obj", "shape_8_pos.obj",
    "shape_9_neg.obj", "shape_9_pos.obj", "shape_3_neg.obj", "shape_3_pos.obj", "shape_4_neg.obj",
    "shape_4_pos.obj", "shape_5_neg.obj", "shape_5_pos.obj", "shape_6_pos.obj", "shape_0_neg.obj",
    "shape_0_pos.obj", "shape_1_neg.obj", "shape_1_pos.obj", "shape_2_neg.obj", "shape_2_pos.obj",
    "exp_8_neg.obj", "exp_9_neg.obj", "exp_9_pos.obj", "exp_4_neg.obj", "exp_5_neg.obj",
    "exp_5_pos.obj", "exp_6_neg.obj", "exp_6_pos.obj", "exp_7_neg.obj", "exp_7_pos.obj",
    "exp_8_pos.obj", "exp_1_neg.obj", "exp_1_pos.obj", "exp_2_neg.obj", "exp_2_pos.obj",
    "exp_3_neg.obj", "exp_3_pos.obj", "exp_4_pos.obj", "exp_0_neg.obj", "exp_0_pos.obj",
    "exp_14_pos.obj", "exp_14_neg.obj", "exp_15_pos.obj", "exp_15_neg.obj", "exp_16_pos.obj",
    "exp_16_neg.obj", "exp_18_pos.obj", "exp_18_neg.obj", "exp_24_pos.obj", "exp_24_neg.obj",
]


def wrap_mesh(
    target_mesh_path: str,
    target_correspondence_path: str,
    output_path: str,
    extra_indices_to_remove=None,
    neutral_mesh_path: str | Path = DEFAULT_NEUTRAL_MESH_PATH,
    pot_template_path: str | Path = DEFAULT_POT_PATH,
) -> None:
    """Deforms neutral_mesh_path (the fixed, zero-shape/zero-expression FLAME
    template -- see render_flame_neutral.py) onto target_mesh_path (this
    actor/expression pair's own scanned .ply, pre-converted to .obj by the
    caller) using triangulated-landmark correspondences, and saves the result
    to output_path."""
    # No wrap_script_facescape/.env.example of its own -- fall back to
    # wrap_script/'s (same WrapCmd/license, not per-directory config) rather
    # than relying on the caller's cwd like the Nersemble version does.
    # WrapCmd / WrapLicense: wrap_script_facescape/.env, loaded into os.environ by facescape_paths.
    wrap_cmd_raw = os.getenv("WrapCmd")
    wrap_license_raw = os.getenv("WrapLicense")
    if not wrap_cmd_raw or not wrap_license_raw:
        raise RuntimeError("WrapCmd or WrapLicense not found in .env file.")
    wrap_cmd = Path(wrap_cmd_raw)
    wrap_license = Path(wrap_license_raw)

    if not wrap_cmd.is_file():
        raise FileNotFoundError(f"WrapCmd not found: {wrap_cmd}")
    if not os.access(wrap_cmd, os.X_OK):
        raise PermissionError(f"WrapCmd is not executable: {wrap_cmd}")
    if not wrap_license.is_file():
        raise FileNotFoundError(f"Wrap license file not found: {wrap_license}")

    process_env = os.environ.copy()
    has_display = bool(process_env.get("DISPLAY") or process_env.get("WAYLAND_DISPLAY"))
    if not has_display and not process_env.get("QT_QPA_PLATFORM"):
        process_env["QT_QPA_PLATFORM"] = "offscreen"
        print("No display detected; using QT_QPA_PLATFORM=offscreen")

    with open(TEMPLATE_PATH, "r") as f:
        data = json.load(f)
    nodes = data["nodes"]

    nodes["NeutralInput"]["params"]["fileName"]["value"] = str(neutral_mesh_path)
    nodes["TargetInput"]["params"]["fileName"]["value"] = target_mesh_path
    nodes["SaveWrap"]["params"]["fileName"]["value"] = output_path
    # $PROJECT_DIR resolves to the per-pair output directory (where
    # output.wrap itself gets written below), not wrap_script_facescape/ --
    # needs an absolute path, same reasoning as neutral_mesh_path/pot_template_path.
    nodes["FaceMask"]["params"]["fileName"]["value"] = str(FACE_MASK_PATH)
    # Same blendshape basis as the Nersemble pipeline (retargeting enabled --
    # see NERSEMBLE_BLENDSHAPE_FILE_BASENAMES above), not a placeholder:
    # verified same FLAME topology (3931 vertices) as our own neutral.obj.
    # Derived fresh from NERSEMBLE_BLENDSHAPE_ROOT here (not precomputed at
    # import time) so a caller can monkeypatch just that one constant.
    blendshape_root = Path(NERSEMBLE_BLENDSHAPE_ROOT)
    nodes["BlendWrapping"]["params"]["neutralFileName"]["value"] = str(blendshape_root / "neutral.obj")
    nodes["BlendWrapping"]["params"]["blendshapeFileNames"]["value"] = [
        str(blendshape_root / name) for name in NERSEMBLE_BLENDSHAPE_FILE_BASENAMES
    ]

    # Rows that couldn't be triangulated at all (no camera had an in-bounds
    # observation -- see triangulate.py's triangulate_point()) are dropped
    # from BOTH correspondence files, positionally, so the remaining rows stay
    # aligned between neutral (POT) and target (per-pair triangulated) sides.
    indices_to_remove = sorted(set(extra_indices_to_remove or []))

    with open(pot_template_path, "r") as f:
        neutral_correspondence_data = np.array(json.load(f))
    neutral_correspondence_path = str(pot_template_path)
    if indices_to_remove:
        # Persisted alongside this pair's other output (keypoint_vertices.json,
        # scan.obj, wrapped_mesh.obj) rather than a deleted tempfile -- which
        # indices get dropped is pair-specific (depends on that pair's own
        # untriangulated_indices), so this filtered POT has no other record
        # once compute finishes, unlike target_correspondence_final_path below
        # (already a filtered view of keypoint_vertices.json, which run_pipeline.py
        # saves in full separately).
        filtered = np.delete(neutral_correspondence_data, indices_to_remove, axis=0)
        neutral_correspondence_path = os.path.join(os.path.dirname(output_path), "neutral_correspondence_filtered.json")
        with open(neutral_correspondence_path, "w") as f:
            json.dump(filtered.tolist(), f)

    with open(target_correspondence_path, "r") as f:
        target_correspondence_data = np.array(json.load(f))
    target_correspondence_final_path = target_correspondence_path
    if indices_to_remove:
        # Persisted next to keypoint_vertices.json (the unfiltered version)
        # rather than a deleted tempfile -- same reasoning as
        # neutral_correspondence_filtered.json above.
        filtered = np.delete(target_correspondence_data, indices_to_remove, axis=0)
        target_correspondence_final_path = os.path.join(os.path.dirname(output_path), "target_correspondence_filtered.json")
        with open(target_correspondence_final_path, "w") as f:
            json.dump(filtered.tolist(), f)

    nodes["CorrespondancePointsNeutral"]["params"]["fileName"]["value"] = neutral_correspondence_path
    nodes["CorrespondenceMapping"]["params"]["fileNameLeft"]["value"] = neutral_correspondence_path
    nodes["CorrespondenceMapping"]["params"]["fileNameRight"]["value"] = target_correspondence_final_path

    # Per-pair project file, same reasoning as run_wrap_script.py's own
    # comment: a shared/predictable name would race under concurrent
    # invocations (e.g. --parallel > 1), since WrapCmd reads it back moments
    # after it's written.
    wrap_project_path = os.path.join(os.path.dirname(output_path), "output.wrap")
    with open(wrap_project_path, "w") as f:
        json.dump(data, f, indent=4)

    activate_cmd = [str(wrap_cmd), "activateNodelocked", str(wrap_license)]
    activate_result = subprocess.run(activate_cmd, check=False, env=process_env, text=True, capture_output=True)
    activate_output = "\n".join(part for part in [activate_result.stdout, activate_result.stderr] if part).strip()
    if activate_output:
        print(activate_output)
    if activate_result.returncode != 0:
        normalized = activate_output.lower()
        already_active = "already" in normalized and "activat" in normalized
        if already_active:
            print("Wrap license appears already activated; continuing.")
        else:
            print(f"Warning: Wrap license activation failed (exit code {activate_result.returncode}). Trying compute anyway.")

    compute_cmd = [str(wrap_cmd), "compute", wrap_project_path]
    compute_result = subprocess.run(compute_cmd, check=False, env=process_env, text=True, capture_output=True)
    compute_output = "\n".join(part for part in [compute_result.stdout, compute_result.stderr] if part).strip()
    if compute_output:
        print(compute_output)
    if compute_result.returncode != 0:
        raise subprocess.CalledProcessError(
            compute_result.returncode, compute_cmd,
            output=compute_result.stdout, stderr=compute_result.stderr,
        )
