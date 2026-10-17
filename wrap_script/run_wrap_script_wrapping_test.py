import argparse
import json
import os
import subprocess
import tempfile
from dotenv import load_dotenv
from pathlib import Path
import numpy as np

def main():

    parser = argparse.ArgumentParser(description="Modify template.wrap and run WrapCmd commands.")
    parser.add_argument('--face_mask_input', required=True, type=str)
    parser.add_argument('--neutral_mesh', required=True, type=str)
    # parser.add_argument('--neutral_texture', required=True, type=str)
    parser.add_argument('--target_mesh', required=True, type=str)
    parser.add_argument('--target_correspondence', required=True, type=str)
    # parser.add_argument('--target_texture', required=True, type=str)
    parser.add_argument('--output_path', required=True, type=str)
    args = parser.parse_args()
    wrap_mesh(args.face_mask_input, args.neutral_mesh, args.target_mesh, args.target_correspondence, args.output_path)

def wrap_mesh(face_mask_path, source_mesh_path, target_mesh_path, target_correspondence_path, output_path, source_mesh_2_path, target_correspondence_2_path, ignore_ears: bool=False, pot_template_path: str | None = None, whitelist_path: str | None = None, extra_indices_to_remove=None):

    # Load .env
    load_dotenv(Path(__file__).resolve().with_name(".env"))  # wrap_script/.env, wherever this is run from
    wrap_cmd_raw = os.getenv('WrapCmd')
    wrap_license_raw = os.getenv('WrapLicense')
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

    # Load the "Wrapping"-node graph (template_wrapping_test.wrap was renamed
    # to template.wrap; the old no-"Wrapping"-node graph is now
    # template_blend.wrap, loaded by run_wrap_script.py instead).
    with open('wrap_template/template.wrap', 'r') as f:
        data = json.load(f)

    # Set node values
    nodes = data['nodes']
    
    temp_face_mask_path = face_mask_path
    if(ignore_ears):
        temp_face_mask_path = face_mask_path + "_without_ears"

    nodes['FaceMask']['params']['fileName']['value'] = temp_face_mask_path + ".txt"
    # nodes['FaceMask1']['params']['fileName']['value'] = face_mask_path

    face_inverse_list = []
    with open(face_mask_path + "_inverse.txt", 'r') as f:
        face_inverse_list = json.load(f)
    
    nodes['FaceMask']['params']['invisiblePolygonIndices'] = {"value": face_inverse_list}
    # nodes['FaceMask1']['params']['invisiblePolygonIndices'] = {"value": face_inverse_list}

    # FaceMaskWrapping feeds the Wrapping node's own polygon selection --
    # written alongside face_mask_path's own .txt/_inverse.txt pair by the
    # caller (face_ray_masking_wrapping_test.py's third/fourth return values),
    # same geom source (NeutralInput) as FaceMask, just a stricter selection
    # (intersected with wrapping_node_mask.txt).
    nodes['FaceMaskWrapping']['params']['fileName']['value'] = face_mask_path + "_wrapping_node.txt"
    with open(face_mask_path + "_wrapping_node_inverse.txt", 'r') as f:
        wrapping_node_inverse_list = json.load(f)
    nodes['FaceMaskWrapping']['params']['invisiblePolygonIndices'] = {"value": wrapping_node_inverse_list}

    nodes['NeutralInput']['params']['fileName']['value'] = source_mesh_path
    # nodes['NeutralInputTexture']['params']['fileName']['value'] = args.neutral_texture
    nodes['TargetInput']['params']['fileName']['value'] = target_mesh_path
        
    whitelist_file = whitelist_path or "whitelist.json"
    with open(whitelist_file, "r") as f:
        whitelist_indices = set(int(idx) for idx in json.load(f))

    # whitelist.json lists the landmark indices that get hand-labeled/corrected
    # (see LandmarkOverlay.tsx / camera_render_landmarks.py) — these are the
    # correspondence points Wrap should fit to. Everything else is removed
    # (plus the ear indices too, when ignore_ears is set). Computed per-file
    # since neutral/target correspondence files aren't necessarily the same length.
    base_indices_to_remove = set([1, 2, 4, 6, 9, 14]) if ignore_ears else set()
    if extra_indices_to_remove:
        # Points triangulate.py couldn't triangulate for this specific frame --
        # no camera had an in-bounds observation at all (e.g. a FLAME index with
        # no sapiens2 equivalent, always (-1, -1, -1); see
        # process_sequence_sapiens.py's SAPIENS_LANDMARK_MAP and
        # triangulate.py's triangulate_point()). Dropped from BOTH correspondence
        # files via the same indices_to_remove_for() used for the whitelist, so
        # Wrap still gets a consistent point-to-point mapping between neutral and
        # target, just without this index, instead of fitting to a meaningless
        # placeholder coordinate.
        base_indices_to_remove |= set(extra_indices_to_remove)

    def indices_to_remove_for(total_len):
        return sorted((set(range(total_len)) - whitelist_indices) | base_indices_to_remove)

    # Every file this function writes as scratch/intermediate state is tracked
    # here and removed in the `finally` block below, once WrapCmd is done reading
    # them -- both to avoid leaving cruft behind (this JSON dump alone is ~550KB,
    # times however many frames get wrapped) and, for output.wrap specifically,
    # because a shared/predictable name is NOT safe for concurrent invocations
    # (see below).
    temp_paths: list[str] = []

    pot_template_file = pot_template_path or "neutral_correspondence_without_eyes_POT.json"
    neutral_correspondence_path = (
        "$PROJECT_DIR/neutral_correspondence_without_eyes_POT.json"
        if pot_template_path is None
        else pot_template_file
    )
    with open(pot_template_file, "r") as f:
        neutral_correspondence_data = np.array(json.load(f))

    neutral_indices_to_remove = indices_to_remove_for(len(neutral_correspondence_data))
    if(neutral_indices_to_remove):
        filtered_neutral_correspondence = np.delete(
            neutral_correspondence_data,
            neutral_indices_to_remove,
            axis=0,
        )

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix="_neutral_correspondence_without_eyes_filtered.json",
            delete=False,
        ) as f:
            json.dump(filtered_neutral_correspondence.tolist(), f)
            neutral_correspondence_path = f.name
            temp_paths.append(neutral_correspondence_path)

    nodes['CorrespondancePointsNeutral']['params']['fileName']['value'] = neutral_correspondence_path
    nodes['CorrespondenceMapping']['params']['fileNameLeft']['value'] = neutral_correspondence_path

    target_correspondence_right_path = target_correspondence_path
    with open(target_correspondence_path, "r") as f:
        target_correspondence_data = np.array(json.load(f))

    target_indices_to_remove = indices_to_remove_for(len(target_correspondence_data))
    if(target_indices_to_remove):
        filtered_target_correspondence = np.delete(
            target_correspondence_data,
            target_indices_to_remove,
            axis=0,
        )

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix="_target_correspondence_filtered.json",
            delete=False,
        ) as f:
            json.dump(filtered_target_correspondence.tolist(), f)
            target_correspondence_right_path = f.name
            temp_paths.append(target_correspondence_right_path)

    nodes['CorrespondenceMapping']['params']['fileNameRight']['value'] = target_correspondence_right_path

    # nodes['CorrespondenceMapping1']['params']['fileNameRight']['value'] = target_correspondence_path
    # nodes['TargetTexture']['params']['fileName']['value'] = args.target_texture
    nodes['SaveWrap']['params']['fileName']['value'] = output_path


    nodes['NeutralInput1']['params']['fileName']['value'] = source_mesh_2_path
    nodes['FaceMask2']['params']['fileName']['value'] = face_mask_path + "_inverse.txt"
    nodes['CorrespondenceMapping2']['params']['fileNameLeft']['value'] = target_correspondence_2_path
    nodes['CorrespondenceMapping2']['params']['fileNameRight']['value'] = target_correspondence_2_path

    # The only $PROJECT_DIR-relative references template.wrap's own node defaults
    # still carry that run_wrap_script.py doesn't already override with an
    # explicit path: the FLAME blendshape basis meshes, which live under this
    # script's own directory (not per-frame). Pin them to absolute paths so
    # they keep resolving correctly regardless of where the project file
    # itself gets saved (see below).
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for blend_node_name in ("BlendWrapping", "BlendWrapping1"):
        blend_node = nodes[blend_node_name]['params']
        blend_node['neutralFileName']['value'] = os.path.join(script_dir, "blendshape", "neutral.obj")
        blend_node['blendshapeFileNames']['value'] = [
            os.path.join(script_dir, "blendshape", os.path.basename(name))
            for name in blend_node['blendshapeFileNames']['value']
        ]

    # Save the project file into the same per-frame output directory as
    # wrapped_mesh.obj itself (mirroring MeshroomCache's own per-frame scoping in
    # run_pipeline.py/run_pipeline_all.py) rather than a single "output.wrap" in
    # this script's own directory. A fixed shared name there is written AND read
    # by WrapCmd moments apart, so two concurrent wrap_mesh() calls (e.g.
    # run_pipeline_all.py under --parallel > 1) can and did race on it: one
    # process's write lands mid-read by another's WrapCmd, producing "empty JSON
    # document", "garbage at the end of the document", or a reference to a temp
    # file that belongs to a different process's run entirely. A per-frame output
    # directory is already guaranteed unique, so this can't collide, and -- like
    # MeshroomCache -- it's left in place afterward rather than cleaned up.
    wrap_project_path = os.path.join(os.path.dirname(output_path), "output.wrap")
    with open(wrap_project_path, 'w') as f:
        json.dump(data, f, indent=4)

    try:
        # Run commands with explicit env so headless sessions do not require xcb.
        activate_cmd = [str(wrap_cmd), "activateNodelocked", str(wrap_license)]
        activate_result = subprocess.run(
            activate_cmd,
            check=False,
            env=process_env,
            text=True,
            capture_output=True,
        )
        activate_output = "\n".join(
            part for part in [activate_result.stdout, activate_result.stderr] if part
        ).strip()
        if activate_output:
            print(activate_output)

        if activate_result.returncode != 0:
            normalized = activate_output.lower()
            already_active = "already" in normalized and "activat" in normalized
            if already_active:
                print("Wrap license appears already activated; continuing.")
            else:
                print(
                    "Warning: Wrap license activation failed "
                    f"(exit code {activate_result.returncode}). Trying compute anyway."
                )

        compute_cmd = [str(wrap_cmd), "compute", wrap_project_path]
        # Caps WrapCmd's own internal thread pool -- without this, each
        # concurrent wrap_mesh() call (e.g. one per --parallel worker in
        # run_pipeline_batch_all.py) independently sizes its thread pool off
        # every core the cgroup allocation exposes, so N concurrent workers
        # can end up running N x that-many threads on a machine with only
        # that-many cores total -- oversubscription that gets WORSE the more
        # cores a node reports (see the Xeon-vs-EPYC slowdown investigation).
        # Set WRAP_MAX_THREADS to (cpus-per-task / --parallel) so each
        # worker's WrapCmd gets an even, non-overlapping share instead.
        wrap_max_threads = os.getenv("WRAP_MAX_THREADS")
        if wrap_max_threads:
            compute_cmd += ["--maxThreads", str(int(wrap_max_threads))]
        compute_result = subprocess.run(
            compute_cmd,
            check=False,
            env=process_env,
            text=True,
            capture_output=True,
        )
        compute_output = "\n".join(
            part for part in [compute_result.stdout, compute_result.stderr] if part
        ).strip()
        if compute_output:
            print(compute_output)

        if compute_result.returncode != 0:
            raise subprocess.CalledProcessError(
                compute_result.returncode,
                compute_cmd,
                output=compute_result.stdout,
                stderr=compute_result.stderr,
            )
    finally:
        for path in temp_paths:
            try:
                os.remove(path)
            except OSError:
                pass

if __name__ == "__main__":
    main()
