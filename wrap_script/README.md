# Wrapping
Here are a series of instructions to get from correspondence points all the way to wrapping. All of these steps may be combined into one script, though it is currently divided into series of steps.

## Correspondence Steps
1) Store landmarks in `(settings.landmarks)/(actor_id)/(sequence_id)/cam_(cam_id)/frame_(frame_num).json`
    - Ensure settings.landmarks is set properly, found in `backend/app/core/config.py`
    
2) Run `triangulate.py (actor_id) (sequence_id) (frame_num/5) --cameras ...`. This will output `keypoint_vertices.json`, which has the 3D correspondence points of the target actor.
Estimated time: 1.56s

3) Run `render_flame_exp_without_eyes.py (actor_id) (sequence_id) (frame_num)`. This will output `expression.obj`, which is a FLAME mesh.
Estimated time: 1.69s

## Actor Mesh Steps
4) Run `extract_frame.py (actor_id) (sequence_id) (frame_num)` to generate frames for photogrammetry in Meshroom
Estimated time:  44.03s

5) Run `generate_sfm.py (actor_id) (sequence_id) (frame_num)` to generate the SFM file used in Meshroom
Estimated time: 0.11s

6) Run `run_meshroom_batch.py (actor_id) (sequence_id) (frame_num)` to output the target mesh. The mesh will be outputted to `meshroom_output/` folder
Estimated time: 188.51s

## Neutral Texture Wrapping
8) TODO: Use Blender

## Wrapping
9) Run `face_ray_masking.py (actor_id) (sequence_id) (frame_num) expression.obj` to get the mask for wrapping. It will output to `wrap_mask.txt`
Estimated time: 49.2s

9) Run `run_wrap_script.py --face_mask_input wrap_mask.txt --neutral_mesh neutral.obj --target_mesh (Meshroom Target Mesh) --target_correspondence keypoint_vertices.json --output_path (output path)` The output path will have the wrapped mesh
Estimated time: 5.22s

## Current Pipeline
- Running `run_pipeline.py` for each data individually
- Running `run_pipeline_batch.py` to run the nersemble data sequentially
- Running `run_meshroom_compare.py` to render a video for visually checking the wrapped mesh results
- Running `python3 analyze_temporal_noise.py <actor_id> <sequence_id> [--top-n N] [--landmarks-only] [--mesh-only]` checks for noise of the wrapped script
