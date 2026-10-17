"""Application settings and path helpers."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field

# Delegation videos (and therefore landmark/frame indices shown in the frontend) are
# subsampled from the raw video: raw frame r == displayed frame r * RAW_FRAME_STRIDE.
# Generated with `select=not(mod(n,5))` in scripts/preprocess/process_sequence_flame.py.
RAW_FRAME_STRIDE = 5

# Machine-specific paths live in FaceView/backend/.env (see .env.example), not in this file.
ENV_PATH = Path(__file__).resolve().parents[2] / ".env"


class Settings(BaseSettings):
    """Central configuration loaded from environment variables when available."""

    project_name: str = Field(default="FaceView HITL Backend", description="Display name for docs")
    api_v1_prefix: str = Field(default="/api", description="Root prefix for REST endpoints")

    data_root: Path = Field(description="Root folder storing raw multiview data")
    processed_root: Path = Field(description="Folder for delegation videos")
    landmarks_root: Path = Field(description="Folder for extracted/edited landmarks")
    line_renders_root: Path = Field(
        
        description=(
            "Folder for test_render_gt.py's overlay videos "
            "(<actor>/<sequence>/line_render_<camera>.mp4), served by the "
            "/labels/overlay-video endpoint for the frontend's Overlay Video tab."
        ),
    )
    point_overlay_renders_root: Path = Field(
        
        description=(
            "Folder for wrap_script/render_point_overlay.py's landmark point-overlay "
            "videos (<actor>_<sequence>_<camera>.mp4: annotated vs. wrapped-mesh "
            "landmarks), served by the /labels/point-overlay-video endpoint for the "
            "frontend's Overlay Landmarks tab."
        ),
    )
    configs_root: Path = Field(default=Path("configs"), description="Configuration files such as face_indices.json")
    output_root: Path = Field(
        
        description=(
            "Root folder for wrap_script's per-frame pipeline outputs "
            "(<actor>/<sequence>/<frame>/{mesh,wrapped_mesh,expression}.obj), shared "
            "with run_pipeline.py/run_pipeline_all.py."
        ),
    )
    final_wrapped_data_root: Path = Field(
        
        description=(
            "Compact per-sequence export of a confirmed pair's wrapped_mesh.obj frames "
            "(<actor>/<sequence>.pt -- faces saved once, vertices stacked per frame), "
            "written by services/wrapped_mesh_export.py when /labels/confirm succeeds."
        ),
    )
    label_tracker_path: Path = Field(
        
        description=(
            "JSON file tracking each actor/sequence's review status (unlabeled/"
            "unreviewed/reviewed — 'no_neutral_label' is the implicit default for "
            "any pair absent from this file), shared with wrap_script tooling so "
            "batch scripts can target only specifically labeled actors/sequences."
        ),
    )
    default_cameras_path: Path = Field(
        default=Path(__file__).resolve().parents[3] / "default_cameras.json",
        description="JSON file listing default camera IDs, shared with wrap_script/run_pipeline.py",
    )

    # FaceScape (static multiview scans, no video) -- a second dataset onboarded
    # alongside Nersemble. Kept as a fully separate tree from the Nersemble
    # settings above (data_root/processed_root/landmarks_root) rather than
    # reusing them, since FaceScape actors are small ints (1, 2, ...) that could
    # otherwise collide with Nersemble actor ids, and the two datasets' directory
    # shapes differ (per-camera static images + params.json vs per-camera video).
    facescape_images_root: Path = Field(
        
        description="Root folder for FaceScape's origin-space-transformed per-camera images + params.json (see wrap_script_facescape/coordinate_transforms).",
    )
    facescape_raw_images_root: Path = Field(
        
        description=(
            "Root folder for FaceScape's PRISTINE, untransformed per-camera images + "
            "params.json -- never written to (matches wrap_script_facescape/coordinate_transforms/"
            "run_batch_transform.py's SOURCE_IMAGES_ROOT). Each pair's own raw params.json here, "
            "compared against its origin-space counterpart in facescape_images_root, is how "
            "services/facescape_coordinate_transform.py recovers that pair's similarity "
            "transform (Phase 1 never persists it to its own file) to invert a confirmed "
            "pair's wrapped mesh back out of origin space."
        ),
    )
    facescape_raw_shape_root: Path = Field(
        
        description=(
            "Root folder for FaceScape's untransformed scanned meshes (<actor>/<expression>.ply) -- "
            "used by the invalid-pair pipeline (wrap_script_facescape/*_invalid.py), whose pairs have "
            "no origin-space version."
        ),
    )
    facescape_shape_root: Path = Field(
        
        description="Root folder for FaceScape's origin-space-transformed scanned meshes (<actor>/<expression>.ply).",
    )
    facescape_landmarks_root: Path = Field(
        
        description=(
            "Folder for FaceScape's extracted landmarks, in the same "
            "<actor>/<expression>/cam_<camera>/frame_000000.json shape as Nersemble's "
            "landmarks_root (always frame_000000 -- FaceScape has exactly one static "
            "image per camera per actor/expression, no time dimension)."
        ),
    )
    facescape_invalid_pairs_path: Path = Field(
        
        description="Actor/expression pairs that failed Phase 1's origin-space alignment -- see wrap_script_facescape/coordinate_transforms/run_batch_transform.py.",
    )
    facescape_cameras_path: Path = Field(
        
        description="Culled camera-id groups (front/left_front/right_front) usable across every processed FaceScape actor/expression -- see wrap_script_facescape/labeling/compute_facescape_cameras.py.",
    )
    facescape_output_root: Path = Field(
        
        description=(
            "Root folder for FaceScape's wrap pipeline output "
            "(<actor>/<expression>/wrapped_mesh.obj), written by "
            "wrap_script_facescape/run_pipeline.py."
        ),
    )
    point_overlay_renders_facescape_root: Path = Field(
        
        description=(
            "Folder for wrap_script_facescape/render_point_overlay_frame.py's landmark "
            "point-overlay renders (<actor>_<expression>_<camera>.jpg), served by the "
            "/facescape/point-overlay-frame endpoint for the frontend's Overlay "
            "Landmarks tab."
        ),
    )
    line_renders_facescape_root: Path = Field(
        
        description=(
            "Folder for wrap_script_facescape/render_overlay_frame.py's keyline "
            "overlay renders (<actor>_<expression>_<camera>.jpg), served by the "
            "/facescape/overlay-frame endpoint for the frontend's Overlay Frame tab."
        ),
    )

    ava256_wraps_root: Path = Field(
        
        description=(
            "Ava-256 wraps: <capture>/<frame>/wrapped_mesh.obj (template frame) plus "
            "<capture>/dynamic_transform_params.npz, which maps them back to world space."
        ),
    )
    line_renders_ava256_root: Path = Field(
        
        description=(
            "wrap_script_ava256/render_capture_line_video.py output: <capture>.mp4 and its "
            "per-frame <capture>.csv (FRAME_ID,STATUS) labeled in the Ava-256 tab."
        ),
    )
    ava256_labels_root: Path = Field(
        
        description="Where the Ava-256 tab's Done button archives a capture's finished <capture>.csv.",
    )

    wrap_script_root: Path = Field(
        description="Nersemble wrap_script/ (services/facescape_ear_points.py imports its modules from here)."
    )
    facescape_coordinate_transforms_root: Path = Field(
        description="wrap_script_facescape/coordinate_transforms/ (services/facescape_coordinate_transform.py)."
    )

    propagate_window: int = Field(default=48, description="Default number of frames to re-track after an edit")
    job_store_path: Path = Field(default=Path("metadata/jobs"), description="Location for job artifacts")

    landmark_frame_stride: int = Field(
        default=1,
        description=(
            "Expected spacing between landmark frames, exposed to the frontend as a UI "
            "hint (e.g. for next/prev-frame navigation step size). Not enforced — the "
            "authoritative source of which frames actually have data is always whichever "
            "frame_*.json files exist on disk (see LandmarkStore.list_available_frame_indices), "
            "since real sparsity doesn't always land exactly on stride multiples."
        ),
    )

    tracker_python_bin: Path = Field(
        
        description="Interpreter used for MediaPipe and CoTracker subprocesses",
    )

    class Config:
        # FaceView/backend/.env, wherever the backend (or a script importing it) is started from.
        env_file = ENV_PATH
        env_file_encoding = "utf-8"
        extra = "ignore"  # backend/.env also holds the Sapiens step's keys


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached settings instance so every module shares the same object."""

    settings = Settings()
    settings.data_root.mkdir(parents=True, exist_ok=True)
    settings.processed_root.mkdir(parents=True, exist_ok=True)
    settings.landmarks_root.mkdir(parents=True, exist_ok=True)
    settings.line_renders_root.mkdir(parents=True, exist_ok=True)
    settings.line_renders_facescape_root.mkdir(parents=True, exist_ok=True)
    settings.point_overlay_renders_root.mkdir(parents=True, exist_ok=True)
    settings.point_overlay_renders_facescape_root.mkdir(parents=True, exist_ok=True)
    settings.job_store_path.mkdir(parents=True, exist_ok=True)
    settings.final_wrapped_data_root.mkdir(parents=True, exist_ok=True)
    return settings
