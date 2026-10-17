"""Application settings and path helpers."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Central configuration loaded from environment variables when available."""

    project_name: str = Field(default="FaceView HITL Backend", description="Display name for docs")
    api_v1_prefix: str = Field(default="/api", description="Root prefix for REST endpoints")

    data_root: Path = Field(description="Root folder storing raw multiview data")
    processed_root: Path = Field(description="Folder for delegation videos")
    landmarks_root: Path = Field(description="Folder for extracted/edited landmarks")
    configs_root: Path = Field(default=Path("configs"), description="Configuration files such as face_indices.json")
    label_tracker_path: Path = Field(
        description=(
            "JSON file tracking each actor/sequence's review status (unlabeled/"
            "unreviewed/reviewed — 'no_neutral_label' is the implicit default for "
            "any pair absent from this file), shared with the FaceView backend so "
            "batch scripts here can target only specifically labeled actors/sequences."
        ),
    )

    propagate_window: int = Field(default=48, description="Default number of frames to re-track after an edit")
    job_store_path: Path = Field(default=Path("metadata/jobs"), description="Location for job artifacts")

    tracker_python_bin: Path = Field(
        description="Interpreter used for MediaPipe and CoTracker subprocesses",
    )

    class Config:
        # wrap_script/.env (shared with nersemble_paths.py), wherever the script is started from; keys this
        # class doesn't define are ignored rather than rejected.
        env_file = Path(__file__).resolve().parents[3] / ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached settings instance so every module shares the same object."""

    settings = Settings()
    settings.data_root.mkdir(parents=True, exist_ok=True)
    settings.processed_root.mkdir(parents=True, exist_ok=True)
    settings.landmarks_root.mkdir(parents=True, exist_ok=True)
    settings.job_store_path.mkdir(parents=True, exist_ok=True)
    return settings
