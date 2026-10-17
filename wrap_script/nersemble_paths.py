"""Filesystem locations for the Nersemble wrap_script pipeline, read from
wrap_script/.env (next to this file, so it works from any working directory
or cluster node) instead of being hard-coded in each script. A variable already
set in the process environment wins over the .env value. Every key is
required: a missing one raises with its name, rather than silently falling
back to a path on one particular cluster.

Usage:  import nersemble_paths
        nersemble_paths.OUTPUT_ROOT / actor / sequence      # pathlib.Path
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ENV_PATH = Path(__file__).resolve().with_name(".env")
load_dotenv(ENV_PATH, override=False)


def _path(key: str) -> Path:
    value = os.environ.get(key)
    if not value:
        raise KeyError(f"{key} is not set -- add it to {ENV_PATH} (see .env.example)")
    return Path(value)


# Raw multiview data (calibration + per-camera videos), per actor.
DATA_ROOT = _path("DATA_ROOT")
# Scratch area the pipeline's outputs live under (one-off / legacy outputs).
WORK_ROOT = _path("WORK_ROOT")
# Delegation videos and per-frame landmark JSONs.
PROCESSED_ROOT = _path("PROCESSED_ROOT")
LANDMARKS_ROOT = _path("LANDMARKS_ROOT")
# Per-frame wraps (output/<actor>/<sequence>/<frame>/wrapped_mesh.obj) and the
# confirmed pairs' compact exports (final_wrapped_data/<actor>/<sequence>.pt).
OUTPUT_ROOT = _path("OUTPUT_ROOT")
FINAL_WRAPPED_DATA_ROOT = _path("FINAL_WRAPPED_DATA_ROOT")
# Review videos FaceView serves.
LINE_RENDERS_ROOT = _path("LINE_RENDERS_ROOT")
POINT_OVERLAY_RENDERS_ROOT = _path("POINT_OVERLAY_RENDERS_ROOT")
FLOWFACE_LINE_RENDER_ROOT = _path("FLOWFACE_LINE_RENDER_ROOT")
LABEL_TRACKER_PATH = _path("LABEL_TRACKER_PATH")
LANDMARK_LINES_PATH = _path("LANDMARK_LINES_PATH")
# Meshroom scans (meshroom-tracking-full/<actor>/<sequence>/frame_<N>/mesh.obj.gz).
MESHROOM_DATA_ROOT = _path("MESHROOM_DATA_ROOT")
# Nersemble tracking results (shared copy, and the local copy).
NERSEMBLE_TRACKING_ROOT = _path("NERSEMBLE_TRACKING_ROOT")
NERSEMBLE_TRACKING_LOCAL_ROOT = _path("NERSEMBLE_TRACKING_LOCAL_ROOT")
# Code / models / tools.
FACEVIEW_ROOT = _path("FACEVIEW_ROOT")
WRAP_SCRIPT_ROOT = _path("WRAP_SCRIPT_ROOT")
MODELS_ROOT = _path("MODELS_ROOT")
SAPIENS_PYTHON = _path("SAPIENS_PYTHON")
GPU_LOCKS_ROOT = _path("GPU_LOCKS_ROOT")
