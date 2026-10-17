"""Filesystem locations for the FaceScape pipeline, read from wrap_script_facescape/.env (next to this file, so it
works from any working directory or cluster node) instead of being hard-coded in each script. Same convention as
wrap_script/nersemble_paths.py: a variable already set in the process environment wins over the .env value, and
every key is required -- a missing one raises with its name rather than falling back to a path on one cluster.
Loading this module also puts the keys into os.environ, so subprocesses (e.g. process_expression_sapiens.py,
WrapCmd) inherit them.

The FaceScape data roots FaceView's backend uses (FACESCAPE_IMAGES_ROOT, FACESCAPE_OUTPUT_ROOT, LABEL_TRACKER_PATH,
...) live in FaceView/backend/.env and are read through backend.app.core.config.get_settings(); keys repeated here
must hold the same values (check_paths() compares them).

Usage:  import facescape_paths as fp
        fp.FACESCAPE_OUTPUT_ROOT / actor / expression      # pathlib.Path
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ENV_PATH = Path(__file__).resolve().with_name(".env")
load_dotenv(ENV_PATH, override=False)


def _path(key: str) -> Path:
    value = os.environ.get(key)
    if not value:
        raise KeyError(f"{key} is not set -- add it to {ENV_PATH} (see .env.example)")
    return Path(value)


# This folder (template.wrap, neutral.obj, POT tables, masks, cameras live here).
FACESCAPE_PIPELINE_DIR = Path(__file__).resolve().parent
# Code that lives next to this folder (shipped in the same repo).
FACEVIEW_ROOT = _path("FACEVIEW_ROOT")              # FaceView/ (backend config + services, co-tracker, Sapiens scripts)
WRAP_SCRIPT_ROOT = _path("WRAP_SCRIPT_ROOT")        # wrap_script/ (Nersemble modules reused: triangulate, skin landmarks, blendshapes)
AVA256_SHARED_ROOT = _path("AVA256_SHARED_ROOT")    # ava256_shared/ (camera selection helpers used by invalid_pairs.py)
# Raw FaceScape download (read only) and the working copy every phase writes into (Phase 0/1 of README.md).
FACESCAPE_RAW_IMAGES_ROOT = _path("FACESCAPE_RAW_IMAGES_ROOT")
FACESCAPE_RAW_SHAPE_ROOT = _path("FACESCAPE_RAW_SHAPE_ROOT")
FACESCAPE_IMAGES_ROOT = _path("FACESCAPE_IMAGES_ROOT")
FACESCAPE_SHAPE_ROOT = _path("FACESCAPE_SHAPE_ROOT")
# Outputs.
FACESCAPE_OUTPUT_ROOT = _path("FACESCAPE_OUTPUT_ROOT")
POINT_OVERLAY_RENDERS_FACESCAPE_ROOT = _path("POINT_OVERLAY_RENDERS_FACESCAPE_ROOT")
FINAL_WRAPPED_BACKUP_ROOT = _path("FINAL_WRAPPED_BACKUP_ROOT")  # run_pipeline_batch.py --rewrap backups
INVALID_PAIRS_PATH = _path("INVALID_PAIRS_PATH")
SANITY_RENDER_ROOT = _path("SANITY_RENDER_ROOT")    # process_expression_sapiens.py debug renders
# Shared data / models / tools.
LANDMARK_LINES_PATH = _path("LANDMARK_LINES_PATH")
RAFT_CHECKPOINT_PATH = _path("RAFT_CHECKPOINT_PATH")
SAPIENS_CHECKPOINT_ROOT = _path("SAPIENS_CHECKPOINT_ROOT")
SAPIENS_PYTHON = _path("SAPIENS_PYTHON")            # faceview_sapiens_venv interpreter (Phase 2)
VENV_PYTHON = _path("VENV_PYTHON")                  # faceview_venv interpreter (phases 1, 3, 4; shell wrappers)
WRAP_CMD = _path("WrapCmd")                         # Faceform WrapCmd + license (run_wrap_script.py)
WRAP_LICENSE = _path("WrapLicense")


def add_faceview_to_sys_path() -> None:
    """FaceView root first on sys.path, for `backend.app.core.config` / `backend.app.services` (wrap_script/ has an
    older backend/ mirror without the FaceScape settings, which must not shadow it)."""
    root = str(FACEVIEW_ROOT)
    if root in sys.path:
        sys.path.remove(root)
    sys.path.insert(0, root)


def check_paths() -> list[str]:
    """Problems with the configured paths: missing code/model/tool paths, and keys repeated in FaceView's
    backend/.env with a different value. Run `python facescape_paths.py` after editing .env."""
    problems = []
    for key in ("FACEVIEW_ROOT", "WRAP_SCRIPT_ROOT", "AVA256_SHARED_ROOT", "LANDMARK_LINES_PATH", "RAFT_CHECKPOINT_PATH",
                "SAPIENS_CHECKPOINT_ROOT", "SAPIENS_PYTHON", "VENV_PYTHON", "WrapCmd", "WrapLicense", "FACESCAPE_IMAGES_ROOT", "FACESCAPE_SHAPE_ROOT"):
        if not Path(os.environ[key]).exists():
            problems.append(f"{key} does not exist: {os.environ[key]}")
    add_faceview_to_sys_path()
    try:
        from backend.app.core.config import get_settings
        st = get_settings()
        for key in ("FACESCAPE_IMAGES_ROOT", "FACESCAPE_SHAPE_ROOT", "FACESCAPE_RAW_IMAGES_ROOT", "FACESCAPE_OUTPUT_ROOT",
                    "POINT_OVERLAY_RENDERS_FACESCAPE_ROOT"):
            backend_value = getattr(st, key.lower(), None)
            if backend_value is not None and Path(backend_value) != Path(os.environ[key]):
                problems.append(f"{key}: {os.environ[key]} here but {backend_value} in FaceView/backend/.env")
        # backend/.env keys that must point inside this repo (export, camera list, shared modules)
        for key, expected in (("facescape_cameras_path", FACESCAPE_PIPELINE_DIR / "facescape_cameras.json"),
                              ("facescape_coordinate_transforms_root", FACESCAPE_PIPELINE_DIR / "coordinate_transforms"),
                              ("wrap_script_root", WRAP_SCRIPT_ROOT)):
            value = getattr(st, key, None)
            if value is not None and Path(value).resolve() != Path(expected).resolve():
                problems.append(f"FaceView/backend/.env {key.upper()} = {value}, expected {expected}")
    except Exception as exc:  # noqa: BLE001
        problems.append(f"could not load FaceView backend settings: {exc!r}")
    return problems


if __name__ == "__main__":
    issues = check_paths()
    print("\n".join(issues) if issues else f"OK: all paths in {ENV_PATH} resolve and match FaceView/backend/.env")
    sys.exit(1 if issues else 0)
