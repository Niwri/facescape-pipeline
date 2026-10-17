# Notes for Claude Code sessions in this repo (FaceScape pipeline)

Read `HANDOFF.md` first. It covers the layout, the four `.env` files, the external dependencies, and the
Phase 0–5 run book with exact commands. `wrap_script_facescape/README.md` explains each phase in depth (its paths
are Medusa examples).

## Paths and configuration
- Never hard-code a cluster path.
  - FaceScape scripts: add a key to `wrap_script_facescape/.env` + `.env.example` and read it through
    `facescape_paths.py` (`import facescape_paths as fp`).
  - FaceView: use `FaceView/backend/.env` via `backend.app.core.config.get_settings()`.
  - Shell wrappers: `source` the `.env` next to them.
- After any `.env` or path change, `$VENV_PYTHON wrap_script_facescape/facescape_paths.py` must print OK.
- `wrap_script/` and `FaceView/` are shared with the Nersemble pipeline (`nersemble-stuff`); make changes to them
  in both repos.

## Running
- **Working directory:** run FaceScape scripts from `wrap_script_facescape/`, or from `coordinate_transforms/` /
  `labeling/` for those.
- **Interpreters:** use `VENV_PYTHON` for phases 1, 3 and 4 and the backend, and `SAPIENS_PYTHON` for phase 2.
  The batch scripts start their per-pair subprocesses with the interpreter that launched them.
- **Pair status flow** (`label_tracker.json`, "FaceScape" key):
  `unlabeled` (P2) → `unreviewed` (P3) → `unconfirmed` (P4) → `confirmed` (P5, with export).
  - `run_pipeline_batch.py` only wraps `unreviewed` pairs.
  - Actors need a saved neutral (`neutral_frame.json`, "Save as Neutral" in FaceView) before Phase 4.
- **Invalid pairs** (Phase 1 failures, `INVALID_PAIRS_PATH`) are skipped by phases 2 and 4. They go through the
  `*_invalid.py` scripts, which work in each pair's raw space.
- **Phase 0 rsync must use `--ignore-existing`.** Never write into the raw FaceScape download (`FACESCAPE_RAW_*`).
- **Template wraps are face-scaled** (`face_scale.py`, "P5": vertical brow-to-chin, horizontal outer eye corners),
  then mapped back to world space; the map is saved in `face_scale.json`. Keep `scan.obj` and
  `keypoint_vertices.json` in world space.

## Testing and safety
- **Sandbox a test:** override `FACESCAPE_OUTPUT_ROOT`, `LABEL_TRACKER_PATH`, `LINE_RENDERS_FACESCAPE_ROOT` and
  `POINT_OVERLAY_RENDERS_FACESCAPE_ROOT` in the shell so nothing real is written. Shell values win over `.env`.
- **Ask first** before:
  - starting batch jobs (the Phase 2 and 4 batch scripts poll forever);
  - deleting outputs;
  - editing `label_tracker.json` or `invalid_pairs.json`;
  - committing or pushing.
- **Back up a file** before overwriting it.
- **Stop processes by PID**, never by name pattern.
