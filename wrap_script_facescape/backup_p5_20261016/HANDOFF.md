# FaceScape pipeline — handoff

The code needed to label, propagate and wrap FaceScape, made portable the same way as `nersemble_pipeline`
(`nersemble-stuff`). Built on 2026-10-16 from the Medusa working folders: `wrap_script_facescape/` (FaceScape
scripts), plus the Nersemble modules it reuses and FaceView. `MANIFEST.txt` lists every source. Nothing in the
code points at a cluster path any more: every location comes from a `.env` file.

## Layout

```
wrap_script_facescape/   the FaceScape pipeline (phases 1-4, see its README.md)
  facescape_paths.py     loads every path from wrap_script_facescape/.env (raises if a key is missing);
                         `python facescape_paths.py` checks the paths and that they match FaceView/backend/.env
  coordinate_transforms/ phase 1 (origin-space alignment; inputs/ holds the solved origin transforms)
  labeling/              FaceScape camera selection (facescape_cameras.json)
  template.wrap neutral.obj neutral_correspondence_without_eyes_POT*.json facescape_mask.txt facescape_cameras.json
wrap_script/             Nersemble modules reused by FaceScape (triangulate, skin_landmarks, propagated_landmarks,
                         render_landmarks, point_overlay, blendshapes, flame_model, landmark region files);
                         same copy as nersemble_pipeline/wrap_script, reads wrap_script/.env via nersemble_paths.py
ava256_shared/           six Ava-256 helpers invalid_pairs.py uses (camera selection / classification)
FaceView/                labelling backend + frontend (FaceScape tabs), co-tracker + checkpoint,
                         scripts/preprocess/process_expression_sapiens*.py (phase 2, Sapiens2)
envs/                    pip freeze of both venvs + rebuild notes (README.md)
```

## Configuration — four `.env` files (each has a `.env.example` next to it)

| File | Read by | Edit on a new cluster |
|---|---|---|
| `wrap_script_facescape/.env` | `facescape_paths.py` (all FaceScape scripts; children inherit it) | everything |
| `FaceView/backend/.env` | `backend/app/core/config.py` (FaceScape roots, `LABEL_TRACKER_PATH`, landmarks/output roots) | the `FACESCAPE_*`, `LABEL_TRACKER_PATH`, `LINE_RENDERS_FACESCAPE_ROOT`, `SAPIENS_*` keys; Nersemble keys can point anywhere |
| `wrap_script/.env` | `nersemble_paths.py` (imported by `render_landmarks.py` / `point_overlay.py`) | every key must exist; `LANDMARK_LINES_PATH`, `FACEVIEW_ROOT`, `WRAP_SCRIPT_ROOT` matter, Nersemble data roots can point anywhere |
| `FaceView/frontend/.env` | `vite.config.ts` | `BACKEND_URL` |

- The FaceScape roots repeated in `wrap_script_facescape/.env` and `FaceView/backend/.env` must agree;
  `python wrap_script_facescape/facescape_paths.py` reports any mismatch or missing path.
- A variable set in the shell overrides the `.env` value. That is how to sandbox a run, e.g.
  `FACESCAPE_OUTPUT_ROOT=/tmp/t/out LABEL_TRACKER_PATH=/tmp/t/tracker.json LINE_RENDERS_FACESCAPE_ROOT=/tmp/t/line
  POINT_OVERLAY_RENDERS_FACESCAPE_ROOT=/tmp/t/point python run_pipeline.py 301 1_neutral`
  (validated this way on Medusa: outputs went only to the sandbox).
- The `.env` files in this folder hold the Medusa values; `.gitignore` keeps every `.env` out of git.

## External dependencies (NOT in this folder)

- **FaceScape data**: the multiview download (images + `params.json`, shape `.ply`) at `FACESCAPE_RAW_*`; phase 0
  rsyncs it to the working copy at `FACESCAPE_IMAGES_ROOT` / `FACESCAPE_SHAPE_ROOT` (see README "Phase 0").
- **Faceform Wrap**: `WrapCmd` + license (`WrapCmd`, `WrapLicense` in `wrap_script_facescape/.env`); the license is
  tied to the account/machine.
- **Python envs**: `faceview_venv` (phases 1, 3, 4: numpy, trimesh, cv2, torch, CoTracker deps, python-dotenv) and
  `faceview_sapiens_venv` (phase 2, `SAPIENS_PYTHON`). Package lists and rebuild steps: `envs/README.md`.
- **Models**: RAFT `raft_large_C_T_SKHT_V2-ff5fadd5.pth` (`RAFT_CHECKPOINT_PATH`), Sapiens2 checkpoints
  (`SAPIENS_CHECKPOINT_ROOT`). CoTracker's `scaled_offline.pth` is included under `FaceView/co-tracker/checkpoints/`.
- **`landmark_lines.npy`** (`LANDMARK_LINES_PATH`), the keyline definition the line renders draw.
- **State**: `label_tracker.json` (`LABEL_TRACKER_PATH`, "FaceScape" key), the landmarks root
  (`FACESCAPE_LANDMARKS_ROOT`), and `invalid_pairs.json` (`INVALID_PAIRS_PATH`). Copy them over to continue
  existing work, or start empty.

## Running

Phases, commands and status transitions are in `wrap_script_facescape/README.md`. Its examples use the Medusa
paths; replace them with your `.env` values. Short version (venv python, cwd `wrap_script_facescape/`):

1. Phase 1: `coordinate_transforms/run_batch_transform.py --actor-min A --actor-max B --parallel 4`
2. Phase 2: `run_process_expression_sapiens_batch.py --gpu-ids 0 1 --parallel 2` (Sapiens2, `SAPIENS_PYTHON`)
3. Phase 3: `run_neutral_skin_propagation.py` (or `script_run_neutral_skin_propagation.sh`)
4. Phase 4: `run_pipeline_batch.py` (wraps every "unreviewed" pair)
5. Phase 5: confirm in the FaceView frontend (exports the wrapped mesh)

Invalid pairs (no origin-space version) go through the `*_invalid.py` variants; see README.

## Known state on Medusa (2026-10-16)

- The working images root only has `params.json` for actors 301+; earlier actors' folders are there without it,
  so re-wrapping them needs phase 1 re-run (or their raw images via the invalid-pair path).
- Actors 32, 41 and 173 wrap badly from scratch (errors of metres in a test): their scan/calibration looks to be
  in a different frame. Check before re-running them.
- The wrap uses no scale normalisation (FaceScape scans are in metres, near template size). A face-size scale
  test lowered the forehead line to Nersemble's level; see the conversation notes, not adopted in this code.
