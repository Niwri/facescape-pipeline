# FaceScape pipeline — handoff

This folder holds the code needed to align, label, propagate, wrap and export FaceScape. It is portable in the same
way as `nersemble_pipeline` (the `nersemble-stuff` repo).
- **Built:** 2026-10-16, from the Medusa working folders (`wrap_script_facescape/`, the Nersemble modules it reuses,
  and FaceView). `MANIFEST.txt` lists every source.
- **No cluster paths in the code:** every location comes from a `.env` file.
- **More detail:** `wrap_script_facescape/README.md` explains each phase in depth. Its paths are Medusa examples;
  this file gives the portable commands.

## Layout

```
wrap_script_facescape/   the FaceScape pipeline (run every phase from here unless noted)
  facescape_paths.py     reads all paths from wrap_script_facescape/.env; `python facescape_paths.py` checks them
  face_scale.py          face-based target scaling ("P5") for template wraps (see Phase 4)
  coordinate_transforms/ Phase 1 (origin-space alignment; inputs/ = the solved origin transforms)
  labeling/              Phase 2 camera subset (writes facescape_cameras.json)
  template.wrap neutral.obj neutral_correspondence_without_eyes_POT*.json facescape_mask.txt facescape_cameras.json
wrap_script/             Nersemble modules reused (triangulation, skin landmarks, point overlay, blendshapes, FLAME)
ava256_shared/           Ava-256 camera helpers used for invalid pairs
FaceView/                labelling backend + frontend (FaceScape tabs, confirm/export), co-tracker + checkpoint,
                         script_process_expression_sapiens.sh + scripts/preprocess/process_expression_sapiens*.py
envs/                    pip freeze of both venvs + rebuild notes (README.md)
```

## Configuration — four `.env` files (each has a `.env.example` next to it)

| File | Read by | Edit on a new cluster |
|---|---|---|
| `wrap_script_facescape/.env` | `facescape_paths.py`, so every FaceScape script; child processes inherit it | everything |
| `FaceView/backend/.env` | `backend/app/core/config.py`, `script_process_expression_sapiens.sh`, `process_expression_sapiens.py` | `FACESCAPE_*`, `LABEL_TRACKER_PATH`, `LINE_RENDERS_FACESCAPE_ROOT`, `FINAL_WRAPPED_DATA_ROOT`, `SAPIENS_*`, `WRAP_SCRIPT_ROOT`. Nersemble keys can point anywhere. |
| `wrap_script/.env` | `nersemble_paths.py` (used by the point overlays) | every key must exist; `FACEVIEW_ROOT`, `WRAP_SCRIPT_ROOT` and `LANDMARK_LINES_PATH` matter |
| `FaceView/frontend/.env` | `vite.config.ts` | `BACKEND_URL` |

**Check after every `.env` edit:**
```bash
cd wrap_script_facescape && $VENV_PYTHON facescape_paths.py
```
This must print `OK`. It checks that the code, model and tool paths exist, and that:
- the roots repeated in both `.env` files agree;
- `FaceView/backend/.env`'s `FACESCAPE_CAMERAS_PATH`, `FACESCAPE_COORDINATE_TRANSFORMS_ROOT` and
  `WRAP_SCRIPT_ROOT` point inside this repo.

**Other notes:**
- A variable set in the shell overrides the `.env` value. That's how to sandbox a run, for example:
  ```bash
  FACESCAPE_OUTPUT_ROOT=/tmp/t/out LABEL_TRACKER_PATH=/tmp/t/tracker.json LINE_RENDERS_FACESCAPE_ROOT=/tmp/t/line \
  POINT_OVERLAY_RENDERS_FACESCAPE_ROOT=/tmp/t/point $VENV_PYTHON run_pipeline.py 301 1_neutral
  ```
- The `.env` files here hold the Medusa values. `.gitignore` keeps every `.env` out of git.

## External dependencies (not in this folder)

- **FaceScape data:** the multiview download (`images/fsmview_trainset/<actor>/<expr>/{<cam>.jpg, params.json}`
  and `shape/fsmview_trainset/<actor>/<expr>.ply`) at `FACESCAPE_RAW_*`. It is never modified; Phase 0 copies it
  to the working copy.
- **Faceform Wrap:** `WrapCmd` plus its license (`WrapCmd`, `WrapLicense` in `wrap_script_facescape/.env`). The
  license is tied to the account and machine.
- **Python environments** (rebuild steps in `envs/README.md`):
  - `VENV_PYTHON` = `faceview_venv`: phases 1, 3 and 4, and the backend.
  - `SAPIENS_PYTHON` = `faceview_sapiens_venv`: phase 2.
- **Models:** RAFT `raft_large_C_T_SKHT_V2-ff5fadd5.pth` (`RAFT_CHECKPOINT_PATH`) and Sapiens2
  (`SAPIENS_CHECKPOINT_ROOT`). CoTracker's `scaled_offline.pth` is included in `FaceView/co-tracker/checkpoints/`.
- **Other:** `landmark_lines.npy` (`LANDMARK_LINES_PATH`), and Node 20.x for the frontend.
- **State, copied over or started empty:**
  - `label_tracker.json` (`LABEL_TRACKER_PATH`, "FaceScape" key);
  - the landmarks root (`FACESCAPE_LANDMARKS_ROOT`);
  - `invalid_pairs.json` (`INVALID_PAIRS_PATH`);
  - `FaceView/completed_expressions_sapiens.json` (Phase 2 progress; created on first run).

## Running: Phase 0 to Phase 5

Below, `$VENV_PYTHON` / `$SAPIENS_PYTHON` are the two interpreters, and `$RAW_*` / `$WORK_*` are the
`FACESCAPE_RAW_*` / `FACESCAPE_*` roots from `.env`. Status changes refer to the `"FaceScape"` key of
`label_tracker.json`.

Each actor/expression pair moves through these statuses:

```
(no entry) --P2--> unlabeled --P3--> unreviewed --P4--> unconfirmed --P5 confirm--> confirmed (+ export)
```

### Phase 0: copy the download to the working tree
`--ignore-existing` is required. Without it, a re-run reverts files that Phase 1 already transformed.
```bash
rsync -av --ignore-existing --exclude='*.zip' $RAW_IMAGES/ $WORK_IMAGES/   # FACESCAPE_RAW_IMAGES_ROOT -> FACESCAPE_IMAGES_ROOT
rsync -av --ignore-existing --exclude='*.zip' $RAW_SHAPE/  $WORK_SHAPE/    # FACESCAPE_RAW_SHAPE_ROOT  -> FACESCAPE_SHAPE_ROOT
```

### Phase 1: align every pair to the shared origin space
```bash
cd wrap_script_facescape/coordinate_transforms
# one-time, only if the origin target (actor 1, 1_neutral) is not in the working tree yet:
$VENV_PYTHON run_origin_transform.py
# new actors (re-runnable; finished pairs are skipped via a .transform_complete marker, --force redoes them):
$VENV_PYTHON run_batch_transform.py --actor-min 21 --actor-max 30 --parallel 4 | tee phase1_21_30.log
```
- **What it does:** solves a similarity transform per pair against the origin target (cameras matched by serial
  number), reindexes the cameras to the target's numbering, and rewrites `params.json` and the `.ply` in place in
  the working tree.
- **Other flags:** `--exclude-expressions`, `--dry-run`, `--rmse-threshold-pos/rot`, `--min-shared-cameras`.
- **Results:** successes go to `manifest.jsonl`, failures to `failures.jsonl`.
- **Then record the invalid pairs** (Phases 2 and 4 skip them; the `*_invalid.py` scripts handle them):
  ```bash
  $VENV_PYTHON - <<'EOF'
  import re, json, os, sys
  sys.path.insert(0, "..")
  import facescape_paths as fp                    # INVALID_PAIRS_PATH from wrap_script_facescape/.env
  pairs = [dict(zip(("actor", "expression", "reason"), m.groups())) for line in open("phase1_21_30.log")
           if (m := re.match(r"^FAIL\s+\[actor=(\S+) expression=(\S+)\]: (.+)$", line.strip()))]
  path = fp.INVALID_PAIRS_PATH                    # merged with any pairs already recorded
  old = json.load(open(path))["pairs"] if os.path.exists(path) else []
  allp = {(p["actor"], p["expression"]): p for p in old + pairs}
  json.dump({"count": len(allp), "pairs": list(allp.values())}, open(path, "w"), indent=2)
  EOF
  ```
- **Check:** most expressions per actor should pass. A whole actor failing uniformly points to bad raw
  calibration; see README "Phase 1" for this and the other checks.

### Phase 2: Sapiens2 standard landmarks (eyelids/lips) on every expression's photos
```bash
# after a large new batch of actors, refresh the labelled camera subset (writes facescape_cameras.json):
cd wrap_script_facescape/labeling && $VENV_PYTHON compute_facescape_cameras.py
# label: either the shell wrapper (one GPU) ...
cd FaceView && bash script_process_expression_sapiens.sh --min-actor 21 --max-actor 30 --parallel 3
# ... or the multi-GPU wrapper (polls forever for new actors; --target-only for a single sweep):
cd wrap_script_facescape && $VENV_PYTHON run_process_expression_sapiens_batch.py --gpu-ids 0 1 2 3 --parallel 3
# one pair while iterating (add --sanity-render to write annotated JPGs to SANITY_RENDER_ROOT):
cd FaceView && $SAPIENS_PYTHON scripts/preprocess/process_expression_sapiens.py 21 1_neutral --parallel 1
```
- **Output:** `FACESCAPE_LANDMARKS_ROOT/<actor>/<expr>/cam_<id>/frame_000000.json`. Each pair goes to
  `unlabeled` (pairs already further along are left alone).
- **Skipped:** invalid pairs.
- **Re-runs:** `--skip-completed-pair` (default in the batch wrapper) skips pairs listed in
  `completed_expressions_sapiens.json`.
- **Invalid pairs:** `run_process_expression_sapiens_batch_invalid.py --gpu-ids ...` (raw cameras).

### Label the neutral expression (FaceView)
Each actor needs a saved neutral expression before Phase 4 will wrap it. Phase 3 falls back to `1_neutral` if none
is saved.
```bash
cd FaceView/backend && $VENV_PYTHON -m uvicorn app.main:app --host 0.0.0.0 --port 8000
cd FaceView/frontend && npm ci && npm run dev          # http://localhost:5173, proxies to BACKEND_URL
```
In the FaceScape tab, open the actor's neutral expression, check its landmarks, and press **Save as Neutral**. This
writes `FACESCAPE_LANDMARKS_ROOT/<actor>/neutral_frame.json`.

### Phase 3: propagate skin and ear/brow/nose/chin points from the neutral to every expression
```bash
cd wrap_script_facescape
$VENV_PYTHON run_neutral_skin_propagation.py                      # sweep every actor with an 'unlabeled' pair
$VENV_PYTHON run_neutral_skin_propagation.py 21 --expressions 2_smile 4_anger   # or one actor / some expressions
bash script_run_neutral_skin_propagation.sh [--dry-run]           # same sweep via the shell wrapper
```
- **How:** for each camera, RAFT or CoTracker tracks each point from the neutral photo to the target photo, then
  the tracks are triangulated.
- **Output:** `skin_frame_000000.json` and `<group>_frame_000000.json` per camera. The neutral's own points come
  from its bootstrap wrap.
- **Status:** `unlabeled` → `unreviewed`.
- **Bootstrap scaling:** the bootstrap neutral wrap goes through `run_pipeline.py`, so it gets the P5 face scaling
  (Phase 4).
- **Not filtered:** it doesn't check `invalid_pairs.json`. Pass `--expressions` to skip known-invalid pairs.
- **Invalid pairs:** `run_neutral_skin_propagation_invalid.py [actor] [--expressions ...]`.

### Phase 4: wrap
```bash
cd wrap_script_facescape
$VENV_PYTHON run_pipeline_batch.py --actor-min 21 --actor-max 30 --parallel 4 --log-dir logs/phase4_21_30
$VENV_PYTHON run_pipeline.py 21 1_neutral                         # one pair
$VENV_PYTHON run_pipeline_invalid.py 21 5_jaw_forward             # one invalid pair (raw space)
```
- **Batch behaviour:** `run_pipeline_batch.py` polls forever for `unreviewed` pairs (gated automatically). It
  wraps each actor's neutral expression first, so the other expressions can chain from it. Actors without
  `neutral_frame.json` are skipped.
- **Batch flags:** `--gpu-ids`, `--pairs`, `--confirmed-only` / `--unconfirmed-only` (re-wrap modes),
  `--no-resume`, `--dry-run`. Run it with `$VENV_PYTHON`, which it also uses for the per-pair subprocesses.
- **Per pair:**
  1. Triangulate the landmarks across all cameras (`keypoint_vertices.json`).
  2. Wrap the FLAME template onto the scan (`wrapped_mesh.obj`).
  3. Set status `unreviewed` → `unconfirmed`.
  4. Render the line and point overlays (`LINE_RENDERS_FACESCAPE_ROOT`, `POINT_OVERLAY_RENDERS_FACESCAPE_ROOT`).
- **P5 face scaling (template wraps only):** this covers the neutral expression and the fallback when the neutral
  wrap is missing.
  - The scan and targets are first scaled so the face matches the template, along the face's own axes:
    vertical = brow-to-chin, horizontal = outer eye corners, depth = their geometric mean.
  - After wrapping, the result is mapped back to world space, and the map is saved in `face_scale.json`.
  - Expressions chained from the neutral wrap are not scaled.
  - `scan.obj` and `keypoint_vertices.json` stay in world space.
- **QA:** `check_skin_landmark_errors.py --threshold 4` reports pairs whose skin landmarks sit more than 4 px off.

### Phase 5: review, confirm and export (FaceView)
- **In the FaceScape tab:** review each `unconfirmed` pair's overlays and confirm it. This moves it to `confirmed`
  and exports in the background:
  - `FINAL_WRAPPED_DATA_ROOT/facescape_<actor>/<expr>.pt` (origin space);
  - a `.ply` in the pair's original scan space, with the inverse alignment re-solved through
    `FACESCAPE_COORDINATE_TRANSFORMS_ROOT`.
- **Unconfirm** reverts the status and deletes both exports.
- **Invalid pairs** are already in raw space, so their export skips the origin-to-raw step.

## Fresh-start checklist for a new cluster
- [ ] Edit all four `.env` files and run `python wrap_script_facescape/facescape_paths.py` until it prints OK.
- [ ] Start from an empty (or copied) `label_tracker.json`, landmarks root and `invalid_pairs.json`. Delete
      `FaceView/completed_expressions_sapiens.json` if it was copied from another cluster.
- [ ] Phase 1 needs the origin target in the working tree. Copy actor 1's transformed `1_neutral` or re-run
      `run_origin_transform.py`.
- [ ] Rebuild both venvs (`envs/README.md`); `npm ci` in `FaceView/frontend`.

## Known state on Medusa (2026-10-17)
- **Missing `params.json`:** the working images root only has it for actors 301+. Earlier actors need Phase 1
  re-run (or the raw/invalid-pair path) before re-wrapping.
- **Bad frames:** actors 32, 41 and 173 wrap badly from scratch (errors of metres); their scan/calibration looks
  to be in a different frame. Check them before re-running.
- **Existing wraps predate P5:** the P5 scaling (2026-10-16) only affects wraps made from now on.
