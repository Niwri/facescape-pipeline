# FaceScape Pipeline — Usage Guide

> **Portable copy:** every path below is a Medusa example. In this repo all paths come from
> `wrap_script_facescape/.env` (via `facescape_paths.py`) and `FaceView/backend/.env`; see `../HANDOFF.md`.

How to onboard FaceScape actor/expression data into the same review pipeline
Nersemble already uses, from raw scans through to a confirmed, exported wrapped
mesh. Written so a **new batch of actors** (e.g. actors 21+, added to
FaceScape later) can be run through end to end by following this document,
without re-deriving the design from scratch.

Six phases, run in order:

| Phase | What runs | Status transition (`label_tracker.json`, `"FaceScape"` key) |
|---|---|---|
| 0 | `rsync` copy to the working tree | none |
| 1 | Coordinate-transform cameras + mesh into shared origin space | none (transform is not persisted — see below) |
| 2 | Sapiens2 landmark labeling | *explicit*: `label_tracker.mark_unlabeled()` writes `"unlabeled"` once a pair's landmarks exist (see below) |
| 3 | `run_neutral_skin_propagation.py` (propagates skin landmarks from `1_neutral`) | `"unlabeled"` → `"unreviewed"` (per pair, via `label_tracker.confirm_labeled()`) |
| 4 | `run_pipeline_batch.py` (wraps the neutral FLAME template onto the scan) | `"unreviewed"` → `"unconfirmed"` (per pair, via `run_pipeline.py`'s `mark_unconfirmed()` call) |
| 5 | Confirming on the frontend | `"unconfirmed"` → `"confirmed"`, plus background export of the wrapped mesh |

Phases 0-1 are self-contained. Phase 2 depends on Phase 1's output. Phase 3
depends on Phase 2's landmarks. Phase 4 is now gated on Phase 3 having run
(it only ever processes pairs at `"unreviewed"`, automatically — see below).
Phase 5 is the reviewer's sign-off action on the frontend.

**Note on phase count**: earlier planning assumed a *standard-landmark*
propagation script analogous to Nersemble's `run_neutral_cotracker_propagation.py`
(propagating landmarks from `1_neutral` over a video via CoTracker). No such
script exists for FaceScape and none is needed for the *standard* 74-point
landmarks — FaceScape already has one static, no-motion-blur capture *per
expression*, so `process_expression_sapiens.py` (Phase 2) just runs sapiens2
directly on every expression's own images. `run_neutral_skin_propagation.py`
*is* a real, required propagation phase (Phase 3) — but only for the
*skin* landmarks (FLAME index ≥ 74), which genuinely have no per-expression
ground truth and must be tracked from `1_neutral`'s own photos into each
target expression's photos via CoTracker, the same way Nersemble propagates
its skin points.

## Prerequisites

- Raw FaceScape data lands at `/scratch/thirty/irwinngo/facescape/multiview/{images,shape}/fsmview_trainset/<actor>/<expression>/...` (images: `<camera_id>.jpg` + `params.json` per actor/expression; shape: `<expression>.ply` per actor).
- If new actors arrive as a `.zip` (as `fsmview_trainset_images_NNN-NNN.zip`), extract with `unzip -n <archive> -d /scratch/thirty/irwinngo/facescape/multiview/images/` — the `-n` (never overwrite) flag makes this safe to re-run against an archive that also contains already-extracted actors.
- `faceview_venv` (`/scratch/ondemand32/irwinngo/envs/faceview_venv/bin/python`) is needed for every phase except Phase 5 (the frontend action) and, partially, Phase 2: `numpy`/`trimesh`/`cv2` for Phases 0-1, `torch`/CoTracker for Phase 3 (`run_neutral_skin_propagation.py`, which appends `FaceView/co-tracker` to `sys.path` itself), and `cv2`/`trimesh` again for Phase 4 (`run_pipeline_batch.py`/`run_pipeline.py`, transitively via `wrap_script/triangulate.py`'s own `cv2` import — the plain system `python3` has neither `cv2` nor `trimesh`, so it cannot run Phase 4). `faceview_sapiens_venv` is needed for Phase 2 (sapiens2/torch) specifically.
- Canonical path settings for all of this live in `FaceView/backend/app/core/config.py`: `facescape_images_root`/`facescape_shape_root` (Phase-1 output, `irwinngo1`), `facescape_raw_images_root` (pristine `irwinngo` source, read-only), `facescape_landmarks_root`, `facescape_invalid_pairs_path`, `facescape_cameras_path`, `facescape_output_root`, `label_tracker_path`.

## Phase 0 — Copy to the working tree

Never modify `/scratch/thirty/irwinngo/facescape` (pristine source). All
downstream work reads from there and writes into
`/scratch/thirty/irwinngo1/facescape`. There is no dedicated script for this
step — it's a literal `rsync` invocation, run by hand:

```bash
rsync -av --ignore-existing --exclude='*.zip' \
  /scratch/thirty/irwinngo/facescape/multiview/images/fsmview_trainset/ \
  /scratch/thirty/irwinngo1/facescape/multiview/images/fsmview_trainset/
rsync -av --ignore-existing --exclude='*.zip' \
  /scratch/thirty/irwinngo/facescape/multiview/shape/fsmview_trainset/ \
  /scratch/thirty/irwinngo1/facescape/multiview/shape/fsmview_trainset/
```

**`--ignore-existing` is load-bearing, not optional.** Plain `rsync -av`
decides what to (re-)copy by comparing size+mtime, and Phase 1 changes both
of those on every file it transforms — so re-running a bare `rsync -av` here
after Phase 1 has already run decides already-transformed `params.json`/image
files "need updating" and silently reverts them to raw, untransformed data.
This happened for real once already (155/400 pairs, including the origin
target itself, got clobbered this way and had to be re-run through Phase 1
with `--force`). `--ignore-existing` makes rsync skip any file that already
exists in the destination at all, regardless of mtime, so it only ever adds
genuinely new actors/expressions and can never touch anything Phase 1 has
already written.

This step is just a bulk starting copy, deliberately blind to whether a file
is actually transformed yet. Phase 1's `run_batch_transform.py` is explicitly
written to *not* trust "does `params.json` exist" as an "already done" signal
for exactly this reason — see its `.transform_complete` marker logic below.

## Phase 1 — Align to origin space

Code: `/scratch/ondemand32/irwinngo/wrap_script_facescape/coordinate_transforms/`.

**Why this exists**: every actor/expression's `params.json` lives in its own
arbitrary coordinate frame. All of them need to be expressed in one shared
"origin space," anchored by a transform solved once in Blender
(`1_1_neutral_to_origin.json`, from `~/test_code/`) that maps actor 1's
`1_neutral` capture onto a fixed origin. Every other actor/expression is then
aligned *directly* to that already-origin-space target, matched by camera
serial number (`<i>_sn` — the one stable physical-camera identity; the raw
per-file camera *index* `i` is **not** stable across actors, see
`params_io.py`'s docstring) — not chained through a per-actor precomputed
transform.

**One-time setup** (only needed once, already done — skip unless the origin
target itself needs regenerating):
```bash
cd /scratch/ondemand32/irwinngo/wrap_script_facescape/coordinate_transforms
/scratch/ondemand32/irwinngo/envs/faceview_venv/bin/python run_origin_transform.py
```
Writes the origin-space version of actor 1's `1_neutral` into `irwinngo1` —
this becomes the alignment target *and* the canonical camera-index source for
everything else (see below).

**For new actors** (this is the step to re-run when actors are added):
```bash
cd /scratch/ondemand32/irwinngo/wrap_script_facescape/coordinate_transforms
/scratch/ondemand32/irwinngo/envs/faceview_venv/bin/python run_batch_transform.py \
    --actor-min 21 --actor-max 30 --parallel 4
```
(Omit `--actor-min`/`--actor-max` to sweep every actor found under the source
images root; already-completed pairs are skipped automatically via a
`.transform_complete` marker file per expression directory — use `--force` to
redo pairs that already have one.)

Per actor/expression, this:
1. Solves a similarity transform `(s, Rg, tg)` (scale + rotation +
   translation) aligning that pair's cameras onto the origin-space target,
   via shared serial number (`solve_alignment.align_params_to_target`).
2. **Reindexes every camera by serial number** to match the target's own
   numbering (`build_canonical_index`/`remap_and_transform` in `params_io.py`)
   — camera *index* `i` means a different physical camera in almost every
   actor's raw file, so this step is not optional. Cameras with no canonical
   match get dropped. The corresponding `<index>.jpg` files get renamed to
   match.
3. Applies the same transform to that expression's scanned mesh (`.ply`),
   via `transform_mesh.transform_mesh_file`.
4. Writes a `.transform_complete` marker on success.

**The solved `(s, Rg, tg)` itself is never written to disk.** Only the
*result* of applying it — the origin-space `params.json` and the transformed
`.ply` — survives. `manifest.jsonl` logs the scalar `scale` (for eyeballing
residuals/spot-checking), but not `Rg`/`tg`, so even the manifest isn't
enough to invert the transform later. Whatever downstream step needs to map
a pair back out of origin space has to re-solve it — see Phase 5 below,
where `facescape_coordinate_transform.solve_facescape_inverse_transform()`
does exactly that by re-running the same solver in the opposite direction
(origin-space `params.json` as source, this pair's own untouched raw
`params.json` as target). That module's own docstring spells out why: adding
a persistence step to Phase 1 now wouldn't retroactively help the thousands
of pairs already processed without one.

Failures (residual too high, too few shared cameras, missing mesh, etc.) are
logged to `failures.jsonl` in the same directory, **not** silently skipped or
written with bad data. Successes are logged to `manifest.jsonl`.

**After running**, regenerate the invalid-pairs record (used downstream by
Phase 2 and Phase 4 to know which pairs to skip — **not** Phase 3, see its
own section below):
```bash
python3 -c "
import re, json
from pathlib import Path
log = 'PATH_TO_THIS_RUNS_STDOUT_LOG'   # capture run_batch_transform.py's own stdout to a file
dest_root = Path('/scratch/thirty/irwinngo1/facescape/multiview/images/fsmview_trainset')
pat = re.compile(r'^FAIL\s+\[actor=(\S+) expression=(\S+)\]: (.+)$')
pairs = []
for line in open(log):
    m = pat.match(line.strip())
    if m:
        pairs.append({'actor': m.group(1), 'expression': m.group(2), 'reason': m.group(3)})
json.dump({'count': len(pairs), 'pairs': pairs}, open('/scratch/thirty/irwinngo1/invalid_pairs.json', 'w'), indent=2)
"
```
(Redirect `run_batch_transform.py`'s stdout to a log file when you run it, so
this step has something to parse. `run_batch_transform.py` itself does not
write `invalid_pairs.json` — this is a separate, manual post-processing
step.)

**Verification** (don't skip this for a new actor batch, especially the first
one from a new source/session):
- Check `manifest.jsonl`/`failures.jsonl` pass rate — a healthy actor should
  see the vast majority of expressions pass, with residuals in the same
  ballpark as already-processed actors (`pos_rmse` ~0.03-0.07, `rot_rmse`
  ~1.9-2.3° was the normal range observed for actors 1-14's calmer
  expressions; big jaw/mouth expressions run somewhat higher and can
  legitimately fail the threshold — that's expected, not a bug).
- If an actor fails **uniformly across every expression** with a much larger,
  smoothly-distributed per-camera rotation error, that's a sign of a genuine
  data problem with that actor's raw calibration (seen with actors 11/14) —
  not something `run_batch_transform.py`'s candidate search can fix by itself.
- Spot check the canonical reindex directly: confirm `<canonical_index>_sn`
  in a newly-transformed actor's `params.json` matches the origin target's
  `<canonical_index>_sn` at a few indices, and open `<canonical_index>.jpg`
  for the new actor and the origin target side by side.

## Phase 2 — Sapiens2 landmark labeling

Code: `FaceView/scripts/preprocess/process_expression_sapiens.py`. FaceScape
counterpart to `process_sequence_sapiens.py` (same sapiens2
keypoints308 detector/pose model and `SAPIENS_LANDMARK_MAP`), adapted for
FaceScape's data shape: one static JPG per `(actor, expression, camera)`
instead of a per-camera video, no FLAME model/projection step. Only the
**eyelid/lip** subset of `SAPIENS_LANDMARK_MAP` (`EYELID_LIP_LANDMARK_MAP`)
is processed this pass — ears/nosebridge/eyebrows/undernose are deliberately
deferred and stay present-but-inert in the map so no index renumbering will
be needed later.

Runs sapiens2 **directly on every expression's own photos** — there is no
propagation step from `1_neutral`, and none is needed: unlike Nersemble
(one manually-labeled neutral video frame, propagated forward/backward via
CoTracker through a whole clip), every FaceScape expression already has its
own full multiview capture, so the detector just runs on each pair
independently.

Reads from `settings.facescape_images_root`, i.e. the Phase-1-transformed
`irwinngo1` copy — never the pristine source. Pairs listed in
`settings.facescape_invalid_pairs_path` (Phase 1 failures) are skipped
automatically.

**Which cameras get labeled**: a culled subset (front/left-front/right-front
groups), computed once by `labeling/compute_facescape_cameras.py` and cached
at `~/wrap_script_facescape/facescape_cameras.json`. Re-run this after adding
a meaningfully large new batch of actors:
```bash
cd /scratch/ondemand32/irwinngo/wrap_script_facescape/labeling
/scratch/ondemand32/irwinngo/envs/faceview_venv/bin/python compute_facescape_cameras.py
```

**Running the labeling batch** (via the shell wrapper, `faceview_sapiens_venv`):
```bash
/scratch/ondemand32/irwinngo/FaceView/script_process_expression_sapiens.sh \
    --min-actor 21 --max-actor 30 --parallel 3
```
Or directly for a single actor/expression while iterating:
```bash
/scratch/ondemand32/irwinngo/envs/faceview_sapiens_venv/bin/python \
    /scratch/ondemand32/irwinngo/FaceView/scripts/preprocess/process_expression_sapiens.py \
    21 1_neutral --parallel 1
```
Flags: `--exclude-expressions` (blacklist — every expression found is
processed by default), `--cameras` (override the culled default),
`--skip-completed` (per-camera, skips a `frame_000000.json` that already
exists), `--skip-completed-pair` (skips whole pairs already recorded in
`completed_expressions_sapiens.json`), `--parallel` (thread pool sharing one
warmed-up model).

Output lands at
`facescape_landmarks_root/<actor>/<expression>/cam_<camera_id>/frame_000000.json`
(default root: `/scratch/ondemand32/irwinngo/landmarks_facescape`).

**Sanity-checking a new actor batch** — add `--sanity-render`:
```bash
/scratch/ondemand32/irwinngo/envs/faceview_sapiens_venv/bin/python \
    /scratch/ondemand32/irwinngo/FaceView/scripts/preprocess/process_expression_sapiens.py \
    21 1_neutral --cameras 1 4 --parallel 1 --sanity-render
```
Writes annotated copies to
`/scratch/ondemand32/irwinngo/temp_renders/<actor>_<expression>/<camera_id>.jpg`.
Points scored below `SAPIENS_SCORE_THRESHOLD` (0.3) are written as `(-1, -1)`
and skipped in the render.

**Status transition is explicit.** Once a pair has at least one camera's
`frame_000000.json` written, `run_pair()` calls
`label_tracker.mark_unlabeled(actor_id, expression_id, dataset="FaceScape")`
— an unconditional write, *except* it's skipped (with a printed note) if the
pair's current status is already `"unreviewed"`/`"unconfirmed"`/`"confirmed"`,
so rerunning sapiens2 on an already-further-along pair (e.g. to pick up an
improved model) can't silently reset its progress.

For pairs processed *before* this explicit write existed, `LabelTracker.get_status(...,
dataset="FaceScape")` (`backend/app/services/label_tracker.py`) still falls
back, for any pair with no tracker entry, to checking whether
`facescape_landmarks_root/<actor>/<expression>` exists on disk: `"unreviewed"`
if it does, else `"unlabeled"` — so old pairs read back as `"unreviewed"`
directly (skipping `"unlabeled"` entirely) rather than needing a backfill.
`"no_neutral_label"` doesn't apply to FaceScape in this fallback (there's no
neutral-frame/CoTracker-propagation concept for the *standard* landmarks here).

## Phase 3 — Skin landmark propagation

Code: `wrap_script_facescape/run_neutral_skin_propagation.py`. Propagates
"skin" points (FLAME indices ≥ 74, beyond the standard 74-point set Phase 2
labels) from `1_neutral` to every other expression, per actor, using
CoTracker on a 2-image `[neutral, target]` "clip" per camera — FaceScape has
no video to propagate through, so this is CoTracker run once per
(expression, camera) pair rather than chained across a clip. `1_neutral`'s
own skin points are written directly (ground-truth reprojection, no
CoTracker needed) via `write_neutral_skin_frames()`, unconditionally before
the propagation loop — not "propagated to itself".

```bash
cd /scratch/ondemand32/irwinngo/wrap_script_facescape
/scratch/ondemand32/irwinngo/envs/faceview_venv/bin/python \
    run_neutral_skin_propagation.py 1 --expressions 2_smile 4_anger
# omit --expressions to propagate to every 'unlabeled' expression found for that actor
```

**Or sweep every actor at once** (mirrors Nersemble's own
`script_run_neutral_cotracker_propagation.sh` — no actor arg needed):
```bash
./script_run_neutral_skin_propagation.sh [--dry-run]
```
`run_neutral_skin_propagation.py` with no `actor_id` given discovers every
actor with at least one `"unlabeled"` expression (`discover_unlabeled_actors()`)
and processes them all sequentially, sharing one warmed-up CoTracker model
across the whole run. A specific `actor_id` (with or without `--expressions`)
still works exactly as before for targeting one actor.

Writes `facescape_landmarks_root/<actor>/<expression>/cam_<camera>/skin_frame_000000.json`
— a separate file from `frame_000000.json`, same per-point schema as
Nersemble's own `skin_frame_<idx>.json`.

**Does not check `invalid_pairs.json`.** Unlike Phase 2 and Phase 4, this
script has no invalid-pairs filter — with no `--expressions` given, it
propagates to every expression directory found under `facescape_images_root`
for that actor, including ones Phase 1 already flagged as failed. Harmless
(Phase 4 will never wrap those pairs regardless), just wasted CoTracker work;
pass `--expressions` explicitly to skip known-invalid ones on purpose.

**Status transition**: on success (and not `--dry-run`), calls
`label_tracker.confirm_labeled(actor_id, expression_id, dataset="FaceScape")`,
moving `"unlabeled"` → `"unreviewed"`. This only fires from exactly
`"unlabeled"` — a no-op (with a printed note) otherwise, e.g. a rerun on a
pair whose skin points just needed refreshing but is already
`"unreviewed"`/`"unconfirmed"`/`"confirmed"`. **This phase is now a real
prerequisite for Phase 4**: `run_pipeline_batch.py` only ever wraps pairs at
`"unreviewed"`, so a pair skips straight from `"unlabeled"` to being
permanently excluded from wrapping until this phase runs for it.

## Phase 4 — Wrapping

Code: `wrap_script_facescape/run_pipeline_batch.py`, which shells out to
`run_pipeline.py` (one subprocess per pair — no shared model/GPU state to
worry about between pairs).

```bash
/scratch/ondemand32/irwinngo/envs/faceview_venv/bin/python run_pipeline_batch.py \
    --actor-min 21 --actor-max 30 --parallel 4 --log-dir /path/to/logs
```
(`run_pipeline_batch.py` dispatches each pair's `run_pipeline.py` as a
subprocess via `sys.executable` — i.e. whatever interpreter launches
`run_pipeline_batch.py` itself is also what runs every pair, so this must be
`faceview_venv`'s `python`, not the bare system `python3`.)

Pairs in `invalid_pairs.json` (Phase 1 failures) are **always** excluded,
unconditionally — there's no flag to re-include them.

**Gating on `"unreviewed"` is automatic, not opt-in.** `run_pipeline_batch.py`
always restricts the discovered pairs to whichever are currently at
`"unreviewed"` (via `label_tracker.get_status(..., dataset="FaceScape")`)
before running anything — so it never re-wraps a pair that hasn't had Phase 3
run for it yet (still `"unlabeled"`), nor one already
`"unconfirmed"`/`"confirmed"` by a prior run or a human reviewer. There is no
flag to disable this gate.

Per pair, `run_pipeline.py` (not `run_pipeline_batch.py` itself):
1. Triangulates sapiens2 landmarks (Phase 2's `frame_000000.json` files)
   across every camera available for that pair.
2. Wraps the fixed `neutral.obj` FLAME template onto the pair's own scanned
   mesh (`facescape_shape_root/<actor>/<expression>.ply`), via
   `run_wrap_script.wrap_mesh`. Output: `facescape_output_root/<actor>/<expression>/wrapped_mesh.obj`.
3. Calls `label_tracker.mark_unconfirmed(actor_id, expression_id, dataset="FaceScape")`
   — a no-op unless the pair's current status is exactly `"unreviewed"`, so a
   rerun on an already-`"unconfirmed"`/`"confirmed"` pair can't clobber a
   later human decision. There is no per-frame loop (FaceScape has one static
   image set per pair, not a video), so this fires once per pair, not once
   per frame the way Nersemble's equivalent does.
4. Best-effort renders overlay frames for a fixed set of cameras
   (`OVERLAY_FRAME_CAMERA_IDS = ["1", "4", "17"]`) via `render_overlay_frame.py`.

## Phase 5 — Confirming and exporting

Trigger: `POST /labels/confirm` (`FaceView/backend/app/api/routes/labels.py`),
the frontend's confirm action, with `dataset="FaceScape"`.

1. `label_tracker.confirm(actor_id, expression_id, dataset="FaceScape")` moves
   the pair `"unconfirmed"` → `"confirmed"` (raises if it isn't currently
   `"unconfirmed"`).
2. If `dataset == "FaceScape"`, the route schedules
   `export_wrapped_facescape_pair(actor_id, expression_id)`
   (`backend/app/services/wrapped_mesh_export.py`) as a background task, so a
   slow export doesn't block the confirm response.

`export_wrapped_facescape_pair`:
- Loads `facescape_output_root/<actor>/<expression>/wrapped_mesh.obj`.
- Writes `final_wrapped_data_root/facescape_<actor>/<expression>.pt`
  (`{faces, verts, frame_ids}`, single-frame) — **in origin space, unchanged**,
  since existing consumers may already depend on that.
- **Also** converts the same mesh into that pair's own original (pre-Phase-1)
  scan space and writes an extra `facescape_<actor>/<expression>.ply`
  alongside it. This does **not** use a transform "saved" from Phase 1 —
  Phase 1 never persists one (see Phase 1 above). Instead it calls
  `facescape_coordinate_transform.solve_facescape_inverse_transform()`, which
  **re-solves** the alignment at confirm time, in the reverse direction: this
  pair's origin-space `params.json` (`facescape_images_root`) as the "source"
  against its own untouched raw `params.json` (`facescape_raw_images_root`)
  as the "target" — reusing `coordinate_transforms/solve_alignment.py`'s same
  tested solver rather than hand-deriving a matrix inverse. Because this is
  recovering an already-applied exact transform rather than fitting
  independent noisy data, it's expected to fit near-exactly; a warning is
  logged if the residual is unexpectedly large (possible sign of a
  mismatched/stale pair). The resulting `(s, Rg, tg)` is applied to the
  wrapped mesh's vertices via `apply_similarity_to_verts()` before writing the
  `.ply`.
- Best-effort: a solve/write failure for the `.ply` doesn't undo the `.pt`
  export that already succeeded.

Reverting a confirmation (`POST /labels/unconfirm-confirmed`) moves
`"confirmed"` → `"unconfirmed"` and deletes both the `.pt` and the `.ply` via
`delete_exported_facescape_pair()`, so a reverted pair stops serving either as
if it were still confirmed.
