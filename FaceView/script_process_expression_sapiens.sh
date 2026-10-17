#!/usr/bin/env bash
set -euo pipefail

# Run from wherever it's invoked, but resolve relative paths against this
# script's own location since process_expression_sapiens.py expects to be run
# with cwd == repo root.
cd "$(dirname "${BASH_SOURCE[0]}")"

# Paths come from backend/.env (SAPIENS_PYTHON, SAPIENS_CHECKPOINT_ROOT, FACESCAPE_*), not this file.
set -a; source backend/.env; set +a
VENV_PYTHON="${SAPIENS_PYTHON}"
MIN_ACTOR="1"
MAX_ACTOR="1"

# Expressions to skip entirely, even if present -- unlike Nersemble's
# TARGET_SEQUENCES allowlist, this is a blacklist: every expression found for
# an actor is processed by default, minus whatever's listed here.
EXCLUDE_EXPRESSIONS=(
)

# Actor/expression pairs to process concurrently, via process_expression_sapiens.py's
# own --parallel (a thread pool sharing one warmed-up model — NOT one process per
# pair, which would multiply GPU memory instead of just wall-clock time).
PARALLEL=3

# Off by default, same as process_expression_sapiens.py's own flag -- writes
# annotated copies of each processed camera to
# /scratch/ondemand32/irwinngo/temp_renders/<actor>_<expression>/<camera_id>.jpg
# for manual visual confirmation. Enable via --sanity-render.
SANITY_RENDER=false

# On by default -- skips any camera that already has a frame_000000.json, so a
# rerun (e.g. after adding new actors) only does new work. Disable via
# --no-skip-completed to force a full regeneration of already-processed
# cameras too (e.g. after a SAPIENS_LANDMARK_MAP change, or to pick up a Phase 1
# coordinate-transform fix that predates an already-written frame file).
SKIP_COMPLETED=true

usage() {
  echo "Usage: $0 [--min-actor N] [--max-actor N] [--parallel N] [--sanity-render] [--no-skip-completed]" >&2
  echo "Runs process_expression_sapiens.py (faceview_sapiens_venv) for actors \$MIN_ACTOR..\$MAX_ACTOR (default: $MIN_ACTOR..$MAX_ACTOR)" >&2
  echo "  --min-actor N         Lowest actor id to process, inclusive (default: $MIN_ACTOR)" >&2
  echo "  --max-actor N         Highest actor id to process, inclusive (default: $MAX_ACTOR)" >&2
  echo "  --parallel N          Actor/expression pairs to process concurrently (default: $PARALLEL)" >&2
  echo "  --sanity-render       Also write annotated overlay JPGs to temp_renders/ for manual review (default: off)" >&2
  echo "  --no-skip-completed   Reprocess cameras that already have a frame_000000.json (default: skip them)" >&2
  echo "Excluded expressions: ${EXCLUDE_EXPRESSIONS[*]:-(none)}" >&2
  echo "Cameras: loaded from facescape_cameras.json (see wrap_script_facescape/labeling/compute_facescape_cameras.py)" >&2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --min-actor)
      [[ $# -ge 2 ]] || { echo "--min-actor requires a value" >&2; exit 1; }
      MIN_ACTOR="$2"
      shift 2
      ;;
    --max-actor)
      [[ $# -ge 2 ]] || { echo "--max-actor requires a value" >&2; exit 1; }
      MAX_ACTOR="$2"
      shift 2
      ;;
    --parallel)
      [[ $# -ge 2 ]] || { echo "--parallel requires a value" >&2; exit 1; }
      PARALLEL="$2"
      shift 2
      ;;
    --sanity-render)
      SANITY_RENDER=true
      shift
      ;;
    --no-skip-completed)
      SKIP_COMPLETED=false
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage
      exit 1
      ;;
  esac
done

if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "venv python not found or not executable: $VENV_PYTHON" >&2
  exit 1
fi

EXTRA_ARGS=()
if [[ "$SANITY_RENDER" == true ]]; then
  EXTRA_ARGS+=(--sanity-render)
fi
if [[ "$SKIP_COMPLETED" == true ]]; then
  EXTRA_ARGS+=(--skip-completed)
fi

# FaceScape actor ids are plain unpadded integers ("1", not "001") -- see
# process_expression_sapiens.py's _iter_targets, which compares them as
# integers, not strings.
exec "$VENV_PYTHON" scripts/preprocess/process_expression_sapiens.py \
    --actor-min "$MIN_ACTOR" --actor-max "$MAX_ACTOR" \
    --exclude-expressions "${EXCLUDE_EXPRESSIONS[@]}" \
    --parallel "$PARALLEL" "${EXTRA_ARGS[@]}"
