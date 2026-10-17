#!/usr/bin/env bash
set -euo pipefail

# Run from wherever it's invoked, but resolve relative paths (template.wrap,
# neutral.obj, keypoint_<region>.txt) against this script's own location since
# run_neutral_skin_propagation.py expects to be run with cwd == this directory.
cd "$(dirname "${BASH_SOURCE[0]}")"

# VENV_PYTHON comes from this folder's .env (see .env.example).
set -a; source .env; set +a
VENV_PYTHON="${VENV_PYTHON:?set VENV_PYTHON in wrap_script_facescape/.env}"

usage() {
  echo "Usage: $0 [--dry-run]" >&2
  echo "Runs run_neutral_skin_propagation.py (faceview_venv) for every FaceScape actor" >&2
  echo "with at least one actor/expression pair currently at 'unlabeled' status in" >&2
  echo "label_tracker.json -- the sweep itself is done by the python script (no actor" >&2
  echo "arg here), which also moves each pair to 'unreviewed' once its propagation" >&2
  echo "finishes (Phase 3 -- see wrap_script_facescape/README.md)." >&2
  echo "  --dry-run   Passed through: run CoTracker/triangulation but write nothing" >&2
}

DRY_RUN=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=(--dry-run)
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

# No actor arg: run_neutral_skin_propagation.py's own main() sweeps every actor
# with at least one 'unlabeled' expression (via label_tracker.json, see
# discover_unlabeled_actors()) and processes them sequentially, sharing one
# warmed-up CoTracker model across the whole run -- same reasoning as
# wrap_script/script_run_neutral_cotracker_propagation.sh's own comment.
exec "$VENV_PYTHON" run_neutral_skin_propagation.py "${DRY_RUN[@]}"
