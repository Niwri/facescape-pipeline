#!/usr/bin/env bash
set -euo pipefail

# Run from wherever it's invoked, but resolve relative paths (flame_model/,
# output/) against this script's own location since process_sequence_sapiens.py
# expects to be run with cwd == repo root.
cd "$(dirname "${BASH_SOURCE[0]}")"

# Paths come from backend/.env (DATA_ROOT, SAPIENS_PYTHON, SAPIENS_CHECKPOINT_ROOT, NERSEMBLE_TRACKING_*),
# exported so process_sequence_sapiens.py sees them too.
set -a; source backend/.env; set +a
VENV_PYTHON="${SAPIENS_PYTHON}"
MIN_ACTOR="298"
MAX_ACTOR="553"
CAMERAS=("221501007" "222200037" "222200039" "222200042" "222200044" "222200046" "222200049" "222200040" "222200043" "222200036" "222200047")
TARGET_SEQUENCES=(
  "EMO-1-shout+laugh"
  "EXP-2-eyes"
  "EXP-3-cheeks+nose"
  "EXP-4-lips"
  "EXP-8-jaw-1"
  "EMO-2-surprise+fear"
  "EMO-3-angry+sad"
  "EMO-4-disgust+happy"
  "EXP-1-head"
  "EXP-5-mouth"
  "EXP-9-jaw-2"
  "FREE"
  "SEN-01-cramp_small_danger"
  "SEN-02-same_phrase_thirty_times"
  "SEN-03-pluck_bright_rose"
  "SEN-04-two_plus_seven"
  "SEN-05-glow_eyes_sweet_girl"
  "SEN-06-problems_wise_chief"
  "SEN-07-fond_note_friend"
  "SEN-08-clothes_and_lodging"
  "SEN-09-frown_events_bad"
  "SEN-10-port_strong_smokey"
)
# Actor/sequence pairs to process concurrently, via process_sequence_sapiens.py's
# own --parallel (a thread pool sharing one warmed-up model — NOT one process per
# pair, which would multiply GPU memory instead of just wall-clock time).
PARALLEL=3
# Off by default, same as process_sequence_sapiens.py's own --v2 flag -- see
# its own --help for what it changes (fit.npz row selection for --v2-style
# tracking fits). Passed through verbatim via --v2 below when set.
V2=false

usage() {
  echo "Usage: $0 [--max-actor N] [--parallel N] [--v2]" >&2
  echo "Runs process_sequence_sapiens.py (faceview_sapiens_venv) for actors $MIN_ACTOR..$MAX_ACTOR under: $DATA_ROOT" >&2
  echo "  --max-actor N   Highest actor id to process, inclusive (default: $MAX_ACTOR)" >&2
  echo "  --parallel N    Actor/sequence pairs to process concurrently (default: $PARALLEL)" >&2
  echo "  --v2            Passed through to process_sequence_sapiens.py's own --v2 (default: off)" >&2
  echo "Sequences: ${TARGET_SEQUENCES[*]}" >&2
  echo "Cameras: ${CAMERAS[*]}" >&2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --max-actor)
      [[ $# -ge 2 ]] || { echo "--max-actor requires a value" >&2; exit 1; }
      MAX_ACTOR="$(printf '%03d' "$2")"
      shift 2
      ;;
    --parallel)
      [[ $# -ge 2 ]] || { echo "--parallel requires a value" >&2; exit 1; }
      PARALLEL="$2"
      shift 2
      ;;
    --v2)
      V2=true
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

# process_sequence_sapiens.py compares actor ids as strings against 3-digit
# folder names ("096"), so both bounds must be zero-padded -- "96" would match
# nothing below "553".
MIN_ACTOR="$(printf '%03d' "$((10#$MIN_ACTOR))")"
MAX_ACTOR="$(printf '%03d' "$((10#$MAX_ACTOR))")"

EXTRA_ARGS=()
if [[ "$V2" == true ]]; then
  EXTRA_ARGS+=(--v2)
fi

if [[ ! -d "$DATA_ROOT" ]]; then
  echo "Data root not found: $DATA_ROOT" >&2
  exit 1
fi

if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "venv python not found or not executable: $VENV_PYTHON" >&2
  exit 1
fi

# One invocation for the whole actor range: process_sequence_sapiens.py's own
# --actor-min/--actor-max/--sequences discover every matching pair, and
# --parallel fans work out across a thread pool that shares a single
# warmed-up model — so the detector + sapiens2_1b pose model get loaded once
# for this whole run, not once per actor.
#
# Deliberately NOT passing --skip-completed here (unlike its previous default):
# that flag skips a camera view that already has ANY extracted frame_<idx>.json,
# which would leave every already-processed camera's landmark files without the
# newly-added chin point (index 50 -> sapiens2 keypoint 77, added to
# SAPIENS_LANDMARK_MAP) forever. This run is meant to force a full
# re-detection/overwrite of every frame so the chin field actually lands
# everywhere, not just on newly-processed cameras.
exec "$VENV_PYTHON" scripts/preprocess/process_sequence_sapiens.py \
    --actor-min "$MIN_ACTOR" --actor-max "$MAX_ACTOR" \
    --sequences "${TARGET_SEQUENCES[@]}" --cameras "${CAMERAS[@]}" \
    --parallel "$PARALLEL" "${EXTRA_ARGS[@]}"
