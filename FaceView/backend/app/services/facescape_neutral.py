"""Reviewer-saved neutral expression for FaceScape actors -- the FaceScape
counterpart to landmark_store.py's Nersemble neutral-frame save/load
(_actor_neutral_frame_path/save_neutral_frame/load_actor_neutral_frame), same
<root>/<actor_id>/neutral_frame.json file-path convention, adapted for
FaceScape's shape: one static expression stands in for Nersemble's video
frame, so the saved payload is just {"expression_id": ...} instead of
{"sequence_id": ..., "frame_idx": ...}.

Consumed by wrap_script_facescape/run_neutral_skin_propagation*.py's own
resolve_neutral_expression() as the first candidate tried, ahead of the
hardcoded NEUTRAL_EXPRESSION ("1_neutral") fallback chain.

save_actor_neutral_expression() also resets every one of this actor's
expression pairs back to "unlabeled" in label_tracker.json, same as
landmark_store.save_neutral_frame's own Nersemble equivalent -- see its
docstring below.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..core.config import get_settings
from .label_tracker import label_tracker


def _actor_neutral_expression_path(actor_id: str) -> Path:
    settings = get_settings()
    actor_dir = settings.facescape_landmarks_root / actor_id
    actor_dir.mkdir(parents=True, exist_ok=True)
    return actor_dir / "neutral_frame.json"


def save_actor_neutral_expression(actor_id: str, expression_id: str) -> None:
    """Save this actor's ONE neutral (reference) expression. Reads the file
    back to confirm the write actually landed before reporting success, same
    verify-after-write discipline as landmark_store.save_neutral_frame (this
    file lives on the same scratch/NFS storage as the rest of the pipeline).
    Overwrites whatever expression this actor's neutral previously pointed
    to -- there is only ever one per actor."""
    path = _actor_neutral_expression_path(actor_id)
    with path.open("w", encoding="utf-8") as handle:
        json.dump({"expression_id": expression_id}, handle)
    with path.open("r", encoding="utf-8") as handle:
        saved = json.load(handle)
    if saved.get("expression_id") != expression_id:
        raise IOError(
            f"Neutral expression verification failed for actor {actor_id}: "
            f"wrote expression_id={expression_id!r}, read back {saved!r} from {path}"
        )

    # A (re)saved neutral expression invalidates every OTHER expression's prior
    # propagated skin/standard-index landmarks (run_neutral_skin_propagation.py
    # seeds every one of them from whichever expression is currently this
    # actor's neutral) -- same "reset every sibling back to unlabeled" reasoning
    # as landmark_store.save_neutral_frame's own Nersemble equivalent. "Every
    # pair connected to this actor" is the same set run_pipeline_batch_temp.py's
    # discover_pairs/facescape_dataset_index.py's own scan treat as this
    # actor's pairs: every facescape_images_root/<actor_id>/<expression_id>
    # with a params.json (Phase 1 has transformed it), regardless of whether
    # it currently has any tracker entry or landmarks yet.
    settings = get_settings()
    actor_images_dir = settings.facescape_images_root / actor_id
    if actor_images_dir.is_dir():
        for expr_dir in sorted(actor_images_dir.iterdir()):
            if expr_dir.is_dir() and (expr_dir / "params.json").exists():
                label_tracker.mark_unlabeled(actor_id, expr_dir.name, dataset="FaceScape")


def load_actor_neutral_expression(actor_id: str) -> str | None:
    """Return this actor's saved neutral expression id, or None if it hasn't
    been saved yet (the normal state before a reviewer has ever clicked
    "Save as Neutral" for this actor)."""
    settings = get_settings()
    path = settings.facescape_landmarks_root / actor_id / "neutral_frame.json"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload.get("expression_id")
