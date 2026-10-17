"""Raw-frame step between wrapped frames, per Nersemble actor.

Landmark files (and the delegation videos) still cover every
LANDMARK_FRAME_STEP-th raw frame for every actor; only which of those frames
get WRAPPED (and rendered) depends on the actor: actors from
SPARSE_WRAP_FIRST_ACTOR on are wrapped every SPARSE_WRAP_FRAME_STEP raw frames
(0, 25, 50, ...), everyone else every LANDMARK_FRAME_STEP (0, 5, 10, ...).
Used by run_pipeline_batch_all.py (which frames to wrap) and by
test_render_gt.py / render_point_overlay.py (which frames to render, and the
video fps).
"""
from __future__ import annotations

LANDMARK_FRAME_STEP = 5
SPARSE_WRAP_FIRST_ACTOR = 102
SPARSE_WRAP_FRAME_STEP = 25


def wrap_frame_step(actor_id: str | int) -> int:
    """Raw-frame step between wrapped frames for this actor (always a
    multiple of LANDMARK_FRAME_STEP). Non-numeric actor ids keep the default."""
    try:
        return SPARSE_WRAP_FRAME_STEP if int(actor_id) >= SPARSE_WRAP_FIRST_ACTOR else LANDMARK_FRAME_STEP
    except (TypeError, ValueError):
        return LANDMARK_FRAME_STEP
