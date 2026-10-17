"""Single source of truth for Nersemble sequences excluded from every
wrap_script tool -- EXP-6-tongue-1/EXP-7-tongue-2 (tongue expressions) are out
of scope for this project's FLAME wrapping pipeline.

Imported by both kinds of sequence-discovery code in this directory:
  - TARGET_SEQUENCES-based batch scripts (run_pipeline_batch*.py,
    batch_render_*.py, reproject_side_cameras.py) -- these already only ever
    list sequences their own TARGET_SEQUENCES allowlist names, which doesn't
    currently include either blacklisted sequence, but filtering against this
    module too means a future edit to TARGET_SEQUENCES can't silently
    reintroduce them.
  - Directory-scan scripts with no allowlist at all (run_neutral_cotracker_
    propagation.py and its _wrapping_test.py twin, backfill_neutral_skin_
    landmarks.py, wrap_all_neutral_frames.py, backfill_camera_landmarks.py,
    relabel_skin_landmarks_after_removal.py) -- these walk landmarks_root/data_root
    directly, so without this filter they pick up whatever sequence
    directories actually exist on disk, blacklisted or not.
"""
from __future__ import annotations

BLACKLISTED_SEQUENCES = frozenset({"EXP-6-tongue-1", "EXP-7-tongue-2"})
