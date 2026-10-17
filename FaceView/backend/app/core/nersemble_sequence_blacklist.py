"""Nersemble sequences excluded from the frontend's dataset listing --
EXP-6-tongue-1/EXP-7-tongue-2 (tongue expressions) are out of scope for this
project's FLAME wrapping pipeline. Mirrors wrap_script/sequence_blacklist.py's
same constant (kept as a separate copy since this backend and wrap_script are
independent codebases/environments, not sharing an import path) -- keep the
two in sync by hand if this list ever changes.
"""
from __future__ import annotations

BLACKLISTED_SEQUENCES = frozenset({"EXP-6-tongue-1", "EXP-7-tongue-2"})
