"""Single source of truth for actor/sequence pairs and whole actors excluded
from wrap_script's Nersemble batch tools (run_pipeline_batch_all.py,
run_neutral_cotracker_propagation.py) -- backed by JSON files rather than
Python constants so both scripts read the exact same list instead of keeping
separate copies that can silently drift apart, and so an entry can be
added/removed without touching either script.

ignore_actor_sequence_pairs.json: a list of {"actor", "sequence", "reason"}
    objects -- "reason" is optional, purely documentation for why that pair
    is excluded (e.g. known-bad upstream data).
ignore_actors.json: a list of actor id strings, each excluded entirely
    (every sequence) -- the coarser cousin of the pairs list above, for when
    a whole actor needs excluding rather than specific pairs.

Both are only consulted by callers that opt in via --ignore-list; loading
them unconditionally at import time is cheap regardless (tiny JSON files, no
heavy dependencies), unlike importing run_pipeline_batch_all.py itself for
these two constants (see that module and run_neutral_cotracker_propagation.py
for why that's avoided -- it would eagerly load SAM/U2Net onto the GPU via
its run_pipeline_all -> face_ray_masking import chain).
"""
from __future__ import annotations

import json
from pathlib import Path

_IGNORE_ACTOR_SEQUENCE_PAIRS_PATH = Path(__file__).with_name("ignore_actor_sequence_pairs.json")
_IGNORE_ACTORS_PATH = Path(__file__).with_name("ignore_actors.json")


def _load_ignore_actor_sequence_pairs() -> list[tuple[str, str]]:
    with _IGNORE_ACTOR_SEQUENCE_PAIRS_PATH.open("r", encoding="utf-8") as f:
        entries = json.load(f)
    return [(entry["actor"], entry["sequence"]) for entry in entries]


def _load_ignore_actors() -> frozenset[str]:
    with _IGNORE_ACTORS_PATH.open("r", encoding="utf-8") as f:
        return frozenset(json.load(f))


IGNORE_ACTOR_SEQUENCE_PAIRS: list[tuple[str, str]] = _load_ignore_actor_sequence_pairs()
IGNORE_ACTORS: frozenset[str] = _load_ignore_actors()
