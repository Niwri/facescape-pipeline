"""File-based tracking of each actor/sequence's review status.

Shared with wrap_script tooling (via the same label_tracker_path setting) so
batch scripts can target only specifically labeled actors/sequences.
"""
from __future__ import annotations

import fcntl
import json
from pathlib import Path
from typing import Dict, Literal

from ..core.config import get_settings
from ..schemas.labels import LabelStatus

Dataset = Literal["Nersemble", "FaceScape"]

# Confirm-*/unconfirm-* transitions only move one step at a time (forward or
# backward); anything else is rejected so the UI can't skip straight from e.g.
# "unlabeled" to "confirmed".
_CONFIRM_LABELED_FROM = "unlabeled"
_CONFIRM_FROM = "unconfirmed"
_UNCONFIRM_LABELED_FROM = "unreviewed"
# unconfirm() undoes mark_unconfirmed()'s automatic unreviewed -> unconfirmed
# step (e.g. to redo run_pipeline_batch_all.py), not confirmed -> unconfirmed --
# that's the separate, explicit unconfirm_confirmed() below.
_UNCONFIRM_FROM = "unconfirmed"

# unconfirm_confirmed() is the deliberate reversal of a person's confirm()
# sign-off (distinct from unconfirm() above, which only undoes the automatic
# mark_unconfirmed() step) -- callers must also delete this pair's
# final_wrapped_data/<actor>/<sequence>.pt export, since it stops being an
# accurate snapshot of a confirmed pair (see api/routes/labels.py).
_UNCONFIRM_CONFIRMED_FROM = "confirmed"

# unreviewed -> unconfirmed is not a person clicking a button -- it's set
# automatically by run_pipeline_batch_all.py once a pair's frames are all
# wrapped with skin landmarks (see mark_unconfirmed below).
_MARK_UNCONFIRMED_FROM = "unreviewed"


class LabelTracker:
    def __init__(self) -> None:
        self.settings = get_settings()

    @property
    def path(self) -> Path:
        return self.settings.label_tracker_path

    def _load(self) -> Dict[str, Dict[str, str]]:
        if not self.path.exists():
            return {}
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except (json.JSONDecodeError, OSError):
            return {}

    def get_all(self, dataset: Dataset) -> Dict[str, Dict[str, str]]:
        """Only the pairs actually recorded in label_tracker.json for this dataset
        namespace -- callers that need the implicit "no_neutral_label"/"unlabeled"
        defaults for pairs absent from it should go through get_status (or pass
        this map to it) rather than treating an absent entry here as meaningful
        on its own."""
        return self._load().get(dataset, {})

    def get_status(
        self,
        actor_id: str,
        sequence_id: str,
        *,
        dataset: Dataset,
        all_statuses: Dict[str, Dict[str, str]] | None = None,
    ) -> LabelStatus:
        data = all_statuses if all_statuses is not None else self._load().get(dataset, {})
        status = data.get(actor_id, {}).get(sequence_id)
        if status is not None:
            return status  # type: ignore[return-value]

        if dataset == "FaceScape":
            # FaceScape has no neutral-frame-selection/CoTracker-propagation step,
            # so "unlabeled" doesn't apply here -- a pair with no tracker entry
            # yet is either not started ("unlabeled") or has landmarks written by
            # process_expression_sapiens.py but not yet moved to "unreviewed" in
            # the tracker; the landmarks directory existing is the only disk
            # signal available (see wrap_script_facescape/README.md Phase 3).
            landmarks_dir = self.settings.facescape_landmarks_root / actor_id / sequence_id
            return "unreviewed" if landmarks_dir.exists() else "unlabeled"

        # No confirm-labeled/confirm ever recorded for this pair. Fall back to
        # whether this ACTOR has a neutral frame on disk (one per actor now,
        # not one per actor/sequence pair -- see LandmarkStore.save_neutral_frame/
        # run_neutral_cotracker_propagation.py's load_actor_neutral_frame), rather
        # than unconditionally reporting "no_neutral_label": every sequence of an
        # actor that has a neutral frame is eligible for cross-sequence
        # propagation, not just the one sequence the neutral was picked from.
        neutral_frame_path = self.settings.landmarks_root / actor_id / "neutral_frame.json"
        return "unlabeled" if neutral_frame_path.exists() else "no_neutral_label"

    def _write(self, dataset: Dataset, actor_id: str, sequence_id: str, status: str) -> None:
        """Read-modify-write, serialized via a flock on a dedicated sibling
        .lock file and made crash-safe via write-to-temp-then-atomic-replace
        on the actual data file -- this file is touched by both the backend
        (confirm-labeled/confirm, mark_unlabeled) and, per its shared config
        setting, wrap_script tooling. Namespaced under a top-level dataset
        key ("Nersemble"/"FaceScape") so the two datasets' actor/sequence ids
        can never collide.

        The lock is deliberately a separate file from the data file, never
        itself replaced -- locking the data file directly would race a
        concurrent replace(): if process A replaces label_tracker.json while
        process B already holds a lock on the (now unlinked) old inode, B's
        read misses A's just-written change and B's own replace silently
        discards it. A lock file that's only ever locked, never swapped,
        avoids that. The replace() itself is what fixes the original bug --
        the previous seek(0)/truncate()/json.dump() sequence left the file
        empty or partially written if the process died in between (e.g. an
        OOM kill), which _load()'s silent except-and-return-{} then
        presented as "no labels recorded" for every pair, not just a load
        error."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.path.with_name(self.path.name + ".lock")
        with lock_path.open("w") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            try:
                data = self._load()
                data.setdefault(dataset, {}).setdefault(actor_id, {})[sequence_id] = status
                tmp_path = self.path.with_name(self.path.name + ".tmp")
                tmp_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
                tmp_path.replace(self.path)
            finally:
                fcntl.flock(lock_file, fcntl.LOCK_UN)

    def mark_unlabeled(self, actor_id: str, sequence_id: str, *, dataset: Dataset = "Nersemble") -> None:
        """Called after a neutral frame is (re)saved -- always resets to
        "unlabeled", since a new neutral frame invalidates any prior
        confirm-labeled/confirm state for this pair."""
        self._write(dataset, actor_id, sequence_id, "unlabeled")

    def confirm_labeled(self, actor_id: str, sequence_id: str, *, dataset: Dataset = "Nersemble") -> LabelStatus:
        current = self.get_status(actor_id, sequence_id, dataset=dataset)
        if current != _CONFIRM_LABELED_FROM:
            raise ValueError(
                f"Cannot confirm labeled from status {current!r} (expected "
                f"{_CONFIRM_LABELED_FROM!r}) for {actor_id}/{sequence_id}"
            )
        self._write(dataset, actor_id, sequence_id, "unreviewed")
        return "unreviewed"

    def mark_unconfirmed(self, actor_id: str, sequence_id: str, *, dataset: Dataset = "Nersemble") -> LabelStatus | None:
        """Called by run_pipeline_batch_all.py once a pair's frames are all wrapped
        with skin landmarks. Only fires from "unreviewed" (the normal state right
        before this pipeline step runs) -- a pair already unconfirmed/confirmed
        from a prior run is left untouched (returns None) rather than resetting
        someone's confirmation on a routine rerun."""
        current = self.get_status(actor_id, sequence_id, dataset=dataset)
        if current != _MARK_UNCONFIRMED_FROM:
            return None
        self._write(dataset, actor_id, sequence_id, "unconfirmed")
        return "unconfirmed"

    def confirm(self, actor_id: str, sequence_id: str, *, dataset: Dataset = "Nersemble") -> LabelStatus:
        current = self.get_status(actor_id, sequence_id, dataset=dataset)
        if current != _CONFIRM_FROM:
            raise ValueError(
                f"Cannot confirm from status {current!r} (expected "
                f"{_CONFIRM_FROM!r}) for {actor_id}/{sequence_id}"
            )
        self._write(dataset, actor_id, sequence_id, "confirmed")
        return "confirmed"

    def unconfirm_labeled(self, actor_id: str, sequence_id: str, *, dataset: Dataset = "Nersemble") -> LabelStatus:
        current = self.get_status(actor_id, sequence_id, dataset=dataset)
        if current != _UNCONFIRM_LABELED_FROM:
            raise ValueError(
                f"Cannot unconfirm labeled from status {current!r} (expected "
                f"{_UNCONFIRM_LABELED_FROM!r}) for {actor_id}/{sequence_id}"
            )
        self._write(dataset, actor_id, sequence_id, "unlabeled")
        return "unlabeled"

    def unconfirm(self, actor_id: str, sequence_id: str, *, dataset: Dataset = "Nersemble") -> LabelStatus:
        """Undoes mark_unconfirmed()'s automatic transition, e.g. so
        run_pipeline_batch_all.py can be rerun on this pair. Distinct from
        unconfirm_confirmed() below, which reverts an actual "confirmed"
        sign-off."""
        current = self.get_status(actor_id, sequence_id, dataset=dataset)
        if current != _UNCONFIRM_FROM:
            raise ValueError(
                f"Cannot unconfirm from status {current!r} (expected "
                f"{_UNCONFIRM_FROM!r}) for {actor_id}/{sequence_id}"
            )
        self._write(dataset, actor_id, sequence_id, "unreviewed")
        return "unreviewed"

    def unconfirm_confirmed(self, actor_id: str, sequence_id: str, *, dataset: Dataset = "Nersemble") -> LabelStatus:
        """Reverts a "confirmed" sign-off back to "unconfirmed" -- e.g. someone
        confirmed a pair before noticing a wrapping issue. Only fires from
        "confirmed"; the route handler is responsible for also deleting this
        pair's final_wrapped_data/<actor>/<sequence>.pt export, since a
        reverted pair shouldn't keep serving a stale compact export."""
        current = self.get_status(actor_id, sequence_id, dataset=dataset)
        if current != _UNCONFIRM_CONFIRMED_FROM:
            raise ValueError(
                f"Cannot unconfirm confirmed from status {current!r} (expected "
                f"{_UNCONFIRM_CONFIRMED_FROM!r}) for {actor_id}/{sequence_id}"
            )
        self._write(dataset, actor_id, sequence_id, "unconfirmed")
        return "unconfirmed"


label_tracker = LabelTracker()
