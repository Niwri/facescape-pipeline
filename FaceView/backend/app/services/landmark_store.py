"""File-based persistence for landmarks."""
from __future__ import annotations

import json
import logging
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable, List

from ..core.config import get_settings, RAW_FRAME_STRIDE
from ..core.nersemble_sequence_blacklist import BLACKLISTED_SEQUENCES
from ..schemas.common import LandmarkFrame, LandmarkPoint
from ..schemas.landmarks import LandmarkPayload
from .label_tracker import label_tracker

log = logging.getLogger(__name__)

# The standard (0-73) index groups run_neutral_cotracker_propagation.py
# propagates from the neutral frame into their own <prefix>_frame_<idx>.json
# files, rather than overwriting frame_<idx>.json's own sapiens2-detected
# values at those same indices (see that script's module docstring). These
# are exactly the indices sapiens2 can't reliably (re)detect per-frame, which
# is why load_frames/load_all_frames below overlay them on top of
# frame_<idx>.json instead of returning sapiens2's raw per-frame values.
STANDARD_PROPAGATED_GROUPS: List[str] = ["ear", "nosebridge", "undernose", "eyebrow", "chin"]

# Loading a camera means reading hundreds of small JSON files from
# landmarks_root, which lives on NFS: each read costs ~50 ms of latency and
# almost nothing in CPU, so reading them one after another made a single
# camera take ~20 s cold. The bulk loaders below list the camera directory
# once (instead of one exists() per possible group file) and read the files
# concurrently through this shared pool.
_READ_POOL = ThreadPoolExecutor(max_workers=32, thread_name_prefix="landmark-read")


def _read_json_or_none(path: Path) -> dict | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        log.warning("Skipping corrupt landmark file %s (invalid JSON)", path)
        return None


def _read_many(paths: List[Path]) -> List[dict | None]:
    """Parsed JSON for each path (None for missing/corrupt), in input order."""
    return list(_READ_POOL.map(_read_json_or_none, paths))


def _list_names(directory: Path) -> set[str]:
    try:
        return set(os.listdir(directory))
    except FileNotFoundError:
        return set()


class LandmarkStore:
    """Simple JSON-backed landmark storage."""

    def __init__(self) -> None:
        self.settings = get_settings()

    def _camera_dir(self, actor_id: str, sequence_id: str, camera_id: str) -> Path:
        return self.settings.landmarks_root / actor_id / sequence_id / camera_id

    def _frame_path(self, actor_id: str, sequence_id: str, camera_id: str, frame_idx: int) -> Path:
        subdir = self._camera_dir(actor_id, sequence_id, camera_id)
        subdir.mkdir(parents=True, exist_ok=True)
        return subdir / f"frame_{frame_idx:06d}.json"

    def _read_frame(self, frame_path: Path) -> LandmarkFrame | None:
        if not frame_path.exists():
            return None
        with frame_path.open("r", encoding="utf-8") as handle:
            try:
                payload = json.load(handle)
            except json.JSONDecodeError:
                # Truncated/empty file, most likely a write interrupted mid-flight
                # (save_frame isn't atomic) -- treat like a missing frame rather than
                # taking down the whole load_all_frames/load_frames request.
                log.warning("Skipping corrupt landmark file %s (invalid JSON)", frame_path)
                return None
        return LandmarkFrame(**payload)

    def load_frames(self, actor_id: str, sequence_id: str, camera_id: str, frame_indices: Iterable[int]) -> List[LandmarkFrame]:
        frames: List[LandmarkFrame] = []
        for idx in frame_indices:
            frame_path = self._frame_path(actor_id, sequence_id, camera_id, idx)
            frame = self._read_frame(frame_path)
            if frame is None:
                continue
            frames.append(self._merge_propagated_groups(actor_id, sequence_id, camera_id, frame))
        return frames

    def list_available_frame_indices(self, actor_id: str, sequence_id: str, camera_id: str) -> List[int]:
        """Return the sorted frame indices that actually have a frame_{idx:06d}.json
        file on disk, derived purely from file names (no JSON parsing) — this is the
        ground truth for which frames are navigable, since actual sparsity (whether
        from a deliberate frame_stride subsample, or unrelated per-frame extraction
        gaps) doesn't always land on tidy stride multiples."""
        camera_dir = self._camera_dir(actor_id, sequence_id, camera_id)
        if not camera_dir.exists():
            return []
        indices: List[int] = []
        for frame_path in camera_dir.glob("frame_*.json"):
            try:
                indices.append(int(frame_path.stem.removeprefix("frame_")))
            except ValueError:
                continue
        return sorted(indices)

    def load_all_frames(self, actor_id: str, sequence_id: str, camera_id: str) -> List[LandmarkFrame]:
        """Every frame_<idx>.json for this camera, each merged with its
        propagated group files (same result as _merge_propagated_groups()),
        read in parallel -- see _READ_POOL."""
        camera_dir = self._camera_dir(actor_id, sequence_id, camera_id)
        names = _list_names(camera_dir)
        frame_names = sorted(n for n in names if n.startswith("frame_") and n.endswith(".json"))
        if not frame_names:
            return []

        # (frame file, [group files that actually exist for it]) per frame.
        plan: List[tuple[str, List[str]]] = []
        for frame_name in frame_names:
            suffix = frame_name[len("frame_"):]
            plan.append((frame_name, [f"{prefix}_frame_{suffix}" for prefix in STANDARD_PROPAGATED_GROUPS
                                      if f"{prefix}_frame_{suffix}" in names]))
        to_read = [camera_dir / n for frame_name, groups in plan for n in (frame_name, *groups)]
        payloads = dict(zip((p.name for p in to_read), _read_many(to_read)))

        frames: List[LandmarkFrame] = []
        for frame_name, groups in plan:
            payload = payloads.get(frame_name)
            if payload is None:
                continue
            frame = LandmarkFrame(**payload)
            overrides: dict[int, LandmarkPoint] = {}
            for group_name in groups:
                for raw_point in (payloads.get(group_name) or {}).get("points", []):
                    overrides[raw_point["index"]] = LandmarkPoint(**raw_point)
            if overrides:
                merged_points = [p for p in frame.points if p.index not in overrides] + list(overrides.values())
                frame = LandmarkFrame(frame_idx=frame.frame_idx, points=merged_points)
            frames.append(frame)
        return frames

    def _group_frame_path(self, actor_id: str, sequence_id: str, camera_id: str, file_prefix: str, frame_idx: int) -> Path:
        return self._camera_dir(actor_id, sequence_id, camera_id) / f"{file_prefix}_frame_{frame_idx:06d}.json"

    def _load_group_points(self, actor_id: str, sequence_id: str, camera_id: str, file_prefix: str, frame_idx: int) -> list[dict]:
        path = self._group_frame_path(actor_id, sequence_id, camera_id, file_prefix, frame_idx)
        if not path.exists():
            return []
        with path.open("r", encoding="utf-8") as handle:
            try:
                payload = json.load(handle)
            except json.JSONDecodeError:
                log.warning("Skipping corrupt landmark file %s (invalid JSON)", path)
                return []
        return payload.get("points", [])

    def _merge_propagated_groups(self, actor_id: str, sequence_id: str, camera_id: str, frame: LandmarkFrame) -> LandmarkFrame:
        """Overlays STANDARD_PROPAGATED_GROUPS' propagated points on top of
        `frame`'s own sapiens2-direct-detected points at the same indices.
        A frame with no propagated data yet (none of the group files exist
        for this frame index) is returned unchanged."""
        overrides: dict[int, LandmarkPoint] = {}
        for file_prefix in STANDARD_PROPAGATED_GROUPS:
            for raw_point in self._load_group_points(actor_id, sequence_id, camera_id, file_prefix, frame.frame_idx):
                overrides[raw_point["index"]] = LandmarkPoint(**raw_point)
        if not overrides:
            return frame
        merged_points = [p for p in frame.points if p.index not in overrides] + list(overrides.values())
        return LandmarkFrame(frame_idx=frame.frame_idx, points=merged_points)

    def load_all_skin_frames(self, actor_id: str, sequence_id: str, camera_id: str) -> List[LandmarkFrame]:
        """Same shape as load_all_frames, but for the optional "skin" landmarks
        (index >= 74) that run_neutral_cotracker_propagation.py writes to
        skin_frame_<idx:06d>.json alongside the standard frame_<idx:06d>.json files.
        Returns an empty list whenever that pair hasn't had skin points propagated
        yet (no special-casing needed — the glob below just finds nothing)."""
        camera_dir = self._camera_dir(actor_id, sequence_id, camera_id)
        paths = [camera_dir / n for n in sorted(_list_names(camera_dir))
                 if n.startswith("skin_frame_") and n.endswith(".json")]
        return [LandmarkFrame(**payload) for payload in _read_many(paths) if payload is not None]

    def save_frame(self, payload: LandmarkPayload) -> Path:
        frame_path = self._frame_path(payload.actor_id, payload.sequence_id, payload.camera_id, payload.frame_idx)
        # Write to a sibling temp file and rename into place -- rename is atomic, so a
        # kill/crash/NFS hiccup mid-write can no longer leave a truncated frame_*.json
        # that then breaks every subsequent load_all_frames/load_frames call for this camera.
        # The temp filename includes a uuid (not just frame_path's own name) so two
        # concurrent writers to the same frame_path (e.g. overlapping propagation jobs
        # re-touching the same frame) never share one temp file -- with a shared name,
        # one writer's replace() can consume the other's temp file out from under it,
        # raising a spurious FileNotFoundError on the second replace() call.
        tmp_path = frame_path.parent / f"{frame_path.name}.{uuid.uuid4().hex}.tmp"
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload.dict(), handle)
        tmp_path.replace(frame_path)
        return frame_path

    def _neutral_frame_path(self, actor_id: str, sequence_id: str) -> Path:
        """Legacy per-sequence location -- kept write-only (see save_neutral_frame)
        for run_pipeline_batch_all.py/run_pipeline_batch.py/rewrap_missing_frames.py/
        etc., which still read this directly for whichever sequence physically has
        a wrapped neutral mesh, to chain their own non-neutral wraps off of it."""
        sequence_dir = self.settings.landmarks_root / actor_id / sequence_id
        sequence_dir.mkdir(parents=True, exist_ok=True)
        return sequence_dir / "neutral_frame.json"

    def _actor_neutral_frame_path(self, actor_id: str) -> Path:
        """Canonical location (one neutral frame per actor, not per sequence) --
        see run_neutral_cotracker_propagation.py's load_actor_neutral_frame. Holds
        {"sequence_id": ..., "frame_idx": ...}: which sequence the neutral frame
        was actually picked from, and its raw frame index within that sequence."""
        actor_dir = self.settings.landmarks_root / actor_id
        actor_dir.mkdir(parents=True, exist_ok=True)
        return actor_dir / "neutral_frame.json"

    def save_neutral_frame(self, actor_id: str, sequence_id: str, displayed_frame_idx: int) -> int:
        """Save this actor's ONE neutral (reference) frame, converting the
        displayed (delegation-video) frame index to the raw frame index used
        by the wrapping pipeline. Reads the file back to confirm the write
        actually landed before reporting success, since neutral_frame.json
        lives on the same scratch/NFS storage as the rest of the pipeline.

        Overwrites whatever sequence/frame this actor's neutral previously
        pointed to -- there is only ever one per actor now, propagated into
        every sequence the actor has (see run_neutral_cotracker_propagation.py's
        cross-sequence propagation), not one per actor/sequence pair."""
        raw_frame_idx = displayed_frame_idx * RAW_FRAME_STRIDE

        actor_frame_path = self._actor_neutral_frame_path(actor_id)
        with actor_frame_path.open("w", encoding="utf-8") as handle:
            json.dump({"sequence_id": sequence_id, "frame_idx": raw_frame_idx}, handle)
        with actor_frame_path.open("r", encoding="utf-8") as handle:
            saved = json.load(handle)
        if saved.get("frame_idx") != raw_frame_idx or saved.get("sequence_id") != sequence_id:
            raise IOError(
                f"Neutral frame verification failed for actor {actor_id}: "
                f"wrote sequence_id={sequence_id!r} frame_idx={raw_frame_idx}, "
                f"read back {saved!r} from {actor_frame_path}"
            )

        # Legacy per-sequence mirror -- see _neutral_frame_path's own docstring.
        legacy_frame_path = self._neutral_frame_path(actor_id, sequence_id)
        with legacy_frame_path.open("w", encoding="utf-8") as handle:
            json.dump({"frame_idx": raw_frame_idx}, handle)

        # A (re)saved actor-level neutral frame invalidates every sequence's
        # prior propagated skin/ear landmarks (they were all seeded from
        # whichever neutral frame used to be current), not just the sequence
        # it was (re)saved from -- reset review status back down to
        # "unlabeled" for every sequence this actor has.
        actor_dir = self.settings.landmarks_root / actor_id
        if actor_dir.is_dir():
            for other_sequence_dir in sorted(actor_dir.iterdir()):
                if other_sequence_dir.is_dir() and other_sequence_dir.name not in BLACKLISTED_SEQUENCES:
                    label_tracker.mark_unlabeled(actor_id, other_sequence_dir.name)
        else:
            label_tracker.mark_unlabeled(actor_id, sequence_id)

        return raw_frame_idx

    def load_actor_neutral_frame(self, actor_id: str) -> tuple[str, int] | None:
        """Return (sequence_id, displayed frame index) of this actor's ONE
        neutral frame, or None if it hasn't been saved yet -- actor-scoped
        (see save_neutral_frame), so this is the same regardless of which
        sequence the caller is currently viewing."""
        actor_frame_path = self._actor_neutral_frame_path(actor_id)
        if not actor_frame_path.exists():
            return None
        with actor_frame_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        sequence_id = payload.get("sequence_id")
        raw_frame_idx = payload.get("frame_idx")
        if sequence_id is None or raw_frame_idx is None:
            return None
        return sequence_id, raw_frame_idx // RAW_FRAME_STRIDE


landmark_store = LandmarkStore()
