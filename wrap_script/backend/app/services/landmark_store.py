"""File-based persistence for landmarks."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

from ..core.config import get_settings
from ..schemas.common import LandmarkFrame
from ..schemas.landmarks import LandmarkPayload


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
            payload = json.load(handle)
        return LandmarkFrame(**payload)

    def load_frames(self, actor_id: str, sequence_id: str, camera_id: str, frame_indices: Iterable[int]) -> List[LandmarkFrame]:
        frames: List[LandmarkFrame] = []
        for idx in frame_indices:
            frame_path = self._frame_path(actor_id, sequence_id, camera_id, idx)
            frame = self._read_frame(frame_path)
            if frame is None:
                continue
            frames.append(frame)
        return frames

    def load_all_frames(self, actor_id: str, sequence_id: str, camera_id: str) -> List[LandmarkFrame]:
        camera_dir = self._camera_dir(actor_id, sequence_id, camera_id)
        if not camera_dir.exists():
            return []
        frames: List[LandmarkFrame] = []
        for frame_path in sorted(camera_dir.glob("frame_*.json")):
            frame = self._read_frame(frame_path)
            if frame is None:
                continue
            frames.append(frame)
        return frames

    def save_frame(self, payload: LandmarkPayload) -> Path:
        frame_path = self._frame_path(payload.actor_id, payload.sequence_id, payload.camera_id, payload.frame_idx)
        with frame_path.open("w", encoding="utf-8") as handle:
            json.dump(payload.dict(), handle)
        return frame_path


landmark_store = LandmarkStore()
