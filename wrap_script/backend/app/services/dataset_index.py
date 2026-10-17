"""Dataset discovery helpers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

from ..core.config import get_settings
from ..schemas.datasets import ActorDescriptor, CameraDescriptor, DatasetResponse, SequenceDescriptor
from ..schemas.common import MediaPath


def _load_face_indices(configs_root: Path) -> dict:
    face_indices_path = configs_root / "face_indices.json"
    if face_indices_path.exists():
        with face_indices_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    return {}


def list_dataset() -> DatasetResponse:
    settings = get_settings()
    actors: List[ActorDescriptor] = []
    for actor_dir in sorted(settings.data_root.glob("*/")):
        actor_id = actor_dir.name
        sequences: List[SequenceDescriptor] = []
        for sequence_dir in sorted((actor_dir / "sequences").glob("*/")):
            sequence_id = sequence_dir.name
            cameras: List[CameraDescriptor] = []
            images_dir = sequence_dir / "images"
            if not images_dir.exists():
                continue
            for video_path in sorted(images_dir.glob("*.mp4")):
                camera_id = video_path.stem
                processed_video = settings.processed_root / actor_id / sequence_id / f"{camera_id}_delegation.mp4"
                landmarks_dir = settings.landmarks_root / actor_id / sequence_id / camera_id
                descriptor = CameraDescriptor(
                    camera_id=camera_id,
                    video=MediaPath(actor_id=actor_id, sequence_id=sequence_id, camera_id=camera_id, video_path=video_path),
                    status="ready" if processed_video.exists() and landmarks_dir.exists() else "pending",
                )
                cameras.append(descriptor)
            sequences.append(
                SequenceDescriptor(
                    sequence_id=sequence_id,
                    cameras=cameras,
                    has_landmarks=any(settings.landmarks_root.joinpath(actor_id, sequence_id, cam.camera_id).exists() for cam in cameras),
                    has_delegation_video=any(
                        settings.processed_root.joinpath(actor_id, sequence_id, f"{cam.camera_id}_delegation.mp4").exists()
                        for cam in cameras
                    ),
                )
            )
        actors.append(ActorDescriptor(actor_id=actor_id, sequences=sequences))
    return DatasetResponse(actors=actors, face_indices=_load_face_indices(settings.configs_root))
