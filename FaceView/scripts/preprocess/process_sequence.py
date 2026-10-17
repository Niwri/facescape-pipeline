"""Generate delegation videos and initial landmark dumps for a sequence."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Iterator, List, Tuple

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from backend.app.core.config import get_settings  # noqa: E402


def _run_ffmpeg(input_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-vf",
        "scale=550:802",
        "-crf",
        "23",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)


def _load_face_indices(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _create_landmarker(model_path: Path) -> vision.FaceLandmarker:
    base_options = python.BaseOptions(model_asset_path=str(model_path))
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1,
    )
    return vision.FaceLandmarker.create_from_options(options)


def _write_frame(payload: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload), encoding="utf-8")


def _extract_landmarks(video_path: Path, actor: str, sequence: str, camera: str, indices: dict) -> None:
    settings = get_settings()
    landmarker = _create_landmarker(Path("face_landmarker_v2_with_blendshapes.task"))
    capture = cv2.VideoCapture(str(video_path))
    frame_idx = 0
    success, frame = capture.read()
    while success:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        detection = landmarker.detect(mp_image)
        if detection.face_landmarks:
            face_landmarks = detection.face_landmarks[0]
            filtered_points = []
            for region, region_indices in indices.items():
                for idx in region_indices:
                    if idx >= len(face_landmarks):
                        continue
                    landmark = face_landmarks[idx]
                    filtered_points.append(
                        {
                            "index": idx,
                            "x": landmark.x,
                            "y": landmark.y,
                            "z": landmark.z,
                            "visibility": landmark.presence if hasattr(landmark, "presence") else None,
                            "region": region,
                        }
                    )
            eye_regions = [385, 387, 160, 158]
            for idx in eye_regions:
                landmark = face_landmarks[idx]
                filtered_points.append(
                    {
                        "index": idx,
                        "x": landmark.x,
                        "y": landmark.y,
                        "z": landmark.z,
                        "visibility": landmark.presence if hasattr(landmark, "presence") else None,
                        "region": "eye",
                    }
                )
            payload = {
                "frame_idx": frame_idx,
                "actor_id": actor,
                "sequence_id": sequence,
                "camera_id": camera,
                "points": filtered_points,
            }
            frame_path = settings.landmarks_root / actor / sequence / camera / f"frame_{frame_idx:06d}.json"
            _write_frame(payload, frame_path)
        frame_idx += 1
        success, frame = capture.read()
    capture.release()


def _iter_cameras(sequence_dir: Path, cameras: Iterable[str] | None = None) -> List[tuple[str, Path]]:
    images_dir = sequence_dir / "images"
    if not images_dir.exists():
        return []
    selected = set(cameras or [])
    result = []
    for video_file in sorted(images_dir.glob("*.mp4")):
        cam_id = video_file.stem
        if selected and cam_id not in selected:
            continue
        result.append((cam_id, video_file))
    return result


def _iter_targets(settings, actor_filter: str | None, sequence_filter: str | None) -> Iterator[Tuple[str, str, Path]]:
    if actor_filter:
        actor_dirs = [settings.data_root / actor_filter]
    else:
        actor_dirs = [p for p in sorted(settings.data_root.glob("*/")) if p.is_dir()]
    for actor_dir in actor_dirs:
        if not actor_dir.exists():
            continue
        sequences_root = actor_dir / "sequences"
        if not sequences_root.exists():
            continue
        if sequence_filter:
            candidate_dirs = [sequences_root / sequence_filter]
        else:
            candidate_dirs = [p for p in sorted(sequences_root.glob("*/")) if p.is_dir()]
        for seq_dir in candidate_dirs:
            if not seq_dir.exists():
                continue
            yield actor_dir.name, seq_dir.name, seq_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("actor_id", nargs="?", default=None, help="Actor identifier; omit to process all actors")
    parser.add_argument("sequence_id", nargs="?", default=None, help="Sequence identifier; omit to process all sequences")
    parser.add_argument("--cameras", nargs="*", default=None, help="Optional list of camera ids to process")
    args = parser.parse_args()

    settings = get_settings()
    targets = list(_iter_targets(settings, args.actor_id, args.sequence_id))
    if not targets:
        raise SystemExit("No sequences found matching the provided filters")

    face_indices = _load_face_indices(settings.configs_root / "face_indices.json")

    for actor_id, sequence_id, sequence_dir in targets:
        print(f"Processing actor={actor_id} sequence={sequence_id}")
        for camera_id, video_path in _iter_cameras(sequence_dir, args.cameras):
            delegation_video = settings.processed_root / actor_id / sequence_id / f"{camera_id}_delegation.mp4"
            _run_ffmpeg(video_path, delegation_video)
            _extract_landmarks(delegation_video, actor_id, sequence_id, camera_id, face_indices)
            print(f"  Completed camera {camera_id} -> {delegation_video}")


if __name__ == "__main__":
    main()
