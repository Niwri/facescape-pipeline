"""Ava-256 capture review: per-frame labels and the Done export.

Each capture is reviewed as one keyline video rendered by
wrap_script_ava256/render_capture_line_video.py into line_renders_ava256_root:
<capture>.mp4 plus <capture>.csv (FRAME_ID,STATUS, one row per video frame in
video order). The Ava-256 tab labels frame ranges Valid / Invalid / Rerun in
that CSV. Done then:
  1. exports the Valid frames' wraps, back in Ava-256 world space (undoing the
     capture's dynamic_transform_params.npz, as the overlay renders do), to
     final_wrapped_data_root/ava256_<capture>.pt,
  2. moves the CSV to ava256_labels_root/<capture>.csv (read-only from then on),
  3. sets label_tracker.json's Ava256[<capture>] to "confirmed".

label_tracker.json's "Ava256" key is FLAT ({capture: status}), unlike the
other datasets' {actor: {sequence: status}} -- see
wrap_script_ava256/label_tracker_ava256.py, which writes the same key with the
same sibling-.lock-file + atomic-replace pattern as label_tracker.py.
"""
from __future__ import annotations

import csv
import fcntl
import json
import logging
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import numpy as np
import torch

from ..core.config import get_settings
from .wrapped_mesh_export import _load_obj_vertices_faces

log = logging.getLogger(__name__)

TRACKER_KEY = "Ava256"
DEFAULT_CAPTURE_STATUS = "unlabeled"
UNCONFIRMED = "unconfirmed"
FRAME_LABELS = ("Valid", "Invalid", "Rerun")
CSV_HEADER = ["FRAME_ID", "STATUS"]


class Ava256Error(Exception):
    """Carries an HTTP status code for the route to return."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


# --- label_tracker.json ------------------------------------------------------

def _load_tracker() -> dict:
    path = get_settings().label_tracker_path
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}


def capture_statuses() -> Dict[str, str]:
    return _load_tracker().get(TRACKER_KEY, {})


def set_capture_status(capture_id: str, status: str) -> None:
    path = get_settings().label_tracker_path
    lock_path = path.with_name(path.name + ".lock")
    with lock_path.open("w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            data = _load_tracker()
            data.setdefault(TRACKER_KEY, {})[capture_id] = status
            tmp_path = path.with_name(path.name + ".tmp")
            tmp_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            tmp_path.replace(path)
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


# --- per-frame CSV -----------------------------------------------------------

def video_path(capture_id: str) -> Path:
    return get_settings().line_renders_ava256_root / f"{capture_id}.mp4"


def _review_csv_path(capture_id: str) -> Path:
    return get_settings().line_renders_ava256_root / f"{capture_id}.csv"


def _archived_csv_path(capture_id: str) -> Path:
    return get_settings().ava256_labels_root / f"{capture_id}.csv"


def is_done(capture_id: str) -> bool:
    return _archived_csv_path(capture_id).exists()


def _csv_path(capture_id: str) -> Path | None:
    for path in (_archived_csv_path(capture_id), _review_csv_path(capture_id)):
        if path.exists():
            return path
    return None


def read_frames(capture_id: str) -> List[Tuple[str, str]]:
    path = _csv_path(capture_id)
    if path is None:
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [(row["FRAME_ID"], row["STATUS"]) for row in csv.DictReader(handle)]


def read_skin_errors(capture_id: str) -> Dict[str, float]:
    """frame_id -> mean skin-landmark pixel error, from
    wrap_script_ava256/check_ava256_skin_landmark_errors.py's
    line_renders_ava256_root/skin_errors/<capture>.json ({} if not run yet)."""
    path = get_settings().line_renders_ava256_root / "skin_errors" / f"{capture_id}.json"
    try:
        frames = json.loads(path.read_text(encoding="utf-8")).get("frames", {})
    except (OSError, json.JSONDecodeError):
        return {}
    return {frame_id: float(entry["px"]) for frame_id, entry in frames.items() if "px" in entry}


def _write_frames(path: Path, rows: List[Tuple[str, str]]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_HEADER)
        writer.writerows(rows)
    tmp.replace(path)


@contextmanager
def _capture_lock(capture_id: str) -> Iterator[None]:
    lock_dir = get_settings().line_renders_ava256_root / ".locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    with (lock_dir / f"{capture_id}.lock").open("w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def apply_ranges(capture_id: str, ranges_by_label: Dict[str, List[List[int]]]) -> Tuple[List[Tuple[str, str]], List[str]]:
    """Sets every frame in each label's inclusive, 1-based ranges (positions in
    the video/CSV) to that label; frames not covered keep their status.
    Rejects unknown labels and ranges that overlap across or within labels.
    Returns (rows, warnings) -- warnings note ranges clamped to the video."""
    unknown = set(ranges_by_label) - set(FRAME_LABELS)
    if unknown:
        raise Ava256Error(400, f"Unknown label(s) {sorted(unknown)}; expected {list(FRAME_LABELS)}")
    if is_done(capture_id):
        raise Ava256Error(409, f"{capture_id} is already done; its labels are archived and read-only")

    with _capture_lock(capture_id):
        path = _review_csv_path(capture_id)
        if not path.exists():
            raise Ava256Error(404, f"No frame CSV for {capture_id} at {path}")
        rows = read_frames(capture_id)
        n = len(rows)
        warnings: List[str] = []
        owner: Dict[int, str] = {}
        for label, ranges in ranges_by_label.items():
            for pair in ranges:
                if len(pair) != 2:
                    raise Ava256Error(400, f"{label}: each range must be [start, end], got {pair}")
                start, end = sorted(int(v) for v in pair)
                if end < 1 or start > n:
                    warnings.append(f"{label} {start}-{end} is outside 1-{n}; ignored")
                    continue
                if start < 1 or end > n:
                    warnings.append(f"{label} {start}-{end} clamped to {max(start, 1)}-{min(end, n)}")
                for position in range(max(start, 1), min(end, n) + 1):
                    if position in owner:
                        other = owner[position]
                        raise Ava256Error(
                            400,
                            f"Frame #{position} is in more than one range ({other} and {label})"
                            if other != label else f"{label} ranges overlap at frame #{position}",
                        )
                    owner[position] = label
        rows = [(frame_id, owner.get(i + 1, status)) for i, (frame_id, status) in enumerate(rows)]
        _write_frames(path, rows)
    return rows, warnings


# --- Done ----------------------------------------------------------------------

def _load_transform_params(capture_id: str) -> dict:
    path = get_settings().ava256_wraps_root / capture_id / "dynamic_transform_params.npz"
    if not path.exists():
        raise Ava256Error(409, f"No dynamic_transform_params.npz for {capture_id} ({path}) -- can't return wraps to world space")
    data = np.load(path)
    return {"scale": float(data["scale"]), "scan_center": data["scan_center"], "neutral_center": data["neutral_center"]}


def _untransform(verts: np.ndarray, params: dict) -> np.ndarray:
    """Same as wrap_script_ava256/dynamic_transform.untransform: template frame -> world."""
    return (verts.astype(np.float64) - params["neutral_center"]) / params["scale"] + params["scan_center"]


def export_path(capture_id: str) -> Path:
    return get_settings().final_wrapped_data_root / f"ava256_{capture_id}.pt"


def finish_capture(capture_id: str) -> dict:
    """The Done button. Refuses while any frame is still unconfirmed."""
    settings = get_settings()
    if is_done(capture_id):
        raise Ava256Error(409, f"{capture_id} is already done")

    with _capture_lock(capture_id):
        review_csv = _review_csv_path(capture_id)
        if not review_csv.exists():
            raise Ava256Error(404, f"No frame CSV for {capture_id} at {review_csv}")
        rows = read_frames(capture_id)
        pending = sum(1 for _, status in rows if status not in FRAME_LABELS)
        if pending:
            raise Ava256Error(409, f"{pending} frame(s) are still unconfirmed -- label every frame before Done")

        valid_ids = [frame_id for frame_id, status in rows if status == "Valid"]
        exported: Path | None = None
        if valid_ids:
            params = _load_transform_params(capture_id)
            faces: np.ndarray | None = None
            verts_by_frame: List[np.ndarray] = []
            for frame_id in valid_ids:
                obj_path = settings.ava256_wraps_root / capture_id / frame_id / "wrapped_mesh.obj"
                try:
                    verts, frame_faces = _load_obj_vertices_faces(obj_path)
                except (OSError, ValueError) as exc:
                    raise Ava256Error(500, f"Cannot read {obj_path}: {exc}") from exc
                if faces is None:
                    faces = frame_faces
                elif not np.array_equal(faces, frame_faces):
                    raise Ava256Error(500, f"{obj_path} has a different face topology than frame {valid_ids[0]}")
                verts_by_frame.append(_untransform(verts, params).astype(np.float32))

            exported = export_path(capture_id)
            exported.parent.mkdir(parents=True, exist_ok=True)
            tmp = exported.with_name(exported.name + ".tmp")
            torch.save(
                {
                    "faces": torch.from_numpy(faces),  # (F, 3) int64
                    "verts": torch.from_numpy(np.stack(verts_by_frame, axis=0)),  # (N, V, 3) float32, world space
                    "frame_ids": torch.tensor([int(f) for f in valid_ids], dtype=torch.int64),  # (N,), same order as verts
                    "frame_id_strings": list(valid_ids),  # zero-padded, as on disk
                },
                tmp,
            )
            tmp.replace(exported)
            log.info("Ava-256 %s: exported %d valid frame(s) to %s", capture_id, len(valid_ids), exported)

        archived = _archived_csv_path(capture_id)
        archived.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(review_csv), str(archived))

    set_capture_status(capture_id, "confirmed")
    counts = {label: sum(1 for _, s in rows if s == label) for label in FRAME_LABELS}
    return {"export_path": str(exported) if exported else None, "labels_path": str(archived), "counts": counts}


# --- listing -------------------------------------------------------------------

def list_captures() -> List[dict]:
    """Every capture in ava256_wraps_root (plus any with only a render or archived
    CSV), with its tracker status, video/CSV presence and per-label counts."""
    settings = get_settings()
    ids = set()
    if settings.ava256_wraps_root.is_dir():
        ids |= {p.name for p in settings.ava256_wraps_root.iterdir() if p.is_dir() and "--" in p.name}
    for root, suffix in ((settings.line_renders_ava256_root, ".mp4"), (settings.ava256_labels_root, ".csv")):
        if root.is_dir():
            ids |= {p.stem for p in root.glob(f"*{suffix}") if "--" in p.stem and not p.stem.endswith("_tmp")}

    statuses = capture_statuses()
    captures = []
    for capture_id in sorted(ids):
        rows = read_frames(capture_id)
        counts = {label: 0 for label in (*FRAME_LABELS, UNCONFIRMED)}
        for _, status in rows:
            counts[status if status in counts else UNCONFIRMED] += 1
        captures.append({
            "capture_id": capture_id,
            "label_status": statuses.get(capture_id, DEFAULT_CAPTURE_STATUS),
            "has_video": video_path(capture_id).exists(),
            "num_frames": len(rows),
            "counts": counts,
            "is_done": is_done(capture_id),
        })
    return captures
