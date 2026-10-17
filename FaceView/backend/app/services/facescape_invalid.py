"""FaceScape "invalid" pairs (settings.facescape_invalid_pairs_path): pairs whose
cameras could not be aligned to the canonical origin rig. They have no
origin-space version, so:
  - their images/params are read from the raw root
    (settings.facescape_raw_images_root),
  - their camera ids are that pair's own raw indices (its params.json), not
    the canonical facescape_cameras.json list,
  - their wrapped mesh is already saved in raw space by
    wrap_script_facescape/run_pipeline_invalid.py, so exporting it on confirm
    must not apply the origin -> raw transform.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import FrozenSet, List, Tuple

from ..core.config import get_settings

_lock = threading.Lock()
_cache: tuple[float, FrozenSet[Tuple[str, str]]] | None = None  # (file mtime, pairs)


def invalid_pairs() -> FrozenSet[Tuple[str, str]]:
    """Cached on the file's mtime, so an updated invalid_pairs.json is picked up."""
    global _cache
    path = get_settings().facescape_invalid_pairs_path
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return frozenset()
    with _lock:
        if _cache is not None and _cache[0] == mtime:
            return _cache[1]
    try:
        with path.open("r", encoding="utf-8") as f:
            pairs = frozenset((str(p["actor"]), str(p["expression"])) for p in json.load(f).get("pairs", []))
    except (OSError, json.JSONDecodeError):
        pairs = frozenset()
    with _lock:
        _cache = (mtime, pairs)
    return pairs


def is_invalid(actor_id: str, expression_id: str) -> bool:
    return (actor_id, expression_id) in invalid_pairs()


def images_root_for(actor_id: str, expression_id: str) -> Path:
    settings = get_settings()
    return settings.facescape_raw_images_root if is_invalid(actor_id, expression_id) else settings.facescape_images_root


def pair_camera_ids(actor_id: str, expression_id: str) -> List[str]:
    """The pair's own camera ids (params.json cameras not flagged invalid)."""
    params_path = images_root_for(actor_id, expression_id) / actor_id / expression_id / "params.json"
    try:
        with params_path.open("r", encoding="utf-8") as f:
            params = json.load(f)
    except (OSError, json.JSONDecodeError):
        return []
    ids = {k.split("_")[0] for k in params if k.endswith("_Rt")}
    return sorted((i for i in ids if params.get(f"{i}_valid", True) is not False), key=int)


def overlay_camera_ids(actor_id: str, expression_id: str) -> List[str]:
    """The three geometric overlay cameras run_pipeline_invalid.py rendered."""
    path = get_settings().facescape_output_root / actor_id / expression_id / "overlay_cameras.json"
    try:
        with path.open("r", encoding="utf-8") as f:
            return [str(c) for c in json.load(f)]
    except (OSError, json.JSONDecodeError):
        return []
