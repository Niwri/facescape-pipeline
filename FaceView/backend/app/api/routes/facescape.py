"""FaceScape dataset endpoints -- static multiview scans, no video. Serves
per-camera JPGs and their sapiens2 landmark overlays for the frontend's
read-only "Cameras" tab; "Overlay Frame" (FaceScape's analog of Nersemble's
"Overlay Video") is plumbing-only until the wrap pipeline produces
wrapped_mesh.obj for FaceScape."""
from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from ...core.config import get_settings
from ...schemas.common import LandmarkFrame
from ...schemas.facescape import (
    FaceScapeDatasetResponse,
    NeutralExpressionQueryResponse,
    NeutralExpressionResponse,
    SaveEarPointRequest,
    SaveEarPointResponse,
    SaveNeutralExpressionRequest,
)
from ...services import facescape_ear_points, facescape_invalid, facescape_neutral
from ...services.facescape_dataset_index import list_facescape_dataset

router = APIRouter(prefix="/facescape", tags=["facescape"])

# The other standard (0-73) index groups wrap_script_facescape/run_neutral_skin_propagation_raft.py
# propagates into their own <prefix>_frame_000000.json files, same reasoning
# as ear_frame_000000.json (frame_000000.json's own sapiens2-detected values
# at these indices are unreliable/absent -- see that script's own
# FACESCAPE_STANDARD_INDEX_REGION_GROUPS). Ear is handled separately below via
# facescape_ear_points.load_ear_points (its own triangulate-and-save flow),
# so it's not repeated here.
PROPAGATED_GROUP_PREFIXES = ["nosebridge", "undernose", "eyebrow", "chin"]


def _load_group_points(actor_id: str, expression_id: str, camera_id: str, file_prefix: str) -> list[dict]:
    settings = get_settings()
    path = (
        settings.facescape_landmarks_root / actor_id / expression_id
        / f"cam_{camera_id}" / f"{file_prefix}_frame_000000.json"
    )
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        try:
            payload = json.load(handle)
        except json.JSONDecodeError:
            return []
    return payload.get("points", [])


@router.get("/dataset", response_model=FaceScapeDatasetResponse)
def get_facescape_dataset() -> FaceScapeDatasetResponse:
    return list_facescape_dataset()


@router.get("/image/{actor_id}/{expression_id}/{camera_id}")
def get_facescape_image(actor_id: str, expression_id: str, camera_id: str) -> FileResponse:
    # Invalid pairs are served from their raw images (no origin-space version).
    image_path = facescape_invalid.images_root_for(actor_id, expression_id) / actor_id / expression_id / f"{camera_id}.jpg"
    if not image_path.exists():
        raise HTTPException(status_code=404, detail=f"No image found at {image_path}")
    # This exact URL can point at different file content over time -- the
    # Phase 1 coordinate-transform pipeline overwrites <camera_id>.jpg in place
    # (same path) when reprocessed. Without this, browsers cache the response
    # by URL alone and can keep serving stale image bytes indefinitely after a
    # reprocess. "no-cache" (not "no-store") still lets the browser reuse the
    # cached bytes once it revalidates via a conditional request, so a rerun
    # with no actual change is still a cheap 304, not a full re-download.
    return FileResponse(image_path, media_type="image/jpeg", headers={"Cache-Control": "no-cache"})


@router.get("/landmarks/skin/{actor_id}/{expression_id}/{camera_id}", response_model=LandmarkFrame)
def get_facescape_skin_landmarks(actor_id: str, expression_id: str, camera_id: str) -> LandmarkFrame:
    """The optional "skin" landmarks (index >= 74) run_neutral_skin_propagation.py
    writes alongside the standard ones. Same graceful-empty convention as
    Nersemble's GET /landmarks/skin: returns an empty points list (not a 404)
    for any actor/expression that hasn't had skin points propagated yet, since
    that's the normal, expected state for most pairs, not an error.

    Registered before /landmarks/{actor_id}/... so "skin" isn't swallowed as
    a path param by the more general route (FastAPI matches in registration
    order)."""
    settings = get_settings()
    frame_path = settings.facescape_landmarks_root / actor_id / expression_id / f"cam_{camera_id}" / "skin_frame_000000.json"
    if not frame_path.exists():
        return LandmarkFrame(frame_idx=0, points=[])
    with frame_path.open("r", encoding="utf-8") as handle:
        try:
            payload = json.load(handle)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=500, detail=f"Corrupt landmark file {frame_path}") from exc
    return LandmarkFrame(**payload)


@router.post("/landmarks/ear", response_model=SaveEarPointResponse)
def save_facescape_ear_point(request: SaveEarPointRequest) -> SaveEarPointResponse:
    """Triangulates one ear-top point (index 2 = left, index 9 = right) from
    the given per-camera clicks, projects it into every camera for this
    actor/expression pair (writing cam_<id>/ear_frame_000000.json -- see
    facescape_ear_points.py). Deliberately triangulate-only -- does NOT
    propagate to any other expression (facescape_ear_propagation.py is no
    longer called from here, per explicit instruction); a reviewer adding ear
    landmarks for one expression only ever affects that expression's own
    cameras. Also deliberately not gated on label_status -- a reviewer can
    add ear points regardless of the pair's current review state."""
    try:
        cameras_written, _world_xyz = facescape_ear_points.save_ear_point(
            request.actor_id, request.expression_id, request.ear, request.clicks,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return SaveEarPointResponse(cameras_written=cameras_written)


@router.post("/neutral_expression", response_model=NeutralExpressionResponse)
def set_facescape_neutral_expression(request: SaveNeutralExpressionRequest) -> NeutralExpressionResponse:
    try:
        facescape_neutral.save_actor_neutral_expression(request.actor_id, request.expression_id)
    except OSError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return NeutralExpressionResponse(expression_id=request.expression_id)


@router.get("/neutral_expression", response_model=NeutralExpressionQueryResponse)
def get_facescape_neutral_expression(actor_id: str) -> NeutralExpressionQueryResponse:
    return NeutralExpressionQueryResponse(expression_id=facescape_neutral.load_actor_neutral_expression(actor_id))


@router.get("/landmarks/{actor_id}/{expression_id}/{camera_id}", response_model=LandmarkFrame)
def get_facescape_landmarks(actor_id: str, expression_id: str, camera_id: str) -> LandmarkFrame:
    settings = get_settings()
    frame_path = settings.facescape_landmarks_root / actor_id / expression_id / f"cam_{camera_id}" / "frame_000000.json"
    if not frame_path.exists():
        raise HTTPException(status_code=404, detail=f"No landmarks found at {frame_path}")
    with frame_path.open("r", encoding="utf-8") as handle:
        try:
            payload = json.load(handle)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=500, detail=f"Corrupt landmark file {frame_path}") from exc

    # Overlay any manually-saved ear point(s) on top of whatever frame_000000.json
    # itself has at indices 2/9 (today, always a garbage (-1,-1,-1) sentinel from
    # process_expression_sapiens.py's SAPIENS_LANDMARK_MAP -- see
    # facescape_ear_points.py's module docstring) -- ear_frame_000000.json is
    # never touched by that script, so this is always the freshest value.
    override_points: dict[int, dict] = {p["index"]: p for p in facescape_ear_points.load_ear_points(actor_id, expression_id, camera_id)}

    # Same overlay for the other four propagated groups -- see
    # PROPAGATED_GROUP_PREFIXES above.
    for file_prefix in PROPAGATED_GROUP_PREFIXES:
        for p in _load_group_points(actor_id, expression_id, camera_id, file_prefix):
            override_points[p["index"]] = p

    if override_points:
        payload["points"] = [p for p in payload["points"] if p["index"] not in override_points] + list(override_points.values())

    return LandmarkFrame(**payload)


@router.get("/overlay-frame/{actor_id}/{expression_id}/{camera_id}")
def get_facescape_overlay_frame(actor_id: str, expression_id: str, camera_id: str) -> FileResponse:
    """Serves the keyline overlay wrap_script_facescape/run_pipeline.py renders
    (via render_overlay_frame.py) once a pair is wrapped, for the frontend's
    Overlay Frame tab -- same role as /labels/overlay-video for Nersemble."""
    settings = get_settings()
    overlay_path = settings.line_renders_facescape_root / f"{actor_id}_{expression_id}_{camera_id}.jpg"
    if not overlay_path.exists():
        raise HTTPException(
            status_code=404,
            detail="FaceScape overlay frame rendering is not available yet for this pair.",
        )
    # Same reasoning as /facescape/image: run_pipeline.py can overwrite this
    # exact path on a rerun, so avoid indefinite browser caching by URL alone.
    return FileResponse(overlay_path, media_type="image/jpeg", headers={"Cache-Control": "no-cache"})


@router.get("/point-overlay-frame/{actor_id}/{expression_id}/{camera_id}")
def get_facescape_point_overlay_frame(actor_id: str, expression_id: str, camera_id: str) -> FileResponse:
    """Serves wrap_script_facescape/render_point_overlay_frame.py's landmark
    point overlay (annotated vs. wrapped-mesh landmarks) for the frontend's
    Overlay Landmarks tab -- same cameras and caching as /overlay-frame."""
    settings = get_settings()
    overlay_path = settings.point_overlay_renders_facescape_root / f"{actor_id}_{expression_id}_{camera_id}.jpg"
    if not overlay_path.exists():
        raise HTTPException(
            status_code=404,
            detail="FaceScape point overlay rendering is not available yet for this pair.",
        )
    return FileResponse(overlay_path, media_type="image/jpeg", headers={"Cache-Control": "no-cache"})
