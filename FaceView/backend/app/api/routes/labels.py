"""Actor/sequence review-label endpoints."""
from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from fastapi.responses import FileResponse

from ...core.config import get_settings
from ...schemas.labels import ConfirmLabelRequest, LabelStatusMapResponse, LabelStatusResponse
from ...services.label_tracker import Dataset, label_tracker
from ...services.wrapped_mesh_export import (
    delete_exported_facescape_pair,
    delete_exported_sequence,
    export_wrapped_facescape_pair,
    export_wrapped_sequence,
    rename_wrapped_to_old_facescape_pair,
    rename_wrapped_to_old_sequence,
)

router = APIRouter(prefix="/labels", tags=["labels"])

# Camera the overlay video is always rendered from -- see run_pipeline_batch_all.py's
# call into test_render_gt.py, which writes to
# line_renders_root/<actor>_<sequence>_<camera>.mp4 (flat, matching test_render_gt.py's
# own output_path convention).
OVERLAY_VIDEO_CAMERA_ID = "222200046"
# Camera for the Overlay Landmarks point-overlay video; must match
# wrap_script/run_pipeline_batch_all.py's POINT_OVERLAY_VIDEO_CAMERA_ID. Videos
# rendered before the switch are on OVERLAY_VIDEO_CAMERA_ID and are served as a
# fallback.
POINT_OVERLAY_VIDEO_CAMERA_ID = "222200037"


@router.post("/confirm-labeled", response_model=LabelStatusResponse)
def confirm_labeled(request: ConfirmLabelRequest) -> LabelStatusResponse:
    try:
        status = label_tracker.confirm_labeled(request.actor_id, request.sequence_id, dataset=request.dataset)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return LabelStatusResponse(actor_id=request.actor_id, sequence_id=request.sequence_id, status=status)


@router.post("/confirm", response_model=LabelStatusResponse)
def confirm(request: ConfirmLabelRequest, background_tasks: BackgroundTasks) -> LabelStatusResponse:
    try:
        status = label_tracker.confirm(request.actor_id, request.sequence_id, dataset=request.dataset)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    # Export this pair's wrapped_mesh.obj(s) to a compact .pt bundle now that
    # it's signed off. Runs after the response is sent (parsing the .obj(s) is
    # the slow part of this endpoint, and the frontend only needs the status
    # flip to update immediately -- see export_wrapped_sequence's/
    # export_wrapped_facescape_pair's own docstrings for why a failure here is
    # safe to defer/ignore rather than something the confirm response needs to
    # wait on or report).
    if request.dataset == "FaceScape":
        background_tasks.add_task(export_wrapped_facescape_pair, request.actor_id, request.sequence_id)

        # FaceScape-only: confirming any one of an actor's "unconfirmed" pairs
        # confirms every OTHER "unconfirmed" pair of that actor too, not just
        # this one -- a reviewer signing off on one expression's wrap quality
        # is in practice signing off on the whole actor's wrap batch (they all
        # went through the same run_pipeline_batch_temp.py pass), so requiring
        # N separate clicks per actor would be pure friction. Read fresh (not
        # the all_statuses map from before label_tracker.confirm() above) so
        # this reflects the just-written status of request.sequence_id itself
        # too, though it's skipped below via the actor_id/expression_id check
        # since it's already "confirmed" by this point, not "unconfirmed".
        sibling_statuses = label_tracker.get_all(dataset="FaceScape").get(request.actor_id, {})
        for sibling_expression_id, sibling_status in sibling_statuses.items():
            if sibling_expression_id == request.sequence_id or sibling_status != "unconfirmed":
                continue
            try:
                label_tracker.confirm(request.actor_id, sibling_expression_id, dataset="FaceScape")
            except ValueError:
                # Lost a race with something else that moved this sibling off
                # "unconfirmed" between the read above and this write -- leave
                # it as whatever it is now rather than failing the whole request.
                continue
            background_tasks.add_task(export_wrapped_facescape_pair, request.actor_id, sibling_expression_id)
    else:
        background_tasks.add_task(export_wrapped_sequence, request.actor_id, request.sequence_id)
    return LabelStatusResponse(actor_id=request.actor_id, sequence_id=request.sequence_id, status=status)


@router.post("/unconfirm-labeled", response_model=LabelStatusResponse)
def unconfirm_labeled(request: ConfirmLabelRequest) -> LabelStatusResponse:
    try:
        status = label_tracker.unconfirm_labeled(request.actor_id, request.sequence_id, dataset=request.dataset)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return LabelStatusResponse(actor_id=request.actor_id, sequence_id=request.sequence_id, status=status)


@router.post("/unconfirm", response_model=LabelStatusResponse)
def unconfirm(request: ConfirmLabelRequest) -> LabelStatusResponse:
    """Reverts an "unconfirmed" pair back to "unreviewed" (undoing
    run_pipeline_batch_all.py's/run_pipeline_batch.py's own automatic
    unreviewed->unconfirmed transition), and renames this pair's
    wrapped_mesh.obj(s) to old_wrapped_mesh.obj -- so a rerun of the batch
    wrap script, which now skips a frame outright once its wrapped_mesh.obj
    exists, actually re-wraps it instead of treating the old, presumably-
    rejected wrap as already done."""
    try:
        status = label_tracker.unconfirm(request.actor_id, request.sequence_id, dataset=request.dataset)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if request.dataset == "FaceScape":
        rename_wrapped_to_old_facescape_pair(request.actor_id, request.sequence_id)
    else:
        rename_wrapped_to_old_sequence(request.actor_id, request.sequence_id)
    return LabelStatusResponse(actor_id=request.actor_id, sequence_id=request.sequence_id, status=status)


@router.post("/unconfirm-confirmed", response_model=LabelStatusResponse)
def unconfirm_confirmed(request: ConfirmLabelRequest) -> LabelStatusResponse:
    """Reverts a "confirmed" sign-off back to "unconfirmed", and deletes this
    pair's final_wrapped_data/<actor>/<sequence>.pt export -- a reverted pair
    shouldn't keep serving a compact export as if it were still confirmed."""
    try:
        status = label_tracker.unconfirm_confirmed(request.actor_id, request.sequence_id, dataset=request.dataset)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if request.dataset == "FaceScape":
        delete_exported_facescape_pair(request.actor_id, request.sequence_id)
    else:
        delete_exported_sequence(request.actor_id, request.sequence_id)
    return LabelStatusResponse(actor_id=request.actor_id, sequence_id=request.sequence_id, status=status)


@router.get("/status/{actor_id}", response_model=LabelStatusMapResponse)
def get_actor_status(actor_id: str, dataset: Dataset = Query("Nersemble")) -> LabelStatusMapResponse:
    """Fresh (uncached) label_status per sequence/expression for one actor, in
    either dataset. Unlike /api/datasets or /api/facescape/dataset (see
    dataset_index.py / facescape_dataset_index.py), which cache their full scan
    for up to 30s -- and can serve several minutes staler than that under NFS
    degradation, per those modules' own docstrings -- this only reads
    label_tracker.json plus a per-sequence/expression directory-existence check,
    cheap enough to call on demand. Meant for the frontend's per-actor and
    "Update all" actions, to pick up out-of-band status changes (wrap_script
    batch tooling -- run_pipeline_batch_all.py's mark_unconfirmed,
    run_neutral_cotracker_propagation.py's confirm_labeled, and their FaceScape
    counterparts -- writes label_tracker.json directly, not through this API)
    without waiting on the dataset cache. `dataset` defaults to "Nersemble" so
    existing frontend calls that predate the FaceScape split keep working
    without sending it."""
    settings = get_settings()
    actor_dir = (settings.facescape_landmarks_root if dataset == "FaceScape" else settings.landmarks_root) / actor_id
    if not actor_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"No landmarks directory for actor {actor_id} in dataset {dataset}")
    all_statuses = label_tracker.get_all(dataset=dataset)
    statuses = {
        sequence_dir.name: label_tracker.get_status(actor_id, sequence_dir.name, dataset=dataset, all_statuses=all_statuses)
        for sequence_dir in sorted(actor_dir.iterdir())
        if sequence_dir.is_dir()
    }
    return LabelStatusMapResponse(statuses={actor_id: statuses}, dataset=dataset)


@router.get("/overlay-video/{actor_id}/{sequence_id}")
def get_overlay_video(actor_id: str, sequence_id: str) -> FileResponse:
    """Serves the test_render_gt.py overlay video run_pipeline_batch_all.py renders
    once a pair reaches "unconfirmed", so the frontend can play it in the
    Overlay Video tab without needing a static-file mount for line_renders_root."""
    settings = get_settings()
    video_path = (
        settings.line_renders_root / f"{actor_id}_{sequence_id}_{OVERLAY_VIDEO_CAMERA_ID}.mp4"
    )
    if not video_path.exists():
        raise HTTPException(status_code=404, detail=f"No overlay video found at {video_path}")
    return FileResponse(video_path, media_type="video/mp4")


@router.get("/point-overlay-video/{actor_id}/{sequence_id}")
def get_point_overlay_video(actor_id: str, sequence_id: str) -> FileResponse:
    """Serves wrap_script/render_point_overlay.py's landmark point-overlay video
    (annotated vs. wrapped-mesh landmarks) for the frontend's Overlay Landmarks
    tab, on POINT_OVERLAY_VIDEO_CAMERA_ID -- falling back to an older video
    rendered on OVERLAY_VIDEO_CAMERA_ID if that's all this pair has."""
    settings = get_settings()
    for camera_id in (POINT_OVERLAY_VIDEO_CAMERA_ID, OVERLAY_VIDEO_CAMERA_ID):
        video_path = settings.point_overlay_renders_root / f"{actor_id}_{sequence_id}_{camera_id}.mp4"
        if video_path.exists():
            return FileResponse(video_path, media_type="video/mp4")
    raise HTTPException(status_code=404, detail=f"No point overlay video found for {actor_id}/{sequence_id}")
