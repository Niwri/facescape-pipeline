"""FaceScape-facing schemas -- static multiview scans, no video, no delegation
pipeline. Deliberately separate from schemas/datasets.py's Nersemble shapes
(CameraDescriptor/SequenceDescriptor/ActorDescriptor/DatasetResponse) rather
than reusing them, since FaceScape has no video/fps/num_frames concept."""
from __future__ import annotations

from typing import Dict, List, Literal, Tuple
from pydantic import BaseModel, Field

from .labels import LabelStatus


class FaceScapeCameraDescriptor(BaseModel):
    camera_id: str
    has_image: bool = False
    has_landmarks: bool = False


class FaceScapeExpressionDescriptor(BaseModel):
    expression_id: str
    cameras: List[FaceScapeCameraDescriptor]
    has_landmarks: bool = False
    label_status: LabelStatus = "unlabeled"
    # Invalid pair (no origin-space alignment, see services/facescape_invalid.py):
    # `cameras` is then the pair's own raw camera list, and its overlay renders
    # use overlay_camera_ids instead of the canonical overlay cameras.
    is_invalid: bool = False
    overlay_camera_ids: List[str] = []


class FaceScapeActorDescriptor(BaseModel):
    actor_id: str
    expressions: List[FaceScapeExpressionDescriptor]


class FaceScapeDatasetResponse(BaseModel):
    actors: List[FaceScapeActorDescriptor]
    # Every camera id this dataset ever labels, grouped by region-of-use --
    # loaded from settings.facescape_cameras_path (see
    # wrap_script_facescape/labeling/compute_facescape_cameras.py). Lets the
    # frontend render a full camera grid (including not-yet-usable cameras
    # for a given actor/expression) without a second request.
    front_cameras: List[str] = []
    left_front_cameras: List[str] = []
    right_front_cameras: List[str] = []


class SaveEarPointRequest(BaseModel):
    """Body for POST /facescape/landmarks/ear -- `clicks` is camera_id ->
    (x, y) normalized [0,1] image coordinates, one entry per camera view the
    reviewer clicked while in "Add Left/Right Ear Top" mode. At least 2
    clicks are required to triangulate (see facescape_ear_points.save_ear_point)."""
    actor_id: str
    expression_id: str
    ear: Literal["left", "right"]
    clicks: Dict[str, Tuple[float, float]] = Field(min_length=2)


class SaveEarPointResponse(BaseModel):
    """Triangulate-and-project only, for whichever actor/expression the point
    was clicked on -- no cross-expression propagation (see
    api/routes/facescape.py's save_facescape_ear_point, per explicit
    instruction not to propagate ear landmarks)."""
    status: Literal["ok"] = "ok"
    cameras_written: int


class SaveNeutralExpressionRequest(BaseModel):
    """Body for POST /facescape/neutral_expression -- FaceScape's counterpart
    to schemas/landmarks.py's NeutralFramePayload, minus the frame/sequence
    concept FaceScape doesn't have (one static expression stands in for
    Nersemble's video frame)."""
    actor_id: str
    expression_id: str


class NeutralExpressionResponse(BaseModel):
    status: Literal["confirmed"] = "confirmed"
    expression_id: str


class NeutralExpressionQueryResponse(BaseModel):
    expression_id: str | None = Field(
        default=None, description="This actor's saved neutral expression id, or null if none has been saved yet"
    )
