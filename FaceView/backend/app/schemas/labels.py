"""Actor/sequence review-label schemas."""
from __future__ import annotations

from typing import Dict, Literal
from pydantic import BaseModel

# "no_neutral_label" is never persisted to label_tracker.json -- it's the implicit
# default for any actor/sequence pair absent from the file (see LabelTracker.get_status).
# "unconfirmed" is set automatically by run_pipeline_batch_all.py once a pair's
# frames are all wrapped with skin landmarks (see mark_unconfirmed in
# label_tracker.py); "confirmed" is a person's manual sign-off after watching
# that pair's overlay video, replacing the old "reviewed" status.
LabelStatus = Literal["no_neutral_label", "unlabeled", "unreviewed", "unconfirmed", "confirmed"]


class LabelStatusResponse(BaseModel):
    actor_id: str
    sequence_id: str
    status: LabelStatus


class LabelStatusMapResponse(BaseModel):
    # actor_id -> sequence_id -> status
    statuses: Dict[str, Dict[str, LabelStatus]]
    dataset: Literal["Nersemble", "FaceScape"] = "Nersemble"


class ConfirmLabelRequest(BaseModel):
    actor_id: str
    sequence_id: str
    # Defaults to "Nersemble" so existing frontend calls (which predate the
    # FaceScape namespace split) keep working without sending this field.
    dataset: Literal["Nersemble", "FaceScape"] = "Nersemble"
