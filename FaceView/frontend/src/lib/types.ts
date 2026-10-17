export interface MediaPath {
  actor_id: string;
  sequence_id: string;
  camera_id: string;
  video_path: string;
  fps?: number;
  num_frames?: number;
}

export interface CameraDescriptor {
  camera_id: string;
  video: MediaPath;
  status: string;
}

// "no_neutral_label" is the implicit default for any pair the backend has never
// recorded a status for (see LabelTracker.get_status). "unconfirmed" is set
// automatically by run_pipeline_batch_all.py; "confirmed" is a person's manual
// sign-off after watching that pair's Overlay Video tab.
export type LabelStatus = "no_neutral_label" | "unlabeled" | "unreviewed" | "unconfirmed" | "confirmed";

export const LABEL_STATUSES: LabelStatus[] = ["no_neutral_label", "unlabeled", "unreviewed", "unconfirmed", "confirmed"];

export interface SequenceDescriptor {
  sequence_id: string;
  cameras: CameraDescriptor[];
  has_landmarks: boolean;
  has_delegation_video: boolean;
  label_status: LabelStatus;
}

export interface ActorDescriptor {
  actor_id: string;
  sequences: SequenceDescriptor[];
}

export interface DatasetResponse {
  actors: ActorDescriptor[];
  face_indices?: Record<string, number[]>;
  default_cameras?: string[];
}

export interface LandmarkPoint {
  index: number;
  x: number;
  y: number;
  z?: number;
  visibility?: number;
  region?: string;
}

export interface LandmarkFrame {
  frame_idx: number;
  points: LandmarkPoint[];
}

export interface LandmarkBatchResponse {
  frames: LandmarkFrame[];
  frame_indices: number[];
  frame_stride: number;
}

export interface LandmarkPayload {
  frame_idx: number;
  camera_id: string;
  sequence_id: string;
  actor_id: string;
  points: LandmarkPoint[];
}

export interface LandmarkBatchPayload {
  payloads: LandmarkPayload[];
  points: number[];
  changed_by_camera: Record<string, number[]>;
}

export interface JobResponse {
  job_id: string;
  job_type: string;
  sequence_id: string;
  camera_id: string;
  status: string;
  message?: string;
}

export interface PreprocessResponse {
  job_id: string;
  status: string;
}

export interface NeutralFrameResponse {
  frame_idx: number;
  status: string;
}

export interface NeutralFrameQueryResponse {
  sequence_id: string | null;
  frame_idx: number | null;
}

export interface LabelStatusResponse {
  actor_id: string;
  sequence_id: string;
  status: LabelStatus;
}

export interface LabelStatusMapResponse {
  // actor_id -> sequence_id -> status
  statuses: Record<string, Record<string, LabelStatus>>;
  dataset?: "Nersemble" | "FaceScape";
}

// FaceScape -- static multiview scans, no video, no delegation pipeline.
// Deliberately separate from CameraDescriptor/SequenceDescriptor/ActorDescriptor/
// DatasetResponse above (Nersemble's shapes) rather than reusing them.
export interface FaceScapeCameraDescriptor {
  camera_id: string;
  has_image: boolean;
  has_landmarks: boolean;
}

export interface FaceScapeExpressionDescriptor {
  expression_id: string;
  cameras: FaceScapeCameraDescriptor[];
  has_landmarks: boolean;
  label_status: LabelStatus;
  // Invalid pair (no origin-space alignment): `cameras` are the pair's own raw
  // cameras, and its overlays use overlay_camera_ids instead of the canonical ones.
  is_invalid: boolean;
  overlay_camera_ids: string[];
}

export interface FaceScapeActorDescriptor {
  actor_id: string;
  expressions: FaceScapeExpressionDescriptor[];
}

export interface FaceScapeDatasetResponse {
  actors: FaceScapeActorDescriptor[];
  front_cameras: string[];
  left_front_cameras: string[];
  right_front_cameras: string[];
}

// "Add Left/Right Ear Top" -- index 2 (left) / 9 (right), the only two
// standard FLAME indices with no sapiens2 auto-detection source. See
// backend/app/services/facescape_ear_points.py.
export interface SaveEarPointRequest {
  actor_id: string;
  expression_id: string;
  ear: "left" | "right";
  clicks: Record<string, [number, number]>;
}

export interface SaveEarPointResponse {
  status: "ok";
  cameras_written: number;
}

// FaceScape's counterpart to NeutralFramePayload/NeutralFrameResponse/
// NeutralFrameQueryResponse above -- no frame/sequence concept, just a saved
// expression id per actor.
export interface SaveNeutralExpressionRequest {
  actor_id: string;
  expression_id: string;
}

export interface NeutralExpressionResponse {
  status: "confirmed";
  expression_id: string;
}

export interface NeutralExpressionQueryResponse {
  expression_id: string | null;
}

// --- Ava-256 (one keyline video per capture, per-frame labels) ---------------
// Per-frame CSV statuses: "unconfirmed" until labeled in the Ava-256 tab.
export type Ava256FrameLabel = "Valid" | "Invalid" | "Rerun";
export const AVA256_FRAME_LABELS: Ava256FrameLabel[] = ["Valid", "Invalid", "Rerun"];
export type Ava256FrameStatus = Ava256FrameLabel | "unconfirmed";

export interface Ava256CaptureDescriptor {
  capture_id: string;
  label_status: LabelStatus;
  has_video: boolean;
  num_frames: number;
  counts: Record<Ava256FrameStatus, number>;
  is_done: boolean;
}

export interface Ava256DatasetResponse {
  captures: Ava256CaptureDescriptor[];
}

export interface Ava256FrameRow {
  frame_id: string;
  status: Ava256FrameStatus;
  // Mean skin-landmark pixel error of the frame's wrap (check_ava256_skin_landmark_errors.py).
  skin_error_px?: number | null;
}

export interface Ava256FramesResponse {
  capture_id: string;
  frames: Ava256FrameRow[];
  is_done: boolean;
  warnings: string[];
}

// Inclusive, 1-based frame positions per label.
export type Ava256RangesByLabel = Partial<Record<Ava256FrameLabel, [number, number][]>>;

export interface Ava256DoneResponse {
  capture_id: string;
  export_path: string | null;
  labels_path: string;
  counts: Record<string, number>;
}
