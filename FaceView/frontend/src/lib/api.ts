import type {
  Ava256DatasetResponse,
  Ava256DoneResponse,
  Ava256FramesResponse,
  Ava256RangesByLabel,
  DatasetResponse,
  FaceScapeDatasetResponse,
  LabelStatusMapResponse,
  LabelStatusResponse,
  LandmarkFrame,
  LandmarkBatchResponse,
  LandmarkBatchPayload,
  LandmarkPayload,
  NeutralFrameResponse,
  NeutralFrameQueryResponse,
  NeutralExpressionResponse,
  NeutralExpressionQueryResponse,
  PreprocessResponse,
  SaveEarPointRequest,
  SaveEarPointResponse,
  SaveNeutralExpressionRequest
} from "@/lib/types";

const API_BASE = import.meta.env.VITE_API_BASE ?? "/api";
const REQUEST_TIMEOUT_MS = Number(import.meta.env.VITE_REQUEST_TIMEOUT_MS ?? 20000);

const WS_BASE = (() => {
  if (import.meta.env.VITE_WS_BASE) {
    return import.meta.env.VITE_WS_BASE as string;
  }
  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  return `${protocol}://${window.location.host}`;
})();

async function fetchJSON<T>(input: RequestInfo, init?: RequestInit): Promise<T> {
  const controller = new AbortController();
  const timeoutId = window.setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  const response = await fetch(input, {
    ...init,
    signal: init?.signal ?? controller.signal
  }).finally(() => {
    window.clearTimeout(timeoutId);
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(`Request failed (${response.status}): ${text}`);
  }
  return (await response.json()) as T;
}

export async function getDataset(): Promise<DatasetResponse> {
  return fetchJSON<DatasetResponse>(`${API_BASE}/datasets`);
}

export interface LandmarksQuery {
  actorId: string;
  sequenceId: string;
  cameraId: string;
  frameIndices?: number[];
  loadAll?: boolean;
}

export async function getLandmarks(query: LandmarksQuery): Promise<LandmarkBatchResponse> {
  const params = new URLSearchParams({
    actor_id: query.actorId,
    sequence_id: query.sequenceId,
    camera_id: query.cameraId
  });
  if (query.loadAll) {
    params.set("load_all", "true");
  } else if (query.frameIndices?.length) {
    query.frameIndices.forEach((frame_idx) => params.append("frame_indices", frame_idx.toString()));
  } else {
    throw new Error("getLandmarks requires frameIndices or loadAll");
  }
  return fetchJSON<LandmarkBatchResponse>(`${API_BASE}/landmarks?${params.toString()}`);
}

export async function getSkinLandmarks(actorId: string, sequenceId: string, cameraId: string): Promise<LandmarkBatchResponse> {
  const params = new URLSearchParams({ actor_id: actorId, sequence_id: sequenceId, camera_id: cameraId });
  return fetchJSON<LandmarkBatchResponse>(`${API_BASE}/landmarks/skin?${params.toString()}`);
}

export async function saveLandmarks(payloads: LandmarkPayload[], changedPointIndices: number[], changedByCamera: Record<string, number[]>): Promise<{ job_id: string; status: string }> {
  return fetchJSON(`${API_BASE}/landmarks`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      payloads,
      points: changedPointIndices,
      changed_by_camera: changedByCamera
    } satisfies LandmarkBatchPayload)
  });
}

export async function saveNeutralFrame(actorId: string, sequenceId: string, frameIdx: number): Promise<NeutralFrameResponse> {
  return fetchJSON<NeutralFrameResponse>(`${API_BASE}/landmarks/neutral_frame`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ actor_id: actorId, sequence_id: sequenceId, frame_idx: frameIdx })
  });
}

export async function getNeutralFrame(actorId: string, sequenceId: string): Promise<NeutralFrameQueryResponse> {
  const params = new URLSearchParams({ actor_id: actorId, sequence_id: sequenceId });
  return fetchJSON<NeutralFrameQueryResponse>(`${API_BASE}/landmarks/neutral_frame?${params.toString()}`);
}

export async function startPreprocess(actorId: string, sequenceId: string, cameras?: string[]): Promise<PreprocessResponse> {
  return fetchJSON<PreprocessResponse>(`${API_BASE}/sequences/${actorId}/${sequenceId}/preprocess`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ cameras })
  });
}

export async function confirmLabeled(actorId: string, sequenceId: string): Promise<LabelStatusResponse> {
  return fetchJSON<LabelStatusResponse>(`${API_BASE}/labels/confirm-labeled`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ actor_id: actorId, sequence_id: sequenceId })
  });
}

export async function confirm(
  actorId: string,
  sequenceId: string,
  dataset: "Nersemble" | "FaceScape" = "Nersemble"
): Promise<LabelStatusResponse> {
  return fetchJSON<LabelStatusResponse>(`${API_BASE}/labels/confirm`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ actor_id: actorId, sequence_id: sequenceId, dataset })
  });
}

export async function unconfirmLabeled(actorId: string, sequenceId: string): Promise<LabelStatusResponse> {
  return fetchJSON<LabelStatusResponse>(`${API_BASE}/labels/unconfirm-labeled`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ actor_id: actorId, sequence_id: sequenceId })
  });
}

export async function unconfirm(
  actorId: string,
  sequenceId: string,
  dataset: "Nersemble" | "FaceScape" = "Nersemble"
): Promise<LabelStatusResponse> {
  return fetchJSON<LabelStatusResponse>(`${API_BASE}/labels/unconfirm`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ actor_id: actorId, sequence_id: sequenceId, dataset })
  });
}

export async function unconfirmConfirmed(
  actorId: string,
  sequenceId: string,
  dataset: "Nersemble" | "FaceScape" = "Nersemble"
): Promise<LabelStatusResponse> {
  return fetchJSON<LabelStatusResponse>(`${API_BASE}/labels/unconfirm-confirmed`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ actor_id: actorId, sequence_id: sequenceId, dataset })
  });
}

export async function getActorStatus(
  actorId: string,
  dataset: "Nersemble" | "FaceScape" = "Nersemble"
): Promise<LabelStatusMapResponse> {
  const params = new URLSearchParams({ dataset });
  return fetchJSON<LabelStatusMapResponse>(`${API_BASE}/labels/status/${encodeURIComponent(actorId)}?${params}`);
}

export function getOverlayVideoUrl(actorId: string, sequenceId: string): string {
  return `${API_BASE}/labels/overlay-video/${encodeURIComponent(actorId)}/${encodeURIComponent(sequenceId)}`;
}

export function getPointOverlayVideoUrl(actorId: string, sequenceId: string): string {
  return `${API_BASE}/labels/point-overlay-video/${encodeURIComponent(actorId)}/${encodeURIComponent(sequenceId)}`;
}

export async function getAva256Dataset(): Promise<Ava256DatasetResponse> {
  return fetchJSON<Ava256DatasetResponse>(`${API_BASE}/ava256/captures`);
}

export function getAva256VideoUrl(captureId: string): string {
  return `${API_BASE}/ava256/${encodeURIComponent(captureId)}/video`;
}

export async function getAva256Frames(captureId: string): Promise<Ava256FramesResponse> {
  return fetchJSON<Ava256FramesResponse>(`${API_BASE}/ava256/${encodeURIComponent(captureId)}/frames`);
}

export async function labelAva256Frames(captureId: string, ranges: Ava256RangesByLabel): Promise<Ava256FramesResponse> {
  return fetchJSON<Ava256FramesResponse>(`${API_BASE}/ava256/${encodeURIComponent(captureId)}/frames`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ranges })
  });
}

export async function finishAva256Capture(captureId: string): Promise<Ava256DoneResponse> {
  // Loads and re-saves every valid frame's mesh -- can take longer than the default timeout.
  const controller = new AbortController();
  const timeoutId = window.setTimeout(() => controller.abort(), 10 * 60 * 1000);
  try {
    return await fetchJSON<Ava256DoneResponse>(`${API_BASE}/ava256/${encodeURIComponent(captureId)}/done`, {
      method: "POST",
      signal: controller.signal
    });
  } finally {
    window.clearTimeout(timeoutId);
  }
}

export async function getFaceScapeDataset(): Promise<FaceScapeDatasetResponse> {
  return fetchJSON<FaceScapeDatasetResponse>(`${API_BASE}/facescape/dataset`);
}

export function getFaceScapeImageUrl(actorId: string, expressionId: string, cameraId: string): string {
  return `${API_BASE}/facescape/image/${encodeURIComponent(actorId)}/${encodeURIComponent(expressionId)}/${encodeURIComponent(cameraId)}`;
}

export async function getFaceScapeLandmarks(actorId: string, expressionId: string, cameraId: string): Promise<LandmarkFrame> {
  return fetchJSON<LandmarkFrame>(
    `${API_BASE}/facescape/landmarks/${encodeURIComponent(actorId)}/${encodeURIComponent(expressionId)}/${encodeURIComponent(cameraId)}`
  );
}

export async function getFaceScapeSkinLandmarks(actorId: string, expressionId: string, cameraId: string): Promise<LandmarkFrame> {
  return fetchJSON<LandmarkFrame>(
    `${API_BASE}/facescape/landmarks/skin/${encodeURIComponent(actorId)}/${encodeURIComponent(expressionId)}/${encodeURIComponent(cameraId)}`
  );
}

export function getFaceScapeOverlayFrameUrl(actorId: string, expressionId: string, cameraId: string): string {
  return `${API_BASE}/facescape/overlay-frame/${encodeURIComponent(actorId)}/${encodeURIComponent(expressionId)}/${encodeURIComponent(cameraId)}`;
}

export function getFaceScapePointOverlayFrameUrl(actorId: string, expressionId: string, cameraId: string): string {
  return `${API_BASE}/facescape/point-overlay-frame/${encodeURIComponent(actorId)}/${encodeURIComponent(expressionId)}/${encodeURIComponent(cameraId)}`;
}

// Triangulate-only for the given actor/expression (no cross-expression
// CoTracker propagation -- see api/routes/facescape.py's
// save_facescape_ear_point), so this fits well within fetchJSON's default
// REQUEST_TIMEOUT_MS.
export async function saveFacescapeEarPoint(
  actorId: string,
  expressionId: string,
  ear: "left" | "right",
  clicks: Record<string, { x: number; y: number }>
): Promise<SaveEarPointResponse> {
  const clicksPayload: Record<string, [number, number]> = {};
  for (const [cameraId, { x, y }] of Object.entries(clicks)) {
    clicksPayload[cameraId] = [x, y];
  }
  return fetchJSON<SaveEarPointResponse>(`${API_BASE}/facescape/landmarks/ear`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      actor_id: actorId,
      expression_id: expressionId,
      ear,
      clicks: clicksPayload
    } satisfies SaveEarPointRequest)
  });
}

export async function saveFacescapeNeutralExpression(actorId: string, expressionId: string): Promise<NeutralExpressionResponse> {
  return fetchJSON<NeutralExpressionResponse>(`${API_BASE}/facescape/neutral_expression`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ actor_id: actorId, expression_id: expressionId } satisfies SaveNeutralExpressionRequest)
  });
}

export async function getFacescapeNeutralExpression(actorId: string): Promise<NeutralExpressionQueryResponse> {
  const params = new URLSearchParams({ actor_id: actorId });
  return fetchJSON<NeutralExpressionQueryResponse>(`${API_BASE}/facescape/neutral_expression?${params.toString()}`);
}

export function createSequenceSocket(sequenceId: string): WebSocket {
  const url = `${WS_BASE}/ws/sequences/${sequenceId}`;
  const socket = new WebSocket(url);
  socket.addEventListener("open", () => socket.send("subscribe"));
  return socket;
}

export const api = {
  getDataset,
  getLandmarks,
  getSkinLandmarks,
  saveLandmarks,
  saveNeutralFrame,
  getNeutralFrame,
  confirmLabeled,
  confirm,
  unconfirmLabeled,
  unconfirm,
  unconfirmConfirmed,
  getActorStatus,
  getOverlayVideoUrl,
  getPointOverlayVideoUrl,
  startPreprocess,
  createSequenceSocket,
  getFaceScapeDataset,
  getAva256Dataset,
  getAva256VideoUrl,
  getAva256Frames,
  labelAva256Frames,
  finishAva256Capture,
  getFaceScapeImageUrl,
  getFaceScapeLandmarks,
  getFaceScapeSkinLandmarks,
  getFaceScapeOverlayFrameUrl,
  getFaceScapePointOverlayFrameUrl,
  saveFacescapeEarPoint,
  saveFacescapeNeutralExpression,
  getFacescapeNeutralExpression
};
