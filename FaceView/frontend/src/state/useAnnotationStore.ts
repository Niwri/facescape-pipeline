import { create, type StateCreator } from "zustand";
import { immer } from "zustand/middleware/immer";

import { api } from "@/lib/api";
import type {
  DatasetResponse,
  FaceScapeDatasetResponse,
  FaceScapeExpressionDescriptor,
  LandmarkFrame,
  LandmarkPayload,
  LandmarkPoint
} from "@/lib/types";

interface FrameCacheEntry {
  cameraId: string;
  frameIdx: number;
  payload: LandmarkFrame;
  dirty: boolean;
}

interface HistoryRecord {
  cameraId: string;
  frameIdx: number;
  before: LandmarkPoint[];
  after: LandmarkPoint[];
}

type CameraLoadPhase = "idle" | "loading" | "loaded" | "error";

export interface ToastMessage {
  id: number;
  tone: "success" | "error";
  message: string;
}

// Which dataset's explorer/player is currently shown. Kept as a single flag
// rather than inferring it from selectedActor/selectedFacescapeActor, since
// both could be set at once (switching tabs shouldn't lose either selection).
// Ava-256 keeps its own slice in useAva256Store.ts.
export type DatasetType = "Nersemble" | "FaceScape" | "Ava256";

export interface AnnotationState {
  activeDatasetType: DatasetType;
  dataset?: DatasetResponse;
  faceIndices: Record<string, number[]>;
  // FaceScape state below is a separate, parallel slice -- deliberately not
  // reusing dataset/selectedActor/selectedSequence/frameCache/etc. above, so
  // every existing Nersemble code path (video playback, drag-to-edit,
  // propagation, undo/redo) stays completely untouched. FaceScape has no
  // video, no editing (read-only Cameras tab), and one static frame per
  // camera (no frame index dimension to cache against).
  facescapeDataset?: FaceScapeDatasetResponse;
  isFacescapeDatasetLoading: boolean;
  facescapeExplorerError?: string;
  selectedFacescapeActor?: string;
  selectedFacescapeExpression?: string;
  selectedFacescapeCameras: string[];
  // Keyed by `${actorId}:${expressionId}:${cameraId}` -- one static frame per
  // camera, so no frame index needed in the key (unlike frameCache above).
  facescapeLandmarkCache: Record<string, LandmarkFrame>;
  facescapeLandmarkLoadStatus: Record<string, CameraLoadPhase>;
  // FaceScape counterpart to skinFrameCache/skinCameraLoadStatus below --
  // populated by run_neutral_skin_propagation.py, keyed the same way as
  // facescapeLandmarkCache (`${actorId}:${expressionId}:${cameraId}`).
  facescapeSkinLandmarkCache: Record<string, LandmarkFrame>;
  facescapeSkinLandmarkLoadStatus: Record<string, CameraLoadPhase>;
  // "Add Left/Right Ear Top" mode state -- null when not active. While set,
  // clicking a FaceScape camera image records a pending (unsaved) click for
  // that camera into whichever of facescapeEarClicksLeft/Right matches the
  // active mode (keyed by camera_id; only meaningful for whichever actor/
  // expression is currently selected, so no further prefix is needed)
  // instead of doing nothing (FaceScape's LandmarkOverlay is otherwise
  // non-interactive). Kept as two separate sides (rather than one dict) so
  // switching modes or hitting "Cancel Mode" never loses the other side's
  // progress -- only "Clear" or a successful save for that side resets it.
  facescapeEarAddMode: "left" | "right" | null;
  facescapeEarClicksLeft: Record<string, { x: number; y: number }>;
  facescapeEarClicksRight: Record<string, { x: number; y: number }>;
  // FaceScape counterpart to neutralFrameSequenceId/neutralFrameIdx below --
  // one saved neutral expression per actor (see facescape_neutral.py), null
  // means this actor has none saved yet.
  facescapeNeutralExpression: string | null;
  selectedActor?: string;
  selectedSequence?: string;
  selectedCameras: string[];
  currentFrame: number;
  // Actor-scoped now (one neutral frame per actor, not per actor/sequence
  // pair) -- the same for every sequence of the actor, so these two always
  // reflect selectedActor's own neutral frame regardless of selectedSequence.
  // null sequenceId/frameIdx means this actor has no neutral frame saved yet.
  neutralFrameSequenceId: string | null;
  neutralFrameIdx: number | null;
  frameCache: Record<string, FrameCacheEntry>;
  cameraFrameCounts: Record<string, number>;
  cameraLoadStatus: Record<string, CameraLoadPhase>;
  // "Skin" landmarks (index >= 74) propagated by run_neutral_cotracker_propagation.py,
  // shown only when the "Show skin landmarks" toggle is on. Keyed the same way as
  // frameCache (`${cameraId}:${frameIdx}`), but read-only — no drag/edit support.
  skinFrameCache: Record<string, LandmarkFrame>;
  skinCameraLoadStatus: Record<string, CameraLoadPhase>;
  // Sorted frame indices that actually have landmark data on disk, per camera —
  // the source of truth for frame navigation, from the backend's frame_indices
  // (file-name-derived, see landmark_store.list_available_frame_indices).
  availableFrameIndices: Record<string, number[]>;
  landmarkFrameStride: number;
  // Real fps/frame count measured directly from each delegation video's own
  // container metadata (mp4box). This is the source of truth for playback
  // range/seeking — NOT landmark file count (which can be much sparser than the
  // video's real frame count) and NOT the dataset listing's video.num_frames
  // (never populated by the backend).
  cameraVideoMeta: Record<string, { fps: number; numFrames: number }>;
  statusMessage?: string;
  toast?: ToastMessage;
  isDatasetLoading: boolean;
  isSaving: boolean;
  propagatingCameras: string[];
  // actor:sequence keys currently mid-flight for confirmLabeled/confirm/
  // unconfirmLabeled/unconfirm -- guards against a double-click firing two
  // requests for the same transition (the second lands after the backend has
  // already moved past the expected "from" status and 409s, even though the
  // first request succeeded).
  pendingLabelActions: string[];
  // actor_ids currently mid-flight for refreshActorStatus -- guards the "Update"
  // button against a double-click firing two requests for the same actor.
  refreshingActors: string[];
  // FaceScape counterpart to refreshingActors, for refreshFacescapeActorStatus.
  refreshingFacescapeActors: string[];
  // True while updateAllActorStatuses is mid-flight, for whichever dataset was
  // active when it was triggered -- guards the "Update all" button the same
  // way refreshingActors/refreshingFacescapeActors guard the per-actor one.
  isUpdatingAll: boolean;
  isPlaying: boolean;
  explorerError?: string;
  history: HistoryRecord[];
  redoStack: HistoryRecord[];
  changedPointIndices: number[];
  changedPointsByCamera: Record<string, number[]>;
  fetchDataset: () => Promise<void>;
  // Fresh, uncached label_status per sequence for one actor -- see
  // backend/app/api/routes/labels.py's /labels/status/{actor_id}. Applies
  // in-place, same as confirmLabeled/confirm/etc., so it doesn't pay the
  // full (cached, up to ~30s+ stale) /api/datasets scan just to pick up
  // out-of-band status changes from wrap_script batch tooling. `silent` skips
  // the per-actor toast -- used by updateAllActorStatuses, which shows one
  // summary toast instead of one per actor.
  refreshActorStatus: (actorId: string, options?: { silent?: boolean }) => Promise<void>;
  selectSequence: (actorId: string, sequenceId: string) => void;
  toggleCamera: (cameraId: string) => void;
  setFrame: (frame: number) => void;
  // Nearest available (has-landmark-data) frame in the given direction from
  // `from`, based on availableFrameIndices for cameraId. Falls back to `from`
  // itself if that camera has no known available frames yet (e.g. still loading).
  getAdjacentAvailableFrame: (cameraId: string, from: number, direction: 1 | -1) => number;
  setCameraVideoMeta: (cameraId: string, fps: number, numFrames: number) => void;
  setPlaying: (playing: boolean) => void;
  togglePlayback: () => void;
  loadFrame: (cameraId: string, frameIdx: number, options?: { force?: boolean }) => Promise<void>;
  loadSkinFrames: (cameraId: string) => Promise<void>;
  updatePoint: (cameraId: string, frameIdx: number, updatedPoint: LandmarkPoint) => void;
  addPoint: (cameraId: string, frameIdx: number, point: LandmarkPoint) => void;
  removePoint: (cameraId: string, frameIdx: number, pointIndex: number) => void;
  saveFrame: (frameIdx: number, cameraIds?: string[]) => Promise<void>;
  saveNeutralFrame: () => Promise<void>;
  fetchNeutralFrame: (actorId: string, sequenceId: string) => Promise<void>;
  confirmLabeled: (actorId: string, sequenceId: string) => Promise<void>;
  confirm: (actorId: string, sequenceId: string) => Promise<void>;
  unconfirmLabeled: (actorId: string, sequenceId: string) => Promise<void>;
  unconfirm: (actorId: string, sequenceId: string) => Promise<void>;
  unconfirmConfirmed: (actorId: string, sequenceId: string) => Promise<void>;
  dismissToast: (id: number) => void;
  triggerPreprocess: (actorId: string, sequenceId: string, cameras?: string[]) => Promise<void>;
  undo: () => void;
  redo: () => void;
  goToNextUnlabeledSequence: () => void;
  handleSocketEvent: (event: Record<string, unknown>) => void;
  setActiveDatasetType: (datasetType: DatasetType) => void;
  fetchFacescapeDataset: () => Promise<void>;
  selectFacescapeExpression: (actorId: string, expressionId: string) => void;
  toggleFacescapeCamera: (cameraId: string) => void;
  loadFacescapeLandmarks: (actorId: string, expressionId: string, cameraId: string) => Promise<void>;
  loadFacescapeSkinLandmarks: (actorId: string, expressionId: string, cameraId: string) => Promise<void>;
  confirmFacescape: (actorId: string, expressionId: string) => Promise<void>;
  unconfirmFacescape: (actorId: string, expressionId: string) => Promise<void>;
  unconfirmConfirmedFacescape: (actorId: string, expressionId: string) => Promise<void>;
  startFacescapeEarAddMode: (ear: "left" | "right") => void;
  cancelFacescapeEarAddMode: () => void;
  clearFacescapeEarClicks: () => void;
  recordFacescapeEarClick: (cameraId: string, x: number, y: number) => void;
  saveFacescapeEarPoints: () => Promise<void>;
  saveFacescapeNeutralExpression: () => Promise<void>;
  fetchFacescapeNeutralExpression: (actorId: string) => Promise<void>;
  // FaceScape counterpart to refreshActorStatus.
  refreshFacescapeActorStatus: (actorId: string, options?: { silent?: boolean }) => Promise<void>;
  // "Update all" button's action -- refreshes every actor in whichever dataset
  // is currently active (activeDatasetType), i.e. the same thing as pressing
  // the per-actor "Update" button on every actor in that dataset, one at a
  // time (see MAX_UPDATE_ALL_CONCURRENCY's comment for why not all at once).
  updateAllActorStatuses: () => Promise<void>;
}

const keyForFrame = (cameraId: string, frameIdx: number) => `${cameraId}:${frameIdx}`;

// camera_id and sequence_id are both reused identically across every actor (same
// physical cameras, same expression names), so propagatingCameras must be keyed
// by actor+sequence+camera -- keying by camera_id alone would show the
// "Propagating..." lock on an unrelated actor's identically-named camera/sequence.
export const propagationKey = (actorId: string, sequenceId: string, cameraId: string) =>
  `${actorId}:${sequenceId}:${cameraId}`;

export const labelActionKey = (actorId: string, sequenceId: string) => `${actorId}:${sequenceId}`;

// Separate prefix from labelActionKey -- FaceScape actor ids are small ints
// ("1", "2", ...) that could otherwise collide with a Nersemble actor id in
// the shared pendingLabelActions list.
export const facescapeLabelActionKey = (actorId: string, expressionId: string) => `facescape:${actorId}:${expressionId}`;

const HISTORY_LIMIT = 100;

// updateAllActorStatuses runs one /labels/status/{actor_id} call per actor in
// the active dataset (up to ~421 actors for Nersemble) -- this session has
// repeatedly seen the backing NFS mount degrade badly under concurrent load
// (multi-minute dataset-scan hangs), so a full unbounded burst risks the same
// kind of wedge. A small bounded concurrency keeps "Update all" moving without
// piling on that many requests at once.
const MAX_UPDATE_ALL_CONCURRENCY = 4;

const clonePoints = (points: LandmarkPoint[]) => points.map((point) => ({ ...point }));

const appendHistory = (state: AnnotationState, record: HistoryRecord) => {
  state.history.push(record);
  if (state.history.length > HISTORY_LIMIT) {
    state.history.shift();
  }
};

const pushHistoryRecord = (state: AnnotationState, record: HistoryRecord) => {
  appendHistory(state, record);
  state.redoStack = [];
};

// Locate a sequence in state.dataset for an in-place label_status update after a
// confirm-labeled/confirm/save-neutral-frame call, so the explorer
// reflects the new status immediately without a full dataset refetch.
const findSequence = (state: AnnotationState, actorId: string, sequenceId: string) =>
  state.dataset?.actors
    .find((actor) => actor.actor_id === actorId)
    ?.sequences.find((sequence) => sequence.sequence_id === sequenceId);

// FaceScape counterpart to findSequence, for confirmFacescape/unconfirmFacescape/
// unconfirmConfirmedFacescape's in-place label_status update.
const findFacescapeExpression = (state: AnnotationState, actorId: string, expressionId: string) =>
  state.facescapeDataset?.actors
    .find((actor) => actor.actor_id === actorId)
    ?.expressions.find((expression) => expression.expression_id === expressionId);

type AnnotationStoreCreator = StateCreator<
  AnnotationState,
  [["zustand/immer", never]],
  [],
  AnnotationState
>;

const storeCreator: AnnotationStoreCreator = (set, get) => ({
    activeDatasetType: "Nersemble",
    dataset: undefined,
    faceIndices: {},
    facescapeDataset: undefined,
    isFacescapeDatasetLoading: false,
    facescapeExplorerError: undefined,
    selectedFacescapeActor: undefined,
    selectedFacescapeExpression: undefined,
    selectedFacescapeCameras: [],
    facescapeLandmarkCache: {},
    facescapeLandmarkLoadStatus: {},
    facescapeSkinLandmarkCache: {},
    facescapeSkinLandmarkLoadStatus: {},
    facescapeEarAddMode: null,
    facescapeEarClicksLeft: {},
    facescapeEarClicksRight: {},
    facescapeNeutralExpression: null,
    selectedActor: undefined,
    selectedSequence: undefined,
    selectedCameras: [],
    currentFrame: 0,
    neutralFrameSequenceId: null,
    neutralFrameIdx: null,
    frameCache: {},
    cameraFrameCounts: {},
    cameraLoadStatus: {},
    skinFrameCache: {},
    skinCameraLoadStatus: {},
    availableFrameIndices: {},
    landmarkFrameStride: 1,
    cameraVideoMeta: {},
    isDatasetLoading: false,
    isSaving: false,
    propagatingCameras: [],
    pendingLabelActions: [],
    refreshingActors: [],
    refreshingFacescapeActors: [],
    isUpdatingAll: false,
    isPlaying: false,
    explorerError: undefined,
    history: [],
    redoStack: [],
    changedPointIndices: [] as number[],
    changedPointsByCamera: {} as Record<string, number[]>,

    async fetchDataset() {
      set((state) => {
        state.isDatasetLoading = true;
        state.explorerError = undefined;
      });
      try {
        const dataset = await api.getDataset();
        set((state) => {
          state.dataset = dataset;
          state.faceIndices = dataset.face_indices ?? {};
          if (dataset.actors.length && (!state.selectedActor || !state.selectedSequence)) {
            const firstActor = dataset.actors[0];
            const firstSequence = firstActor.sequences[0];
            state.selectedActor = firstActor.actor_id;
            state.selectedSequence = firstSequence?.sequence_id;
            const available = new Set(firstSequence?.cameras.map((cam) => cam.camera_id) ?? []);
            const defaultCameras = dataset.default_cameras ?? [];
            const preferred = defaultCameras.filter((cameraId) => available.has(cameraId));
            state.selectedCameras = preferred.length > 0
              ? preferred
              : firstSequence?.cameras.slice(0, 2).map((cam) => cam.camera_id) ?? [];
          }
        });
      } catch (error) {
        set((state) => {
          state.explorerError = error instanceof Error ? error.message : String(error);
        });
      } finally {
        set((state) => {
          state.isDatasetLoading = false;
        });
      }
    },

    async refreshActorStatus(actorId: string, options?: { silent?: boolean }) {
      const silent = options?.silent ?? false;
      if (get().refreshingActors.includes(actorId)) return;
      set((state) => {
        state.refreshingActors.push(actorId);
      });
      try {
        const response = await api.getActorStatus(actorId);
        const statuses = response.statuses[actorId] ?? {};
        set((state) => {
          const actor = state.dataset?.actors.find((a) => a.actor_id === actorId);
          if (actor) {
            for (const sequence of actor.sequences) {
              const status = statuses[sequence.sequence_id];
              if (status) sequence.label_status = status;
            }
          }
          if (!silent) {
            state.toast = {
              id: Date.now(),
              tone: "success",
              message: `${actorId}: statuses updated`
            };
          }
        });
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        if (!silent) {
          set((state) => {
            state.toast = { id: Date.now(), tone: "error", message: `Failed to update ${actorId}: ${message}` };
          });
        }
        // Bulk callers (updateAllActorStatuses) need to know this one failed --
        // rethrow only in silent mode so the single-actor "Update" button (which
        // already handled the error above) doesn't also see an unhandled rejection.
        if (silent) throw error;
      } finally {
        set((state) => {
          state.refreshingActors = state.refreshingActors.filter((existing) => existing !== actorId);
        });
      }
    },

    selectSequence(actorId: string, sequenceId: string) {
      set((state) => {
        state.selectedActor = actorId;
        state.selectedSequence = sequenceId;
        state.selectedCameras = [];
        state.frameCache = {};
        state.cameraFrameCounts = {};
        state.cameraLoadStatus = {};
        state.skinFrameCache = {};
        state.skinCameraLoadStatus = {};
        state.availableFrameIndices = {};
        state.cameraVideoMeta = {};
        state.currentFrame = 0;
        state.neutralFrameSequenceId = null;
        state.neutralFrameIdx = null;
        state.isPlaying = false;
        state.history = [];
        state.redoStack = [];
      });
      const dataset = get().dataset;
      const sequence = dataset?.actors
        .find((actor) => actor.actor_id === actorId)?.sequences
        .find((seq) => seq.sequence_id === sequenceId);
      if (sequence) {
        set((state) => {
          const available = new Set(sequence.cameras.map((cam) => cam.camera_id));
          const defaultCameras = dataset?.default_cameras ?? [];
          const preferred = defaultCameras.filter((cameraId) => available.has(cameraId));
          state.selectedCameras = preferred.length > 0
            ? preferred
            : sequence.cameras.slice(0, 2).map((cam) => cam.camera_id);
        });
      }
    },

    toggleCamera(cameraId: string) {
      set((state) => {
        if (!state.selectedCameras.includes(cameraId)) {
          state.selectedCameras.push(cameraId);
          state.cameraLoadStatus[cameraId] = state.cameraLoadStatus[cameraId] ?? "idle";
        } else if (state.selectedCameras.length > 1) {
          state.selectedCameras = state.selectedCameras.filter((id) => id !== cameraId);
        }
      });
    },

    setFrame(frame: number) {
      set((state) => {
        state.currentFrame = Math.max(0, Math.floor(frame));
      });
    },

    getAdjacentAvailableFrame(cameraId: string, from: number, direction: 1 | -1) {
      const available = get().availableFrameIndices[cameraId];
      if (!available || available.length === 0) return from;

      if (direction === 1) {
        const next = available.find((idx) => idx > from);
        return next ?? available[available.length - 1];
      }
      for (let i = available.length - 1; i >= 0; i -= 1) {
        if (available[i] < from) return available[i];
      }
      return available[0];
    },

    setCameraVideoMeta(cameraId: string, fps: number, numFrames: number) {
      set((state) => {
        state.cameraVideoMeta[cameraId] = { fps, numFrames };
      });
    },

    setPlaying(playing: boolean) {
      set((state) => {
        state.isPlaying = playing;
      });
    },

    togglePlayback() {
      set((state) => {
        state.isPlaying = !state.isPlaying;
      });
    },

    async loadFrame(cameraId: string, frameIdx: number, options?: { force?: boolean }) {
      const actorId = get().selectedActor;
      const sequenceId = get().selectedSequence;
      if (!actorId || !sequenceId) return;
      const key = keyForFrame(cameraId, frameIdx);
      
      if (!options?.force && get().frameCache[key]) return;

      const cameraStatus = get().cameraLoadStatus[cameraId];
      if (!options?.force && (cameraStatus === "loading" || cameraStatus === "loaded")) return;

      set((state) => {
        state.cameraLoadStatus[cameraId] = "loading";
      });

      try {
        const response = await api.getLandmarks({
          actorId,
          sequenceId,
          cameraId,
          loadAll: true
        });
        set((state) => {
          response.frames.forEach((frame) => {
            const frameKey = keyForFrame(cameraId, frame.frame_idx);
            state.frameCache[frameKey] = {
              cameraId,
              frameIdx: frame.frame_idx,
              payload: frame,
              dirty: false
            };
          });
          state.cameraFrameCounts[cameraId] = response.frames.length;
          state.availableFrameIndices[cameraId] = response.frame_indices;
          state.landmarkFrameStride = response.frame_stride;
          state.cameraLoadStatus[cameraId] = "loaded";
          if (!response.frames.length) {
            state.statusMessage = `No landmarks found for ${cameraId}`;
          }
        });
      } catch (error) {
        set((state) => {
          state.cameraLoadStatus[cameraId] = "error";
          state.statusMessage = error instanceof Error ? error.message : String(error);
        });
      }
    },

    async loadSkinFrames(cameraId: string) {
      const actorId = get().selectedActor;
      const sequenceId = get().selectedSequence;
      if (!actorId || !sequenceId) return;

      const status = get().skinCameraLoadStatus[cameraId];
      if (status === "loading" || status === "loaded") return;

      set((state) => {
        state.skinCameraLoadStatus[cameraId] = "loading";
      });

      try {
        const response = await api.getSkinLandmarks(actorId, sequenceId, cameraId);
        set((state) => {
          response.frames.forEach((frame) => {
            state.skinFrameCache[keyForFrame(cameraId, frame.frame_idx)] = frame;
          });
          state.skinCameraLoadStatus[cameraId] = "loaded";
        });
      } catch (error) {
        set((state) => {
          state.skinCameraLoadStatus[cameraId] = "error";
          state.statusMessage = error instanceof Error ? error.message : String(error);
        });
      }
    },

    updatePoint(cameraId: string, frameIdx: number, updatedPoint: LandmarkPoint) {
      const key = keyForFrame(cameraId, frameIdx);
      set((state) => {
        const entry = state.frameCache[key];
        if (!entry) return;
        const before = clonePoints(entry.payload.points);
        let changed = false;
        entry.payload.points = entry.payload.points.map((point) => {
          if (point.index === updatedPoint.index) {
            // Check if position changed (x, y, z)
            if (point.x !== updatedPoint.x || point.y !== updatedPoint.y) {
              changed = true;
            }
            return { ...point, ...updatedPoint };
          }
          return point;
        });
        
        // entry.payload.points = entry.payload.points.map((point) =>
        //   point.index === updatedPoint.index ? { ...point, ...updatedPoint } : point
        // );
        if (changed) {
          if (!state.changedPointIndices.includes(updatedPoint.index)) {
            state.changedPointIndices.push(updatedPoint.index);
          }
          // Track which camera this point was moved in, so propagation only
          // triangulates from the cameras the user actually edited.
          const cameraChanges = state.changedPointsByCamera[cameraId] ?? [];
          if (!cameraChanges.includes(updatedPoint.index)) {
            cameraChanges.push(updatedPoint.index);
          }
          state.changedPointsByCamera[cameraId] = cameraChanges;
        }
        entry.dirty = true;
        pushHistoryRecord(state, {
          cameraId,
          frameIdx,
          before,
          after: clonePoints(entry.payload.points)
        });
      });
    },

    addPoint(cameraId: string, frameIdx: number, point: LandmarkPoint) {
      const key = keyForFrame(cameraId, frameIdx);
      set((state) => {
        const entry = state.frameCache[key];
        if (!entry) return;
        const before = clonePoints(entry.payload.points);
        const nextIndex = typeof point.index === "number"
          ? point.index
          : entry.payload.points.reduce((max, current) => Math.max(max, current.index), -1) + 1;
        entry.payload.points = [
          ...entry.payload.points,
          { ...point, index: nextIndex }
        ];
        entry.dirty = true;
        pushHistoryRecord(state, {
          cameraId,
          frameIdx,
          before,
          after: clonePoints(entry.payload.points)
        });
      });
    },

    removePoint(cameraId: string, frameIdx: number, pointIndex: number) {
      const key = keyForFrame(cameraId, frameIdx);
      set((state) => {
        const entry = state.frameCache[key];
        if (!entry) return;
        const before = clonePoints(entry.payload.points);
        const filtered = entry.payload.points.filter((point) => point.index !== pointIndex);
        if (filtered.length === entry.payload.points.length) return;
        entry.payload.points = filtered;
        entry.dirty = true;
        pushHistoryRecord(state, {
          cameraId,
          frameIdx,
          before,
          after: clonePoints(entry.payload.points)
        });
      });
    },

    async saveFrame(frameIdx: number, cameraIds?: string[]) {
      const actorId = get().selectedActor;
      const sequenceId = get().selectedSequence;
      if (!actorId || !sequenceId) return;

      const candidateCameras = cameraIds?.length ? cameraIds : get().selectedCameras;
      const targetCameraIds = candidateCameras.filter((cameraId) => {
        const key = keyForFrame(cameraId, frameIdx);
        return Boolean(get().frameCache[key]);
      });
      if (!targetCameraIds.length) return;

      const changedPointIndices = get().changedPointIndices;
      // Only send the per-camera changes for the cameras actually being saved.
      const allChangedByCamera = get().changedPointsByCamera;
      const changedByCamera: Record<string, number[]> = {};
      targetCameraIds.forEach((cameraId) => {
        if (allChangedByCamera[cameraId]?.length) {
          changedByCamera[cameraId] = allChangedByCamera[cameraId];
        }
      });
      set((state) => {
        state.isSaving = true;
      });

      const payloads: LandmarkPayload[] = targetCameraIds.map((cameraId) => {
        const key = keyForFrame(cameraId, frameIdx);
        const entry = get().frameCache[key];
        if (!entry) {
          throw new Error(`Frame not loaded for ${cameraId}:${frameIdx}`);
        }
        return {
          actor_id: actorId,
          sequence_id: sequenceId,
          camera_id: cameraId,
          frame_idx: frameIdx,
          points: entry.payload.points
        };
      });

      try {
        await api.saveLandmarks(payloads, changedPointIndices, changedByCamera);
        set((state) => {
          targetCameraIds.forEach((cameraId) => {
            const key = keyForFrame(cameraId, frameIdx);
            const target = state.frameCache[key];
            if (target) target.dirty = false;
            const propKey = propagationKey(actorId, sequenceId, cameraId);
            if (!state.propagatingCameras.includes(propKey)) {
              state.propagatingCameras.push(propKey);
            }
          });
          if (targetCameraIds.length === 1) {
            state.statusMessage = `Saved frame ${frameIdx} for ${targetCameraIds[0]}. Propagating...`;
          } else {
            state.statusMessage = `Saved frame ${frameIdx} for ${targetCameraIds.length} cameras. Propagating...`;
          }
        });
      } catch (error) {
        set((state) => {
          state.statusMessage = error instanceof Error ? error.message : String(error);
        });
      } finally {
        set((state) => {
          state.isSaving = false;
          state.changedPointIndices = [];
          state.changedPointsByCamera = {};
        });
      }
    },

    async saveNeutralFrame() {
      const actorId = get().selectedActor;
      const sequenceId = get().selectedSequence;
      if (!actorId || !sequenceId) return;
      const frameIdx = get().currentFrame;
      try {
        const response = await api.saveNeutralFrame(actorId, sequenceId, frameIdx);
        set((state) => {
          state.statusMessage = `Saved frame ${frameIdx} as neutral (raw frame ${response.frame_idx}) for actor ${actorId} (from ${sequenceId})`;
          if (response.status === "confirmed") {
            state.neutralFrameSequenceId = sequenceId;
            state.neutralFrameIdx = frameIdx;
            state.toast = {
              id: Date.now(),
              tone: "success",
              message: `Neutral frame saved for actor ${actorId} (${sequenceId} frame ${frameIdx})`
            };
            // Mirrors the backend: a (re)saved actor-level neutral frame
            // invalidates every sequence's prior propagated landmarks, not
            // just the one it was saved from -- reset all of this actor's
            // sequences back down to "unlabeled".
            const actor = state.dataset?.actors.find((a) => a.actor_id === actorId);
            if (actor) {
              for (const sequence of actor.sequences) {
                sequence.label_status = "unlabeled";
              }
            }
          }
        });
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        set((state) => {
          state.statusMessage = message;
          state.toast = {
            id: Date.now(),
            tone: "error",
            message: `Failed to save neutral frame: ${message}`
          };
        });
      }
    },

    async fetchNeutralFrame(actorId: string, sequenceId: string) {
      // sequenceId is only forwarded for the API call's own URL shape -- the
      // response is this actor's ONE neutral frame (with whichever sequence
      // it actually came from), the same regardless of sequenceId.
      try {
        const response = await api.getNeutralFrame(actorId, sequenceId);
        set((state) => {
          state.neutralFrameSequenceId = response.sequence_id;
          state.neutralFrameIdx = response.frame_idx;
        });
      } catch {
        set((state) => {
          state.neutralFrameSequenceId = null;
          state.neutralFrameIdx = null;
        });
      }
    },

    async confirmLabeled(actorId: string, sequenceId: string) {
      const key = labelActionKey(actorId, sequenceId);
      if (get().pendingLabelActions.includes(key)) return;
      set((state) => {
        state.pendingLabelActions.push(key);
      });
      try {
        const response = await api.confirmLabeled(actorId, sequenceId);
        set((state) => {
          const sequence = findSequence(state, actorId, sequenceId);
          if (sequence) sequence.label_status = response.status;
          state.toast = {
            id: Date.now(),
            tone: "success",
            message: `${actorId}/${sequenceId} confirmed labeled`
          };
        });
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        set((state) => {
          state.toast = { id: Date.now(), tone: "error", message: `Failed to confirm labeled: ${message}` };
        });
      } finally {
        set((state) => {
          state.pendingLabelActions = state.pendingLabelActions.filter((existing) => existing !== key);
        });
      }
    },

    async confirm(actorId: string, sequenceId: string) {
      const key = labelActionKey(actorId, sequenceId);
      if (get().pendingLabelActions.includes(key)) return;
      set((state) => {
        state.pendingLabelActions.push(key);
      });
      try {
        const response = await api.confirm(actorId, sequenceId);
        set((state) => {
          const sequence = findSequence(state, actorId, sequenceId);
          if (sequence) sequence.label_status = response.status;
          state.toast = {
            id: Date.now(),
            tone: "success",
            message: `${actorId}/${sequenceId} confirmed`
          };
        });
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        set((state) => {
          state.toast = { id: Date.now(), tone: "error", message: `Failed to confirm: ${message}` };
        });
      } finally {
        set((state) => {
          state.pendingLabelActions = state.pendingLabelActions.filter((existing) => existing !== key);
        });
      }
    },

    async unconfirmLabeled(actorId: string, sequenceId: string) {
      const key = labelActionKey(actorId, sequenceId);
      if (get().pendingLabelActions.includes(key)) return;
      set((state) => {
        state.pendingLabelActions.push(key);
      });
      try {
        const response = await api.unconfirmLabeled(actorId, sequenceId);
        set((state) => {
          const sequence = findSequence(state, actorId, sequenceId);
          if (sequence) sequence.label_status = response.status;
          state.toast = {
            id: Date.now(),
            tone: "success",
            message: `${actorId}/${sequenceId} unconfirmed labeled`
          };
        });
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        set((state) => {
          state.toast = { id: Date.now(), tone: "error", message: `Failed to unconfirm labeled: ${message}` };
        });
      } finally {
        set((state) => {
          state.pendingLabelActions = state.pendingLabelActions.filter((existing) => existing !== key);
        });
      }
    },

    async unconfirm(actorId: string, sequenceId: string) {
      const key = labelActionKey(actorId, sequenceId);
      if (get().pendingLabelActions.includes(key)) return;
      set((state) => {
        state.pendingLabelActions.push(key);
      });
      try {
        const response = await api.unconfirm(actorId, sequenceId);
        set((state) => {
          const sequence = findSequence(state, actorId, sequenceId);
          if (sequence) sequence.label_status = response.status;
          state.toast = {
            id: Date.now(),
            tone: "success",
            message: `${actorId}/${sequenceId} unconfirmed`
          };
        });
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        set((state) => {
          state.toast = { id: Date.now(), tone: "error", message: `Failed to unconfirm: ${message}` };
        });
      } finally {
        set((state) => {
          state.pendingLabelActions = state.pendingLabelActions.filter((existing) => existing !== key);
        });
      }
    },

    async unconfirmConfirmed(actorId: string, sequenceId: string) {
      const key = labelActionKey(actorId, sequenceId);
      if (get().pendingLabelActions.includes(key)) return;
      set((state) => {
        state.pendingLabelActions.push(key);
      });
      try {
        const response = await api.unconfirmConfirmed(actorId, sequenceId);
        set((state) => {
          const sequence = findSequence(state, actorId, sequenceId);
          if (sequence) sequence.label_status = response.status;
          state.toast = {
            id: Date.now(),
            tone: "success",
            message: `${actorId}/${sequenceId} reverted to unconfirmed`
          };
        });
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        set((state) => {
          state.toast = { id: Date.now(), tone: "error", message: `Failed to unconfirm: ${message}` };
        });
      } finally {
        set((state) => {
          state.pendingLabelActions = state.pendingLabelActions.filter((existing) => existing !== key);
        });
      }
    },

    async confirmFacescape(actorId: string, expressionId: string) {
      const key = facescapeLabelActionKey(actorId, expressionId);
      if (get().pendingLabelActions.includes(key)) return;
      set((state) => {
        state.pendingLabelActions.push(key);
      });
      try {
        const response = await api.confirm(actorId, expressionId, "FaceScape");
        set((state) => {
          const expression = findFacescapeExpression(state, actorId, expressionId);
          if (expression) expression.label_status = response.status;
          state.toast = {
            id: Date.now(),
            tone: "success",
            message: `${actorId}/${expressionId} confirmed`
          };
        });
        // Backend also confirms every other "unconfirmed" pair of this actor
        // (see /labels/confirm's own note) -- refetch the whole actor's
        // statuses so those siblings flip to "confirmed" here too, not just
        // the one expression just clicked.
        await get().refreshFacescapeActorStatus(actorId, { silent: true });
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        set((state) => {
          state.toast = { id: Date.now(), tone: "error", message: `Failed to confirm: ${message}` };
        });
      } finally {
        set((state) => {
          state.pendingLabelActions = state.pendingLabelActions.filter((existing) => existing !== key);
        });
      }
    },

    async unconfirmFacescape(actorId: string, expressionId: string) {
      const key = facescapeLabelActionKey(actorId, expressionId);
      if (get().pendingLabelActions.includes(key)) return;
      set((state) => {
        state.pendingLabelActions.push(key);
      });
      try {
        const response = await api.unconfirm(actorId, expressionId, "FaceScape");
        set((state) => {
          const expression = findFacescapeExpression(state, actorId, expressionId);
          if (expression) expression.label_status = response.status;
          state.toast = {
            id: Date.now(),
            tone: "success",
            message: `${actorId}/${expressionId} unconfirmed`
          };
        });
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        set((state) => {
          state.toast = { id: Date.now(), tone: "error", message: `Failed to unconfirm: ${message}` };
        });
      } finally {
        set((state) => {
          state.pendingLabelActions = state.pendingLabelActions.filter((existing) => existing !== key);
        });
      }
    },

    async unconfirmConfirmedFacescape(actorId: string, expressionId: string) {
      const key = facescapeLabelActionKey(actorId, expressionId);
      if (get().pendingLabelActions.includes(key)) return;
      set((state) => {
        state.pendingLabelActions.push(key);
      });
      try {
        // Backend also deletes final_wrapped_data/facescape_<actor>/<expression>.pt
        // as part of this call -- see labels.py's /unconfirm-confirmed handler.
        const response = await api.unconfirmConfirmed(actorId, expressionId, "FaceScape");
        set((state) => {
          const expression = findFacescapeExpression(state, actorId, expressionId);
          if (expression) expression.label_status = response.status;
          state.toast = {
            id: Date.now(),
            tone: "success",
            message: `${actorId}/${expressionId} reverted to unconfirmed`
          };
        });
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        set((state) => {
          state.toast = { id: Date.now(), tone: "error", message: `Failed to unconfirm: ${message}` };
        });
      } finally {
        set((state) => {
          state.pendingLabelActions = state.pendingLabelActions.filter((existing) => existing !== key);
        });
      }
    },

    startFacescapeEarAddMode(ear: "left" | "right") {
      // Only switches which side new clicks record into -- does NOT clear
      // either side's already-recorded clicks, so hopping between "Add Left"
      // and "Add Right" (or coming back to one later) keeps prior progress.
      set((state) => {
        state.facescapeEarAddMode = ear;
      });
    },

    cancelFacescapeEarAddMode() {
      // Only exits add-mode (so clicking the image stops recording) --
      // deliberately does NOT clear facescapeEarClicksLeft/Right, so
      // "Cancel Mode" can be used to pause without losing progress. Use
      // clearFacescapeEarClicks() to actually discard recorded clicks.
      set((state) => {
        state.facescapeEarAddMode = null;
      });
    },

    clearFacescapeEarClicks() {
      set((state) => {
        state.facescapeEarClicksLeft = {};
        state.facescapeEarClicksRight = {};
      });
    },

    recordFacescapeEarClick(cameraId: string, x: number, y: number) {
      set((state) => {
        if (state.facescapeEarAddMode === "left") {
          state.facescapeEarClicksLeft[cameraId] = { x, y };
        } else if (state.facescapeEarAddMode === "right") {
          state.facescapeEarClicksRight[cameraId] = { x, y };
        }
      });
    },

    async saveFacescapeEarPoints() {
      const { selectedFacescapeActor, selectedFacescapeExpression, facescapeEarClicksLeft, facescapeEarClicksRight } = get();
      if (!selectedFacescapeActor || !selectedFacescapeExpression) return;

      const sides: Array<["left" | "right", Record<string, { x: number; y: number }>]> = [
        ["left", facescapeEarClicksLeft],
        ["right", facescapeEarClicksRight]
      ].filter(([, clicks]) => Object.keys(clicks).length >= 2) as Array<
        ["left" | "right", Record<string, { x: number; y: number }>]
      >;
      if (sides.length === 0) return;

      // Actor-scoped (not actor+expression): the backend blocks the whole
      // request until it's also propagated this point to every OTHER
      // expression for this actor (see facescape_ear_propagation.py), so
      // this actor is "busy" for the whole duration, not just this one
      // expression -- a different actor's Save is never blocked by it.
      const key = `facescape-ear:${selectedFacescapeActor}`;
      if (get().pendingLabelActions.includes(key)) return;
      set((state) => {
        state.pendingLabelActions.push(key);
      });
      try {
        const savedSides: string[] = [];
        const failedSides: string[] = [];
        let totalCameras = 0;
        for (const [ear, clicks] of sides) {
          try {
            const response = await api.saveFacescapeEarPoint(selectedFacescapeActor, selectedFacescapeExpression, ear, clicks);
            totalCameras += response.cameras_written;
            savedSides.push(ear);
            set((state) => {
              if (ear === "left") state.facescapeEarClicksLeft = {};
              else state.facescapeEarClicksRight = {};
            });
          } catch (error) {
            failedSides.push(ear);
            const message = error instanceof Error ? error.message : String(error);
            console.error(`Failed to save ${ear} ear point:`, message);
          }
        }

        if (savedSides.length > 0) {
          // Re-fetch every camera already cached for this pair -- the backend
          // just (re)projected the saved point(s) into all of them, so their
          // cached LandmarkFrame is now stale at index 2/9 (see
          // facescape_ear_points.py's save_ear_point).
          const prefix = `${selectedFacescapeActor}:${selectedFacescapeExpression}:`;
          const cameraIds = Object.keys(get().facescapeLandmarkCache)
            .filter((cacheKey) => cacheKey.startsWith(prefix))
            .map((cacheKey) => cacheKey.slice(prefix.length));
          await Promise.all(
            cameraIds.map(async (cameraId) => {
              const frame = await api.getFaceScapeLandmarks(selectedFacescapeActor, selectedFacescapeExpression, cameraId);
              set((state) => {
                state.facescapeLandmarkCache[`${prefix}${cameraId}`] = frame;
              });
            })
          );
        }

        set((state) => {
          state.facescapeEarAddMode = null;
          if (failedSides.length === 0) {
            state.toast = {
              id: Date.now(),
              tone: "success",
              message: `Saved ${savedSides.join(" + ")} ear top for ${selectedFacescapeActor}/${selectedFacescapeExpression} `
                + `(${totalCameras} camera(s))`
            };
          } else {
            state.toast = {
              id: Date.now(),
              tone: "error",
              message: `Saved ${savedSides.join(", ") || "none"}; failed to save ${failedSides.join(", ")} -- see console`
            };
          }
        });
      } finally {
        set((state) => {
          state.pendingLabelActions = state.pendingLabelActions.filter((existing) => existing !== key);
        });
      }
    },

    async saveFacescapeNeutralExpression() {
      const { selectedFacescapeActor, selectedFacescapeExpression } = get();
      if (!selectedFacescapeActor || !selectedFacescapeExpression) return;
      try {
        const response = await api.saveFacescapeNeutralExpression(selectedFacescapeActor, selectedFacescapeExpression);
        set((state) => {
          state.facescapeNeutralExpression = response.expression_id;
          state.toast = {
            id: Date.now(),
            tone: "success",
            message: `Neutral expression saved for actor ${selectedFacescapeActor} (${response.expression_id})`
          };
        });
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        set((state) => {
          state.toast = { id: Date.now(), tone: "error", message: `Failed to save neutral expression: ${message}` };
        });
      }
    },

    async fetchFacescapeNeutralExpression(actorId: string) {
      try {
        const response = await api.getFacescapeNeutralExpression(actorId);
        set((state) => {
          state.facescapeNeutralExpression = response.expression_id;
        });
      } catch {
        set((state) => {
          state.facescapeNeutralExpression = null;
        });
      }
    },

    async refreshFacescapeActorStatus(actorId: string, options?: { silent?: boolean }) {
      const silent = options?.silent ?? false;
      if (get().refreshingFacescapeActors.includes(actorId)) return;
      set((state) => {
        state.refreshingFacescapeActors.push(actorId);
      });
      try {
        const response = await api.getActorStatus(actorId, "FaceScape");
        const statuses = response.statuses[actorId] ?? {};
        set((state) => {
          const actor = state.facescapeDataset?.actors.find((a) => a.actor_id === actorId);
          if (actor) {
            for (const expression of actor.expressions) {
              const status = statuses[expression.expression_id];
              if (status) expression.label_status = status;
            }
          }
          if (!silent) {
            state.toast = {
              id: Date.now(),
              tone: "success",
              message: `${actorId}: statuses updated`
            };
          }
        });
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        if (!silent) {
          set((state) => {
            state.toast = { id: Date.now(), tone: "error", message: `Failed to update ${actorId}: ${message}` };
          });
        }
        if (silent) throw error;
      } finally {
        set((state) => {
          state.refreshingFacescapeActors = state.refreshingFacescapeActors.filter((existing) => existing !== actorId);
        });
      }
    },

    async updateAllActorStatuses() {
      if (get().isUpdatingAll) return;
      const datasetType = get().activeDatasetType;
      const actorIds = datasetType === "FaceScape"
        ? (get().facescapeDataset?.actors ?? []).map((actor) => actor.actor_id)
        : (get().dataset?.actors ?? []).map((actor) => actor.actor_id);
      if (actorIds.length === 0) return;

      set((state) => {
        state.isUpdatingAll = true;
      });

      let succeeded = 0;
      let failed = 0;
      try {
        const refreshOne = datasetType === "FaceScape" ? get().refreshFacescapeActorStatus : get().refreshActorStatus;
        let nextIndex = 0;
        const worker = async () => {
          while (nextIndex < actorIds.length) {
            const actorId = actorIds[nextIndex++];
            try {
              await refreshOne(actorId, { silent: true });
              succeeded += 1;
            } catch {
              failed += 1;
            }
          }
        };
        const workerCount = Math.min(MAX_UPDATE_ALL_CONCURRENCY, actorIds.length);
        await Promise.all(Array.from({ length: workerCount }, worker));
      } finally {
        set((state) => {
          state.isUpdatingAll = false;
          state.toast = {
            id: Date.now(),
            tone: failed > 0 ? "error" : "success",
            message: failed > 0
              ? `Updated ${succeeded}/${actorIds.length} ${datasetType} actors (${failed} failed)`
              : `Updated all ${succeeded} ${datasetType} actor${succeeded === 1 ? "" : "s"}`
          };
        });
      }
    },

    dismissToast(id: number) {
      set((state) => {
        if (state.toast?.id === id) {
          state.toast = undefined;
        }
      });
    },

    async triggerPreprocess(actorId: string, sequenceId: string, cameras?: string[]) {
      try {
        const job = await api.startPreprocess(actorId, sequenceId, cameras);
        set((state) => {
          state.statusMessage = `Preprocess job ${job.job_id} accepted`;
        });
      } catch (error) {
        set((state) => {
          state.statusMessage = error instanceof Error ? error.message : String(error);
        });
      }
    },

    undo() {
      set((state) => {
        const record = state.history.pop();
        if (!record) return;
        const key = keyForFrame(record.cameraId, record.frameIdx);
        const entry = state.frameCache[key];
        if (!entry) return;
        entry.payload.points = clonePoints(record.before);
        entry.dirty = true;
        state.redoStack.push(record);
        state.isPlaying = false;
      });
    },

    redo() {
      set((state) => {
        const record = state.redoStack.pop();
        if (!record) return;
        const key = keyForFrame(record.cameraId, record.frameIdx);
        const entry = state.frameCache[key];
        if (!entry) return;
        entry.payload.points = clonePoints(record.after);
        entry.dirty = true;
        appendHistory(state, record);
        state.isPlaying = false;
      });
    },

    goToNextUnlabeledSequence() {
      const dataset = get().dataset;
      if (!dataset) return;
      const flattened = dataset.actors.flatMap((actor) =>
        actor.sequences.map((sequence) => ({ actorId: actor.actor_id, sequence }))
      );
      if (!flattened.length) return;
      const currentActor = get().selectedActor;
      const currentSequence = get().selectedSequence;
      const currentIndex = flattened.findIndex((item) =>
        item.actorId === currentActor && item.sequence.sequence_id === currentSequence
      );
      const ordered = currentIndex >= 0
        ? [...flattened.slice(currentIndex + 1), ...flattened.slice(0, currentIndex + 1)]
        : flattened;
      const target = ordered.find((item) => !item.sequence.has_landmarks);
      if (!target) {
        set((state) => {
          state.statusMessage = "No unlabeled sequences remaining";
        });
        return;
      }
      get().selectSequence(target.actorId, target.sequence.sequence_id);
      set((state) => {
        state.statusMessage = `Switched to ${target.actorId} / ${target.sequence.sequence_id}`;
      });
    },

    setActiveDatasetType(datasetType: DatasetType) {
      set((state) => {
        state.activeDatasetType = datasetType;
      });
      if (datasetType === "FaceScape" && !get().facescapeDataset && !get().isFacescapeDatasetLoading) {
        void get().fetchFacescapeDataset();
      }
    },

    async fetchFacescapeDataset() {
      set((state) => {
        state.isFacescapeDatasetLoading = true;
        state.facescapeExplorerError = undefined;
      });
      try {
        const dataset = await api.getFaceScapeDataset();
        set((state) => {
          state.facescapeDataset = dataset;
          if (dataset.actors.length && (!state.selectedFacescapeActor || !state.selectedFacescapeExpression)) {
            const firstActor = dataset.actors[0];
            const firstExpression = firstActor.expressions[0];
            state.selectedFacescapeActor = firstActor.actor_id;
            state.selectedFacescapeExpression = firstExpression?.expression_id;
            state.selectedFacescapeCameras = firstExpression ? defaultFacescapeCameras(firstExpression) : [];
          }
        });
      } catch (error) {
        set((state) => {
          state.facescapeExplorerError = error instanceof Error ? error.message : String(error);
        });
      } finally {
        set((state) => {
          state.isFacescapeDatasetLoading = false;
        });
      }
    },

    selectFacescapeExpression(actorId: string, expressionId: string) {
      set((state) => {
        state.selectedFacescapeActor = actorId;
        state.selectedFacescapeExpression = expressionId;
        state.selectedFacescapeCameras = [];
        state.facescapeLandmarkCache = {};
        state.facescapeLandmarkLoadStatus = {};
        state.facescapeSkinLandmarkCache = {};
        state.facescapeSkinLandmarkLoadStatus = {};
      });
      const expression = get().facescapeDataset?.actors
        .find((actor) => actor.actor_id === actorId)?.expressions
        .find((expr) => expr.expression_id === expressionId);
      if (expression) {
        set((state) => {
          state.selectedFacescapeCameras = defaultFacescapeCameras(expression);
        });
      }
    },

    toggleFacescapeCamera(cameraId: string) {
      set((state) => {
        if (!state.selectedFacescapeCameras.includes(cameraId)) {
          state.selectedFacescapeCameras.push(cameraId);
        } else if (state.selectedFacescapeCameras.length > 1) {
          state.selectedFacescapeCameras = state.selectedFacescapeCameras.filter((id) => id !== cameraId);
        }
      });
    },

    async loadFacescapeLandmarks(actorId: string, expressionId: string, cameraId: string) {
      const key = `${actorId}:${expressionId}:${cameraId}`;
      const status = get().facescapeLandmarkLoadStatus[key];
      if (status === "loading" || status === "loaded") return;

      set((state) => {
        state.facescapeLandmarkLoadStatus[key] = "loading";
      });
      try {
        const frame = await api.getFaceScapeLandmarks(actorId, expressionId, cameraId);
        set((state) => {
          state.facescapeLandmarkCache[key] = frame;
          state.facescapeLandmarkLoadStatus[key] = "loaded";
        });
      } catch (error) {
        set((state) => {
          state.facescapeLandmarkLoadStatus[key] = "error";
          state.statusMessage = error instanceof Error ? error.message : String(error);
        });
      }
    },

    async loadFacescapeSkinLandmarks(actorId: string, expressionId: string, cameraId: string) {
      const key = `${actorId}:${expressionId}:${cameraId}`;
      const status = get().facescapeSkinLandmarkLoadStatus[key];
      if (status === "loading" || status === "loaded") return;

      set((state) => {
        state.facescapeSkinLandmarkLoadStatus[key] = "loading";
      });
      try {
        const frame = await api.getFaceScapeSkinLandmarks(actorId, expressionId, cameraId);
        set((state) => {
          state.facescapeSkinLandmarkCache[key] = frame;
          state.facescapeSkinLandmarkLoadStatus[key] = "loaded";
        });
      } catch (error) {
        set((state) => {
          state.facescapeSkinLandmarkLoadStatus[key] = "error";
          state.statusMessage = error instanceof Error ? error.message : String(error);
        });
      }
    },

    handleSocketEvent(event: Record<string, unknown>) {
      const eventName = event.event as string | undefined;
      
      if (eventName === "propagation_complete") {
        const eventActorId = event.actor_id as string;
        const cameraId = event.camera_id as string;
        // The WS channel is subscribed by sequence_id alone (reused identically
        // across actors), so this event may belong to a different actor's
        // identically-named sequence/camera -- always clear its own
        // propagationKey below, but only touch frameCache/cameraLoadStatus (which
        // are keyed by camera_id alone, no actor) when it's for the actor
        // currently on screen, to avoid evicting an unrelated actor's cache.
        const propKey = propagationKey(eventActorId, event.sequence_id as string, cameraId);
        const isForCurrentActor =
          eventActorId === get().selectedActor && event.sequence_id === get().selectedSequence;
        // The edited frame itself is included: the backend reprojects onto
        // passive (selected-but-unedited) cameras there too, not just frame_idx+1
        // onward, so their cache needs invalidating at the edit frame as well.
        const editFrame = event.frame_idx as number;
        const window = event.window as number;
        const endFrame = editFrame + 1 + window;
        const currentFrame = get().currentFrame;

        if (isForCurrentActor) {
          set((state) => {
            state.statusMessage = `Propagation complete for camera ${cameraId}`;

            // Invalidate cache for propagated frames, BUT skip current frame to avoid flicker
            for (let i = editFrame; i < endFrame; i++) {
               if (i === currentFrame) continue;
               const key = keyForFrame(cameraId, i);
               if (state.frameCache[key]) {
                 delete state.frameCache[key];
               }
            }
            // Force reload when needed
            state.cameraLoadStatus[cameraId] = "idle";
          });
        }

        // If current frame was affected, force reload it
        if (isForCurrentActor && currentFrame >= editFrame && currentFrame < endFrame) {
           get().loadFrame(cameraId, currentFrame, { force: true }).finally(() => {
             set((state) => {
               state.propagatingCameras = state.propagatingCameras.filter((key) => key !== propKey);
             });
           });
        } else {
           set((state) => {
             state.propagatingCameras = state.propagatingCameras.filter((key) => key !== propKey);
           });
        }

      } else if (eventName === "propagation_failed") {
        const eventActorId = event.actor_id as string;
        const cameraId = event.camera_id as string;
        const propKey = propagationKey(eventActorId, event.sequence_id as string, cameraId);
        set((state) => {
          if (eventActorId === get().selectedActor && event.sequence_id === get().selectedSequence) {
            state.statusMessage = `Propagation failed: ${event.error as string}`;
          }
          state.propagatingCameras = state.propagatingCameras.filter((key) => key !== propKey);
        });
      }
    }
  });

// Cameras shown by default for a FaceScape pair: every camera with an image,
// except an invalid pair (its own full raw rig, ~66 cameras) once wrapped --
// then just its three overlay cameras; the rest stay one click away.
function defaultFacescapeCameras(expression: FaceScapeExpressionDescriptor): string[] {
  if (expression.is_invalid && expression.overlay_camera_ids.length) {
    return expression.overlay_camera_ids;
  }
  return expression.cameras.filter((cam) => cam.has_image).map((cam) => cam.camera_id);
}

export const useAnnotationStore = create<AnnotationState>()(immer(storeCreator));
