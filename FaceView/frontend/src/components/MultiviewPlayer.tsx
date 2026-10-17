import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { DEFAULT_FPS, MEDIA_BASE } from "@/config";
import { api } from "@/lib/api";
import type { LandmarkFrame } from "@/lib/types";
import { propagationKey, useAnnotationStore } from "@/state/useAnnotationStore";
import { LandmarkOverlay } from "./LandmarkOverlay";
import { ImageTile } from "./ImageTile";
import { Ava256Player } from "./Ava256Player";

import * as MP4Box from "mp4box";
import type { Movie, MP4BoxBuffer, Track } from "mp4box";

interface VideoMeta {
  fps: number;
  numFrames: number;
}

function getVideoMetaFromSrc(videoSrc: string): Promise<VideoMeta | null> {
  return new Promise((resolve, reject) => {
    const mp4boxFile = MP4Box.createFile();

    mp4boxFile.onError = (error: string) => {
      reject(new Error(`mp4box parse error: ${error}`));
    };

    mp4boxFile.onReady = (info: Movie) => {
      const videoTrack = info.videoTracks?.[0] as Track | undefined;
      if (!videoTrack) {
        resolve(null);
        return;
      }
      // nb_samples / duration (in track timescale units) = fps. nb_samples is also
      // the video's true frame count — the authoritative source for playback range,
      // since landmark file count can be much sparser than the actual video.
      const durationSeconds = videoTrack.duration / videoTrack.timescale;
      const fps = durationSeconds > 0 ? videoTrack.nb_samples / durationSeconds : null;
      if (fps === null || !videoTrack.nb_samples) {
        resolve(null);
        return;
      }
      resolve({ fps, numFrames: videoTrack.nb_samples });
    };

    fetch(videoSrc)
      .then((res) => res.arrayBuffer())
      .then((buffer) => {
        const mp4Buffer = buffer as MP4BoxBuffer;
        mp4Buffer.fileStart = 0;
        mp4boxFile.appendBuffer(mp4Buffer);
        mp4boxFile.flush();
      })
      .catch(reject);
  });
}

function useVideoMeta(videoSrc?: string): VideoMeta | null {
  const [meta, setMeta] = useState<VideoMeta | null>(null);

  useEffect(() => {
    if (!videoSrc) {
      setMeta(null);
      return;
    }
    let cancelled = false;

    getVideoMetaFromSrc(videoSrc)
      .then((result) => {
        if (!cancelled) setMeta(result);
      })
      .catch((err) => {
        console.warn("video meta parse failed:", err);
        if (!cancelled) setMeta(null);
      });

    return () => {
      cancelled = true;
    };
  }, [videoSrc]); // only re-runs when videoSrc actually changes

  return meta;
}

export function MultiviewPlayer() {
  const activeDatasetType = useAnnotationStore((state) => state.activeDatasetType);
  if (activeDatasetType === "FaceScape") {
    return <FaceScapePlayer />;
  }
  if (activeDatasetType === "Ava256") {
    return <Ava256Player />;
  }
  return <NersembleMultiviewPlayer />;
}

function NersembleMultiviewPlayer() {
  const {
    dataset,
    selectedActor,
    selectedSequence,
    selectedCameras,
    currentFrame,
    neutralFrameSequenceId,
    neutralFrameIdx,
    setFrame,
    getAdjacentAvailableFrame,
    frameCache,
    cameraFrameCounts,
    cameraVideoMeta,
    setCameraVideoMeta,
    loadFrame,
    loadSkinFrames,
    skinFrameCache,
    isPlaying,
    togglePlayback,
    setPlaying,
    saveFrame,
    saveNeutralFrame,
    fetchNeutralFrame,
    undo,
    redo,
    goToNextUnlabeledSequence,
    propagatingCameras
  } = useAnnotationStore((state) => ({
    dataset: state.dataset,
    selectedActor: state.selectedActor,
    selectedSequence: state.selectedSequence,
    selectedCameras: state.selectedCameras,
    currentFrame: state.currentFrame,
    neutralFrameSequenceId: state.neutralFrameSequenceId,
    neutralFrameIdx: state.neutralFrameIdx,
    setFrame: state.setFrame,
    getAdjacentAvailableFrame: state.getAdjacentAvailableFrame,
    frameCache: state.frameCache,
    cameraFrameCounts: state.cameraFrameCounts,
    cameraVideoMeta: state.cameraVideoMeta,
    setCameraVideoMeta: state.setCameraVideoMeta,
    loadFrame: state.loadFrame,
    loadSkinFrames: state.loadSkinFrames,
    skinFrameCache: state.skinFrameCache,
    isPlaying: state.isPlaying,
    togglePlayback: state.togglePlayback,
    setPlaying: state.setPlaying,
    saveFrame: state.saveFrame,
    saveNeutralFrame: state.saveNeutralFrame,
    fetchNeutralFrame: state.fetchNeutralFrame,
    undo: state.undo,
    redo: state.redo,
    goToNextUnlabeledSequence: state.goToNextUnlabeledSequence,
    propagatingCameras: state.propagatingCameras
  }));
  const [showIndices, setShowIndices] = useState(false);
  const [showSkinLandmarks, setShowSkinLandmarks] = useState(false);
  const frameRef = useRef(currentFrame);

  useEffect(() => {
    selectedCameras.forEach((cameraId) => {
      void loadFrame(cameraId, currentFrame);
    });
  }, [selectedCameras, currentFrame, loadFrame]);

  useEffect(() => {
    if (!showSkinLandmarks) return;
    selectedCameras.forEach((cameraId) => {
      void loadSkinFrames(cameraId);
    });
  }, [showSkinLandmarks, selectedCameras, loadSkinFrames]);

  useEffect(() => {
    if (selectedActor && selectedSequence) {
      void fetchNeutralFrame(selectedActor, selectedSequence);
    }
  }, [selectedActor, selectedSequence, fetchNeutralFrame]);

  useEffect(() => {
    frameRef.current = currentFrame;
  }, [currentFrame]);

  const sequence = useMemo(() => {
    return dataset?.actors
      .find((actor) => actor.actor_id === selectedActor)?.sequences
      .find((seq) => seq.sequence_id === selectedSequence);
  }, [dataset, selectedActor, selectedSequence]);

  const resolvedFrameCount = useMemo(() => {
    if (!sequence || !selectedCameras.length) {
      return sequence?.cameras[0]?.video.num_frames ?? 500;
    }
    const primaryCameraId = selectedCameras[0];
    // Real frame count measured from the delegation video's own container
    // metadata — the authoritative source. Landmark file count is NOT a valid
    // proxy for this: it can be much sparser than the actual video (e.g. only
    // every 5th frame kept), which would wrongly cap scrubbing/seeking range.
    const measuredCount = cameraVideoMeta[primaryCameraId]?.numFrames;
    if (measuredCount && measuredCount > 0) {
      return measuredCount;
    }
    const cameraMeta = sequence.cameras.find((camera) => camera.camera_id === primaryCameraId);
    if (cameraMeta?.video.num_frames) {
      return cameraMeta.video.num_frames;
    }
    const loadedCount = cameraFrameCounts[primaryCameraId];
    if (loadedCount && loadedCount > 0) {
      return loadedCount;
    }
    return sequence.cameras[0]?.video.num_frames ?? 500;
  }, [sequence, selectedCameras, cameraVideoMeta, cameraFrameCounts]);

  const maxFrameIndex = Math.max(0, Math.floor(resolvedFrameCount) - 1);
  const fps = cameraVideoMeta[selectedCameras[0]]?.fps
    ?? sequence?.cameras[0]?.video.fps
    ?? DEFAULT_FPS;

  useEffect(() => {
    if (currentFrame > maxFrameIndex) {
      setFrame(maxFrameIndex);
    }
  }, [currentFrame, maxFrameIndex, setFrame]);

  useEffect(() => {
    if (!isPlaying) return;
    let animationId: number;
    const startTime = performance.now();
    const startFrame = frameRef.current;

    const tick = () => {
      const now = performance.now();
      const elapsed = (now - startTime) / 1000;
      const nextFrame = Math.min(maxFrameIndex, Math.floor(startFrame + elapsed * fps));

      if (nextFrame > maxFrameIndex) {
        setFrame(maxFrameIndex);
        setPlaying(false);
        return;
      }

      if (nextFrame !== frameRef.current) {
        setFrame(nextFrame);
      }
      animationId = requestAnimationFrame(tick);
    };
    animationId = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(animationId);
  }, [isPlaying, fps, maxFrameIndex, setFrame, setPlaying]);

  useEffect(() => {
    if (isPlaying && currentFrame >= maxFrameIndex) {
      setPlaying(false);
    }
  }, [currentFrame, isPlaying, maxFrameIndex, setPlaying]);

  useEffect(() => {
    if (!selectedCameras.length && isPlaying) {
      setPlaying(false);
    }
  }, [selectedCameras.length, isPlaying, setPlaying]);

  const clampFrame = useCallback((value: number) => {
    if (!Number.isFinite(value)) return 0;
    return Math.min(maxFrameIndex, Math.max(0, Math.floor(value)));
  }, [maxFrameIndex]);

  const handlePrevFrame = useCallback(() => {
    setPlaying(false);
    const primaryCamera = selectedCameras[0];
    const prev = primaryCamera
      ? getAdjacentAvailableFrame(primaryCamera, currentFrame, -1)
      : currentFrame - 1;
    setFrame(clampFrame(prev));
  }, [clampFrame, currentFrame, selectedCameras, getAdjacentAvailableFrame, setFrame, setPlaying]);

  const handleNextFrame = useCallback(() => {
    setPlaying(false);
    const primaryCamera = selectedCameras[0];
    const next = primaryCamera
      ? getAdjacentAvailableFrame(primaryCamera, currentFrame, 1)
      : currentFrame + 1;
    setFrame(clampFrame(next));
  }, [clampFrame, currentFrame, selectedCameras, getAdjacentAvailableFrame, setFrame, setPlaying]);

  const handleSliderChange = (value: number) => {
    setPlaying(false);
    setFrame(clampFrame(value));
  };

  const handlePlayToggle = () => {
    if (!selectedCameras.length) return;
    togglePlayback();
  };

  const handleSaveAll = useCallback(() => {
    if (!selectedCameras.length) return;
    void saveFrame(currentFrame, selectedCameras);
  }, [selectedCameras, saveFrame, currentFrame]);

  const handleSaveNeutral = useCallback(() => {
    void saveNeutralFrame();
  }, [saveNeutralFrame]);

  const [activeTab, setActiveTab] = useState<"cameras" | "overlay" | "pointOverlay">("cameras");
  useEffect(() => {
    setActiveTab("cameras");
  }, [selectedActor, selectedSequence]);

  // Overlay Video tab is only relevant once run_pipeline_batch_all.py has actually
  // rendered one (status "unconfirmed") or a person has already confirmed it
  // (status "confirmed", kept visible so it can still be rewatched afterward).
  const currentLabelStatus = dataset?.actors
    .find((a) => a.actor_id === selectedActor)
    ?.sequences.find((s) => s.sequence_id === selectedSequence)?.label_status;
  const hasOverlayVideo = currentLabelStatus === "unconfirmed" || currentLabelStatus === "confirmed";

  const dirtyCameras = selectedCameras.filter((cameraId) => frameCache[`${cameraId}:${currentFrame}`]?.dirty);
  const hasDirtyFrames = dirtyCameras.length > 0;
  const timeLabel = `${formatFrameLabel(currentFrame, fps)} / ${formatFrameLabel(maxFrameIndex, fps)}`;
  const buttonClass = "px-3 py-1.5 rounded border border-slate-700 text-xs font-medium text-slate-100 hover:bg-slate-800 disabled:opacity-40 disabled:cursor-not-allowed";

  return (
    <section className="flex flex-col gap-4 flex-1">
      <header className="flex flex-col gap-2 lg:flex-row lg:items-center lg:justify-between">
        <div>
          <h2 className="text-xl font-semibold text-white">
            {selectedActor ?? "Select an actor"} · {selectedSequence ?? "sequence"}
          </h2>
          <p className="text-xs text-slate-400">
            Frame {currentFrame} · {fps.toFixed(2)} fps · Neutral Frame Index:{" "}
            {neutralFrameIdx !== null ? `${neutralFrameIdx} (${neutralFrameSequenceId})` : "N/A"}
          </p>
        </div>
        <div className="flex flex-wrap gap-2 text-xs text-slate-300">
          <button className={buttonClass} onClick={handlePrevFrame} disabled={currentFrame === 0}>Prev Frame</button>
          <button className={buttonClass} onClick={handlePlayToggle} disabled={!selectedCameras.length}>
            {isPlaying ? "Pause" : "Play"}
          </button>
          <button className={buttonClass} onClick={handleNextFrame} disabled={currentFrame >= maxFrameIndex}>Next Frame</button>
          <button className={buttonClass} onClick={handleSaveAll} disabled={!hasDirtyFrames}>
            Save {dirtyCameras.length ? `(${dirtyCameras.length})` : ""}
          </button>
          <button className={buttonClass} onClick={undo}>Undo</button>
          <button className={buttonClass} onClick={redo}>Redo</button>
          <button className={buttonClass} onClick={handleSaveNeutral} disabled={!selectedActor || !selectedSequence}>Save as Neutral</button>
          <button className={buttonClass} onClick={goToNextUnlabeledSequence}>Next Unlabeled</button>
          <label className="ml-2 inline-flex items-center gap-2">
            <input
              type="checkbox"
              checked={showIndices}
              onChange={(event) => setShowIndices(event.target.checked)}
            />
            Show indices
          </label>
          <label className="inline-flex items-center gap-2">
            <input
              type="checkbox"
              checked={showSkinLandmarks}
              onChange={(event) => setShowSkinLandmarks(event.target.checked)}
            />
            Show skin landmarks
          </label>
        </div>
      </header>

      {hasOverlayVideo && (
        <div className="flex gap-1 border-b border-slate-800">
          <button
            className={`px-3 py-1.5 text-xs font-medium ${
              activeTab === "cameras"
                ? "border-b-2 border-sky-400 text-white"
                : "text-slate-400 hover:text-slate-200"
            }`}
            onClick={() => setActiveTab("cameras")}
          >
            Cameras
          </button>
          <button
            className={`px-3 py-1.5 text-xs font-medium ${
              activeTab === "overlay"
                ? "border-b-2 border-sky-400 text-white"
                : "text-slate-400 hover:text-slate-200"
            }`}
            onClick={() => setActiveTab("overlay")}
          >
            Overlay Video
          </button>
          <button
            className={`px-3 py-1.5 text-xs font-medium ${
              activeTab === "pointOverlay"
                ? "border-b-2 border-sky-400 text-white"
                : "text-slate-400 hover:text-slate-200"
            }`}
            onClick={() => setActiveTab("pointOverlay")}
          >
            Overlay Landmarks
          </button>
        </div>
      )}

      {activeTab === "overlay" && hasOverlayVideo && selectedActor && selectedSequence && (
        <div className="rounded-lg border border-slate-800 bg-slate-900/40 p-4">
          <video
            key={`${selectedActor}/${selectedSequence}`}
            controls
            className="w-full max-h-[75vh]"
            src={api.getOverlayVideoUrl(selectedActor, selectedSequence)}
          />
        </div>
      )}

      {/* Annotated (filled dots, Cameras-tab colors) vs. wrapped-mesh landmarks
          (lighter rings), rendered by wrap_script/render_point_overlay.py on the
          same camera as the Overlay Video. */}
      {activeTab === "pointOverlay" && hasOverlayVideo && selectedActor && selectedSequence && (
        <div className="rounded-lg border border-slate-800 bg-slate-900/40 p-4">
          <video
            key={`${selectedActor}/${selectedSequence}/points`}
            controls
            className="w-full max-h-[75vh]"
            src={api.getPointOverlayVideoUrl(selectedActor, selectedSequence)}
          />
        </div>
      )}

      {activeTab === "cameras" && (
      <>
      <div className="rounded-lg border border-slate-800 bg-slate-900/40 p-4 space-y-3">
        <div className="flex items-center gap-3">
          <input
            type="range"
            min={0}
            max={maxFrameIndex}
            value={currentFrame}
            onChange={(event) => handleSliderChange(Number(event.target.value))}
            className="flex-1"
          />
          <input
            type="number"
            min={0}
            max={maxFrameIndex}
            value={currentFrame}
            onChange={(event) => handleSliderChange(Number(event.target.value))}
            className="w-20 rounded bg-slate-800 border border-slate-700 px-2 py-1 text-right"
          />
          <span className="text-xs text-slate-400 tabular-nums">{timeLabel}</span>
        </div>dd
        <p className="text-[11px] text-slate-500">
          Space: Play/Pause · A/D: Step frame · Shift+Click: Add landmark · Alt+Click: Delete landmark · Drag to adjust · Shift+U: Next unlabeled · Ctrl+S: Save
        </p>
      </div>

      <div className="grid multiview-grid gap-6 justify-items-center">
        {selectedCameras.map((cameraId) => {
          const key = `${cameraId}:${currentFrame}`;
          const entry = frameCache[key];
          const frame = entry?.payload;
          const dirty = entry?.dirty;
          const skinFrame = skinFrameCache[key];
          const videoSrc = buildDelegationSrc(selectedActor, selectedSequence, cameraId);
          return (
            <VideoTile
              key={cameraId}
              cameraId={cameraId}
              videoSrc={videoSrc}
              frame={frame}
              dirty={dirty}
              currentFrame={currentFrame}
              fps={fps}
              isPlaying={isPlaying}
              showIndices={showIndices}
              skinFrame={skinFrame}
              showSkinLandmarks={showSkinLandmarks}
              isPropagating={
                Boolean(selectedActor && selectedSequence) &&
                propagatingCameras.includes(propagationKey(selectedActor!, selectedSequence!, cameraId))
              }
              onVideoMeta={setCameraVideoMeta}
            />
          );
        })}
        {selectedCameras.length === 0 && (
          <div className="rounded-lg border border-dashed border-slate-700 p-8 text-center text-slate-400 col-span-full">
            Select at least one camera to begin annotating.
          </div>
        )}
      </div>
      </>
      )}
    </section>
  );
}

// Cameras wrap_script_facescape/run_pipeline.py always renders the Overlay
// Frame keyline renders from -- matches its own OVERLAY_FRAME_CAMERA_IDS.
const FACESCAPE_OVERLAY_FRAME_CAMERA_IDS = ["1", "4", "17"];

// FaceScape has no video, no editing, and one static frame per camera -- always
// shows two tabs (Cameras / Overlay Frame), no play/pause/frame-slider/save/
// undo/neutral-frame controls, all of which are Nersemble/video-specific.
function FaceScapePlayer() {
  const {
    facescapeDataset,
    selectedFacescapeActor,
    selectedFacescapeExpression,
    selectedFacescapeCameras,
    facescapeLandmarkCache,
    loadFacescapeLandmarks,
    facescapeSkinLandmarkCache,
    loadFacescapeSkinLandmarks,
    facescapeEarAddMode,
    facescapeEarClicksLeft,
    facescapeEarClicksRight,
    startFacescapeEarAddMode,
    cancelFacescapeEarAddMode,
    clearFacescapeEarClicks,
    recordFacescapeEarClick,
    saveFacescapeEarPoints,
    facescapeNeutralExpression,
    saveFacescapeNeutralExpression,
    fetchFacescapeNeutralExpression,
    pendingLabelActions
  } = useAnnotationStore((state) => ({
    facescapeDataset: state.facescapeDataset,
    selectedFacescapeActor: state.selectedFacescapeActor,
    selectedFacescapeExpression: state.selectedFacescapeExpression,
    selectedFacescapeCameras: state.selectedFacescapeCameras,
    facescapeLandmarkCache: state.facescapeLandmarkCache,
    loadFacescapeLandmarks: state.loadFacescapeLandmarks,
    facescapeSkinLandmarkCache: state.facescapeSkinLandmarkCache,
    loadFacescapeSkinLandmarks: state.loadFacescapeSkinLandmarks,
    facescapeEarAddMode: state.facescapeEarAddMode,
    facescapeEarClicksLeft: state.facescapeEarClicksLeft,
    facescapeEarClicksRight: state.facescapeEarClicksRight,
    startFacescapeEarAddMode: state.startFacescapeEarAddMode,
    cancelFacescapeEarAddMode: state.cancelFacescapeEarAddMode,
    clearFacescapeEarClicks: state.clearFacescapeEarClicks,
    recordFacescapeEarClick: state.recordFacescapeEarClick,
    saveFacescapeEarPoints: state.saveFacescapeEarPoints,
    facescapeNeutralExpression: state.facescapeNeutralExpression,
    saveFacescapeNeutralExpression: state.saveFacescapeNeutralExpression,
    fetchFacescapeNeutralExpression: state.fetchFacescapeNeutralExpression,
    pendingLabelActions: state.pendingLabelActions
  }));

  const [activeTab, setActiveTab] = useState<"cameras" | "overlay" | "pointOverlay">("cameras");
  const [showSkinLandmarks, setShowSkinLandmarks] = useState(false);
  useEffect(() => {
    setActiveTab("cameras");
  }, [selectedFacescapeActor, selectedFacescapeExpression]);

  useEffect(() => {
    if (!selectedFacescapeActor) return;
    void fetchFacescapeNeutralExpression(selectedFacescapeActor);
  }, [selectedFacescapeActor, fetchFacescapeNeutralExpression]);

  useEffect(() => {
    if (!selectedFacescapeActor || !selectedFacescapeExpression) return;
    selectedFacescapeCameras.forEach((cameraId) => {
      void loadFacescapeLandmarks(selectedFacescapeActor, selectedFacescapeExpression, cameraId);
    });
  }, [selectedFacescapeActor, selectedFacescapeExpression, selectedFacescapeCameras, loadFacescapeLandmarks]);

  useEffect(() => {
    if (!showSkinLandmarks || !selectedFacescapeActor || !selectedFacescapeExpression) return;
    selectedFacescapeCameras.forEach((cameraId) => {
      void loadFacescapeSkinLandmarks(selectedFacescapeActor, selectedFacescapeExpression, cameraId);
    });
  }, [showSkinLandmarks, selectedFacescapeActor, selectedFacescapeExpression, selectedFacescapeCameras, loadFacescapeSkinLandmarks]);

  const expression = facescapeDataset?.actors
    .find((actor) => actor.actor_id === selectedFacescapeActor)?.expressions
    .find((expr) => expr.expression_id === selectedFacescapeExpression);

  const buttonClass = "px-3 py-1.5 text-xs font-medium";

  // Ear-add mode works regardless of the pair's label_status (a reviewer can
  // label ear-top points whether or not the rest of the pair is reviewed
  // yet) -- so these buttons are only gated on an actor/expression actually
  // being selected, not on `expression?.label_status`.
  const canAddEarPoints = Boolean(selectedFacescapeActor && selectedFacescapeExpression);
  // Actor-scoped (not actor+expression) -- Save Point triangulates AND
  // propagates to every other expression for this actor in one blocking
  // request (see facescape_ear_propagation.py), so the whole actor is
  // "busy" until that finishes, not just the one expression being saved.
  // Other actors are never blocked by this.
  const earSaveKey = selectedFacescapeActor ? `facescape-ear:${selectedFacescapeActor}` : undefined;
  const isSavingEarPoint = earSaveKey ? pendingLabelActions.includes(earSaveKey) : false;
  const leftEarClickCount = Object.keys(facescapeEarClicksLeft).length;
  const rightEarClickCount = Object.keys(facescapeEarClicksRight).length;
  const canSaveEarPoints = leftEarClickCount >= 2 || rightEarClickCount >= 2;
  const canClearEarPoints = leftEarClickCount > 0 || rightEarClickCount > 0;

  return (
    <section className="flex flex-col gap-4 flex-1">
      <header>
        <h2 className="text-xl font-semibold text-white">
          {selectedFacescapeActor ?? "Select an actor"} · {selectedFacescapeExpression ?? "expression"}
        </h2>
        <p className="text-xs text-slate-400">
          FaceScape · static multiview scan, read-only · Neutral expression:{" "}
          {facescapeNeutralExpression ?? "N/A"}
        </p>
        <label className="mt-2 inline-flex items-center gap-2 text-xs text-slate-300">
          <input
            type="checkbox"
            checked={showSkinLandmarks}
            onChange={(event) => setShowSkinLandmarks(event.target.checked)}
          />
          Show skin landmarks
        </label>
        <div className="mt-2 flex items-center gap-2">
          <button
            className={`${buttonClass} rounded border border-slate-700 text-slate-300 hover:text-white disabled:opacity-40 disabled:cursor-not-allowed`}
            disabled={!selectedFacescapeActor || !selectedFacescapeExpression}
            onClick={() => void saveFacescapeNeutralExpression()}
          >
            Save as Neutral
          </button>
          <button
            className={`${buttonClass} rounded border ${
              facescapeEarAddMode === "left"
                ? "border-amber-400 bg-amber-500/20 text-amber-200"
                : "border-slate-700 text-slate-300 hover:text-white"
            }`}
            disabled={!canAddEarPoints}
            onClick={() =>
              facescapeEarAddMode === "left" ? cancelFacescapeEarAddMode() : startFacescapeEarAddMode("left")
            }
          >
            {facescapeEarAddMode === "left" ? "Cancel Mode" : "Add Left Ear Top"}
          </button>
          <button
            className={`${buttonClass} rounded border ${
              facescapeEarAddMode === "right"
                ? "border-amber-400 bg-amber-500/20 text-amber-200"
                : "border-slate-700 text-slate-300 hover:text-white"
            }`}
            disabled={!canAddEarPoints}
            onClick={() =>
              facescapeEarAddMode === "right" ? cancelFacescapeEarAddMode() : startFacescapeEarAddMode("right")
            }
          >
            {facescapeEarAddMode === "right" ? "Cancel Mode" : "Add Right Ear Top"}
          </button>
          {(facescapeEarAddMode || canClearEarPoints) && (
            <>
              <button
                className={`${buttonClass} rounded border border-emerald-500 bg-emerald-500/20 text-emerald-200 disabled:opacity-40 disabled:cursor-not-allowed`}
                disabled={!canSaveEarPoints || isSavingEarPoint}
                onClick={() => void saveFacescapeEarPoints()}
              >
                {isSavingEarPoint ? "Saving + propagating..." : "Save"}
              </button>
              <button
                className={`${buttonClass} rounded border border-slate-700 text-slate-300 hover:text-white disabled:opacity-40 disabled:cursor-not-allowed`}
                disabled={!canClearEarPoints || isSavingEarPoint}
                onClick={() => clearFacescapeEarClicks()}
              >
                Clear
              </button>
              <span className="text-xs text-slate-400">
                left: {leftEarClickCount} camera{leftEarClickCount === 1 ? "" : "s"} · right: {rightEarClickCount} camera{rightEarClickCount === 1 ? "" : "s"} (need at least 2 per side to save that side)
              </span>
            </>
          )}
        </div>
      </header>

      <div className="flex gap-1 border-b border-slate-800">
        <button
          className={`${buttonClass} ${activeTab === "cameras" ? "border-b-2 border-sky-400 text-white" : "text-slate-400 hover:text-slate-200"}`}
          onClick={() => setActiveTab("cameras")}
        >
          Cameras
        </button>
        <button
          className={`${buttonClass} ${activeTab === "overlay" ? "border-b-2 border-sky-400 text-white" : "text-slate-400 hover:text-slate-200"}`}
          onClick={() => setActiveTab("overlay")}
        >
          Overlay Frame
        </button>
        <button
          className={`${buttonClass} ${activeTab === "pointOverlay" ? "border-b-2 border-sky-400 text-white" : "text-slate-400 hover:text-slate-200"}`}
          onClick={() => setActiveTab("pointOverlay")}
        >
          Overlay Landmarks
        </button>
      </div>

      {activeTab === "cameras" && (
        <div className="grid multiview-grid gap-6 justify-items-center">
          {selectedFacescapeCameras.map((cameraId) => {
            if (!selectedFacescapeActor || !selectedFacescapeExpression) return null;
            const key = `${selectedFacescapeActor}:${selectedFacescapeExpression}:${cameraId}`;
            const frame = facescapeLandmarkCache[key];
            const skinFrame = facescapeSkinLandmarkCache[key];
            const imageSrc = api.getFaceScapeImageUrl(selectedFacescapeActor, selectedFacescapeExpression, cameraId);
            return (
              <ImageTile
                key={cameraId}
                cameraId={cameraId}
                imageSrc={imageSrc}
                frame={frame}
                skinFrame={skinFrame}
                showSkinLandmarks={showSkinLandmarks}
                earAddMode={facescapeEarAddMode !== null}
                pendingEarClick={
                  facescapeEarAddMode === "left"
                    ? facescapeEarClicksLeft[cameraId]
                    : facescapeEarAddMode === "right"
                      ? facescapeEarClicksRight[cameraId]
                      : undefined
                }
                onEarClick={(x, y) => recordFacescapeEarClick(cameraId, x, y)}
              />
            );
          })}
          {selectedFacescapeCameras.length === 0 && (
            <div className="rounded-lg border border-dashed border-slate-700 p-8 text-center text-slate-400 col-span-full">
              Select at least one camera in the sidebar to view.
            </div>
          )}
        </div>
      )}

      {(activeTab === "overlay" || activeTab === "pointOverlay") && (() => {
        // "overlay" = keyline renders (render_overlay_frame.py); "pointOverlay" =
        // annotated vs. wrapped-mesh landmark renders (render_point_overlay_frame.py),
        // same cameras.
        const getFrameUrl =
          activeTab === "pointOverlay" ? api.getFaceScapePointOverlayFrameUrl : api.getFaceScapeOverlayFrameUrl;
        // Same gate as Nersemble's hasOverlayVideo: run_pipeline.py only moves a
        // pair past "unreviewed" (and only renders the overlay) once it's
        // actually wrapped -- see run_pipeline.py's mark_unconfirmed() call.
        const hasOverlayFrame = expression?.label_status === "unconfirmed" || expression?.label_status === "confirmed";
        if (hasOverlayFrame && selectedFacescapeActor && selectedFacescapeExpression) {
          return (
            <div className="grid multiview-grid gap-6 justify-items-center">
              {(expression?.is_invalid && expression.overlay_camera_ids.length
                ? expression.overlay_camera_ids
                : FACESCAPE_OVERLAY_FRAME_CAMERA_IDS
              ).map((cameraId) => (
                <div
                  key={`${activeTab}/${selectedFacescapeActor}/${selectedFacescapeExpression}/${cameraId}`}
                  className="rounded-lg border border-slate-800 bg-slate-900/40 p-4"
                >
                  <img
                    src={getFrameUrl(selectedFacescapeActor, selectedFacescapeExpression, cameraId)}
                    className="w-full max-h-[75vh] object-contain"
                    // run_pipeline.py renders each camera's overlay best-effort --
                    // one camera failing doesn't block the others, so a pair can
                    // have some cameras rendered and some not. Hide whichever
                    // ones 404 instead of showing a broken-image icon.
                    onError={(event) => {
                      event.currentTarget.style.display = "none";
                    }}
                  />
                </div>
              ))}
            </div>
          );
        }
        return (
          <div className="rounded-lg border border-dashed border-slate-700 p-8 text-center text-slate-400">
            {activeTab === "pointOverlay" ? "Overlay Landmarks" : "Overlay Frame"} is not available yet -- this pair hasn't been wrapped.
            {expression && !expression.has_landmarks && " (No sapiens2 landmarks yet for this pair either.)"}
          </div>
        );
      })()}
    </section>
  );
}

interface VideoTileProps {
  cameraId: string;
  videoSrc?: string;
  frame?: LandmarkFrame;
  dirty?: boolean;
  currentFrame: number;
  fps: number;
  isPlaying: boolean;
  showIndices: boolean;
  skinFrame?: LandmarkFrame;
  showSkinLandmarks: boolean;
  isPropagating: boolean;
  onVideoMeta: (cameraId: string, fps: number, numFrames: number) => void;
}

function VideoTile({
  cameraId,
  videoSrc,
  frame,
  dirty,
  currentFrame,
  fps: fallbackFps,
  isPlaying,
  showIndices,
  skinFrame,
  showSkinLandmarks,
  isPropagating,
  onVideoMeta
}: VideoTileProps) {
  const [dimensions, setDimensions] = useState({ width: 640, height: 360 });
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const measuredMeta = useVideoMeta(videoSrc);
  const fps = measuredMeta?.fps ?? fallbackFps;

  useEffect(() => {
    if (measuredMeta) {
      onVideoMeta(cameraId, measuredMeta.fps, measuredMeta.numFrames);
    }
  }, [cameraId, measuredMeta, onVideoMeta]);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    const updateSize = () => {
      setDimensions({
        width: video.clientWidth || video.videoWidth || 300,
        height: video.clientHeight || video.videoHeight || 200
      });
    };
    updateSize();
    video.addEventListener("loadedmetadata", updateSize);

    const observer = new ResizeObserver(updateSize);
    observer.observe(video);

    return () => {
      video.removeEventListener("loadedmetadata", updateSize);
      observer.disconnect();
    };
  }, [videoSrc]);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    const targetTime = (currentFrame + 0.9) / Math.max(1, fps);
    const delta = Math.abs(video.currentTime - targetTime);
    // Use a looser threshold during playback to prevent stuttering from frequent seeking
    const threshold = isPlaying ? 0.1 : 1 / Math.max(30, fps);
    if (delta > threshold) {
      video.currentTime = targetTime;
    }
  }, [currentFrame, fps, isPlaying]);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    if (isPlaying) {
      void video.play().catch(() => {
        // Ignore autoplay failures; manual playback is still synced via currentTime updates.
      });
    } else {
      video.pause();
    }
  }, [isPlaying]);

  return (
    <div className="relative rounded-lg overflow-hidden bg-black shadow-lg w-full max-w-[320px]">
      {videoSrc ? (
        <video
          ref={videoRef}
          id={`video-${cameraId}`}
          src={videoSrc}
          style={{ maxWidth: "300px" }}
          muted
          playsInline
          className="w-full h-auto object-contain"
        />
      ) : (
        <div className="h-64 flex items-center justify-center text-slate-500">
          Delegation video not available yet. Trigger preprocessing.
        </div>
      )}
      <LandmarkOverlay
        cameraId={cameraId}
        frameIdx={currentFrame}
        frame={frame}
        width={dimensions.width}
        height={dimensions.height}
        showIndices={showIndices}
        skinFrame={skinFrame}
        showSkinLandmarks={showSkinLandmarks}
        interactive
      />
      <div className="absolute bottom-2 left-2 text-xs bg-slate-900/70 px-2 py-1 rounded">
        {cameraId}
      </div>
      {dirty && (
        <span className="absolute top-2 right-2 bg-amber-500 text-xs px-2 py-1 rounded">Unsaved</span>
      )}
      {isPropagating && (
        <div className="absolute inset-0 bg-black/50 flex items-center justify-center z-10">
          <div className="flex flex-col items-center gap-2">
            <div className="w-8 h-8 border-4 border-blue-500 border-t-transparent rounded-full animate-spin"></div>
            <span className="text-xs font-medium text-blue-200">Propagating...</span>
          </div>
        </div>
      )}
    </div>
  );
}

function buildDelegationSrc(actorId?: string, sequenceId?: string, cameraId?: string) {
  if (!actorId || !sequenceId || !cameraId) return undefined;
  return `${MEDIA_BASE}/${actorId}/${sequenceId}/${cameraId}_delegation.mp4`;
}

function formatFrameLabel(frame: number, fps: number) {
  const safeFps = Math.max(1, Math.round(fps || 0));
  const totalSeconds = frame / Math.max(fps, 0.0001);
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = Math.floor(totalSeconds % 60);
  const frames = Math.floor(frame % safeFps);
  return `${padNumber(minutes)}:${padNumber(seconds)}:${padNumber(frames)}`;
}

function padNumber(value: number) {
  return value.toString().padStart(2, "0");
}
