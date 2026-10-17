import { useEffect, useMemo } from "react";

import { useAnnotationStore } from "@/state/useAnnotationStore";

export function useKeyboardShortcuts() {
  const {
    currentFrame,
    setFrame,
    getAdjacentAvailableFrame,
    selectedCameras,
    saveFrame,
    dataset,
    selectedActor,
    selectedSequence,
    cameraVideoMeta,
    togglePlayback,
    undo,
    redo,
    goToNextUnlabeledSequence
  } = useAnnotationStore((state) => ({
    currentFrame: state.currentFrame,
    setFrame: state.setFrame,
    getAdjacentAvailableFrame: state.getAdjacentAvailableFrame,
    selectedCameras: state.selectedCameras,
    saveFrame: state.saveFrame,
    dataset: state.dataset,
    selectedActor: state.selectedActor,
    selectedSequence: state.selectedSequence,
    cameraVideoMeta: state.cameraVideoMeta,
    togglePlayback: state.togglePlayback,
    undo: state.undo,
    redo: state.redo,
    goToNextUnlabeledSequence: state.goToNextUnlabeledSequence
  }));

  const maxFrameIndex = useMemo(() => {
    // Real frame count measured from the video itself takes priority — see
    // MultiviewPlayer's resolvedFrameCount for why landmark file count and
    // dataset.video.num_frames are both unreliable (sparse landmarks, and the
    // backend never actually populates num_frames, respectively).
    const measuredCount = cameraVideoMeta[selectedCameras[0]]?.numFrames;
    if (measuredCount && measuredCount > 0) {
      return Math.max(0, measuredCount - 1);
    }
    const sequence = dataset?.actors
      .find((actor) => actor.actor_id === selectedActor)?.sequences
      .find((seq) => seq.sequence_id === selectedSequence);
    const totalFrames = sequence?.cameras[0]?.video.num_frames;
    const fallback = typeof totalFrames === "number" && totalFrames > 0 ? totalFrames : 500;
    return Math.max(0, fallback - 1);
  }, [cameraVideoMeta, selectedCameras, dataset, selectedActor, selectedSequence]);

  useEffect(() => {
    const handler = (event: KeyboardEvent) => {
      if (event.target instanceof HTMLInputElement || event.target instanceof HTMLTextAreaElement) {
        return;
      }
      if (event.key.toLowerCase() === "d") {
        event.preventDefault();
        const primaryCamera = selectedCameras[0];
        const next = primaryCamera
          ? getAdjacentAvailableFrame(primaryCamera, currentFrame, 1)
          : currentFrame + 1;
        setFrame(Math.min(maxFrameIndex, next));
      } else if (event.key.toLowerCase() === "a") {
        event.preventDefault();
        const primaryCamera = selectedCameras[0];
        const prev = primaryCamera
          ? getAdjacentAvailableFrame(primaryCamera, currentFrame, -1)
          : currentFrame - 1;
        setFrame(Math.max(0, prev));
      } else if (event.code === "Space") {
        event.preventDefault();
        togglePlayback();
      } else if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "s") {
        event.preventDefault();

        // Always send every selected camera, not just the ones with local edits:
        // triangulation needs an observation from each view to fuse a 3D point,
        // so a save that only includes the edited camera can never triangulate.
        if (selectedCameras.length) {
          void saveFrame(currentFrame, selectedCameras);
        }
      } else if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "z") {
        event.preventDefault();
        if (event.shiftKey) {
          redo();
        } else {
          undo();
        }
      } else if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "y") {
        event.preventDefault();
        redo();
      } else if (event.shiftKey && event.key.toLowerCase() === "u") {
        event.preventDefault();
        goToNextUnlabeledSequence();
      }
    };

    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [currentFrame, maxFrameIndex, selectedCameras, setFrame, getAdjacentAvailableFrame, saveFrame, togglePlayback, undo, redo, goToNextUnlabeledSequence]);
}