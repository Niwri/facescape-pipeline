import { useEffect, useRef, useState } from "react";

import type { LandmarkFrame } from "@/lib/types";
import { LandmarkOverlay } from "./LandmarkOverlay";

interface ImageTileProps {
  cameraId: string;
  imageSrc?: string;
  frame?: LandmarkFrame;
  skinFrame?: LandmarkFrame;
  showSkinLandmarks?: boolean;
  // "Add Left/Right Ear Top" mode pass-through -- see LandmarkOverlay's own
  // earAddMode/pendingEarClick/onEarClick docs. Owned by the parent
  // (FaceScapePlayer), which reads/writes useAnnotationStore's
  // facescapeEarAddMode/facescapeEarClicksLeft/facescapeEarClicksRight so
  // every camera tile in the grid shares one mode toggle.
  earAddMode?: boolean;
  pendingEarClick?: { x: number; y: number };
  onEarClick?: (x: number, y: number) => void;
}

// Shown for FaceScape only, not Nersemble -- see LandmarkOverlay's
// extraVisibleIndices docs for why these aren't just added to whitelist.json.
const FACESCAPE_EXTRA_VISIBLE_INDICES = [69, 72];

// Read-only counterpart to MultiviewPlayer.tsx's VideoTile -- no playback/seek/
// dirty-state, since FaceScape has exactly one static image per camera and this
// tab is view-only (no drag-to-edit this pass, see wrap_script_facescape/README.md
// Phase 4).
export function ImageTile({
  cameraId,
  imageSrc,
  frame,
  skinFrame,
  showSkinLandmarks,
  earAddMode,
  pendingEarClick,
  onEarClick
}: ImageTileProps) {
  const [dimensions, setDimensions] = useState({ width: 320, height: 480 });
  const imgRef = useRef<HTMLImageElement | null>(null);

  useEffect(() => {
    const img = imgRef.current;
    if (!img) return;
    const updateSize = () => {
      setDimensions({
        width: img.clientWidth || img.naturalWidth || 320,
        height: img.clientHeight || img.naturalHeight || 480
      });
    };
    if (img.complete) updateSize();
    img.addEventListener("load", updateSize);
    const observer = new ResizeObserver(updateSize);
    observer.observe(img);
    return () => {
      img.removeEventListener("load", updateSize);
      observer.disconnect();
    };
  }, [imageSrc]);

  return (
    <div className="relative rounded-lg overflow-hidden bg-black shadow-lg w-full max-w-[320px]">
      {imageSrc ? (
        <img
          ref={imgRef}
          src={imageSrc}
          style={{ maxWidth: "300px" }}
          className="w-full h-auto object-contain"
        />
      ) : (
        <div className="h-64 flex items-center justify-center text-slate-500">
          No image available for this camera.
        </div>
      )}
      <LandmarkOverlay
        cameraId={cameraId}
        frameIdx={0}
        frame={frame}
        width={dimensions.width}
        height={dimensions.height}
        extraVisibleIndices={FACESCAPE_EXTRA_VISIBLE_INDICES}
        skinFrame={skinFrame}
        showSkinLandmarks={showSkinLandmarks}
        earAddMode={earAddMode}
        pendingEarClick={pendingEarClick}
        onEarClick={onEarClick}
      />
      <div className="absolute bottom-2 left-2 text-xs bg-slate-900/70 px-2 py-1 rounded">
        {cameraId}
      </div>
    </div>
  );
}
