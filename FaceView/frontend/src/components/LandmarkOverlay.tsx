import { useEffect, useRef, useState } from "react";
import type { PointerEvent as ReactPointerEvent } from "react";

import type { LandmarkFrame, LandmarkPoint } from "@/lib/types";
import { useAnnotationStore } from "@/state/useAnnotationStore";
import WhiteList from "../../../whitelist.json";

interface LandmarkOverlayProps {
  cameraId: string;
  frameIdx: number;
  frame?: LandmarkFrame;
  width: number;
  height: number;
  showIndices?: boolean;
  interactive?: boolean;
  // "Skin" landmarks (index >= 74) from run_neutral_cotracker_propagation.py —
  // read-only (no drag/add/delete), only drawn when showSkinLandmarks is true.
  // Not every actor/sequence has these yet (see LandmarkOverlay's skinFrame
  // usage below): skinFrame is simply undefined/pointless in that case, so
  // nothing extra renders — no sentinel values needed.
  skinFrame?: LandmarkFrame;
  showSkinLandmarks?: boolean;
  // Indices to show in addition to the shared whitelist.json set, without
  // touching that file -- it's also read by the Python wrap pipeline
  // (run_pipeline_all.py, ensure_extended_correspondence_files) for actual
  // mesh-wrapping, not just display, so a dataset-specific display tweak here
  // shouldn't change what that pipeline does for every dataset.
  extraVisibleIndices?: number[];
  // FaceScape "Add Left/Right Ear Top" mode: while true, clicking anywhere on
  // the image calls onEarClick(x, y) instead of the normal drag/shift-click/
  // alt-click point-editing behavior (which FaceScape never enables via
  // `interactive` anyway). pendingEarClick renders this camera's not-yet-saved
  // click, if one has been recorded, distinctly from confirmed landmarks.
  earAddMode?: boolean;
  pendingEarClick?: { x: number; y: number };
  onEarClick?: (x: number, y: number) => void;
}

const JAW = [31, 28, 12, 53, 48, 49, 52, 51, 50, 66, 67, 64, 63, 68, 3, 7, 10];
const LEFT_EAR = [2, 4, 1];
const RIGHT_EAR = [9, 14, 6];
const LEFT_BROW = [8, 30, 0, 73, 72];
const RIGHT_BROW = [29, 47, 5, 70, 69];
const NOSE = [60, 62, 54, 57];
const LEFT_EYE_UP = [15, 17];
const LEFT_EYE = [71, 16, 13, 11];
const RIGHT_EYE_UP = [34, 36];
const RIGHT_EYE = [35, 33, 32, 65];
const UNDER_NOSE = [38, 41, 61, 19, 22];
const RIGHT_MOUTH_CORNER = [37, 43];
const LEFT_MOUTH_CORNER = [18, 24];
const OUTER_MOUTH_RIGHT = [42, 44];
const OUTER_MOUTH_LEFT = [23, 25];
const OUTER_MOUTH_MIDDLE_OUTER = [39, 20, 46, 27];
const INNER_MOUTH_MIDDLE_OUTER = [40, 21, 45, 26];
const OUTER_MOUTH_MIDDLE_MIDDLE = [59, 55];
const INNER_MOUTH_MIDDLE_MIDDLE = [56, 58];

const WHITE_LIST_SET = new Set<number>(WhiteList);
const WHITE_LIST_INDICES = Array.from(WHITE_LIST_SET).sort((a, b) => a - b);

const JAW_COLOR = "255, 0, 0";                // red
const LEFT_EAR_COLOR = "0, 255, 0";           // green
const RIGHT_EAR_COLOR = "0, 0, 255";          // blue
const LEFT_BROW_COLOR = "255, 165, 0";        // orange
const RIGHT_BROW_COLOR = "128, 0, 128";       // purple
const NOSE_COLOR = "255, 255, 0";             // yellow
const LEFT_EYE_UP_COLOR = "0, 200, 200";         // cyan
const LEFT_EYE_COLOR = "0, 255, 255";         // cyan
const RIGHT_EYE_UP_COLOR = "200, 0, 200";         // cyan
const RIGHT_EYE_COLOR = "255, 0, 255";        // magenta
const UNDER_NOSE_COLOR = "0, 128, 0";         // dark green
const RIGHT_MOUTH_CORNER_COLOR = "128, 0, 0"; // dark red
const LEFT_MOUTH_CORNER_COLOR = "0, 0, 128";  // navy
const OUTER_MOUTH_RIGHT_COLOR = "255, 105, 180"; // hot pink
const OUTER_MOUTH_LEFT_COLOR = "0, 128, 128";    // teal
const OUTER_MOUTH_MIDDLE_OUTER_COLOR = "210, 105, 30"; // chocolate
const INNER_MOUTH_MIDDLE_OUTER_COLOR = "75, 0, 130";   // indigo
const OUTER_MOUTH_MIDDLE_MIDDLE_COLOR = "154, 205, 50"; // yellow‑green
const INNER_MOUTH_MIDDLE_MIDDLE_COLOR = "70, 130, 180"; // steel blue

const SKIN_POINT_COLOR = "rgba(148, 163, 184, 0.85)"; // slate — distinct from every whitelist region color
const PENDING_EAR_CLICK_COLOR = "rgba(250, 204, 21, 0.95)"; // amber — distinct from every landmark/skin color above
const POINT_COLOR = "rgba(56, 189, 248, 0.9)";
const DIRTY_COLOR = "rgba(249, 115, 22, 0.9)";

const HOVER_COLOR = "rgba(168, 85, 247, 0.9)";
const ACTIVE_COLOR = "rgba(16, 185, 129, 0.95)";
const SELECTION_RADIUS_PX = 16;

export function LandmarkOverlay({
  cameraId,
  frameIdx,
  frame,
  width,
  height,
  showIndices,
  interactive = false,
  skinFrame,
  showSkinLandmarks = false,
  extraVisibleIndices,
  earAddMode = false,
  pendingEarClick,
  onEarClick
}: LandmarkOverlayProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);
  const [activeIndex, setActiveIndex] = useState<number | null>(null);
  const draggingRef = useRef(false);
  const { updatePoint, addPoint, removePoint } = useAnnotationStore((state) => ({
    updatePoint: state.updatePoint,
    addPoint: state.addPoint,
    removePoint: state.removePoint
  }));

  const points: LandmarkPoint[] = frame?.points ?? [];
  const filteredPoints = points.filter(
    (point) => WHITE_LIST_SET.has(point.index) || extraVisibleIndices?.includes(point.index)
  );
  const skinPoints: LandmarkPoint[] = showSkinLandmarks ? (skinFrame?.points ?? []) : [];

  useEffect(() => {
    setHoveredIndex(null);
    setActiveIndex(null);
    draggingRef.current = false;
  }, [frameIdx, frame?.frame_idx]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, width, height);
    if (!filteredPoints.length && !skinPoints.length) return;
    ctx.font = "10px Inter";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    filteredPoints.forEach((point) => {
      const x = point.x * width;
      const y = point.y * height;
      ctx.beginPath();
      ctx.arc(x, y, 2, 0, Math.PI * 2);
      ctx.closePath();

      // const baseColor = (point.visibility ?? 0.5) < 0.2 ? DIRTY_COLOR : POINT_COLOR;
      var baseColor = "rgba(";
      if (JAW.includes(point.index))
        baseColor += JAW_COLOR;
      else if (LEFT_EAR.includes(point.index))
        baseColor += LEFT_EAR_COLOR;
      else if (RIGHT_EAR.includes(point.index))
        baseColor += RIGHT_EAR_COLOR;
      else if (LEFT_BROW.includes(point.index))
        baseColor += LEFT_BROW_COLOR;
      else if (RIGHT_BROW.includes(point.index))
        baseColor += RIGHT_BROW_COLOR;
      else if (NOSE.includes(point.index))
        baseColor += NOSE_COLOR;
      else if (LEFT_EYE.includes(point.index))
        baseColor += LEFT_EYE_COLOR;
      else if (RIGHT_EYE.includes(point.index))
        baseColor += RIGHT_EYE_COLOR;
      else if (RIGHT_EYE_UP.includes(point.index))
        baseColor += RIGHT_EYE_UP_COLOR;
      else if (LEFT_EYE_UP.includes(point.index))
        baseColor += LEFT_EYE_UP_COLOR;
      else if (UNDER_NOSE.includes(point.index))
        baseColor += UNDER_NOSE_COLOR;
      else if (RIGHT_MOUTH_CORNER.includes(point.index))
        baseColor += RIGHT_MOUTH_CORNER_COLOR;
      else if (LEFT_MOUTH_CORNER.includes(point.index))
        baseColor += LEFT_MOUTH_CORNER_COLOR;
      else if (OUTER_MOUTH_RIGHT.includes(point.index))
        baseColor += OUTER_MOUTH_RIGHT_COLOR;
      else if (OUTER_MOUTH_LEFT.includes(point.index))
        baseColor += OUTER_MOUTH_LEFT_COLOR;
      else if (OUTER_MOUTH_MIDDLE_OUTER.includes(point.index))
        baseColor += OUTER_MOUTH_MIDDLE_OUTER_COLOR;
      else if (INNER_MOUTH_MIDDLE_OUTER.includes(point.index))
        baseColor += INNER_MOUTH_MIDDLE_OUTER_COLOR;
      else if (OUTER_MOUTH_MIDDLE_MIDDLE.includes(point.index))
        baseColor += OUTER_MOUTH_MIDDLE_MIDDLE_COLOR;
      else if (INNER_MOUTH_MIDDLE_MIDDLE.includes(point.index))
        baseColor += INNER_MOUTH_MIDDLE_MIDDLE_COLOR;

      baseColor += ", 0.7)";
      // console.log(baseColor)


      if (point.index === activeIndex) {
        ctx.fillStyle = ACTIVE_COLOR;
      } else if (point.index === hoveredIndex) {
        ctx.fillStyle = HOVER_COLOR;
      } else {
        ctx.fillStyle = baseColor;
      }
      ctx.fill();
      if (showIndices) {
        ctx.fillStyle = "rgba(226, 232, 240, 0.9)";
        ctx.fillText(String(point.index), x + 10, y);
      }
    });

    // Skin landmarks are read-only (no drag/add/delete), so no hover/active
    // color handling — just a plain marker distinct from the editable colors above.
    skinPoints.forEach((point) => {
      const x = point.x * width;
      const y = point.y * height;
      ctx.beginPath();
      ctx.arc(x, y, 2, 0, Math.PI * 2);
      ctx.closePath();
      ctx.fillStyle = SKIN_POINT_COLOR;
      ctx.fill();
      if (showIndices) {
        ctx.fillStyle = "rgba(226, 232, 240, 0.9)";
        ctx.fillText(String(point.index), x + 10, y);
      }
    });

    if (pendingEarClick) {
      const x = pendingEarClick.x * width;
      const y = pendingEarClick.y * height;
      ctx.beginPath();
      ctx.arc(x, y, 6, 0, Math.PI * 2);
      ctx.strokeStyle = PENDING_EAR_CLICK_COLOR;
      ctx.lineWidth = 2;
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(x, y, 2, 0, Math.PI * 2);
      ctx.fillStyle = PENDING_EAR_CLICK_COLOR;
      ctx.fill();
    }
  }, [filteredPoints, skinPoints, width, height, showIndices, hoveredIndex, activeIndex, pendingEarClick]);

  const clickable = interactive || earAddMode;
  const pointerClass = clickable ? "absolute inset-0" : "absolute inset-0 pointer-events-none";

  const handlePointerPosition = (event: ReactPointerEvent<HTMLCanvasElement>) => {
    const rect = event.currentTarget.getBoundingClientRect();
    const x = (event.clientX - rect.left) / rect.width;
    const y = (event.clientY - rect.top) / rect.height;
    return {
      x: clampNormalized(x),
      y: clampNormalized(y)
    };
  };

  const findNearestPoint = (x: number, y: number) => {
    let closest: LandmarkPoint | undefined;
    let bestDistance = Number.POSITIVE_INFINITY;
    filteredPoints.forEach((point) => {
      const px = point.x * width;
      const py = point.y * height;
      const distance = Math.hypot(px - x * width, py - y * height);
      if (distance <= SELECTION_RADIUS_PX && distance < bestDistance) {
        closest = point;
        bestDistance = distance;
      }
    });
    return closest;
  };

  const handlePointerDown = (event: ReactPointerEvent<HTMLCanvasElement>) => {
    if (earAddMode) {
      event.preventDefault();
      const { x, y } = handlePointerPosition(event);
      onEarClick?.(x, y);
      return;
    }
    if (!interactive || !frame) return;
    event.preventDefault();
    const { x, y } = handlePointerPosition(event);
    const target = findNearestPoint(x, y);
    if (event.altKey && target) {
      removePoint(cameraId, frameIdx, target.index);
      return;
    }
    if (event.shiftKey) {
      const nextIndex = WHITE_LIST_INDICES.find((index) => !filteredPoints.some((point) => point.index === index));
      if (nextIndex === undefined) {
        return;
      }
      addPoint(cameraId, frameIdx, { index: nextIndex, x, y, visibility: 1 });
      return;
    }
    if (target) {
      setActiveIndex(target.index);
      draggingRef.current = true;
    }
  };

  const handlePointerMove = (event: ReactPointerEvent<HTMLCanvasElement>) => {
    if (earAddMode || !interactive || !frame) return;
    const { x, y } = handlePointerPosition(event);
    const target = findNearestPoint(x, y);
    setHoveredIndex(target?.index ?? null);
    if (draggingRef.current && activeIndex !== null) {
      updatePoint(cameraId, frameIdx, { index: activeIndex, x, y });
    }
  };

  const handlePointerUp = () => {
    draggingRef.current = false;
    setActiveIndex(null);
  };

  return (
    <canvas
      ref={canvasRef}
      className={`${pointerClass} ${clickable ? "cursor-crosshair" : ""}`}
      onPointerDown={clickable ? handlePointerDown : undefined}
      onPointerMove={clickable ? handlePointerMove : undefined}
      onPointerUp={clickable ? handlePointerUp : undefined}
      onPointerLeave={clickable ? handlePointerUp : undefined}
    />
  );
}

const clampNormalized = (value: number) => Math.min(1, Math.max(0, value));
