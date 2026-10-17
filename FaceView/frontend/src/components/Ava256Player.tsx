import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import classNames from "classnames";

import { api } from "@/lib/api";
import { AVA256_FRAME_LABELS, type Ava256FrameLabel, type Ava256FrameStatus, type Ava256RangesByLabel } from "@/lib/types";
import { useAva256Store } from "@/state/useAva256Store";

// Frame positions are 1-based and match the "#N" burned into each video frame
// by render_capture_line_video.py and the rows of the capture's CSV.

const STATUS_STYLE: Record<Ava256FrameStatus, { cell: string; text: string; chip: string }> = {
  Valid: { cell: "bg-emerald-500", text: "text-emerald-300", chip: "border-emerald-500/60 text-emerald-200" },
  Invalid: { cell: "bg-rose-500", text: "text-rose-300", chip: "border-rose-500/60 text-rose-200" },
  Rerun: { cell: "bg-amber-400", text: "text-amber-300", chip: "border-amber-400/60 text-amber-200" },
  unconfirmed: { cell: "bg-slate-700", text: "text-slate-400", chip: "border-slate-600 text-slate-300" }
};

// Same default threshold as wrap_script_ava256/check_ava256_skin_landmark_errors.py.
const SKIN_ERROR_FLAG_PX = 4.0;

type Range = [number, number];
type RangeText = Record<Ava256FrameLabel, string>;
const EMPTY_TEXT: RangeText = { Valid: "", Invalid: "", Rerun: "" };

function parseRanges(text: string, n: number): { ranges: Range[]; errors: string[] } {
  const ranges: Range[] = [];
  const errors: string[] = [];
  for (const raw of text.split(",")) {
    const token = raw.trim();
    if (!token) continue;
    const match = token.match(/^(\d+)\s*(?:-\s*(\d+))?$/);
    if (!match) {
      errors.push(`"${token}" isn't a frame or range`);
      continue;
    }
    let start = Number(match[1]);
    let end = match[2] ? Number(match[2]) : start;
    if (start > end) [start, end] = [end, start];
    if (start < 1 || end > n) {
      errors.push(`${token} is outside 1-${n}`);
      continue;
    }
    ranges.push([start, end]);
  }
  return { ranges, errors };
}

function formatRanges(ranges: Range[]): string {
  return ranges.map(([a, b]) => (a === b ? `${a}` : `${a}-${b}`)).join(", ");
}

// Contiguous runs of each label in the saved statuses -> range text.
function rangesFromStatuses(statuses: Ava256FrameStatus[]): RangeText {
  const out: Record<Ava256FrameLabel, Range[]> = { Valid: [], Invalid: [], Rerun: [] };
  let i = 0;
  while (i < statuses.length) {
    const status = statuses[i];
    let j = i;
    while (j + 1 < statuses.length && statuses[j + 1] === status) j += 1;
    if (status !== "unconfirmed") out[status].push([i + 1, j + 1]);
    i = j + 1;
  }
  return { Valid: formatRanges(out.Valid), Invalid: formatRanges(out.Invalid), Rerun: formatRanges(out.Rerun) };
}

export function Ava256Player() {
  const { selectedCapture, frames, isDone, isFramesLoading, isSaving, isFinishing, applyRanges, finish, dataset } =
    useAva256Store((state) => ({
      selectedCapture: state.selectedCapture,
      frames: state.frames,
      isDone: state.isDone,
      isFramesLoading: state.isFramesLoading,
      isSaving: state.isSaving,
      isFinishing: state.isFinishing,
      applyRanges: state.applyRanges,
      finish: state.finish,
      dataset: state.dataset
    }));

  const videoRef = useRef<HTMLVideoElement>(null);
  const [duration, setDuration] = useState(0);
  const [position, setPosition] = useState(1);
  const [text, setText] = useState<RangeText>(EMPTY_TEXT);
  const [selection, setSelection] = useState<Range | null>(null);
  const [dragAnchor, setDragAnchor] = useState<number | null>(null);
  const [confirmingDone, setConfirmingDone] = useState(false);

  const n = frames.length;
  const savedStatuses = useMemo(() => frames.map((f) => f.status), [frames]);
  const fps = duration > 0 && n > 0 ? n / duration : 24;

  // Prefill the editor from the saved CSV whenever a capture's frames (re)load
  // or are saved -- never while typing.
  useEffect(() => {
    setText(rangesFromStatuses(savedStatuses));
    setSelection(null);
    setConfirmingDone(false);
  }, [savedStatuses]);

  useEffect(() => {
    setDuration(0);
    setPosition(1);
  }, [selectedCapture]);

  // Track the playhead frame precisely while playing.
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    let handle = 0;
    const tick = () => {
      if (n > 0) setPosition(Math.min(n, Math.floor(video.currentTime * fps + 1e-3) + 1));
      handle = window.requestAnimationFrame(tick);
    };
    handle = window.requestAnimationFrame(tick);
    return () => window.cancelAnimationFrame(handle);
  }, [n, fps, selectedCapture]);

  const seek = useCallback(
    (pos: number) => {
      const video = videoRef.current;
      if (!video || n === 0) return;
      const clamped = Math.max(1, Math.min(n, pos));
      video.pause();
      video.currentTime = (clamped - 0.5) / fps; // middle of the frame
      setPosition(clamped);
    },
    [n, fps]
  );

  const parsed = useMemo(() => {
    const result = {} as Record<Ava256FrameLabel, { ranges: Range[]; errors: string[] }>;
    for (const label of AVA256_FRAME_LABELS) result[label] = parseRanges(text[label], n);
    return result;
  }, [text, n]);

  // Pending result = saved status overridden by every range in the editor;
  // a frame claimed by two labels (or twice) is a conflict.
  const { preview, conflicts } = useMemo(() => {
    const owner: (Ava256FrameLabel | null)[] = new Array(n).fill(null);
    const conflictSet = new Set<number>();
    for (const label of AVA256_FRAME_LABELS) {
      for (const [a, b] of parsed[label].ranges) {
        for (let p = a; p <= b; p += 1) {
          if (owner[p - 1] !== null) conflictSet.add(p);
          owner[p - 1] = label;
        }
      }
    }
    return {
      preview: savedStatuses.map((status, i) => owner[i] ?? status),
      conflicts: conflictSet
    };
  }, [parsed, savedStatuses, n]);

  const parseErrors = AVA256_FRAME_LABELS.flatMap((label) => parsed[label].errors.map((e) => `${label}: ${e}`));
  const isDirty = preview.some((status, i) => status !== savedStatuses[i]);
  const savedUnconfirmed = savedStatuses.filter((s) => s === "unconfirmed").length;
  const previewCounts = useMemo(() => {
    const counts: Record<Ava256FrameStatus, number> = { Valid: 0, Invalid: 0, Rerun: 0, unconfirmed: 0 };
    for (const status of preview) counts[status] += 1;
    return counts;
  }, [preview]);

  const addSelectionTo = (label: Ava256FrameLabel) => {
    if (!selection) return;
    const [a, b] = selection;
    setText((prev) => ({ ...prev, [label]: formatRanges([...parsed[label].ranges, [a, b]]) }));
    setSelection(null);
  };

  // Every frame still unlabeled in the preview (saved "unconfirmed" and not in
  // any pending range) -> contiguous ranges appended to this label's row.
  const addUnlabeledTo = (label: Ava256FrameLabel) => {
    const runs: Range[] = [];
    preview.forEach((status, i) => {
      if (status !== "unconfirmed") return;
      const pos = i + 1;
      const last = runs[runs.length - 1];
      if (last && last[1] === pos - 1) last[1] = pos;
      else runs.push([pos, pos]);
    });
    if (!runs.length) return;
    setText((prev) => ({ ...prev, [label]: formatRanges([...parsed[label].ranges, ...runs]) }));
  };

  const removeRange = (label: Ava256FrameLabel, index: number) => {
    setText((prev) => ({ ...prev, [label]: formatRanges(parsed[label].ranges.filter((_, i) => i !== index)) }));
  };

  const onApply = async () => {
    const ranges: Ava256RangesByLabel = {};
    for (const label of AVA256_FRAME_LABELS) ranges[label] = parsed[label].ranges;
    await applyRanges(ranges);
  };

  useEffect(() => {
    if (dragAnchor === null) return;
    const up = () => setDragAnchor(null);
    window.addEventListener("mouseup", up);
    return () => window.removeEventListener("mouseup", up);
  }, [dragAnchor]);

  if (!selectedCapture) {
    return <div className="text-slate-400">Select an Ava-256 capture on the left.</div>;
  }
  const capture = dataset?.captures.find((c) => c.capture_id === selectedCapture);
  const currentFrame = frames[position - 1];

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h2 className="text-lg font-semibold text-slate-100 break-all">{selectedCapture}</h2>
        <span className="text-xs text-slate-400">
          {isDone ? "Done -- labels archived, read-only" : capture?.label_status ?? ""}
        </span>
      </div>

      {capture?.has_video ? (
        <div className="rounded-lg border border-slate-800 bg-slate-900/40 p-4">
          <video
            key={selectedCapture}
            ref={videoRef}
            controls
            className="w-full max-h-[60vh]"
            src={api.getAva256VideoUrl(selectedCapture)}
            onLoadedMetadata={(event) => setDuration(event.currentTarget.duration)}
          />
          <div className="mt-2 flex flex-wrap items-center gap-3 text-sm">
            <button className="rounded bg-slate-800 px-2 py-1 hover:bg-slate-700" onClick={() => seek(position - 1)}>
              ◂ Prev
            </button>
            <button className="rounded bg-slate-800 px-2 py-1 hover:bg-slate-700" onClick={() => seek(position + 1)}>
              Next ▸
            </button>
            <span className="tabular-nums text-slate-200">
              #{position} / {n}
              {currentFrame && (
                <>
                  {" "}· frame {currentFrame.frame_id} ·{" "}
                  <span className={STATUS_STYLE[currentFrame.status].text}>{currentFrame.status}</span>
                  {currentFrame.skin_error_px != null && (
                    <span className={currentFrame.skin_error_px > SKIN_ERROR_FLAG_PX ? "text-rose-300" : "text-slate-400"}>
                      {" "}· skin error {currentFrame.skin_error_px.toFixed(2)} px
                    </span>
                  )}
                </>
              )}
            </span>
            <label className="flex items-center gap-1 text-xs text-slate-400">
              Go to #
              <input
                type="number"
                min={1}
                max={n}
                className="w-20 rounded bg-slate-800 border border-slate-700 px-1.5 py-0.5 text-slate-100"
                onKeyDown={(event) => {
                  if (event.key === "Enter") seek(Number((event.target as HTMLInputElement).value));
                }}
              />
            </label>
          </div>
        </div>
      ) : (
        <div className="text-sm text-slate-400">No line render video for this capture.</div>
      )}

      {isFramesLoading && <p className="text-xs text-slate-400">Loading frames…</p>}

      {n > 0 && (
        <div className="rounded-lg border border-slate-800 bg-slate-900/40 p-4 space-y-3">
          <div className="flex items-center justify-between text-xs text-slate-400">
            <span>
              Timeline -- click a frame to seek{isDone ? "" : ", drag to select a range"}
            </span>
            <span className="tabular-nums">
              {AVA256_FRAME_LABELS.map((label) => (
                <span key={label} className={classNames("ml-3", STATUS_STYLE[label].text)}>
                  {label} {previewCounts[label]}
                </span>
              ))}
              <span className="ml-3 text-slate-400">unlabeled {previewCounts.unconfirmed}</span>
            </span>
          </div>
          <div className="flex h-8 w-full select-none overflow-hidden rounded">
            {preview.map((status, i) => {
              const pos = i + 1;
              const inSelection = selection && pos >= selection[0] && pos <= selection[1];
              return (
                <div
                  key={pos}
                  title={`#${pos} · ${frames[i].frame_id} · ${status}` +
                    (frames[i].skin_error_px != null ? ` · skin error ${frames[i].skin_error_px!.toFixed(2)} px` : "")}
                  className={classNames(
                    "relative flex-1 min-w-0 cursor-pointer",
                    conflicts.has(pos) ? "bg-red-600 animate-pulse" : STATUS_STYLE[status].cell,
                    status !== savedStatuses[i] && "opacity-70",
                    inSelection && "brightness-150"
                  )}
                  onMouseDown={() => {
                    if (isDone) {
                      seek(pos);
                      return;
                    }
                    setDragAnchor(pos);
                    setSelection([pos, pos]);
                    seek(pos);
                  }}
                  onMouseEnter={() => {
                    if (dragAnchor !== null) setSelection([Math.min(dragAnchor, pos), Math.max(dragAnchor, pos)]);
                  }}
                >
                  {inSelection && <div className="absolute inset-x-0 top-0 h-1 bg-white" />}
                  {(frames[i].skin_error_px ?? 0) > SKIN_ERROR_FLAG_PX && (
                    <div className="absolute inset-x-0 bottom-0 h-1 bg-black/60" />
                  )}
                  {pos === position && <div className="absolute inset-y-0 left-1/2 w-0.5 -translate-x-1/2 bg-white" />}
                </div>
              );
            })}
          </div>

          {!isDone && (
            <>
              <div className="flex flex-wrap items-center gap-2 text-xs">
                <span className="text-slate-400">Selection:</span>
                <button
                  className="rounded bg-slate-800 px-2 py-1 hover:bg-slate-700"
                  onClick={() => setSelection((prev) => {
                    const end = prev ? prev[1] : position;
                    return [Math.min(position, end), Math.max(position, end)];
                  })}
                >
                  Start at playhead
                </button>
                <button
                  className="rounded bg-slate-800 px-2 py-1 hover:bg-slate-700"
                  onClick={() => setSelection((prev) => {
                    const start = prev ? prev[0] : position;
                    return [Math.min(start, position), Math.max(start, position)];
                  })}
                >
                  End at playhead
                </button>
                <span className="tabular-nums text-slate-200">
                  {selection ? `#${selection[0]}-#${selection[1]}` : "none"}
                </span>
                {AVA256_FRAME_LABELS.map((label) => (
                  <button
                    key={label}
                    disabled={!selection}
                    className={classNames(
                      "rounded border px-2 py-1 disabled:opacity-40 disabled:cursor-not-allowed hover:bg-slate-800",
                      STATUS_STYLE[label].chip
                    )}
                    onClick={() => addSelectionTo(label)}
                  >
                    Add to {label}
                  </button>
                ))}
              </div>

              <div className="flex flex-wrap items-center gap-2 text-xs">
                <span className="text-slate-400">Remaining unlabeled ({previewCounts.unconfirmed}):</span>
                {(["Valid", "Rerun"] as const).map((label) => (
                  <button
                    key={label}
                    disabled={previewCounts.unconfirmed === 0}
                    className={classNames(
                      "rounded border px-2 py-1 disabled:opacity-40 disabled:cursor-not-allowed hover:bg-slate-800",
                      STATUS_STYLE[label].chip
                    )}
                    title={`Add every frame that is still unlabeled (not in any range) to the ${label} row -- then Apply to save`}
                    onClick={() => addUnlabeledTo(label)}
                  >
                    Add Unlabeled to {label}
                  </button>
                ))}
              </div>

              <div className="space-y-2">
                {AVA256_FRAME_LABELS.map((label) => (
                  <div key={label} className="flex flex-wrap items-center gap-2">
                    <span className={classNames("w-16 text-sm font-medium", STATUS_STYLE[label].text)}>{label}</span>
                    <input
                      type="text"
                      value={text[label]}
                      onChange={(event) => setText((prev) => ({ ...prev, [label]: event.target.value }))}
                      placeholder="e.g. 1-100, 150-300"
                      className="min-w-0 flex-1 rounded bg-slate-800 border border-slate-700 px-2 py-1 text-sm text-slate-100 placeholder:text-slate-500 focus:outline-none focus:border-brand-400"
                    />
                    <div className="flex w-full flex-wrap gap-1 pl-[4.5rem]">
                      {parsed[label].ranges.map(([a, b], index) => (
                        <span
                          key={`${a}-${b}-${index}`}
                          className={classNames("rounded-full border px-2 py-0.5 text-[11px] tabular-nums", STATUS_STYLE[label].chip)}
                        >
                          {a === b ? a : `${a}-${b}`}
                          <button className="ml-1 opacity-70 hover:opacity-100" onClick={() => removeRange(label, index)}>
                            ×
                          </button>
                        </span>
                      ))}
                    </div>
                  </div>
                ))}
              </div>

              {parseErrors.length > 0 && <p className="text-xs text-rose-400">{parseErrors.join(" · ")}</p>}
              {conflicts.size > 0 && (
                <p className="text-xs text-rose-400">
                  {conflicts.size} frame(s) are in more than one range (red on the timeline).
                </p>
              )}
              <p className="text-xs text-slate-500">
                Frames not in any range keep their saved label. Removing a range doesn't unlabel frames; relabel them instead.
              </p>

              <div className="flex flex-wrap items-center gap-2">
                <button
                  className="rounded bg-brand-500 px-3 py-1.5 text-sm text-white hover:bg-brand-400 disabled:opacity-40 disabled:cursor-not-allowed"
                  disabled={isSaving || parseErrors.length > 0 || conflicts.size > 0 || !isDirty}
                  onClick={() => void onApply()}
                >
                  {isSaving ? "Saving…" : "Apply"}
                </button>
                <button
                  className="rounded bg-slate-800 px-3 py-1.5 text-sm text-slate-300 hover:bg-slate-700"
                  onClick={() => setText(EMPTY_TEXT)}
                >
                  Clear
                </button>
                <button
                  className="rounded bg-slate-800 px-3 py-1.5 text-sm text-slate-300 hover:bg-slate-700"
                  onClick={() => setText(rangesFromStatuses(savedStatuses))}
                >
                  Reset to saved
                </button>

                <div className="ml-auto flex items-center gap-2">
                  {confirmingDone ? (
                    <>
                      <span className="text-xs text-slate-300">
                        Export {savedStatuses.filter((s) => s === "Valid").length} valid frame(s) and archive the labels?
                      </span>
                      <button
                        className="rounded bg-slate-800 px-3 py-1.5 text-sm text-slate-300 hover:bg-slate-700"
                        onClick={() => setConfirmingDone(false)}
                      >
                        Cancel
                      </button>
                      <button
                        className="rounded bg-emerald-600 px-3 py-1.5 text-sm text-white hover:bg-emerald-500 disabled:opacity-40"
                        disabled={isFinishing}
                        onClick={() => void finish().then(() => setConfirmingDone(false))}
                      >
                        {isFinishing ? "Exporting…" : "Confirm Done"}
                      </button>
                    </>
                  ) : (
                    <button
                      className="rounded bg-emerald-700 px-3 py-1.5 text-sm text-white hover:bg-emerald-600 disabled:opacity-40 disabled:cursor-not-allowed"
                      disabled={savedUnconfirmed > 0 || isDirty}
                      title={
                        savedUnconfirmed > 0
                          ? `${savedUnconfirmed} frame(s) are still unlabeled`
                          : isDirty
                            ? "Apply your pending changes first"
                            : "Export valid frames and archive the labels"
                      }
                      onClick={() => setConfirmingDone(true)}
                    >
                      Done
                    </button>
                  )}
                </div>
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
}
