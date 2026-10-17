import { create } from "zustand";
import { immer } from "zustand/middleware/immer";

import { api } from "@/lib/api";
import type { Ava256DatasetResponse, Ava256FrameRow, Ava256RangesByLabel } from "@/lib/types";
import { useAnnotationStore } from "@/state/useAnnotationStore";

// Ava-256 slice, separate from useAnnotationStore so the Nersemble/FaceScape
// code paths stay untouched. One capture = one keyline video + per-frame CSV
// (backend services/ava256_labels.py).
interface Ava256State {
  dataset?: Ava256DatasetResponse;
  isLoading: boolean;
  error?: string;
  selectedCapture?: string;
  frames: Ava256FrameRow[];
  isDone: boolean;
  isFramesLoading: boolean;
  isSaving: boolean;
  isFinishing: boolean;
  fetchDataset: () => Promise<void>;
  selectCapture: (captureId: string) => Promise<void>;
  applyRanges: (ranges: Ava256RangesByLabel) => Promise<boolean>;
  finish: () => Promise<boolean>;
}

function toast(tone: "success" | "error", message: string) {
  useAnnotationStore.setState((state) => {
    state.toast = { id: Date.now(), tone, message };
  });
}

// fetchJSON errors look like 'Request failed (409): {"detail":"..."}'.
function errorText(error: unknown): string {
  const message = error instanceof Error ? error.message : String(error);
  const match = message.match(/^Request failed \((\d+)\): (.*)$/s);
  if (!match) return message;
  try {
    const detail = JSON.parse(match[2]).detail;
    return typeof detail === "string" ? detail : JSON.stringify(detail);
  } catch {
    return match[2] || message;
  }
}

export const useAva256Store = create<Ava256State>()(
  immer((set, get) => ({
    dataset: undefined,
    isLoading: false,
    error: undefined,
    selectedCapture: undefined,
    frames: [],
    isDone: false,
    isFramesLoading: false,
    isSaving: false,
    isFinishing: false,

    async fetchDataset() {
      set((state) => {
        state.isLoading = true;
        state.error = undefined;
      });
      try {
        const dataset = await api.getAva256Dataset();
        set((state) => {
          state.dataset = dataset;
        });
      } catch (error) {
        set((state) => {
          state.error = errorText(error);
        });
      } finally {
        set((state) => {
          state.isLoading = false;
        });
      }
    },

    async selectCapture(captureId: string) {
      set((state) => {
        state.selectedCapture = captureId;
        state.frames = [];
        state.isDone = false;
        state.isFramesLoading = true;
      });
      try {
        const response = await api.getAva256Frames(captureId);
        if (get().selectedCapture !== captureId) return;
        set((state) => {
          state.frames = response.frames;
          state.isDone = response.is_done;
        });
      } catch (error) {
        toast("error", `Failed to load frames for ${captureId}: ${errorText(error)}`);
      } finally {
        set((state) => {
          state.isFramesLoading = false;
        });
      }
    },

    async applyRanges(ranges: Ava256RangesByLabel) {
      const captureId = get().selectedCapture;
      if (!captureId) return false;
      set((state) => {
        state.isSaving = true;
      });
      try {
        const response = await api.labelAva256Frames(captureId, ranges);
        set((state) => {
          if (state.selectedCapture === captureId) state.frames = response.frames;
          const capture = state.dataset?.captures.find((c) => c.capture_id === captureId);
          if (capture) {
            capture.counts = { Valid: 0, Invalid: 0, Rerun: 0, unconfirmed: 0 };
            for (const frame of response.frames) capture.counts[frame.status] += 1;
          }
        });
        toast(
          "success",
          `Labels saved for ${captureId}` + (response.warnings.length ? ` (${response.warnings.join("; ")})` : "")
        );
        return true;
      } catch (error) {
        toast("error", `Failed to save labels: ${errorText(error)}`);
        return false;
      } finally {
        set((state) => {
          state.isSaving = false;
        });
      }
    },

    async finish() {
      const captureId = get().selectedCapture;
      if (!captureId) return false;
      set((state) => {
        state.isFinishing = true;
      });
      try {
        const response = await api.finishAva256Capture(captureId);
        set((state) => {
          if (state.selectedCapture === captureId) state.isDone = true;
          const capture = state.dataset?.captures.find((c) => c.capture_id === captureId);
          if (capture) {
            capture.is_done = true;
            capture.label_status = "confirmed";
          }
        });
        toast(
          "success",
          `${captureId} done: ${response.counts.Valid ?? 0} valid frame(s)` +
            (response.export_path ? ` exported to ${response.export_path}` : " (nothing exported)")
        );
        return true;
      } catch (error) {
        toast("error", `Done failed: ${errorText(error)}`);
        return false;
      } finally {
        set((state) => {
          state.isFinishing = false;
        });
      }
    }
  }))
);
