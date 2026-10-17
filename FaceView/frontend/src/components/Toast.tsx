import { useEffect } from "react";

import { useAnnotationStore } from "@/state/useAnnotationStore";

const TOAST_DURATION_MS = 3000;

export function Toast() {
  const { toast, dismissToast } = useAnnotationStore((state) => ({
    toast: state.toast,
    dismissToast: state.dismissToast
  }));

  useEffect(() => {
    if (!toast) return;
    const id = toast.id;
    const timeoutId = window.setTimeout(() => dismissToast(id), TOAST_DURATION_MS);
    return () => window.clearTimeout(timeoutId);
  }, [toast, dismissToast]);

  if (!toast) return null;

  const toneClass = toast.tone === "success"
    ? "bg-emerald-600 border-emerald-500"
    : "bg-rose-600 border-rose-500";

  return (
    <div
      role="status"
      className={`fixed bottom-16 right-6 z-50 rounded-lg border px-4 py-2 text-sm text-white shadow-lg ${toneClass}`}
    >
      {toast.message}
    </div>
  );
}
