import { useEffect } from "react";

import { api } from "@/lib/api";
import { useAnnotationStore } from "@/state/useAnnotationStore";

const RECONNECT_DELAY_MS = 1000;

export function useSequenceSocket(sequenceId?: string) {
  const handleSocketEvent = useAnnotationStore((state) => state.handleSocketEvent);

  useEffect(() => {
    if (!sequenceId) return;

    let socket: WebSocket | null = null;
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
    let cancelled = false;

    const connect = () => {
      socket = api.createSequenceSocket(sequenceId);

      socket.onmessage = (event) => {
        try {
          const payload = JSON.parse(event.data);
          handleSocketEvent(payload);
        } catch (error) {
          console.warn("Failed to parse socket payload", error);
        }
      };

      socket.onclose = () => {
        if (cancelled) return;
        // Backend restarted (or the connection otherwise dropped) — reconnect so
        // completion events like propagation_complete aren't silently lost.
        reconnectTimer = setTimeout(connect, RECONNECT_DELAY_MS);
      };

      socket.onerror = () => {
        socket?.close();
      };
    };

    connect();

    return () => {
      cancelled = true;
      if (reconnectTimer) clearTimeout(reconnectTimer);
      socket?.close();
    };
  }, [sequenceId, handleSocketEvent]);
}
