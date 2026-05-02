/**
 * Connection status tracking with auto-retry.
 * Shows reconnecting spinner on connection loss, exponential backoff.
 */

import { useState, useCallback, useRef } from "react";

export type ConnectionState = "connected" | "reconnecting" | "error";

interface ConnectionStatus {
  state: ConnectionState;
  error: string | null;
  retryCount: number;
}

const MAX_RETRIES = 5;
const BASE_DELAY_MS = 1000;

export function useConnectionStatus() {
  const [status, setStatus] = useState<ConnectionStatus>({
    state: "connected",
    error: null,
    retryCount: 0,
  });
  const retryTimerRef = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);

  const onError = useCallback((error: string) => {
    setStatus((prev) => {
      if (prev.retryCount >= MAX_RETRIES) {
        return { state: "error", error, retryCount: prev.retryCount };
      }
      return {
        state: "reconnecting",
        error,
        retryCount: prev.retryCount + 1,
      };
    });
  }, []);

  const onConnected = useCallback(() => {
    if (retryTimerRef.current) {
      clearTimeout(retryTimerRef.current);
    }
    setStatus({ state: "connected", error: null, retryCount: 0 });
  }, []);

  const scheduleRetry = useCallback(
    (retryFn: () => void) => {
      const delay = BASE_DELAY_MS * Math.pow(2, status.retryCount - 1);
      retryTimerRef.current = setTimeout(retryFn, delay);
    },
    [status.retryCount],
  );

  return { ...status, onError, onConnected, scheduleRetry };
}
