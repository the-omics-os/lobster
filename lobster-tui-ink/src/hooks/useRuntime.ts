import { useState, useEffect, useCallback, useRef } from "react";
import { useDataStreamRuntime } from "@assistant-ui/react-data-stream";
import type { AppConfig } from "../config.js";
import { authHeaders } from "../config.js";
import { hydrateMessages } from "../utils/hydration.js";
import { resolveSessionId } from "../api/sessions.js";
import {
  processStatePatch,
} from "../utils/stateHandlers.js";
import { useSSEReconnect } from "./useSSEReconnect.js";
import {
  applyAppStatePatch,
  createAppStateStore,
  resetAppStateStore,
} from "../utils/appStateStore.js";

export function useRuntime(config: AppConfig) {
  const [sessionId, setSessionId] = useState<string | undefined>(
    config.sessionId
  );
  const appStateStoreRef = useRef(createAppStateStore());
  const appStateStore = appStateStoreRef.current;

  const sse = useSSEReconnect(sessionId);

  // Resolve session ID on mount
  useEffect(() => {
    resolveSessionId(config).then(setSessionId);
  }, [config.apiUrl, config.sessionId]);

  // State patch handler (protocol §1.3) + SSE event ID tracking
  const onData = useCallback(
    (data: { type: string; name: string; data: unknown; id?: string }) => {
      // Track SSE event IDs for reconnection (protocol §5.2)
      if (data.id) {
        sse.trackEventId(data.id);
      }

      const result = processStatePatch(data.name, data.data);
      if (result) {
        applyAppStatePatch(appStateStore, result.key, result.data);
      }
    },
    [appStateStore, sse.trackEventId],
  );

  // Build headers: auth + Last-Event-ID for reconnection
  const headers: Record<string, string> = {
    ...authHeaders(config),
    ...sse.reconnectHeaders(),
  };

  const api = `${config.apiUrl}/sessions/${sessionId ?? "pending"}/chat/stream`;

  const runtime = useDataStreamRuntime({
    api,
    headers: Object.keys(headers).length > 0 ? headers : undefined,
    onData,
  });

  // Hydrate message history when resuming a session (deferred via runtime.thread.reset)
  useEffect(() => {
    if (!sessionId) return;
    hydrateMessages(config, sessionId).then((msgs) => {
      if (msgs.length > 0) {
        runtime.thread.reset(msgs);
      }
    });
  }, [sessionId]);

  /** Clear thread messages and reset app state (for /clear). */
  const clearThread = useCallback(() => {
    resetAppStateStore(appStateStore);
    runtime.thread.reset();
    // Request a new session so old messages don't rehydrate
    resolveSessionId({ ...config, sessionId: undefined }).then(setSessionId);
  }, [appStateStore, config, runtime]);

  return { runtime, appStateStore, sessionId, sse, clearThread };
}
