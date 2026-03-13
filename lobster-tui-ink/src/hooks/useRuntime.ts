import { useState, useEffect, useCallback, useRef } from "react";
import { useDataStreamRuntime } from "@assistant-ui/react-data-stream";
import type { ThreadMessageLike } from "@assistant-ui/react-ink";
import type { AppConfig } from "../config.js";
import { authHeaders } from "../config.js";
import { hydrateMessages } from "../utils/hydration.js";
import { resolveSessionId } from "../api/sessions.js";
import {
  createInitialState,
  applyStatePatch,
  processStatePatch,
  type AppState,
} from "../utils/stateHandlers.js";
import { useSSEReconnect, loadLastEventId } from "./useSSEReconnect.js";

export function useRuntime(config: AppConfig) {
  const [sessionId, setSessionId] = useState<string | undefined>(
    config.sessionId
  );
  const [initialMessages, setInitialMessages] = useState<
    ThreadMessageLike[] | undefined
  >(undefined);
  const [appState, setAppState] = useState<AppState>(createInitialState);
  const appStateRef = useRef(appState);

  const sse = useSSEReconnect(sessionId);

  // Resolve session ID on mount
  useEffect(() => {
    resolveSessionId(config).then(setSessionId);
  }, [config.apiUrl, config.sessionId]);

  // Hydrate message history when resuming a session
  useEffect(() => {
    if (!sessionId) return;
    hydrateMessages(config, sessionId).then((msgs) => {
      if (msgs.length > 0) {
        setInitialMessages(msgs);
      }
    });
  }, [sessionId]);

  // State patch handler (protocol §1.3) + SSE event ID tracking
  const onData = useCallback(
    (data: { type: string; name: string; data: unknown; id?: string }) => {
      // Track SSE event IDs for reconnection (protocol §5.2)
      if (data.id) {
        sse.trackEventId(data.id);
      }

      const result = processStatePatch(data.name, data.data);
      if (result) {
        setAppState((prev) => {
          const next = applyStatePatch(prev, result.key, result.data);
          appStateRef.current = next;
          return next;
        });
      }
    },
    [sse.trackEventId],
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
    initialMessages,
    onData,
  });

  return { runtime, appState, sessionId, sse };
}
