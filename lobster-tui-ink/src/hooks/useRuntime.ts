import { useState, useEffect, useCallback, useRef } from "react";
import { useDataStreamRuntime } from "@assistant-ui/react-data-stream";
import type { ThreadMessageLike } from "@assistant-ui/react-ink";
import type { AppConfig } from "../config.js";
import { hydrateMessages } from "../utils/hydration.js";
import {
  createInitialState,
  applyStatePatch,
  processStatePatch,
  type AppState,
  type StateKey,
} from "../utils/stateHandlers.js";

export function useRuntime(config: AppConfig) {
  const [initialMessages, setInitialMessages] = useState<
    ThreadMessageLike[] | undefined
  >(undefined);
  const [appState, setAppState] = useState<AppState>(createInitialState);
  const appStateRef = useRef(appState);

  // Hydrate message history when resuming a session
  useEffect(() => {
    if (!config.sessionId) return;
    hydrateMessages(config, config.sessionId).then((msgs) => {
      if (msgs.length > 0) {
        setInitialMessages(msgs);
      }
    });
  }, [config.sessionId]);

  // State patch handler (protocol §1.3)
  const onData = useCallback(
    (data: { type: string; name: string; data: unknown }) => {
      const result = processStatePatch(data.name, data.data);
      if (result) {
        setAppState((prev) => {
          const next = applyStatePatch(prev, result.key, result.data);
          appStateRef.current = next;
          return next;
        });
      }
    },
    []
  );

  const headers: Record<string, string> = {};
  if (config.token) {
    headers["Authorization"] = `Bearer ${config.token}`;
  }

  const runtime = useDataStreamRuntime({
    api: `${config.apiUrl}/sessions/${config.sessionId ?? "new"}/chat/stream`,
    headers: Object.keys(headers).length > 0 ? headers : undefined,
    initialMessages,
    onData,
  });

  return { runtime, appState };
}
