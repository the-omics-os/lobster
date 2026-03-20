import { useState, useEffect, useCallback, useRef } from "react";
import { useDataStreamRuntime } from "@assistant-ui/react-data-stream";
import type { AppConfig } from "../config.js";
import { freshAuthHeaders } from "../config.js";
import { hydrateMessages } from "../utils/hydration.js";
import { resolveSessionId } from "../api/sessions.js";
import { apiFetch } from "../api/apiClient.js";
import {
  processStatePatch,
} from "../utils/stateHandlers.js";
import { useSSEReconnect } from "./useSSEReconnect.js";
import {
  applyAppStatePatch,
  clearRunActivity,
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
    resolveSessionId(config).then(setSessionId).catch((error) => {
      const message = error instanceof Error ? error.message : String(error);
      let alertMessage: string;
      if (message.includes("401") || message.includes("Authentication")) {
        alertMessage = "Not authenticated. Run: lobster cloud login";
      } else if (message.includes("ECONNREFUSED") || message.includes("fetch failed")) {
        alertMessage = "Cannot reach backend. Is the server running?";
      } else {
        alertMessage = `Failed to create session: ${message}`;
      }
      applyAppStatePatch(appStateStore, "alerts", [
        { level: "error", title: "Session Error", message: alertMessage },
      ]);
    });
  }, [config.apiUrl, config.sessionId]);

  // Refs for callbacks that need current values
  const configRef = useRef(config);
  configRef.current = config;
  const sseRef = useRef(sse);
  sseRef.current = sse;
  const sessionIdRef = useRef(sessionId);
  sessionIdRef.current = sessionId;

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

  // Post-stream REST fallback — refetch modalities/files/plots (matches web app onFinish pattern)
  const refetchAfterStream = useCallback(async () => {
    const sid = sessionIdRef.current;
    const cfg = configRef.current;
    if (!sid) return;

    const fetches = [
      apiFetch<{ modalities?: unknown[] }>(cfg, `/sessions/${sid}/modalities`).then(
        (d) => applyAppStatePatch(appStateStore, "modalities", d.modalities ?? []),
      ),
      apiFetch<{ files?: unknown[] }>(cfg, `/sessions/${sid}/files`).then(
        (d) => applyAppStatePatch(appStateStore, "files", d.files ?? []),
      ),
      apiFetch<{ plots?: unknown[] }>(cfg, `/sessions/${sid}/plots`).then(
        (d) => applyAppStatePatch(appStateStore, "plots", d.plots ?? []),
      ),
    ];
    await Promise.allSettled(fetches);
  }, [appStateStore]);

  // Error handler — display in TUI instead of crashing
  const onError = useCallback((error: unknown) => {
    const message = error instanceof Error ? error.message : String(error);

    let title = "Connection Error";
    let alertMessage: string;
    if (message.includes("401")) {
      alertMessage = "Authentication expired. Run: lobster cloud login";
    } else if (message.includes("429")) {
      alertMessage = "Rate limited. Wait a moment and try again.";
    } else if (message.includes("402")) {
      alertMessage = "Budget exhausted. Upgrade your tier at app.omics-os.com";
    } else if (message.includes("security token") || message.includes("UnrecognizedClientException")) {
      title = "Provider Error";
      alertMessage = "AWS credentials invalid or expired. Run: aws configure\nOr switch provider: lobster config --provider gemini";
    } else if (message.includes("ConverseStream") || message.includes("InvokeModel") || message.includes("bedrock")) {
      title = "Provider Error";
      alertMessage = "AWS Bedrock access denied. Check IAM permissions or switch provider:\n  lobster config --provider gemini";
    } else if (message.includes("ECONNREFUSED") || message.includes("fetch failed")) {
      alertMessage = "Cannot reach backend. Is the server running?";
    } else if (message.includes("timeout") || message.includes("ETIMEDOUT")) {
      alertMessage = "Request timed out. Check your network connection.";
    } else {
      alertMessage = `Stream error: ${message}`;
    }

    applyAppStatePatch(appStateStore, "alerts", [
      { level: "error", title, message: alertMessage },
    ]);
    clearRunActivity(appStateStore);
  }, [appStateStore]);

  // Finish handler — reset active agent + REST fallback + auto-rename
  const onFinish = useCallback(async () => {
    clearRunActivity(appStateStore);
    await refetchAfterStream();

    // Auto-rename session from session_title patch
    const sid = sessionIdRef.current;
    const title = appStateStore.getState().sessionTitle;
    if (sid && title) {
      apiFetch(configRef.current, `/sessions/${sid}`, {
        method: "PATCH",
        body: { name: title },
      }).catch(() => {});
    }
  }, [appStateStore, refetchAfterStream]);

  // Async headers function — refreshes expired OAuth tokens before each request
  const headers = useCallback(async (): Promise<Record<string, string>> => {
    const auth = await freshAuthHeaders(configRef.current);
    const reconnect = sseRef.current.reconnectHeaders();
    return { ...auth, ...reconnect };
  }, []);

  const api = `${config.apiUrl}/sessions/${sessionId ?? "pending"}/chat/stream`;

  const runtime = useDataStreamRuntime({
    api,
    headers,
    // Cloud backend uses assistant-stream DataStreamResponse (no [DONE] marker).
    // Local ink_launcher uses SSE format with [DONE] (ui-message-stream).
    protocol: config.isCloud ? "data-stream" : "ui-message-stream",
    onData,
    onError,
    onFinish,
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
