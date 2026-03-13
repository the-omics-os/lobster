import { useState, useEffect } from "react";
import { useDataStreamRuntime } from "@assistant-ui/react-data-stream";
import type { ThreadMessageLike } from "@assistant-ui/react-ink";
import type { AppConfig } from "../config.js";
import { hydrateMessages } from "../utils/hydration.js";

export function useRuntime(config: AppConfig) {
  const [initialMessages, setInitialMessages] = useState<
    ThreadMessageLike[] | undefined
  >(undefined);

  // Hydrate message history when resuming a session
  useEffect(() => {
    if (!config.sessionId) return;
    hydrateMessages(config, config.sessionId).then((msgs) => {
      if (msgs.length > 0) {
        setInitialMessages(msgs);
      }
    });
  }, [config.sessionId]);

  const headers: Record<string, string> = {};
  if (config.token) {
    headers["Authorization"] = `Bearer ${config.token}`;
  }

  return useDataStreamRuntime({
    api: `${config.apiUrl}/sessions/${config.sessionId ?? "new"}/chat/stream`,
    headers: Object.keys(headers).length > 0 ? headers : undefined,
    initialMessages,
  });
}
