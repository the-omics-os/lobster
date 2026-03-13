import { useDataStreamRuntime } from "@assistant-ui/react-data-stream";
import type { AppConfig } from "../config.js";

export function useRuntime(config: AppConfig) {
  const headers: Record<string, string> = {};
  if (config.token) {
    headers["Authorization"] = `Bearer ${config.token}`;
  }

  return useDataStreamRuntime({
    api: `${config.apiUrl}/sessions/${config.sessionId ?? "new"}/chat/stream`,
    headers: Object.keys(headers).length > 0 ? headers : undefined,
  });
}
