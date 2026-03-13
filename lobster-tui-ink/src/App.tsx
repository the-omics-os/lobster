import React from "react";
import { Text } from "ink";
import { AssistantRuntimeProvider } from "@assistant-ui/react-ink";
import { useRuntime } from "./hooks/useRuntime.js";
import type { AppConfig } from "./config.js";

function AppInner({ config }: { config: AppConfig }) {
  return (
    <Text>
      Connected to {config.apiUrl}
      {config.sessionId ? ` (session: ${config.sessionId})` : ""}
    </Text>
  );
}

export function App({ config }: { config: AppConfig }) {
  const runtime = useRuntime(config);

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <AppInner config={config} />
    </AssistantRuntimeProvider>
  );
}
