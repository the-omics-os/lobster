import React from "react";
import { Box } from "ink";
import { AssistantRuntimeProvider } from "@assistant-ui/react-ink";
import { useRuntime } from "./hooks/useRuntime.js";
import { Thread } from "./components/Thread.js";
import type { AppConfig } from "./config.js";

export function App({ config }: { config: AppConfig }) {
  const runtime = useRuntime(config);

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <Box flexDirection="column">
        <Thread />
      </Box>
    </AssistantRuntimeProvider>
  );
}
