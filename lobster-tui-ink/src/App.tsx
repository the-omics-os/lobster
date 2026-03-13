import React from "react";
import { Box } from "ink";
import { AssistantRuntimeProvider } from "@assistant-ui/react-ink";
import { useRuntime } from "./hooks/useRuntime.js";
import { Header } from "./components/Header.js";
import { Thread } from "./components/Thread.js";
import { Composer } from "./components/Composer.js";
import { StatusBar } from "./components/StatusBar.js";
import type { AppConfig } from "./config.js";

export function App({ config }: { config: AppConfig }) {
  const { runtime, appState } = useRuntime(config);

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <Box flexDirection="column">
        <Header
          agentName={appState.activeAgent ?? undefined}
          sessionTitle={appState.sessionTitle ?? undefined}
          sessionId={config.sessionId}
        />
        <Thread />
        <Composer />
        <StatusBar
          appState={appState}
          sessionId={config.sessionId}
        />
      </Box>
    </AssistantRuntimeProvider>
  );
}
