import React, { useCallback } from "react";
import { Box, Text } from "ink";
import { AssistantRuntimeProvider } from "@assistant-ui/react-ink";
import { useRuntime } from "./hooks/useRuntime.js";
import { useCancelHandler } from "./hooks/useCancelHandler.js";
import { Header } from "./components/Header.js";
import { Thread } from "./components/Thread.js";
import { Composer } from "./components/Composer.js";
import { StatusBar } from "./components/StatusBar.js";
import { ActivityFeed } from "./components/ActivityFeed.js";
import type { AppConfig } from "./config.js";

export function App({ config }: { config: AppConfig }) {
  const { runtime, appState, sessionId } = useRuntime(config);
  const handleCancel = useCallback(() => {
    runtime.thread.cancelRun();
  }, [runtime]);
  const cancelState = useCancelHandler(handleCancel);

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <Box flexDirection="column">
        <Header
          agentName={appState.activeAgent ?? undefined}
          sessionTitle={appState.sessionTitle ?? undefined}
          sessionId={sessionId}
        />
        <Thread />
        <Composer />
        {cancelState.showWarning && (
          <Text color="yellow">Press Ctrl+C again to cancel</Text>
        )}
        <ActivityFeed events={appState.activityEvents} />
        <StatusBar
          appState={appState}
          sessionId={sessionId}
        />
      </Box>
    </AssistantRuntimeProvider>
  );
}
