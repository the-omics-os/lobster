import React, { useCallback, useState, useEffect } from "react";
import { Box, Text } from "ink";
import Spinner from "ink-spinner";
import { AssistantRuntimeProvider } from "@assistant-ui/react-ink";
import { useRuntime } from "./hooks/useRuntime.js";
import { useCancelHandler } from "./hooks/useCancelHandler.js";
import { useSlashCommands } from "./hooks/useSlashCommands.js";
import { useConnectionStatus } from "./hooks/useConnectionStatus.js";
import { ErrorBoundary } from "./components/ErrorBoundary.js";
import { Header } from "./components/Header.js";
import { Thread } from "./components/Thread.js";
import { Composer } from "./components/Composer.js";
import { StatusBar } from "./components/StatusBar.js";
import { ActivityFeed } from "./components/ActivityFeed.js";
import { TemplateSelector } from "./components/TemplateSelector.js";
import {
  ConfirmPromptUI,
  SelectPromptUI,
  TextInputPromptUI,
  ThresholdSliderUI,
  CellTypeSelectorUI,
  QCDashboardUI,
} from "./components/HITL/index.js";
import type { AppConfig } from "./config.js";
import { fetchFeatureFlags, type FeatureFlags } from "./api/featureFlags.js";
import { fetchTemplates, type PromptTemplate } from "./api/templates.js";

export function App({ config }: { config: AppConfig }) {
  const { runtime, appState, sessionId } = useRuntime(config);
  const handleCancel = useCallback(() => {
    runtime.thread.cancelRun();
  }, [runtime]);
  const cancelState = useCancelHandler(handleCancel);
  const slashCmds = useSlashCommands(appState, config, sessionId);
  const connection = useConnectionStatus();

  const [flags, setFlags] = useState<FeatureFlags | undefined>();
  const [templates, setTemplates] = useState<PromptTemplate[]>([]);
  const [showTemplates, setShowTemplates] = useState(false);

  // Fetch feature flags on startup
  useEffect(() => {
    fetchFeatureFlags(config).then(setFlags);
  }, [config.apiUrl]);

  // Fetch templates on new session (no --session-id)
  useEffect(() => {
    if (config.sessionId) return;
    fetchTemplates(config).then((tpls) => {
      if (tpls.length > 0) {
        setTemplates(tpls);
        setShowTemplates(true);
      }
    });
  }, [config.apiUrl, config.sessionId]);

  const handleTemplateSelect = useCallback(
    (text: string) => {
      setShowTemplates(false);
      runtime.thread.append({
        role: "user",
        content: [{ type: "text", text }],
      });
    },
    [runtime],
  );

  const handleTemplateDismiss = useCallback(() => {
    setShowTemplates(false);
  }, []);

  // Handle /exit
  useEffect(() => {
    if (slashCmds.exitRequested) {
      process.exit(0);
    }
  }, [slashCmds.exitRequested]);

  return (
    <ErrorBoundary>
      <AssistantRuntimeProvider runtime={runtime}>
        <ConfirmPromptUI />
        <SelectPromptUI />
        <TextInputPromptUI />
        <ThresholdSliderUI />
        <CellTypeSelectorUI />
        <QCDashboardUI />
        <Box flexDirection="column">
          <Header
            agentName={appState.activeAgent ?? undefined}
            sessionTitle={appState.sessionTitle ?? undefined}
            sessionId={sessionId}
          />
          {connection.state === "reconnecting" && (
            <Box paddingX={1}>
              <Text color="yellow">
                <Spinner type="line" />{" "}
                Reconnecting... (attempt {connection.retryCount})
              </Text>
            </Box>
          )}
          {connection.state === "error" && (
            <Box paddingX={1}>
              <Text color="red">
                Connection lost: {connection.error ?? "unknown error"}
              </Text>
            </Box>
          )}
          {showTemplates ? (
            <TemplateSelector
              templates={templates}
              onSelect={handleTemplateSelect}
              onDismiss={handleTemplateDismiss}
            />
          ) : (
            <>
              <Thread />
              {slashCmds.loading && (
                <Box paddingX={1}>
                  <Text color="yellow">Running command...</Text>
                </Box>
              )}
              {slashCmds.commandOutput && (
                <Box paddingX={1} marginY={1}>
                  <Text color="gray">{slashCmds.commandOutput}</Text>
                </Box>
              )}
              <Composer onIntercept={slashCmds.handleInput} />
            </>
          )}
          {cancelState.showWarning && (
            <Text color="yellow">Press Ctrl+C again to cancel</Text>
          )}
          <ActivityFeed events={appState.activityEvents} />
          <StatusBar
            appState={appState}
            sessionId={sessionId}
            flags={flags}
          />
        </Box>
      </AssistantRuntimeProvider>
    </ErrorBoundary>
  );
}
