import React, { useCallback, useState, useEffect } from "react";
import { Box, Text } from "ink";
import Spinner from "ink-spinner";
import { AssistantRuntimeProvider } from "@assistant-ui/react-ink";
import { useRuntime } from "./hooks/useRuntime.js";
import { useCancelHandler } from "./hooks/useCancelHandler.js";
import { useSlashCommands } from "./hooks/useSlashCommands.js";
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
import { fetchResources, type Resource } from "./api/resources.js";

export function App({ config }: { config: AppConfig }) {
  const { runtime, appState, sessionId, sse } = useRuntime(config);
  const handleCancel = useCallback(() => {
    runtime.thread.cancelRun();
  }, [runtime]);
  const cancelState = useCancelHandler(handleCancel);
  const slashCmds = useSlashCommands(appState, config, sessionId);

  const [flags, setFlags] = useState<FeatureFlags | undefined>();
  const [templates, setTemplates] = useState<PromptTemplate[]>([]);
  const [showTemplates, setShowTemplates] = useState(false);
  const [resources, setResources] = useState<Resource[]>([]);

  // Fetch feature flags + resources on startup
  useEffect(() => {
    fetchFeatureFlags(config).then(setFlags);
    fetchResources(config).then(setResources);
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

  // Handle degraded mode exit (protocol §5.5: > 2min with 3 failed retries)
  useEffect(() => {
    if (sse.degradedLevel === "exit" && sessionId) {
      console.log(
        `\nSession ${sessionId} may still be running server-side.\n` +
        `Resume: lobster cloud chat --resume ${sessionId}\n` +
        `View:   app.omics-os.com/sessions/${sessionId}\n`,
      );
      process.exit(1);
    }
  }, [sse.degradedLevel, sessionId]);

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
          {sse.degradedLevel === "reconnecting" && (
            <Box paddingX={1}>
              <Text color="yellow">
                <Spinner type="line" />{" "}
                Reconnecting... (attempt {sse.retryCount})
              </Text>
            </Box>
          )}
          {sse.degradedLevel === "lost" && (
            <Box paddingX={1}>
              <Text color="red">
                Connection lost. Retrying... (Ctrl+C to exit)
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
              <Composer onIntercept={slashCmds.handleInput} resources={resources} />
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
