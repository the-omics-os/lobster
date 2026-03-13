import React, { useCallback, useState, useEffect } from "react";
import { Box, Text } from "ink";
import { AssistantRuntimeProvider } from "@assistant-ui/react-ink";
import { BrailleSpinner } from "./components/BrailleSpinner.js";
import { theme } from "./theme.js";
import { useRuntime } from "./hooks/useRuntime.js";
import { useCancelHandler } from "./hooks/useCancelHandler.js";
import { useSlashCommands } from "./hooks/useSlashCommands.js";
import { useLayout } from "./hooks/useLayout.js";
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

  // 4-layer layout engine
  const activityEventCount = appState.activityEvents?.length ?? 0;
  const layout = useLayout({
    inputLineCount: 1,
    completionMenuOpen: false,
    activityEventCount: Math.min(activityEventCount, 5),
  });

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
        <Box flexDirection="column" height={layout.rows}>
          {/* Layer 1: Header (fixed) */}
          <Box height={layout.headerRows} flexShrink={0}>
            <Header
              agentName={appState.activeAgent ?? undefined}
              sessionTitle={appState.sessionTitle ?? undefined}
              sessionId={sessionId}
            />
          </Box>

          {/* Connection status banners (overlay into viewport space) */}
          {sse.degradedLevel === "reconnecting" && (
            <Box paddingX={1}>
              <BrailleSpinner label={`Reconnecting... (attempt ${sse.retryCount})`} color={theme.warning} />
            </Box>
          )}
          {sse.degradedLevel === "lost" && (
            <Box paddingX={1}>
              <Text color={theme.error}>
                Connection lost. Retrying... (Ctrl+C to exit)
              </Text>
            </Box>
          )}

          {/* Layer 2: Viewport (flex) */}
          {showTemplates ? (
            <Box height={layout.viewportRows} flexShrink={0}>
              <TemplateSelector
                templates={templates}
                onSelect={handleTemplateSelect}
                onDismiss={handleTemplateDismiss}
              />
            </Box>
          ) : (
            <>
              <Thread viewportHeight={layout.viewportRows} />
              {slashCmds.loading && (
                <Box paddingX={1}>
                  <Text color={theme.warning}>Running command...</Text>
                </Box>
              )}
              {slashCmds.commandOutput && (
                <Box paddingX={1} marginY={1}>
                  <Text color={theme.textMuted}>{slashCmds.commandOutput}</Text>
                </Box>
              )}
            </>
          )}

          {/* Layer 3: Input (dynamic) */}
          <Box flexShrink={0}>
            <Composer onIntercept={slashCmds.handleInput} resources={resources} />
          </Box>

          {/* Layer 4: Footer (fixed) */}
          <Box flexDirection="column" flexShrink={0}>
            {cancelState.showWarning && (
              <Text color={theme.warning}>Press Ctrl+C again to cancel</Text>
            )}
            <ActivityFeed events={appState.activityEvents} />
            <StatusBar
              appState={appState}
              sessionId={sessionId}
              flags={flags}
            />
          </Box>
        </Box>
      </AssistantRuntimeProvider>
    </ErrorBoundary>
  );
}
