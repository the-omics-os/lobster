import React, { useCallback, useState, useEffect } from "react";
import { Box, Text } from "ink";
import { AssistantRuntimeProvider } from "@assistant-ui/react-ink";
import { BrailleSpinner } from "./components/BrailleSpinner.js";
import { useRuntime } from "./hooks/useRuntime.js";
import { useCancelHandler } from "./hooks/useCancelHandler.js";
import { useSlashCommands } from "./hooks/useSlashCommands.js";
import { useTheme } from "./hooks/useTheme.js";
import { ErrorBoundary } from "./components/ErrorBoundary.js";
import { Header } from "./components/Header.js";
import { Thread } from "./components/Thread.js";
import { Composer } from "./components/Composer.js";
import { StatusBar } from "./components/StatusBar.js";
import { ActivityFeed } from "./components/ActivityFeed.js";
import { Alerts } from "./components/Alerts.js";
import { TemplateSelector } from "./components/TemplateSelector.js";
import { WelcomeAnimation } from "./components/WelcomeAnimation.js";
import { CommandOutput } from "./components/CommandOutput.js";
import {
  ConfirmPromptUI,
  SelectPromptUI,
  TextInputPromptUI,
  ThresholdSliderUI,
  CellTypeSelectorUI,
  QCDashboardUI,
} from "./components/HITL/index.js";
import type { AppConfig } from "./config.js";
import type { FeatureFlags } from "./api/featureFlags.js";
import type { PromptTemplate } from "./api/templates.js";
import type { Resource } from "./api/resources.js";
import { fetchBootstrap } from "./api/bootstrap.js";

export function App({ config }: { config: AppConfig }) {
  const theme = useTheme();
  const [backendReady, setBackendReady] = useState(false);
  const [backendError, setBackendError] = useState<string | null>(null);

  // Poll /health until backend agent is initialized
  useEffect(() => {
    if (config.isCloud) {
      // Cloud mode — backend is always ready
      setBackendReady(true);
      return;
    }

    let cancelled = false;
    const poll = async () => {
      while (!cancelled) {
        try {
          const resp = await fetch(`${config.apiUrl}/health`);
          if (resp.ok) {
            const data = (await resp.json()) as { ready: boolean; error: string | null };
            if (data.ready) {
              setBackendReady(true);
              return;
            }
            if (data.error) {
              setBackendError(data.error);
              return;
            }
          }
        } catch {
          // Server not ready yet — keep polling
        }
        await new Promise((r) => setTimeout(r, 300));
      }
    };
    poll();
    return () => { cancelled = true; };
  }, [config.apiUrl, config.isCloud]);

  const { runtime, appStateStore, sessionId, clearThread } = useRuntime(config);
  const handleCancel = useCallback(() => {
    runtime.thread.cancelRun();
  }, [runtime]);
  const cancelState = useCancelHandler(handleCancel);
  const slashCmds = useSlashCommands(appStateStore, config, sessionId);

  const [flags, setFlags] = useState<FeatureFlags | undefined>();
  const [templates, setTemplates] = useState<PromptTemplate[]>([]);
  const [showTemplates, setShowTemplates] = useState(false);
  const [resources, setResources] = useState<Resource[]>([]);

  // Bootstrap: single fetch for flags, resources, and templates (after backend ready)
  useEffect(() => {
    if (!backendReady) return;
    fetchBootstrap(config).catch(() => {}).then((bootstrap) => {
      if (!bootstrap) return;
      setFlags(bootstrap.flags);
      setResources(bootstrap.resources);
      if (!config.sessionId && bootstrap.templates.length > 0) {
        setTemplates(bootstrap.templates);
        setShowTemplates(true);
      }
    }).catch(() => {});
  }, [config.apiUrl, backendReady]);

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

  // Handle /clear — wipe terminal scrollback + reset thread and app state
  useEffect(() => {
    if (slashCmds.clearRequested) {
      process.stdout.write("\x1b[2J\x1b[3J\x1b[H");
      clearThread();
      slashCmds.resetClear();
    }
  }, [slashCmds.clearRequested, clearThread, slashCmds.resetClear]);

  // Loading gate — show spinner until backend is ready
  if (!backendReady) {
    return (
      <Box flexDirection="column" paddingX={1} paddingY={1}>
        {backendError ? (
          <Box flexDirection="column" marginTop={1}>
            <Text color={theme.error}>Failed to initialize agent: {backendError}</Text>
            <Text dimColor>Run &apos;lobster init&apos; to reconfigure.</Text>
          </Box>
        ) : (
          <Box flexDirection="column" marginTop={1} gap={1}>
            {!config.isResume ? <WelcomeAnimation /> : null}
            <BrailleSpinner label="Loading agent..." color={theme.primary} />
          </Box>
        )}
      </Box>
    );
  }

  return (
    <ErrorBoundary>
      <AssistantRuntimeProvider runtime={runtime}>
        <ConfirmPromptUI />
        <SelectPromptUI />
        <TextInputPromptUI />
        <ThresholdSliderUI />
        <CellTypeSelectorUI />
        <QCDashboardUI />
        {/* Header — printed once at start */}
        <Header
          appStateStore={appStateStore}
          sessionId={sessionId}
        />

        {/* Messages — <Static> completed + live streaming (inside Thread) */}
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
              <Box>
                <Text color={theme.warning}>Running command...</Text>
              </Box>
            )}
            {slashCmds.commandOutput && (
              <Box flexDirection="column">
                <CommandOutput output={slashCmds.commandOutput} />
              </Box>
            )}
            <ActivityFeed appStateStore={appStateStore} />
          </>
        )}

        <Alerts appStateStore={appStateStore} />

        {/* Composer — live area (re-renders in place) */}
        {!showTemplates && (
          <Composer
            onIntercept={slashCmds.handleInput}
            onSubmit={slashCmds.dismissOutput}
            resources={resources}
            appStateStore={appStateStore}
          />
        )}

        {/* Footer — live area */}
        {cancelState.showWarning && (
          <Text color={theme.warning}>Press Ctrl+C again to cancel</Text>
        )}
        <StatusBar
          appStateStore={appStateStore}
          sessionId={sessionId}
          flags={flags}
        />
      </AssistantRuntimeProvider>
    </ErrorBoundary>
  );
}
