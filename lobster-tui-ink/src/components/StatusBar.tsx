import React from "react";
import { Box, Text } from "ink";
import { useAuiState } from "@assistant-ui/store";
import { useStore } from "zustand";
import { BrailleSpinner } from "./BrailleSpinner.js";
import { useTheme } from "../hooks/useTheme.js";
import type { FeatureFlags } from "../api/featureFlags.js";
import type { AppStateStore } from "../utils/appStateStore.js";

interface StatusBarProps {
  appStateStore: AppStateStore;
  sessionId?: string;
  isStreaming?: boolean;
  flags?: FeatureFlags;
}

/**
 * Bottom-pinned status bar showing active agent, session ID, and token usage.
 * Updates from aui-state: patches (active_agent, token_usage).
 */
export function StatusBar({ appStateStore, sessionId, isStreaming, flags }: StatusBarProps) {
  const theme = useTheme();
  const activeAgent = useStore(appStateStore, (state) => state.activeAgent);
  const tokenUsage = useStore(appStateStore, (state) => state.tokenUsage);
  const runtimeRunning = useAuiState((state) => state.thread.isRunning);
  const running = isStreaming ?? runtimeRunning;
  const statusLabel = running ? activeAgent ?? "thinking" : "idle";

  const tokenDisplay =
    tokenUsage && (tokenUsage.promptTokens || tokenUsage.completionTokens)
      ? `${(tokenUsage.promptTokens ?? 0) + (tokenUsage.completionTokens ?? 0)} tok`
      : null;

  return (
    <Box>
      <Box gap={1}>
        {running ? (
          <BrailleSpinner color={theme.warning} animated={false} />
        ) : (
          <Text color={theme.success}>●</Text>
        )}
        {running && activeAgent ? (
          <Text color={theme.accent1}>{statusLabel}</Text>
        ) : (
          <Text dimColor>{statusLabel}</Text>
        )}
      </Box>
      <Box flexGrow={1} />
      <Box gap={2}>
        {tokenDisplay && <Text dimColor>{tokenDisplay}</Text>}
        {sessionId && <Text dimColor>{sessionId.slice(0, 8)}</Text>}
      </Box>
    </Box>
  );
}
