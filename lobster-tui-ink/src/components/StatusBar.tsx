import React from "react";
import { Box, Text } from "ink";
import { BrailleSpinner } from "./BrailleSpinner.js";
import { theme } from "../theme.js";
import type { AppState } from "../utils/stateHandlers.js";
import type { FeatureFlags } from "../api/featureFlags.js";

interface StatusBarProps {
  appState: AppState;
  sessionId?: string;
  isStreaming?: boolean;
  flags?: FeatureFlags;
}

/**
 * Bottom-pinned status bar showing active agent, session ID, and token usage.
 * Updates from aui-state: patches (active_agent, token_usage).
 */
export function StatusBar({ appState, sessionId, isStreaming, flags }: StatusBarProps) {
  const { activeAgent, tokenUsage } = appState;

  const tokenDisplay =
    tokenUsage && (tokenUsage.promptTokens || tokenUsage.completionTokens)
      ? `${(tokenUsage.promptTokens ?? 0) + (tokenUsage.completionTokens ?? 0)} tok`
      : null;

  return (
    <Box
      borderStyle="single"
      borderColor={theme.textMuted}
      paddingX={1}
      justifyContent="space-between"
    >
      <Box gap={1}>
        {isStreaming ? (
          <BrailleSpinner color={theme.warning} />
        ) : (
          <Text color={theme.success}>●</Text>
        )}
        {activeAgent ? (
          <Text color={theme.accent1}>{activeAgent}</Text>
        ) : (
          <Text dimColor>idle</Text>
        )}
      </Box>
      <Box gap={2}>
        {tokenDisplay && <Text dimColor>{tokenDisplay}</Text>}
        {sessionId && <Text dimColor>{sessionId.slice(0, 8)}</Text>}
      </Box>
    </Box>
  );
}
