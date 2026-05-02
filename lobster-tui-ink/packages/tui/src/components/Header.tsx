import React from "react";
import { Box, Text } from "ink";
import { useAuiState } from "@assistant-ui/store";
import { useStore } from "zustand";
import { useTheme } from "../hooks/useTheme.js";
import type { AppStateStore } from "../utils/appStateStore.js";

export interface HeaderProps {
  appStateStore: AppStateStore;
  sessionId?: string;
}

export function Header({ appStateStore, sessionId }: HeaderProps) {
  const theme = useTheme();
  const hasMessages = useAuiState((state) => state.thread.messages.length > 0);
  const sessionTitle = useStore(appStateStore, (state) => state.sessionTitle);
  const displaySession =
    sessionTitle ?? (sessionId ? truncateMiddle(sessionId, 28) : undefined);

  return (
    <Box>
      {hasMessages ? (
        <Text bold color={theme.primary}>
          Lobster AI
        </Text>
      ) : null}
      <Box flexGrow={1} />
      {displaySession && (
        <Text dimColor>{displaySession}</Text>
      )}
    </Box>
  );
}

function truncateMiddle(value: string, max: number): string {
  if (value.length <= max) {
    return value;
  }

  const visible = Math.max(1, max - 3);
  const head = Math.ceil(visible / 2);
  const tail = Math.floor(visible / 2);
  return `${value.slice(0, head)}...${value.slice(-tail)}`;
}
