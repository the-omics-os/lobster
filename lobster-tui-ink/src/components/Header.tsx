import React from "react";
import { Box, Text } from "ink";
import { theme } from "../theme.js";

export interface HeaderProps {
  agentName?: string;
  sessionTitle?: string;
  sessionId?: string;
}

export function Header({ agentName, sessionTitle, sessionId }: HeaderProps) {
  const displaySession =
    sessionTitle ?? (sessionId ? truncate(sessionId, 12) : undefined);

  return (
    <Box borderStyle="single" borderColor={theme.primary} paddingX={1}>
      <Text bold color={theme.primary}>
        Lobster AI
      </Text>
      {agentName && (
        <Text>
          {"  "}
          <Text dimColor>[</Text>
          <Text color={theme.accent2}>{agentName}</Text>
          <Text dimColor>]</Text>
        </Text>
      )}
      {displaySession && (
        <Text>
          {"  "}
          <Text dimColor>{displaySession}</Text>
        </Text>
      )}
    </Box>
  );
}

function truncate(s: string, max: number): string {
  return s.length > max ? s.slice(0, max) + "..." : s;
}
