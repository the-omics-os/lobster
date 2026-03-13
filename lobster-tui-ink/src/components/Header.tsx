import React from "react";
import { Box, Text } from "ink";

export interface HeaderProps {
  agentName?: string;
  sessionTitle?: string;
  sessionId?: string;
}

export function Header({ agentName, sessionTitle, sessionId }: HeaderProps) {
  const displaySession =
    sessionTitle ?? (sessionId ? truncate(sessionId, 12) : undefined);

  return (
    <Box borderStyle="single" borderColor="red" paddingX={1}>
      <Text bold color="red">
        Lobster AI
      </Text>
      {agentName && (
        <Text>
          {"  "}
          <Text dimColor>[</Text>
          <Text color="yellow">{agentName}</Text>
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
