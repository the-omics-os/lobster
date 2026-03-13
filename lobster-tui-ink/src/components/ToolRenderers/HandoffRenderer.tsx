import React from "react";
import { Box, Text } from "ink";
import type { ToolCallMessagePartProps } from "@assistant-ui/react-ink";

const HANDOFF_PATTERN = /^handoff_to_(.+)$/;
const TRANSFER_BACK_PATTERN = /^transfer_back_to_(.+)$/;

/**
 * Checks if a tool name is a handoff/transfer tool.
 */
export function isHandoffTool(toolName: string): boolean {
  return HANDOFF_PATTERN.test(toolName) || TRANSFER_BACK_PATTERN.test(toolName);
}

/**
 * Renders agent handoff transitions matching Go TUI style:
 * "-> Delegating to transcriptomics_expert..."
 * "<- Returning to supervisor..."
 */
export function HandoffRenderer(props: ToolCallMessagePartProps) {
  const { toolName, args } = props;

  const handoffMatch = toolName.match(HANDOFF_PATTERN);
  const transferMatch = toolName.match(TRANSFER_BACK_PATTERN);

  const agentName = handoffMatch?.[1] ?? transferMatch?.[1] ?? toolName;
  const isReturn = !!transferMatch;
  const displayName = agentName.replace(/_/g, " ");
  const reason =
    typeof args === "object" && args !== null && "reason" in args
      ? String((args as Record<string, unknown>).reason)
      : undefined;

  return (
    <Box gap={1}>
      <Text color={isReturn ? "blue" : "cyan"}>
        {isReturn ? "<-" : "->"}
      </Text>
      <Text>
        {isReturn ? "Returning to " : "Delegating to "}
        <Text bold color={isReturn ? "blue" : "cyan"}>
          {displayName}
        </Text>
        {reason ? <Text dimColor> — {reason}</Text> : null}
      </Text>
    </Box>
  );
}
