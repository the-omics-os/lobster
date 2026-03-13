/**
 * HITL Confirm component — renders Y/n confirmation for tool interrupts.
 *
 * When a tool has status "requires-action", this shows a ConfirmInput
 * and calls addResult(true/false) to resume the graph.
 */
import React from "react";
import { Box, Text } from "ink";
import { ConfirmInput } from "@inkjs/ui";
import { makeAssistantToolUI } from "@assistant-ui/react-ink";

interface ConfirmArgs {
  message?: string;
  title?: string;
}

export const ConfirmPromptUI = makeAssistantToolUI<ConfirmArgs, boolean>({
  toolName: "ask_for_confirmation",
  render: ({ args, status, addResult }) => {
    if (status.type !== "requires-action") {
      return (
        <Box>
          <Text color="gray">
            {args.title ?? "Confirm"}: {status.type === "complete" ? "answered" : "..."}
          </Text>
        </Box>
      );
    }

    return (
      <Box flexDirection="column" marginY={1}>
        <Text bold color="yellow">
          {args.title ?? "Confirmation required"}
        </Text>
        {args.message && <Text>{args.message}</Text>}
        <Box marginTop={1}>
          <Text>Confirm? (Y/n) </Text>
          <ConfirmInput
            defaultChoice="confirm"
            onConfirm={() => addResult(true)}
            onCancel={() => addResult(false)}
          />
        </Box>
      </Box>
    );
  },
});
