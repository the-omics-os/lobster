/**
 * HITL Text Input component — renders free-form text input for tool interrupts.
 *
 * Renders @inkjs/ui TextInput, calls addResult(value) on Enter.
 */
import React from "react";
import { Box, Text } from "ink";
import { TextInput } from "@inkjs/ui";
import { makeAssistantToolUI } from "@assistant-ui/react-ink";

interface TextInputArgs {
  message?: string;
  title?: string;
  placeholder?: string;
  default_value?: string;
}

export const TextInputPromptUI = makeAssistantToolUI<TextInputArgs, string>({
  toolName: "ask_for_text_input",
  render: ({ args, status, addResult }) => {
    if (status.type !== "requires-action") {
      return (
        <Box>
          <Text color="gray">
            {args.title ?? "Input"}: {status.type === "complete" ? "answered" : "..."}
          </Text>
        </Box>
      );
    }

    return (
      <Box flexDirection="column" marginY={1}>
        <Text bold color="yellow">
          {args.title ?? "Input required"}
        </Text>
        {args.message && <Text>{args.message}</Text>}
        <Box marginTop={1}>
          <Text>{"> "}</Text>
          <TextInput
            placeholder={args.placeholder ?? "Type your answer..."}
            defaultValue={args.default_value}
            onSubmit={(value) => addResult(value)}
          />
        </Box>
      </Box>
    );
  },
});
