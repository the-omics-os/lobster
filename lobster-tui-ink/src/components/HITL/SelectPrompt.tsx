/**
 * HITL Select component — renders option list for tool interrupts.
 *
 * Renders @inkjs/ui Select with options from tool args,
 * calls addResult(selected_value) to resume the graph.
 */
import React from "react";
import { Box, Text } from "ink";
import { Select } from "@inkjs/ui";
import { makeAssistantToolUI } from "@assistant-ui/react-ink";

interface SelectOption {
  label: string;
  value: string;
}

interface SelectArgs {
  message?: string;
  title?: string;
  options: SelectOption[];
}

export const SelectPromptUI = makeAssistantToolUI<SelectArgs, string>({
  toolName: "ask_for_selection",
  render: ({ args, status, addResult }) => {
    if (status.type !== "requires-action") {
      return (
        <Box>
          <Text color="gray">
            {args.title ?? "Select"}: {status.type === "complete" ? "answered" : "..."}
          </Text>
        </Box>
      );
    }

    const options = (args.options ?? []).map((opt) =>
      typeof opt === "string"
        ? { label: opt, value: opt }
        : { label: opt.label, value: opt.value }
    );

    return (
      <Box flexDirection="column" marginY={1}>
        <Text bold color="yellow">
          {args.title ?? "Selection required"}
        </Text>
        {args.message && <Text>{args.message}</Text>}
        <Box marginTop={1}>
          <Select options={options} onChange={(value) => addResult(value)} />
        </Box>
      </Box>
    );
  },
});
