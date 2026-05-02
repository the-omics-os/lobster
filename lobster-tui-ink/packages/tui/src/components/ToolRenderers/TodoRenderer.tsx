import React from "react";
import { Box, Text } from "ink";
import type { ToolCallMessagePartProps } from "@assistant-ui/react-ink";
import { theme } from "../../theme.js";

interface TodoItem {
  task: string;
  done?: boolean;
  status?: string;
}

/**
 * Renders todo/task lists from write_todos tool calls.
 */
export function TodoRenderer({ result, isError }: ToolCallMessagePartProps) {
  if (isError || !result) {
    return null;
  }

  const items: TodoItem[] = Array.isArray(result)
    ? (result as TodoItem[])
    : typeof result === "object" && result !== null && "todos" in result
      ? ((result as Record<string, unknown>).todos as TodoItem[])
      : [];

  if (!items.length) return null;

  return (
    <Box flexDirection="column">
      {items.map((item, i) => (
        <Box key={i} gap={1}>
          <Text color={item.done ? theme.success : theme.warning}>
            {item.done ? "\u2713" : "\u25CB"}
          </Text>
          <Text>{item.task ?? String(item)}</Text>
        </Box>
      ))}
    </Box>
  );
}
