import React from "react";
import { useStdout } from "ink";
import { ToolCallFallback } from "@assistant-ui/react-ink";
import type {
  ToolCallMessagePart,
  ToolCallMessagePartStatus,
} from "@assistant-ui/core";

interface ToolCallRendererProps {
  part: ToolCallMessagePart;
  index: number;
}

function inferStatus(part: ToolCallMessagePart): ToolCallMessagePartStatus {
  if (part.isError)
    return { type: "incomplete", reason: "error" };
  if (part.interrupt)
    return { type: "requires-action", reason: "interrupt" };
  if (part.result !== undefined)
    return { type: "complete" };
  return { type: "running" };
}

/**
 * Generic tool call renderer with terminal-width-aware truncation.
 * Wraps the built-in ToolCallFallback with adaptive max lines.
 */
export function ToolCallRenderer({ part }: ToolCallRendererProps) {
  const { stdout } = useStdout();
  const columns = stdout?.columns ?? 80;

  // Wider terminals get more visible lines
  const maxArgLines = columns >= 120 ? 15 : columns >= 80 ? 10 : 5;
  const maxResultLines = columns >= 120 ? 15 : columns >= 80 ? 10 : 5;

  return (
    <ToolCallFallback
      type="tool-call"
      status={inferStatus(part)}
      toolName={part.toolName}
      toolCallId={part.toolCallId}
      args={part.args}
      argsText={part.argsText}
      result={part.result}
      isError={part.isError}
      interrupt={part.interrupt}
      maxArgLines={maxArgLines}
      maxResultLines={maxResultLines}
      maxResultPreviewLines={1}
    />
  );
}
