import React from "react";
import { Box, Text, useStdout } from "ink";
import type {
  ToolCallMessagePart,
  ToolCallMessagePartStatus,
} from "@assistant-ui/core";
import { useTheme } from "../../hooks/useTheme.js";

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

function prettifyName(value: string) {
  return value.replace(/_/g, " ");
}

function stringifyPreview(value: unknown): string | null {
  if (value === undefined || value === null) {
    return null;
  }

  if (typeof value === "string") {
    const trimmed = value.trim();
    return trimmed.length > 0 ? trimmed : null;
  }

  if (typeof value === "number" || typeof value === "boolean") {
    return String(value);
  }

  if (Array.isArray(value)) {
    if (value.length === 0) {
      return "[]";
    }
    if (value.length <= 3) {
      try {
        return JSON.stringify(value);
      } catch {
        return `${value.length} items`;
      }
    }
    return `${value.length} items`;
  }

  if (typeof value === "object") {
    try {
      const record = value as Record<string, unknown>;
      const entries = Object.entries(record).slice(0, 4);
      if (entries.length === 0) {
        return "{}";
      }
      const preview = entries
        .map(([key, entryValue]) => `${key}=${stringifyScalar(entryValue)}`)
        .join(", ");
      const suffix = Object.keys(record).length > entries.length ? ", ..." : "";
      return preview + suffix;
    } catch {
      return "[object]";
    }
  }

  return null;
}

function stringifyScalar(value: unknown) {
  if (typeof value === "string") {
    return value.length > 32 ? `${value.slice(0, 31)}…` : value;
  }
  if (typeof value === "number" || typeof value === "boolean") {
    return String(value);
  }
  if (value === null || value === undefined) {
    return String(value);
  }
  if (Array.isArray(value)) {
    return `${value.length} items`;
  }
  return "{…}";
}

function wrapPreview(text: string | null, width: number, maxLines: number) {
  if (!text) {
    return [];
  }

  const normalized = text.replace(/\s+/g, " ").trim();
  if (!normalized) {
    return [];
  }

  const words = normalized.split(" ");
  const lines: string[] = [];
  let current = "";

  for (const word of words) {
    const candidate = current ? `${current} ${word}` : word;
    if (candidate.length <= width) {
      current = candidate;
      continue;
    }

    if (current) {
      lines.push(current);
      if (lines.length === maxLines) {
        return withEllipsis(lines);
      }
    }

    if (word.length <= width) {
      current = word;
      continue;
    }

    lines.push(word.slice(0, Math.max(1, width - 1)) + "…");
    if (lines.length === maxLines) {
      return withEllipsis(lines);
    }
    current = "";
  }

  if (current) {
    lines.push(current);
  }

  return lines.length > maxLines ? withEllipsis(lines.slice(0, maxLines)) : lines;
}

function withEllipsis(lines: string[]) {
  if (lines.length === 0) {
    return lines;
  }

  const next = [...lines];
  const last = next[next.length - 1] ?? "";
  next[next.length - 1] = last.endsWith("…") ? last : `${last}…`;
  return next;
}

/**
 * Generic tool call renderer using Lobster-native text lines rather than
 * the boxed assistant-ui fallback. This keeps tool activity visually aligned
 * with the transcript and avoids nested panel chrome inside messages.
 */
export function ToolCallRenderer({ part }: ToolCallRendererProps) {
  const theme = useTheme();
  const { stdout } = useStdout();
  const columns = Math.max(24, (stdout?.columns ?? 80) - 8);
  const status = inferStatus(part);
  const isRunning = status.type === "running";
  const isError = status.type === "incomplete" && status.reason === "error";
  const requiresAction = status.type === "requires-action";
  const icon = isError ? "✗" : requiresAction ? "?" : isRunning ? "⋯" : "✓";
  const color = isError
    ? theme.error
    : requiresAction
      ? theme.warning
      : isRunning
        ? theme.warning
        : theme.success;
  const label = prettifyName(part.toolName);
  const argsPreview = wrapPreview(stringifyPreview(part.argsText ?? part.args), columns, 2);
  const resultPreview = wrapPreview(stringifyPreview(part.result), columns, 4);

  return (
    <Box flexDirection="column" marginLeft={2}>
      <Box gap={1}>
        <Text color={color}>{icon}</Text>
        <Text color={isError ? theme.error : theme.textMuted}>{label}</Text>
      </Box>
      {argsPreview.map((line, index) => (
        <Text key={`args-${index}`} color={theme.textMuted}>
          {"  "}args: {line}
        </Text>
      ))}
      {requiresAction && (
        <Text color={theme.warning}>{"  "}awaiting user input</Text>
      )}
      {!requiresAction &&
        resultPreview.map((line, index) => (
          <Text
            key={`result-${index}`}
            color={isError ? theme.error : theme.textMuted}
          >
            {"  "}result: {line}
          </Text>
        ))}
    </Box>
  );
}
