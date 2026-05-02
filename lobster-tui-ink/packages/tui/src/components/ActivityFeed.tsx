import React from "react";
import { Box, Text } from "ink";
import { useAuiState } from "@assistant-ui/store";
import { useStore } from "zustand";
import { BrailleSpinner } from "./BrailleSpinner.js";
import { IndeterminateSpinner, ProgressBar } from "./Progress.js";
import { useTheme } from "../hooks/useTheme.js";
import type { ActivityEvent, ProgressEntry } from "../utils/stateHandlers.js";
import type { AppStateStore } from "../utils/appStateStore.js";

interface ActivityFeedProps {
  appStateStore: AppStateStore;
}

const RING_SIZE = 5;

function normalizeToolName(event: ActivityEvent) {
  return event.tool_name ?? event.tool ?? "tool";
}

function prettifyName(value: string) {
  return value.replace(/_/g, " ");
}

function normalizeToolKey(event: ActivityEvent) {
  return event.tool_call_id ?? event.id ?? null;
}

function formatDuration(durationMs: number | undefined) {
  if (typeof durationMs !== "number" || !Number.isFinite(durationMs) || durationMs <= 0) {
    return null;
  }
  return `${(durationMs / 1000).toFixed(1)}s`;
}

function shouldDisplayEvent(event: ActivityEvent) {
  return event.type !== "agent_content";
}

function compactEvents(events: ActivityEvent[]) {
  const compacted: ActivityEvent[] = [];

  for (const event of events) {
    if (!shouldDisplayEvent(event)) {
      continue;
    }

    const key = normalizeToolKey(event);

    if (key && (event.type === "tool_complete" || event.type === "tool_error")) {
      const existingIndex = compacted.findIndex(
        (candidate) => normalizeToolKey(candidate) === key,
      );
      if (existingIndex >= 0) {
        compacted[existingIndex] = {
          ...compacted[existingIndex],
          ...event,
        };
        continue;
      }
    }

    if (event.type === "agent_handoff" || event.type === "agent_transition") {
      const existingIndex = compacted.findIndex(
        (candidate) =>
          (candidate.type === "agent_handoff" || candidate.type === "agent_transition") &&
          candidate.from_agent === event.from_agent &&
          candidate.to_agent === event.to_agent &&
          candidate.status === event.status,
      );
      if (existingIndex >= 0) {
        const existing = compacted[existingIndex]!;
        compacted[existingIndex] =
          event.task_description && !existing.task_description
            ? { ...existing, ...event }
            : existing;
        continue;
      }
    }

    compacted.push(event);
  }

  return compacted.slice(-RING_SIZE);
}

/**
 * Activity feed from aui-state: activity_events.
 * Ring buffer display showing last 5 events (matches Go TUI toolFeed).
 */
export function ActivityFeed({ appStateStore }: ActivityFeedProps) {
  const theme = useTheme();
  const events = useStore(appStateStore, (state) => state.activityEvents);
  const progress = useStore(appStateStore, (state) => state.progress);
  const activeAgent = useStore(appStateStore, (state) => state.activeAgent);
  const runtimeRunning = useAuiState((state) => state.thread.isRunning);
  const display = compactEvents(events);
  const progressEntries = Object.values(progress ?? {}).filter((entry) => !entry.done);
  const activeLabel = activeAgent ? prettifyName(activeAgent) : null;

  if (!display.length && progressEntries.length === 0) return null;

  return (
    <Box flexDirection="column" marginTop={1} paddingLeft={2}>
      <Text color={theme.textMuted}>
        {runtimeRunning ? "Working" : "Recent work"}
        {runtimeRunning && activeLabel ? ` — ${activeLabel}` : ""}
      </Text>
      {progressEntries.map((entry) =>
        entry.total < 0 ? (
          <IndeterminateSpinner key={entry.id} label={entry.label} />
        ) : (
          <ProgressBar
            key={entry.id}
            label={entry.label}
            value={entry.total > 0 ? entry.current / entry.total : 0}
          />
        ),
      )}
      {display.map((evt, i) => (
        <ActivityEventLine key={i} event={evt} />
      ))}
    </Box>
  );
}

function ActivityEventLine({ event }: { event: ActivityEvent }) {
  const theme = useTheme();
  const toolName = prettifyName(normalizeToolName(event));
  const agent = event.agent ?? event.to_agent ?? event.from_agent;
  const agentLabel = agent ? prettifyName(agent) : null;

  switch (event.type) {
    case "tool_start":
      return (
        <Box gap={1}>
          <BrailleSpinner color={theme.warning} animated={false} />
          <Text color={theme.textMuted}>
            {toolName}
            {agentLabel ? ` · ${agentLabel}` : ""}
          </Text>
        </Box>
      );
    case "tool_complete":
    case "tool_error": {
      const failed = event.type === "tool_error" || event.error === true;
      const duration = formatDuration(event.duration_ms);
      const summary =
        typeof event.error === "string"
          ? event.error
          : event.summary ?? event.message ?? undefined;
      const summaryLabel =
        summary && summary !== duration ? summary : undefined;

      return (
        <Text>
          <Text color={failed ? theme.error : theme.success} bold>
            {failed ? "✗ " : "✓ "}
          </Text>
          <Text color={failed ? theme.error : theme.textMuted}>
            {toolName}
            {duration ? ` (${duration})` : ""}
            {agentLabel ? ` · ${agentLabel}` : ""}
            {summaryLabel ? ` — ${summaryLabel}` : ""}
          </Text>
        </Text>
      );
    }
    case "agent_handoff":
    case "agent_transition":
      return (
        <Text>
          <Text color={theme.accent2} italic>
            {event.status === "complete" ? "← " : "→ "}
          </Text>
          <Text color={theme.accent2}>
            {prettifyName(
              event.status === "complete"
                ? event.to_agent ?? event.from_agent ?? "supervisor"
                : event.to_agent ?? event.agent ?? "supervisor",
            )}
            {event.task_description ? ` — ${event.task_description}` : ""}
          </Text>
        </Text>
      );
    case "context_compaction":
      return (
        <Text>
          <Text color={theme.info}>i </Text>
          <Text color={theme.textMuted}>Context compacted</Text>
        </Text>
      );
    case "agent_content":
      return null;
    case "error":
      return <Text color={theme.error}>{event.message ?? "Unknown error"}</Text>;
    default:
      return <Text dimColor>[{event.type}]</Text>;
  }
}
