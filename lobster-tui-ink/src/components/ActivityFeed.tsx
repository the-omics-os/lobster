import React from "react";
import { Box, Text } from "ink";

interface ActivityEvent {
  type: string;
  agent?: string;
  tool_name?: string;
  to_agent?: string;
  task_description?: string;
  duration_ms?: number;
  error?: string;
}

interface ActivityFeedProps {
  events: unknown[];
}

const RING_SIZE = 5;

/**
 * Activity feed from aui-state: activity_events.
 * Ring buffer display showing last 5 events (matches Go TUI toolFeed).
 */
export function ActivityFeed({ events }: ActivityFeedProps) {
  const typed = events as ActivityEvent[];
  const display = typed.slice(-RING_SIZE);

  if (!display.length) return null;

  return (
    <Box flexDirection="column">
      {display.map((evt, i) => (
        <ActivityEventLine key={i} event={evt} />
      ))}
    </Box>
  );
}

function ActivityEventLine({ event }: { event: ActivityEvent }) {
  switch (event.type) {
    case "tool_start":
      return (
        <Text dimColor>
          [{event.agent ?? "?"}] {event.tool_name}...
        </Text>
      );
    case "tool_complete":
      return (
        <Text dimColor>
          [{event.agent ?? "?"}] {event.tool_name} (
          {event.duration_ms ?? 0}ms)
        </Text>
      );
    case "tool_error":
      return (
        <Text color="red">
          [{event.agent ?? "?"}] {event.tool_name} failed
          {event.error ? `: ${event.error}` : ""}
        </Text>
      );
    case "agent_handoff":
      return (
        <Text color="cyan">
          → Delegating to {event.to_agent}
          {event.task_description ? `: ${event.task_description}` : ""}
        </Text>
      );
    default:
      return <Text dimColor>[{event.type}]</Text>;
  }
}
