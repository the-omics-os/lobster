import React from "react";
import { Box, Text } from "ink";
import { theme } from "../theme.js";

type AlertLevel = "success" | "warning" | "error" | "info";

interface AlertProps {
  level: AlertLevel;
  message: string;
}

const LEVEL_CONFIG: Record<AlertLevel, { icon: string; color: string }> = {
  success: { icon: "\u2713", color: theme.success },
  warning: { icon: "\u26A0", color: theme.warning },
  error: { icon: "\u2716", color: theme.error },
  info: { icon: "\u2139", color: theme.info },
};

/**
 * Parse title from message. Supports two formats:
 * - "[Title] rest of message"
 * - "Title: rest of message"
 */
function parseTitle(msg: string): { title?: string; body: string } {
  const bracketMatch = msg.match(/^\[([^\]]+)\]\s*(.*)$/s);
  if (bracketMatch) return { title: bracketMatch[1], body: bracketMatch[2] ?? "" };

  const colonIdx = msg.indexOf(": ");
  if (colonIdx > 0 && colonIdx < 40) {
    return { title: msg.slice(0, colonIdx), body: msg.slice(colonIdx + 2) };
  }

  return { body: msg };
}

/**
 * Colored alert messages matching Go TUI renderAlert() style.
 * Left border in level color. Icon per level. Title extracted from message.
 */
export function Alert({ level, message }: AlertProps) {
  const { icon, color } = LEVEL_CONFIG[level];
  const { title, body } = parseTitle(message);

  return (
    <Box borderStyle="single" borderLeft borderTop={false} borderRight={false} borderBottom={false} borderColor={color} paddingLeft={1} gap={1}>
      <Text color={color} bold>
        {icon}
      </Text>
      {title ? (
        <Text>
          <Text color={color} bold>{title}</Text>
          {body ? <Text>{` ${body}`}</Text> : null}
        </Text>
      ) : (
        <Text>{body}</Text>
      )}
    </Box>
  );
}
