import React from "react";
import { Box, Text } from "ink";
import { theme } from "../theme.js";

type AlertLevel = "success" | "warning" | "error" | "info";

interface AlertBlockProps {
  level: AlertLevel;
  title?: string;
  message: string;
}

const ALERT_LEVELS: Record<AlertLevel, { color: string; icon: string }> = {
  success: { color: theme.success, icon: "\u2713" },
  warning: { color: theme.warning, icon: "\u26A0" },
  error: { color: theme.error, icon: "\u2716" },
  info: { color: theme.info, icon: "\u2139" },
};

export function AlertBlock({ level, title, message }: AlertBlockProps) {
  const config = ALERT_LEVELS[level];

  return (
    <Box
      borderStyle="single"
      borderLeft
      borderTop={false}
      borderRight={false}
      borderBottom={false}
      borderColor={config.color}
      flexDirection="column"
      paddingLeft={1}
    >
      <Box gap={1}>
        <Text color={config.color} bold>
          {config.icon}
        </Text>
        <Text>
          {title ? (
            <Text color={config.color} bold>
              {title}
            </Text>
          ) : null}
          {title ? " " : ""}
          <Text>{message}</Text>
        </Text>
      </Box>
    </Box>
  );
}
