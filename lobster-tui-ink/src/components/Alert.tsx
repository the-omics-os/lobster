import React from "react";
import { Box, Text } from "ink";

type AlertLevel = "success" | "warning" | "error" | "info";

interface AlertProps {
  level: AlertLevel;
  message: string;
}

const LEVEL_CONFIG: Record<AlertLevel, { icon: string; color: string }> = {
  success: { icon: "+", color: "green" },
  warning: { icon: "!", color: "yellow" },
  error: { icon: "x", color: "red" },
  info: { icon: "i", color: "blue" },
};

/**
 * Colored alert messages matching Go TUI renderAlert() style.
 */
export function Alert({ level, message }: AlertProps) {
  const { icon, color } = LEVEL_CONFIG[level];

  return (
    <Box gap={1}>
      <Text color={color} bold>
        [{icon}]
      </Text>
      <Text color={color}>{message}</Text>
    </Box>
  );
}
