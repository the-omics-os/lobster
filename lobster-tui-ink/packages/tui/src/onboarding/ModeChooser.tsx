import React from "react";
import { Box, Text } from "ink";
import { Select } from "@inkjs/ui";
import { useTheme } from "../hooks/useTheme.js";

interface Props {
  onSelect: (mode: "cloud" | "local") => void;
}

export function ModeChooser({ onSelect }: Props) {
  const theme = useTheme();

  return (
    <Box flexDirection="column" paddingX={2} marginY={1}>
      <Text bold color={theme.primary}>
        Lobster AI
      </Text>
      <Text color={theme.textMuted}>
        The Operating System for Biology
      </Text>

      <Box marginTop={1}>
        <Text bold>How would you like to connect?</Text>
      </Box>

      <Box marginTop={1}>
        <Select
          options={[
            {
              label: "Cloud (recommended) — No setup required",
              value: "cloud",
            },
            {
              label: "Local — Run agents on your machine (requires Python)",
              value: "local",
            },
          ]}
          onChange={(value) => onSelect(value as "cloud" | "local")}
        />
      </Box>
    </Box>
  );
}
