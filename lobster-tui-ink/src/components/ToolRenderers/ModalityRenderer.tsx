import React from "react";
import { Box, Text } from "ink";
import type { ToolCallMessagePartProps } from "@assistant-ui/react-ink";
import { theme } from "../../theme.js";

/**
 * Renders modality load results: "Loaded: GSE12345 (2000 obs x 18000 vars)"
 */
export function ModalityRenderer({ result, isError }: ToolCallMessagePartProps) {
  if (isError) {
    return (
      <Box gap={1}>
        <Text color={theme.error}>{"\u2716"}</Text>
        <Text color={theme.error}>Modality load failed: {String(result)}</Text>
      </Box>
    );
  }

  if (!result) {
    return (
      <Box gap={1}>
        <Text color={theme.warning}>{"\u27F3"}</Text>
        <Text dimColor>Loading modality...</Text>
      </Box>
    );
  }

  const r = result as Record<string, unknown>;
  const name = r.name ?? r.accession ?? "unknown";
  const obs = r.n_obs ?? r.observations ?? "?";
  const vars = r.n_vars ?? r.variables ?? "?";

  return (
    <Box gap={1}>
      <Text color={theme.success}>{"\u2713"}</Text>
      <Text>
        Loaded: <Text bold>{String(name)}</Text> ({String(obs)} obs x{" "}
        {String(vars)} vars)
      </Text>
    </Box>
  );
}
