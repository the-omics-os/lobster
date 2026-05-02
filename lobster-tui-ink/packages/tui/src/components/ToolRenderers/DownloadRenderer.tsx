import React from "react";
import { Box, Text } from "ink";
import type { ToolCallMessagePartProps } from "@assistant-ui/react-ink";
import { ProgressBar, IndeterminateSpinner } from "../Progress.js";
import { theme } from "../../theme.js";

/**
 * Renders download progress with file name.
 */
export function DownloadRenderer({
  args,
  result,
  isError,
}: ToolCallMessagePartProps) {
  const a = args as Record<string, unknown> | undefined;
  const fileName = a?.file_name ?? a?.filename ?? a?.path ?? "file";

  if (isError) {
    return (
      <Box gap={1}>
        <Text color={theme.error}>{"\u2716"}</Text>
        <Text color={theme.error}>Download failed: {String(fileName)}</Text>
      </Box>
    );
  }

  if (result !== undefined) {
    return (
      <Box gap={1}>
        <Text color={theme.success}>{"\u2713"}</Text>
        <Text>Downloaded: <Text bold>{String(fileName)}</Text></Text>
      </Box>
    );
  }

  // Still in progress
  const r = result as Record<string, unknown> | undefined;
  const progress = typeof r?.progress === "number" ? r.progress : undefined;

  if (progress !== undefined) {
    return <ProgressBar label={String(fileName)} value={progress} />;
  }

  return <IndeterminateSpinner label={`Downloading ${String(fileName)}...`} />;
}
