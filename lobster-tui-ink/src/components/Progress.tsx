import React from "react";
import { Box, Text } from "ink";
import Spinner from "ink-spinner";

interface ProgressBarProps {
  label: string;
  value: number; // 0-1
}

const BAR_WIDTH = 30;

/**
 * Determinate progress bar for downloads and long operations.
 * Wired to aui-state: progress patches.
 */
export function ProgressBar({ label, value }: ProgressBarProps) {
  const clamped = Math.max(0, Math.min(1, value));
  const filled = Math.round(clamped * BAR_WIDTH);
  const empty = BAR_WIDTH - filled;
  const pct = Math.round(clamped * 100);

  return (
    <Box gap={1}>
      <Text dimColor>{label}</Text>
      <Text>
        <Text color="green">{"█".repeat(filled)}</Text>
        <Text dimColor>{"░".repeat(empty)}</Text>
      </Text>
      <Text>{pct}%</Text>
    </Box>
  );
}

interface IndeterminateSpinnerProps {
  label: string;
}

/**
 * Indeterminate spinner for operations without known progress.
 */
export function IndeterminateSpinner({ label }: IndeterminateSpinnerProps) {
  return (
    <Box gap={1}>
      <Text color="yellow">
        <Spinner type="line" />
      </Text>
      <Text dimColor>{label}</Text>
    </Box>
  );
}
