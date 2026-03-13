import React from "react";
import { Box, Text } from "ink";
import { BrailleSpinner } from "./BrailleSpinner.js";
import { BrailleProgressBar } from "./BrailleProgressBar.js";

interface ProgressBarProps {
  label: string;
  value: number; // 0-1
}

/**
 * Determinate progress bar for downloads and long operations.
 * Delegates to BrailleProgressBar for consistent visual style.
 */
export function ProgressBar({ label, value }: ProgressBarProps) {
  return <BrailleProgressBar label={label} value={value} />;
}

interface IndeterminateSpinnerProps {
  label: string;
}

/**
 * Indeterminate spinner for operations without known progress.
 * Delegates to BrailleSpinner for consistent visual style.
 */
export function IndeterminateSpinner({ label }: IndeterminateSpinnerProps) {
  return (
    <Box gap={1}>
      <BrailleSpinner label={label} />
    </Box>
  );
}
