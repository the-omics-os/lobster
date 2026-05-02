/**
 * BrailleProgressBar — braille-block progress bar matching Go Charm TUI style.
 *
 * Uses braille characters: filled = \u28FF, empty = \u2880
 * Shows: label [braille bar] percentage
 */
import React from "react";
import { Box, Text } from "ink";
import { useTheme } from "../hooks/useTheme.js";

const BAR_WIDTH = 30;
const FILLED_CHAR = "\u28FF";  // braille full block
const EMPTY_CHAR = "\u2880";   // braille lower dots

interface BrailleProgressBarProps {
  label: string;
  /** Progress value between 0 and 1. */
  value: number;
  /** Override filled-bar color (defaults to theme.accent3). */
  color?: string;
}

export function BrailleProgressBar({
  label,
  value,
  color,
}: BrailleProgressBarProps) {
  const theme = useTheme();
  const barColor = color ?? theme.accent3;
  const clamped = Math.max(0, Math.min(1, value));
  const filled = Math.round(clamped * BAR_WIDTH);
  const empty = BAR_WIDTH - filled;
  const pct = Math.round(clamped * 100);

  return (
    <Box gap={1}>
      <Text color={theme.textMuted}>{label}</Text>
      <Text>
        <Text color={barColor}>{FILLED_CHAR.repeat(filled)}</Text>
        <Text color={theme.textDim}>{EMPTY_CHAR.repeat(empty)}</Text>
      </Text>
      <Text>{pct}%</Text>
    </Box>
  );
}
