/**
 * BrailleSpinner — custom 10-frame braille spinner matching the Go Charm TUI.
 *
 * Frames: braille dots at 80ms interval.
 * Optional rotating tips displayed every 5 seconds.
 */
import React, { useState, useEffect } from "react";
import { Text } from "ink";
import { theme } from "../theme.js";

const FRAMES = ["\u280B", "\u2819", "\u2839", "\u2838", "\u283C", "\u2834", "\u2826", "\u2827", "\u2807", "\u280F"];
const INTERVAL_MS = 80;

const TIPS = [
  "Try /help for available commands",
  "Use @mention to reference datasets",
  "Press Ctrl+C to cancel a running task",
  "Type /status to check session info",
  "Use arrow keys for command history",
];
const TIP_ROTATE_MS = 5_000;

interface BrailleSpinnerProps {
  /** Text label shown next to the spinner. */
  label?: string;
  /** Show rotating tips below the spinner. */
  showTips?: boolean;
  /** Override spinner color (defaults to theme.warning). */
  color?: string;
}

export function BrailleSpinner({
  label,
  showTips = false,
  color = theme.warning,
}: BrailleSpinnerProps) {
  const [frameIdx, setFrameIdx] = useState(0);
  const [tipIdx, setTipIdx] = useState(0);

  useEffect(() => {
    const id = setInterval(() => {
      setFrameIdx((i) => (i + 1) % FRAMES.length);
    }, INTERVAL_MS);
    return () => clearInterval(id);
  }, []);

  useEffect(() => {
    if (!showTips) return;
    const id = setInterval(() => {
      setTipIdx((i) => (i + 1) % TIPS.length);
    }, TIP_ROTATE_MS);
    return () => clearInterval(id);
  }, [showTips]);

  return (
    <Text>
      <Text color={color}>{FRAMES[frameIdx]}</Text>
      {label ? <Text> {label}</Text> : null}
      {showTips ? (
        <Text color={theme.textMuted}>{`  \u{1F4A1} ${TIPS[tipIdx]}`}</Text>
      ) : null}
    </Text>
  );
}
