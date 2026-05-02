import React, { useEffect, useState } from "react";
import { Text } from "ink";
import { useTheme } from "../hooks/useTheme.js";

const TIPS = [
  "Lobster AI has 22 specialist agents across 10 domains",
  "Use /data to inspect loaded modalities",
  "Sessions auto-save — use --session-id latest to resume",
  "Use Tab to autocomplete slash commands",
  "Alt+Enter inserts a newline in the composer",
  "Use @mentions to reference resources in your messages",
  "Try /help to see all available commands",
  "Press Ctrl+C twice to cancel a running operation",
];

const ROTATE_MS = 5000;

export function LoadingTips() {
  const theme = useTheme();
  const [tipIndex, setTipIndex] = useState(0);

  useEffect(() => {
    const timer = setInterval(() => {
      setTipIndex((current) => (current + 1) % TIPS.length);
    }, ROTATE_MS);
    return () => clearInterval(timer);
  }, []);

  return <Text color={theme.textMuted}>{`Tip: ${TIPS[tipIndex]}`}</Text>;
}
