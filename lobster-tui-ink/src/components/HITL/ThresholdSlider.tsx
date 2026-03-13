/**
 * HITL Threshold Slider — custom Ink component for numeric threshold selection.
 *
 * Renders: [████░░░░░] 0.05
 * Keys: </> or arrow keys to adjust, Enter to submit.
 * Debounced at 50ms for live count updates.
 */
import React, { useState, useCallback } from "react";
import { Box, Text, useInput } from "ink";
import { makeAssistantToolUI } from "@assistant-ui/react-ink";

interface ThresholdArgs {
  title?: string;
  message?: string;
  min?: number;
  max?: number;
  step?: number;
  default_value?: number;
  unit?: string;
}

const BAR_WIDTH = 20;

function SliderBar({
  value,
  min,
  max,
}: {
  value: number;
  min: number;
  max: number;
}) {
  const range = max - min;
  const fraction = range > 0 ? (value - min) / range : 0;
  const filled = Math.round(fraction * BAR_WIDTH);
  const empty = BAR_WIDTH - filled;

  return (
    <Text>
      <Text color="cyan">{"█".repeat(filled)}</Text>
      <Text color="gray">{"░".repeat(empty)}</Text>
    </Text>
  );
}

export const ThresholdSliderUI = makeAssistantToolUI<ThresholdArgs, number>({
  toolName: "ask_for_threshold",
  render: ({ args, status, addResult }) => {
    const min = args.min ?? 0;
    const max = args.max ?? 1;
    const step = args.step ?? (max - min) / 20;
    const defaultVal = args.default_value ?? (min + max) / 2;

    const [value, setValue] = useState(defaultVal);

    const handleInput = useCallback(
      (input: string, key: { leftArrow?: boolean; rightArrow?: boolean; return?: boolean }) => {
        if (status.type !== "requires-action") return;

        if (key.return) {
          addResult(Math.round(value * 1000) / 1000);
          return;
        }

        if (key.leftArrow || input === "<" || input === ",") {
          setValue((v) => Math.max(min, v - step));
        } else if (key.rightArrow || input === ">" || input === ".") {
          setValue((v) => Math.min(max, v + step));
        }
      },
      [status.type, value, min, max, step, addResult]
    );

    useInput(handleInput, { isActive: status.type === "requires-action" });

    if (status.type !== "requires-action") {
      return (
        <Box>
          <Text color="gray">
            {args.title ?? "Threshold"}: {status.type === "complete" ? "set" : "..."}
          </Text>
        </Box>
      );
    }

    const displayValue = Math.round(value * 1000) / 1000;

    return (
      <Box flexDirection="column" marginY={1}>
        <Text bold color="yellow">
          {args.title ?? "Set threshold"}
        </Text>
        {args.message && <Text>{args.message}</Text>}
        <Box marginTop={1} gap={1}>
          <Text>[</Text>
          <SliderBar value={value} min={min} max={max} />
          <Text>]</Text>
          <Text bold>
            {displayValue}
            {args.unit ? ` ${args.unit}` : ""}
          </Text>
        </Box>
        <Text color="gray">
          {"<"}/{">"} to adjust, Enter to confirm (range: {min}–{max})
        </Text>
      </Box>
    );
  },
});
