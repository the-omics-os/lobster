import React from "react";
import { Box, Text } from "ink";
import { ChainOfThoughtPrimitive } from "@assistant-ui/react-ink";
import { useAuiState } from "@assistant-ui/store";
import { ToolRouter } from "./ToolRenderers/ToolRouter.js";
import { useTheme } from "../hooks/useTheme.js";

function ReasoningPart({ text }: { text: string }) {
  return (
    <Box marginLeft={2}>
      <Text dimColor>{text}</Text>
    </Box>
  );
}

/**
 * ChainOfThought groups reasoning + tool calls into a collapsible block.
 * Expanded while streaming, collapsed after completion.
 */
export function ChainOfThought() {
  const collapsed = useAuiState((s) => s.chainOfThought.collapsed);
  const theme = useTheme();

  return (
    <ChainOfThoughtPrimitive.Root flexDirection="column" marginY={0}>
      <ChainOfThoughtPrimitive.AccordionTrigger>
        <Box gap={1}>
          <Text color={theme.warning}>{collapsed ? "\u25B6" : "\u25BC"}</Text>
          <Text dimColor>Work details</Text>
        </Box>
      </ChainOfThoughtPrimitive.AccordionTrigger>
      <ChainOfThoughtPrimitive.Parts
        components={{
          Reasoning: ReasoningPart,
          tools: {
            Fallback: ToolRouter,
          },
        }}
      />
    </ChainOfThoughtPrimitive.Root>
  );
}
