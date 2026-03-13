import React from "react";
import { Box, Text } from "ink";
import { MessagePrimitive } from "@assistant-ui/react-ink";
import { MarkdownText } from "@assistant-ui/react-ink-markdown";
import { ChainOfThought } from "./ChainOfThought.js";
import { theme } from "../theme.js";

function TextPart({ text }: { text: string }) {
  return <MarkdownText text={text} />;
}

export function AssistantMessage() {
  return (
    <MessagePrimitive.Root>
      <Box flexDirection="column" marginY={0}>
        <Text bold color={theme.accent1}>
          Lobster:
        </Text>
        <Box marginLeft={2} flexDirection="column">
          <MessagePrimitive.Parts
            components={{
              Text: TextPart,
              ChainOfThought,
            }}
          />
        </Box>
      </Box>
    </MessagePrimitive.Root>
  );
}
