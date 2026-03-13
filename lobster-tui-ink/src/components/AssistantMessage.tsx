import React from "react";
import { Box, Text } from "ink";
import { MessagePrimitive } from "@assistant-ui/react-ink";
import { MarkdownText } from "@assistant-ui/react-ink-markdown";
import { ChainOfThought } from "./ChainOfThought.js";

function TextPart({ text }: { text: string }) {
  return <MarkdownText text={text} />;
}

export function AssistantMessage() {
  return (
    <MessagePrimitive.Root>
      <Box flexDirection="column" marginY={0}>
        <Text bold color="magenta">
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
