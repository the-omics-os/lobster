import React from "react";
import { Box, Text } from "ink";
import { MessagePrimitive } from "@assistant-ui/react-ink";
import { MarkdownText } from "@assistant-ui/react-ink-markdown";

export function AssistantMessage() {
  return (
    <MessagePrimitive.Root>
      <Box flexDirection="column" marginY={0}>
        <Text bold color="magenta">
          Lobster:
        </Text>
        <Box marginLeft={2}>
          <MessagePrimitive.Content
            renderText={({ part }) => <MarkdownText text={part.text} />}
          />
        </Box>
      </Box>
    </MessagePrimitive.Root>
  );
}
