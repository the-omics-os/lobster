import React from "react";
import { Box, Text } from "ink";
import { MessagePrimitive } from "@assistant-ui/react-ink";
import { MarkdownText } from "@assistant-ui/react-ink-markdown";
import { ToolCallRenderer } from "./ToolRenderers/ToolCallRenderer.js";

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
            renderToolCall={({ part, index }) => (
              <ToolCallRenderer part={part} index={index} />
            )}
          />
        </Box>
      </Box>
    </MessagePrimitive.Root>
  );
}
