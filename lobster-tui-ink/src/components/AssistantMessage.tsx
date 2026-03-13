import React from "react";
import { Box, Text } from "ink";
import { MessageRoot, MessageParts } from "@assistant-ui/react-ink";
import { MarkdownText } from "@assistant-ui/react-ink-markdown";
import { ChainOfThought } from "./ChainOfThought.js";
import { theme } from "../theme.js";

function TextPart({ text }: { text: string }) {
  return <MarkdownText text={text} />;
}

export function AssistantMessage() {
  return (
    <MessageRoot>
      <Box flexDirection="column" marginY={0}>
        <Text bold color={theme.accent1}>
          Lobster:
        </Text>
        <Box marginLeft={2} flexDirection="column">
          <MessageParts
            components={{
              Text: TextPart,
              ChainOfThought,
            }}
          />
        </Box>
      </Box>
    </MessageRoot>
  );
}
