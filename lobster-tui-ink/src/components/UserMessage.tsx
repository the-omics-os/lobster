import React from "react";
import { Box, Text } from "ink";
import { MessagePrimitive } from "@assistant-ui/react-ink";

export function UserMessage() {
  return (
    <MessagePrimitive.Root>
      <Box marginY={0}>
        <Text bold color="cyan">
          You:{" "}
        </Text>
        <MessagePrimitive.Content />
      </Box>
    </MessagePrimitive.Root>
  );
}
