import React from "react";
import { Box, Text } from "ink";
import { MessagePrimitive } from "@assistant-ui/react-ink";
import { theme } from "../theme.js";

export function UserMessage() {
  return (
    <MessagePrimitive.Root>
      <Box marginY={0} borderStyle="single" borderLeft borderTop={false} borderRight={false} borderBottom={false} borderColor={theme.primary} paddingLeft={1}>
        <Text bold color={theme.primary}>
          You:{" "}
        </Text>
        <MessagePrimitive.Content />
      </Box>
    </MessagePrimitive.Root>
  );
}
