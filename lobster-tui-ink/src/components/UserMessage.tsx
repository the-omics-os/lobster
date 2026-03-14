import React from "react";
import { Box, Text } from "ink";
import { MessagePrimitive } from "@assistant-ui/react-ink";
import { useTheme } from "../hooks/useTheme.js";

export function UserMessage() {
  const theme = useTheme();

  return (
    <MessagePrimitive.Root>
      <Box
        flexDirection="column"
        marginBottom={1}
        borderStyle="single"
        borderLeft
        borderTop={false}
        borderRight={false}
        borderBottom={false}
        borderColor={theme.primary}
        paddingLeft={1}
      >
        <Text bold color={theme.primary}>
          You
        </Text>
        <MessagePrimitive.Content />
      </Box>
    </MessagePrimitive.Root>
  );
}
