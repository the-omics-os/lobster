import React from "react";
import { Box, Text } from "ink";
import { ComposerPrimitive } from "@assistant-ui/react-ink";

export function Composer() {
  return (
    <ComposerPrimitive.Root>
      <Box borderStyle="single" borderColor="gray">
        <Text dimColor>{"> "}</Text>
        <ComposerPrimitive.Input
          submitOnEnter
          autoFocus
          placeholder="Type a message..."
        />
      </Box>
    </ComposerPrimitive.Root>
  );
}
