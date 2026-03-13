import React, { useState, useCallback } from "react";
import { Box, Text } from "ink";
import { TextInput } from "@inkjs/ui";
import { useAssistantRuntime } from "@assistant-ui/react-ink";

interface ComposerProps {
  /** Intercept input before sending to runtime. Return true if handled. */
  onIntercept?: (input: string) => boolean;
}

export function Composer({ onIntercept }: ComposerProps) {
  const runtime = useAssistantRuntime();
  const threadRuntime = runtime.thread;
  const [key, setKey] = useState(0);

  const handleSubmit = useCallback(
    (value: string) => {
      const trimmed = value.trim();
      if (!trimmed) return;

      // Check if a slash command handler wants to intercept
      if (onIntercept && onIntercept(trimmed)) {
        // Force re-render to clear the input
        setKey((k) => k + 1);
        return;
      }

      // Send to runtime as a user message
      threadRuntime.append({
        role: "user",
        content: [{ type: "text", text: trimmed }],
      });
      setKey((k) => k + 1);
    },
    [onIntercept, threadRuntime],
  );

  return (
    <Box borderStyle="single" borderColor="gray">
      <Text dimColor>{"> "}</Text>
      <TextInput
        key={key}
        placeholder="Type a message or /help..."
        onSubmit={handleSubmit}
      />
    </Box>
  );
}
