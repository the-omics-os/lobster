import React, { useState, useCallback, useMemo } from "react";
import { Box, Text } from "ink";
import { TextInput } from "@inkjs/ui";
import { useAssistantRuntime } from "@assistant-ui/react-ink";
import { getCommandNames } from "../commands/dispatcher.js";

interface ComposerProps {
  /** Intercept input before sending to runtime. Return true if handled. */
  onIntercept?: (input: string) => boolean;
}

export function Composer({ onIntercept }: ComposerProps) {
  const runtime = useAssistantRuntime();
  const threadRuntime = runtime.thread;
  const [key, setKey] = useState(0);
  const [currentInput, setCurrentInput] = useState("");

  // Build suggestions: when input starts with /, suggest command names
  const suggestions = useMemo(() => {
    if (!currentInput.startsWith("/")) return undefined;
    const partial = currentInput.slice(1).toLowerCase();
    return getCommandNames()
      .filter((name) => name.startsWith(partial))
      .map((name) => `/${name}`);
  }, [currentInput]);

  const handleChange = useCallback((value: string) => {
    setCurrentInput(value);
  }, []);

  const handleSubmit = useCallback(
    (value: string) => {
      const trimmed = value.trim();
      if (!trimmed) return;

      if (onIntercept && onIntercept(trimmed)) {
        setKey((k) => k + 1);
        setCurrentInput("");
        return;
      }

      threadRuntime.append({
        role: "user",
        content: [{ type: "text", text: trimmed }],
      });
      setKey((k) => k + 1);
      setCurrentInput("");
    },
    [onIntercept, threadRuntime],
  );

  return (
    <Box borderStyle="single" borderColor="gray">
      <Text dimColor>{"> "}</Text>
      <TextInput
        key={key}
        placeholder="Type a message or /help..."
        suggestions={suggestions}
        onChange={handleChange}
        onSubmit={handleSubmit}
      />
    </Box>
  );
}
