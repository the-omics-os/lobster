import React, { useState, useCallback, useMemo } from "react";
import { Box, Text, useInput } from "ink";
import { TextInput } from "@inkjs/ui";
import { useAssistantRuntime } from "@assistant-ui/react-ink";
import { getCommandNames } from "../commands/dispatcher.js";
import { useHistory } from "../hooks/useHistory.js";
import { filterResources, type Resource } from "../api/resources.js";

interface ComposerProps {
  /** Intercept input before sending to runtime. Return true if handled. */
  onIntercept?: (input: string) => boolean;
  /** Resources catalog for @mention autocomplete. */
  resources?: Resource[];
}

export function Composer({ onIntercept, resources }: ComposerProps) {
  const runtime = useAssistantRuntime();
  const threadRuntime = runtime.thread;
  const [key, setKey] = useState(0);
  const [currentInput, setCurrentInput] = useState("");
  const [defaultValue, setDefaultValue] = useState("");
  const history = useHistory();

  // Arrow-up/down for history navigation
  useInput((_input, k) => {
    if (k.upArrow) {
      const entry = history.up(currentInput);
      if (entry !== undefined) {
        setDefaultValue(entry);
        setCurrentInput(entry);
        setKey((prev) => prev + 1);
      }
    } else if (k.downArrow) {
      const entry = history.down();
      if (entry !== undefined) {
        setDefaultValue(entry);
        setCurrentInput(entry);
        setKey((prev) => prev + 1);
      }
    }
  });

  // Build suggestions: / commands or @resource mentions
  const suggestions = useMemo(() => {
    if (currentInput.startsWith("/")) {
      const partial = currentInput.slice(1).toLowerCase();
      return getCommandNames()
        .filter((name) => name.startsWith(partial))
        .map((name) => `/${name}`);
    }

    // Check for @mention: find the last @ in the input
    const atIdx = currentInput.lastIndexOf("@");
    if (atIdx >= 0 && resources && resources.length > 0) {
      const prefix = currentInput.slice(atIdx + 1);
      if (prefix.length > 0 && !prefix.includes(" ")) {
        const matches = filterResources(resources, prefix);
        return matches.slice(0, 10).map((r) => {
          const before = currentInput.slice(0, atIdx);
          return `${before}@${r.id}`;
        });
      }
    }

    return undefined;
  }, [currentInput, resources]);

  const handleChange = useCallback((value: string) => {
    setCurrentInput(value);
  }, []);

  const handleSubmit = useCallback(
    (value: string) => {
      const trimmed = value.trim();
      if (!trimmed) return;

      history.push(trimmed);

      if (onIntercept && onIntercept(trimmed)) {
        setKey((k) => k + 1);
        setCurrentInput("");
        setDefaultValue("");
        return;
      }

      threadRuntime.append({
        role: "user",
        content: [{ type: "text", text: trimmed }],
      });
      setKey((k) => k + 1);
      setCurrentInput("");
      setDefaultValue("");
    },
    [onIntercept, threadRuntime, history],
  );

  return (
    <Box borderStyle="single" borderColor="gray">
      <Text dimColor>{"> "}</Text>
      <TextInput
        key={key}
        defaultValue={defaultValue}
        placeholder="Type a message or /help..."
        suggestions={suggestions}
        onChange={handleChange}
        onSubmit={handleSubmit}
      />
    </Box>
  );
}
