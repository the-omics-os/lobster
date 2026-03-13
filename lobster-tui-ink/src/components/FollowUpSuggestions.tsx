import React from "react";
import { Box, Text } from "ink";

interface FollowUpSuggestionsProps {
  suggestions: string[];
}

/**
 * Numbered follow-up suggestion list below an assistant message.
 * Rendered from the durable message envelope's follow_ups field.
 */
export function FollowUpSuggestions({ suggestions }: FollowUpSuggestionsProps) {
  if (!suggestions.length) return null;

  return (
    <Box flexDirection="column" marginTop={1}>
      {suggestions.map((text, i) => (
        <Text key={i} dimColor>
          [{i + 1}] {text}
        </Text>
      ))}
    </Box>
  );
}
