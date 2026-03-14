import React from "react";
import { Box, Text } from "ink";
import { theme } from "../theme.js";

interface CodeBlockProps {
  code: string;
  language?: string;
  title?: string;
}

export function CodeBlock({ code, language, title }: CodeBlockProps) {
  return (
    <Box borderStyle="single" borderColor={theme.textMuted} flexDirection="column" paddingX={1}>
      <Box justifyContent="space-between">
        <Text dimColor>{title ?? " "}</Text>
        <Text dimColor>{language ?? "text"}</Text>
      </Box>
      {code.split("\n").map((line, index) => (
        <Text key={`${index}:${line}`}>{line || " "}</Text>
      ))}
    </Box>
  );
}
