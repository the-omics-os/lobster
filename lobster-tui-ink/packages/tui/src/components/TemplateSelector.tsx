/**
 * TemplateSelector — shows prompt templates on new session.
 * "Choose a template or type freely:"
 * User types a number → pre-fills composer with template prompt.
 * User types freely → normal input.
 */
import React, { useCallback } from "react";
import { Box, Text } from "ink";
import { TextInput } from "@inkjs/ui";
import type { PromptTemplate } from "../api/templates.js";
import { useTheme } from "../hooks/useTheme.js";

const CATEGORY_LABELS: Record<string, string> = {
  omics: "Omics",
  visualization: "Visualization",
  data: "Data",
  literature: "Literature",
};

interface TemplateSelectorProps {
  templates: PromptTemplate[];
  onSelect: (template: string) => void;
  onDismiss: () => void;
}

export function TemplateSelector({
  templates,
  onSelect,
}: TemplateSelectorProps) {
  const theme = useTheme();

  const handleSubmit = useCallback(
    (value: string) => {
      const trimmed = value.trim();
      if (!trimmed) return;

      const num = parseInt(trimmed, 10);
      const selected = !isNaN(num) ? templates[num - 1] : undefined;
      if (selected) {
        onSelect(selected.template);
      } else {
        onSelect(trimmed);
      }
    },
    [templates, onSelect],
  );

  // Group by category
  const grouped = new Map<string, { index: number; tpl: PromptTemplate }[]>();
  for (const [i, tpl] of templates.entries()) {
    const cat = tpl.category;
    const items = grouped.get(cat) ?? [];
    items.push({ index: i + 1, tpl });
    grouped.set(cat, items);
  }

  return (
    <Box flexDirection="column" paddingX={1} marginBottom={1}>
      <Text bold color={theme.info}>
        Choose a template or type freely:
      </Text>
      <Text> </Text>
      {Array.from(grouped.entries()).map(([category, items]) => (
        <Box key={category} flexDirection="column" marginBottom={0}>
          <Text bold dimColor>
            {CATEGORY_LABELS[category] ?? category}
          </Text>
          {items.map(({ index, tpl }) => (
            <Text key={tpl.id}>
              <Text color={theme.warning}>{`  ${String(index).padStart(2)} `}</Text>
              <Text>{tpl.title}</Text>
            </Text>
          ))}
        </Box>
      ))}
      <Text> </Text>
      <Box>
        <Text dimColor>{"> "}</Text>
        <TextInput
          placeholder="Type a number or your own prompt..."
          onSubmit={handleSubmit}
        />
      </Box>
    </Box>
  );
}
