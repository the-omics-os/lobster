import React from "react";
import { Box, Text } from "ink";
import type { CommandOutputPayload } from "../commands/dispatcher.js";
import { CodeBlock } from "./CodeBlock.js";
import { RichTable } from "./RichTable.js";
import { AlertBlock } from "./AlertBlock.js";
import { useTheme } from "../hooks/useTheme.js";

interface CommandOutputProps {
  output: CommandOutputPayload;
}

function renderSection(title: string | undefined, body: string | undefined) {
  const lines = [title, body].filter(Boolean).join("\n").split("\n");
  return lines.map((line, index) => <Text key={index}>{line}</Text>);
}

export function CommandOutput({ output }: CommandOutputProps) {
  const theme = useTheme();

  return (
    <Box flexDirection="column">
      {output.blocks.map((block, index) => {
        const key = `${block.kind}:${index}`;

        if (block.kind === "section" || block.kind === "hint") {
          return (
            <Box key={key} flexDirection="column">
              {renderSection(block.title, block.body ?? block.message)}
            </Box>
          );
        }

        if (block.kind === "alert") {
          return (
            <AlertBlock
              key={key}
              level={
                block.level === "success" ||
                block.level === "warning" ||
                block.level === "error" ||
                block.level === "info"
                  ? block.level
                  : "info"
              }
              title={block.title}
              message={block.message ?? block.body ?? ""}
            />
          );
        }

        if (block.kind === "code") {
          return (
            <CodeBlock
              key={key}
              title={block.title}
              language={block.language}
              code={block.code ?? ""}
            />
          );
        }

        if (block.kind === "list") {
          return (
            <Box key={key} flexDirection="column">
              {block.title ? <Text bold>{block.title}</Text> : null}
              {(block.items ?? []).map((item, itemIndex) => (
                <Text key={itemIndex}>
                  {block.ordered ? `${itemIndex + 1}. ` : "- "}
                  {item}
                </Text>
              ))}
            </Box>
          );
        }

        if (block.kind === "kv") {
          return (
            <RichTable
              key={key}
              title={block.title}
              headers={[
                block.key_label ?? "Field",
                block.value_label ?? "Value",
              ]}
              rows={block.rows ?? []}
            />
          );
        }

        if (block.kind === "table") {
          return (
            <RichTable
              key={key}
              title={block.title}
              headers={(block.columns ?? []).length > 0
                ? (block.columns ?? []).map((column) => String(column.name ?? ""))
                : (block.rows?.[0]?.map((_, rowIndex) => `Column ${rowIndex + 1}`) ?? [])}
              rows={block.rows ?? []}
              columns={(block.columns ?? []).map((column) => ({
                name: String(column.name ?? ""),
                width:
                  typeof column.width === "number" ? column.width : undefined,
                maxWidth:
                  typeof column.max_width === "number"
                    ? column.max_width
                    : typeof column.maxWidth === "number"
                      ? column.maxWidth
                      : undefined,
                justify:
                  column.justify === "left" ||
                  column.justify === "right" ||
                  column.justify === "center"
                    ? column.justify
                    : undefined,
              }))}
            />
          );
        }

        return (
          <Text key={key} color={theme.textMuted}>
            {JSON.stringify(block)}
          </Text>
        );
      })}
    </Box>
  );
}
