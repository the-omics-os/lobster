import React from "react";
import { Box } from "ink";
import { MessagePrimitive } from "@assistant-ui/react-ink";
import type { DataMessagePartProps } from "@assistant-ui/core/react";
import { useAuiState } from "@assistant-ui/store";
import { MarkdownText } from "@assistant-ui/react-ink-markdown";
import { ChainOfThought } from "./ChainOfThought.js";
import { CodeBlock } from "./CodeBlock.js";
import { RichTable } from "./RichTable.js";
import { AlertBlock } from "./AlertBlock.js";
import { useTerminalSize } from "../hooks/useTerminalSize.js";
import {
  coerceAlertBlockData,
  coerceCodeBlockData,
  coerceRichTableData,
} from "../utils/richBlocks.js";

function CodeDataPart({ data }: DataMessagePartProps) {
  const block = coerceCodeBlockData(data);
  if (!block) {
    return null;
  }
  return <CodeBlock code={block.code} language={block.language} title={block.title} />;
}

function TableDataPart({ data }: DataMessagePartProps) {
  const table = coerceRichTableData(data);
  if (!table) {
    return null;
  }
  return (
    <RichTable
      headers={table.headers}
      rows={table.rows}
      columns={table.columns}
      title={table.title}
    />
  );
}

function AlertDataPart({ data }: DataMessagePartProps) {
  const alert = coerceAlertBlockData(data);
  if (!alert) {
    return null;
  }
  return (
    <AlertBlock
      level={alert.level}
      title={alert.title}
      message={alert.message}
    />
  );
}

export function AssistantMessage() {
  const partCount = useAuiState((state) => state.message.parts.length);
  const { columns } = useTerminalSize();
  const markdownWidth = Math.max(24, columns - 4);

  if (partCount === 0) {
    return null;
  }

  return (
    <MessagePrimitive.Root>
      <Box marginBottom={1} paddingLeft={2} flexDirection="column">
        <MessagePrimitive.Parts
          components={{
            Text: ({ text }) => (
              <MarkdownText
                text={text}
                width={markdownWidth}
                listIndent={2}
                tableBorder="unicode"
              />
            ),
            ChainOfThought,
            data: {
              by_name: {
                code: CodeDataPart,
                table: TableDataPart,
                alert: AlertDataPart,
              },
            },
          }}
        />
      </Box>
    </MessagePrimitive.Root>
  );
}
