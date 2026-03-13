/**
 * DataTablePreviewRenderer — extracts tabular data from tool results
 * and renders an inline preview using the DataTable component.
 */
import React from "react";
import { Box, Text } from "ink";
import type { ToolCallMessagePartProps } from "@assistant-ui/react-ink";
import { DataTable } from "../DataTable.js";

function extractTableData(
  result: unknown,
): Record<string, unknown>[] | undefined {
  if (!result) return undefined;

  try {
    const parsed = typeof result === "string" ? JSON.parse(result) : result;

    // Direct array of objects
    if (Array.isArray(parsed) && parsed.length > 0 && typeof parsed[0] === "object") {
      return parsed as Record<string, unknown>[];
    }

    // Object with a data/rows/results/items array
    if (parsed && typeof parsed === "object") {
      const obj = parsed as Record<string, unknown>;
      for (const key of ["data", "rows", "results", "items", "records"]) {
        const arr = obj[key];
        if (Array.isArray(arr) && arr.length > 0 && typeof arr[0] === "object") {
          return arr as Record<string, unknown>[];
        }
      }
    }
  } catch {
    // Not JSON or not tabular
  }

  return undefined;
}

export function DataTablePreviewRenderer(props: ToolCallMessagePartProps) {
  const { toolName, result } = props;
  const status = result !== undefined ? "complete" : "running";

  if (status !== "complete") {
    return (
      <Box gap={1}>
        <Text color="yellow">⏳</Text>
        <Text dimColor>{toolName}...</Text>
      </Box>
    );
  }

  const tableData = extractTableData(result);

  if (!tableData) {
    return (
      <Box gap={1}>
        <Text color="green">✓</Text>
        <Text dimColor>{toolName}</Text>
      </Box>
    );
  }

  return (
    <Box flexDirection="column" marginY={1}>
      <DataTable data={tableData} title={toolName} />
      <Text dimColor>Press 'o' to open full table in browser</Text>
    </Box>
  );
}
