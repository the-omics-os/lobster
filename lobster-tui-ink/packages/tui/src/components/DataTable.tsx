import React from "react";
import { Box, Text, useStdout } from "ink";

const MAX_ROWS = 10;
const MAX_COLS = 5;

interface DataTableProps {
  data: Record<string, unknown>[];
  title?: string;
}

/**
 * Terminal-width-aware table rendering.
 * Truncates to MAX_ROWS x MAX_COLS with summary footer.
 */
export function DataTable({ data, title }: DataTableProps) {
  const { stdout } = useStdout();
  const columns = stdout?.columns ?? 80;

  if (!data.length) {
    return <Text dimColor>Empty table</Text>;
  }

  const allKeys = Object.keys(data[0]!);
  const displayKeys = allKeys.slice(0, MAX_COLS);
  const displayRows = data.slice(0, MAX_ROWS);
  const colWidth = Math.max(
    8,
    Math.floor((columns - displayKeys.length - 1) / displayKeys.length)
  );

  const truncCell = (val: unknown): string => {
    const s = val === null || val === undefined ? "" : String(val);
    return s.length > colWidth ? s.slice(0, colWidth - 1) + "…" : s;
  };

  const padCell = (s: string): string => s.padEnd(colWidth).slice(0, colWidth);

  return (
    <Box flexDirection="column">
      {title && <Text bold>{title}</Text>}
      <Text bold dimColor>
        {displayKeys.map((k) => padCell(truncCell(k))).join(" ")}
      </Text>
      <Text dimColor>{"─".repeat(Math.min(columns - 2, colWidth * displayKeys.length + displayKeys.length - 1))}</Text>
      {displayRows.map((row, i) => (
        <Text key={i}>
          {displayKeys
            .map((k) => padCell(truncCell(row[k])))
            .join(" ")}
        </Text>
      ))}
      {(data.length > MAX_ROWS || allKeys.length > MAX_COLS) && (
        <Text dimColor>
          Showing {displayRows.length} of {data.length} rows
          {allKeys.length > MAX_COLS
            ? `, ${displayKeys.length} of ${allKeys.length} columns`
            : ""}
        </Text>
      )}
    </Box>
  );
}
