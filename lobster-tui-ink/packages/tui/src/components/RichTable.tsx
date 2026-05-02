import React from "react";
import { Box, Text, useStdout } from "ink";
import { theme } from "../theme.js";
import type { RichTableColumn } from "../utils/richBlocks.js";

interface RichTableProps {
  headers: string[];
  rows: string[][];
  columns?: RichTableColumn[];
  title?: string;
}

function truncateCell(value: string, width: number) {
  if (value.length <= width) return value.padEnd(width);
  if (width <= 1) return value.slice(0, width);
  return `${value.slice(0, width - 1)}…`;
}

export function RichTable({ headers, rows, columns: columnDefs, title }: RichTableProps) {
  const { stdout } = useStdout();
  const columns = stdout?.columns ?? 80;

  if (!headers.length) return null;

  const maxContentWidths = headers.map((header, columnIndex) => {
    const definedWidth = columnDefs?.[columnIndex]?.width;
    const maxWidth = columnDefs?.[columnIndex]?.maxWidth;
    const measured = Math.max(
      header.length,
      ...rows.map((row) => (row[columnIndex] ?? "").length),
    );
    if (typeof definedWidth === "number") return definedWidth;
    if (typeof maxWidth === "number") return Math.min(maxWidth, measured);
    return measured;
  });

  const availableWidth = Math.max(columns - 4 - headers.length, headers.length * 6);
  const naturalWidth = maxContentWidths.reduce((sum, width) => sum + width, 0);
  const scale = naturalWidth > availableWidth ? availableWidth / naturalWidth : 1;
  const widths = maxContentWidths.map((width) => Math.max(4, Math.floor(width * scale)));

  const horizontal = widths.map((width) => "─".repeat(width + 2)).join("┬");
  const divider = widths.map((width) => "─".repeat(width + 2)).join("┼");
  const footer = widths.map((width) => "─".repeat(width + 2)).join("┴");

  const renderRow = (cells: string[]) =>
    `│ ${cells.map((cell, index) => truncateCell(cell ?? "", widths[index] ?? 4)).join(" │ ")} │`;

  return (
    <Box flexDirection="column">
      {title ? <Text bold>{title}</Text> : null}
      <Text dimColor>{`┌${horizontal}┐`}</Text>
      <Text bold>{renderRow(headers)}</Text>
      <Text dimColor>{`├${divider}┤`}</Text>
      {rows.map((row, index) => (
        <Text key={`${index}:${row.join("\u0000")}`} dimColor={index % 2 === 1}>
          {renderRow(row)}
        </Text>
      ))}
      <Text dimColor color={theme.textMuted}>{`└${footer}┘`}</Text>
    </Box>
  );
}
