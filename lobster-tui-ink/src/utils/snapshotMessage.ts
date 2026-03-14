/**
 * Pure function to snapshot a completed message as a styled terminal string.
 * Used by Thread to promote finished messages into inert <Static> <Text> nodes,
 * eliminating React hook subscriptions and VDOM diffing for scrollback.
 */

import chalk from "chalk";
import {
  coerceAlertBlockData,
  coerceCodeBlockData,
  coerceRichTableData,
} from "./richBlocks.js";

const PRIMARY = chalk.hex("#e45c47");
const ACCENT = chalk.hex("#e45c47").bold;
const DIM = chalk.gray;
const MUTED = chalk.gray;
const SUCCESS = chalk.hex("#28a745");
const WARNING = chalk.hex("#ffc107");
const ERROR = chalk.hex("#dc3545");
const INFO = chalk.hex("#17a2b8");

interface MessagePart {
  type: string;
  text?: string;
  toolName?: string;
  toolCallId?: string;
  result?: unknown;
  isError?: boolean;
  name?: string;
  data?: unknown;
}

interface SnapshotableMessage {
  role: string;
  content: readonly MessagePart[];
  status?: { type: string };
}

function widthOf(value: string) {
  return Array.from(value).length;
}

function truncate(value: string, width: number) {
  const chars = Array.from(value);
  if (chars.length <= width) {
    return value;
  }
  if (width <= 1) {
    return chars.slice(0, width).join("");
  }
  return `${chars.slice(0, width - 1).join("")}…`;
}

function formatCodeBlock(data: unknown) {
  const block = coerceCodeBlockData(data);
  if (!block) return [] as string[];

  const lines: string[] = [];
  if (block.title || block.language) {
    lines.push(`  ${MUTED(block.title ?? block.language ?? "")}`);
  }
  lines.push(`  ${MUTED("┌" + "─".repeat(46) + "┐")}`);
  for (const line of block.code.split("\n")) {
    lines.push(`  ${MUTED("│")} ${line}`);
  }
  lines.push(`  ${MUTED("└" + "─".repeat(46) + "┘")}`);
  return lines;
}

function formatTable(data: unknown) {
  const table = coerceRichTableData(data);
  if (!table) return [] as string[];

  const rows = table.rows.map((row) =>
    Array.from({ length: table.headers.length }, (_, index) => String(row[index] ?? "")),
  );
  const widths = table.headers.map((header, index) =>
    Math.min(
      24,
      Math.max(widthOf(header), ...rows.map((row) => widthOf(row[index] ?? ""))),
    ),
  );

  const borderTop = `  ${MUTED(`┌${widths.map((width) => "─".repeat(width)).join("┬")}┐`)}`;
  const borderMid = `  ${MUTED(`├${widths.map((width) => "─".repeat(width)).join("┼")}┤`)}`;
  const borderBottom = `  ${MUTED(`└${widths.map((width) => "─".repeat(width)).join("┴")}┘`)}`;
  const renderRow = (row: string[]) =>
    `  │${row
      .map((value, index) => truncate(value, widths[index]!).padEnd(widths[index]!))
      .join("│")}│`;

  return [
    ...(table.title ? [`  ${table.title}`] : []),
    borderTop,
    renderRow(table.headers),
    borderMid,
    ...rows.map(renderRow),
    borderBottom,
  ];
}

function formatAlert(data: unknown) {
  const alert = coerceAlertBlockData(data);
  if (!alert) return [] as string[];

  const color =
    alert.level === "success"
      ? SUCCESS
      : alert.level === "warning"
        ? WARNING
        : alert.level === "error"
          ? ERROR
          : INFO;

  const heading = [alert.title].filter(Boolean).join(" ");
  const lines = alert.message.split("\n");

  return [
    `  ${color("│")} ${color(heading || alert.level.toUpperCase())}`,
    ...lines.map((line) => `  ${color("│")} ${line}`),
  ];
}

export function snapshotMessage(message: SnapshotableMessage): string {
  const lines: string[] = [];

  if (message.role === "user") {
    const text = message.content
      .filter((p) => p.type === "text")
      .map((p) => p.text ?? "")
      .join("");
    lines.push(`${PRIMARY.bold("You:")} ${text}`);
  } else if (message.role === "assistant") {
    lines.push(ACCENT("Lobster:"));
    for (const part of message.content) {
      switch (part.type) {
        case "text":
          if (part.text) {
            for (const line of part.text.split("\n")) {
              lines.push(`  ${line}`);
            }
          }
          break;
        case "tool-call":
          lines.push(
            `  ${DIM("⚙")} ${DIM(part.toolName ?? "tool")} ${DIM("→")} ${DIM(part.result !== undefined ? "done" : "…")}`,
          );
          break;
        case "reasoning":
          lines.push(`  ${DIM("▶ Thinking... (collapsed)")}`);
          break;
        case "data":
          if (part.name === "code") {
            lines.push(...formatCodeBlock(part.data));
          } else if (part.name === "table") {
            lines.push(...formatTable(part.data));
          } else if (part.name === "alert") {
            lines.push(...formatAlert(part.data));
          }
          break;
        case "data-code":
          lines.push(...formatCodeBlock(part.data));
          break;
        case "data-table":
          lines.push(...formatTable(part.data));
          break;
        case "data-alert":
          lines.push(...formatAlert(part.data));
          break;
      }
    }
  }

  return lines.join("\n");
}
