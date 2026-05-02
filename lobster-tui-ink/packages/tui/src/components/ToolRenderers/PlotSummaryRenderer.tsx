/**
 * PlotSummaryRenderer — extract stats from Plotly JSON and render as text.
 * Includes canonical URL per spec §7.1.
 */
import React from "react";
import { Box, Text } from "ink";
import type { ToolCallMessagePartProps } from "@assistant-ui/react-ink";
import { theme } from "../../theme.js";

interface PlotlyTrace {
  type?: string;
  x?: unknown[];
  y?: unknown[];
  name?: string;
  mode?: string;
}

interface PlotlyLayout {
  title?: string | { text?: string };
  xaxis?: { title?: string | { text?: string } };
  yaxis?: { title?: string | { text?: string } };
}

interface PlotlyFigure {
  data?: PlotlyTrace[];
  layout?: PlotlyLayout;
}

function extractTitle(layout: PlotlyLayout): string | undefined {
  if (typeof layout.title === "string") return layout.title;
  return layout.title?.text;
}

function extractAxisLabel(
  axis?: { title?: string | { text?: string } },
): string | undefined {
  if (!axis) return undefined;
  if (typeof axis.title === "string") return axis.title;
  return axis.title?.text;
}

function summarizePlot(figure: PlotlyFigure): string {
  const parts: string[] = [];
  const layout = figure.layout ?? {};
  const data = figure.data ?? [];

  const title = extractTitle(layout);
  if (title) parts.push(title);

  const traceCount = data.length;
  const totalPoints = data.reduce((sum, t) => sum + (t.y?.length ?? t.x?.length ?? 0), 0);

  if (traceCount > 0) {
    const types = [...new Set(data.map((t) => t.type ?? "scatter"))];
    parts.push(`${types.join("/")} (${traceCount} trace${traceCount > 1 ? "s" : ""}, ${totalPoints} points)`);
  }

  const xLabel = extractAxisLabel(layout.xaxis);
  const yLabel = extractAxisLabel(layout.yaxis);
  if (xLabel || yLabel) {
    parts.push(`Axes: ${xLabel ?? "?"} vs ${yLabel ?? "?"}`);
  }

  // Top trace names
  const names = data
    .map((t) => t.name)
    .filter(Boolean)
    .slice(0, 5);
  if (names.length > 0) {
    parts.push(`Traces: ${names.join(", ")}`);
  }

  return parts.join(". ") || "Plot";
}

export function PlotSummaryRenderer(props: ToolCallMessagePartProps) {
  const { args, result, toolName } = props;

  let summary = "Plot generated";
  let plotId: string | undefined;

  // Try to extract Plotly figure from result
  try {
    const res = typeof result === "string" ? JSON.parse(result) : result;
    if (res && typeof res === "object") {
      const fig = res as PlotlyFigure & { plot_id?: string; file_path?: string };
      if (fig.data || fig.layout) {
        summary = summarizePlot(fig);
      }
      plotId = fig.plot_id;
    }
  } catch {
    // Fall through to args-based summary
  }

  // Also try args for pre-generation info
  if (summary === "Plot generated" && args && typeof args === "object") {
    const a = args as Record<string, unknown>;
    const plotType = a.plot_type ?? a.type ?? toolName;
    summary = `${plotType}`;
  }

  const status = result !== undefined ? "complete" : "running";

  return (
    <Box flexDirection="column" paddingX={1}>
      <Box gap={1}>
        <Text color={status === "complete" ? theme.success : theme.warning}>
          {status === "complete" ? "\u2713" : "\u23F3"}
        </Text>
        <Text>{summary}</Text>
      </Box>
      {status === "complete" && (
        <Text dimColor>
          Press 'o' to open in browser
          {plotId ? ` | View at app.omics-os.com/sessions/...#plot=${plotId}` : ""}
        </Text>
      )}
    </Box>
  );
}
