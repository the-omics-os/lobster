/**
 * HITL QC Dashboard — read-only inline display of QC metrics.
 *
 * Renders a table with columns: sample, reads, genes, mito%, doublet score.
 * Streaming updates as metrics arrive (tool result updates the display).
 */
import React from "react";
import { Box, Text } from "ink";
import { makeAssistantToolUI } from "@assistant-ui/react-ink";

interface QCMetric {
  sample: string;
  total_reads?: number;
  genes_detected?: number;
  mito_pct?: number;
  doublet_score?: number;
  pass?: boolean;
}

interface QCDashboardArgs {
  title?: string;
  metrics: QCMetric[];
}

function formatNumber(n: number | undefined): string {
  if (n === undefined) return "-";
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return String(n);
}

function formatPct(n: number | undefined): string {
  if (n === undefined) return "-";
  return `${(n * 100).toFixed(1)}%`;
}

export const QCDashboardUI = makeAssistantToolUI<QCDashboardArgs, null>({
  toolName: "display_qc_dashboard",
  render: ({ args }) => {
    const metrics = args.metrics ?? [];
    const colWidths = { sample: 20, reads: 10, genes: 10, mito: 8, doublet: 10, status: 6 };

    return (
      <Box flexDirection="column" marginY={1}>
        <Text bold color="cyan">
          {args.title ?? "QC Dashboard"}
        </Text>

        {/* Header row */}
        <Box>
          <Text bold>
            {"Sample".padEnd(colWidths.sample)}
            {"Reads".padEnd(colWidths.reads)}
            {"Genes".padEnd(colWidths.genes)}
            {"Mito%".padEnd(colWidths.mito)}
            {"Doublet".padEnd(colWidths.doublet)}
            {"Status".padEnd(colWidths.status)}
          </Text>
        </Box>
        <Text color="gray">{"─".repeat(64)}</Text>

        {/* Data rows */}
        {metrics.map((m) => {
          const pass = m.pass !== false;
          return (
            <Box key={m.sample}>
              <Text>
                {m.sample.slice(0, colWidths.sample - 1).padEnd(colWidths.sample)}
              </Text>
              <Text>{formatNumber(m.total_reads).padEnd(colWidths.reads)}</Text>
              <Text>{formatNumber(m.genes_detected).padEnd(colWidths.genes)}</Text>
              <Text
                color={
                  m.mito_pct !== undefined && m.mito_pct > 0.2
                    ? "red"
                    : undefined
                }
              >
                {formatPct(m.mito_pct).padEnd(colWidths.mito)}
              </Text>
              <Text>{(m.doublet_score?.toFixed(3) ?? "-").padEnd(colWidths.doublet)}</Text>
              <Text color={pass ? "green" : "red"}>
                {pass ? "PASS" : "FAIL"}
              </Text>
            </Box>
          );
        })}

        {metrics.length === 0 && (
          <Text color="gray">Waiting for QC metrics...</Text>
        )}
      </Box>
    );
  },
});
