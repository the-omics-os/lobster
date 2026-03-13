/**
 * HITL Cell Type Selector — cluster list with marker genes, user assigns labels.
 *
 * Renders cluster rows with: cluster_id, size, top markers.
 * User types label per cluster, Enter to advance, Enter on last row to submit.
 */
import React, { useState, useCallback } from "react";
import { Box, Text } from "ink";
import { TextInput } from "@inkjs/ui";
import { makeAssistantToolUI } from "@assistant-ui/react-ink";
import { theme } from "../../theme.js";

interface Cluster {
  cluster_id: string | number;
  size: number;
  markers?: string[];
}

interface CellTypeSelectorArgs {
  title?: string;
  message?: string;
  clusters: Cluster[];
}

export const CellTypeSelectorUI = makeAssistantToolUI<
  CellTypeSelectorArgs,
  Record<string, string>
>({
  toolName: "ask_for_cell_types",
  render: ({ args, status, addResult }) => {
    const clusters = args.clusters ?? [];
    const [currentIdx, setCurrentIdx] = useState(0);
    const [labels, setLabels] = useState<Record<string, string>>({});

    const handleSubmit = useCallback(
      (value: string) => {
        const cluster = clusters[currentIdx];
        if (!cluster) return;

        const key = String(cluster.cluster_id);
        const newLabels = { ...labels, [key]: value || `Cluster ${key}` };
        setLabels(newLabels);

        if (currentIdx + 1 < clusters.length) {
          setCurrentIdx(currentIdx + 1);
        } else {
          addResult(newLabels);
        }
      },
      [currentIdx, clusters, labels, addResult]
    );

    if (status.type !== "requires-action") {
      return (
        <Box>
          <Text color={theme.textMuted}>
            Cell type annotation: {status.type === "complete" ? "done" : "..."}
          </Text>
        </Box>
      );
    }

    return (
      <Box flexDirection="column" marginY={1}>
        <Text bold color={theme.warning}>
          {args.title ?? "Assign cell type labels"}
        </Text>
        {args.message && <Text>{args.message}</Text>}
        <Text color={theme.textMuted}>
          ({currentIdx + 1}/{clusters.length}) Type label, Enter to advance
        </Text>

        <Box flexDirection="column" marginTop={1}>
          {clusters.map((cluster, i) => {
            const key = String(cluster.cluster_id);
            const isActive = i === currentIdx;
            const label = labels[key];
            const markers = cluster.markers?.slice(0, 5).join(", ") ?? "";

            return (
              <Box key={key} gap={1}>
                <Text color={isActive ? theme.info : label ? theme.success : theme.textMuted}>
                  {isActive ? ">" : " "}
                </Text>
                <Text bold={isActive}>
                  C{key}
                </Text>
                <Text color={theme.textMuted}>({cluster.size})</Text>
                {markers && <Text color={theme.textMuted}>[{markers}]</Text>}
                {label && <Text color={theme.success}>{label}</Text>}
                {isActive && !label && (
                  <TextInput
                    placeholder="cell type..."
                    onSubmit={handleSubmit}
                  />
                )}
              </Box>
            );
          })}
        </Box>
      </Box>
    );
  },
});
