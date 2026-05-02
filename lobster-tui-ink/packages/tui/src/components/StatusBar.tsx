import React from "react";
import { Box, Text } from "ink";
import { useAuiState } from "@assistant-ui/store";
import { useStore } from "zustand";
import { BrailleSpinner } from "./BrailleSpinner.js";
import { useTheme } from "../hooks/useTheme.js";
import { useTerminalSize } from "../hooks/useTerminalSize.js";
import type { RuntimeInfo } from "../api/bootstrap.js";
import type { AppStateStore } from "../utils/appStateStore.js";
import type { FooterStateStore } from "../utils/footerStateStore.js";
import type { DataStatusSummary } from "../utils/stateHandlers.js";

interface StatusBarProps {
  appStateStore: AppStateStore;
  footerStateStore: FooterStateStore;
  sessionId?: string;
  runtimeInfo?: RuntimeInfo;
  isStreaming?: boolean;
  isCloud?: boolean;
}

type FooterTone = "accent" | "muted" | "dim" | "success" | "warning";

export interface FooterPart {
  text: string;
  tone: FooterTone;
}

interface BuildFooterPartsInput {
  running: boolean;
  activeAgent: string | null;
  tokenUsage:
    | {
        promptTokens?: number;
        completionTokens?: number;
        duration?: number;
      }
    | null;
  dataStatus: DataStatusSummary | null;
  sessionId?: string;
  runtimeInfo?: RuntimeInfo;
  isCloud?: boolean;
  completionVisible: boolean;
  inputBlocked: boolean;
  multiline: boolean;
}

const STATUS_ICON_WIDTH = 2;
const FOOTER_SEPARATOR = " · ";
const FOOTER_SEPARATOR_WIDTH = FOOTER_SEPARATOR.length;

function prettifyName(value: string) {
  return value.replace(/[_-]/g, " ");
}

function truncateMiddle(value: string, max: number) {
  if (value.length <= max) {
    return value;
  }

  const visible = Math.max(1, max - 3);
  const head = Math.ceil(visible / 2);
  const tail = Math.floor(visible / 2);
  return `${value.slice(0, head)}...${value.slice(-tail)}`;
}

function formatTokenDisplay(
  tokenUsage: BuildFooterPartsInput["tokenUsage"],
) {
  if (!tokenUsage) return null;

  const total =
    (tokenUsage.promptTokens ?? 0) + (tokenUsage.completionTokens ?? 0);
  if (total <= 0) return null;
  return `${total.toLocaleString()} tok`;
}

function formatDurationDisplay(
  tokenUsage: BuildFooterPartsInput["tokenUsage"],
) {
  if (!tokenUsage || typeof tokenUsage.duration !== "number") {
    return null;
  }

  if (!Number.isFinite(tokenUsage.duration) || tokenUsage.duration <= 0) {
    return null;
  }

  return `${tokenUsage.duration.toFixed(1)}s`;
}

function buildRuntimeLabel(runtimeInfo?: RuntimeInfo, isCloud?: boolean) {
  const provider = runtimeInfo?.provider?.trim();
  const model = runtimeInfo?.model?.trim();
  if (provider && model) {
    return `${provider} / ${truncateMiddle(model, 28)}`;
  }
  if (model) {
    return truncateMiddle(model, 32);
  }
  if (provider) {
    return provider;
  }
  return isCloud ? "cloud" : "local";
}

function buildStatusPart(
  running: boolean,
  activeAgent: string | null,
): FooterPart {
  if (!running) {
    return { text: "ready", tone: "success" };
  }

  if (!activeAgent || activeAgent === "supervisor") {
    return { text: "thinking", tone: "warning" };
  }

  return {
    text: prettifyName(activeAgent),
    tone: "accent",
  };
}

function buildGuidanceParts(input: {
  running: boolean;
  completionVisible: boolean;
  inputBlocked: boolean;
  multiline: boolean;
}): FooterPart[] {
  if (input.inputBlocked) {
    return [
      { text: "respond above", tone: "accent" },
      { text: "/help", tone: "dim" },
    ];
  }

  if (input.completionVisible) {
    return [
      { text: "Tab accept", tone: "accent" },
      { text: "↑/↓ move", tone: "dim" },
      { text: "Esc dismiss", tone: "dim" },
    ];
  }

  const parts: FooterPart[] = [];
  if (input.running) {
    parts.push({ text: "Ctrl+C cancel", tone: "accent" });
  }
  parts.push({ text: "Tab complete", tone: "dim" });
  if (!input.multiline) {
    parts.push({ text: "↑/↓ history", tone: "dim" });
  }
  parts.push({ text: "Shift+Enter newline", tone: "dim" });
  parts.push({ text: "/help", tone: "dim" });
  return parts;
}

function buildDataStatusPart(
  dataStatus: DataStatusSummary | null,
): FooterPart | null {
  if (!dataStatus) return null;
  const { cold, warm } = dataStatus;
  if (cold === 0 && warm === 0) return null;
  if (warm > 0 && cold > 0) {
    return { text: `${warm} loading, ${cold} on disk`, tone: "warning" };
  }
  if (warm > 0) {
    return { text: `${warm} loading`, tone: "warning" };
  }
  return { text: `${cold} on disk`, tone: "muted" };
}

export function buildFooterParts({
  running,
  activeAgent,
  tokenUsage,
  dataStatus,
  sessionId,
  runtimeInfo,
  isCloud,
  completionVisible,
  inputBlocked,
  multiline,
}: BuildFooterPartsInput): FooterPart[] {
  const parts: FooterPart[] = [buildStatusPart(running, activeAgent)];

  const dspart = buildDataStatusPart(dataStatus);
  if (dspart) parts.push(dspart);

  parts.push({
    text: buildRuntimeLabel(runtimeInfo, isCloud),
    tone: "muted",
  });

  const tokenDisplay = formatTokenDisplay(tokenUsage);
  if (tokenDisplay) {
    parts.push({ text: tokenDisplay, tone: "muted" });
  }

  parts.push(
    ...buildGuidanceParts({
      running,
      completionVisible,
      inputBlocked,
      multiline,
    }),
  );

  const durationDisplay = formatDurationDisplay(tokenUsage);
  if (durationDisplay) {
    parts.push({ text: durationDisplay, tone: "muted" });
  }

  if (sessionId) {
    parts.push({
      text: `session ${sessionId.slice(0, 8)}`,
      tone: "dim",
    });
  }

  return parts.filter((part) => part.text.trim().length > 0);
}

export function selectVisibleFooterParts(
  parts: FooterPart[],
  columns: number,
) {
  const visible: FooterPart[] = [];
  let used = STATUS_ICON_WIDTH;

  for (const part of parts) {
    const width = Array.from(part.text).length;
    const nextWidth =
      used +
      (visible.length > 0 ? FOOTER_SEPARATOR_WIDTH : 0) +
      width;

    if (nextWidth > columns) {
      if (visible.length === 0) {
        const remaining = Math.max(1, columns - STATUS_ICON_WIDTH);
        visible.push({
          ...part,
          text: truncateMiddle(part.text, remaining),
        });
      }
      break;
    }

    visible.push(part);
    used = nextWidth;
  }

  return visible;
}

function FooterPartText({ part }: { part: FooterPart }) {
  const theme = useTheme();

  switch (part.tone) {
    case "accent":
      return <Text color={theme.accent1}>{part.text}</Text>;
    case "success":
      return <Text color={theme.success}>{part.text}</Text>;
    case "warning":
      return <Text color={theme.warning}>{part.text}</Text>;
    case "muted":
      return <Text color={theme.textMuted}>{part.text}</Text>;
    case "dim":
    default:
      return <Text dimColor>{part.text}</Text>;
  }
}

/**
 * Constant bottom status line mirroring the Go TUI's live footer model:
 * one calm strip that combines run state, runtime context, and current hints.
 */
export function StatusBar({
  appStateStore,
  footerStateStore,
  sessionId,
  runtimeInfo,
  isStreaming,
  isCloud,
}: StatusBarProps) {
  const theme = useTheme();
  const { columns } = useTerminalSize();
  const activeAgent = useStore(appStateStore, (state) => state.activeAgent);
  const tokenUsage = useStore(appStateStore, (state) => state.tokenUsage);
  const dataStatus = useStore(appStateStore, (state) => state.dataStatus);
  const completionVisible = useStore(
    footerStateStore,
    (state) => state.completionVisible,
  );
  const inputBlocked = useStore(
    footerStateStore,
    (state) => state.inputBlocked,
  );
  const multiline = useStore(
    footerStateStore,
    (state) => state.multiline,
  );
  const runtimeRunning = useAuiState((state) => state.thread.isRunning);
  const running = isStreaming ?? runtimeRunning;

  const visibleParts = selectVisibleFooterParts(
    buildFooterParts({
      running,
      activeAgent,
      tokenUsage,
      dataStatus,
      sessionId,
      runtimeInfo,
      isCloud,
      completionVisible,
      inputBlocked,
      multiline,
    }),
    columns,
  );

  return (
    <Box>
      {running ? (
        <BrailleSpinner color={theme.warning} animated={false} />
      ) : (
        <Text color={theme.success}>●</Text>
      )}
      <Text> </Text>
      {visibleParts.map((part, index) => (
        <React.Fragment key={`${part.text}:${index}`}>
          {index > 0 ? <Text dimColor>{FOOTER_SEPARATOR}</Text> : null}
          <FooterPartText part={part} />
        </React.Fragment>
      ))}
    </Box>
  );
}
