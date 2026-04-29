/**
 * Bridge cloud `aui-state:` patches into appStateStore.
 *
 * In cloud mode, state patches arrive as `aui-state:` chunks in the data-stream
 * protocol. assistant-ui accumulates these into `thread.state` (the most recent
 * assistant message's `metadata.unstable_state`). This hook subscribes to that
 * state and mirrors known keys into appStateStore so the StatusBar, ActivityFeed,
 * and other UI consumers see cloud state changes.
 *
 * Unlike the local onData path which receives incremental deltas (and uses
 * append semantics for activity_events/alerts), the cloud bridge receives
 * accumulated snapshots from thread.state. We bypass applyStatePatch and
 * write directly to the zustand store with replace semantics to avoid
 * duplicating append-style entries.
 *
 * In local mode (ui-message-stream protocol), state patches arrive via the
 * `onData` callback and go directly into appStateStore — this hook is a no-op.
 */

import { useEffect, useRef } from "react";
import { useAuiState } from "@assistant-ui/store";
import type { AppStateStore } from "../utils/appStateStore.js";
import type {
  AppState,
  ActivityEvent,
  AlertEvent,
  ProgressEntry,
  DataStatusSummary,
} from "../utils/stateHandlers.js";

function summarizeDataStatus(modalities: unknown[]): DataStatusSummary | null {
  let cold = 0;
  let warm = 0;
  let hot = 0;
  for (const raw of modalities) {
    if (!raw || typeof raw !== "object") continue;
    const status = (raw as Record<string, unknown>).data_status;
    if (status === "cold") cold++;
    else if (status === "warm") warm++;
    else hot++;
  }
  const total = cold + warm + hot;
  if (total === 0) return null;
  return { cold, warm, hot, total };
}

function normalizeTokenUsage(
  raw: unknown,
): AppState["tokenUsage"] {
  if (!raw || typeof raw !== "object") return null;
  const r = raw as Record<string, unknown>;
  const promptTokens = (r.promptTokens as number | undefined)
    ?? (r.input_tokens as number | undefined);
  const completionTokens = (r.completionTokens as number | undefined)
    ?? (r.output_tokens as number | undefined);
  if (promptTokens === undefined && completionTokens === undefined) return null;
  return { promptTokens, completionTokens };
}

function normalizeProgress(raw: unknown): Record<string, ProgressEntry> {
  if (!raw || typeof raw !== "object") return {};
  if (Array.isArray(raw)) {
    const result: Record<string, ProgressEntry> = {};
    for (const entry of raw) {
      if (entry && typeof entry === "object" && typeof (entry as Record<string, unknown>).id === "string") {
        const e = entry as ProgressEntry;
        if (!e.done) result[e.id] = e;
      }
    }
    return result;
  }
  return raw as Record<string, ProgressEntry>;
}

export function useCloudStateBridge(
  appStateStore: AppStateStore,
  isCloud: boolean,
) {
  const threadState = useAuiState((s) => s.thread.state) as Record<string, unknown> | null;
  const prevRef = useRef<Record<string, unknown> | null>(null);

  useEffect(() => {
    if (!isCloud) return;

    // Reset bookkeeping when thread state becomes null (new run, /clear, etc.)
    if (!threadState) {
      prevRef.current = null;
      return;
    }

    const prev = prevRef.current;
    prevRef.current = threadState;

    // Build a partial AppState update with snapshot-replace semantics.
    // This bypasses applyStatePatch to avoid append duplication.
    const patch: Partial<AppState> = {};
    let hasChanges = false;

    function changed(key: string): boolean {
      return threadState![key] !== undefined && (!prev || prev[key] !== threadState![key]);
    }

    if (changed("active_agent")) {
      patch.activeAgent = typeof threadState.active_agent === "string"
        ? threadState.active_agent : null;
      hasChanges = true;
    }
    if (changed("agent_status")) {
      patch.agentStatus = typeof threadState.agent_status === "string"
        ? threadState.agent_status : null;
      hasChanges = true;
    }
    if (changed("session_title")) {
      patch.sessionTitle = typeof threadState.session_title === "string"
        ? threadState.session_title : null;
      hasChanges = true;
    }
    if (changed("error_detail")) {
      patch.errorDetail = typeof threadState.error_detail === "string"
        ? threadState.error_detail : null;
      hasChanges = true;
    }
    if (changed("token_usage")) {
      patch.tokenUsage = normalizeTokenUsage(threadState.token_usage);
      hasChanges = true;
    }
    if (changed("activity_events")) {
      patch.activityEvents = Array.isArray(threadState.activity_events)
        ? (threadState.activity_events as ActivityEvent[])
        : [];
      hasChanges = true;
    }
    if (changed("alerts")) {
      patch.alerts = Array.isArray(threadState.alerts)
        ? (threadState.alerts as AlertEvent[])
        : [];
      hasChanges = true;
    }
    if (changed("progress")) {
      patch.progress = normalizeProgress(threadState.progress);
      hasChanges = true;
    }
    if (changed("modalities")) {
      const list = Array.isArray(threadState.modalities) ? threadState.modalities : [];
      patch.modalities = list;
      patch.dataStatus = summarizeDataStatus(list);
      hasChanges = true;
    }
    if (changed("plots")) {
      patch.plots = Array.isArray(threadState.plots) ? threadState.plots : [];
      hasChanges = true;
    }
    if (changed("files")) {
      patch.files = Array.isArray(threadState.files) ? threadState.files : [];
      hasChanges = true;
    }

    if (hasChanges) {
      appStateStore.setState((state) => ({ ...state, ...patch }));
    }
  }, [appStateStore, isCloud, threadState]);
}
