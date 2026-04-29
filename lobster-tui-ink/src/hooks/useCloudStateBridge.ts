/**
 * Bridge cloud `aui-state:` patches into appStateStore.
 *
 * In cloud mode, state patches arrive as `aui-state:` chunks in the data-stream
 * protocol. assistant-ui accumulates these into `thread.state` (the most recent
 * assistant message's `metadata.unstable_state`). This hook subscribes to that
 * state and mirrors known keys into appStateStore so the StatusBar, ActivityFeed,
 * and other UI consumers see cloud state changes.
 *
 * In local mode (ui-message-stream protocol), state patches arrive via the
 * `onData` callback and go directly into appStateStore — this hook is a no-op.
 */

import { useEffect, useRef } from "react";
import { useAuiState } from "@assistant-ui/store";
import { applyAppStatePatch, type AppStateStore } from "../utils/appStateStore.js";
import type { StateKey } from "../utils/stateHandlers.js";

const CLOUD_STATE_KEYS: readonly StateKey[] = [
  "active_agent",
  "agent_status",
  "activity_events",
  "progress",
  "token_usage",
  "session_title",
  "modalities",
  "plots",
  "files",
  "alerts",
  "error_detail",
];

export function useCloudStateBridge(
  appStateStore: AppStateStore,
  isCloud: boolean,
) {
  const threadState = useAuiState((s) => s.thread.state) as Record<string, unknown> | null;
  const prevRef = useRef<Record<string, unknown> | null>(null);

  useEffect(() => {
    if (!isCloud || !threadState) return;

    const prev = prevRef.current;
    prevRef.current = threadState;

    for (const key of CLOUD_STATE_KEYS) {
      const value = threadState[key];
      if (value === undefined) continue;

      // Only apply if changed from previous snapshot
      if (prev && prev[key] === value) continue;

      applyAppStatePatch(appStateStore, key, value);
    }
  }, [appStateStore, isCloud, threadState]);
}
