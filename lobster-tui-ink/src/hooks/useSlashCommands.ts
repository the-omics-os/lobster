/**
 * Hook for slash command interception (native + bridged).
 * Returns a submit handler that checks for slash commands before
 * passing through to the runtime.
 */

import { useState, useCallback, useRef } from "react";
import {
  dispatchCommand,
  type CommandOutputPayload,
  type CommandResult,
  type CommandContext,
} from "../commands/dispatcher.js";
import type { AppConfig } from "../config.js";
import type { AppStateStore } from "../utils/appStateStore.js";

interface SlashCommandState {
  commandOutput: CommandOutputPayload | null;
  clearRequested: boolean;
  exitRequested: boolean;
  loading: boolean;
}

export function useSlashCommands(
  appStateStore: AppStateStore,
  config: AppConfig,
  sessionId?: string,
) {
  const [state, setState] = useState<SlashCommandState>({
    commandOutput: null,
    clearRequested: false,
    exitRequested: false,
    loading: false,
  });
  const latestRequestId = useRef(0);

  const applyResult = useCallback((result: CommandResult, requestId: number) => {
    if (requestId !== latestRequestId.current) {
      return;
    }

    if (result.type === "output") {
      setState((prev) => ({
        ...prev,
        commandOutput:
          result.output ??
          (result.text
            ? { blocks: [{ kind: "section", body: result.text }] }
            : null),
        loading: false,
      }));
    } else if (result.action === "clear") {
      setState((prev) => ({
        ...prev,
        clearRequested: true,
        commandOutput: null,
        loading: false,
      }));
    } else if (result.action === "exit") {
      setState((prev) => ({
        ...prev,
        exitRequested: true,
        loading: false,
      }));
    }
  }, []);

  /** Try to handle input as a slash command. Returns true if handled. */
  const handleInput = useCallback(
    (input: string): boolean => {
      const ctx: CommandContext = {
        state: appStateStore.getState(),
        config,
        sessionId,
      };
      const result = dispatchCommand(input, ctx);
      if (!result) return false;
      const requestId = latestRequestId.current + 1;
      latestRequestId.current = requestId;

      if (result instanceof Promise) {
        setState((prev) => ({ ...prev, commandOutput: null, loading: true }));
        result.then((resolved) => applyResult(resolved, requestId)).catch((e) => {
          if (requestId !== latestRequestId.current) {
            return;
          }

          setState((prev) => ({
            ...prev,
            commandOutput: {
              blocks: [{
                kind: "alert",
                level: "error",
                message: `Error: ${e instanceof Error ? e.message : String(e)}`,
              }],
            },
            loading: false,
          }));
        });
      } else {
        applyResult(result, requestId);
      }

      return true;
    },
    [appStateStore, config, sessionId, applyResult],
  );

  const dismissOutput = useCallback(() => {
    latestRequestId.current += 1;
    setState((prev) => ({ ...prev, commandOutput: null, loading: false }));
  }, []);

  const resetClear = useCallback(() => {
    setState((prev) => ({ ...prev, clearRequested: false }));
  }, []);

  return {
    ...state,
    handleInput,
    dismissOutput,
    resetClear,
  };
}
