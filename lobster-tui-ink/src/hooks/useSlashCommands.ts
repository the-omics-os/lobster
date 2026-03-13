/**
 * Hook for slash command interception (native + bridged).
 * Returns a submit handler that checks for slash commands before
 * passing through to the runtime.
 */

import { useState, useCallback } from "react";
import {
  dispatchCommand,
  type CommandResult,
  type CommandContext,
} from "../commands/dispatcher.js";
import type { AppConfig } from "../config.js";
import type { AppState } from "../utils/stateHandlers.js";

interface SlashCommandState {
  commandOutput: string | null;
  clearRequested: boolean;
  exitRequested: boolean;
  loading: boolean;
}

export function useSlashCommands(
  appState: AppState,
  config: AppConfig,
  sessionId?: string,
) {
  const [state, setState] = useState<SlashCommandState>({
    commandOutput: null,
    clearRequested: false,
    exitRequested: false,
    loading: false,
  });

  const applyResult = useCallback((result: CommandResult) => {
    if (result.type === "output") {
      setState((prev) => ({
        ...prev,
        commandOutput: result.text ?? null,
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
      const ctx: CommandContext = { state: appState, config, sessionId };
      const result = dispatchCommand(input, ctx);
      if (!result) return false;

      if (result instanceof Promise) {
        setState((prev) => ({ ...prev, loading: true }));
        result.then(applyResult).catch((e) => {
          setState((prev) => ({
            ...prev,
            commandOutput: `Error: ${e instanceof Error ? e.message : String(e)}`,
            loading: false,
          }));
        });
      } else {
        applyResult(result);
      }

      return true;
    },
    [appState, config, sessionId, applyResult],
  );

  const dismissOutput = useCallback(() => {
    setState((prev) => ({ ...prev, commandOutput: null }));
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
