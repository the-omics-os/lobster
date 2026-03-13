/**
 * Hook for slash command interception.
 * Returns a submit handler that checks for slash commands before
 * passing through to the runtime.
 */

import { useState, useCallback } from "react";
import { dispatchCommand, type CommandResult } from "../commands/dispatcher.js";
import type { AppState } from "../utils/stateHandlers.js";

interface SlashCommandState {
  /** Latest command output text to display. */
  commandOutput: string | null;
  /** Whether a /clear was triggered. */
  clearRequested: boolean;
  /** Whether /exit or /quit was triggered. */
  exitRequested: boolean;
}

export function useSlashCommands(appState: AppState) {
  const [state, setState] = useState<SlashCommandState>({
    commandOutput: null,
    clearRequested: false,
    exitRequested: false,
  });

  /** Try to handle input as a slash command. Returns true if handled. */
  const handleInput = useCallback(
    (input: string): boolean => {
      const result = dispatchCommand(input, appState);
      if (!result) return false;

      if (result.type === "output") {
        setState((prev) => ({
          ...prev,
          commandOutput: result.text ?? null,
        }));
      } else if (result.action === "clear") {
        setState((prev) => ({
          ...prev,
          clearRequested: true,
          commandOutput: null,
        }));
      } else if (result.action === "exit") {
        setState((prev) => ({
          ...prev,
          exitRequested: true,
        }));
      }

      return true;
    },
    [appState],
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
