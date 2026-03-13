/**
 * Slash command dispatcher — routes /commands to native handlers.
 * Native commands execute client-side without backend calls.
 */

import type { AppState } from "../utils/stateHandlers.js";

export interface CommandResult {
  type: "output" | "action";
  text?: string;
  action?: "clear" | "exit";
}

export interface CommandDef {
  name: string;
  description: string;
  handler: (args: string, state: AppState) => CommandResult;
}

const NATIVE_COMMANDS: CommandDef[] = [
  {
    name: "help",
    description: "Show available commands",
    handler: (_args, _state) => {
      const lines = [
        "Available commands:",
        "",
        ...NATIVE_COMMANDS.map(
          (c) => `  /${c.name.padEnd(12)} ${c.description}`,
        ),
        "",
        "Bridged commands (sent to backend):",
        "  /files        List workspace files",
        "  /pipeline     Show pipeline status",
        "  /status       Show agent status",
        "  /tokens       Show token usage",
      ];
      return { type: "output", text: lines.join("\n") };
    },
  },
  {
    name: "clear",
    description: "Clear the viewport",
    handler: () => ({ type: "action", action: "clear" }),
  },
  {
    name: "exit",
    description: "Exit the application",
    handler: () => ({ type: "action", action: "exit" }),
  },
  {
    name: "quit",
    description: "Exit the application",
    handler: () => ({ type: "action", action: "exit" }),
  },
  {
    name: "data",
    description: "Show loaded modalities",
    handler: (_args, state) => {
      if (state.modalities.length === 0) {
        return { type: "output", text: "No modalities loaded." };
      }
      const lines = state.modalities.map((m, i) => {
        const mod = m as Record<string, unknown>;
        const name = mod.name ?? mod.type ?? `modality-${i + 1}`;
        const samples = mod.sample_count ?? mod.samples ?? "?";
        return `  ${String(i + 1).padStart(2)}. ${name} (${samples} samples)`;
      });
      return {
        type: "output",
        text: `Loaded modalities:\n${lines.join("\n")}`,
      };
    },
  },
];

/** Parse input and dispatch if it's a slash command. Returns null if not a command. */
export function dispatchCommand(
  input: string,
  state: AppState,
): CommandResult | null {
  const trimmed = input.trim();
  if (!trimmed.startsWith("/")) return null;

  const spaceIdx = trimmed.indexOf(" ");
  const name = (spaceIdx === -1 ? trimmed.slice(1) : trimmed.slice(1, spaceIdx)).toLowerCase();
  const args = spaceIdx === -1 ? "" : trimmed.slice(spaceIdx + 1).trim();

  const cmd = NATIVE_COMMANDS.find((c) => c.name === name);
  if (!cmd) return null;

  return cmd.handler(args, state);
}

/** Check if input looks like a slash command (starts with /). */
export function isSlashCommand(input: string): boolean {
  return input.trim().startsWith("/");
}

/** Get all command names for autocomplete. */
export function getCommandNames(): string[] {
  return NATIVE_COMMANDS.map((c) => c.name);
}
