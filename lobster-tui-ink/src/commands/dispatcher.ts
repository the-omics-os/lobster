/**
 * Slash command dispatcher — routes /commands to native or bridged handlers.
 * Native commands execute client-side. Bridged commands POST to backend.
 */

import type { AppConfig } from "../config.js";
import { apiFetch } from "../api/apiClient.js";
import type { AppState } from "../utils/stateHandlers.js";
import { isFeatureEnabled } from "../api/featureFlags.js";
import { listProjects } from "../api/projects.js";
import { listAgents, getAgentInfo } from "../api/agents.js";

export interface CommandResult {
  type: "output" | "action";
  text?: string;
  action?: "clear" | "exit";
}

export interface CommandDef {
  name: string;
  description: string;
  bridged?: boolean;
  handler: (args: string, ctx: CommandContext) => CommandResult | Promise<CommandResult>;
}

export interface CommandContext {
  state: AppState;
  config: AppConfig;
  sessionId?: string;
}

const BRIDGED_COMMANDS: CommandDef[] = [
  {
    name: "files",
    description: "List workspace files",
    bridged: true,
    handler: async (_args, ctx) => bridgedPost(ctx, "/files"),
  },
  {
    name: "pipeline",
    description: "Show pipeline status",
    bridged: true,
    handler: async (_args, ctx) => bridgedPost(ctx, "/pipeline"),
  },
  {
    name: "status",
    description: "Show agent status",
    bridged: true,
    handler: async (_args, ctx) => bridgedPost(ctx, "/status"),
  },
  {
    name: "tokens",
    description: "Show token usage",
    bridged: true,
    handler: (_args, ctx) => {
      const usage = ctx.state.tokenUsage;
      if (!usage) {
        return { type: "output", text: "No token usage data available." };
      }
      const prompt = usage.promptTokens ?? 0;
      const completion = usage.completionTokens ?? 0;
      const lines = [
        "Token usage:",
        `  Prompt:     ${prompt.toLocaleString()}`,
        `  Completion: ${completion.toLocaleString()}`,
        `  Total:      ${(prompt + completion).toLocaleString()}`,
      ];
      return { type: "output", text: lines.join("\n") };
    },
  },
];

const NATIVE_COMMANDS: CommandDef[] = [
  {
    name: "help",
    description: "Show available commands",
    handler: (_args, _ctx) => {
      const all = ALL_COMMANDS;
      const lines = [
        "Available commands:",
        "",
        ...all.map(
          (c) => `  /${c.name.padEnd(12)} ${c.description}${c.bridged ? " *" : ""}`,
        ),
        "",
        "* = sent to backend",
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
    handler: (_args, ctx) => {
      if (ctx.state.modalities.length === 0) {
        return { type: "output", text: "No modalities loaded." };
      }
      const lines = ctx.state.modalities.map((m, i) => {
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

const PROJECT_COMMANDS: CommandDef[] = [
  {
    name: "projects",
    description: "List cloud projects",
    bridged: true,
    handler: async (_args, ctx) => {
      if (!isFeatureEnabled("projects_datasets")) {
        return { type: "output", text: "Projects feature is not enabled." };
      }
      try {
        const projects = await listProjects(ctx.config);
        if (projects.length === 0) {
          return { type: "output", text: "No projects found." };
        }
        const lines = [
          "Projects:",
          "",
          ...projects.map(
            (p) =>
              `  ${p.name.padEnd(24)} ${String(p.dataset_count ?? 0).padStart(3)} datasets  ${p.id}`,
          ),
        ];
        return { type: "output", text: lines.join("\n") };
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        return { type: "output", text: `Error: ${msg}` };
      }
    },
  },
  {
    name: "datasets",
    description: "List datasets in a project",
    bridged: true,
    handler: async (args, ctx) => {
      if (!isFeatureEnabled("projects_datasets")) {
        return { type: "output", text: "Projects feature is not enabled." };
      }
      if (!args) {
        return { type: "output", text: "Usage: /datasets <project_id>" };
      }
      try {
        const data = await apiFetch<{ datasets: Array<{ id: string; name: string; file_count?: number }> }>(
          ctx.config,
          `/projects/${args}/datasets`,
        );
        const datasets = data.datasets ?? [];
        if (datasets.length === 0) {
          return { type: "output", text: "No datasets found." };
        }
        const lines = [
          "Datasets:",
          "",
          ...datasets.map(
            (d) => `  ${d.name.padEnd(24)} ${String(d.file_count ?? 0).padStart(3)} files  ${d.id}`,
          ),
        ];
        return { type: "output", text: lines.join("\n") };
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        return { type: "output", text: `Error: ${msg}` };
      }
    },
  },
];

const AGENT_COMMANDS: CommandDef[] = [
  {
    name: "agents",
    description: "Browse curated agents",
    bridged: true,
    handler: async (_args, ctx) => {
      if (!isFeatureEnabled("curated_agent_store")) {
        return { type: "output", text: "Agent store feature is not enabled." };
      }
      try {
        const agents = await listAgents(ctx.config);
        if (agents.length === 0) {
          return { type: "output", text: "No curated agents found." };
        }
        const lines = [
          "Curated agents:",
          "",
          ...agents.map(
            (a) =>
              `  ${a.name.padEnd(28)} v${a.version.padEnd(8)} ${a.description ?? ""}`,
          ),
          "",
          "Use /agent-info <name> for details.",
        ];
        return { type: "output", text: lines.join("\n") };
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        return { type: "output", text: `Error: ${msg}` };
      }
    },
  },
  {
    name: "agent-info",
    description: "Show agent details",
    bridged: true,
    handler: async (args, ctx) => {
      if (!isFeatureEnabled("curated_agent_store")) {
        return { type: "output", text: "Agent store feature is not enabled." };
      }
      if (!args) {
        return { type: "output", text: "Usage: /agent-info <agent_name>" };
      }
      try {
        const agent = await getAgentInfo(ctx.config, args.trim());
        if (!agent) {
          return { type: "output", text: `Agent "${args.trim()}" not found.` };
        }
        const lines = [
          `${agent.name} v${agent.version}`,
          "",
          agent.description ?? "(no description)",
        ];
        if (agent.author) lines.push(`Author: ${agent.author}`);
        if (agent.capabilities && agent.capabilities.length > 0) {
          lines.push("", "Capabilities:");
          for (const cap of agent.capabilities) {
            lines.push(`  - ${cap}`);
          }
        }
        if (agent.tools && agent.tools.length > 0) {
          lines.push("", `Tools: ${agent.tools.join(", ")}`);
        }
        return { type: "output", text: lines.join("\n") };
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        return { type: "output", text: `Error: ${msg}` };
      }
    },
  },
];

const ALL_COMMANDS = [...NATIVE_COMMANDS, ...BRIDGED_COMMANDS, ...PROJECT_COMMANDS, ...AGENT_COMMANDS];

/** POST a slash command to backend and format the response. */
async function bridgedPost(
  ctx: CommandContext,
  endpoint: string,
): Promise<CommandResult> {
  try {
    const sessionPath = ctx.sessionId
      ? `/sessions/${ctx.sessionId}/commands${endpoint}`
      : `/commands${endpoint}`;
    const data = await apiFetch<Record<string, unknown>>(
      ctx.config,
      sessionPath,
    );
    // Format response as readable text
    const text =
      typeof data === "string"
        ? data
        : JSON.stringify(data, null, 2);
    return { type: "output", text };
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    return { type: "output", text: `Error: ${msg}` };
  }
}

/** Parse and dispatch a slash command. Returns null if not a command. */
export function dispatchCommand(
  input: string,
  ctx: CommandContext,
): CommandResult | Promise<CommandResult> | null {
  const trimmed = input.trim();
  if (!trimmed.startsWith("/")) return null;

  const spaceIdx = trimmed.indexOf(" ");
  const name = (
    spaceIdx === -1 ? trimmed.slice(1) : trimmed.slice(1, spaceIdx)
  ).toLowerCase();
  const args = spaceIdx === -1 ? "" : trimmed.slice(spaceIdx + 1).trim();

  const cmd = ALL_COMMANDS.find((c) => c.name === name);
  if (!cmd) return null;

  return cmd.handler(args, ctx);
}

/** Check if input starts with /. */
export function isSlashCommand(input: string): boolean {
  return input.trim().startsWith("/");
}

/** Get all command names for autocomplete. */
export function getCommandNames(): string[] {
  return ALL_COMMANDS.map((c) => c.name);
}
