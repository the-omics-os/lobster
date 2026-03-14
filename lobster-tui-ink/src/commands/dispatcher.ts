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
import {
  PARITY_COMMANDS,
  getGroupSubcommands,
  getNestedSubcommands,
  getParityCommand,
  isGroupedCommand,
} from "./subcommands.js";

export interface CommandOutputBlock {
  kind: string;
  title?: string;
  body?: string;
  style?: string;
  message?: string;
  level?: string;
  columns?: Array<Record<string, unknown>>;
  rows?: string[][];
  width?: number;
  items?: string[];
  ordered?: boolean;
  code?: string;
  language?: string;
  key_label?: string;
  value_label?: string;
}

export interface CommandOutputPayload {
  command?: string;
  summary?: string;
  blocks: CommandOutputBlock[];
}

export interface CommandResult {
  type: "output" | "action";
  text?: string;
  output?: CommandOutputPayload;
  action?: "clear" | "exit";
}

export interface CommandDef {
  name: string;
  description: string;
  bridged?: boolean;
  handler: (args: string, ctx: CommandContext) => CommandResult | Promise<CommandResult>;
}

export interface CommandCompletion {
  name: string;
  description: string;
}

export interface CommandContext {
  state: AppState;
  config: AppConfig;
  sessionId?: string;
}

interface BridgedCommandRequest {
  pathSegments: string[];
  args: string;
}

const NATIVE_COMMANDS: CommandDef[] = [
  {
    name: "help",
    description: "Show available commands",
    handler: () => {
      const lines = [
        "Available commands:",
        "",
        ...HELP_COMMANDS.map(
          (command) =>
            `  /${command.name.padEnd(14)} ${command.description}${command.bridged ? " *" : ""}`,
        ),
        "",
        "* = sent to backend",
        "Composer: Enter sends. Alt+Enter inserts a newline.",
        "Shift+Enter also inserts a newline when the terminal reports it.",
      ];
      return {
        type: "output",
        output: {
          blocks: [{ kind: "section", body: lines.join("\n") }],
        },
      };
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
];

const PROJECT_COMMANDS: CommandDef[] = [
  {
    name: "projects",
    description: "List cloud projects",
    bridged: true,
    handler: async (_args, ctx) => {
      if (!isFeatureEnabled("projects_datasets")) {
        return buildTextResult("Projects feature is not enabled.");
      }
      try {
        const projects = await listProjects(ctx.config);
        if (projects.length === 0) {
          return buildTextResult("No projects found.");
        }
        const lines = [
          "Projects:",
          "",
          ...projects.map(
            (project) =>
              `  ${project.name.padEnd(24)} ${String(project.dataset_count ?? 0).padStart(3)} datasets  ${project.id}`,
          ),
        ];
        return buildTextResult(lines.join("\n"));
      } catch (error) {
        return buildTextResult(asErrorMessage(error));
      }
    },
  },
  {
    name: "datasets",
    description: "List datasets in a project",
    bridged: true,
    handler: async (args, ctx) => {
      if (!isFeatureEnabled("projects_datasets")) {
        return buildTextResult("Projects feature is not enabled.");
      }
      if (!args) {
        return buildTextResult("Usage: /datasets <project_id>");
      }
      try {
        const data = await apiFetch<{
          datasets: Array<{ id: string; name: string; file_count?: number }>;
        }>(ctx.config, `/projects/${args}/datasets`);
        const datasets = data.datasets ?? [];
        if (datasets.length === 0) {
          return buildTextResult("No datasets found.");
        }
        const lines = [
          "Datasets:",
          "",
          ...datasets.map(
            (dataset) =>
              `  ${dataset.name.padEnd(24)} ${String(dataset.file_count ?? 0).padStart(3)} files  ${dataset.id}`,
          ),
        ];
        return buildTextResult(lines.join("\n"));
      } catch (error) {
        return buildTextResult(asErrorMessage(error));
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
        return buildTextResult("Agent store feature is not enabled.");
      }
      try {
        const agents = await listAgents(ctx.config);
        if (agents.length === 0) {
          return buildTextResult("No curated agents found.");
        }
        const lines = [
          "Curated agents:",
          "",
          ...agents.map(
            (agent) =>
              `  ${agent.name.padEnd(28)} v${agent.version.padEnd(8)} ${agent.description ?? ""}`,
          ),
          "",
          "Use /agent-info <name> for details.",
        ];
        return buildTextResult(lines.join("\n"));
      } catch (error) {
        return buildTextResult(asErrorMessage(error));
      }
    },
  },
  {
    name: "agent-info",
    description: "Show agent details",
    bridged: true,
    handler: async (args, ctx) => {
      if (!isFeatureEnabled("curated_agent_store")) {
        return buildTextResult("Agent store feature is not enabled.");
      }
      if (!args) {
        return buildTextResult("Usage: /agent-info <agent_name>");
      }
      try {
        const agent = await getAgentInfo(ctx.config, args.trim());
        if (!agent) {
          return buildTextResult(`Agent "${args.trim()}" not found.`);
        }
        const lines = [
          `${agent.name} v${agent.version}`,
          "",
          agent.description ?? "(no description)",
        ];
        if (agent.author) lines.push(`Author: ${agent.author}`);
        if (agent.capabilities && agent.capabilities.length > 0) {
          lines.push("", "Capabilities:");
          for (const capability of agent.capabilities) {
            lines.push(`  - ${capability}`);
          }
        }
        if (agent.tools && agent.tools.length > 0) {
          lines.push("", `Tools: ${agent.tools.join(", ")}`);
        }
        return buildTextResult(lines.join("\n"));
      } catch (error) {
        return buildTextResult(asErrorMessage(error));
      }
    },
  },
];

const BRIDGED_HELP_COMMANDS = PARITY_COMMANDS.filter(
  (command) => !["help", "clear", "exit"].includes(command.name),
).map((command) => ({ ...command, bridged: true }));

const HELP_COMMANDS = [
  ...NATIVE_COMMANDS,
  ...BRIDGED_HELP_COMMANDS,
  ...PROJECT_COMMANDS,
  ...AGENT_COMMANDS,
];

function asErrorMessage(error: unknown) {
  const message = error instanceof Error ? error.message : String(error);
  return `Error: ${message}`;
}

function buildTextResult(text: string): CommandResult {
  return {
    type: "output",
    output: {
      blocks: [{ kind: "section", body: text }],
    },
    text,
  };
}

function normalizeCommandOutput(data: unknown): CommandOutputPayload {
  if (typeof data === "string") {
    return { blocks: [{ kind: "section", body: data }] };
  }

  if (!data || typeof data !== "object") {
    return {
      blocks: [
        {
          kind: "code",
          code: JSON.stringify(data, null, 2),
          language: "json",
        },
      ],
    };
  }

  const record = data as Record<string, unknown>;
  if (Array.isArray(record.blocks)) {
    return {
      command: typeof record.command === "string" ? record.command : undefined,
      summary: typeof record.summary === "string" ? record.summary : undefined,
      blocks: record.blocks as CommandOutputBlock[],
    };
  }

  if (typeof record.result === "string") {
    return { blocks: [{ kind: "section", body: record.result }] };
  }

  if (typeof record.error === "string") {
    return {
      blocks: [{ kind: "alert", level: "error", message: record.error }],
    };
  }

  return {
    blocks: [
      {
        kind: "code",
        code: JSON.stringify(record, null, 2),
        language: "json",
      },
    ],
  };
}

async function bridgedPost(
  ctx: CommandContext,
  request: BridgedCommandRequest,
): Promise<CommandResult> {
  try {
    const endpoint = request.pathSegments.join("/");
    const sessionPath = ctx.sessionId
      ? `/sessions/${ctx.sessionId}/commands/${endpoint}`
      : `/commands/${endpoint}`;
    const data = await apiFetch<Record<string, unknown>>(ctx.config, sessionPath, {
      method: "POST",
      body: request.args ? { args: request.args } : {},
      timeoutMs: 15000,
    });
    const output = normalizeCommandOutput(data);
    return { type: "output", output, text: outputToText(output) };
  } catch (error) {
    const message = asErrorMessage(error);
    return {
      type: "output",
      output: {
        blocks: [{ kind: "alert", level: "error", message }],
      },
      text: message,
    };
  }
}

function tokenizeArgs(input: string) {
  return input.trim().split(/\s+/).filter(Boolean);
}

function buildBridgedRequest(input: string): BridgedCommandRequest | null {
  const trimmed = input.trim();
  if (!trimmed.startsWith("/")) return null;

  const rawArgs = trimmed.slice(1).trim();
  const tokens = tokenizeArgs(rawArgs);
  const topLevel = tokens[0]?.toLowerCase();
  if (!topLevel || !getParityCommand(topLevel)) {
    return null;
  }

  if (!isGroupedCommand(topLevel)) {
    return {
      pathSegments: [topLevel],
      args: rawArgs.slice(topLevel.length).trim(),
    };
  }

  const secondToken = tokens[1]?.toLowerCase();
  const matchedSubcommand = secondToken
    ? getGroupSubcommands(topLevel).find((subcommand) => subcommand.name === secondToken)
    : undefined;

  if (!matchedSubcommand) {
    return {
      pathSegments: [topLevel],
      args: rawArgs.slice(topLevel.length).trim(),
    };
  }

  const args = rawArgs
    .slice(topLevel.length)
    .trim()
    .slice((secondToken ?? matchedSubcommand.name).length)
    .trim();

  return {
    pathSegments: [topLevel, matchedSubcommand.name],
    args,
  };
}

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

  const custom = [...NATIVE_COMMANDS, ...PROJECT_COMMANDS, ...AGENT_COMMANDS].find(
    (command) => command.name === name,
  );
  if (custom) {
    return custom.handler(args, ctx);
  }

  const bridgedRequest = buildBridgedRequest(trimmed);
  if (bridgedRequest) {
    return bridgedPost(ctx, bridgedRequest);
  }

  if (getParityCommand(name)) {
    return bridgedPost(ctx, {
      pathSegments: [name],
      args,
    });
  }

  return null;
}

export function isSlashCommand(input: string): boolean {
  return input.trim().startsWith("/");
}

export function getCommandNames(): string[] {
  return HELP_COMMANDS.map((command) => command.name);
}

export function getCommandCompletions(): CommandCompletion[] {
  return HELP_COMMANDS.map(({ name, description }) => ({ name, description }));
}

export function getSubcommandCompletions(group: string): CommandCompletion[] {
  return getGroupSubcommands(group).map(({ name, description }) => ({
    name,
    description,
  }));
}

export function getNestedCommandCompletions(path: string): CommandCompletion[] {
  return getNestedSubcommands(path).map((name) => ({
    name,
    description: "",
  }));
}

export function hasNestedCompletions(topLevel: string, subcommand: string) {
  return getNestedSubcommands(`${topLevel} ${subcommand}`).length > 0;
}

export function getParityCommandNames(): string[] {
  return PARITY_COMMANDS.map((command) => command.name);
}

export function isGroupedParityCommand(name: string) {
  return isGroupedCommand(name);
}

export function getParityCommandDefinition(name: string) {
  return getParityCommand(name);
}

export function parseCommandTokens(input: string) {
  const trimmed = input.trimStart();
  if (!trimmed.startsWith("/")) {
    return [];
  }

  return trimmed.slice(1).split(/\s+/).filter(Boolean);
}

export function outputToText(output: CommandOutputPayload): string {
  return output.blocks
    .map((block) => {
      if (block.kind === "section") {
        return [block.title, block.body].filter(Boolean).join("\n");
      }
      if (block.kind === "alert") {
        return [block.title, block.message].filter(Boolean).join(": ");
      }
      if (block.kind === "hint") {
        return [block.title, block.message].filter(Boolean).join("\n");
      }
      if (block.kind === "list") {
        return (block.items ?? []).map((item) => `- ${item}`).join("\n");
      }
      if (block.kind === "kv" || block.kind === "table") {
        return JSON.stringify(block.rows ?? [], null, 2);
      }
      if (block.kind === "code") {
        return block.code ?? "";
      }
      return JSON.stringify(block, null, 2);
    })
    .filter(Boolean)
    .join("\n\n");
}
