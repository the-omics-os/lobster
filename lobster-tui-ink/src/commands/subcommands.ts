export interface TopLevelCommandDef {
  name: string;
  description: string;
}

export interface GroupSubcommandDef {
  name: string;
  description: string;
}

export const PARITY_COMMANDS: TopLevelCommandDef[] = [
  { name: "data", description: "Current data summary" },
  { name: "help", description: "Show available commands" },
  { name: "input-features", description: "Show input capabilities" },
  { name: "read", description: "View file contents" },
  { name: "workspace", description: "Workspace info and management" },
  { name: "plots", description: "List generated plots" },
  { name: "tokens", description: "Token usage and costs" },
  { name: "session", description: "Current session status" },
  { name: "status", description: "Subscription tier and agents" },
  { name: "status-panel", description: "System status dashboard" },
  { name: "workspace-info", description: "Workspace overview dashboard" },
  { name: "analysis-dash", description: "Analysis dashboard" },
  { name: "progress", description: "Progress monitor" },
  { name: "pipeline", description: "Pipeline operations" },
  { name: "config", description: "Configuration settings" },
  { name: "queue", description: "Download queue management" },
  { name: "metadata", description: "Metadata overview" },
  { name: "modalities", description: "Detailed modality info" },
  { name: "describe", description: "Describe a specific modality" },
  { name: "export", description: "Export to ZIP" },
  { name: "save", description: "Save state to workspace" },
  { name: "files", description: "List workspace files" },
  { name: "tree", description: "Directory tree view" },
  { name: "plot", description: "Open plots directory or a specific plot" },
  { name: "open", description: "Open file in system app" },
  { name: "restore", description: "Restore previous datasets" },
  { name: "vector-search", description: "Search ontology collections" },
  { name: "reset", description: "Reset conversation" },
  { name: "clear", description: "Clear the viewport" },
  { name: "dashboard", description: "Open the classic dashboard" },
  { name: "exit", description: "Exit the application" },
];

export const GROUPED_COMMANDS: Record<string, readonly GroupSubcommandDef[]> = {
  workspace: [
    { name: "list", description: "List datasets in the workspace" },
    { name: "load", description: "Load a dataset or file into the workspace" },
    { name: "info", description: "Show dataset information" },
    { name: "remove", description: "Remove a dataset from the workspace" },
    { name: "save", description: "Save modalities to the workspace" },
    { name: "status", description: "Show workspace status" },
  ],
  queue: [
    { name: "load", description: "Load a file into the queue" },
    { name: "list", description: "List queue entries" },
    { name: "clear", description: "Clear queued entries" },
    { name: "export", description: "Export the queue to the workspace" },
    { name: "import", description: "Import a queue export" },
  ],
  metadata: [
    { name: "publications", description: "Show publication queue metadata" },
    { name: "samples", description: "Show sample metadata statistics" },
    { name: "workspace", description: "Show workspace metadata inventory" },
    { name: "exports", description: "Show exported metadata files" },
    { name: "list", description: "Show the detailed metadata list" },
    { name: "clear", description: "Clear metadata state" },
  ],
  pipeline: [
    { name: "export", description: "Export a notebook" },
    { name: "list", description: "List saved notebooks" },
    { name: "run", description: "Run a notebook" },
    { name: "info", description: "Show pipeline information" },
  ],
  config: [
    { name: "show", description: "Show current configuration" },
    { name: "provider", description: "Inspect or switch LLM providers" },
    { name: "model", description: "Inspect or switch models" },
  ],
};

export const NESTED_SUBCOMMANDS: Record<string, readonly string[]> = {
  "queue clear": ["all", "download"],
  "queue list": ["download", "publication"],
  "metadata clear": ["all", "exports"],
  "config provider": ["list", "switch"],
  "config model": ["list", "switch"],
  save: ["--force"],
};

export function getParityCommand(name: string) {
  return PARITY_COMMANDS.find((command) => command.name === name);
}

export function getGroupSubcommands(name: string): readonly GroupSubcommandDef[] {
  return GROUPED_COMMANDS[name] ?? [];
}

export function getNestedSubcommands(path: string): readonly string[] {
  return NESTED_SUBCOMMANDS[path] ?? [];
}

export function isGroupedCommand(name: string) {
  return Object.hasOwn(GROUPED_COMMANDS, name);
}
