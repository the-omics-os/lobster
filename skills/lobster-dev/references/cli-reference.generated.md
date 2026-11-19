<!-- Generated from lobster.commands/1 by scripts/generate_cli_reference.py; do not edit -->

# Lobster CLI reference (generated)

## CLI commands

### `lobster agent`

Run Lobster as a machine interface for other programs (NDJSON on stdin and stdout).

  - `stdio` (boolean, optional) - Speak protocol v1 NDJSON on stdin and stdout.
  - `workspace` (path, optional) - Workspace directory. Can also be set via LOBSTER_WORKSPACE env var. Default: ./.lobster_workspace
  - `session-id` (string, optional) - Session ID to continue (use 'latest' for most recent session in workspace)
  - `provider` (string, optional) - LLM provider to use (anthropic, bedrock, ollama, gemini, azure, openai, openrouter, nebius, ollama-cloud, omics-os). Overrides auto-detection.
  - `model` (string, optional) - Model to use (e.g., claude-4-sonnet, llama3:70b-instruct, mixtral:8x7b). Overrides configuration.

### `lobster agents info`

Show detailed agent information.

  - `agent_name` (string, required) - Agent name to inspect

### `lobster agents install`

Install an agent package and auto-enable its agents.

  - `package_name` (string, required) - Package name to install
  - `workspace` (path, optional) - Path to workspace directory

### `lobster agents list`

List installed agents with status.

  - `workspace` (path, optional) - Path to workspace directory

### `lobster agents uninstall`

Uninstall an agent package and remove its agents from config.

  - `package_name` (string, required) - Package name to uninstall
  - `workspace` (path, optional) - Path to workspace directory

### `lobster auth login`

Login to an LLM provider via OAuth.

  - `provider` (string, required) - Provider to authenticate with (e.g., 'anthropic').
  - `no-browser` (boolean, optional) - Don't auto-open browser; print URL instead.
  - `timeout` (integer, optional) - Seconds to wait for browser callback.

### `lobster auth logout`

Remove stored OAuth credentials for a provider.

  - `provider` (string, required) - Provider to log out from (e.g., 'anthropic').

### `lobster auth status`

Show authentication status for all providers.

### `lobster chat`

Start an interactive chat session with the multi-agent system.

  - `workspace` (path, optional) - Workspace directory. Can also be set via LOBSTER_WORKSPACE env var. Default: ./.lobster_workspace
  - `session-id` (string, optional) - Session ID to continue (use 'latest' for most recent session in workspace)
  - `reasoning` (boolean, optional) - Show agent reasoning and thinking process
  - `verbose` (boolean, optional) - Show detailed tool usage and agent activity
  - `debug` (boolean, optional) - Enable debug mode with enhanced error reporting
  - `profile-timings` (boolean, optional) - Enable timing diagnostics for data manager operations
  - `provider` (string, optional) - LLM provider to use (anthropic, bedrock, ollama, gemini, azure, openai, openrouter, nebius, ollama-cloud, omics-os). Overrides auto-detection.
  - `model` (string, optional) - Model to use (e.g., claude-4-sonnet, llama3:70b-instruct, mixtral:8x7b). Overrides configuration.
  - `no-intro` (boolean, optional) - Disable the Go TUI inline intro animation (useful for automation and PTY capture).
  - `stream` (boolean, optional) - Enable real-time text streaming (default: on)
  - `ui` (string, optional) - UI mode: auto (Go TUI, then classic), go (Go TUI), classic (Rich terminal)
  - `classic` (boolean, optional) - Shorthand for --ui classic (force Rich/Textual UI)

### `lobster cloud account`

Show your Omics-OS Cloud account summary.

### `lobster cloud chat`

Start an interactive cloud chat session (lobster-cli, direct connection).

  - `session-id` (string, optional) - Resume an existing cloud session
  - `token` (string, optional) - Override stored auth token (prefer LOBSTER_CLOUD_TOKEN env var)
  - `endpoint` (string, optional) - Custom cloud API endpoint
  - `stream-endpoint` (string, optional) - Custom stream origin (e.g., https://stream.omics-os.com)
  - `project-id` (string, optional) - Associate session with a cloud project

### `lobster cloud login`

Authenticate the Python local gateway provider with --api-key <key>.

  - `api-key` (string, optional) - Omics-OS API key from app.omics-os.com/account

### `lobster cloud logout`

Log out of Omics-OS Cloud.

### `lobster cloud projects`

List your Omics-OS Cloud projects (also: omics-os lobster projects list).

  - `json` (boolean, optional) - Output JSON

### `lobster cloud query`

Send a single query to Omics-OS Cloud (agents run on ECS Fargate).

  - `question` (string, required) - Question to ask
  - `session-id` (string, optional) - Session UUID to continue (or 'latest')
  - `token` (string, optional) - Override stored auth token (prefer OMICS_OS_API_KEY env var)
  - `endpoint` (string, optional) - Custom REST API origin (e.g., https://app.omics-os.com)
  - `stream-endpoint` (string, optional) - Custom stream origin (e.g., https://stream.omics-os.com)
  - `stream` (boolean, optional) - Stream text as it arrives
  - `json` (boolean, optional) - Output JSON only on stdout
  - `project-id` (string, optional) - Associate session with a cloud project (personal projects only)

### `lobster cloud status`

Show current Omics-OS Cloud connection and usage.

### `lobster command`

Execute workspace commands without an LLM session.

  - `cmd` (string, required) - Command to execute, e.g. "data", "workspace list", "files"
  - `workspace` (path, optional) - Workspace directory. Default: ./.lobster_workspace
  - `session-id` (string, optional) - Session ID to load. Use 'latest' for most recent. Defaults to last session if not specified.
  - `json` (boolean, optional) - Output as JSON for programmatic consumption

### `lobster commands`

List Lobster's commands; --json prints the machine-readable manifest.

  - `json` (boolean, optional) - Print the lobster.commands/1 manifest as JSON.

### `lobster config create-custom`

Interactive creation of custom configuration.

### `lobster config generate-env`

Generate .env template with all available options.

### `lobster config list-models`

List all available model presets.

### `lobster config list-profiles`

List all available testing profiles.

### `lobster config models`

Interactive per-agent model configuration.

  - `workspace` (string, optional) - Workspace path (default: current directory)

### `lobster config show`

Display current configuration including agent composition.

  - `workspace` (string, optional) - Workspace directory to use

### `lobster config show-config`

Show current runtime configuration from ConfigResolver and ProviderRegistry.

  - `workspace` (path, optional) - Workspace path (default: current directory)
  - `show-all` (boolean, optional) - Show all configured agents regardless of license tier

### `lobster config test`

Test LLM provider connectivity and configuration.

  - `profile` (string, optional) - Profile to test (optional, auto-detects if not provided)
  - `agent` (string, optional) - Specific agent to test

### `lobster config-test`

Test API connectivity and validate configuration.

  - `json` (boolean, optional) - Output results as JSON (machine-readable)

### `lobster dashboard`

Launch interactive dashboard (Textual UI).

  - `workspace` (path, optional) - Workspace directory. Can also be set via LOBSTER_WORKSPACE env var. Default: ./.lobster_workspace

### `lobster init`

Initialize Lobster AI configuration.

  - `folder` (path, optional) - Folder to set up as a project (default: the current folder)
  - `global` (boolean, optional) - Save configuration globally (~/.config/lobster/) for all workspaces
  - `force` (boolean, optional) - Overwrite existing configuration
  - `non-interactive` (boolean, optional) - Non-interactive mode for CI/CD (requires API key flags)
  - `use-omics-os` (boolean, optional) - Use Omics-OS Cloud with OMICS_OS_API_KEY or saved credentials (non-interactive mode)
  - `omics-os-key` (string, optional) - Omics-OS Cloud gateway API key (non-interactive mode; saved, not checked)
  - `anthropic-key` (string, optional) - Claude API key (non-interactive mode)
  - `bedrock-access-key` (string, optional) - AWS Bedrock access key (non-interactive mode)
  - `bedrock-secret-key` (string, optional) - AWS Bedrock secret key (non-interactive mode)
  - `use-ollama` (boolean, optional) - Use Ollama (local LLM) provider (non-interactive mode)
  - `ollama-model` (string, optional) - Ollama model name (default: llama3:8b-instruct, non-interactive mode)
  - `model` (string, optional) - Default model ID for the selected provider. Saved to workspace config.
  - `gemini-key` (string, optional) - Google API key (non-interactive mode)
  - `openai-key` (string, optional) - OpenAI API key (non-interactive mode)
  - `openrouter-key` (string, optional) - OpenRouter API key (non-interactive mode)
  - `nebius-key` (string, optional) - Nebius AI Studio API key (non-interactive mode)
  - `ollama-cloud-key` (string, optional) - Ollama Cloud API key (non-interactive mode)
  - `azure-endpoint` (string, optional) - Azure AI endpoint URL (non-interactive mode)
  - `azure-credential` (string, optional) - Azure AI API credential (non-interactive mode)
  - `profile` (string, optional) - Agent profile (development, production, performance, max). Only for Anthropic/Bedrock providers.
  - `ncbi-key` (string, optional) - NCBI API key (optional, non-interactive mode)
  - `cloud-key` (string, optional) - Omics-OS Cloud API key (optional, enables premium tier)
  - `cloud-endpoint` (string, optional) - Custom cloud endpoint URL (optional)
  - `skip-ssl-test` (boolean, optional) - Skip SSL connectivity test during init
  - `ssl-verify` (boolean, optional) - Enable/disable SSL verification (non-interactive mode)
  - `ssl-cert-path` (string, optional) - Path to custom CA certificate bundle
  - `agents` (string, optional) - Comma-separated list of agents to enable (non-interactive mode)
  - `preset` (string, optional) - Agent preset name (scrna-basic, scrna-full, multiomics-full)
  - `auto-agents` (boolean, optional) - Use LLM to suggest agents based on description (requires --agents-description)
  - `agents-description` (string, optional) - Workflow description for LLM agent suggestion (with --auto-agents)
  - `skip-docling` (boolean, optional) - Skip docling installation prompt (non-interactive mode)
  - `install-docling` (boolean, optional) - Install docling for PDF intelligence (non-interactive mode)
  - `install-vector-search` (boolean, optional) - Install Smart Standardization dependencies (the lobster-vector package)
  - `skip-extras` (boolean, optional) - Skip optional package installation (provider, TUI, extended-data)
  - `ui` (string, optional) - UI mode for interactive init: auto (Go TUI if available, else questionary, else classic), go (require Go TUI), classic (Rich prompts only)
  - `name` (string, optional) - Project name (default: the folder name)
  - `provider` (string, optional) - Provider for a project made without a terminal (no key is read or saved)
  - `lamin` (boolean, optional) - Keep a LaminDB catalog (default: ask; No)

### `lobster lamin init`

Switch the catalog on for this workspace and create it now.

  - `workspace` (string, optional) - Workspace directory. Default: ./.lobster_workspace

### `lobster lamin ls`

List the files or sessions in the workspace catalog.

  - `runs` (boolean, optional) - List sessions
  - `artifacts` (boolean, optional) - List files (default)
  - `limit` (integer, optional) - Maximum number of rows
  - `workspace` (string, optional) - Workspace directory. Default: ./.lobster_workspace
  - `json` (boolean, optional) - Output results as JSON (machine-readable)

### `lobster lamin prune`

Remove the kept copies of earlier versions; the records stay.

  - `dry-run` (boolean, optional) - Show what would be removed; remove nothing
  - `yes` (boolean, optional) - Do not ask first
  - `workspace` (string, optional) - Workspace directory. Default: ./.lobster_workspace
  - `json` (boolean, optional) - Output results as JSON (machine-readable)

### `lobster lamin status`

Show whether this workspace is recorded, and where.

  - `workspace` (string, optional) - Workspace directory. Default: ./.lobster_workspace
  - `json` (boolean, optional) - Output results as JSON (machine-readable)

### `lobster lamin use`

Record this workspace in a LaminDB instance of your own instead.

  - `target` (string, optional) - Instance as owner/name, or the storage folder of a local instance
  - `clear` (boolean, optional) - Go back to the workspace catalog
  - `workspace` (string, optional) - Workspace directory. Default: ./.lobster_workspace

### `lobster mcp`

Run Lobster as an MCP server for desktop and agent hosts (JSON-RPC on stdin and stdout).

  - `stdio` (boolean, optional) - Speak MCP (JSON-RPC) on stdin and stdout.
  - `workspace` (path, optional) - Folder the Lobster project is found from (default: the current folder).

### `lobster metadata`

Display metadata overview and statistics.

  - `subcommand` (string, optional) - Subcommand: overview, publications, samples, workspace, exports, list, clear
  - `target` (string, optional) - For clear: exports or all
  - `workspace` (path, optional) - Workspace directory. Default: ./.lobster_workspace
  - `status` (string, optional) - Status filter for publications subcommand

### `lobster projects delete`

Delete a project's Lobster workspace; the folder and your other files stay.

  - `folder` (path, optional) - Project folder (default: the project in the current folder)
  - `yes` (boolean, optional) - Delete without asking (needed without a terminal)
  - `dry-run` (boolean, optional) - List what would be removed; change nothing
  - `json` (boolean, optional) - Print the result as lobster.project/1 JSON

### `lobster projects info`

Show a project's identity, provider, model, experts and runs.

  - `folder` (path, optional) - Project folder (default: the project that contains the current folder)
  - `json` (boolean, optional) - Print the lobster.project/1 record as JSON

### `lobster projects list`

List the projects this machine has set up.

  - `json` (boolean, optional) - Print the lobster.project/1 listing as JSON

### `lobster purge`

Remove all Lobster AI files from your system.

  - `scope` (string, optional) - What to purge: 'global' (~/.lobster, ~/.config/lobster), 'workspace' (current), or 'all'
  - `workspace` (path, optional) - Workspace directory to purge (default: current workspace)
  - `dry-run` (boolean, optional) - Show what would be deleted without actually deleting
  - `force` (boolean, optional) - Skip confirmation prompt

### `lobster query`

Send a single query, or run a local slash command when QUESTION starts with '/'.

  - `question` (string, required)
  - `workspace` (path, optional) - Workspace directory. Defaults to ./.lobster_workspace or LOBSTER_WORKSPACE.
  - `session-id` (string, optional) - Continue a saved session. Use 'latest' for the most recent one.
  - `reasoning` (boolean, optional) - Show agent reasoning.
  - `verbose` (boolean, optional) - Show agent and tool trace details.
  - `debug` (boolean, optional) - Enable debug logging.
  - `output` (path, optional) - Write the final answer to a file.
  - `profile-timings` (boolean, optional) - Show data-manager timing diagnostics.
  - `provider` (string, optional) - Provider override. Uses the configured provider when omitted.
  - `model` (string, optional) - Model override. Uses the configured model when omitted.
  - `stream` (boolean, optional) - Stream text as it arrives. Disabled automatically in trace mode.
  - `json` (boolean, optional) - Emit JSON only on stdout.

### `lobster replay`

Re-run an exported notebook and compare its output with the saved file. Exit codes: 0 MATCH, 1 MISMATCH, 2 usage error, 3 NOT_COMPARED, 4 REFUSED, 5 the notebook failed to run, 6 MATCH_CUSTOM_POLICY, 130 interrupted. Compares what the notebook writes against what the session saved, and detects changes it can see; it does not prove how the outputs were computed.

  - `verify` (path, required) - Exported notebook to re-run and check
  - `input` (path, optional) - Input data file of the notebook (.h5ad)
  - `policy` (path, optional) - Comparison policy (JSON). Default: exact values, without the session's own provenance and lineage records
  - `workspace` (string, optional) - Workspace directory. Default: ./.lobster_workspace
  - `json` (boolean, optional) - Output results as JSON (machine-readable)
  - `timeout` (integer, optional) - Time limit per cell, in seconds; a cell that runs longer stops the check (0: no limit)

### `lobster scaffold agent`

Generate a complete agent plugin package.

  - `name` (string, required) - Agent name in snake_case (e.g., epigenomics_expert)
  - `display-name` (string, required) - Human-readable name (e.g., 'Epigenomics Expert')
  - `description` (string, required) - Agent capabilities description
  - `children` (string, optional) - Comma-separated child agent names
  - `output-dir` (path, optional) - Output directory
  - `author-name` (string, optional) - Package author name
  - `author-email` (string, optional) - Package author email

### `lobster serve`

Start the agent system as an API server (for React UI).

  - `port` (integer, optional)
  - `host` (string, optional) - Bind address. Anything other than localhost requires a token.
  - `token` (string, optional) - Bearer token. Prefer LOBSTER_SERVE_TOKEN: --token shows in ps.

### `lobster sessions list`

List saved sessions in a workspace, newest first.

  - `workspace` (path, optional) - Workspace directory. Defaults to ./.lobster_workspace or LOBSTER_WORKSPACE.
  - `json` (boolean, optional) - Print the lobster.sessions/1 listing as JSON.

### `lobster skills install`

Copy the shipped skills into coding-agent skill folders (Claude Code, Codex, ...).

  - `harness` (string, optional) - Harness to install into (claude-code, codex, cursor, os, pi); repeatable. Default: every harness found in your home directory.
  - `target` (path, optional) - Install into this skill folder instead.
  - `dry-run` (boolean, optional) - Show what would be copied; write nothing.
  - `json` (boolean, optional) - Print the lobster.skills-install/1 plan as JSON.

### `lobster skills list`

List the agent skills shipped with Lobster.

  - `json` (boolean, optional) - Print the shipped skills as JSON.

### `lobster status`

Display subscription tier, installed packages, and available agents.

### `lobster upgrade`

Upgrade lobster-ai to the latest published version.

  - `dry-run` (boolean, optional) - Print the upgrade command without running it

### `lobster vector-search`

Search all ontology collections via semantic vector similarity.

  - `query_text` (string, required) - Biomedical term to search
  - `top-k` (integer, optional) - Results per collection (default: 5)
  - `pretty` (boolean, optional) - Pretty-print or compact JSON

## Slash commands

### `/analysis-dash`

Show analysis monitoring dashboard

### `/clear`

Clear screen

### `/config`

Show current configuration (provider, profile, config files)

### `/config model`

List available Ollama models

### `/config model <name>`

Switch to specified Ollama model (runtime only)

### `/config model <name> --save`

Switch Ollama model and persist to workspace

### `/config provider`

List available LLM providers

### `/config provider <name>`

Switch to specified provider (runtime only)

### `/config provider <name> --save`

Switch provider and persist to workspace

### `/dashboard`

Switch to interactive dashboard (Textual UI)

### `/data`

Show current data summary

### `/describe`

Show detailed information about a specific modality

### `/exit`

Exit the chat

### `/export`

Export to ZIP: data + saved figures with HTML views and every recorded PNG; no new PNG is rendered, --no-png only disables generation (--force to re-serialize all)

### `/files`

List workspace files

### `/help`

Show this help message

### `/input-features`

Show input capabilities and navigation features

### `/lang`

Set the UI language (en, zh) for this session and the next launch

### `/metadata`

Show smart metadata overview with stats & next steps

### `/metadata clear`

Clear metadata (memory + workspace/metadata/)

### `/metadata clear all`

Clear ALL metadata (memory + disk + exports)

### `/metadata clear exports`

Clear export files (workspace/exports/)

### `/metadata exports`

Export files with categories & usage hints

### `/metadata list`

Legacy detailed list (use overview instead)

### `/metadata publications`

Publication queue status breakdown

### `/metadata samples`

Aggregated sample statistics & disease coverage

### `/metadata workspace`

Categorized file inventory across all locations

### `/modalities`

Show detailed modality information

### `/open`

Open file or folder in system default application

### `/plot`

Open plots directory or specific plot

### `/plots`

List all generated plots

### `/progress`

Show multi-task progress monitor

### `/queue`

Show queue status

### `/queue clear`

Clear publication queue

### `/queue clear all`

Clear all queues (publication + download)

### `/queue clear download`

Clear download queue

### `/queue export`

Export queue to workspace for persistence

### `/queue list`

List queued items

### `/queue load`

Load file into queue (supports .ris, more coming)

### `/read`

View file contents (inspection only)

### `/reset`

Reset conversation

### `/restore`

Restore previous session datasets

### `/save`

Save current state to workspace (--force to re-save all)

### `/session`

Show current session status

### `/status`

Show subscription tier, packages, and agents

### `/status-panel`

Show comprehensive system status panel

### `/tokens`

Execute /tokens command

### `/tree`

Show directory tree view

### `/vector-search`

Search all ontology collections (semantic vector similarity)

### `/workspace`

Show workspace status and information

### `/workspace info`

Show dataset information

### `/workspace list`

List available datasets in workspace

### `/workspace load`

Load dataset or file into workspace

### `/workspace remove`

Remove modality(ies) by index, pattern, or name

### `/workspace save`

Save modality to workspace

### `/workspace-info`

Show detailed workspace overview
