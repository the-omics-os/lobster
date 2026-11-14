# MCP host configurations for `lobster mcp --stdio`

| File | Host | Where it goes |
|---|---|---|
| `claude-code.mcp.json` | Claude Code | `<project>/.mcp.json`, or `claude mcp add --transport stdio lobster -- lobster mcp --stdio` |
| `claude-desktop.config.json` | Claude Desktop | `claude_desktop_config.json` (macOS: `~/Library/Application Support/Claude/`) |
| `codex.config.toml` | Codex CLI | `~/.codex/config.toml` |
| `cursor.mcp.json` | Cursor | `<project>/.cursor/mcp.json` or `~/.cursor/mcp.json` |
| `open-science.connector.json` | AIPOCH Open Science | Import as a custom connector |

The server lists nine tools: `lobster_run`, `lobster_continue`, `lobster_answer`,
`lobster_cancel`, and `lobster_overview`, `lobster_status`, `lobster_wait`,
`lobster_inspect` and `lobster_docs` (read-only; `lobster_docs` reads
docs.omics-os.com). Hosts ask before the tools that are not read-only; the Codex
file pre-approves only `lobster_overview`, `lobster_status` and `lobster_wait`.

The server finds the Lobster project from the folder it starts in, or from the
folder named by `lobster mcp --stdio --workspace <folder>` (for desktop hosts,
which start servers elsewhere). Either way it works in the project's folder, and
a project made later above that folder is entered on the next call. A run uses
the project's provider, model and experts, even when a global provider setup or
`LOBSTER_LLM_PROVIDER` names another, and the key from the project folder's
`.env`, the server's environment or the global setup.

Desktop apps do not start from your shell and may not find `lobster` on PATH:
replace `"lobster"` with the absolute path printed by `which lobster`.
