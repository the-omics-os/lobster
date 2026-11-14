# MCP host configurations for `lobster mcp --stdio`

| File | Host | Where it goes |
|---|---|---|
| `claude-code.mcp.json` | Claude Code | `<project>/.mcp.json`, or `claude mcp add --transport stdio lobster -- lobster mcp --stdio` |
| `claude-desktop.config.json` | Claude Desktop | `claude_desktop_config.json` (macOS: `~/Library/Application Support/Claude/`) |
| `codex.config.toml` | Codex CLI | `~/.codex/config.toml` |
| `cursor.mcp.json` | Cursor | `<project>/.cursor/mcp.json` or `~/.cursor/mcp.json` |
| `open-science.connector.json` | AIPOCH Open Science | Import as a custom connector |

The server lists six tools: `lobster_run`, `lobster_continue`, `lobster_answer`,
`lobster_cancel`, and `lobster_status` and `lobster_inspect` (both read-only).
Hosts ask before the tools that are not read-only; the Codex file pre-approves
only `lobster_status`.

Desktop apps do not start from your shell and may not find `lobster` on PATH:
replace `"lobster"` with the absolute path printed by `which lobster`.
