# MCP host configurations for `lobster mcp --stdio`

Three hosts add the server with one command, run in your project folder:

```bash
claude mcp add --transport stdio lobster -- lobster mcp --stdio   # Claude Code
codex mcp add lobster -- lobster mcp --stdio                      # Codex
pi mcp add lobster --exposure direct -- lobster mcp --stdio       # pi
pi install npm:@omicsos/pi-lobster                                # pi: short rows for Lobster's tools
```

The files below are the full configurations, including the longer time limit
per tool call described further down:

| File | Host | Where it goes |
|---|---|---|
| `claude-code.mcp.json` | Claude Code | `<project>/.mcp.json`, or `claude mcp add --transport stdio lobster -- lobster mcp --stdio` |
| `claude-desktop.config.json` | Claude Desktop | `claude_desktop_config.json` (macOS: `~/Library/Application Support/Claude/`) |
| `codex.config.toml` | Codex CLI | `~/.codex/config.toml` |
| `cursor.mcp.json` | Cursor | `<project>/.cursor/mcp.json` or `~/.cursor/mcp.json` |
| `open-science.connector.json` | AIPOCH Open Science | Import as a custom connector |
| `opencode.json` | opencode | `<project>/opencode.json` or `~/.config/opencode/opencode.json` |
| `pi.mcp.json` | pi (built-in MCP or `pi-mcp-adapter`) | `~/.pi/agent/mcp.json` or `.pi/mcp.json` |

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

`lobster_wait` without `timeout_s` returns only when the run ends or needs a
decision, which can take longer than a host allows a tool call. Codex and pi
stop a tool call after 60 seconds by default, and `codex mcp add` and
`pi mcp add` cannot change that: a longer wait is cut off and the model has to
call again. The Codex and pi files raise that limit to an hour
(`tool_timeout_sec = 3600`, `"timeout": 3600`); add the same key to an entry
made with `mcp add` to get it. In a host with a short limit the model passes
`timeout_s` under it. The pi file registers the tools directly under
their own names (`directTools`, `toolPrefix: "none"`), so the model calls
`lobster_run` rather than going through the `mcp` proxy tool; `timeout` is pi's
built-in MCP limit in seconds and `requestTimeoutMs` the adapter's;
`exposure: "direct"` declares the tools to the model in pi's built-in MCP. Each
ignores the other's keys. With the `@omicsos/pi-lobster` package, pi shows each
call as one short row.

Desktop apps do not start from your shell and may not find `lobster` on PATH:
replace `"lobster"` with the absolute path printed by `which lobster`.
