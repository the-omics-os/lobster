# Projects, MCP Server, LaminDB Catalog and Replay (Lobster AI 1.3)

Everything here needs Lobster AI **1.3.0 or later**. Check `lobster --version`.
Obtain permission before installing, writing credentials or starting a run.

## Install

```bash
uv tool install --python 3.12 lobster-ai            # venv: uv venv --python 3.12, then uv pip install lobster-ai
lobster init ~/my-study                              # a project folder (home itself is refused); asks which analysis packages to install
cd ~/my-study                                        # run Lobster from inside the project
lobster skills list                                  # skills shipped inside the wheel
lobster skills install [--harness claude-code|codex|cursor|os|pi] [--dry-run]
```

`lobster skills install` never replaces a skill folder it did not create.
Plain `lobster-ai` is small and includes the default provider, Omics-OS Cloud;
`lobster init` installs the analysis packages and other providers the user picks.
`'lobster-ai[full]'` installs everything at once (a larger install).

## Projects

```bash
lobster init [folder]                     # interactive; makes the folder a project
lobster init <folder> --provider <name> [--model <m>] [--agents a,b | --preset p]
lobster projects list | info | delete     # each takes --json; `project` also works
lobster projects delete --dry-run         # lists what would go; --yes for scripts
```

- A project is a folder with a hidden `.lobster_workspace`. Lobster finds it by
  walking up from the current folder; `--workspace` or `LOBSTER_WORKSPACE` win.
  A workspace made before 1.3 still works as in 1.2.
- `init` refuses the home folder and any folder above it (and `/`): use a
  subfolder such as `~/my-study`. `--global` sets user-wide defaults instead.
- Without a terminal, `init <folder>` takes the provider from `--provider`, the
  project file, the global provider file or `LOBSTER_LLM_PROVIDER`; with none it
  exits 1 (`no_provider_setup`) and writes nothing. It never reads a key flag.
- `projects delete` removes only what Lobster created, never the folder itself.
- Projects are local: no sync with Omics-OS Cloud projects.

## MCP server (`lobster mcp --stdio`)

Lobster runs as a child process of an MCP host. Config for Claude Code, Claude
Desktop or Cursor:

```json
{ "mcpServers": { "lobster": { "command": "lobster", "args": ["mcp", "--stdio"] } } }
```

(Claude Code also takes `"type": "stdio"`.) Add `"--workspace", "<project folder>"`
to the args to pin the project. The server reads its provider key from its own
environment, the global credentials or the project's `.env`.

Nine tools: `lobster_overview` (project, provider, model, whether a key is present,
experts), `lobster_run` (a free-text question, optional local file), `lobster_status`,
`lobster_wait` (returns at a question or the end, or after an optional `timeout_s`),
`lobster_answer`, `lobster_cancel`, `lobster_continue` (one follow-up instruction on
a completed run), `lobster_inspect` (structure of a saved `.h5ad`, no matrix load),
`lobster_docs` (public docs search, read-only).

- Plan first: a run with a file loads and describes it, then asks the host to
  approve a plan. Nothing else runs until the plan is answered.
- Without `output_dir`, a run goes to `runs/<UTC time>` in the project. The folder
  holds `result.json`, `lobster_run.ipynb`, saved data and `plots/` (HTML only).
- One run at a time per server; absolute paths only; a stopped first turn cannot be
  continued. A second process on the same run folder is refused (`output_dir_in_use`).
- `lobster agent --stdio` is the JSON Lines host protocol (one terminal `run_state`
  per run); `lobster commands --json` lists commands as data.

## Optional LaminDB catalog (off by default)

With it off, Lobster never imports LaminDB and makes no `.lamindb` folder. It needs
the `[lamin]` extra (not part of `[full]`): `lobster init` installs it at its LaminDB
step (`lobster init --lamin`); `lobster lamin init` or `LOBSTER_LAMIN=1` turn it on later.
It records sessions, saved datasets and
exported notebooks in `<workspace>/.lamindb/` (local SQLite, no account).

```bash
lobster lamin status [--json]
lobster lamin ls [--runs] [--json]
lobster lamin prune [--dry-run] [--yes]    # removes kept older file versions only
lobster lamin use <instance> | --clear     # your own LaminDB instance
```

If the extra is missing while the catalog is on, Lobster warns and runs unrecorded.

## Verify an exported notebook (`lobster replay --verify`)

```bash
lobster replay --verify <notebook.ipynb> -w <workspace> [--input <file>] [--json]
```

It re-runs the notebook and compares what it writes with the files the session
saved. It detects changes it can see; it does not prove how the outputs were computed.

| Exit | Verdict | Meaning |
|---|---|---|
| 0 | MATCH | Every saved file was compared (or was the replay's own input) and equal |
| 1 | MISMATCH | A compared file differs |
| 3 | NOT_COMPARED | Not enough recorded to compare, e.g. a session migrated from 1.2.x |
| 4 | REFUSED | A reference changed, or `--input` is not a file the session loaded |
| 5 | NOTEBOOK_FAILED | The notebook did not run |
| 6 | MATCH_CUSTOM_POLICY | A match under the user's own `--policy` |

A file the user loaded needs `--input`; a GEO download is found automatically. For an
MCP run pass the run folder as `-w`. Plot cells run but figures are not compared.
Report the verdict and exit code as they are; never present NOT_COMPARED as a pass.
