# Build and test the local CLI

Use a separate checkout with its own environment. This runs the real Python CLI, agents, and Go terminal UI from source without replacing any global `lobster` command.

One-time setup (Python 3.12+, Go, and uv required), from the checkout root:

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e '.[full,openai,dev,classic-tui]' -e packages/lobster-ai-tui -e packages/lobster-research -e packages/lobster-transcriptomics -e packages/lobster-visualization -e packages/lobster-genomics -e packages/lobster-proteomics
```

Rebuild the terminal UI and point the CLI at it:

```bash
source .venv/bin/activate
make dev-build
export LOBSTER_TUI_BINARY="$PWD/lobster-tui/lobster-tui"
```

The explicit binary override is required. Otherwise an installed TUI wheel takes precedence over the development build.

Run setup and chat from a separate test directory, so test configuration stays out of the checkout:

```bash
mkdir -p ~/lobster-dev-test && cd ~/lobster-dev-test
lobster init      # writes .lobster_workspace/ and .env in this directory
lobster status
lobster chat
```

`lobster init` and every other command use the same workspace: `LOBSTER_WORKSPACE` if set, otherwise `.lobster_workspace` in the current directory. Credentials from setup are written to `.env` in the current directory. Avoid `--global` when you want isolated configuration. The `.env` credentials file stays in the directory where you ran init, so run later commands from that directory (or export the key) even when `LOBSTER_WORKSPACE` points elsewhere.

## Language

`--lang en|zh` overrides `LOBSTER_LANG`, which overrides the language saved in the workspace. `/lang zh` inside chat saves the selection; restart chat to refresh the terminal UI. Unknown languages fall back to English. Command names, flags, tool names, and JSON keys stay in English.

## Logging

Command output is quiet by default. Set `LOBSTER_LOG_LEVEL=INFO` or `DEBUG`, or pass `--debug` to chat, for diagnostic logs.

## Upgrading

`lobster upgrade --dry-run` prints the upgrade command without running it. Do not run a real upgrade from a development checkout: it replaces the editable install with a published package.
