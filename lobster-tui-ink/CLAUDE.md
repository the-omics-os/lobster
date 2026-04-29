# lobster-tui-ink — React Ink Terminal UI

React Ink terminal interface for Lobster AI built on `@assistant-ui/react-ink` and `@assistant-ui/react-data-stream`.

This surface is the primary interactive CLI target in this worktree. The Go/Charm TUI remains the protected reference implementation and fallback surface. Ink-side work must not silently regress Go behavior or shared protocol semantics.

## Surface Summary

- **Framework:** React 19 + Ink 6 + `@assistant-ui/react-ink`
- **Runtime:** `useDataStreamRuntime` over the `ui-message-stream` / DataStream protocol
- **Binary:** Bun-compiled `dist/lobster-chat`
- **Modes:**
  - **Local bridge mode:** Python launches a local HTTP server via `lobster/cli_internal/ink_launcher.py`
  - **Cloud-direct mode:** the binary connects directly to `https://app.omics-os.com/api/v1`
- **Session model:** session creation, hydration, resume, and state patches are shared conceptually with the web app

## Entry Points

| File | Responsibility |
|------|----------------|
| `src/cli.tsx` | Parse binary args, resolve config, launch chat app or init wizard |
| `src/App.tsx` | Shell layout, runtime wiring, backend readiness, empty state, transcript structure |
| `src/config.ts` | API URL resolution, local vs cloud mode, auth header construction |
| `src/hooks/useRuntime.ts` | `useDataStreamRuntime`, session resolution, hydration, state patch application |
| `src/utils/stateHandlers.ts` | Protocol-aware state patch handling with `_v` checks |
| `src/utils/hydration.ts` | Durable message hydration and `message_id` dedup |
| `src/commands/dispatcher.ts` | Native and bridged slash-command routing |
| `src/components/` | Transcript, prompt, footer, activity, tool/handoff renderers, intro |
| `lobster/cli_internal/ink_launcher.py` | Local bridge server, slash-command bridge, SSE/DataStream adaptation |

## Runtime Architecture

### 1. Config Resolution

`src/config.ts` determines:
- whether the app is running in local or cloud mode
- which API base URL to use
- whether auth is `Bearer`, `X-API-Key`, or none
- whether the session is a fresh launch or a resume path

Auth precedence:
1. `LOBSTER_TOKEN`
2. `--token`
3. stored credentials at `~/.config/omics-os/credentials.json` for cloud mode

### 2. Session Resolution and Hydration

`src/hooks/useRuntime.ts` is the runtime entry point:
- resolves or creates a session via `src/api/sessions.ts`
- constructs the stream endpoint
- attaches auth headers plus reconnect headers
- hands state patches to `processStatePatch(...)`
- hydrates existing durable messages from `GET /sessions/{id}/messages`
- resets thread state cleanly on `/clear`

Rules that must remain true:
- message hydration dedups by `message_id`
- unknown `aui-state` keys are ignored
- known keys with unsupported `_v` are ignored
- request bodies stay minimal; optional fields are not required

### 3. State Patches and UI State

The runtime consumes state patches for:
- `active_agent`
- `agent_status`
- `activity_events`
- `progress`
- `alerts`
- `token_usage`
- `session_title`
- `plots`
- `modalities`

Rapidly changing protocol state is intentionally pushed into external stores rather than broad top-level React state:
- `src/utils/appStateStore.ts`
- `src/utils/footerStateStore.ts`

This is important for render calmness. Avoid reintroducing global rerender churn through coarse React state subscriptions.

## Local Bridge Mode

`lobster/cli_internal/ink_launcher.py` launches the Ink binary and hosts a local HTTP bridge that exposes:
- `/health`
- `/bootstrap`
- `/sessions`
- `/sessions/{id}/chat/stream`
- `/sessions/{id}/commands/...`

The launcher adapts backend events into assistant-ui/DataStream-compatible chunks:
- text deltas
- reasoning deltas
- tool call lifecycle
- transient `data-*` state events for `onData`

Important bridge behaviors:
- slash commands can execute through backend POST handlers
- completed HTTP responses must be closed promptly or the single-threaded bridge can wedge later requests
- Python-owned stdout/stderr/logging must stay quarantined so background warnings do not leak into the TUI

## Cloud-Direct Mode — VALIDATED WORKING (2026-04-28)

The binary supports direct cloud connectivity, **live-tested end-to-end on 2026-04-28**:
- `--cloud` + `--api-url=https://app.omics-os.com/api/v1`
- `--token` (explicit override)
- stored credentials from `~/.config/omics-os/credentials.json` (OAuth or API key)

Confirmed working: session creation, SSE streaming from ECS Fargate, message send/receive, session resume via `--session-id <UUID>`, `/sessions` and `/cloud account` slash commands inside TUI, full message hydration on resume.

Known bug: `--session-id latest` passes literal "latest" to API instead of resolving to most recent UUID. Fix written in `src/api/sessions.ts:73`, awaiting binary rebuild.

In cloud-direct mode, the Ink app talks to the Omics-OS Cloud API directly and uses the same session/message/state model as the web app. CLI/web session continuity is architecturally real — same session UUIDs, same backend, same agents.

## Rendering Model

### Transcript Structure

The intended shell is text-first:
- one-line header
- transcript in normal flow
- inline prompt
- constant footer/status line

Do not regress toward box-heavy panel UI unless there is a very strong reason.

### Tool and Handoff Visibility

Tool calls and agent handoffs should read as part of the transcript, not as hidden peripheral state.

Relevant files:
- `src/components/ChainOfThought.tsx`
- `src/components/ActivityFeed.tsx`
- `src/components/StatusBar.tsx`
- `src/components/ToolRenderers/ToolRouter.tsx`
- `src/components/ToolRenderers/ToolCallRenderer.tsx`
- `src/components/ToolRenderers/HandoffRenderer.tsx`

### Empty State and Intro

`src/components/WelcomeAnimation.tsx` owns the hero animation and idle shell feel.

The target is closer to the Go/Charm experience:
- large ASCII hero
- restrained flicker/spark behavior
- breathing room above the prompt
- footer anchored visually at the bottom

## Slash Commands

Slash-command behavior is split between:
- **native client-side commands** for immediate UI actions
- **bridged backend commands** for filesystem/workspace/session actions

Relevant files:
- `src/commands/dispatcher.ts`
- `src/commands/subcommands.ts`
- `src/hooks/useSlashCommands.ts`
- `src/components/CommandOutput.tsx`

When modifying slash commands:
- keep command output out of the durable assistant transcript unless intentionally modeled there
- ensure command output is cleared appropriately on the next normal user turn
- validate both direct command execution and `chat -> command -> chat` transitions

## Validation Commands

```bash
cd lobster-tui-ink && bun run typecheck
cd lobster-tui-ink && bun test
cd lobster-tui-ink && bun run build
python3 -m py_compile lobster/cli_internal/ink_launcher.py
```

Useful live checks:

```bash
./.venv/bin/lobster chat --ui ink --debug
./.venv/bin/lobster chat --ui ink --debug --no-intro
./.venv/bin/lobster query "Reply with exactly QUERY_OK and nothing else."
./.venv/bin/lobster init
```

## Guardrails

1. Do not modify Go/Charm files just to make Ink look better.
2. If you must touch shared protocol behavior, keep changes additive and backward-compatible.
3. Re-run Go regression coverage after shared launcher/protocol changes.
4. Keep the shell calm: avoid unnecessary animation or broad rerender triggers.
5. Preserve session continuity behavior across fresh launch, resume, and hydration.

## Current Known Gaps

Active parity concerns (last reviewed 2026-04-28):
- streamed assistant text still redraws more roughly than the Go client in PTY capture
- startup latency is still higher and more variable than desired
- some structured outputs remain heavier than the Go/Charm transcript style
- long multi-agent runs can still expose completion/run-loop edge cases
- full-height Ink layout can still feel more like a screen-clearing window than the Go client's inline terminal model
- `--session-id latest` not resolved in cloud mode (fix written in `src/api/sessions.ts`, needs binary rebuild with `bun`)
- `data_status` (cold/warm/hot) modality state not consumed — merge blocker #11

**No longer gaps (confirmed working 2026-04-28):**
- Cloud-direct mode: session create, SSE stream, message send/receive — all working
- Session resume via `--session-id <UUID>` — working with full message hydration
- OAuth credential flow: login → stored credentials → auto-auth in Ink — working
- Cloud slash commands (`/sessions`, `/cloud account`) inside TUI — working

## Related Docs

- `/Users/tyo/Omics-OS/lobster/.claude/docs/ink-tui-architecture.md`
- `/Users/tyo/Omics-OS/lobster/.claude/docs/assistant-ui-cloud-sync.md`
- `/Users/tyo/Omics-OS/lobster-cloud/.planning/cross_surface_protocol.md`
- `../.planning/IMPLEMENTATION_STATE.md`

## Universal CLI / Data Accessing Layer (NEXUS Plan)

The Universal CLI initiative has been expanded into the **NEXUS Data Accessing Layer** and consolidated under lobster-cloud:

**Canonical location:** `/Users/tyo/Omics-OS/lobster-cloud/.planning/NEXUS_DATA_LAYER/`

This contains:
- `ROADMAP.md`, `STATE.md` — master plan and live tracker
- `PHASE_0_FOUNDATION.md` through `PHASE_5_INK_SLASH.md` — CLI phases (originated here, copied verbatim)
- `PHASE_6+` — Connector Framework phases (new, added by lobster-cloud ultrathink)
- `codex/` — GPT-5.5 validation reports
- `reports/` — 5 team research reports

**Why moved:** Kevin expanded the initiative to combine Universal CLI + Connector Framework (DataConnector ABC, OpenCures API, agent tools, SDK). The backend-heavy connector work lives in lobster-cloud, so the consolidated plan lives there too.

**Rule:** When executing CLI phases (P0-P5), reference the NEXUS_DATA_LAYER location. Local copies in `.planning/UNIVERSAL_CLI/` are historical — the canonical files are in lobster-cloud.
