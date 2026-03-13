# lobster-tui-ink — React Ink Terminal UI

Terminal interface for Lobster AI built with React Ink + assistant-ui. Shares the same DataStream protocol as the Omics-OS Cloud web app.

## Architecture

- **Framework:** React 19 + Ink 6 + @assistant-ui/react-ink + @assistant-ui/react-data-stream
- **Runtime:** Bun (compile to standalone binary)
- **Protocol:** ui-message-stream (SSE with JSON payloads)
- **Two modes:** Local (Python DataStream bridge) and Cloud (direct to app.omics-os.com)

## Commands

```bash
# Dev
bun install
bun run typecheck    # TypeScript validation
bun run dev          # Dev mode

# Build
bun run build        # Compile to dist/lobster-chat binary
```

## Key Files

| File | Purpose |
|------|---------|
| `src/cli.tsx` | Entry point (parseArgs, mode routing) |
| `src/App.tsx` | Main app shell (runtime, hooks, layout) |
| `src/config.ts` | API URL + auth configuration |
| `src/hooks/useRuntime.ts` | DataStream runtime + session + state |
| `src/hooks/useSlashCommands.ts` | Slash command interception |
| `src/hooks/useHistory.ts` | Command history with persistence |
| `src/hooks/useCancelHandler.ts` | Two-phase Ctrl+C |
| `src/commands/dispatcher.ts` | Native + bridged slash commands |
| `src/api/` | REST client, sessions, feature flags, templates |
| `src/components/` | UI components (Thread, Composer, StatusBar, etc.) |
| `src/components/HITL/` | 6 human-in-the-loop components |
| `src/components/ToolRenderers/` | Specialized tool call renderers |
| `src/wizard/` | Init wizard (WizardManifest consumer) |
| `src/utils/` | Hydration, state handlers, browser open |
| `src/query.ts` | Non-interactive query mode |

## Protocol Compliance

- Schema versioning (§1.3): `stateHandlers.ts` checks `_v` field
- Message dedup (§3): `hydration.ts` deduplicates by `message_id`
- Feature flags (§4.1): `featureFlags.ts` caches flags for session
- Prompt templates (§4.2): `templates.ts` fetches on new session
