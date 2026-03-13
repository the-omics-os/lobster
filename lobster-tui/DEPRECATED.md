# Go TUI — Deprecated

The Go Charm TUI (`lobster-tui/`) is deprecated in favor of the React Ink TUI (`lobster-tui-ink/`).

## Migration

The Ink TUI is now the default when running `lobster chat`. No action required for end users.

### For developers

- **New TUI source:** `lobster-tui-ink/` (TypeScript, React Ink, Bun)
- **Protocol:** ui-message-stream (shared with Cloud web app)
- **Binary:** `lobster-chat` (compiled via `bun build --compile`)

### Timeline

- **Current:** `--ui go` shows deprecation warning but still works
- **Next minor:** Go binary no longer included in default distribution
- **Future:** `lobster-tui/` directory removed

### Why

The Go TUI required a custom 33-type JSON-lines protocol and a Python bridge. The Ink TUI uses the same DataStream protocol as the Cloud web app, eliminating protocol duplication and enabling feature parity across all surfaces.
