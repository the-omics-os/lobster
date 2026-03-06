# lobster-ai-tui

Platform-specific Go TUI binary for [Lobster AI](https://lobsterbio.com).

This package provides the `lobster-tui` binary — a native terminal UI built with the
[Charm](https://charm.sh) stack (BubbleTea, Bubbles, Lipgloss, Glamour). It is
automatically installed as a dependency of `lobster-ai` on supported platforms.

## Supported Platforms

- Linux x86_64 (amd64)
- Linux ARM64 (aarch64)
- macOS ARM64 (Apple Silicon)
- macOS x86_64 (Intel)

## Usage

This package is not intended to be used directly. Install `lobster-ai` and run:

```bash
lobster chat          # Uses Go TUI by default
lobster chat --classic  # Fall back to Python Rich/Textual UI
```

## Binary Discovery

The `lobster-ai` CLI finds the binary via `lobster_ai_tui.get_binary_path()`:

```python
from lobster_ai_tui import get_binary_path
binary = get_binary_path()  # Returns Path to lobster-tui binary
```
