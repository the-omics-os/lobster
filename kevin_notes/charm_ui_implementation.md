# Charm TUI Frontend — Implementation Spec & Plan

> **Goal:** Replace Lobster AI's Python Rich/Textual terminal interface with a compiled Go binary
> using the Charm ecosystem (bubbletea, huh, lipgloss, glamour), communicating with the Python
> LangGraph backend via a JSON Lines protocol over stdio.
>
> **Author:** ultrathink (CTO) + Kevin Yar
> **Date:** 2026-03-04
> **Status:** Design approved, implementation pending

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Why This Matters](#2-why-this-matters)
3. [Reference Implementation: AgentUI](#3-reference-implementation-agentui)
4. [Brutalist Audit Findings](#4-brutalist-audit-findings)
5. [Architecture Design](#5-architecture-design)
6. [Protocol Specification](#6-protocol-specification)
7. [Lobster-Specific Extensions](#7-lobster-specific-extensions)
8. [Distribution Strategy](#8-distribution-strategy)
9. [Phase Plan](#9-phase-plan)
10. [Risk Mitigations](#10-risk-mitigations)
11. [Go Dependency Stack](#11-go-dependency-stack)
12. [Key Files & Resources](#12-key-files--resources)
13. [Decision Log](#13-decision-log)

---

## 1. Executive Summary

Lobster AI currently uses Python Rich (CLI) + Textual (optional TUI) for its terminal interface.
This works but has structural limitations:

- **Rendering performance:** O(n²) markdown re-renders on streaming, widget memory leaks
- **Interaction model:** Numbered string choices (`"1"`, `"2"`), no arrow-key navigation
- **Visual quality:** Below what modern CLI tools (Charm Crush, Claude Code, Gemini CLI) deliver
- **Reliability:** 68 silent `except: pass` paths in UI code, broken cancel, state corruption hacks

**The plan:** Build `lobster-tui`, a compiled Go binary using the Charm ecosystem, that handles
ALL terminal rendering. Python becomes a headless computation engine. They communicate via
JSON Lines over stdio — the same pattern used by LSP, MCP, and Neovim.

**Key constraint:** Rich CLI fallback is ALWAYS maintained. Users without the Go binary get a
functional (if less polished) experience. The Go TUI is additive, never required.

---

## 2. Why This Matters

### 2.1 Market Position

No Python AI agent platform has a compiled TUI frontend:

| Tool | Stars | Stack | TUI Quality |
|------|-------|-------|-------------|
| Charm Crush | 20,863 | Pure Go | Gold standard |
| Claude Code | 73,686 | TypeScript/Ink | Excellent |
| Gemini CLI | 96,436 | TypeScript | Excellent |
| Codex CLI | 63,009 | Rust | Excellent |
| aider | 41,413 | Pure Python | Basic readline |
| open-interpreter | 62,518 | Pure Python | Basic streaming |
| **Lobster AI** | — | **Python → Go (planned)** | **First hybrid** |

Every polished AI TUI uses a compiled language. Every Python AI tool has a basic terminal.
Lobster would be the first to bridge the gap.

### 2.2 Current Pain Points (Evidence-Based)

From brutalist code audit (Codex agent, line-level evidence):

| Bug | File | Impact |
|-----|------|--------|
| **Ctrl+C cancel broken** | `analysis_screen.py:447` — worker never assigned to `current_worker` | Users can't cancel long queries |
| **68 silent exception swallows** | `textual_callback.py:157`, `analysis_screen.py:810`, etc. | "Looks fine but wrong" states |
| **O(n²) markdown streaming** | `chat_message.py:41` — full content re-render per chunk | UI lag on long responses |
| **Widget memory leak** | `activity_log.py:103` — trims list but not mounted widgets | Long sessions degrade |
| **Workspace rescan on hot path** | `analysis_screen.py:873` — recursive size calc after every query | Stalls on large workspaces |
| **State corruption workaround** | `textual_callback.py:440` — heuristic "must be supervisor" fallback | Wrong agent attribution |

These are not "5 lines of fixes" — they're symptoms of architectural stress in the Python
rendering layer. A compiled frontend with a clean protocol boundary eliminates this class of bugs.

### 2.3 Strategic Value

- **Fundability:** Investors see professional UX, not a homebrew Python TUI
- **SSH access:** Charm's `wish` library enables `ssh app.omics-os.com` → live TUI (cloud differentiator)
- **Theming:** JSON-based themes, 8+ built-in (Dracula, Catppuccin, Nord...) + custom Lobster theme
- **Accessibility:** Charm's `huh` has built-in accessibility mode for screen readers
- **Speed:** Go renders at native speed, no GIL, no Python overhead

---

## 3. Reference Implementation: AgentUI

**Repository:** `~/GITHUB/agentui/` (cloned from `github.com/flight505/agentui`)

AgentUI is the only project that implements the exact pattern we need: Go bubbletea TUI
communicating with a Python backend via JSON Lines over stdio.

### 3.1 What to Copy

| Component | AgentUI Location | Lines | Copy Strategy |
|-----------|-----------------|-------|---------------|
| Protocol spec (message types) | `src/agentui/protocol.py` | 311 | Fork + extend for Lobster |
| Go protocol handler | `internal/protocol/handler.go` | 191 | Copy verbatim |
| Go message types | `internal/protocol/types.go` | 265 | Fork + extend |
| Python TUIBridge | `src/agentui/bridge/tui_bridge.py` | 509 | Fork, remove AgentCore dep |
| Python CLIBridge fallback | `src/agentui/bridge/cli_bridge.py` | 281 | Adapt to Rich/questionary |
| Go bubbletea model | `internal/app/app.go` | 1008 | Rewrite for Lobster layout |
| Theme system | `internal/theme/` | ~700 | Copy + add Lobster theme |
| View components | `internal/ui/views/views.go` | 656 | Copy + extend |
| Form components | `internal/ui/components/forms.go` | 707 | Replace with `huh` |
| Spring animations | `internal/ui/animations/spring.go` | 237 | Copy verbatim |

### 3.2 What to Skip

| Component | Reason |
|-----------|--------|
| `src/agentui/core/agent.py` | We have LangGraph — don't need their agent loop |
| `src/agentui/providers/` | We have our own provider system |
| `src/agentui/app.py` | Lobster has its own CLI entry points |
| `src/agentui/component_selector.py` | Generative UI — not needed for init/chat |
| `src/agentui/skills/` | We have our own skill system |

### 3.3 What to Improve Over AgentUI

| Improvement | Reason |
|-------------|--------|
| Use `huh` for forms instead of custom components | Battle-tested, accessibility mode, 5 themes |
| Target bubbletea v2 (not v1) | Crush uses v2, it's the future |
| Add protocol versioning/handshake | Brutalist found no version negotiation = drift risk |
| Redirect Python stderr properly | Brutalist found stdio corruption risk from C extensions |
| Add message history cap in Go viewport | AgentUI has unbounded message growth too |

---

## 4. Brutalist Audit Findings

Two AI critics (GPT-5.3-Codex, Gemini-3.1-Pro) independently reviewed this plan.
Their feedback is integrated throughout, but the critical findings are:

### 4.1 Risks That Must Be Mitigated

| Risk | Severity | Source | Mitigation |
|------|----------|--------|------------|
| **stdio corruption from C extensions** | CRITICAL | Gemini | Redirect Python subprocess to use dedicated fd pair (fd 3/4), NOT stdout. stderr → log file. See §10.1 |
| **Protocol drift (22 types × 2 languages)** | CRITICAL | Both | Protocol version in handshake. Codegen from shared schema. See §10.2 |
| **Solo founder bandwidth** | HIGH | Gemini | Phased rollout. Phase 0 is one-shot subprocess (no IPC bridge). See §9 |
| **HPC/air-gapped binary distribution** | HIGH | Gemini | Platform wheels + offline install path. See §8 |
| **`init` doesn't prove streaming** | MEDIUM | Gemini | Phase 0 is intentionally simple. Phase 1 proves streaming. See §9 |
| **Zombie Go/Python processes** | HIGH | Gemini | PID tracking, signal forwarding, process group kill. See §10.3 |

### 4.2 Current Python TUI Bugs to Fix Regardless

These should be fixed in parallel with Go TUI development:

1. `analysis_screen.py:447` — assign `current_worker` after `run_worker()`
2. `textual_callback.py` — replace `except: pass` with proper logging (68 occurrences)
3. `chat_message.py:41` — debounce markdown re-render (throttle to every 100ms)
4. `activity_log.py:103` — prune mounted widgets when trimming event list
5. `analysis_screen.py:873` — move workspace rescan to background worker

---

## 5. Architecture Design

### 5.1 High-Level Architecture

```
                        ┌─────────────────────────────────────────┐
                        │           lobster-tui (Go binary)       │
                        │                                         │
                        │  ┌─────────┐  ┌──────────┐  ┌────────┐│
                        │  │ huh     │  │bubbletea │  │lipgloss││
                        │  │ (forms) │  │(viewport)│  │(styles) ││
                        │  └────┬────┘  └────┬─────┘  └────────┘│
                        │       │            │                    │
                        │  ┌────▼────────────▼─────────────────┐ │
                        │  │     Protocol Handler (Go)          │ │
                        │  │     readLoop() / writeLoop()       │ │
                        │  └────────────────┬──────────────────┘ │
                        └───────────────────┼────────────────────┘
                                            │
                               JSON Lines over fd 3/4
                              (NOT stdout — avoids C ext pollution)
                                            │
                        ┌───────────────────┼────────────────────┐
                        │                   │                     │
                        │  ┌────────────────▼──────────────────┐ │
                        │  │     LobsterTUIBridge (Python)      │ │
                        │  │     _read_loop() / _write_loop()   │ │
                        │  │     _pending_requests{}             │ │
                        │  └────────────────┬──────────────────┘ │
                        │                   │                     │
                        │  ┌────────────────▼──────────────────┐ │
                        │  │  TextualCallbackHandler (adapted)  │ │
                        │  │  → protocol messages instead of    │ │
                        │  │    direct widget updates           │ │
                        │  └────────────────┬──────────────────┘ │
                        │                   │                     │
                        │  ┌────────────────▼──────────────────┐ │
                        │  │  LangGraph Agent Execution         │ │
                        │  │  22 agents across 10 packages      │ │
                        │  │  DataManagerV2, Provenance, etc.   │ │
                        │  └──────────────────────────────────┘ │
                        │           Python Process               │
                        └────────────────────────────────────────┘
```

### 5.2 Process Lifecycle

**Phase 0 (init wizard) — One-shot subprocess:**
```
Python: lobster init
  → find/download lobster-tui binary
  → subprocess.run([lobster-tui, "init", "--theme", "lobster-dark"],
                   capture_output=True)
  → parse JSON output
  → apply config to .env, providers.json, agents.json
  → done (Go process exits)
```

**Phase 1+ (chat) — Long-lived IPC:**
```
Python: lobster chat
  → spawn lobster-tui as subprocess with fd 3/4 pipes
  → LobsterTUIBridge manages async read/write
  → LangGraph events → protocol messages → Go renders
  → Go user input → protocol messages → Python processes
  → Ctrl+C → graceful shutdown (signal forwarding)
```

### 5.3 Fallback Behavior

```python
# lobster/ui/bridge/__init__.py
def create_ui_bridge(config: TUIConfig) -> BaseBridge:
    """Try Go TUI first, fall back to Python CLI."""
    binary = _find_tui_binary()
    if binary:
        return GoTUIBridge(binary, config)

    # Fallback: questionary + Rich (Python-native)
    return PythonCLIBridge(config)
```

The fallback uses `questionary` for arrow-key selects + `Rich` for styled output.
This is already better than current `Prompt.ask(choices=["1","2"])`.

---

## 6. Protocol Specification

### 6.1 Wire Format

JSON Lines over dedicated file descriptors (NOT stdout/stdin — see §10.1):

```
Python → Go:  fd 3 (write) → fd 3 (read)
Go → Python:  fd 4 (write) → fd 4 (read)
stdout:       reserved for Python C extension noise (redirected to /dev/null or log)
stderr:       reserved for debug logging (both processes)
```

Each message is a single JSON object terminated by `\n`:
```json
{"type":"form","id":"uuid-123","version":1,"payload":{"title":"Provider","fields":[...]}}
```

### 6.2 Message Envelope

```
{
  "type":    string,          // MessageType enum
  "id":      string | null,   // UUID for request/response correlation
  "version": int,             // Protocol version (starts at 1)
  "payload": object | null    // Type-specific data
}
```

**Improvement over AgentUI:** Added `version` field for protocol evolution.

### 6.3 Message Types

#### Python → Go (Render Commands)

| Type | Payload | Blocking | Use Case |
|------|---------|----------|----------|
| `text` | `{content, done: bool}` | No | Streaming LLM responses |
| `markdown` | `{content, title?}` | No | Formatted content blocks |
| `code` | `{code, language, title?, lineNumbers?}` | No | Code display, notebook preview |
| `table` | `{title?, columns[], rows[][], footer?}` | No | Agent listing, modality info |
| `form` | `{title, description?, fields[], submitLabel?, cancelLabel?}` | Yes | Init wizard steps, config |
| `confirm` | `{message, destructive?, confirmLabel?, cancelLabel?}` | Yes | Dangerous operations |
| `select` | `{label, options[], default?}` | Yes | Provider selection, profile |
| `progress` | `{message, percent?, steps[]?}` | No | Package install, SSL test |
| `alert` | `{message, title?, severity}` | No | Info/success/warning/error |
| `spinner` | `{message}` | No | Loading states |
| `status` | `{message, tokens?, activeAgent?}` | No | Status bar updates |
| `clear` | `{scope: "all"|"progress"|"messages"}` | No | Reset UI sections |
| `done` | `{summary?}` | No | Agent finished |
| `agent_transition` | `{from, to, reason?}` | No | **Lobster-specific:** agent handoff viz |
| `modality_loaded` | `{name, type, shape, path}` | No | **Lobster-specific:** data loaded |
| `tool_execution` | `{tool, agent, status, result?}` | No | **Lobster-specific:** tool call viz |

#### Go → Python (User Events)

| Type | Payload | Use Case |
|------|---------|----------|
| `input` | `{content}` | User typed a message |
| `form_response` | `{values: {field: value}}` | Form submitted (matched by `id`) |
| `confirm_response` | `{confirmed: bool}` | Confirmation answered |
| `select_response` | `{value: string}` | Selection made |
| `cancel` | `{}` | User cancelled current operation |
| `quit` | `{}` | User wants to exit (Ctrl+C) |
| `resize` | `{width, height}` | Terminal resized |
| `slash_command` | `{command, args?}` | **Lobster-specific:** /data, /help, etc. |

### 6.4 Form Field Types

```json
{
  "name": "api_key",
  "type": "password",       // text | password | select | checkbox | number
  "label": "Claude API Key",
  "required": true,
  "placeholder": "sk-ant-...",
  "description": "Get your key from console.anthropic.com",
  "options": ["opt1", "opt2"],  // only for type=select
  "default": "opt1"
}
```

### 6.5 Request/Response Correlation

Blocking messages (form, confirm, select) use UUIDs:

```
Python sends:  {"type":"form", "id":"abc-123", "payload":{...}}
Go renders form, user fills it out
Go sends:      {"type":"form_response", "id":"abc-123", "payload":{"values":{...}}}
Python matches id → unblocks waiting asyncio.Future
```

### 6.6 Protocol Handshake

On connection, Go sends a handshake message:

```json
{"type":"handshake","payload":{"protocol_version":1,"tui_version":"0.1.0","features":["form","table","code","streaming"]}}
```

Python validates version compatibility before proceeding.

---

## 7. Lobster-Specific Extensions

Beyond AgentUI's base protocol, Lobster needs:

### 7.1 Agent Transition Visualization

```json
{"type":"agent_transition","payload":{
  "from":"supervisor",
  "to":"transcriptomics_expert",
  "reason":"User asked about scRNA-seq QC",
  "active_agents":["supervisor","transcriptomics_expert"]
}}
```

Go renders: animated agent panel showing active/idle agents with handoff arrows.

### 7.2 Modality Data Panel

```json
{"type":"modality_loaded","payload":{
  "name":"pbmc3k",
  "type":"single_cell_transcriptomics",
  "shape":[2700, 32738],
  "path":"workspace/pbmc3k.h5ad"
}}
```

Go renders: data panel showing loaded datasets with type icons.

### 7.3 Tool Execution Feed

```json
{"type":"tool_execution","payload":{
  "tool":"run_scanpy_qc",
  "agent":"transcriptomics_expert",
  "status":"running",
  "category":"ANALYZE"
}}
```

Go renders: activity log with tool name, agent, AQUADIF category badge.

### 7.4 Slash Command Passthrough

Go intercepts `/help`, `/data`, `/files`, etc. and sends them as `slash_command` events.
Python handles them via existing `OutputAdapter` pattern, sends results back as
`table`/`markdown`/`code` messages.

---

## 8. Distribution Strategy

### 8.1 Primary: Platform-Specific Python Wheels

Following the ruff/uv model — ship Go binaries inside Python wheels:

```
lobster-ai-tui-0.1.0-py3-none-macosx_arm64.whl       (~10MB)
lobster-ai-tui-0.1.0-py3-none-macosx_x86_64.whl
lobster-ai-tui-0.1.0-py3-none-manylinux2014_x86_64.whl
lobster-ai-tui-0.1.0-py3-none-win_amd64.whl
```

Install: `uv tool install 'lobster-ai[tui-go]'`

The wheel contains a single file: `lobster_ai_tui/bin/lobster-tui`
Python finds it via: `importlib.resources.files("lobster_ai_tui") / "bin" / "lobster-tui"`

### 8.2 Secondary: Lazy Download

If platform wheel is not installed, download on first use:

```python
def _ensure_tui_binary() -> Path | None:
    # 1. Check platform wheel
    try:
        import lobster_ai_tui
        return lobster_ai_tui.get_binary_path()
    except ImportError:
        pass

    # 2. Check cache
    cache = Path.home() / ".cache" / "lobster" / "bin" / "lobster-tui"
    if cache.exists() and _verify_checksum(cache):
        return cache

    # 3. Download (with user consent)
    if not _has_internet():
        return None  # Fall back to Python CLI

    platform = _detect_platform()  # darwin-arm64, linux-amd64, etc.
    url = f"https://github.com/the-omics-os/lobster/releases/download/tui-v{VERSION}/lobster-tui-{platform}"
    _download_with_progress(url, cache)
    cache.chmod(0o755)
    return cache
```

### 8.3 Tertiary: Manual Install

```bash
# Homebrew (macOS/Linux)
brew install the-omics-os/tap/lobster-tui

# Go install (developers)
go install github.com/the-omics-os/lobster-tui@latest

# Direct binary download
curl -fsSL https://install.lobsterbio.com/tui | bash
```

### 8.4 Offline/HPC Environments

For air-gapped clusters (critical for bioinformatics users):

```bash
# On a machine with internet:
lobster tui download --platform linux-amd64 --output lobster-tui

# Transfer to HPC:
scp lobster-tui user@cluster:~/.cache/lobster/bin/lobster-tui

# Or: install the platform wheel offline:
pip download lobster-ai-tui --platform manylinux2014_x86_64
scp lobster_ai_tui-*.whl user@cluster:
pip install lobster_ai_tui-*.whl
```

### 8.5 CI/CD Pipeline for Go Binary

```yaml
# .github/workflows/build-tui.yml
name: Build TUI Binary
on:
  push:
    tags: ['tui-v*']

jobs:
  build:
    strategy:
      matrix:
        include:
          - os: macos-latest
            goos: darwin
            goarch: arm64
          - os: macos-13
            goos: darwin
            goarch: amd64
          - os: ubuntu-latest
            goos: linux
            goarch: amd64
          - os: windows-latest
            goos: windows
            goarch: amd64
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/setup-go@v5
        with:
          go-version: '1.22'
      - run: |
          GOOS=${{ matrix.goos }} GOARCH=${{ matrix.goarch }} \
          go build -ldflags="-s -w" -o lobster-tui-${{ matrix.goos }}-${{ matrix.goarch }} \
          ./cmd/lobster-tui/
      - uses: actions/upload-artifact@v4
        # ... upload to GitHub Release

  publish-wheels:
    needs: build
    # Build platform-specific Python wheels containing the binary
    # Uses a custom script similar to ruff's wheel builder
```

---

## 9. Phase Plan

### Phase 0: Init Wizard (One-Shot Subprocess) ✅ COMPLETE (2026-03-04)
**Goal:** Prove Go build pipeline, distribution, and UX. Zero IPC complexity.
**Duration:** 2-3 weeks → completed in 1 session
**Risk:** Low

**What:**
- Go binary with `huh` forms for the `lobster init` wizard
- One-shot subprocess: Python calls `lobster-tui init`, Go renders forms, outputs JSON to stdout
- Python reads JSON output, applies config (writes .env, providers.json, agents.json)
- NO long-lived IPC bridge — subprocess runs and exits
- Fallback: Python `questionary` + Rich if Go binary unavailable

**Go binary scope:**
```
lobster-tui init
  Step 1: Agent package selection (huh MultiSelect)
  Step 2: Provider selection (huh Select: Anthropic/Bedrock/Ollama/Gemini/Azure/OpenAI/OpenRouter)
  Step 3: API key entry (huh Input with password masking)
  Step 4: Profile selection (huh Select: dev/production/performance/max)
  Step 5: Optional keys — NCBI, Cloud (huh Confirm + Input)
  Output: JSON to stdout with all selections
```

**Python scope:**
```python
def _init_with_tui(binary_path):
    result = subprocess.run(
        [binary_path, "init", "--theme", "lobster-dark"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise TUIError(result.stderr)
    config = json.loads(result.stdout)
    _apply_init_config(config)  # Existing logic: write .env, install packages, SSL test
```

**Deliverables:**
- [x] Go module at `lobster-tui/` in the monorepo
- [x] `cmd/lobster-tui/main.go` — entry point with `init` subcommand
- [x] `internal/initwizard/wizard.go` — huh form wizard (renamed from `init/` — Go reserved word)
- [x] `internal/theme/lobster.go` — Lobster theme (#e45c47 primary)
- [ ] CI workflow for cross-compilation (4 platforms) — deferred to Phase 3
- [x] Python `find_tui_binary()` in `ui/bridge/binary_finder.py`
- [x] Python `apply_tui_init_result()` in `ui/bridge/init_adapter.py`
- [x] Python `questionary` fallback in `ui/bridge/questionary_fallback.py`
- [x] `--ui` flag (auto/go/classic) on `lobster init`
- [x] Manual test: `lobster-tui init` on macOS with Lobster Dark theme — verified by Kevin

**Success criteria:** A bioinformatician runs `lobster init` and sees a beautiful, arrow-key
navigable wizard with the Lobster orange theme. If Go binary is missing, they get the same
flow via questionary with slightly less visual polish.

---

### Phase 1: Protocol Foundation & Streaming Chat ✅ COMPLETE (2026-03-04)
**Goal:** Prove the full IPC bridge with real-time LLM streaming.
**Duration:** 3-4 weeks → completed in 1 session
**Risk:** Medium (this is where the hard IPC problems live)
**Prerequisite:** Phase 0 validated

**What:**
- Implement the full JSON Lines protocol (§6)
- Build `LobsterTUIBridge` (Python) — fork of AgentUI's TUIBridge
- Build Go protocol handler — fork of AgentUI's handler.go
- Implement `lobster-tui chat` — bubbletea viewport + input + status bar
- Wire LangGraph callbacks → protocol messages → Go rendering
- Handle streaming text (token-by-token), code blocks, tables
- Handle Ctrl+C graceful shutdown with signal forwarding

**Critical IPC decisions (from brutalist feedback):**
- Use fd 3/4 for protocol, NOT stdout (§10.1)
- Implement protocol handshake with version (§6.6)
- 30s timeout on blocking requests with cleanup
- PID tracking for zombie process prevention

**Deliverables:**
- [x] `lobster/cli_internal/go_tui_launcher.py` — lightweight IPC bridge (stdlib only, <200ms startup)
- [x] `lobster/ui/callbacks/protocol_callback.py` — `ProtocolCallbackHandler` (adapted from Textual)
- [x] `lobster-tui/internal/protocol/` — Go-side handler + types (38 message types defined)
- [x] `lobster-tui/internal/chat/` — bubbletea chat model (viewport + input + status + spinner)
- [x] `lobster-tui/internal/theme/` — theme system with Lobster Dark/Light themes
- [x] Heartbeat monitoring (3s interval, 15s timeout detection)
- [x] Signal forwarding: Ctrl+C → process group kill, clean exit
- [ ] Integration test: stream a mock LLM response through the full pipeline
- [ ] Stress test: 10K token response without UI degradation

**Success criteria:** ✅ `lobster chat --ui go` streams LLM responses smoothly, shows
active agent, handles tool calls, and Ctrl+C cleanly stops both processes.

**Architecture notes:**
- IPC uses inherited fd pairs (not stdin/stdout) — avoids C extension stdio corruption
- Python `_LightBridge` inlined in `go_tui_launcher.py` — avoids importing Rich/LangChain at module level
- Stderr suppressed during heavy imports, restored after init
- `*strings.Builder` (pointer) in Model to avoid BubbleTea value-copy panic

---

### Phase 2: Rich Rendering & UX Polish ✅ COMPLETE (2026-03-04)
**Goal:** Polished UX — glamour markdown, metadata display, loading tips.
**Duration:** 3-4 weeks → completed in 1 session
**Prerequisite:** Phase 1 validated

**What (original plan + actual deliverables merged):**
- ~~Agent transition panel (show active/idle agents with handoff animations)~~ → deferred to Phase 3
- ~~Modality data panel (loaded datasets with type/shape info)~~ → deferred to Phase 3
- Tool execution activity log (ring buffer, 5 lines) ✅ (Phase 1)
- Slash command passthrough (/data, /help, /files, /pipeline, /status) ✅
- Token usage display in status bar ✅
- Session continuity (`--session-id`) ✅
- **Glamour markdown rendering** for assistant messages ✅
- **TypeMarkdown/TypeCode protocol handlers** (forward-compatible) ✅
- **Loading tips** during init spinner (8 tips, 5s rotation) ✅
- **Session ID in header** (right-aligned, truncated >32 chars) ✅
- **Rich status** after queries (tokens, cost, duration) ✅

**Bug fixes (from CLI_BUG_REPORT.md, 2026-03-04):**
- [x] Bug 1: Slash commands not recognized — Go strips `/`, Python expects it. Fixed: prepend `/` in `_handle_slash_command`
- [x] Bug 2: Chat unusable after suspend — missing `tea.ResumeMsg` handler broke protocol loop. Fixed: added handler in Update
- [x] Bug 3: Session ID truncated — `sid[len-20:]` cut `session_` prefix. Fixed: only truncate >32 chars with preserved prefix
- [x] Bug 4: Non-TTY hang — no stdin check before Go TUI launch. Fixed: `os.isatty(0)` guard + error for `--ui go`
- [x] Bug 5: Invalid `--ui` value accepted — no validation. Fixed: early validation against `{auto, go, classic}`
- [x] Bug 6: `strings.Builder` panic — copied by value in BubbleTea Update. Fixed: changed to `*strings.Builder`

**Deliverables:**
- [x] Glamour dependency (`github.com/charmbracelet/glamour v0.10.0`) with auto dark/light detection
- [x] Cached markdown renderer (lazy init, recreated on width change)
- [x] `renderMessage()` renders assistant content through glamour (headers, bold, code fences, lists, tables)
- [x] `TypeMarkdown` and `TypeCode` handlers in `handleProtocol()`
- [x] `_format_usage()` sends `"Tokens: N | Cost: $X.XXXX · Duration: Ys"` after queries
- [x] Session ID sent from Python after `init_client()`, displayed in header
- [x] 8 rotating loading tips during init spinner
- [x] `--ui` validation, non-TTY detection, slash command `/` prefix fix
- [ ] ~~`lobster-tui/internal/widgets/agents_panel.go`~~ → deferred to Phase 3
- [ ] ~~`lobster-tui/internal/widgets/modality_panel.go`~~ → deferred to Phase 3

**Charm ecosystem research (2026-03-04):**
Explored 5 cloned Charm repos (`~/GITHUB/{glamour,bubbles,crush,gum,log}`) via sub-agents:
- **glamour**: `NewTermRenderer(WithAutoStyle(), WithWordWrap(w))`, reuse renderer, `RenderBytes` for perf
- **crush**: `Styles` struct pattern, `MarkdownRenderer()` helper, `anim` package, `Scrollbar()`, gradient utilities, `charmtone` palette, icon constants
- **bubbles v2**: viewport (SoftWrap, LeftGutterFunc, StyleLineFunc), spinner (ID-tagged), textarea (chat input), table, help bar, progress
- **gum**: glamour patterns (`WithAutoStyle()`), spinner + background command, viewport height budgeting, input auto-width
- **log**: `Styles` struct pattern, TTY-aware renderer caching, structured key=value with multiline indent

---

### Phase 3: Protocol Handlers & Native Interactions ✅ COMPLETE (2026-03-04)
**Goal:** Complete all 20 protocol handlers, native forms/confirm/select, progress bar.
**Duration:** 1 session
**Prerequisite:** Phase 2 validated

**What (delivered):**
- [x] 20 protocol handlers in `handleProtocol()` (all message types from spec)
- [x] `TypeForm` handler — suspends TUI via `tea.Exec` with `huh` forms (`forms.go`)
- [x] `TypeConfirm` handler — inline yes/no dialog with y/n/Enter keys
- [x] `TypeSelect` handler — inline arrow-navigated selector with Up/Down/Enter
- [x] `TypeProgress` handler — visual progress bar with label and percentage
- [x] `TypeTable` handler — markdown table rendering from `headers[]` + `rows[][]`
- [x] `TypeModalityLoaded` handler — system message with data shape info
- [x] `TypeClear` handler — target-specific clearing (output/status/all)
- [x] Agent transition displayed as styled system messages
- [x] Tool execution ring buffer (5 entries) with in-place finish/error updates

**Files:**
- `lobster-tui/internal/chat/model.go` — all protocol cases
- `lobster-tui/internal/chat/forms.go` — NEW: huh form suspension
- `lobster-tui/internal/chat/views.go` — rendering functions for all UI elements

---

### Phase 4: Native Slash Commands, Event Wiring & Autocomplete ✅ COMPLETE (2026-03-04)
**Goal:** Eliminate TUI suspension for slash commands. Go-native `/help`, `/clear`, `/exit`, `/data`.
**Duration:** 1 session
**Prerequisite:** Phase 3 validated

**What (delivered):**

**Go-native commands (zero Python):**
- [x] `/help` — renders markdown command table via glamour
- [x] `/clear` — clears messages, stream buffer, tool feed, modality cache
- [x] `/exit`, `/quit` — clean shutdown with quit signal
- [x] `/data` — renders from cached `ModalityInfo` (populated by `modality_loaded` events)
- [x] `/dashboard` — shows "not available in Go TUI" alert

**ProtocolOutputAdapter (Python → Go bridge for slash commands):**
- [x] `ProtocolOutputAdapter` — 4th `OutputAdapter` subclass (~50 lines)
- [x] `_handle_slash_command` rewritten — no more suspend/resume/Press Enter
- [x] `_execute_command` accepts `output` parameter — all 13 command branches adapted
- [x] `/help` and `/tokens` fully converted from Rich Tables to `output.print_table()`
- [x] Spinner shown during heavy imports (~1-2s first call)

**Event wiring:**
- [x] `DataManagerV2.on_modality_loaded` callback — fires at end of `load_modality()`
- [x] Wired in `go_tui_launcher.py` to emit `modality_loaded` protocol messages
- [x] `DownloadOrchestrator.progress_callback` — emits before/after download

**Autocomplete:**
- [x] `completions.go` — NEW: context-aware autocomplete using `textinput.SetSuggestions()`
- [x] 23 top-level commands, 5 subcommand groups, 3 deep subcommand levels
- [x] Modality name completion for `/describe <name>` from `m.modalities` cache
- [x] Provider name completion for `/config provider <name>`
- [x] Viewport scrolling gated when suggestions active (Up/Down/Tab for suggestions)
- [x] Completion preview styled with theme's `Dimmed` style

**Files:**
- `lobster-tui/internal/chat/model.go` — slash switch, ModalityInfo cache, autocomplete wiring
- `lobster-tui/internal/chat/completions.go` — NEW: suggestion builder
- `lobster/cli_internal/commands/output_adapter.py` — `ProtocolOutputAdapter`
- `lobster/cli_internal/go_tui_launcher.py` — rewritten `_handle_slash_command`, modality wiring
- `lobster/cli_internal/commands/heavy/slash_commands.py` — `output` param, converted commands
- `lobster/core/runtime/data_manager.py` — `on_modality_loaded` callback
- `lobster/tools/download_orchestrator.py` — `progress_callback`

---

### Phase 5: BioCharm Components
**Goal:** Domain-specific bioinformatics TUI components invokable by LangGraph agents.
**Duration:** 3-4 weeks
**Prerequisite:** Phase 4 validated

**Architecture:** Fully designed in `.claude/docs/charm-tui-architecture.md`. Uses generic 3-message protocol (`component_render`, `component_close`, `component_response`) — new components require Go-side changes only, no protocol modifications.

**Components:**

| Component | Mode | Interactive | Description |
|-----------|------|-------------|-------------|
| `dna_animation` | inline | No | ASCII double helix loading animation |
| `qc_dashboard` | inline | No (streaming) | Multi-metric quality panel |
| `cell_type_selector` | overlay | Yes | Cluster annotation with marker genes |
| `threshold_slider` | overlay | Yes (streaming) | p-value/FC cutoff with live gene count |
| `ontology_browser` | overlay | Yes (lazy-load) | GO/Reactome tree navigation |
| `sequence_input` | fullscreen | Yes | DNA/RNA/protein entry with validation |

**Deliverables:**
- [ ] `internal/biocomp/registry.go` — `BioComponent` interface + factory map
- [ ] Component rendering in `model.go` — `handleComponentRender()`, `activeComponents` map
- [ ] Agent transition cleanup (close overlay/fullscreen on handoff)
- [ ] Error boundaries (bad payload → error UI, not crash)
- [ ] Python `BioComponentMixin` for `ProtocolCallbackHandler`
- [ ] LangGraph `interrupt()` integration for interactive components
- [ ] 6 component packages in `internal/biocomp/`

---

### Phase 6: Distribution & Cross-Platform
**Goal:** Production-ready distribution and UX polish.
**Duration:** 2-3 weeks
**Prerequisite:** Phase 5 working

**What:**
- Platform-specific wheel packaging (ruff model)
- Lazy download fallback
- Offline/HPC install path (`lobster tui download`)
- Homebrew tap
- Multiple themes (Lobster Dark, Lobster Light, Catppuccin, Dracula, Nord)
- `lobster tui --list-themes` and `lobster tui --theme <name>`
- Documentation: installation, theming, HPC deployment

**Deliverables:**
- [ ] CI workflow: Go cross-compile → platform wheels → PyPI
- [ ] `lobster tui download` command
- [ ] Homebrew formula
- [ ] 5+ built-in themes
- [ ] User documentation on docs.omics-os.com

---

### Phase 7: SSH & Cloud (Future)
**Goal:** Serve the TUI over SSH for cloud users.
**Duration:** TBD (post-funding)

**What:**
- Charm's `wish` library integration
- `ssh app.omics-os.com` → lobster chat TUI
- Authentication via SSH keys or tokens
- Cloud session management

This is a major cloud differentiator — no other bioinformatics platform offers this.

---

## 10. Risk Mitigations

### 10.1 stdio Corruption (CRITICAL)

**Problem:** Python C extensions (numpy, scanpy, pandas) can write directly to stdout/stderr
at the C level, bypassing Python's `sys.stdout`. This corrupts the JSON Lines stream.

**Solution:** Do NOT use stdout/stdin for the protocol. Use dedicated file descriptors:

```python
# Python side: create pipe pair
proto_read_fd, proto_write_fd = os.pipe()
response_read_fd, response_write_fd = os.pipe()

process = subprocess.Popen(
    [binary, "chat", "--proto-fd-in", str(proto_read_fd),
                     "--proto-fd-out", str(response_write_fd)],
    pass_fds=(proto_read_fd, response_write_fd),
    stdout=subprocess.DEVNULL,  # Swallow any C extension noise
    stderr=log_file,            # Debug logging
)
```

```go
// Go side: read from fd passed via flag
protoIn := os.NewFile(uintptr(protoFdIn), "proto-in")
protoOut := os.NewFile(uintptr(protoFdOut), "proto-out")
handler := protocol.NewHandler(protoIn, protoOut)
```

**For Phase 0 (init):** Not needed — one-shot subprocess uses capture_output + JSON on stdout.
C extensions don't load during init.

**For Phase 1+ (chat):** Required. LangGraph loads scanpy, pandas, etc.

### 10.2 Protocol Drift (CRITICAL)

**Problem:** 22+ message types defined in both Python and Go. Manual sync = guaranteed drift.

**Solution:** Single-source schema with code generation.

```
lobster-tui/protocol/
  schema.json          # Single source of truth (JSON Schema)
  generate.py          # Generates Python dataclasses
  generate.go          # Generates Go structs
```

OR: Use Protocol Buffers with `protoc` generating both Python and Go types.

**Minimum viable:** Protocol version in handshake (§6.6). If versions mismatch, fall back
to Python CLI. This buys time before implementing codegen.

### 10.3 Zombie Processes (HIGH)

**Problem:** If Python crashes, Go process stays alive. If Go crashes, Python may hang.

**Solution:**

```python
# Python: use process group for clean shutdown
import os, signal

process = subprocess.Popen(
    [binary, "chat", ...],
    preexec_fn=os.setsid  # New process group
)

def _cleanup():
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    except ProcessLookupError:
        pass

import atexit
atexit.register(_cleanup)
```

```go
// Go: detect parent death (Linux: prctl, macOS: kqueue)
// If parent PID changes to 1 (init), exit immediately
go func() {
    ppid := os.Getppid()
    for {
        time.Sleep(1 * time.Second)
        if os.Getppid() != ppid {
            os.Exit(0)  // Parent died, exit cleanly
        }
    }
}()
```

### 10.4 HPC Distribution (HIGH)

**Problem:** Air-gapped HPC clusters can't download binaries at runtime.

**Solution:** Multiple install paths (§8). Key addition:

```bash
# Pre-download for offline transfer
lobster tui download --platform linux-amd64 --output /path/to/lobster-tui

# Or install the platform wheel via pip (works offline if wheel is present)
pip install lobster-ai-tui-*.whl
```

Document this prominently in the HPC deployment guide.

### 10.5 Feature Velocity (HIGH)

**Problem:** Every new UI feature requires changes in both Python and Go.

**Mitigation:**
1. **OutputAdapter stays:** Commands still render through Python adapters. Go TUI is just
   another adapter target (like Console vs Dashboard vs JSON).
2. **Generic message types:** Most features use `table`, `markdown`, `code` — not custom types.
   Only truly new UI patterns require Go changes.
3. **The bridge is stable:** Once the protocol is proven, most development happens in
   Python (new agents, services, tools) or Go (new widgets). Rarely both.

---

## 11. Go Dependency Stack

### 11.1 Direct Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `charmbracelet/bubbletea` | v2 (latest) | TUI framework (Elm architecture) |
| `charmbracelet/huh` | latest | Interactive forms/prompts (for init wizard) |
| `charmbracelet/lipgloss` | v2 (latest) | Terminal styling + layout |
| `charmbracelet/glamour` | v2 (latest) | Markdown rendering |
| `charmbracelet/bubbles` | v2 (latest) | Text input, viewport, spinner, progress |
| `charmbracelet/harmonica` | latest | Spring physics animations |
| `alecthomas/chroma/v2` | latest | Syntax highlighting |

### 11.2 Go Module Setup

```
lobster-tui/
├── go.mod
├── go.sum
├── cmd/lobster-tui/
│   └── main.go              # Entry point: init | chat subcommands
├── internal/
│   ├── protocol/
│   │   ├── types.go          # Message definitions (from schema)
│   │   └── handler.go        # Async read/write over fd pair
│   ├── init/
│   │   └── wizard.go         # huh-based init wizard
│   ├── chat/
│   │   ├── model.go          # bubbletea Model (state machine)
│   │   └── views.go          # View rendering
│   ├── widgets/
│   │   ├── agents_panel.go   # Agent status display
│   │   ├── modality_panel.go # Loaded data display
│   │   ├── activity_log.go   # Tool execution feed
│   │   └── status_bar.go     # Token counts, active agent
│   ├── theme/
│   │   ├── theme.go          # Theme registry + builder
│   │   ├── lobster.go        # Lobster Dark/Light themes
│   │   ├── community.go      # Catppuccin, Dracula, Nord, etc.
│   │   └── loader.go         # JSON theme loading
│   └── ui/
│       └── views/
│           ├── markdown.go   # Glamour markdown rendering
│           ├── table.go      # Table rendering
│           ├── code.go       # Syntax-highlighted code
│           └── progress.go   # Progress bars + steps
└── themes/
    ├── lobster-dark.json
    ├── lobster-light.json
    └── catppuccin-mocha.json
```

---

## 12. Key Files & Resources

### 12.1 Reference Implementations

| Project | Location | What to Study |
|---------|----------|---------------|
| **AgentUI** | `~/GITHUB/agentui/` | Protocol, bridge, Go TUI architecture |
| **Charm Crush** | `github.com/charmbracelet/crush` | Gold standard Go AI TUI, uses fantasy+huh+bubbletea v2 |
| **Charm Mods** | `github.com/charmbracelet/mods` | Simpler AI TUI, good for understanding streaming pattern |
| **Charm huh** | `github.com/charmbracelet/huh` | Form library (for init wizard) |

### 12.2 Current Lobster Files (to be modified)

| File | Current Purpose | Planned Change |
|------|----------------|----------------|
| `lobster/cli.py:3623-4977` | `lobster init` (Rich prompts) | Add Go TUI path with fallback |
| `lobster/cli.py:1048` | `lobster dashboard` (Textual) | Add `lobster chat --tui` Go path |
| `lobster/ui/callbacks/textual_callback.py` | LangGraph → Textual bridge | Fork into `ProtocolCallbackHandler` |
| `lobster/cli_internal/commands/output_adapter.py` | OutputAdapter ABC | Add `GoTUIOutputAdapter` |
| `lobster/ui/themes.py` | LobsterTheme (Rich styles) | Mirror in Go theme system |
| `lobster/integrations/langchain_tui_backend.py` | BackendProtocol | Adapt for Go TUI bridge |

### 12.3 AgentUI Files to Fork

| File | Lines | What to Take |
|------|-------|-------------|
| `src/agentui/protocol.py` | 311 | Message types, serialization, payload builders |
| `src/agentui/bridge/tui_bridge.py` | 509 | Async subprocess management, request/response |
| `src/agentui/bridge/cli_bridge.py` | 281 | Rich fallback pattern |
| `src/agentui/bridge/base.py` | ~50 | Bridge ABC |
| `internal/protocol/handler.go` | 191 | Go-side async read/write |
| `internal/protocol/types.go` | 265 | Go message definitions |
| `internal/theme/theme.go` | 317 | Theme registry + style builder |
| `internal/theme/loader.go` | 215 | JSON theme loading |
| `internal/ui/views/views.go` | 656 | Markdown, table, code, progress views |
| `internal/ui/animations/spring.go` | 237 | Spring physics for modal transitions |

### 12.4 Documentation

| Resource | URL/Path |
|----------|----------|
| Bubbletea docs | https://github.com/charmbracelet/bubbletea |
| Huh docs | https://github.com/charmbracelet/huh |
| Lipgloss docs | https://github.com/charmbracelet/lipgloss |
| Glamour docs | https://github.com/charmbracelet/glamour |
| Crush source (UX reference) | https://github.com/charmbracelet/crush |
| AgentUI source (architecture) | ~/GITHUB/agentui/ |
| Lobster CLAUDE.md | ~/Omics-OS/lobster/CLAUDE.md |
| Lobster architecture docs | ~/Omics-OS/lobster/.claude/docs/architecture.md |

---

## 13. Decision Log

| # | Decision | Rationale | Date |
|---|----------|-----------|------|
| 1 | **Go + Charm ecosystem** (not Rust, not Node.js) | Most complete TUI ecosystem (40K stars). Single binary, easy cross-compile. Charm team well-funded. `wish` enables SSH TUI. | 2026-03-04 |
| 2 | **JSON Lines protocol** (not gRPC, not HTTP) | Simplest IPC. Same pattern as LSP, MCP, Neovim. No port management. Proven by AgentUI. | 2026-03-04 |
| 3 | **Dedicated fd pair** (not stdout) | Brutalist finding: Python C extensions (numpy, scanpy) can corrupt stdout. fd 3/4 isolates protocol from library noise. | 2026-03-04 |
| 4 | **huh for forms** (not custom bubbletea components) | Battle-tested, 5 themes, accessibility mode. AgentUI's custom forms are inferior. | 2026-03-04 |
| 5 | **Phase 0 is one-shot subprocess** (not IPC bridge) | Brutalist feedback: init doesn't prove streaming. But it DOES prove: Go build pipeline, distribution, UX, and theming. Zero IPC risk. | 2026-03-04 |
| 6 | **Platform wheels as primary distribution** | Follows ruff/uv model. Zero user friction. `uv tool install 'lobster-ai[tui-go]'` just works. | 2026-03-04 |
| 7 | **Rich CLI fallback is permanent** | Bioinformaticians on HPC may never get the Go binary. The Python path must always work. | 2026-03-04 |
| 8 | **Protocol versioning from day 1** | Brutalist finding: no version negotiation = guaranteed drift. Version in handshake + codegen from schema. | 2026-03-04 |
| 9 | **Lobster theme: #e45c47 primary** | Matches existing LobsterTheme.PRIMARY_ORANGE. Brand consistency across Rich and Go. | 2026-03-04 |
| 10 | **Bubbletea v2 + bubbles v2** (not v1) | Crush uses v2. It's the current version. AgentUI uses v1 — we upgrade. | 2026-03-04 |

---

## Appendix A: Lobster Theme Specification

```json
{
  "id": "lobster-dark",
  "name": "Lobster Dark",
  "description": "Official Lobster AI dark theme",
  "author": "Omics-OS",
  "version": "1.0.0",
  "colors": {
    "primary": "#e45c47",
    "secondary": "#CC2C18",
    "background": "#1a1a2e",
    "surface": "#252538",
    "overlay": "#2a2a40",
    "text": "#FAFAFA",
    "text_muted": "#888888",
    "text_dim": "#555555",
    "success": "#28a745",
    "warning": "#ffc107",
    "error": "#dc3545",
    "info": "#17a2b8",
    "accent1": "#e45c47",
    "accent2": "#FF6B4A",
    "accent3": "#4CAF50"
  }
}
```

Maps to existing Python theme:
- `LobsterTheme.PRIMARY_ORANGE` → `primary: #e45c47`
- `LobsterTheme.COLORS["genomics"]` → `accent3: #4CAF50`
- `LobsterTheme.COLORS["proteomics"]` → could add as extended palette

## Appendix B: Comparison — Before and After

### Before (Python Rich — current `lobster init`)

```
Select your LLM provider:
  1 - Claude API (Anthropic)
  2 - AWS Bedrock
  3 - Ollama (Local)
  4 - Google Gemini
  5 - Azure AI
  6 - OpenAI
  7 - OpenRouter
Choose provider [1]: _
```

### After (Go + huh — planned)

```
┌  lobster init                          lobster-dark
│
◇  Which LLM provider?
│  ● Anthropic (Claude) — Quick testing, development
│  ○ AWS Bedrock — Production, enterprise
│  ○ Ollama — Privacy, zero cost, offline
│  ○ Google Gemini — Latest models with thinking
│  ○ Azure AI — Enterprise Azure deployments
│  ○ OpenAI — GPT-4o, reasoning models
│  ○ OpenRouter — 600+ models via one API key
│
◇  Enter your Claude API key
│  sk-ant-••••••••••••
│
◆  Select agent packages  (Space to toggle)
│  ◼ lobster-transcriptomics  (3 agents — scRNA-seq, bulk RNA)
│  ◼ lobster-research         (2 agents — PubMed, GEO, data)
│  ◻ lobster-genomics         (2 agents — VCF, GWAS)
│  ◻ lobster-proteomics       (3 agents — MS, affinity, biomarkers)
│  ◻ lobster-metabolomics     (1 agent — LC-MS, GC-MS, NMR)
│  ◻ lobster-visualization    (1 agent — Plotly charts)
│
◇  Agent profile?
│  ○ Development   (Sonnet 4 — fastest, most affordable)
│  ● Production    (Sonnet 4 + Sonnet 4.5 supervisor) [Recommended]
│  ○ Performance   (Sonnet 4.5 — highest quality)
│  ○ Max           (Opus 4.5 supervisor — most capable, expensive)
│
⠋ Validating API key...
✓ API key valid
⠋ Testing NCBI connectivity...
✓ NCBI connection OK
│
└  Done! Configuration saved.
   Run lobster chat to start analyzing.
```

---

## 14. Execution Addendum (Owner Constraints, 2026-03-04)

This addendum converts the approved design into an implementation-ready execution plan.
It captures your latest direction and marks unresolved items explicitly.

### 14.1 Hard Direction (from your latest responses)

1. **Primary migration order is mandatory:**
   - Replace `chat` first.
   - Replace `init`, `query`, and then all command surfaces.
   - Keep legacy chat accessible behind an explicit tag/flag.
2. **At least one requirement is unresolved** (`"could you elaborate?"`).
   - Treated as **blocked** until clarified (see §14.8).
3. One previously discussed item is **optional**:
   - Can be dropped for MVP; keep as post-MVP nice-to-have.
4. **Under-the-hood transparency is critical**:
   - Users must be able to clearly inspect agent/tool flow and runtime events.
5. One previously discussed item is explicitly **not accepted** (`"No"`):
   - Treated as excluded from MVP unless re-opened.
6. One previously discussed item remains **not finalized** (`"elaborate"`):
   - Also blocked for final scope until clarified (see §14.8).
7. **Isolated worktree is required** for this effort.
8. UX directive: **best UX, no split layout** (`NO SPLI` interpreted as no split-pane UI).

### 14.2 Command Migration Contract (concrete parity targets)

| Surface | Go-UI Target | Legacy Access Requirement | Parity Exit Criteria |
|---|---|---|---|
| `lobster chat` | Default in `--ui auto` when binary exists | `lobster chat --ui classic` (or equivalent explicit legacy tag) | Input, streaming, slash commands, session continuity, error handling parity |
| `lobster init` | Go wizard path first | `lobster init --ui classic` | All provider/profile/agent-selection flows preserved |
| `lobster query` | Single-turn render in Go UI mode | `--json` remains classic/no-Go by design | Output correctness and session behavior parity |
| Slash/command surfaces | Routed through protocol adapter | Classic dispatch still callable | Shared command behavior unchanged |
| Dashboard/Textual | Out of immediate replacement scope unless explicitly added | Existing command remains callable | No regressions in existing users until formal deprecation decision |

### 14.3 Legacy Tag Policy (explicit and safe)

- Legacy path must remain first-class during rollout.
- Recommended contract (plan-level, not implementation):
  - `--ui classic` always forces current Python CLI UX.
  - `--ui go` hard-fails if binary missing (no silent fallback).
  - `--ui auto` prefers Go when available, otherwise classic with clear notice.
- Deprecation policy (later phase): only after stable telemetry and user sign-off.

### 14.4 UX Contract: "No Split" Single-Surface Layout

To satisfy "best UX, NO SPLI":

- **One primary conversation surface** (no permanent side-by-side panes).
- Secondary information appears as:
  - inline event feed blocks,
  - modal overlays,
  - or collapsible inspector drawer.
- Prohibited for MVP:
  - persistent left-right split chat+telemetry layouts,
  - multi-pane dashboards inside `chat`.

### 14.5 Under-the-Hood Transparency (Critical Requirement)

Must-have inspection features in MVP (not optional):

1. **Agent transition feed** (`supervisor -> specialist -> supervisor`).
2. **Tool execution feed** (tool name, agent, status, duration).
3. **Status line** with active phase and token/cost snapshot when available.
4. **Command visibility**: slash command dispatch and command result summaries visible.
5. **Failure visibility**: protocol errors, tool errors, and fallbacks are surfaced clearly.

Recommended interaction model:

- Default view: clean conversation stream.
- Toggle key/command: opens detailed runtime inspector.
- No hidden "silent pass" behavior for runtime events.

### 14.6 Integration Complexity Review (Go frontend <-> Python client)

Overall complexity to connect Go UI with existing Python client: **High**.

Breakdown:

- **Protocol + IPC bridge:** High
  - Dedicated fd pair, blocking request/response correlation, lifecycle control.
- **Callback adaptation:** Medium-High
  - Existing callback stack is console-centric; must emit structured protocol events.
- **Slash command interoperability:** Medium
  - Good foundation exists (`OutputAdapter` + `_dispatch_command`), but event plumbing is non-trivial.
- **Session/state parity:** Medium
  - Existing `session_id`, save/load semantics are reusable; behavior drift risk remains.
- **Fallback correctness:** Medium
  - Must avoid ambiguous auto behavior and hidden backend switching.
- **Distribution/install path:** High
  - Binary discovery, wheel packaging, offline/HPC workflows, version compatibility.

Practical conclusion:

- This is not a UI repaint; this is a **frontend architecture migration**.
- Fastest safe path remains phased rollout with strict parity gates per command.

### 14.7 Isolated Worktree Operating Model (Required)

Execution rule:

- This project runs in a dedicated worktree and branch; no mixed edits with unrelated feature work.

Recommended setup:

```bash
git worktree add ../lobster-charm-ui -b feature/charm-ui
cd ../lobster-charm-ui
```

Guardrails:

- Keep PRs phase-scoped (`init`, `chat`, `query`, command parity, packaging).
- Do not mix protocol refactors with unrelated agent/service changes.
- Maintain a rollback-safe path to `--ui classic` in every phase.

### 14.8 Not Yet Fully Answered (Blocked Questions)

The following items are still unresolved and must be closed before full implementation lock:

1. **Response #2 unresolved requirement**
   - You requested elaboration; the original question/options should be restated and finalized.
2. **Response #6 unresolved requirement**
   - Same status: needs a concrete decision, not "elaborate".
3. **Response #5 "No" mapping**
   - Confirm exactly which capability was rejected so it is formally excluded from MVP scope.
4. **"NO SPLI" interpretation check**
   - Current plan assumes "no split-pane layout." Confirm this interpretation explicitly.
5. **Legacy tag format**
   - Confirm whether legacy access should be only `--ui classic` or also a dedicated alias command.

### 14.9 Biggest Risks (Updated Priority)

1. **Scope ambiguity from unresolved decisions (#2, #6, #5 mapping).**
   - Risk: implementation churn and rework.
   - Mitigation: close §14.8 before coding each affected phase.
2. **Protocol/backend drift across Python and Go.**
   - Risk: runtime incompatibilities and brittle releases.
   - Mitigation: handshake version checks, schema-first generation, strict compatibility tests.
3. **Behavior regression during command migration.**
   - Risk: slash command inconsistency and broken workflows.
   - Mitigation: command parity matrix + golden transcript tests.
4. **Process lifecycle failures (zombies, hung bridge).**
   - Risk: broken terminal sessions and user distrust.
   - Mitigation: process group management, explicit shutdown protocol, timeout handling.
5. **Distribution friction in HPC/offline environments.**
   - Risk: enterprise adoption blockers.
   - Mitigation: wheel-first + offline binary transfer path documented and tested.

### 14.10 Phase Gate Checklist (do-not-pass criteria)

Before declaring each phase complete:

- `chat`: streaming, slash commands, transparency feed, and classic fallback verified.
- `init`: all provider/profile/agent-selection branches verified in Go and classic.
- `query`: output correctness parity (`text`, `--json`, session continuation) verified.
- Commands: shared dispatch parity and no command-class regressions.
- UX: no split-pane violations and transparency requirement demonstrably satisfied.
