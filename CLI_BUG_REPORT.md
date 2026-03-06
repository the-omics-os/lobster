# CLI Bug Report (Charm/Go UI)

Date: 2026-03-04 19:00:48 PST  
Repo: `lobster-charm-ui`  
Commit: `742c99c`  
CLI: `lobster version 1.0.14`  
Python: `3.13.9`  
OS: `Darwin 25.3.0 (arm64)`

## Scope
Test target was the new Charm-based CLI via:
- `lobster chat --ui go`

Also compared behavior with:
- `lobster chat --ui auto`
- `lobster chat --ui classic`
- invalid UI selection: `lobster chat --ui banana`

## Findings

### 1) Slash commands are not recognized in Go UI
Severity: High

Steps:
1. Run `source .venv/bin/activate`
2. Run `lobster chat --ui go`
3. Wait until UI shows `Ready`
4. Enter `/help` (also reproduced with `/status` and `/files`)

Expected:
- Command executes (as documented in CLAUDE/CLI docs)

Actual:
- Error modal: `Unknown command: help` (or `status`, `files`)

Evidence:
- Reproduced manually and with `expect`.
- Example output: `⚠️ Error Unknown command: files`

---

### 2) After unknown-command error, chat becomes unusable
Severity: High

Steps:
1. Start Go UI and wait for `Ready`
2. Enter `/files` and get error modal
3. Press Enter (`Press Enter to return to chat...`)
4. Type `hello` and press Enter

Expected:
- Return to interactive chat and process next input

Actual:
- Input is echoed but no `You: hello` event appears; no response is produced

Evidence:
- `expect` automation timed out waiting for follow-up input handling:
  - `EXPECT followup_input_timeout`

---

### 3) Session label/header is truncated on ready screen
Severity: Medium

Steps:
1. Start `lobster chat --ui go`
2. Wait for ready state

Expected:
- Full session label (e.g., `session_...`) displayed correctly

Actual:
- Header shows truncated text like `sion_20260304_...`

Evidence:
- Multiple runs showed: `Lobster AI  sion_20260304_...`

---

### 4) Non-interactive mode hangs indefinitely for `--ui go` and `--ui auto`
Severity: High

Steps:
1. Run from non-TTY context (e.g., subprocess):
   - `lobster chat --ui go`
   - `lobster chat --ui auto`

Expected:
- Fast failure with clear non-interactive error OR fallback behavior

Actual:
- Only prints `Warning: Input is not a terminal (fd=0).`
- Process hangs (timed out after 8s in automated check)

Evidence:
- Automated timeout results for both commands

---

### 5) Invalid `--ui` value is accepted and falls through to runtime behavior
Severity: Medium

Steps:
1. Run `lobster chat --ui banana`

Expected:
- Immediate CLI validation error for unsupported UI value

Actual:
- Command proceeds into runtime UI flow (classic-style output), then ends in interactive fallback abort path

Evidence:
- No immediate option validation error was shown

---

## Notes
- Existing working tree had pre-existing changes in Go TUI-related files:
  - `lobster-tui/go.mod`
  - `lobster-tui/go.sum`
  - `lobster-tui/internal/chat/model.go`
  - `lobster-tui/internal/chat/views.go`
  - `lobster/cli_internal/go_tui_launcher.py`

