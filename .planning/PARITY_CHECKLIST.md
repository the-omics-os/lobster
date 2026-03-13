# Feature Parity + Protocol Compliance Verification

## Go TUI Feature Parity Matrix

| Feature | Go Implementation | Ink Implementation | Status |
|---------|------------------|-------------------|--------|
| Streaming text | streamBuf accumulation | Built-in MessageContent | PASS |
| Markdown rendering | glamour.TermRenderer | @assistant-ui/react-ink-markdown | PASS |
| Scrollable chat viewport | viewport.Model | ink-scroll-view (ChatViewport) | PASS |
| User input + submit | textinput.Model | Custom Composer with TextInput | PASS |
| Header (logo + agent + session) | renderHeader() | Header.tsx | PASS |
| Status bar (spinner + text) | renderStatusBar() | StatusBar.tsx | PASS |
| Agent transitions | TypeAgentTransition | HandoffRenderer.tsx | PASS |
| Tool feed (ring buffer of 5) | toolFeed[] | ActivityFeed.tsx | PASS |
| Tool identity threading | toolFeedIndex | ToolCallPrimitive tracking | PASS |
| Progress bar | renderProgressBar() | Progress.tsx (ProgressBar + IndeterminateSpinner) | PASS |
| Alert messages (colored) | renderAlert() | Alert.tsx (4 levels) | PASS |
| Table rendering | custom table views.go | DataTable.tsx | PASS |
| ChainOfThought / reasoning | N/A | ChainOfThought.tsx | PASS (new) |
| Init wizard (5 steps) | huh forms | InitWizard.tsx (WizardManifest) | PASS |
| Auth (local + cloud) | Config + env vars | config.ts + authHeaders() | PASS |
| Session management | Python bridge | sessions.ts (REST CRUD) | PASS |
| HITL: confirm (Y/n) | pendingConfirm | ConfirmPrompt.tsx | PASS |
| HITL: select (choice list) | pendingSelect | SelectPrompt.tsx | PASS |
| HITL: text input | pendingComponentID | TextInputPrompt.tsx | PASS |
| HITL: threshold slider | BioComp | ThresholdSlider.tsx | PASS |
| HITL: cell type selector | BioComp | CellTypeSelector.tsx | PASS |
| HITL: QC dashboard | BioComp | QCDashboard.tsx | PASS |
| Slash commands (native) | /help, /clear, /exit, /data | dispatcher.ts | PASS |
| Slash commands (bridged) | TypeSlashCommand → Python | dispatcher.ts (async POST) | PASS |
| Command autocomplete | textinput.SetSuggestions() | TextInput suggestions prop | PASS |
| Command history (arrow-up) | Custom history ring | useHistory.ts + persistence | PASS |
| Two-phase Ctrl+C cancel | isCanceling + 2s timer | useCancelHandler.ts | PASS |
| Non-interactive mode | lobster query | --query flag + query.ts | PASS |
| Plot summaries | N/A (files) | PlotSummaryRenderer.tsx | PASS (new) |
| Data table preview | N/A | DataTablePreviewRenderer.tsx | PASS (new) |
| Terminal resize | BubbleTea WindowSizeMsg | useTerminalSize.ts + Ink native | PASS |
| Heartbeat/keepalive | heartbeat protocol msg | DataStream built-in | PASS |
| Binary distribution | publish-tui.yml | build-ink-tui.yml (6 targets) | PASS |
| Python bootstrap | packages/lobster-ai-tui | ink_bootstrap.py (download + SHA256) | PASS |
| Error recovery | N/A | ErrorBoundary + useConnectionStatus | PASS (new) |
| Feature flags | N/A | featureFlags.ts (gating) | PASS (new) |
| Prompt templates | N/A | TemplateSelector.tsx | PASS (new) |
| Scientific review | N/A | ScientificReview.tsx | PASS (new) |
| Follow-up suggestions | N/A | FollowUpSuggestions.tsx | PASS (new) |
| Browser open | N/A | useBrowserOpen + openBrowser.ts | PASS (new) |

## Protocol Compliance (spec §10)

| Rule | Description | Status |
|------|-------------|--------|
| §1.3 R1 | Unknown aui-state keys silently ignored | PASS (stateHandlers.ts) |
| §1.3 R2 | Known key with _v > supported → ignore value | PASS (processStatePatch) |
| §1.3 R3 | New optional fields tolerated (no crash) | PASS (unknown typed as `unknown`) |
| §3 | Message dedup by message_id during hydration | PASS (hydration.ts) |
| §4.1 | Feature flags from GET /config/flags | PASS (featureFlags.ts) |
| §4.2 | Prompt templates from GET /config/templates | PASS (templates.ts) |
| §10.3 | No required fields on request bodies | PASS (minimal payloads) |

## Blocked Items (require backend prerequisites)

| Feature | Blocked on | Step |
|---------|-----------|------|
| SSE reconnection + degraded mode | SSE event IDs, ring buffer, Last-Event-ID | 5.9 |
| Resources @autocomplete | GET /resources | 5.10 |
| Project & dataset CLI commands | projects_datasets flag + endpoints | 5.11 |
| Agent store CLI commands | curated_agent_store flag + endpoints | 5.12 |

These items will be completed when backend prerequisites are available.
