# Roadmap: AQUADIF Refactor

## Milestones

- ✅ **v1.0 AQUADIF Refactor** — Phases 1-4 (shipped 2026-03-01)
- 📋 **v1.1 Monitoring & Validation** — Phases 5-7 (planned)

## Phases

<details>
<summary>✅ v1.0 AQUADIF Refactor (Phases 1-4) — SHIPPED 2026-03-01</summary>

- [x] Phase 1: AQUADIF Skill Creation (3/3 plans) — completed 2026-02-28
- [x] Phase 2: Contract Test Infrastructure (2/2 plans) — completed 2026-02-28
- [x] Phase 3: Reference Implementation (2/2 plans) — completed 2026-03-01
- [x] Phase 4: Agent Rollout (7/7 plans) — completed 2026-03-01

Full details: `.planning/milestones/v1.0-ROADMAP.md`

</details>

### 📋 v1.1 Monitoring & Validation (Planned)

- [x] **Phase 5: Monitoring Infrastructure** — Shared AquadifMonitor service injected into existing callback chain (not a new handler) (completed 2026-03-01)
- [ ] **Phase 6: Extension Case Study** — Validate AI self-extension via epigenomics package creation (with control group + cross-domain)
- [ ] **Phase 7: Documentation & Release** — Update all docs, architecture files, and skill references to reflect AQUADIF as shipped

### Phase 5: Monitoring Infrastructure
**Goal**: Enable runtime introspection of tool category usage and provenance compliance via a shared monitor service injected into the existing callback chain
**Depends on**: v1.0 (Phase 4)
**Requirements**: MON-01, MON-02, MON-03, MON-04, MON-05, MON-06
**Plans:** 2/2 plans complete

Plans:
- [ ] 05-01-PLAN.md — AquadifMonitor service class with TDD (thread-safe, fail-open, bounded data structures)
- [ ] 05-02-PLAN.md — Wire monitor into client/graph/callback/DataManagerV2 chain

**Architecture** (revised after Codex + Gemini brutalist review 2026-03-01):

Shared monitor service — NOT a new callback handler. `AquadifMonitor` is a thread-safe stateful service called from the always-on `TokenTrackingCallback.on_tool_start`. No additional handler in the callback chain.

```
AquadifMonitor (lobster/core/aquadif_monitor.py)
├── __init__(tool_metadata_map: dict)          # tool_name → {categories, provenance} from graph.py
├── record_tool_invocation(tool_name)          # Called ONLY from TokenTrackingCallback
├── record_provenance_call(tool_name, has_real_ir: bool)  # Called from DataManagerV2.log_tool_usage
├── get_category_distribution() → dict         # {"ANALYZE": 42, "IMPORT": 12, ...}
├── get_provenance_status() → dict             # {real_ir: [...], hollow_ir: [...], missing: [...]}
├── get_code_exec_log() → deque(maxlen=100)    # Bounded circular buffer
├── get_session_summary() → dict               # Structured event for cloud
└── All mutable state protected by threading.Lock
```

**Key design decisions (from brutalist review):**

1. **Path B only for metadata access** — build `tool_name → metadata` lookup dict at graph construction time in `graph.py`. Do NOT rely on LangChain passing `.tags`/`.metadata` through `on_tool_start` kwargs — undocumented, fragile, breaks on LangChain upgrades.

2. **Single injection point** — only `TokenTrackingCallback` (always-on) calls the monitor. Display handlers (Terminal, Textual, Streaming) do NOT call it. Eliminates double-counting in cloud sessions where both TokenTracker and StreamingCallback are active.

3. **Provenance via DataManagerV2 instrumentation** — do NOT parse tool output strings. Instead, add a hook in `DataManagerV2.log_tool_usage()` that calls `monitor.record_provenance_call(tool_name, has_real_ir=(ir is not None))`. This observes the actual call, not a guess.

4. **ir=None whitelist** — 20+ tools currently pass `ir=None` (hollow provenance from v1.0 bridge pattern). Monitor tracks these separately as `hollow_ir` status, not as violations. Prevents dashboard noise on day 1. Optionally report as warnings, not errors.

5. **Fail-open** — all monitor calls wrapped in try/except inside callback methods. Monitor exception never crashes a tool invocation or LLM call. `aquadif_monitor=None` disables all monitoring with zero overhead.

6. **Bounded data structures** — `code_exec_log` uses `collections.deque(maxlen=100)`. Category counters are simple dicts (thread-safe single-op under GIL). No unbounded lists. Session summary is O(1) to compute from counters.

7. **Thread safety** — `threading.Lock` around all compound state mutations. Simple counter increments are GIL-safe. Cloud `StreamingCallbackHandler` reads summary via `get_session_summary()` which acquires lock for consistent snapshot.

**Success Criteria** (what must be TRUE):
  1. `AquadifMonitor` class exists in `lobster/core/aquadif_monitor.py` with thread-safe state, bounded data structures, and fail-open error handling
  2. `graph.py` builds `tool_name → {categories, provenance}` lookup dict at graph construction time and passes it to monitor
  3. `TokenTrackingCallback.on_tool_start` calls `monitor.record_tool_invocation()` (single injection point, no other handlers call monitor)
  4. `DataManagerV2.log_tool_usage` calls `monitor.record_provenance_call(tool_name, has_real_ir)` — provenance detection by observation, not output parsing
  5. CODE_EXEC invocations logged to bounded `deque` with tool name, timestamp, agent attribution
  6. Provenance status distinguishes `real_ir` / `hollow_ir` / `missing` — ir=None tools tracked as hollow, not violations
  7. `get_session_summary()` returns structured dict consumable by Omics-OS Cloud SSE enrichment
  8. Monitor is opt-in: `aquadif_monitor=None` disables all monitoring; monitor exceptions never crash tool invocations
  9. All existing tests pass — zero behavioral changes to tool execution

**Brutalist review findings addressed:**
- ~~Provenance via output parsing~~ → DataManagerV2 instrumentation (Codex #1, Gemini #3)
- ~~Rely on LangChain kwargs~~ → Path B lookup dict (Codex #4, Gemini #2)
- ~~Multiple handlers call monitor~~ → Single injection via TokenTrackingCallback (Codex #7)
- ~~Unbounded memory~~ → Bounded deque + simple counters (Codex #9, Gemini #1)
- ~~ir=None floods dashboard~~ → Whitelist as hollow_ir status (Codex #2, Gemini #4)
- ~~Monitor crash kills requests~~ → Fail-open try/except (Codex #6)
- ~~Thread safety unaddressed~~ → threading.Lock on compound mutations (Codex #3, Gemini #1)

### Phase 6: Extension Case Study
**Goal**: Provide concrete evidence that a coding agent can build a new domain package (epigenomics) following the AQUADIF skill with minimal correction cycles
**Depends on**: Phase 5
**Requirements**: CASE-01, CASE-02, CASE-03, CASE-04, CASE-05, CASE-06
**Success Criteria** (what must be TRUE):
  1. A coding agent (Claude Code) designs an epigenomics tool set with correct AQUADIF categories by reading the updated lobster-dev skill
  2. Agent generates a complete epigenomics package structure following modular package patterns from the skill
  3. Generated package passes all AQUADIF contract tests on first or second attempt (correction cycle count measured)
  4. Package auto-registers via entry points and works with the supervisor without core modification
  5. Supervisor correctly routes epigenomics queries to the new agent in end-to-end testing
  6. Metrics are collected and documented: total time from skill invocation to passing tests, LOC generated vs edited, correction cycles, contract test first-attempt pass rate
**Plans**: TBD

**Prior eval data:** 4 iterations complete (smoke-01, iter-02, iter-03, iter-04) — see `skills/CLAUDE.md`
**Required:** Control group (WITHOUT condition), cross-domain trial (non-linear domain)

### Phase 7: Documentation & Release
**Goal**: Update all documentation, architecture files, and skill references to reflect AQUADIF as a shipped, validated system
**Depends on**: Phase 6
**Requirements**: DOC-01, DOC-02, DOC-03, DOC-04, DOC-05, DOC-06
**Success Criteria** (what must be TRUE):
  1. Root `CLAUDE.md` and `lobster/CLAUDE.md` reflect AQUADIF taxonomy in architecture sections
  2. `.github/CLAUDE.md` documents AQUADIF metadata as a requirement for new tools and PRs
  3. `docs-site/` has an AQUADIF page explaining the taxonomy
  4. `lobster-dev` skill references are current with scaffold workflow + AQUADIF validation
  5. `lobster-use` skill references mention AQUADIF categories where relevant
  6. `master_mermaid.md` includes AQUADIF metadata flow and contract test infrastructure
**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4 → 5 → 6 → 7

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1. AQUADIF Skill Creation | v1.0 | 3/3 | ✓ Complete | 2026-02-28 |
| 2. Contract Test Infrastructure | v1.0 | 2/2 | ✓ Complete | 2026-02-28 |
| 3. Reference Implementation | v1.0 | 2/2 | ✓ Complete | 2026-03-01 |
| 4. Agent Rollout | v1.0 | 7/7 | ✓ Complete | 2026-03-01 |
| 5. Monitoring Infrastructure | 2/2 | Complete   | 2026-03-01 | - |
| 6. Extension Case Study | v1.1 | 0/? | Not started | - |
| 7. Documentation & Release | v1.1 | 0/? | Not started | - |
