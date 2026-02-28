---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: unknown
last_updated: "2026-02-28T09:26:57.030Z"
progress:
  total_phases: 2
  completed_phases: 2
  total_plans: 5
  completed_plans: 5
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-28)

**Core value:** Every tool in Lobster AI declares what it does (category) and whether it must produce provenance — making the system introspectable, enforceable, and teachable to coding agents.
**Current focus:** Phase 2: Contract Test Infrastructure

## Current Position

Phase: 2 of 6 (Contract Test Infrastructure)
Plan: 2 of 2 (complete)
Status: Active
Last activity: 2026-02-28 — Phase 2 Plan 2 complete: AST provenance validation + parent agent tests + smoke tests (3060fd1)

Progress: [███░░░░░░░] 33%

## Phase 1 Completion Summary

**Phase 1: AQUADIF Skill Creation — COMPLETE** (2026-02-28)

All 5 success criteria met. Key deliverables:
- `skills/lobster-dev/references/aquadif-contract.md` — 10-category taxonomy contract (2520 words)
- 5 existing skill files updated with AQUADIF integration
- MANIFEST updated for installer distribution

**Teachability validation (exceeded plan scope):**
- 3 Claude trials (Phase 1 eval): All discovered + applied AQUADIF autonomously
- 3-agent cross-eval (iter-02): Claude B+, Gemini C, Codex B-
- Brutalist peer review: identified control group gap, metadata-runtime disconnect
- 3 skill fixes applied post-eval (committed 2a196cf)

**Publication evidence:** `~/omics-os/publications/papers/lobster-system/research/evidence/05-extension-easy-epigenomics.md`

## Phase 2 Completion Summary

**Phase 2: Contract Test Infrastructure — COMPLETE** (2026-02-28)

All 8 test requirements (TEST-01 through TEST-08) implemented. Key deliverables:
- `lobster/config/aquadif.py` — AquadifCategory enum + PROVENANCE_REQUIRED set
- `lobster/testing/contract_mixins.py` — 13 test methods (6 Phase 1 + 7 Phase 2)
- `tests/unit/agents/test_aquadif_compliance.py` — 7-test smoke suite
- `pytest.ini` + `pyproject.toml` — contract marker registration

**Critical tests implemented:**
- TEST-08 (AST provenance validation): Catches metadata-runtime disconnect found in 100% of Phase 1 eval agents
- TEST-06 (minimum viable parent): Ensures domain-expert parent agents maintain IMPORT + QUALITY + ANALYZE/DELEGATE
- TEST-07 (metadata uniqueness): Prevents shared metadata dict bug

**Test coverage:** All 8 requirements from Phase 2 research complete.

## Performance Metrics

**Velocity:**
- Total plans completed: 5
- Phase 1 plan execution: 229s (2 automated plans)
- Phase 1 checkpoint: multi-day eval cycle (exceeded scope — full cross-agent validation)
- Phase 2 plan 1 execution: 181s (2 tasks, fully automated)
- Phase 2 plan 2 execution: 332s (2 tasks, fully automated)

**By Phase:**

| Phase | Plans | Status | Date |
|-------|-------|--------|------|
| 01 | 3/3 | ✓ Complete | 2026-02-28 |
| 02 | 2/2 | ✓ Complete | 2026-02-28 |

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting Phase 2:

- **Skill-first validated:** Teachability confirmed across 3 agents — safe to invest in contract test infrastructure
- **Agents follow code examples over checklist text:** When __init__.py example showed bad pattern, 100% used it; after fix, 0%. Contract tests must enforce rules that skill docs can't
- **Taxonomy understanding ≠ ecosystem integration:** Codex scored best on AQUADIF categories (9/10) but worst on package structure (5/10). Contract tests must cover both
- **Metadata-runtime disconnect is the real gap:** Agents copy provenance boilerplate mechanically. AST-based tests needed to validate metadata matches runtime behavior
- **String enum for AQUADIF (02-01):** Use `(str, Enum)` base for AquadifCategory to enable string comparisons while maintaining type safety and Python 3.12 compatibility
- **AST validation implemented (02-02):** Use ast.parse() + ast.walk() to find log_tool_usage(ir=ir) calls — only reliable way to validate provenance tracking without executing tools
- **Smoke tests use mock tools (02-02):** Validate contract test infrastructure with @tool decorator + manual metadata, not real factories (fast, no LLM dependencies)
- **Contract marker in both configs (02-02):** Register in pytest.ini AND pyproject.toml for compatibility regardless of which file is authoritative

### Phase 2 Requirements (from eval findings)

These findings from Phase 1 eval MUST inform Phase 2 contract test design:

1. **Metadata-runtime consistency test (critical):** AST validation that provenance-flagged tools call `log_tool_usage(ir=ir)` AND non-provenance tools do NOT
2. **`try/except ImportError` detection:** 3/3 Phase 1 trials used prohibited pattern. Contract test should flag this
3. **`.tags` auto-population helper:** Consider `aquadif_metadata()` helper to eliminate manual mirroring (brutalist: engineering bug not docs fix)
4. **Package structure validation:** Codex failed PEP 420 layout, entry points, factory return type. Contract tests should cover these
5. **Test template standardization:** All 3 agents independently invented AQUADIF validation tests — include copy-paste template in contract

### Eval Infrastructure Available

Built during Phase 1 eval, reusable for Phase 2 and Phase 6:
- `.testing/skill-eval/` — Docker-based eval harness (Claude/Gemini/Codex support)
- `extract-metrics.py` AST logic design → directly informs contract test implementation
- Ground truth: `.testing/skill-eval/ground-truth/epigenomics.json`
- Full experimental protocol: `publications/papers/lobster-system/research/aquadif_validation_infrastructure_plan.md`

### Pending Skill Fixes (iter-03)

3 gaps identified in iter-02 cross-agent eval, not yet applied:
1. **Lazy import enforcement** — Gemini uses module-level imports in factory (2/3 agents failed)
2. **Flat layout requirement** — Codex used src/ layout breaking PEP 420 (1/3 failed)
3. **Plugin-not-standalone messaging** — Codex built standalone package instead of Lobster plugin (1/3 failed)

These can be applied as a quick task before or during Phase 2.

### Blockers/Concerns

**Phase 2:**
- Metadata propagation through LangGraph delegation chains not explicitly documented by LangChain — needs end-to-end contract test validation
- Should `.tags` helper be in Phase 2 scope or deferred?

**Phase 6:**
- Must include control group (WITHOUT condition) + cross-domain (multi-omics) trials
- Full 36-trial campaign infrastructure designed but not yet built (Gemini/Codex Dockerfiles, analysis pipeline, campaign orchestrator)

### Quick Tasks Completed

| # | Description | Date | Commit | Directory |
|---|-------------|------|--------|-----------|
| 1 | Refactor lobster-dev SKILL.md: replace hardcoded package tree with conceptual description, add subtle Omics-OS branding, optimize data flow section for AI readability | 2026-02-28 | b3785dd | [1-refactor-lobster-dev-skill-md-replace-ha](./quick/1-refactor-lobster-dev-skill-md-replace-ha/) |

## Session Continuity

Last session: 2026-02-28T09:25:53Z (Phase 2 Plan 2 execution)
Stopped at: Completed 02-02-PLAN.md — AST provenance validation + parent agent tests + smoke tests
Resume: Phase 2 complete — begin Phase 3 (agent package adoption) or Phase 6 (validation campaign)
Key artifacts:
- Phase 2 Plan 2 SUMMARY: `.planning/phases/02-contract-test-infrastructure/02-02-SUMMARY.md`
- Contract test mixin: `lobster/testing/contract_mixins.py` (13 test methods: 6 Phase 1 + 7 Phase 2)
- Smoke test suite: `tests/unit/agents/test_aquadif_compliance.py` (7 tests, all passing)
- AST helper: `_has_log_tool_usage_with_ir()` validates metadata-runtime consistency
- Contract marker: registered in pytest.ini and pyproject.toml
