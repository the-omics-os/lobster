# Roadmap: AQUADIF Refactor

## Overview

This roadmap transforms Lobster AI's ~180 tool bindings across 18 agents by adding a 10-category metadata taxonomy (AQUADIF) using native LangChain `BaseTool.metadata` and `BaseTool.tags` fields. The journey starts with skill creation to validate teachability BEFORE touching any code, then builds contract tests to define correctness, applies the pattern to a reference implementation (transcriptomics expert), rolls out systematically to all remaining agents, adds runtime monitoring infrastructure, validates via an extension case study where a coding agent builds an epigenomics package following the AQUADIF skill, and closes with comprehensive documentation updates across all repos and skill references.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: AQUADIF Skill Creation** - Create and validate lobster-dev skill for tool categorization (2026-02-28)
- [x] **Phase 2: Contract Test Infrastructure** - Implement automated validation of taxonomy compliance (2026-02-28)
- [ ] **Phase 3: Reference Implementation** - Apply pattern to transcriptomics expert (22 tools)
- [ ] **Phase 4: Agent Rollout** - Tag all remaining 156 tools across 16 agents with metadata
- [ ] **Phase 5: Monitoring Infrastructure** - Build runtime callback handler for category tracking and provenance enforcement
- [ ] **Phase 6: Extension Case Study** - Validate AI self-extension via epigenomics package creation
- [ ] **Phase 7: Documentation & Release** - Update all docs, architecture files, and skill references to reflect AQUADIF as shipped

## Phase Details

### Phase 1: AQUADIF Skill Creation
**Goal**: Validate that the AQUADIF categorization pattern is teachable to coding agents before investing in implementation
**Depends on**: Nothing (first phase)
**Requirements**: SKIL-01, SKIL-02, SKIL-03, SKIL-04, SKIL-05
**Success Criteria** (what must be TRUE):
  1. Contract reference document exists at `skills/lobster-dev/references/aquadif-contract.md` with all 10 category definitions, metadata assignment pattern, contract test requirements, and a complete example
  2. Existing skill files (`creating-agents.md`, `planning-workflow.md`, `SKILL.md`) reference AQUADIF as the tool organization framework
  3. A coding agent (Claude Code) can design a hypothetical epigenomics tool set with correct AQUADIF categories without additional prompting beyond the skill
  4. Skill follows Anthropic Skills Guide format (progressive disclosure, under 5K words in SKILL.md body)
  5. MANIFEST file includes the new reference file and skill installation works end-to-end
**Plans**: 3 plans

Plans:
- [x] 01-01: Create AQUADIF contract reference document and update MANIFEST
- [x] 01-02: Integrate AQUADIF into existing skill files (creating-agents, planning-workflow, SKILL.md, architecture, creating-services)
- [x] 01-03: Teachability validation — epigenomics dry run (checkpoint) + cross-agent eval (Claude B+, Gemini C, Codex B-)

### Phase 2: Contract Test Infrastructure
**Goal**: Define and implement automated validation that enforces taxonomy compliance at test time
**Depends on**: Phase 1
**Requirements**: TEST-01, TEST-02, TEST-03, TEST-04, TEST-05, TEST-06, TEST-07, TEST-08
**Success Criteria** (what must be TRUE):
  1. `AquadifCategory` enum exists in `lobster/config/aquadif.py` with all 10 categories and `PROVENANCE_REQUIRED` set defining the 7 categories that require provenance
  2. Contract test mixin in `lobster/testing/contract_mixins.py` has 8 test methods covering metadata presence, category validity, category cap, provenance compliance, minimum viable parent, metadata uniqueness, and AST-based provenance call validation
  3. Contract tests can be run against any agent's tool factory and produce clear pass/fail results
  4. Tests catch all known pitfalls (closure scoping drift, category inflation, missing provenance calls)
  5. Documentation explains how to run contract tests during agent development
**Plans**: 2 plans

**Phase 1 Eval Findings (must be addressed):**
- **Metadata-runtime consistency test (critical):** AST-based validation that provenance-flagged tools actually call `log_tool_usage(ir=ir)` AND that non-provenance tools do NOT log provenance. Phase 1 eval found agents copy boilerplate mechanically without adjusting based on metadata flag.
- **`try/except ImportError` detection:** 3/3 eval trials used prohibited pattern in `__init__.py`. Contract test should flag this.
- **`.tags` auto-population helper:** Consider adding `aquadif_metadata()` helper that sets both `.metadata` and `.tags` from a single call to eliminate manual mirroring (brutalist: engineering bug, not docs fix).
- **Test template for skill docs:** All 3 eval agents independently invented AQUADIF validation tests. Include a copy-paste test template in the contract to standardize.

Plans:
- [x] 02-01: Create AquadifCategory enum and add 5 basic contract test methods to mixin (TEST-01, TEST-02, TEST-03, TEST-04, TEST-05, TEST-07) — Complete (2026-02-28)
- [x] 02-02: Add parent agent and AST provenance validation tests, smoke test, pytest marker (TEST-06, TEST-08) — Complete (2026-02-28)

### Phase 3: Reference Implementation
**Goal**: Apply the AQUADIF pattern to transcriptomics expert to validate it works on a complete agent before rolling out to 156 additional tools
**Depends on**: Phase 2
**Requirements**: IMPL-01, IMPL-02, IMPL-03, IMPL-04
**Success Criteria** (what must be TRUE):
  1. All 22 tools (15 in transcriptomics_expert.py + 7 in shared_tools.py) have `.metadata` dict with `categories` and `provenance` keys and `.tags` list set correctly
  2. Shared tools in the transcriptomics package have correct metadata set at the source (shared_tools.py)
  3. All existing transcriptomics tests still pass (backward compatibility validated)
  4. Contract tests pass for the transcriptomics package
  5. Reference implementation pattern is documented as a template for Phase 4 rollout
**Plans**: 2 plans

Plans:
- [x] 03-01-PLAN.md — Tag all 22 tools with AQUADIF metadata and create contract tests (IMPL-01, IMPL-02, IMPL-03) — Complete (2026-03-01)
- [x] 03-02-PLAN.md — Create migration reference template for Phase 4 rollout (IMPL-04) — Complete (2026-03-01)

### Phase 4: Agent Rollout
**Goal**: Systematically tag all remaining 156 tools across 16 agents with AQUADIF metadata following the validated reference pattern
**Depends on**: Phase 3
**Requirements**: ROLL-01, ROLL-02, ROLL-03, ROLL-04, ROLL-05, ROLL-06, ROLL-07, ROLL-08, ROLL-09, ROLL-10
**Success Criteria** (what must be TRUE):
  1. All 10 agent packages (genomics, proteomics, metabolomics, research, visualization, metadata, structural-viz, ml, drug-discovery) have metadata on every tool
  2. Dynamic delegation tools in `graph.py` (`_create_lazy_delegation_tool`) have DELEGATE category metadata
  3. Contract tests pass for all 10 packages with zero failures
  4. Multi-category usage is under 40% across all ~222 tools (category minimalism validated)
  5. All existing tests across all packages still pass (no backward compatibility breaks)
**Plans**: 7 plans

Plans:
- [ ] 04-01-PLAN.md — Tag structural-viz + metabolomics + graph.py delegation tools (ROLL-03, ROLL-08)
- [ ] 04-02-PLAN.md — Tag metadata + transcriptomics children (ROLL-04, ROLL-07 partial)
- [ ] 04-03-PLAN.md — Tag genomics + visualization (ROLL-01, ROLL-07 partial)
- [ ] 04-04-PLAN.md — Tag ML package (ROLL-05)
- [ ] 04-05-PLAN.md — Tag proteomics (ROLL-02)
- [ ] 04-06-PLAN.md — Tag research (ROLL-06)
- [ ] 04-07-PLAN.md — Tag drug-discovery + global ROLL-09 validation (ROLL-09, ROLL-10)

### Phase 5: Monitoring Infrastructure
**Goal**: Enable runtime introspection of tool category usage and automated provenance compliance checking
**Depends on**: Phase 4
**Requirements**: MON-01, MON-02, MON-03, MON-04, MON-05, MON-06
**Success Criteria** (what must be TRUE):
  1. `AquadifCallbackHandler` class exists in `lobster/core/aquadif_callback.py` and logs category with every tool invocation
  2. Callback handler tracks category distribution per session and exposes it via structured events
  3. CODE_EXEC usage is flagged with details (which tool, when, what code executed)
  4. Provenance compliance is checked at runtime (callback detects if provenance-required tools completed without calling `log_tool_usage`)
  5. Callback handler integrates cleanly with existing callback infrastructure in `graph.py` without breaking existing monitoring
  6. Structured events are consumable by Omics-OS Cloud for cross-session analytics
**Plans**: TBD

Plans:
- [ ] 05-01: TBD

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

**Phase 1 Eval Findings (must be addressed):**
- **Control group required:** Run baseline trial WITHOUT `aquadif-contract.md` to measure actual teaching effect vs pretraining knowledge (brutalist critique)
- **Cross-domain trial required:** Test non-linear pipeline domain (e.g., spatial transcriptomics) to validate generalization beyond linear workflows (brutalist critique)
- **Metadata-runtime consistency:** Validate that provenance flag in `.metadata` matches actual `log_tool_usage` calls in generated code (eval found disconnect)
- **Evidence template:** Follow structured schema in `publications/papers/lobster-system/research/evidence/05-extension-easy-epigenomics.md`
- **Phase 1 early data:** 3 Docker trials already collected (naive/professional/publication prompts, 13-16 tools each, 7/10 categories, ~7-8m). Raw data in `.planning/phases/01-aquadif-skill-creation/eval/`

Plans:
- [ ] 06-01: TBD

### Phase 7: Documentation & Release
**Goal**: Update all documentation, architecture files, and skill references to reflect AQUADIF as a shipped, validated system — ensuring new contributors, coding agents, and users see the current state
**Depends on**: Phase 6
**Requirements**: DOC-01, DOC-02, DOC-03, DOC-04, DOC-05, DOC-06
**Success Criteria** (what must be TRUE):
  1. Root `CLAUDE.md` and `lobster/CLAUDE.md` reflect AQUADIF taxonomy in architecture sections (agent table includes categories, service pattern shows metadata assignment)
  2. `.github/CLAUDE.md` (contributor guidelines) documents AQUADIF metadata as a requirement for new tools and PRs
  3. `docs-site/` has an AQUADIF page (or section) explaining the taxonomy for end users and contributors
  4. `lobster-dev` skill references are current: `creating-agents.md`, `creating-services.md`, `architecture.md`, `code-layout.md` all reflect scaffold workflow + AQUADIF validation
  5. `lobster-use` skill references mention AQUADIF categories where relevant (e.g., tool capabilities per agent)
  6. `master_mermaid.md` architecture diagram includes AQUADIF metadata flow and contract test infrastructure
**Plans**: TBD

Plans:
- [ ] 07-01: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4 → 5 → 6 → 7

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. AQUADIF Skill Creation | 3/3 | ✓ Complete | 2026-02-28 |
| 2. Contract Test Infrastructure | 2/2 | ✓ Complete | 2026-02-28 |
| 3. Reference Implementation | 2/2 | ✓ Complete | 2026-03-01 |
| 4. Agent Rollout | 2/7 | In Progress|  |
| 5. Monitoring Infrastructure | 0/? | Not started | - |
| 6. Extension Case Study | 0/? | Not started | - |
| 7. Documentation & Release | 0/? | Not started | - |
