# Requirements: AQUADIF Refactor

**Defined:** 2026-02-27
**Core Value:** Every tool in Lobster AI declares what it does (category) and whether it must produce provenance — making the system introspectable, enforceable, and teachable to coding agents.

## v1 Requirements

Requirements for the AQUADIF refactor. Each maps to roadmap phases.

### Skill Creation

- [x] **SKIL-01**: AQUADIF contract reference document created at `skills/lobster-dev/references/aquadif-contract.md` with 10 category definitions, metadata assignment pattern, contract test requirements, and full example for a hypothetical domain agent
- [x] **SKIL-02**: `skills/lobster-dev/references/creating-agents.md` updated to include AQUADIF as the tool organization framework with assignment pattern examples
- [x] **SKIL-03**: `skills/lobster-dev/references/planning-workflow.md` updated so Phase 3 (domain knowledge) maps to AQUADIF categories
- [x] **SKIL-04**: `skills/lobster-dev/SKILL.md` references AQUADIF in main instructions and `MANIFEST` includes new reference file
- [ ] **SKIL-05**: Skill validated by having a coding agent design an epigenomics tool set with correct categories without additional prompting

### Contract Tests

- [x] **TEST-01**: `AquadifCategory` enum created in `lobster/config/aquadif.py` with 10 categories and `PROVENANCE_REQUIRED` set (7 categories) — Complete (02-01)
- [x] **TEST-02**: Contract test `test_tools_have_metadata` — every tool returned by factory has `.metadata` with `categories` key — Complete (02-01)
- [x] **TEST-03**: Contract test `test_categories_are_valid` — all categories are from the AQUADIF 10-category set — Complete (02-01)
- [x] **TEST-04**: Contract test `test_categories_capped_at_three` — no tool has more than 3 categories — Complete (02-01)
- [x] **TEST-05**: Contract test `test_provenance_tools_have_flag` — tools with provenance-requiring primary categories declare `provenance: True` — Complete (02-01)
- [x] **TEST-06**: Contract test `test_minimum_viable_parent` — parent agents have IMPORT + QUALITY + one of ANALYZE/DELEGATE — Complete (02-02)
- [x] **TEST-07**: Contract test for metadata uniqueness — tools don't share metadata dict objects (closure scoping pitfall prevention) — Complete (02-01)
- [x] **TEST-08**: AST-based provenance validation — provenance-required tools contain `log_tool_usage` call in their body — Complete (02-02)

### Reference Implementation

- [x] **IMPL-01**: All 15 tools in `transcriptomics_expert.py` have correct `metadata` and `tags` set per dry run mapping (15 tools in transcriptomics_expert.py + 7 shared tools = 22 total in package) — Complete (03-01)
- [x] **IMPL-02**: Shared tools in transcriptomics package (if any) have correct metadata — Complete (03-01)
- [x] **IMPL-03**: All existing transcriptomics tests still pass after metadata addition — Complete (03-01)
- [x] **IMPL-04**: Reference implementation pattern documented for other agents to follow

### Agent Rollout

- [x] **ROLL-01**: `genomics_expert` (12 tools) and `variant_analysis_expert` (8 tools) have metadata and pass contract tests
- [x] **ROLL-02**: `proteomics_expert` (21 tools), `proteomics_de_analysis_expert` (7 tools), and `biomarker_discovery_expert` (7 tools) have metadata and pass contract tests
- [x] **ROLL-03**: `metabolomics_expert` (10 tools) has metadata and passes contract tests
- [x] **ROLL-04**: `annotation_expert` (10 tools) and `de_analysis_expert` (14 tools) have metadata and pass contract tests
- [x] **ROLL-05**: `machine_learning_expert` (9 tools), `feature_selection_expert` (6 tools), and `survival_analysis_expert` (6 tools) have metadata and pass contract tests
- [x] **ROLL-06**: `research_agent` (13 tools) and `data_expert` (10 tools) have metadata and pass contract tests
- [x] **ROLL-07**: `visualization_expert` (11 tools), `metadata_assistant` (~8 tools), and `protein_structure_visualization_expert` (~4 tools) have metadata and pass contract tests
- [x] **ROLL-08**: Dynamic DELEGATE tools in `graph.py` (`_create_lazy_delegation_tool`) have DELEGATE metadata
- [x] **ROLL-09**: All ~180 tools across 18 agents pass contract tests; multi-category usage is <40% — 1/221 tools multi-category (0.5%)
- [x] **ROLL-10**: `drug_discovery_expert` (10 shared tools), `cheminformatics_expert` (9 tools), `clinical_dev_expert` (8 tools), and `pharmacogenomics_expert` (8 tools) have metadata and pass contract tests

### Monitoring

- [x] **MON-01**: `AquadifMonitor` in `lobster/core/aquadif_monitor.py` logs category with every tool invocation via `record_tool_invocation()` — Complete (05-01, 05-02)
- [x] **MON-02**: Monitor tracks category distribution per session via `get_category_distribution()` — Complete (05-01)
- [x] **MON-03**: Monitor flags CODE_EXEC usage with details via bounded `deque(maxlen=100)` log and `get_code_exec_log()` — Complete (05-01)
- [x] **MON-04**: Monitor checks provenance compliance at runtime via `record_provenance_call()` called from `DataManagerV2.log_tool_usage` — Complete (05-01, 05-02)
- [x] **MON-05**: Monitor injected into existing callback chain — `TokenTrackingCallback.on_tool_start` calls `record_tool_invocation`, `graph.py` builds tool metadata map — Complete (05-02)
- [x] **MON-06**: Monitor emits structured events via `get_session_summary()` consumable by Omics-OS Cloud — Complete (05-01)

### Documentation & Release

- [x] **DOC-01**: Root `CLAUDE.md` and `lobster/CLAUDE.md` reflect AQUADIF taxonomy in architecture sections — Complete (06-01)
- [x] **DOC-02**: `.github/CLAUDE.md` documents AQUADIF metadata as a requirement for new tools and PRs (Hard Rules 11-12) — Complete (06-01)
- [x] **DOC-03**: `docs-site/` has an AQUADIF page explaining the taxonomy at `extending/aquadif-taxonomy.mdx` — Complete (06-02)
- [x] **DOC-04**: `lobster-dev` skill references are current with scaffold workflow + AQUADIF validation — Complete (06-02)
- [x] **DOC-05**: `lobster-use` skill references mention AQUADIF categories where relevant — Complete (06-02)
- [x] **DOC-06**: `master_mermaid.md` includes AQUADIF metadata flow and contract test infrastructure (section 12) — Complete (06-01)

### Extension Case Study

- [ ] **CASE-01**: Coding agent (Claude Code) loads updated lobster-dev skill and designs epigenomics (ChIP-seq) tool set with correct AQUADIF categories
- [ ] **CASE-02**: Coding agent generates epigenomics package structure following lobster-dev skill
- [ ] **CASE-03**: Generated package passes all AQUADIF contract tests on first or second attempt
- [ ] **CASE-04**: Package auto-registers via entry points without core modification
- [ ] **CASE-05**: Supervisor correctly routes epigenomics queries to the new agent
- [ ] **CASE-06**: Metrics collected: time from skill invocation to passing tests, LOC generated vs edited, correction cycles, contract test first-attempt pass rate

## v2 Requirements

Deferred to future release. Tracked but not in current roadmap.

### Prompt Auto-Generation

- **PRMT-01**: Utility function `generate_tool_prompt_section(tools)` groups tools by primary category
- **PRMT-02**: Agent `prompts.py` files updated to use generator instead of hardcoded tool lists
- **PRMT-03**: A/B test validates auto-generated prompts perform within 5% of handcrafted on tool selection accuracy

### Analytics & Access Control

- **ANLC-01**: Cross-session analytics aggregate category usage across Omics-OS Cloud users
- **ANLC-02**: Tool recommendation engine suggests related tools based on category co-occurrence
- **ANLC-03**: Category-based access control restricts CODE_EXEC to admin users in enterprise deployments

## Out of Scope

| Feature | Reason |
|---------|--------|
| Two-layer taxonomy | Rejected after brutalist review — single layer with multi-category is sufficient |
| Custom `@tool_meta` decorator | Breaks `inspect.signature`, conflicts with LangChain schema extraction |
| Centralized metadata dict on AGENT_CONFIG | Drift risk, doesn't work with dynamic tools, not self-documenting |
| Per-domain taxonomies | Fragmentation — domain-agnostic categories (IMPORT, QUALITY, ANALYZE) work everywhere |
| Pure delegation model for parents | Loses domain-specific computation; parents keep ANALYZE tools |
| Mandatory SYNTHESIZE implementations | Honest gap is better science — 0 implementations is intentional |
| Runtime category mutation | Breaks monitoring, confuses LLMs, defeats caching — categories fixed at tool creation |
| Hierarchical tool dependencies in metadata | That's workflow logic, belongs in agent prompts or LangGraph state transitions |
| Automatic tool selection via category | Too coarse — LLM must select specific tool by name, categories inform grouping |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| SKIL-01 | Phase 1 | Complete |
| SKIL-02 | Phase 1 | Complete |
| SKIL-03 | Phase 1 | Complete |
| SKIL-04 | Phase 1 | Complete |
| SKIL-05 | Phase 1 | Pending |
| TEST-01 | Phase 2 | Complete (02-01) |
| TEST-02 | Phase 2 | Complete (02-01) |
| TEST-03 | Phase 2 | Complete (02-01) |
| TEST-04 | Phase 2 | Complete (02-01) |
| TEST-05 | Phase 2 | Complete (02-01) |
| TEST-06 | Phase 2 | Complete (02-02) |
| TEST-07 | Phase 2 | Complete (02-01) |
| TEST-08 | Phase 2 | Complete (02-02) |
| IMPL-01 | Phase 3 | Complete (03-01) |
| IMPL-02 | Phase 3 | Complete (03-01) |
| IMPL-03 | Phase 3 | Complete (03-01) |
| IMPL-04 | Phase 3 | Complete |
| ROLL-01 | Phase 4 | Complete |
| ROLL-02 | Phase 4 | Complete |
| ROLL-03 | Phase 4 | Complete |
| ROLL-04 | Phase 4 | Complete |
| ROLL-05 | Phase 4 | Complete |
| ROLL-06 | Phase 4 | Complete |
| ROLL-07 | Phase 4 | Complete |
| ROLL-08 | Phase 4 | Complete |
| ROLL-09 | Phase 4 | Complete |
| ROLL-10 | Phase 4 | Complete |
| MON-01 | Phase 5 | Complete (05-01, 05-02) |
| MON-02 | Phase 5 | Complete (05-01) |
| MON-03 | Phase 5 | Complete (05-01) |
| MON-04 | Phase 5 | Complete (05-01, 05-02) |
| MON-05 | Phase 5 | Complete (05-02) |
| MON-06 | Phase 5 | Complete (05-01) |
| DOC-01 | Phase 6 | Complete (06-01) |
| DOC-02 | Phase 6 | Complete (06-01) |
| DOC-03 | Phase 6 | Complete (06-02) |
| DOC-04 | Phase 6 | Complete (06-02) |
| DOC-05 | Phase 6 | Complete (06-02) |
| DOC-06 | Phase 6 | Complete (06-01) |
| CASE-01 | Phase 6 | Pending |
| CASE-02 | Phase 6 | Pending |
| CASE-03 | Phase 6 | Pending |
| CASE-04 | Phase 6 | Pending |
| CASE-05 | Phase 6 | Pending |
| CASE-06 | Phase 6 | Pending |

**Coverage:**
- v1 requirements: 37 total
- Mapped to phases: 37
- Unmapped: 0

---
*Requirements defined: 2026-02-27*
*Last updated: 2026-03-01 — DOC-01..DOC-06 complete: all internal docs, public docs-site page, skill references, and master_mermaid updated for AQUADIF*
