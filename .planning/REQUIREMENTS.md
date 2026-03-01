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
- [ ] **ROLL-02**: `proteomics_expert` (21 tools), `proteomics_de_analysis_expert` (7 tools), and `biomarker_discovery_expert` (7 tools) have metadata and pass contract tests
- [x] **ROLL-03**: `metabolomics_expert` (10 tools) has metadata and passes contract tests
- [x] **ROLL-04**: `annotation_expert` (10 tools) and `de_analysis_expert` (14 tools) have metadata and pass contract tests
- [x] **ROLL-05**: `machine_learning_expert` (9 tools), `feature_selection_expert` (6 tools), and `survival_analysis_expert` (6 tools) have metadata and pass contract tests
- [ ] **ROLL-06**: `research_agent` (13 tools) and `data_expert` (10 tools) have metadata and pass contract tests
- [x] **ROLL-07**: `visualization_expert` (11 tools), `metadata_assistant` (~8 tools), and `protein_structure_visualization_expert` (~4 tools) have metadata and pass contract tests
- [x] **ROLL-08**: Dynamic DELEGATE tools in `graph.py` (`_create_lazy_delegation_tool`) have DELEGATE metadata
- [ ] **ROLL-09**: All ~180 tools across 18 agents pass contract tests; multi-category usage is <40%
- [ ] **ROLL-10**: `drug_discovery_expert` (10 shared tools), `cheminformatics_expert` (9 tools), `clinical_dev_expert` (8 tools), and `pharmacogenomics_expert` (8 tools) have metadata and pass contract tests

### Monitoring

- [ ] **MON-01**: `AquadifCallbackHandler` in `lobster/core/aquadif_callback.py` logs category with every tool invocation
- [ ] **MON-02**: Callback handler tracks category distribution per session
- [ ] **MON-03**: Callback handler flags CODE_EXEC usage with details
- [ ] **MON-04**: Callback handler checks provenance compliance at runtime (was `log_tool_usage` called for provenance-required tools?)
- [ ] **MON-05**: Callback handler integrated with existing callback infrastructure in `graph.py`
- [ ] **MON-06**: Callback handler emits structured events consumable by Omics-OS Cloud

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
| ROLL-02 | Phase 4 | Pending |
| ROLL-03 | Phase 4 | Complete |
| ROLL-04 | Phase 4 | Complete |
| ROLL-05 | Phase 4 | Complete |
| ROLL-06 | Phase 4 | Pending |
| ROLL-07 | Phase 4 | Complete |
| ROLL-08 | Phase 4 | Complete |
| ROLL-09 | Phase 4 | Pending |
| ROLL-10 | Phase 4 | Pending |
| MON-01 | Phase 5 | Pending |
| MON-02 | Phase 5 | Pending |
| MON-03 | Phase 5 | Pending |
| MON-04 | Phase 5 | Pending |
| MON-05 | Phase 5 | Pending |
| MON-06 | Phase 5 | Pending |
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
*Last updated: 2026-02-28 — Fixed IMPL-01 tool count: 24 -> 15 (verified: 15 @tool in transcriptomics_expert.py, 7 in shared_tools.py = 22 total)*
