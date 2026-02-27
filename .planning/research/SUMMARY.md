# Project Research Summary

**Project:** AQUADIF Tool Taxonomy Refactor
**Domain:** Multi-agent bioinformatics platform tool metadata system
**Researched:** 2026-02-27
**Confidence:** HIGH

## Executive Summary

AQUADIF is a structured 10-category taxonomy for organizing ~180 tool bindings across 18 specialist agents in Lobster AI. The research validates that this refactor should leverage native LangChain `BaseTool.metadata` and `.tags` fields rather than custom decorators or registries. These fields propagate automatically to callback handlers, enabling runtime monitoring, provenance enforcement, and prompt auto-generation without modifying core LangGraph execution flow.

The recommended approach is skill-first validation: create the lobster-dev AQUADIF skill BEFORE touching any code to validate that the pattern is teachable to coding agents. If Phase 1 (skill creation) fails — meaning Claude Code or similar agents cannot follow the instructions to correctly categorize tools — the approach needs simplification before proceeding. This front-loads risk and prevents a large-scale refactor based on a flawed abstraction. Once teachability is proven, systematic rollout follows: contract tests define correctness, reference implementation on transcriptomics_expert validates the pattern, then all 180 tools get tagged with metadata co-located at their definitions.

Key risks center on metadata propagation blackholes (LangGraph config overwrites), closure scoping bugs in factory functions, and category inflation from over-tagging. Mitigation involves end-to-end contract tests, immediate post-`@tool` metadata assignment, category minimalism enforcement via code review, and A/B testing before auto-generating prompts from metadata. The 7-phase roadmap front-loads validation (skill → tests → reference) before full implementation, with continuous enforcement preventing taxonomy drift over time.

## Key Findings

### Recommended Stack

The stack is intentionally minimal because LangChain already provides what's needed. No new dependencies are required — this is about using existing infrastructure correctly.

**Core technologies:**
- **langchain-core ≥0.3.79**: Native `BaseTool.metadata` and `.tags` fields with callback propagation — this is the foundation, zero additional libraries needed
- **langgraph ≥1.0.5**: Multi-agent orchestration already in use, compatible with metadata propagation through graph execution
- **pytest ≥7.0.0**: Contract test validation with fixtures for metadata schemas and parametrized tests for multi-category validation

**Supporting libraries:**
- **pydantic ≥2.0**: For `AquadifCategory` enum and metadata structure validation (already a LangChain dependency)
- **Langfuse ≥3.2.6** (optional): Self-hosted observability alternative if LangSmith not desired, already in optional dependencies

**Why native fields over custom decorators:** Decorator stacking causes signature inspection conflicts, metadata hidden in decorator stack isn't discoverable via `tool.__dict__`, and custom patterns require domain-specific documentation that AI agents can't generalize. Native LangChain fields are AI-teachable, propagate to callbacks automatically, and have been stable since langchain-core 0.3.x (2024-2025).

### Expected Features

**Must have (table stakes):**
- Category assignment — tools declare what they do via primary category from 10-option taxonomy
- Metadata co-location — metadata lives WITH tool definition to avoid drift from implementation
- Runtime introspection — system can query "what tools exist in category X?" for monitoring and filtering
- Multi-category support — real tools span categories (up to 3, first is primary)
- Contract tests — enforce taxonomy compliance at test time to prevent regression
- Backward compatibility — adding metadata must not break existing tool execution

**Should have (differentiators):**
- Provenance enforcement — 7 categories require `provenance: True` and must produce W3C-PROV IR, validated at runtime via callback
- Prompt auto-generation — agent prompts generate `<Your_Tools>` sections from metadata, eliminating drift from hardcoded lists
- AI self-extension recipe — coding agents (Claude Code, Codex, Gemini CLI) can follow AQUADIF skill to build new domain packages with correct categorization (killer publication feature)
- Honest gap documentation — SYNTHESIZE category exists with 0 implementations, publication admits limitation rather than forcing bad implementations
- Category-based monitoring — runtime callback tracks category distribution per session, flags anomalies (e.g., excessive CODE_EXEC usage)

**Defer (v2+):**
- Cross-session analytics — aggregate category usage across Omics-OS Cloud users to identify patterns
- Tool recommendation engine — suggest related tools based on category co-occurrence
- Auto-generated documentation — API docs grouped by category with usage examples
- Category-based access control — enterprise deployments may restrict CODE_EXEC to admin users

### Architecture Approach

AQUADIF integrates with existing LangChain/LangGraph architecture through native metadata propagation. Metadata flows FROM tool definition → THROUGH LangGraph → TO callback handlers as first-class citizens of execution context. This enables runtime monitoring, provenance enforcement, and prompt generation without bypassing or extending LangGraph's callback system.

**Major components:**
1. **AQUADIF Configuration** (`lobster/config/aquadif.py`) — NEW file with `AquadifCategory` enum and `PROVENANCE_REQUIRED` set (pure configuration, no runtime logic)
2. **Agent Factory Metadata Assignment** — MODIFIED files across 9 packages, metadata assigned after `@tool` decorator but before agent creation in factory closures
3. **Delegation Tool Metadata** (`lobster/agents/graph.py`) — MODIFIED function `_create_lazy_delegation_tool()` assigns DELEGATE metadata to dynamic handoff tools
4. **AQUADIF Callback Handler** (`lobster/core/aquadif_callback.py`) — NEW file, runtime monitoring via LangChain callbacks (logs categories, checks provenance compliance, tracks CODE_EXEC usage)
5. **Contract Test Mixins** (`lobster/testing/contract_mixins.py`) — MODIFIED class with 5 new test methods for AQUADIF validation
6. **Prompt Auto-Generation** (`lobster/agents/prompts.py` per package) — MODIFIED utilities, generates tool sections from metadata (Phase 6 work)

**Critical architectural insight:** LangChain's `RunnableConfig` propagates `metadata` and `tags` from tool definitions to callback handlers at every invocation. No custom propagation logic needed. Metadata assignment MUST happen after `@tool` creation but before agent creation to avoid schema extraction conflicts.

### Critical Pitfalls

1. **Tool Factory Closure Scoping Drift** — Python late binding causes all tools in a factory to share metadata from the LAST tool defined if assignment happens in batch. Prevention: assign metadata IMMEDIATELY after each `@tool` definition (not in loop), contract tests validate metadata uniqueness via `id(tool.metadata)`.

2. **LangChain Metadata Propagation Blackholes** — `BaseTool.metadata` propagates to callbacks, BUT LangGraph constructs (conditionals, subgraphs) can override via `RunnableConfig`, dropping tool metadata. Prevention: test propagation end-to-end (not just tool definition), implement merge pattern in `graph.py` delegation tools, build defensive callback that checks multiple metadata sources.

3. **Multi-Category Becomes Meaningless Noise** — Without enforcement, developers add 2-3 categories to every tool "just in case," flattening distribution to ~80% multi-category. Prevention: category minimalism rule (use 1 category unless tool genuinely performs two DISTINCT operations), bright-line test in code review ("if you remove secondary category, does description become incomplete?"), contract tests flag tools with 3 categories for manual review.

4. **Provenance Flag Becomes Ignored Configuration** — Developers set `provenance: True` but forget to call `log_tool_usage(..., ir=ir)`, metadata lies about compliance. Prevention: AST-based contract test validates call exists in tool body, runtime callback detects tool completion without IR emission.

5. **Prompt Auto-Generation Produces Worse Tool Selection** — Phase 6 replaces handcrafted prompts with auto-generated text from metadata, LLM tool selection accuracy drops because generated text loses semantic grouping cues. Prevention: A/B test auto-generated vs handcrafted on single agent BEFORE full rollout, keep handcrafted for complex agents, use hybrid approach (auto-generate list, manual workflow guidance).

6. **Category Drift Over Time** — Six months post-refactor, new agents lack proper metadata because skill not loaded, contract tests skipped, category definitions evolve informally. Prevention: CI enforcement (PR fails if contract tests not run), pre-commit hooks, quarterly taxonomy review, skill versioning, decision log for edge cases.

## Implications for Roadmap

Based on research, suggested 7-phase structure with front-loaded validation:

### Phase 1: AQUADIF Skill Creation (lobster-dev update)
**Rationale:** Validate teachability BEFORE touching any code. If coding agents can't follow the skill to correctly categorize tools, the pattern needs simplification. This prevents large-scale refactor based on flawed abstraction.
**Delivers:** Updated `skills/lobster-dev/` with AQUADIF contract specification, category definitions, assignment pattern examples, pitfall troubleshooting.
**Addresses:** AI self-extension recipe (differentiator), skill-first development pattern
**Avoids:** Building 7 phases of work on an approach that isn't AI-teachable (Pitfall validation)
**Research Flags:** SKIP RESEARCH — skill format follows Anthropic Skills Guide, pattern is LangChain native

### Phase 2: Contract Test Specification
**Rationale:** Define "correct" before implementing. Tests encode taxonomy rules as enforceable constraints. Without validation, no way to know if metadata is right.
**Delivers:** `lobster/config/aquadif.py` (enum + provenance set), updated `lobster/testing/contract_mixins.py` (5 new test methods: metadata presence, category validity, cap at 3, provenance compliance, minimum viable parent)
**Uses:** pytest fixtures, parametrized tests, AST parsing for provenance call validation
**Implements:** Contract test component boundary
**Addresses:** Validation at test time (table stakes), provenance enforcement foundation
**Avoids:** Pitfall 1 (metadata uniqueness test), Pitfall 3 (category distribution metrics), Pitfall 4 (AST validation)
**Research Flags:** SKIP RESEARCH — pytest patterns are standard, AST parsing well-documented

### Phase 3: Reference Implementation (Transcriptomics)
**Rationale:** Validate pattern on most mature agent (24 tools) before rolling out to 156 additional tools. Serves as reference for Phase 4. Catches pattern issues on manageable scale.
**Delivers:** All 24 tools in `transcriptomics_expert.py` have metadata, contract tests pass, external review confirms no scoping/over-tagging issues
**Uses:** Post-decorator assignment pattern, immediate metadata capture per tool
**Implements:** Agent factory metadata assignment component boundary
**Addresses:** Metadata co-location (table stakes), multi-category support (table stakes)
**Avoids:** Pitfall 1 (immediate assignment prevents closure bugs), Pitfall 3 (reference sets category minimalism example)
**Research Flags:** SKIP RESEARCH — pattern proven in dry run, just systematic application

### Phase 4: Roll Out to All Agents (16 remaining agents)
**Rationale:** Systematic application of validated reference pattern. Contract tests validate each agent. Achieves full system coverage (~156 additional tools).
**Delivers:** All 180 tools across 18 agents have metadata, contract tests pass for all 9 packages, category distribution measured (<40% multi-category target)
**Uses:** Reference implementation template, code review enforcement
**Implements:** Complete agent factory metadata assignment
**Addresses:** Runtime introspection (table stakes), backward compatibility (table stakes)
**Avoids:** Pitfall 1 (code review checks immediate assignment), Pitfall 3 (review justifies multi-category), Pitfall 6 (CI enforcement starts here)
**Research Flags:** SKIP RESEARCH — repetitive application of Phase 3 pattern

### Phase 5: Monitoring Callback Handler (parallel with Phase 6)
**Rationale:** Runtime monitoring infrastructure requires all tools to have metadata (dependency on Phase 4). Can be built in parallel with Phase 6 since both consume metadata.
**Delivers:** `lobster/core/aquadif_callback.py` with `AquadifCallbackHandler` class (logs categories, checks provenance compliance, tracks CODE_EXEC usage, emits structured events for Omics-OS Cloud)
**Uses:** LangChain `BaseCallbackHandler.on_tool_start()`, `pending_provenance_checks` state tracking
**Implements:** AQUADIF callback handler component boundary, provenance compliance check flow
**Addresses:** Callback propagation (differentiator), category-based monitoring (differentiator), provenance enforcement runtime validation
**Avoids:** Pitfall 2 (defensive callback checks multiple metadata sources), Pitfall 4 (runtime detection of missing IR)
**Research Flags:** SKIP RESEARCH — LangChain callback API is stable, pattern clear from architecture

### Phase 6: Prompt Auto-Generation (parallel with Phase 5)
**Rationale:** Eliminate hardcoded tool lists, prompts stay in sync with tool changes. Requires all tools to have metadata (dependency on Phase 4). High risk of accuracy regression.
**Delivers:** `generate_tool_prompt_section()` utility, A/B test on 2 agents (visualization_expert + one complex agent), updated prompts for agents where auto-gen performs within 5% of handcrafted
**Uses:** Category-based prompt generation pattern, tool grouping by primary category
**Implements:** Prompt auto-generation component boundary
**Addresses:** Prompt auto-generation (differentiator), domain-agnostic taxonomy validation across omics + non-omics agents
**Avoids:** Pitfall 5 (A/B test REQUIRED before rollout, hybrid approach if needed)
**Research Flags:** NEED RESEARCH — A/B test methodology for LLM tool selection accuracy, metrics for task completion rate

### Phase 7: Extension Case Study (Epigenomics)
**Rationale:** Publication evidence for AI self-extension. Requires everything working end-to-end (Phases 1-6 complete). Validates teachability under realistic conditions (new domain package from scratch).
**Delivers:** Epigenomics package built by coding agent following AQUADIF skill, metrics collected (time to passing contract tests, LOC generated vs edited, correction cycles, first-attempt pass rate), documentation in skill maintenance runbook
**Uses:** Complete AQUADIF infrastructure, lobster-dev skill, contract tests
**Addresses:** AI self-extension recipe (killer differentiator), skill-first validation proof, honest gap documentation (show SYNTHESIZE remains empty)
**Avoids:** Pitfall 6 (maintenance runbook documents ongoing enforcement, skill versioning established)
**Research Flags:** SKIP RESEARCH — case study validates existing skill, no new patterns

### Phase Ordering Rationale

- **Skill-first (Phase 1):** Front-loads risk. If approach isn't teachable, we know before touching 180 tools. Failure triggers simplification, not wasted implementation work.
- **Tests before implementation (Phase 2 before 3-4):** Defines correctness before coding. Contract tests catch pitfalls (closure scoping, category inflation) at development time.
- **Reference before rollout (Phase 3 before 4):** Validates pattern on manageable scale (24 tools), catches issues before applying to 156 additional tools.
- **Full coverage before consumption (Phase 4 before 5-6):** Monitoring and prompt generation require all tools to have metadata. Parallel phases 5-6 both depend on completed Phase 4.
- **Case study last (Phase 7):** Requires entire system working. Validates teachability under realistic conditions, not controlled examples.

**Dependency chain:**
```
Phase 1 (Skill)
    └── Phase 2 (Tests) — skill defines what tests validate
            └── Phase 3 (Reference) — tests validate reference
                    └── Phase 4 (Rollout) — reference pattern applied to all
                            ├── Phase 5 (Monitoring) — consumes metadata
                            └── Phase 6 (Prompts) — consumes metadata
                                    └── Phase 7 (Case Study) — end-to-end validation
```

### Research Flags

**Phases with standard patterns (skip research-phase):**
- **Phase 1:** Anthropic Skills Guide provides format, LangChain native pattern well-documented
- **Phase 2:** pytest patterns standard, AST parsing well-documented
- **Phase 3:** Reference implementation applies proven pattern from dry run
- **Phase 4:** Repetitive application of Phase 3 pattern
- **Phase 5:** LangChain callback API stable, architecture clear
- **Phase 7:** Validates existing skill, no new patterns

**Phase needing deeper research:**
- **Phase 6:** A/B test methodology for LLM tool selection accuracy (metrics: task completion rate, tool selection accuracy, user correction frequency, prompt token budget). Need research on: evaluation harness design, statistical significance thresholds, hybrid prompt design if auto-gen underperforms.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Native LangChain fields verified via source code inspection (`langchain_core/tools/base.py`), callback propagation validated empirically, no external dependencies needed |
| Features | MEDIUM | High confidence on Lobster-specific features (codebase inspection), medium confidence on ecosystem comparison (no recent external research, Brave API unavailable), but AQUADIF solves real internal problems regardless |
| Architecture | HIGH | LangChain/LangGraph integration verified via codebase inspection, callback propagation tested, component boundaries clear, build order validated against existing patterns |
| Pitfalls | HIGH | All pitfalls grounded in codebase inspection (closure patterns in `graph.py`, existing provenance calls), Python closure behavior well-understood, recovery strategies practical |

**Overall confidence:** HIGH

Research is grounded in direct codebase inspection, LangChain source code verification, and empirical testing. External ecosystem comparison is medium confidence but not blocking — AQUADIF addresses real needs in existing Lobster system (provenance enforcement, category-based monitoring, AI self-extension) regardless of what other frameworks do.

### Gaps to Address

**Skill teachability validation (Phase 1):**
- **Gap:** Unknown if coding agents can follow AQUADIF skill to correctly categorize tools for new domain (epigenomics)
- **Resolution:** Phase 1 includes dry run with Claude Code — provide skill, ask agent to design epigenomics tool set with categories, validate output manually. If agent fails to produce correct categories without additional prompting, simplify skill before proceeding to Phase 2.

**Metadata propagation through delegation chains (Phase 2):**
- **Gap:** LangGraph documentation doesn't explicitly address metadata survival through nested agent invocations (supervisor → parent → child)
- **Resolution:** Phase 2 contract tests include end-to-end propagation test (invoke tool via supervisor, verify callback receives metadata). If metadata lost, implement merge pattern in `graph.py` config construction.

**Multi-category usage patterns (Phase 4):**
- **Gap:** Will most tools be single-category (simple) or multi-category (complex)? Transcriptomics dry run shows 87.5% single-category, but that's one agent.
- **Resolution:** Phase 4 measures actual distribution across all agents. If >50% become multi-category, category inflation is occurring — trigger code review enforcement and skill clarification.

**Prompt auto-generation quality (Phase 6):**
- **Gap:** Will LLM tool selection improve, degrade, or stay neutral with auto-generated vs handcrafted prompts?
- **Resolution:** Phase 6 REQUIRES A/B test on 2 agents before full rollout. Metrics: tool selection accuracy, task completion rate. If auto-gen underperforms by >5%, use hybrid approach (auto-generate list, manual workflow guidance) or keep handcrafted for complex agents.

**CODE_EXEC usage frequency (Phase 5):**
- **Gap:** How often will agents use CODE_EXEC escape hatch vs typed tools? If >20%, taxonomy has limited coverage.
- **Resolution:** Phase 5 monitoring tracks CODE_EXEC frequency per session. If exceeds threshold, investigate why agents bypass typed tools — may indicate missing categories or tools.

## Sources

### Primary (HIGH confidence)
- **Lobster AI codebase** — direct inspection of `lobster/agents/graph.py` (closure patterns, delegation tools), `lobster/core/data_manager_v2.py` (provenance tracking), `lobster/testing/contract_mixins.py` (existing test infrastructure), agent factories across 9 packages
- **LangChain core source code** — `langchain_core/tools/base.py` lines 480-497 (BaseTool.metadata and .tags fields), `langchain_core/callbacks/base.py` (on_tool_start signature)
- **Empirical testing** — metadata propagation to callback handlers verified via Python test script
- **AQUADIF planning docs** — brainstorming handoff, tool_meta research, refactor plan, dry run results (transcriptomics expert)
- **Anthropic Skills Guide** — `skills/anthropic_skills_guide.md` for skill format and progressive disclosure pattern

### Secondary (MEDIUM confidence)
- **LangChain callback documentation** — redirected during research, relied on source code inspection instead
- **pytest documentation** — fixtures and parametrization best practices for 2025/2026
- **Multi-agent system patterns** — BDI/FIPA from training data (LOW) + Lobster ecosystem observation (HIGH)

### Tertiary (LOW confidence, needs validation)
- **LangGraph callback propagation edge cases** — assumed metadata survives multi-agent handoffs, needs Phase 2 integration test validation
- **Function calling standards** — OpenAI/Anthropic tool schemas from training data, not verified with recent API docs
- **Langfuse vs LangSmith tradeoffs** — both mentioned in docs but no head-to-head comparison found

---
*Research completed: 2026-02-27*
*Ready for roadmap: yes*
