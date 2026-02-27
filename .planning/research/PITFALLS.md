# Pitfalls Research: AQUADIF Tool Taxonomy Metadata Refactor

**Domain:** Multi-agent bioinformatics platform tool taxonomy
**Researched:** 2026-02-27
**Confidence:** HIGH (based on codebase inspection + architecture analysis)

## Critical Pitfalls

### Pitfall 1: Tool Factory Closure Scoping Drift

**What goes wrong:**
When adding metadata assignment after `@tool` decoration inside factory closures, Python's late binding causes metadata to reference the wrong closure variables if not explicitly captured. All tools in a factory get the metadata from the LAST tool defined, not their own.

**Why it happens:**
Factory functions like `create_shared_tools()` define 10+ tool functions in a loop or sequence, each using `@tool` decorator. When you assign `tool.metadata = {...}` AFTER defining all tools, Python closures capture variables by reference, not value. The metadata assignment happens when the closure variable points to the last tool's context.

**Example of the failure:**
```python
def create_tools():
    tools = []

    @tool
    def quality_tool(): pass

    @tool
    def analyze_tool(): pass

    # WRONG: Both get ANALYZE metadata
    for t in [quality_tool, analyze_tool]:
        if "quality" in t.name:
            t.metadata = {"categories": ["QUALITY"]}
        else:
            t.metadata = {"categories": ["ANALYZE"]}

    # quality_tool.metadata ends up as ANALYZE if closure isn't careful
```

**How to avoid:**
- Assign metadata IMMEDIATELY after each `@tool` definition, not in batch
- Use explicit variable capture: `_tool = tool_func` before metadata assignment
- See `graph.py:222-225` for the exact pattern Lobster already uses for delegation tools

**Warning signs:**
- Contract tests pass for first agent, fail for second with "wrong category" errors
- Multiple tools in same factory share the same metadata dict ID (`id(tool.metadata)` is identical)
- Tools defined last in factory have correct metadata, earlier ones don't

**Phase to address:**
- **Phase 1 (Skill)**: Document the immediate-assignment pattern with examples
- **Phase 2 (Contract Tests)**: Add test that validates metadata uniqueness (`id(tool.metadata)` must differ across tools)
- **Phase 3 (Reference Impl)**: Apply pattern correctly in transcriptomics expert, create template

---

### Pitfall 2: LangChain Metadata Propagation Blackholes

**What goes wrong:**
`BaseTool.metadata` and `BaseTool.tags` propagate to callback handlers at tool invocation, BUT if tools are wrapped in certain LangGraph constructs (conditionals, subgraphs, retry decorators), metadata can be dropped or replaced by outer config. The callback handler receives empty/wrong metadata even though tools have it set.

**Why it happens:**
LangGraph's `create_react_agent()` wraps tools in execution layers. Each layer can override `RunnableConfig`. When a parent layer sets `config={"metadata": {...}}`, it REPLACES child metadata instead of merging. Graph delegation tools (created in `graph.py:228`) explicitly set config — this overrides tool metadata.

**Example from existing code:**
```python
# graph.py:254-258 — delegation tools set config
config = {
    "run_name": _name,
    "tags": [_name],  # This REPLACES tool.tags
    "metadata": {"agent_name": _name},  # This REPLACES tool.metadata
}
result = agent.invoke({...}, config=config)
```

**How to avoid:**
- Test metadata propagation end-to-end, not just tool definition
- In callback handler, check BOTH `tool.metadata` and `run.extra["metadata"]`
- For delegation tools in `graph.py`, MERGE tool metadata into config instead of replacing:
  ```python
  config = {
      "metadata": {**tool.metadata, "agent_name": _name},
      "tags": tool.tags + [_name],
  }
  ```

**Warning signs:**
- Contract tests pass (tool has metadata) but callback handler logs show empty
- Delegation tool categories never appear in monitoring
- Metadata works for direct tools, fails for child agent tools

**Phase to address:**
- **Phase 2 (Contract Tests)**: Add end-to-end test that validates metadata reaches callback handler
- **Phase 4 (Delegation Tools)**: Audit `graph.py` for config overwrites, implement merge pattern
- **Phase 5 (Monitoring Callback)**: Build defensive callback that checks multiple metadata sources

---

### Pitfall 3: Multi-Category Becomes Meaningless Noise

**What goes wrong:**
Without strict enforcement, developers add 2-3 categories to every tool "just in case," making categories worthless for filtering/routing. Tools with `["QUALITY", "ANALYZE", "FILTER"]` provide no signal. Category distribution flattens to ~80% of tools having 3 categories.

**Why it happens:**
- Ambiguous category boundaries (is batch detection QUALITY or PREPROCESS?)
- Fear of missing the "right" category leads to defensive over-tagging
- Contract tests allow up to 3, so developers use all 3
- No CODE REVIEW enforcement of category minimalism

**Real examples from transcriptomics dry run:**
| Tool | Proposed | Better |
|------|----------|--------|
| `evaluate_clustering_quality` | `["QUALITY", "ANALYZE"]` | `["QUALITY"]` (primary intent) |
| `check_data_status` | `["UTILITY", "QUALITY"]` | `["UTILITY"]` (it's read-only inspection) |
| `create_analysis_summary` | `["UTILITY", "SYNTHESIZE"]` | `["UTILITY"]` (SYNTHESIZE is aspirational) |

**How to avoid:**
- **Category minimalism rule**: Use 1 category unless the tool genuinely performs two DISTINCT operations in sequence
- **Bright-line test**: If you remove secondary category, does tool description become incomplete? If no → single category
- **Code review checklist**: Reviewer asks "Why 2 categories?" for any multi-category tool
- **Contract test addition**: Flag tools with 3 categories for manual review (not auto-fail, but logged)

**Warning signs:**
- Phase 4 (All Agents) produces >60% tools with 2+ categories
- New agent PRs default to 2 categories even for obvious single-purpose tools
- Category-based filtering returns 80% of tools regardless of filter

**Phase to address:**
- **Phase 1 (Skill)**: Emphasize "single category is normal, multi is exception"
- **Phase 2 (Contract Tests)**: Add metrics report: "X% of tools have 1 category" (target: >70%)
- **Phase 4 (All Agents)**: Code review enforcement, explicit rationale for multi-category in docstrings

---

### Pitfall 4: Provenance Flag Becomes Ignored Configuration

**What goes wrong:**
Developers set `provenance: True` but forget to actually call `log_tool_usage(..., ir=ir)` in the tool body. The metadata lies about provenance compliance. Runtime detection only happens if monitoring callback explicitly checks — but by then, data is already incomplete.

**Why it happens:**
- Metadata and implementation are decoupled (metadata at tool bottom, `log_tool_usage` buried 50 lines above)
- Copy-paste from existing tools propagates missing provenance
- No compile-time enforcement — Python can't validate "if provenance=True then must call X"
- Contract tests validate metadata format, not behavior

**Existing vulnerability:**
Lobster already has this issue. The 3-tuple service pattern requires calling `log_tool_usage(..., ir=ir)`, but there's no enforcement. Transcriptomics shared_tools.py calls it correctly (lines 270-272), but nothing prevents a new tool from skipping it.

**How to avoid:**
- **Phase 2 contract test addition**:
  ```python
  def test_provenance_tools_call_log_tool_usage(self):
      """Provenance-required tools must call data_manager.log_tool_usage with ir param."""
      # Use AST parsing to validate call exists in tool body
      # See implementation notes in Phase 2 spec
  ```
- **Linter rule**: Custom pylint/ruff check for provenance-required categories
- **Runtime callback**: Detect when tool with `provenance: True` completes without IR emission

**Warning signs:**
- `/pipeline export` generates notebooks with missing steps (gaps in provenance chain)
- New agent's contract tests pass but integration tests show incomplete IR
- DataManagerV2 tool usage logs show 0 IR for ANALYZE tools

**Phase to address:**
- **Phase 2 (Contract Tests)**: AST-based validation that `log_tool_usage` with `ir=` exists
- **Phase 5 (Monitoring Callback)**: Runtime flag: tool finished, provenance expected, IR missing

---

### Pitfall 5: Prompt Auto-Generation Produces Worse Tool Selection

**What goes wrong:**
Phase 6 replaces hardcoded `<Your_Tools>` prompt sections with auto-generated text from tool metadata. LLM tool selection accuracy DROPS because:
- Generated text loses semantic grouping cues from handcrafted prompts
- Category-based grouping doesn't match how LLM reasons about tool use
- Verbose metadata produces longer prompts → worse instruction following

**Why it happens:**
- Handcrafted prompts encode years of trial-and-error about what helps LLMs pick tools
- Auto-generation produces technically correct but pedagogically weak prompts
- No A/B testing before full rollout
- Prompt templates tuned for Claude may fail for other providers

**Example:**
```
# Handcrafted (good)
<QC_and_Filtering>
Use `assess_data_quality` FIRST to understand your data.
Then `filter_and_normalize` to prepare for clustering.
Never cluster unfiltered data.
</QC_and_Filtering>

# Auto-generated from metadata (worse)
QUALITY category tools:
- assess_data_quality: Run comprehensive quality control
- detect_doublets: Identify doublet cells

FILTER category tools:
- filter_and_normalize: Filter and normalize data
```

**How to avoid:**
- A/B test auto-generated prompts against hardcoded on a single agent BEFORE Phase 6 rollout
- Metrics: tool selection accuracy, task completion rate, user correction frequency
- Fallback: Keep handcrafted prompts, use metadata for SUPPLEMENTARY tool tables only
- Hybrid approach: Auto-generate tool list, manually curate workflow guidance

**Warning signs:**
- Agent starts selecting wrong tools after Phase 6 deployment
- User queries require more back-and-forth to get correct tool sequence
- LLM repeatedly uses DELEGATE when it should ANALYZE directly

**Phase to address:**
- **Phase 6 (Prompts)**: DO NOT auto-replace all prompts. Start with visualization_expert (simple agent, low risk)
- **Between Phase 6 and 7**: Run comparison on transcriptomics expert — 50 test queries, measure deltas
- **Decision gate**: Only proceed if auto-generated performs within 5% of handcrafted

---

### Pitfall 6: Category Drift Over Time (Maintenance Decay)

**What goes wrong:**
Six months post-refactor, new agents and tools are added without proper AQUADIF metadata because:
- Skill isn't loaded (new contributors skip onboarding)
- Contract tests are skipped ("just a quick fix")
- Category definitions evolve informally without updating skill
- Edge cases accumulate until taxonomy is inconsistent

**Why it happens:**
- No automated enforcement at PR merge time
- Skill distribution relies on manual install (`curl ... | bash`)
- Contributors work from older repo checkouts with outdated skills
- Category boundary decisions aren't documented beyond initial skill

**Example trajectory:**
```
v1.0 (Phase 7 complete): 100% AQUADIF compliance
v1.1 (+2 agents):        95% compliant (one agent skipped contract tests)
v1.3 (+5 agents):        78% compliant (new pattern emerges, contradicts skill)
v1.5 (+8 agents):        62% compliant (category drift is substantial)
```

**How to avoid:**
- **CI enforcement**: PR checks fail if contract tests not run
- **Pre-commit hook**: Validate all `@tool` definitions have metadata (can be local only)
- **Quarterly taxonomy review**: Check category distribution, identify drift, update skill
- **Skill versioning**: Track skill version in SKILL.md, reference in contract tests
- **Decision log**: Document edge cases in `skills/lobster-dev/references/aquadif-decisions.md`

**Warning signs:**
- GitHub PR descriptions say "skipped contract tests, will add later"
- New package doesn't inherit from `AgentContractTestMixin`
- Issue reports: "Tool X appears in wrong category in UI"
- Callback monitoring shows 15% tools with no category metadata

**Phase to address:**
- **Phase 1 (Skill)**: Include versioning in skill frontmatter
- **Phase 2 (Contract Tests)**: Make contract tests MANDATORY for `make test`
- **Phase 7 (Case Study)**: Document maintenance runbook in skill
- **Post-launch**: Set up quarterly taxonomy health check

---

## Technical Debt Patterns

Shortcuts that seem reasonable but create long-term problems.

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Batch metadata assignment in loop | Fewer lines of code | Closure scoping bugs (Pitfall 1) | Never |
| Skip contract tests for "simple" agents | Faster development | Inconsistent metadata, taxonomy drift | Never |
| Multi-category as default for ambiguous tools | Less decision-making | Category inflation (Pitfall 3) | Only if documented rationale in docstring |
| Manual tool list in prompts instead of auto-gen | Better initial accuracy | Maintenance burden, drift from metadata | Acceptable IF A/B test shows >10% accuracy delta |
| Copy-paste tool factories without metadata review | Fast new agent creation | Inherited incorrect categories | MVP only; must audit before first release |

---

## LangChain-Specific Gotchas

Common mistakes when working with LangChain's tool metadata system.

| Gotcha | What Goes Wrong | Correct Approach |
|--------|-----------------|------------------|
| Metadata assigned before `@tool` decoration | Assignment happens to function object, decorator creates NEW StructuredTool without metadata | Always assign AFTER decoration: `@tool\ndef x(): ...\nx.metadata = {...}` |
| Assuming `.tags` and `.metadata` always propagate | RunnableConfig overwrites can drop metadata (Pitfall 2) | Test propagation end-to-end, build defensive callback |
| Mutating shared metadata dict | Multiple tools reference same dict ID, mutations affect all | Each tool gets fresh dict: `tool.metadata = {"categories": ["X"].copy()}` |
| Metadata in tool docstring vs. `.metadata` field | LLM sees docstring, callbacks see `.metadata` — can diverge | Single source of truth: generate docstring FROM metadata |
| Using `tool.metadata` before schema extraction | LangChain `@tool` extracts schema via `inspect.signature`, metadata can interfere | Assign metadata in final step, after all other tool config |

---

## Integration Gotchas

Common mistakes when integrating AQUADIF with existing Lobster infrastructure.

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| Dynamic delegation tools (`graph.py`) | Metadata assigned at tool creation, but tools created in loop → closure issue | Use explicit capture pattern from `graph.py:222`: `_name = agent_name` before `@tool` |
| Entry point agent discovery | Metadata added AFTER entry point loaded, but discovery happens at import | Metadata must be in-line at tool definition, not added post-hoc by registry |
| Tool wrappers/decorators | Custom wrappers lose metadata because they return new function | Preserve metadata: `wrapper.metadata = original.metadata` |
| Cloud callback handlers | Metadata serialization for Omics-OS Cloud analytics | Ensure all metadata values are JSON-serializable (no functions, classes) |
| Multi-package coordination | Core defines categories, packages define tools — version skew | Pin `lobster-ai` core version in package dependencies |

---

## Performance Traps

Patterns that work at small scale but fail as usage grows.

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Per-tool metadata validation at runtime | Callback handler validates all metadata on every tool call → 5-10ms overhead per call | Validate at factory creation, cache result | >1000 tool calls/session |
| Category-based filtering with list comprehension | `[t for t in tools if "ANALYZE" in t.metadata["categories"]]` re-scans all tools | Build category→tools index once at graph creation | >50 tools in graph |
| Dynamic prompt generation on every LLM call | Regenerating tool list from metadata adds 50-100ms | Generate once at agent creation, cache as string | Every message in conversation |
| Metadata propagation through deep delegation | 4-level delegation chains (supervisor→parent→child→grandchild) lose metadata | Flatten hierarchy OR implement metadata stack merge | >3 delegation levels |
| AST-based contract test for provenance calls | Parsing 180 tool bodies at test time → 2-3 second overhead | Lazy evaluation, cache parsing results | 10+ packages in monorepo |

---

## Category Boundary Ambiguities

Specific decision guidance for tools that span multiple categories.

| Ambiguous Tool Pattern | Primary Category | Rationale |
|------------------------|------------------|-----------|
| QC metrics that filter data | `QUALITY` | Primary intent is assessment; filtering is side effect |
| Batch correction that integrates datasets | `PREPROCESS` | Transformation is primary; integration is context |
| Feature selection during dimensionality reduction | `PREPROCESS` | Representation change is primary |
| Clustering quality evaluation | `QUALITY` | Evaluating existing result, not creating new analysis |
| Marker gene finding | `ANALYZE` | Generating new statistical result |
| Cell type annotation (automated) | `ANNOTATE` | Adding biological meaning |
| Cell type annotation (manual) | `ANNOTATE` | Same operation, different input source |
| Differential expression with pathway enrichment | `ANALYZE` | Both are analytical computation; GSEA is secondary |
| Import + automatic QC | Split into 2 tools | Separate concerns, enable modular workflows |
| Status check with conditional filtering | `UTILITY` | Read-only inspection; mention filtering in description |

**Rule of thumb**: Ask "What is the DELIVERABLE?" That's the primary category. Secondary operations are implementation details.

---

## "Looks Done But Isn't" Checklist

Things that appear complete but are missing critical pieces.

- [ ] **Metadata added to all tools**: Verify shared_tools.py AND agent-specific files (often missed)
- [ ] **Delegation tools in graph.py tagged**: Dynamic tools need manual metadata in factory
- [ ] **Contract tests passing**: Agents without test file need new test class created
- [ ] **Prompt uses metadata**: Check prompts.py actually imports/calls auto-gen function
- [ ] **Callback handler wired into graph**: Handler registered in `create_bioinformatics_graph`, not just defined
- [ ] **Provenance validation works**: Test that missing IR triggers detection (don't just check happy path)
- [ ] **Multi-category has rationale**: Docstring explains why 2+ categories, not just default
- [ ] **Skill references updated**: MANIFEST has aquadif-contract.md, creating-agents.md imports updated
- [ ] **Category enum imported**: Tools reference `AquadifCategory.ANALYZE`, not magic strings
- [ ] **CI includes contract tests**: `make test` fails if metadata missing, not skipped

---

## Recovery Strategies

When pitfalls occur despite prevention, how to recover.

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Closure scoping breaks metadata (Pitfall 1) | MEDIUM (1-2 days) | 1. Identify affected factory. 2. Refactor to immediate assignment. 3. Add contract test for metadata uniqueness. 4. Re-run all agent tests. |
| Metadata doesn't reach callback (Pitfall 2) | MEDIUM (2-3 days) | 1. Add logging at tool invoke. 2. Trace config overwrites. 3. Implement merge pattern in graph.py. 4. Validate end-to-end propagation. |
| Category inflation (Pitfall 3) | HIGH (1-2 weeks) | 1. Audit all tools, justify multi-category. 2. Remove speculative categories. 3. Update skill with bright-line rules. 4. Add code review gate. |
| Missing provenance calls (Pitfall 4) | MEDIUM (1 week) | 1. AST scan all tools. 2. Add missing `log_tool_usage` calls. 3. Backfill IR for affected sessions. 4. Add runtime detection callback. |
| Auto-gen prompts degrade accuracy (Pitfall 5) | LOW (1-2 days) | 1. Rollback to handcrafted prompts. 2. Refine generation template. 3. Re-test on single agent. 4. Hybrid approach if needed. |
| Taxonomy drift over time (Pitfall 6) | HIGH (2-3 weeks) | 1. Full audit of all agents. 2. Category realignment. 3. Update skill. 4. Mandatory CI enforcement going forward. |

**Prevention is cheaper than recovery for ALL pitfalls.** Phase 1 (Skill) and Phase 2 (Contract Tests) are critical investment.

---

## Pitfall-to-Phase Mapping

How roadmap phases should address these pitfalls.

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| **Closure scoping drift (Pitfall 1)** | Phase 1 (Skill), Phase 2 (Tests), Phase 3 (Reference) | Metadata uniqueness test passes for all tools |
| **Metadata propagation blackholes (Pitfall 2)** | Phase 2 (Tests), Phase 5 (Monitoring) | End-to-end callback test receives correct metadata |
| **Multi-category noise (Pitfall 3)** | Phase 1 (Skill), Phase 4 (All Agents) | <40% of tools have 2+ categories |
| **Ignored provenance flag (Pitfall 4)** | Phase 2 (Tests), Phase 5 (Monitoring) | AST test + runtime detection both implemented |
| **Prompt auto-gen regression (Pitfall 5)** | Phase 6 (Prompts) | A/B test shows <5% accuracy delta |
| **Category drift (Pitfall 6)** | Phase 1 (Skill), Phase 2 (Tests), Phase 7 (Case Study) | CI enforces contract tests, skill has maintenance runbook |

**Critical path dependencies:**
- Pitfall 1 MUST be prevented in Phase 1-3 (reference implementation sets pattern)
- Pitfall 2 and 4 require Phase 5 monitoring callback (can't validate without runtime)
- Pitfall 3 and 6 require ongoing enforcement (not one-time fix)

---

## Phase-Specific Warnings

| Phase | Likely Pitfall | Mitigation Strategy |
|-------|---------------|---------------------|
| **Phase 1 (Skill)** | Skill too abstract, no concrete examples | Include full agent example with 15+ tools properly categorized |
| **Phase 2 (Contract Tests)** | Tests validate format but not behavior | Add AST-based validation for provenance calls |
| **Phase 3 (Reference Impl)** | Reference has subtle bugs, 16 agents copy them | External review of transcriptomics metadata before Phase 4 |
| **Phase 4 (All Agents)** | Batch assignment used for speed, breaks closure scoping | Code review EVERY factory for immediate assignment pattern |
| **Phase 5 (Monitoring)** | Callback too verbose, logs fill disk | Rate-limit category distribution logging to 1/minute |
| **Phase 6 (Prompts)** | Auto-gen rolled out without testing | REQUIRE A/B test on 2 agents before full rollout |
| **Phase 7 (Case Study)** | Epigenomics too complex, hides skill weaknesses | Start with metabolomics (simpler) if case study fails |

---

## Validation Checklist (Before Phase Completion)

Each phase should pass these gates before proceeding:

**Phase 1 (Skill) Complete When:**
- [ ] Skill loaded by Claude Code produces correct tool categorization without additional prompting
- [ ] Skill includes troubleshooting for all 6 critical pitfalls
- [ ] Example agent in skill uses immediate assignment pattern for metadata

**Phase 2 (Contract Tests) Complete When:**
- [ ] All 5 core tests implemented and passing on mock agent
- [ ] Metadata uniqueness test catches closure scoping bug
- [ ] AST-based provenance validation detects missing `log_tool_usage`

**Phase 3 (Reference Impl) Complete When:**
- [ ] All 24 transcriptomics tools have metadata
- [ ] Contract tests pass for lobster-transcriptomics package
- [ ] External review confirms no Pitfall 1 or 3 issues

**Phase 4 (All Agents) Complete When:**
- [ ] All 180 tools have metadata
- [ ] Contract tests pass for all 9 packages
- [ ] Category distribution: <40% multi-category, <5% with 3 categories

**Phase 5 (Monitoring) Complete When:**
- [ ] Callback receives metadata for direct tools AND delegation tools
- [ ] Missing provenance detection works at runtime
- [ ] Category distribution logged per session

**Phase 6 (Prompts) Complete When:**
- [ ] A/B test on 2 agents shows <5% accuracy delta
- [ ] Handcrafted prompts kept for agents with complex workflows
- [ ] Auto-gen used for simple agents (visualization, utility)

**Phase 7 (Case Study) Complete When:**
- [ ] Coding agent builds epigenomics package following skill
- [ ] Contract tests pass on first attempt (or documented why not)
- [ ] Metrics collected for paper (time, LOC, correction cycles)

---

## Sources

- **Lobster AI codebase inspection**:
  - `lobster/agents/graph.py` (lines 222-225: closure capture pattern)
  - `packages/lobster-transcriptomics/lobster/agents/transcriptomics/shared_tools.py` (lines 270-272: provenance pattern)
  - `lobster/testing/contract_mixins.py` (existing contract test infrastructure)

- **AQUADIF refactor plan**: `.planning/QALITA_refactor/gsd_project_aquadif_refactor_plan.md`

- **Architecture docs**:
  - `.claude/docs/development-rules.md` (3-tuple service pattern, provenance requirements)
  - `.claude/docs/agent_architecture_guidelines.md` (tool factory patterns)

- **LangChain documentation**:
  - `langchain_core/tools/base.py:480-497` (native metadata/tags fields)
  - RunnableConfig behavior (metadata overwriting in nested calls)

- **Domain expertise**:
  - Multi-agent system metadata consistency challenges
  - Python closure scoping behavior
  - Tool taxonomy evolution in production systems

---

*Pitfalls research for: AQUADIF Tool Taxonomy Metadata Refactor*
*Researched: 2026-02-27*
*Next: Use this document to inform Phase 1 (Skill creation) and Phase 2 (Contract test spec)*
