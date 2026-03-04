# Publication Requirements: AQUADIF Skill-Guided Self-Extension Evaluation

> Reproducible experimental pipeline for validating Lobster AI's self-extending nature.
> Target venue: NeurIPS 2025 Datasets & Benchmarks track.
> Last updated: 2026-03-01

---

## 1. Central Thesis

**Claim:** A structured development skill (lobster-dev) encoding a 10-category tool taxonomy (AQUADIF) enables state-of-the-art AI coding agents to scaffold publication-quality domain packages for a multi-agent bioinformatics system — and the resulting packages enable accurate task routing on small, locally-deployable models at zero marginal inference cost.

**Three testable sub-claims:**

| # | Sub-Claim | Evidence Required | Paper Section |
|---|-----------|-------------------|---------------|
| C1 | The skill reliably teaches AQUADIF metadata annotation | WITH vs WITHOUT comparison across 3 agents | Sec 07.4 |
| C2 | Iterative skill refinement measurably improves agent output quality | Longitudinal iter-02 → iter-05 progression | Sec 05.5 |
| C3 | Skill-built packages enable accurate routing on small local models | Model size scaling experiment (frontier → 8B) | Sec 07.5 |

---

## 2. Research Questions & Hypotheses

### RQ1: Skill Teaching Effect (C1)
**Question:** Does the AQUADIF skill improve the structural quality of AI-generated domain packages compared to a no-skill baseline?

**H1a (Metadata):** Agents in the WITH-skill condition will produce tools with AQUADIF metadata annotations at a rate ≥90%, while WITHOUT-skill agents will produce ≤10%.

**H1b (Category Accuracy):** WITH-skill agents will assign tools to correct AQUADIF categories at ≥80% accuracy (measured against expert ground truth), while WITHOUT-skill agents will not produce recognizable category assignments.

**H1c (Provenance):** WITH-skill agents will produce provenance-compliant tools (calling `log_tool_usage(ir=ir)` in provenance-required categories) at ≥80%, while WITHOUT-skill agents will produce ≤5%.

**H1d (Structural Compliance):** WITH-skill packages will pass ≥75% of the 8-point `validate_plugin.py` structural checks; WITHOUT-skill packages will pass ≤25%.

### RQ2: Cross-Agent Generalization (C1)
**Question:** Does the skill teaching effect transfer across fundamentally different AI coding agents?

**H2a:** The skill teaching effect (WITH > WITHOUT) will be statistically significant for each of the three agents independently (Claude Opus 4.6, Gemini 3.1 Pro, Codex 5.3).

**H2b:** Agent "personality" (depth vs breadth vs speed) will produce measurably different output profiles despite identical skill input — but all will pass minimum quality thresholds.

### RQ3: Cross-Domain Generalization (C1)
**Question:** Does the skill teaching effect generalize across domains of different complexity?

**H3a:** Metadata presence and category accuracy will remain ≥80% for both epigenomics (easy: single-modality, well-defined tools) and multi-omics integration (hard: cross-modal, ambiguous boundaries).

**H3b:** Multi-omics packages will exercise more AQUADIF categories (target: 9-10 of 10) including DELEGATE and SYNTHESIZE, which are unnecessary for single-domain packages.

### RQ4: Iterative Skill Improvement (C2)
**Question:** Does the test-identify-fix-retest cycle measurably improve agent output quality?

**H4a:** Mean quality grade will increase monotonically from iter-02 to iter-05 across all agents.

**H4b:** Specific skill fixes will produce measurable reductions in targeted anti-patterns (e.g., `try/except ImportError` elimination, scaffold adoption rate).

### RQ5: Small-Model Deployment (C3)
**Question:** Can packages built by frontier models route queries accurately when executed by small, locally-deployable models?

**H5a:** Routing accuracy on skill-built packages will exceed 80% for models ≥14B parameters, enabled by structured tool categories and auto-generated prompts.

**H5b:** There exists a "capability cliff" below which routing accuracy degrades sharply — identifying this threshold is a key practical finding.

**H5c:** Cost per correct routing drops from $X (frontier API) to ~$0 (local Ollama), demonstrating the "build big, run small" deployment model.

---

## 3. Experimental Design

### 3.1 Campaign Structure

#### Campaign A: WITH vs WITHOUT (Primary Evidence — C1)

The definitive experiment. Tests whether the skill causes the quality difference.

```
3 agents × 2 conditions × 2 domains × 3 replications = 36 trials

Agents:    Claude Opus 4.6, Gemini 3.1 Pro, Codex 5.3
Conditions: WITH skill (treatment), WITHOUT skill (control)
Domains:   Epigenomics (easy), Multi-omics (hard)
Reps:      3 per cell (for variance estimation)
```

| Cell ID | Agent | Condition | Domain | Trials | Status |
|---------|-------|-----------|--------|--------|--------|
| A01-A03 | Claude | WITH | Epigenomics | 3 | iter-05 t1 done; need t2, t3 |
| A04-A06 | Claude | WITHOUT | Epigenomics | 3 | **NOT STARTED** |
| A07-A09 | Claude | WITH | Multi-omics | 3 | **NOT STARTED** |
| A10-A12 | Claude | WITHOUT | Multi-omics | 3 | **NOT STARTED** |
| B01-B03 | Gemini | WITH | Epigenomics | 3 | iter-05 t1 done; need t2, t3 |
| B04-B06 | Gemini | WITHOUT | Epigenomics | 3 | **NOT STARTED** |
| B07-B09 | Gemini | WITH | Multi-omics | 3 | **NOT STARTED** |
| B10-B12 | Gemini | WITHOUT | Multi-omics | 3 | **NOT STARTED** |
| C01-C03 | Codex | WITH | Epigenomics | 3 | iter-05 t1 done; need t2, t3 |
| C04-C06 | Codex | WITHOUT | Epigenomics | 3 | **NOT STARTED** |
| C07-C09 | Codex | WITH | Multi-omics | 3 | **NOT STARTED** |
| C10-C12 | Codex | WITHOUT | Multi-omics | 3 | **NOT STARTED** |

**Controls:**
- Same Docker base image for WITH and WITHOUT
- Same prompt text (domain task description)
- Same autonomy suffix
- Same model version (pinned in `metadata.json`)
- Same Lobster codebase installed (lobster-ai + lobster-research)
- WITHOUT image has NO skill files, NO AGENT.md, NO references/

**What differs (treatment only):**
- WITH: Skill files in `/workspace/SKILLS/lobster-dev/` + consolidated `AGENT.md`
- WITHOUT: Clean workspace with only Lobster source code

#### Campaign B: Longitudinal Skill Improvement (C2)

Already collected across iter-02 → iter-05. No new trials needed — extract and formalize metrics from existing data.

```
4 iterations × 3 agents × 1 condition (WITH) × 1 domain (epigenomics) × 1 trial = 12 data points
```

| Iter | Skill Version | Key Change | Data Location |
|------|--------------|------------|---------------|
| iter-02 | v1 (post-smoke-01 fix) | init.py example fixed | `results/iter-02/` + `results/smoke-01/` |
| iter-03 | v2 (3 fixes applied) | Lazy imports, flat layout, plugin messaging | `results/iter-03/` |
| iter-04 | v2 + infra fixes | Scaffold templates bundled, PATH fixed | `results/iter-04/` |
| iter-05 | v3 (examples + validator) | Concrete examples section, validate-plugin in skill | `results/iter-05/` |

**Metrics to extract per iteration per agent:**
- Grade (A/A-/B+/B/B-/C/D/F → numeric 4.0/3.7/3.3/3.0/2.7/2.0/1.0/0)
- Tool count
- Category coverage (N of 10)
- Ground truth pattern match (N of 9)
- Implementation depth (% real vs stub)
- Anti-pattern count (try/except, module-level imports, wrong layout)
- Scaffold usage (binary)
- Validator usage (binary)
- Validator pass rate (N of N checks)
- Duration (seconds)

#### Campaign C: Model Size Scaling (C3)

Uses the BEST skill-built package from Campaign A as a fixed artifact. Tests whether the structured AQUADIF metadata enables accurate routing on progressively smaller models.

```
1 package × 5 model sizes × 20 queries × 3 replications = 300 routing evaluations
```

**Package selection:** Highest-graded Claude WITH epigenomics trial from Campaign A (expected: A or A+).

**Model ladder:**

| Tier | Model | Parameters | Runtime | Expected Cost/Query |
|------|-------|-----------|---------|-------------------|
| Frontier | Claude Opus 4.6 | ~2T (est.) | AWS Bedrock | ~$0.15 |
| Mid-tier | Claude Haiku 4.5 | ~70B (est.) | AWS Bedrock | ~$0.01 |
| Local Large | Llama 3.3 70B | 70B | Ollama (M3 Max) | ~$0 |
| Local Medium | Phi-4 14B | 14B | Ollama (M3 Max) | ~$0 |
| Local Small | Llama 3.1 8B | 8B | Ollama (M3 Max) | ~$0 |

**Query design:** 20 standardized routing queries that collectively cover all 10 AQUADIF categories:

| Category | Example Query | Expected Tool |
|----------|---------------|---------------|
| IMPORT | "Load ChIP-seq peaks from BED file" | `load_bed_file` |
| QUALITY | "Assess quality of methylation data" | `assess_methylation_quality` |
| FILTER | "Remove low-quality peaks" | `filter_peaks` |
| PREPROCESS | "Normalize methylation beta values" | `normalize_methylation` |
| ANALYZE | "Find differentially methylated regions" | `find_dmrs` |
| ANNOTATE | "Map peaks to nearest genes" | `annotate_peaks_to_genes` |
| DELEGATE | "Analyze chromatin accessibility" | `handoff_to_atac_expert` |
| SYNTHESIZE | "Integrate ChIP-seq and methylation results" | (expected: fallback or no tool) |
| UTILITY | "Show available data modalities" | `list_modalities` |
| CODE_EXEC | "Run custom visualization script" | `execute_custom_code` |

**Evaluation:** Exact-match on tool selection. Partial credit for correct category but wrong specific tool.

**Key metric:** Routing accuracy (%) per model size. The curve shape determines the "build big, run small" narrative:
- Flat (>80% at 8B): "Structured metadata enables even tiny models to route correctly"
- Graceful degradation: "Accuracy degrades linearly; 14B is the practical minimum"
- Cliff: "Below X parameters, routing fails — structured metadata has limits"

---

### 3.2 Minimum Viable Campaign (If Time-Constrained)

If the full 36-trial Campaign A is infeasible, the absolute minimum for a credible publication:

```
Minimum: 3 agents × 2 conditions × 1 domain × 1 trial = 6 trials (3 new WITHOUT runs)
```

Plus 3 model sizes (Opus, Llama 70B, Llama 8B) × 20 queries = 60 routing evaluations.

**Total new work: 6 trials + 60 routing evals.**

This sacrifices replications and multi-omics generalization but preserves the two critical comparisons (WITH vs WITHOUT, frontier vs local).

---

## 4. Metrics & Scoring

### 4.1 Primary Metrics (Automated — No Human Judgment)

These metrics are extracted programmatically from generated source code via AST parsing and regex matching. No subjective evaluation.

| # | Metric | Type | Extraction Method | Appears In |
|---|--------|------|-------------------|-----------|
| M1 | Metadata presence | % (0-100) | Count tools with `.metadata` dict / total `@tool` decorators | Sec 07.4 Table |
| M2 | Category assignment accuracy | % (0-100) | Compare `.metadata["categories"][0]` to ground truth primary category | Sec 07.4 Table |
| M3 | Category coverage | Count (0-10) | Unique categories across all tools | Sec 07.4 Table |
| M4 | Provenance compliance | % (0-100) | Tools in provenance-required categories that call `log_tool_usage(ir=` | Sec 07.4 Table |
| M5 | Structural compliance | Count (0-8) | `validate_plugin.py` check pass count | Sec 07.4 Table |
| M6 | Tool count | Integer | Count `@tool` decorators in agent module | Sec 07.4 Table |
| M7 | Ground truth coverage | % (0-100) | Regex match against `ground-truth/{domain}.json` required patterns | Sec 07.4 Table |
| M8 | Duration | Seconds | `metadata.json` → `duration_seconds` | Appendix |

### 4.2 Secondary Metrics (Automated — Heuristic)

| # | Metric | Type | Extraction Method | Appears In |
|---|--------|------|-------------------|-----------|
| M9 | Implementation depth | % real (0-100) | Heuristic: service functions with >3 lines of non-boilerplate logic | Appendix |
| M10 | Anti-pattern count | Integer | Count: try/except ImportError, module-level heavy imports, compat shims | Appendix |
| M11 | Scaffold usage | Binary | Grep workspace for `lobster scaffold` in trace.jsonl or output.log | Sec 05.5 |
| M12 | Validator usage | Binary | Grep for `validate_plugin` in trace.jsonl or output.log | Sec 05.5 |
| M13 | Validator pass rate | Fraction | Parse validator output from output.log (e.g., "12/12 PASS") | Sec 05.5 |
| M14 | .tags consistency | % (0-100) | Check `.tags` matches `.metadata["categories"]` for each tool | Appendix |
| M15 | 3-tuple compliance | % (0-100) | Service functions returning `(AnnData, Dict, AnalysisStep)` pattern | Appendix |

### 4.3 Composite Grade (For Longitudinal Analysis Only)

Used in Campaign B to track improvement across iterations. NOT used as a primary metric in Campaign A.

```
Grade = weighted_sum(
    0.25 × M1_metadata_presence,
    0.20 × M2_category_accuracy,
    0.15 × M7_ground_truth_coverage,
    0.15 × M4_provenance_compliance,
    0.15 × M5_structural_compliance / 8 × 100,
    0.10 × M9_implementation_depth
)

Scale: A (≥90), A- (≥87), B+ (≥83), B (≥80), B- (≥77), C+ (≥73), C (≥70), D (≥60), F (<60)
```

### 4.4 Model Scaling Metrics (Campaign C)

| # | Metric | Type | Method |
|---|--------|------|--------|
| S1 | Routing accuracy | % (0-100) | Exact tool name match against expected tool |
| S2 | Category accuracy | % (0-100) | Correct category even if wrong specific tool |
| S3 | Latency per query | Seconds | Time from query to tool selection |
| S4 | Cost per query | USD | API cost (Bedrock) or $0 (Ollama) |
| S5 | Refusal rate | % | Model refuses to select a tool or hallucinates non-existent tool |

---

## 5. Infrastructure Requirements

### 5.1 Existing Infrastructure (Ready)

| Component | Path | Status |
|-----------|------|--------|
| Docker image builder | `.testing/skill-eval/build.sh` | Working (317 lines, builds 6 images) |
| Trial runner | `.testing/skill-eval/run-test.sh` | Working (393 lines, handles 3 agents) |
| Plugin validator | `skills/lobster-dev/references/validate_plugin.py` | Working (8 structural checks) |
| Ground truth (epigenomics) | `.testing/skill-eval/ground-truth/epigenomics.json` | Complete (9 required patterns) |
| Domain prompt (epigenomics) | `.testing/skill-eval/prompts/epigenomics.txt` | Complete |
| Domain prompt (multi-omics) | `.testing/skill-eval/prompts/multi-omics.txt` | Complete |
| AQUADIF contract | `skills/lobster-dev/references/aquadif-contract.md` | Complete (10 categories) |
| Skill files (lobster-dev) | `skills/lobster-dev/` | Complete (11 reference files) |
| Results (iter-02 → iter-05) | `.testing/skill-eval/results/` | 12 WITH trials collected |

### 5.2 Infrastructure To Build

| Component | Path | Description | Priority | Est. Effort |
|-----------|------|-------------|----------|-------------|
| **Automated metric extractor** | `.testing/skill-eval/analysis/extract_metrics.py` | Parse generated source code, compute M1-M15 per trial | **P0** | 1 day |
| **Ground truth validator** | `.testing/skill-eval/analysis/validate_ground_truth.py` | Score tool list against `ground-truth/{domain}.json` | **P0** | 0.5 day |
| **Campaign orchestrator** | `.testing/skill-eval/run-campaign.sh` | Run all 36 trials sequentially or in parallel (with rate limiting) | **P0** | 0.5 day |
| **Results aggregator** | `.testing/skill-eval/analysis/aggregate.py` | Merge all trial metrics into `campaign_results.csv` | **P0** | 0.5 day |
| **Statistical analysis** | `.testing/skill-eval/analysis/statistics.py` | WITH vs WITHOUT tests, effect sizes, confidence intervals | **P1** | 0.5 day |
| **Figure generator** | `.testing/skill-eval/analysis/plot_figures.py` | Publication-quality matplotlib figures → PDF | **P1** | 1 day |
| **LaTeX table emitter** | `.testing/skill-eval/analysis/emit_tables.py` | Auto-generate `.tex` table fragments from CSV | **P1** | 0.5 day |
| **Ground truth (multi-omics)** | `.testing/skill-eval/ground-truth/multi-omics.json` | Expert-defined rubric for multi-omics domain | **P1** | 0.5 day |
| **Model scaling harness** | `.testing/skill-eval/model-scaling/run-scaling.sh` | Route 20 queries through 5 model sizes via Lobster | **P1** | 1 day |
| **Scaling query set** | `.testing/skill-eval/model-scaling/queries.json` | 20 standardized queries covering 10 categories | **P1** | 0.5 day |
| **Trace parser** | `.testing/skill-eval/analysis/parse_trace.py` | Extract behavioral metrics from `trace.jsonl` | **P2** | 1 day |
| **Reproducibility manifest** | `.testing/skill-eval/REPRODUCIBILITY.md` | Exact versions, hashes, env requirements | **P2** | 0.5 day |

### 5.3 Authentication & Cost

| Agent | Auth Mechanism | Estimated Cost per Trial | Source |
|-------|---------------|-------------------------|--------|
| Claude Opus 4.6 | AWS Bedrock (IAM role) | ~$5-15 (varies by output length) | `AWS_ACCESS_KEY_ID` + region |
| Gemini 3.1 Pro | Google AI API key | ~$2-8 | `GEMINI_API_KEY` |
| Codex 5.3 | OpenAI API key | ~$3-10 | `OPENAI_API_KEY` |

**Full Campaign A cost estimate:** 36 trials × ~$8 avg = ~$288
**Campaign C cost estimate:** API models ~$10; Ollama models ~$0 → ~$10 total
**Total estimated budget: ~$300**

### 5.4 Hardware Requirements

| Component | Requirement | Available? |
|-----------|------------|-----------|
| Docker runtime | Docker Desktop or Colima | Yes |
| Ollama for local models | M-series Mac with ≥32GB RAM (for 70B) | Verify |
| Network access | API endpoints for 3 providers | Yes |
| Disk space | ~2GB per trial workspace × 36 trials ≈ 72GB | Verify |

---

## 6. Data Collection & Storage

### 6.1 Trial Output Contract

Every trial MUST produce this exact structure:

```
results/{campaign}/{agent}-{condition}-{domain}-t{trial}/
├── metadata.json              # Run metadata (agent, model, duration, exit code)
├── output.log                 # Final agent response (human-readable)
├── docker.log                 # Container stderr (diagnostics)
├── .full-prompt.txt           # Exact prompt sent (for reproducibility)
├── metrics.json               # AUTO-GENERATED by extract_metrics.py
└── workspace/
    ├── trace.jsonl            # Raw agent event stream
    ├── AGENT.md               # Skill files (WITH only)
    ├── packages/
    │   └── lobster-{domain}/
    │       ├── pyproject.toml
    │       ├── lobster/agents/{domain}/
    │       │   ├── __init__.py
    │       │   ├── {domain}_expert.py
    │       │   ├── shared_tools.py
    │       │   └── config.py
    │       ├── lobster/services/{domain}/
    │       └── tests/
    └── pyproject.toml         # Sometimes at root level
```

### 6.2 Metrics Output Contract

`extract_metrics.py` produces `metrics.json` per trial:

```json
{
  "trial_id": "claude-with-epigenomics-t1",
  "campaign": "campaign-a",
  "agent": "claude",
  "condition": "with",
  "domain": "epigenomics",

  "primary_metrics": {
    "M1_metadata_presence": 100.0,
    "M2_category_accuracy": 85.7,
    "M3_category_coverage": 7,
    "M4_provenance_compliance": 100.0,
    "M5_structural_compliance": 7,
    "M6_tool_count": 14,
    "M7_ground_truth_coverage": 88.9,
    "M8_duration_seconds": 643
  },

  "secondary_metrics": {
    "M9_implementation_depth": 92.9,
    "M10_antipattern_count": 0,
    "M11_scaffold_used": true,
    "M12_validator_used": true,
    "M13_validator_pass_rate": "12/12",
    "M14_tags_consistency": 100.0,
    "M15_three_tuple_compliance": 85.7
  },

  "tool_inventory": [
    {"name": "load_bed_file", "category": "IMPORT", "provenance": true, "has_metadata": true, "ground_truth_match": "import.*peak|load.*bed"},
    ...
  ],

  "categories_exercised": ["IMPORT", "QUALITY", "FILTER", "PREPROCESS", "ANALYZE", "ANNOTATE", "UTILITY"],
  "categories_missing": ["DELEGATE", "SYNTHESIZE", "CODE_EXEC"],

  "antipatterns_found": [],

  "extraction_timestamp": "2026-03-02T10:00:00Z",
  "extractor_version": "1.0.0"
}
```

### 6.3 Aggregate Output Contract

`aggregate.py` produces `campaign_results.csv`:

```
trial_id,campaign,agent,condition,domain,rep,M1,M2,M3,M4,M5,M6,M7,M8,M9,M10,M11,M12,M13,M14,M15,grade
claude-with-epigenomics-t1,campaign-a,claude,with,epigenomics,1,100.0,85.7,7,100.0,7,14,88.9,643,92.9,0,1,1,12/12,100.0,85.7,A
claude-without-epigenomics-t1,campaign-a,claude,without,epigenomics,1,0.0,...
...
```

### 6.4 Model Scaling Output Contract

```
model-scaling/results/{package_source}/{model_id}/
├── routing_results.json       # Per-query: query, expected_tool, actual_tool, correct, latency
├── summary.json               # Aggregate accuracy, latency, cost
└── raw_responses/             # Full LLM responses for each query
    ├── query_01.json
    ├── query_02.json
    └── ...
```

---

## 7. Statistical Analysis Plan

### 7.1 Campaign A: WITH vs WITHOUT

**Primary analysis:** Two-way ANOVA (agent × condition) for each primary metric, with domain as a blocking factor.

| Test | DV | IV | Method | Justification |
|------|----|----|--------|---------------|
| Skill effect on metadata | M1 | Condition (WITH/WITHOUT) | Mann-Whitney U (per agent) | Non-normal, small N |
| Skill effect on accuracy | M2 | Condition (WITH/WITHOUT) | Mann-Whitney U (per agent) | Non-normal, small N |
| Cross-agent consistency | M1-M7 | Agent × Condition | Kruskal-Wallis | 3 groups, small N |
| Domain generalization | M1-M7 | Domain × Condition | Paired comparison (same agent) | Tests H3a |

**Effect size:** Cohen's d for each metric × condition comparison. Report 95% bootstrap confidence intervals (10,000 resamples) given small N.

**Multiple comparison correction:** Bonferroni for the 7 primary metrics (α = 0.05/7 = 0.007).

**Expected pattern:** Large effect sizes (d > 2.0) for metadata presence (M1) — this is essentially binary (100% vs ~0%). Moderate effects (d = 0.5-1.5) for accuracy and coverage metrics.

**Note on small N:** With 3 replications per cell, statistical power is limited. We acknowledge this honestly and frame results as effect sizes with confidence intervals, not p-values alone. The longitudinal Campaign B data (12 data points) provides corroborating evidence.

### 7.2 Campaign B: Longitudinal Improvement

**Analysis:** Spearman rank correlation between iteration number (ordinal: iter-02=1, iter-03=2, iter-04=3, iter-05=4) and composite grade per agent.

**Visualization:** Line chart showing grade progression per agent across 4 iterations with trend lines.

**Anti-pattern analysis:** Binary occurrence tracking (try/except present: 1/0) across iterations. Report as contingency table with Fisher's exact test.

### 7.3 Campaign C: Model Scaling

**Analysis:** Logistic regression of routing accuracy on log(model_parameters) to characterize the accuracy-size relationship.

**Visualization:** Line chart with 95% CI bands (from 3 replications per model size). X-axis: model size (log scale). Y-axis: routing accuracy (%).

**Cost-accuracy frontier:** Plot cost per correct routing vs accuracy for all 5 model sizes. Identify Pareto-optimal points.

### 7.4 Reporting Standards

All results reported following:
- **Effect sizes** with 95% confidence intervals (not just p-values)
- **Raw counts** alongside percentages (e.g., "13/14 tools, 92.9%")
- **Individual trial data** in appendix (no aggregation-only reporting)
- **Negative results** reported explicitly (BioMLBench 0%, Gemini depth ceiling, SYNTHESIZE gap)

---

## 8. Ground Truth Definitions

### 8.1 Epigenomics Ground Truth (Existing — Verify)

Source: `.testing/skill-eval/ground-truth/epigenomics.json`

| Pattern # | Regex | Category | Description |
|-----------|-------|----------|-------------|
| GT-E1 | `import.*peak\|load.*bed\|import.*narrowpeak` | IMPORT | Load BED/narrowPeak/broadPeak |
| GT-E2 | `import.*methyl\|load.*idat\|import.*beta` | IMPORT | Load methylation data |
| GT-E3 | `import.*atac\|import.*fragment\|load.*fragment` | IMPORT | Load ATAC-seq fragments/signal |
| GT-E4 | `quality\|qc\|assess.*quality` | QUALITY | QC assessment |
| GT-E5 | `filter.*peak\|filter.*probe\|filter.*region` | FILTER | Data subsetting |
| GT-E6 | `normalize\|batch.*correct\|impute` | PREPROCESS | Data transformation |
| GT-E7 | `differential\|enrichment\|pca\|cluster\|dmr` | ANALYZE | Analytical computation |
| GT-E8 | `annotate\|gene.*map\|ontology\|pathway` | ANNOTATE | Biological annotation |
| GT-E9 | `list.*modal\|status\|export\|summarize` | UTILITY | Workspace management |

**Expected categories:** IMPORT, QUALITY, FILTER, PREPROCESS, ANALYZE, ANNOTATE, UTILITY (7 of 10)
**Not expected:** DELEGATE (no child agents), SYNTHESIZE (known gap), CODE_EXEC (optional)
**Tool count range:** 10-20 (below 10 = too sparse, above 20 = bloated)

### 8.2 Multi-Omics Ground Truth (To Create)

Must be defined BEFORE running multi-omics trials. Expert-authored, versioned in `ground-truth/multi-omics.json`.

| Pattern # | Regex | Category | Description |
|-----------|-------|----------|-------------|
| GT-M1 | `load.*transcriptom\|import.*expression` | IMPORT | Load expression data |
| GT-M2 | `load.*proteom\|import.*protein` | IMPORT | Load proteomics data |
| GT-M3 | `load.*metabol\|import.*metabol` | IMPORT | Load metabolomics data |
| GT-M4 | `quality\|qc\|assess` | QUALITY | Cross-modal QC |
| GT-M5 | `filter\|subset\|remove` | FILTER | Multi-modal filtering |
| GT-M6 | `normalize\|batch.*correct\|harmonize` | PREPROCESS | Cross-modal normalization |
| GT-M7 | `integrat\|correlat\|factor.*analysis\|mofa\|mcia` | ANALYZE | Multi-omics integration |
| GT-M8 | `pathway\|enrich\|annotate\|ontology` | ANNOTATE | Biological annotation |
| GT-M9 | `handoff\|delegate\|transfer.*to` | DELEGATE | Specialist delegation |
| GT-M10 | `synthesize\|summarize.*result\|cross.*modal.*report` | SYNTHESIZE | Cross-result integration |
| GT-M11 | `list\|status\|export` | UTILITY | Workspace ops |
| GT-M12 | `execute.*code\|custom.*script` | CODE_EXEC | Escape hatch |

**Expected categories:** All 10 (multi-omics requires cross-agent delegation and synthesis)
**Tool count range:** 12-18

### 8.3 Ground Truth Versioning

Ground truth files are immutable per campaign. If the rubric changes, a new campaign starts. This prevents p-hacking by adjusting ground truth after seeing results.

```
ground-truth/
├── epigenomics.json          # v1.0 — used since iter-02
├── multi-omics.json          # v1.0 — to be created before Campaign A
└── CHANGELOG.md              # Version history with rationale
```

---

## 9. Figures & Tables (Publication Deliverables)

### 9.1 Main Paper Figures

| Figure | Content | Data Source | Generator |
|--------|---------|-------------|-----------|
| Fig 3a | WITH vs WITHOUT: metadata presence + category accuracy per agent | `campaign_results.csv` | `plot_figures.py` |
| Fig 3b | Category coverage heatmap: 10 categories × 3 agents × 2 conditions | `campaign_results.csv` | `plot_figures.py` |
| Fig 4 | Model scaling: routing accuracy vs model size with cost annotations | `model-scaling/results/` | `plot_figures.py` |
| Fig 5 | Benchmark results: grouped bars (SCBench + LabBench2 key tags) | Existing benchmark data | `plot_figures.py` |

### 9.2 Main Paper Tables

| Table | Content | Data Source | Generator |
|-------|---------|-------------|-----------|
| Tab 4 | Tool category taxonomy across 4 omics domains | Codebase + generated packages | Manual + `emit_tables.py` |
| Tab 5 | AQUADIF eval summary: 7 primary metrics × 3 agents × 2 conditions | `campaign_results.csv` | `emit_tables.py` |
| Tab 6 | Model scaling: accuracy + cost × 5 model sizes | `model-scaling/results/` | `emit_tables.py` |

### 9.3 Appendix Tables

| Table | Content | Data Source |
|-------|---------|-------------|
| Tab A.6 | Full per-trial Campaign A results (36 rows × 15 metrics) | `campaign_results.csv` |
| Tab A.7 | Tool inventory per agent per condition (all tools with categories) | `metrics.json` per trial |
| Tab A.8 | Anti-pattern tracking across 4 iterations (Campaign B) | `metrics.json` per iter |
| Tab A.9 | Model scaling per-query breakdown (20 queries × 5 models) | `model-scaling/results/` |
| Tab A.10 | Ground truth rubrics (epigenomics + multi-omics) | `ground-truth/*.json` |

### 9.4 Auto-Generation Pipeline

```
extract_metrics.py → metrics.json (per trial)
       ↓
aggregate.py → campaign_results.csv (all trials)
       ↓
   ┌───┴───┐
   ↓       ↓
statistics.py    plot_figures.py    emit_tables.py
   ↓                ↓                    ↓
stats.json     figures/*.pdf       tables/*.tex
```

All figures and tables are regenerated from raw data with a single command:

```bash
make figures    # in .testing/skill-eval/
```

No manual editing of figures or tables. If a number changes, rerun the pipeline.

---

## 10. Reproducibility Guarantees

### 10.1 Version Pinning

| Component | How Pinned | Where Recorded |
|-----------|-----------|---------------|
| Agent models | Model ID string in metadata.json | `metadata.json → model_id` |
| Lobster version | Git commit SHA of installed version | `metadata.json → lobster_sha` |
| Skill version | Git commit SHA of skills/ directory | `metadata.json → skill_sha` |
| Docker base image | Digest in Dockerfile.base | `build.sh` output log |
| Python version | Fixed in Dockerfile (3.12) | Dockerfile.base |
| Ground truth | Versioned JSON, immutable per campaign | `ground-truth/CHANGELOG.md` |
| Prompts | Exact text captured | `.full-prompt.txt` per trial |

### 10.2 Metadata Enhancements (add to run-test.sh)

Current `metadata.json` captures: trial_id, agent, model_id, condition, domain, timestamps, duration, exit_code.

**Add:**
- `lobster_sha`: `git -C $REPO_ROOT rev-parse HEAD`
- `skill_sha`: `git -C $REPO_ROOT/skills log -1 --format=%H -- lobster-dev/`
- `docker_image_digest`: `docker inspect --format='{{.Id}}' $IMAGE`
- `prompt_sha256`: `sha256sum .full-prompt.txt`
- `ground_truth_sha256`: `sha256sum ground-truth/{domain}.json`
- `build_sh_sha256`: `sha256sum build.sh`
- `run_test_sh_sha256`: `sha256sum run-test.sh`

### 10.3 Replication Instructions

The paper's appendix or supplementary material includes:

```bash
# 1. Clone repository at pinned commit
git clone https://github.com/the-omics-os/lobster.git
cd lobster && git checkout <SHA>

# 2. Build evaluation images
cd .testing/skill-eval
./build.sh

# 3. Set API credentials
export ANTHROPIC_API_KEY=...
export GEMINI_API_KEY=...
export OPENAI_API_KEY=...

# 4. Run full campaign
./run-campaign.sh --campaign campaign-a

# 5. Extract metrics and generate figures
cd analysis
python extract_metrics.py ../results/campaign-a/
python aggregate.py ../results/campaign-a/
python statistics.py campaign_results.csv
python plot_figures.py campaign_results.csv --output ../../figures/
python emit_tables.py campaign_results.csv --output ../../tables/
```

---

## 11. Known Limitations & Honest Declarations

These limitations MUST appear in the paper (Sec 08). Hiding them invites rejection.

| # | Limitation | Severity | Mitigation |
|---|-----------|----------|------------|
| L1 | Small sample size (3 reps per cell) | Moderate | Report effect sizes with bootstrap CIs; longitudinal data (Campaign B) provides corroboration |
| L2 | Single evaluator for ground truth | Moderate | Ground truth is regex-based and versioned; publish rubric for community audit |
| L3 | Epigenomics is well-scoped (easy domain) | Low | Multi-omics domain tests harder boundaries; acknowledge in Discussion |
| L4 | Gemini depth ceiling is model capability, not skill | Low | Report honestly; not every model benefits equally; skill teaches structure, not depth |
| L5 | No WITHOUT control for longitudinal data (Campaign B) | Moderate | Campaign A provides the controlled comparison; Campaign B shows improvement trajectory |
| L6 | SYNTHESIZE category has 0 implementations in ground truth | Low | Declared as intentional gap; measurable progress indicator |
| L7 | Model scaling tests routing only, not end-to-end task completion | Moderate | Routing is the architecture's contribution; end-to-end requires benchmark integration (future work) |
| L8 | API model versions may change between publication and replication | Low | Pin model IDs in metadata; note that exact replication requires same model version |
| L9 | Signal track/fragment import (GT-E3) never implemented by any agent | Low | This is a finding about model training data gaps, not a skill failure |
| L10 | Manual comparison.md files from iter-02 → iter-05 used different rubrics | Moderate | Retroactively extract metrics via extract_metrics.py from saved workspaces for consistent scoring |

---

## 12. Execution Timeline

### Phase 1: Automation (Days 1-2)
Build the analysis pipeline so that all subsequent results are automatically processed.

| Task | Output | Est. Hours |
|------|--------|-----------|
| Write `extract_metrics.py` | Automated M1-M15 extraction from workspace source | 4h |
| Write `validate_ground_truth.py` | Automated ground truth scoring | 2h |
| Write `aggregate.py` | Campaign-wide CSV aggregation | 2h |
| Retroactively extract metrics from iter-02 → iter-05 | `metrics.json` for all 12 existing trials | 2h |
| Validate extracted metrics against manual comparison.md grades | Calibration check | 1h |

### Phase 2: WITHOUT Trials (Days 3-4)
Run the control condition. This is the single most important missing piece.

| Task | Output | Est. Hours |
|------|--------|-----------|
| Verify WITHOUT Docker images build correctly | 3 images with no skill files | 1h |
| Run: Claude WITHOUT epigenomics × 1 trial | Baseline package | 0.5h (automated) |
| Run: Gemini WITHOUT epigenomics × 1 trial | Baseline package | 0.5h (automated) |
| Run: Codex WITHOUT epigenomics × 1 trial | Baseline package | 0.5h (automated) |
| Extract metrics from 3 WITHOUT trials | WITH vs WITHOUT comparison data | 1h |
| Run additional WITH trials (t2, t3) for replications | 6 more trials | 3h (automated) |
| Run additional WITHOUT trials (t2, t3) for replications | 6 more trials | 3h (automated) |

### Phase 3: Model Scaling (Days 5-6)
Test whether skill-built packages enable small-model routing.

| Task | Output | Est. Hours |
|------|--------|-----------|
| Design 20 standardized routing queries | `queries.json` | 2h |
| Install best Claude WITH package into Lobster | Working domain agent | 1h |
| Write `run-scaling.sh` harness | Automated query routing | 2h |
| Download Ollama models (70B, 14B, 8B) | Local model weights | 1h (download) |
| Run 5 model sizes × 20 queries × 3 reps | 300 routing evaluations | 4h |
| Analyze scaling results | Routing accuracy curve | 1h |

### Phase 4: Multi-Omics Domain (Days 7-8) — If Time Permits
Tests cross-domain generalization. Valuable but not strictly blocking.

| Task | Output | Est. Hours |
|------|--------|-----------|
| Author `ground-truth/multi-omics.json` | Expert rubric for hard domain | 2h |
| Run 3 agents × 2 conditions × 1 trial | 6 multi-omics trials | 3h |
| Extract metrics and add to aggregate | Cross-domain comparison | 1h |

### Phase 5: Statistics & Figures (Days 9-10)

| Task | Output | Est. Hours |
|------|--------|-----------|
| Write `statistics.py` | WITH vs WITHOUT effect sizes, CIs, p-values | 3h |
| Write `plot_figures.py` | 4 publication-quality figures → PDF | 4h |
| Write `emit_tables.py` | 6 LaTeX table fragments | 2h |
| Generate all figures and tables | `figures/*.pdf`, `tables/*.tex` | 0.5h (automated) |

### Phase 6: Paper Writing (Days 11-14)

| Task | Output | Est. Hours |
|------|--------|-----------|
| Rewrite Sec 05 (skill-guided extension) | Fill Table 4, fix mechanism, write case study | 6h |
| Rewrite Sec 07.4 (AQUADIF eval) | WITH vs WITHOUT results, figures, tables | 4h |
| Write Sec 07.5 (model scaling) | Scaling curve, cost analysis | 3h |
| Update abstract, intro, discussion, conclusion | Consistent numbers throughout | 4h |
| Condense all sections to 9-page budget | Camera-ready content pages | 6h |
| Create Fig 2 (AQUADIF taxonomy diagram) | TikZ or draw.io → PDF | 3h |
| Expand appendix with detailed tables | A.6-A.10 | 3h |
| Final 18-point verification | Clean submission | 2h |

---

## 13. Success Criteria

The evaluation is publishable if ALL of the following hold:

| # | Criterion | Threshold | Measurement |
|---|-----------|-----------|-------------|
| SC1 | WITH > WITHOUT on metadata presence | Effect size d > 2.0, p < 0.01 | Campaign A, all 3 agents |
| SC2 | WITH > WITHOUT on category accuracy | Effect size d > 0.8, p < 0.05 | Campaign A, at least 2 of 3 agents |
| SC3 | Longitudinal improvement visible | Spearman ρ > 0.5 for composite grade vs iteration | Campaign B |
| SC4 | Model scaling shows practical threshold | ≥1 local model achieves >70% routing accuracy | Campaign C |
| SC5 | All metrics extracted automatically | Zero manual scoring in Campaign A | Pipeline verification |
| SC6 | All figures generated from data | `make figures` reproduces all publication figures | Pipeline verification |

**If SC1 fails:** The skill doesn't teach what we claim. Reassess the entire Innovation 2 framing.
**If SC4 fails:** Drop the "build big, run small" claim. Focus on skill teaching effect alone.
**If SC3 fails:** Drop the iterative improvement narrative. Focus on WITH vs WITHOUT only.

---

## 14. Relationship to Paper Sections

| Paper Section | Primary Evidence | Campaign | Figures/Tables |
|--------------|-----------------|----------|---------------|
| Sec 05.4 (Contract Tests) | `validate_plugin.py` pass rates across trials | A + B | — |
| Sec 05.5 (Case Study) | Best Claude WITH trial narrative + iter-02→05 progression | B + best from A | — |
| Sec 07.4 (Self-Extension Eval) | WITH vs WITHOUT across 3 agents, 2 domains | A | Fig 3, Tab 5 |
| Sec 07.5 (Model Scaling) | Routing accuracy × 5 model sizes | C | Fig 4, Tab 6 |
| Sec 07.6 (Cost Analysis) | Frontier vs local inference cost | C | Integrated in Fig 4 |
| Sec 08 (Discussion) | Agent personality variance, Gemini ceiling, Codex taxonomy ≠ integration | A + B | — |
| Appendix A.6-A.10 | Full per-trial data, tool inventories, anti-pattern tracking | All | Tab A.6-A.10 |

---

## 15. Open Questions Requiring Human Decision

| # | Question | Options | Deadline |
|---|----------|---------|----------|
| Q1 | Include multi-omics domain or epigenomics-only? | Full (both) or minimum (epigenomics only) | Before Campaign A starts |
| Q2 | Number of replications per cell? | 3 (planned) or 1 (minimum viable)? | Before Campaign A starts |
| Q3 | Which Ollama models for scaling? | Llama 3.3 70B + Phi-4 14B + Llama 3.1 8B, or different set? | Before Campaign C |
| Q4 | Should Campaign B metrics be retroactively re-extracted from saved workspaces? | Yes (consistent scoring) or use existing manual grades? | Before writing Sec 05.5 |
| Q5 | Is the composite grade formula (Sec 4.3) the right weighting? | Current weights or revise? | Before aggregation |
| Q6 | For model scaling, test routing only or also end-to-end execution? | Routing only (feasible) or full execution (weeks of work)? | Before Campaign C |
