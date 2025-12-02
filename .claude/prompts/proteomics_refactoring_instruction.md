# Proteomics Agent Refactoring Instruction Prompt

**For: Claude Code / lo-ass agents**
**Created from**: Transcriptomics refactoring learnings
**Date**: 2024-11-30

---

## CONTEXT & OBJECTIVE

You are a world-class Python software engineer and expert in agentic Generative AI with 20+ years in bioinformatics. You only write bug-free modular code which respects the lobster codebase. You report if an implementation is badly written.

**Overall Goal**: The `ms_proteomics_expert.py` and `affinity_proteomics_expert.py` agents have high code duplication (~70-80% structural similarity). Refactor them into a unified `proteomics_expert` architecture following the pattern established by the transcriptomics refactoring.

---

## CURRENT STATE ANALYSIS

### Tool Inventory

| Agent | Total Tools | Unique Tools |
|-------|-------------|--------------|
| `ms_proteomics_expert.py` | 8 | 1 (`add_peptide_mapping_to_ms_modality`) |
| `affinity_proteomics_expert.py` | 8 | 1 (`validate_antibody_specificity`) |

### Tool Overlap (7/8 tools are parallel implementations)

| Function | MS Version | Affinity Version | Key Differences |
|----------|------------|------------------|-----------------|
| Data Status | `check_ms_proteomics_data_status` | `check_affinity_proteomics_data_status` | Column detection patterns |
| Quality | `assess_ms_proteomics_quality` | `assess_affinity_proteomics_quality` | Thresholds: missing 0.7 vs 0.3, CV 50% vs 30% |
| Filter | `filter_ms_proteomics_data` | `filter_affinity_proteomics_data` | MS: peptide counts, contaminants; Affinity: CV, failed antibodies |
| Normalize | `normalize_ms_proteomics_data` | `normalize_affinity_proteomics_data` | MS: log_transform=True; Affinity: log_transform=False |
| Patterns | `analyze_ms_proteomics_patterns` | `analyze_affinity_proteomics_patterns` | PCA components: 15 vs 10 |
| Differential | `find_differential_proteins_ms` | `find_differential_proteins_affinity` | FC threshold: 1.5 vs 1.2, method: limma vs t_test |
| Summary | `create_ms_proteomics_summary` | `create_affinity_proteomics_summary` | Modality filtering patterns |

### Shared Services (ALL 4 services are identical between agents)

- `ProteomicsQualityService`
- `ProteomicsPreprocessingService`
- `ProteomicsAnalysisService`
- `ProteomicsDifferentialService`

---

## SCIENTIFIC ISSUES TO FIX

### CRITICAL (Must fix during refactoring)

1. **3-Tuple Pattern Non-Compliance**: All proteomics services return 2-tuples `(AnnData, Dict)` instead of required 3-tuples `(AnnData, Dict, AnalysisStep)`. This breaks:
   - Provenance tracking (W3C-PROV)
   - Notebook export via `/pipeline export`
   - Reproducibility guarantees

   **Fix**: Each agent tool must create AnalysisStep IR even if service doesn't return it, OR update services to return IR.

2. **Simulated Pathway Enrichment**: `ProteomicsAnalysisService.perform_pathway_enrichment` generates fake pathway data instead of querying real databases.

   **Fix**: Add clear warning in tool docstring: "NOTE: Currently uses simulated pathway data. Real database integration (GO, KEGG, Reactome) planned for future release."

### MEDIUM (Should fix)

3. **Misleading Method Names**:
   - `_vsn_normalization` is `arcsinh(X/2)`, NOT true VSN (which uses MLE)
   - `_perform_umap_like` is just variance-weighted PCA, NOT UMAP

   **Fix**: Rename or add docstring warnings about approximations.

4. **Unit Inconsistency**: CV threshold is percentage (30.0) in visualization but fraction (0.2) in quality service.

   **Fix**: Standardize to percentage throughout, or clearly document units in all docstrings.

5. **Missing Empirical Bayes**: `_estimate_prior_variance` is implemented but never called in `_moderated_t_test`.

   **Fix**: Either call it or remove dead code. Document that limma-like is simplified.

---

## RECOMMENDED ARCHITECTURE

Given the high overlap and simpler structure (16 tools total → 10 unique), use **Option 2: Unified Agent with Platform Config** rather than sub-agents.

```
lobster/agents/proteomics/
├── __init__.py                    # Module exports
├── proteomics_expert.py           # Unified parent agent (10 tools)
├── platform_config.py             # Platform-specific defaults
├── shared_tools.py                # Shared tools with auto-detection
├── state.py                       # ProteomicsExpertState
├── deprecated.py                  # Backwards-compatible aliases
└── prompts.py                     # System prompts (optional, can inline)
```

### Platform Configuration Pattern

```python
# platform_config.py
from typing import Literal
from dataclasses import dataclass

PlatformType = Literal["mass_spec", "affinity"]

@dataclass
class PlatformConfig:
    """Platform-specific defaults for proteomics analysis."""
    missing_threshold: float
    cv_threshold: float
    fc_threshold: float
    n_pca_components: int
    log_transform: bool
    default_stat_method: str
    # Platform-specific metadata columns
    expected_columns: list[str]

PLATFORM_CONFIGS = {
    "mass_spec": PlatformConfig(
        missing_threshold=0.7,
        cv_threshold=50.0,  # Use percentage throughout
        fc_threshold=1.5,
        n_pca_components=15,
        log_transform=True,
        default_stat_method="limma_moderated",
        expected_columns=["peptide_count", "contaminant", "reverse"],
    ),
    "affinity": PlatformConfig(
        missing_threshold=0.3,
        cv_threshold=30.0,
        fc_threshold=1.2,
        n_pca_components=10,
        log_transform=False,
        default_stat_method="t_test",
        expected_columns=["antibody_id", "panel", "plate_id"],
    ),
}

def detect_platform_type(adata) -> PlatformType:
    """Auto-detect platform from data characteristics."""
    obs_cols = set(adata.obs.columns.str.lower())
    var_cols = set(adata.var.columns.str.lower())
    all_cols = obs_cols | var_cols

    # Check for MS-specific columns
    ms_indicators = {"peptide", "peptide_count", "contaminant", "reverse", "razor"}
    if any(ind in col for col in all_cols for ind in ms_indicators):
        return "mass_spec"

    # Check for affinity-specific columns
    affinity_indicators = {"antibody", "panel", "plate", "olink", "npx"}
    if any(ind in col for col in all_cols for ind in affinity_indicators):
        return "affinity"

    # Default heuristics
    if adata.n_vars > 5000:  # MS typically has more proteins
        return "mass_spec"
    return "affinity"
```

### Tool Distribution (10 tools total)

| Tool | Description | Platform-Aware |
|------|-------------|----------------|
| `check_proteomics_data_status` | Check modality status | Yes (auto-detect) |
| `assess_proteomics_quality` | QC with platform defaults | Yes |
| `filter_proteomics_data` | Filter with platform defaults | Yes |
| `normalize_proteomics_data` | Normalize with platform defaults | Yes |
| `analyze_proteomics_patterns` | PCA/clustering | Yes |
| `find_differential_proteins` | Statistical testing | Yes |
| `add_peptide_mapping` | MS-specific: peptide→protein | No (MS only) |
| `validate_antibody_specificity` | Affinity-specific: cross-reactivity | No (Affinity only) |
| `correct_plate_effects` | Affinity-specific (extract from normalize) | No (Affinity only) |
| `create_proteomics_summary` | Analysis summary | Yes |

---

## IMPLEMENTATION STEPS

### Phase 1: Directory Structure & Scaffolding
1. Create `lobster/agents/proteomics/` directory
2. Create `__init__.py` with exports
3. Create `state.py` with `ProteomicsExpertState`
4. Create `platform_config.py` with platform detection and configs

### Phase 2: Shared Tools
1. Create `shared_tools.py` with platform-aware tools
2. Implement auto-detection in each tool
3. Apply parameter overrides when explicitly provided
4. **FIX**: Create AnalysisStep IR in each tool for provenance

### Phase 3: Main Agent
1. Create `proteomics_expert.py` with factory function
2. Import shared tools
3. Add MS-specific tool: `add_peptide_mapping`
4. Add Affinity-specific tools: `validate_antibody_specificity`, `correct_plate_effects`
5. Create comprehensive system prompt

### Phase 4: Deprecation & Registry
1. Create `deprecated.py` with `ms_proteomics_alias` and `affinity_proteomics_alias`
2. Update `config/agent_registry.py`:
   - Add `proteomics_expert` (supervisor-accessible)
   - Update `ms_proteomics_expert_agent` to deprecated alias
   - Uncomment and update `affinity_proteomics_expert_agent` to deprecated alias

### Phase 5: Verification
1. Run import tests
2. Verify registry configuration
3. Test deprecation warnings

---

## TOOL PATTERN TO FOLLOW

```python
@tool
def assess_proteomics_quality(
    modality_name: str,
    platform_type: Optional[Literal["mass_spec", "affinity"]] = None,
    missing_threshold: Optional[float] = None,
    cv_threshold: Optional[float] = None,
    # ... other params with None defaults
) -> str:
    """Assess proteomics data quality with platform-appropriate defaults.

    Automatically detects platform type (mass spectrometry vs affinity) and
    applies appropriate QC thresholds. All parameters can be overridden.

    Args:
        modality_name: Name of the modality to assess
        platform_type: "mass_spec" or "affinity". Auto-detected if not specified.
        missing_threshold: Max fraction of missing values (MS: 0.7, Affinity: 0.3)
        cv_threshold: Max CV percentage (MS: 50%, Affinity: 30%)

    Returns:
        Quality assessment summary with metrics and recommendations.
    """
    adata = data_manager.get_modality(modality_name)

    # Auto-detect platform if not specified
    if platform_type is None:
        platform_type = detect_platform_type(adata)

    config = PLATFORM_CONFIGS[platform_type]

    # Use explicit params if provided, else platform defaults
    missing_threshold = missing_threshold if missing_threshold is not None else config.missing_threshold
    cv_threshold = cv_threshold if cv_threshold is not None else config.cv_threshold

    # Call service (returns 2-tuple, we need to create IR)
    result, stats = quality_service.assess_missing_value_patterns(
        adata, sample_threshold=missing_threshold
    )

    # Create IR for provenance (since service doesn't return it)
    ir = AnalysisStep(
        operation="proteomics_quality_assessment",
        tool_name="assess_proteomics_quality",
        description=f"Quality assessment for {platform_type} proteomics data",
        library="lobster.services.quality.proteomics_quality_service",
        parameters={
            "platform_type": platform_type,
            "missing_threshold": missing_threshold,
            "cv_threshold": cv_threshold,
        },
        code_template='''# Proteomics QC
from lobster.services.quality.proteomics_quality_service import ProteomicsQualityService
quality_service = ProteomicsQualityService()
result, stats = quality_service.assess_missing_value_patterns(
    adata, sample_threshold={{ missing_threshold }}
)''',
        imports=["from lobster.services.quality.proteomics_quality_service import ProteomicsQualityService"],
    )

    # Store result
    new_name = f"{modality_name}_quality_assessed"
    data_manager.modalities[new_name] = result
    data_manager.log_tool_usage(
        "assess_proteomics_quality",
        {"platform_type": platform_type, "missing_threshold": missing_threshold},
        stats,
        ir=ir  # IR is mandatory!
    )

    return f"Quality assessment complete ({platform_type}): {stats}"
```

---

## REGISTRY CONFIGURATION

```python
# In config/agent_registry.py

"proteomics_expert": AgentRegistryConfig(
    name="proteomics_expert",
    display_name="Proteomics Expert",
    description="Unified expert for mass spectrometry AND affinity proteomics. Auto-detects platform type. Handles QC, normalization, differential analysis, peptide mapping (MS), antibody validation (affinity).",
    factory_function="lobster.agents.proteomics.proteomics_expert.proteomics_expert",
    handoff_tool_name="handoff_to_proteomics_expert",
    handoff_tool_description="Assign ALL proteomics analysis tasks (mass spectrometry OR affinity platforms): QC, normalization, batch correction, differential protein expression, peptide mapping, antibody validation",
),

# DEPRECATED aliases
"ms_proteomics_expert_agent": AgentRegistryConfig(
    name="ms_proteomics_expert_agent",
    display_name="MS Proteomics Expert (DEPRECATED)",
    description="DEPRECATED: Use proteomics_expert instead. Routes to proteomics_expert.",
    factory_function="lobster.agents.proteomics.deprecated.ms_proteomics_alias",
    handoff_tool_name="handoff_to_ms_proteomics_expert_agent",
    handoff_tool_description="DEPRECATED: Routes to proteomics_expert.",
),

"affinity_proteomics_expert_agent": AgentRegistryConfig(
    name="affinity_proteomics_expert_agent",
    display_name="Affinity Proteomics Expert (DEPRECATED)",
    description="DEPRECATED: Use proteomics_expert instead. Routes to proteomics_expert.",
    factory_function="lobster.agents.proteomics.deprecated.affinity_proteomics_alias",
    handoff_tool_name="handoff_to_affinity_proteomics_expert_agent",
    handoff_tool_description="DEPRECATED: Routes to proteomics_expert.",
),
```

---

## KEY DIFFERENCES FROM TRANSCRIPTOMICS REFACTORING

| Aspect | Transcriptomics | Proteomics |
|--------|-----------------|------------|
| Original agents | 2 (32 + 15 tools) | 2 (8 + 8 tools) |
| Tool overlap | ~30% functional | ~87% functional |
| Unique tools | Many (clustering, annotation) | Few (2: peptide mapping, antibody validation) |
| Sub-agents needed | Yes (annotation, DE) | No (too few tools) |
| Services compliant | Yes (3-tuple) | No (2-tuple - must fix) |
| Architecture | Parent + 2 sub-agents | Single unified agent |
| Platform detection | Based on cell count, obs columns | Based on column names, n_vars |

---

## CONSTRAINTS

1. **Follow existing lobster patterns exactly** - study `data_expert.py`, `singlecell_expert.py`
2. **All tools must log with `ir=ir`** - create IR even if service doesn't return it
3. **Use `@tool` decorator** from langchain
4. **Deprecation warnings** - use `warnings.warn()` with `DeprecationWarning` and `stacklevel=2`
5. **Do NOT edit `pyproject.toml`** - dependency changes require human approval
6. **Add scientific warnings** in docstrings for simulated/approximate methods

---

## DELIVERABLES CHECKLIST

- [ ] `lobster/agents/proteomics/__init__.py` - Module exports
- [ ] `lobster/agents/proteomics/state.py` - ProteomicsExpertState class
- [ ] `lobster/agents/proteomics/platform_config.py` - Platform detection and configs
- [ ] `lobster/agents/proteomics/shared_tools.py` - 7 platform-aware tools
- [ ] `lobster/agents/proteomics/proteomics_expert.py` - Main agent factory (10 tools)
- [ ] `lobster/agents/proteomics/deprecated.py` - Alias wrappers
- [ ] Updated `config/agent_registry.py` - New + deprecated entries
- [ ] All imports verified working
- [ ] Scientific fixes applied (IR creation, warnings for simulated methods)

---

## EXECUTION STRATEGY

**Recommended**: Use lo-ass sub-agents for parallel implementation:

1. **Agent 1**: Create `platform_config.py` and `state.py`
2. **Agent 2**: Create `shared_tools.py` (7 platform-aware tools)
3. **Agent 3**: Create `proteomics_expert.py` (main agent + 3 platform-specific tools)
4. **Agent 4**: Create `deprecated.py` and update registry

Then orchestrator reviews, integrates, and verifies imports.
