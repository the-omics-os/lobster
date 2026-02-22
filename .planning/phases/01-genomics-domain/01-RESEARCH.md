# Phase 1: Genomics Domain - Research

**Researched:** 2026-02-22
**Domain:** Genomics pipeline completion (GWAS) + clinical variant analysis (new child agent)
**Confidence:** HIGH

## Summary

Phase 1 completes the genomics domain by (a) filling three pipeline gaps in the parent genomics_expert (LD pruning, kinship, clumping), (b) creating a new variant_analysis_expert child agent for clinical genomics interpretation, (c) fixing provenance gaps in data loading tools, and (d) merging two redundant helper tools into one.

The existing codebase is well-structured. The parent agent already has 12 tools, 3 services (GenomicsQualityService, GWASService, VariantAnnotationService), 2 adapters (VCF, PLINK), and an established sgkit-based pipeline. The LD pruning TODO is explicitly flagged in the code. sgkit provides native functions for all three new parent tools: `ld_prune`, `genomic_relationship` (GRM/kinship), and we have the `ld_matrix` infrastructure for clumping. The variant_analysis_expert child follows the exact pattern of annotation_expert in lobster-transcriptomics.

For the child agent's external API tools (gnomAD, ClinVar, VEP), the existing `EnsemblService` and `VariantAnnotationService` already cover VEP consequence prediction, gnomAD allele frequencies via VEP colocated_variants, ClinVar significance via VEP colocated_variants, and SIFT/PolyPhen scores. The child agent's new tools can largely compose existing service methods or extend them with targeted API calls.

**Primary recommendation:** Implement in two plans: Plan 01 for parent agent improvements (GEN-01 through GEN-07, GEN-16, DOC-01 parent portion), Plan 02 for new child agent creation (GEN-08 through GEN-15, GEN-16 shared, DOC-01 child portion).

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| sgkit | >=0.7.0 | LD pruning, PCA, GRM (kinship), GWAS | Already in pyproject.toml; provides `ld_prune`, `genomic_relationship`, `pc_relate` |
| genebe | >=0.2.0 | Batch variant annotation (VEP, gnomAD, ClinVar, CADD) | Already optional dep; used by VariantAnnotationService |
| requests | >=2.31.0 | Ensembl REST API calls (VEP, sequence, xrefs) | Already in pyproject.toml; used by EnsemblService |
| cyvcf2 | >=0.30.0 | VCF parsing | Already in pyproject.toml; used by VCFAdapter |
| numpy | >=1.23.0 | Genotype matrix operations | Already in pyproject.toml |
| scipy | >=1.10.0 | Statistical tests (chi2, HWE) | Already in pyproject.toml |
| anndata | >=0.9.0 | Data storage (samples x variants) | Standard Lobster data format |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| xarray | latest | sgkit Dataset interop | Required by sgkit for Dataset operations |
| statsmodels | >=0.14.0 | Multiple testing correction (FDR) | Already in pyproject.toml for GWAS |
| pandas | >=1.5.0 | Variant dataframes, metadata manipulation | Already in pyproject.toml |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| sgkit `ld_prune` | PLINK2 subprocess | PLINK2 is faster for large datasets but requires binary installation; sgkit keeps everything in-process and Python-native |
| sgkit `genomic_relationship` | KING (via subprocess) | KING is industry standard but requires external binary; sgkit GRM is sufficient for Lobster's use case (flagging related pairs, not fine-grained inference) |
| Ensembl VEP REST API | Local VEP installation | Local VEP is faster for large datasets but requires 20GB+ cache; REST API is zero-config and sufficient for typical variant lists (<10K) |
| genebe for batch annotation | myvariant.info API | myvariant.info is faster but less comprehensive; genebe already integrated and working |

**Installation:** No new dependencies required. All libraries already in `packages/lobster-genomics/pyproject.toml`.

## Architecture Patterns

### Recommended Project Structure

The new child agent follows the established modular folder pattern:

```
packages/lobster-genomics/
├── lobster/
│   ├── agents/
│   │   └── genomics/
│   │       ├── __init__.py          # UPDATE: add variant_analysis_expert exports
│   │       ├── config.py            # UPDATE: add child agent constants
│   │       ├── genomics_expert.py   # UPDATE: add ld_prune, compute_kinship, clump_results; remove predict_variant_consequences, get_ensembl_sequence; add child_agents config; add IR to load tools; merge list/get_modality
│   │       ├── prompts.py           # UPDATE: add create_variant_analysis_expert_prompt(); update genomics_expert prompt
│   │       └── variant_analysis_expert.py  # NEW: child agent factory
│   └── services/
│       └── analysis/
│           ├── gwas_service.py              # UPDATE: add ld_prune_variants, compute_kinship, clump_gwas_results methods
│           └── variant_annotation_service.py # UPDATE: add normalize_variants, query_gnomad, query_clinvar, prioritize_variants methods
├── tests/
│   ├── agents/
│   │   └── test_variant_analysis_expert.py  # NEW
│   └── services/
│       └── analysis/
│           ├── test_gwas_service.py          # UPDATE: add tests for new methods
│           └── test_variant_annotation_service.py  # UPDATE: add tests for new methods
└── pyproject.toml                   # UPDATE: add variant_analysis_expert entry point
```

### Pattern 1: Parent-Child Agent Delegation (Established)

**What:** Parent agent owns delegation_tools list; graph builder creates lazy delegation tools from `child_agents` config.
**When to use:** genomics_expert -> variant_analysis_expert handoff after GWAS significant hits.
**How it works in the codebase:**

```python
# In genomics_expert.py AGENT_CONFIG (parent):
AGENT_CONFIG = AgentRegistryConfig(
    name="genomics_expert",
    ...
    child_agents=["variant_analysis_expert"],  # NEW
    supervisor_accessible=True,
)

# In variant_analysis_expert.py AGENT_CONFIG (child):
AGENT_CONFIG = AgentRegistryConfig(
    name="variant_analysis_expert",
    ...
    handoff_tool_name=None,       # Not directly accessible from supervisor
    handoff_tool_description=None,
    supervisor_accessible=False,  # Only via genomics_expert
)
```

Source: `lobster/agents/graph.py` lines 430-486 handle delegation_tools creation automatically from `child_agents` list. The parent factory receives `delegation_tools` parameter with lazy delegation tool closures.

### Pattern 2: Service 3-Tuple Return (Mandatory)

**What:** All service methods return `(AnnData, Dict[str, Any], AnalysisStep)`.
**When to use:** Every new service method (ld_prune_variants, compute_kinship, clump_gwas_results, normalize_variants, etc.).
**Example from existing code:**

```python
def ld_prune_variants(
    self,
    adata: anndata.AnnData,
    threshold: float = 0.2,
    window_size: int = 500,
    genotype_layer: str = "GT",
) -> Tuple[anndata.AnnData, Dict[str, Any], AnalysisStep]:
    # ... implementation using sgkit.ld_prune
    return adata_pruned, stats, ir
```

### Pattern 3: Knowledgebase Tool Factory (Established)

**What:** Tool factories in `lobster/tools/knowledgebase_tools.py` return `@tool`-decorated closures with lazy service loading.
**When to use:** Tools that wrap EnsemblService or UniProtService API calls.
**How GEN-05/GEN-06/GEN-15 work:** The existing `create_variant_consequence_tool` and `create_sequence_retrieval_tool` factories in `knowledgebase_tools.py` are already used by genomics_expert. For GEN-05/GEN-06, we REMOVE them from the parent's tool list. For GEN-15, we ADD `create_sequence_retrieval_tool` to the child agent.

### Pattern 4: sgkit Dataset Conversion (Critical for LD/Kinship)

**What:** Converting AnnData genotype data to sgkit xarray.Dataset requires specific dimension ordering and windowing.
**Critical detail:** sgkit `ld_prune` requires the dataset to be windowed FIRST via `window_by_variant` or `window_by_position`. The existing `_adata_to_sgkit` method in `GWASService` handles genotype conversion but does NOT add windowing. New LD pruning method must add windowing step.

```python
# Required sequence for ld_prune:
ds = self._adata_to_sgkit(adata, None, None, genotype_layer)
ds["call_dosage"] = ds["call_genotype"].sum(dim="ploidy")

# CRITICAL: Must add variant_contig for windowing
# variant_contig is required by window_by_variant
ds["variant_contig"] = ("variants", np.zeros(ds.sizes["variants"], dtype=int))

# CRITICAL: Must window before LD prune
ds = sg.window_by_variant(ds, size=window_size)

# Now LD prune will work
ds_pruned = sg.ld_prune(ds, threshold=threshold)
```

### Pattern 5: Provenance IR for Load Tools (GEN-07 Bug Fix)

**What:** `load_vcf` and `load_plink` currently call `data_manager.log_tool_usage()` WITHOUT `ir=` parameter, making them non-reproducible in notebook export.
**Fix pattern:**

```python
# Create IR for data loading
ir = AnalysisStep(
    operation="cyvcf2.VCF",
    tool_name="load_vcf",
    description="Load VCF file into AnnData",
    library="cyvcf2",
    code_template="""...""",
    imports=["import cyvcf2", "import anndata as ad"],
    parameters={...},
    parameter_schema={...},
    input_entities=[],
    output_entities=["adata"],
)
# Pass ir to log_tool_usage
data_manager.log_tool_usage(..., ir=ir)
```

### Anti-Patterns to Avoid

- **DO NOT add `lobster/__init__.py`** — PEP 420 namespace package requirement.
- **DO NOT import heavy libraries at module top in agent files** — AGENT_CONFIG must be defined before heavy imports for <50ms entry point discovery.
- **DO NOT use `try/except ImportError`** for feature gating — use `component_registry`.
- **DO NOT call `component_registry.get_service()` at module level** — causes slow startup.
- **DO NOT make tools that do network calls in the parent agent for clinical interpretation** — clinical genomics tools belong in variant_analysis_expert child.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| LD pruning | Custom r2 computation loop | `sgkit.ld_prune` with `window_by_variant` | Correct sparse matrix handling, maximal independent set algorithm |
| Kinship matrix | Manual IBS/IBD calculation | `sgkit.pc_relate` or `sgkit.genomic_relationship` | VanRaden GRM estimator handles missing data, ploidy |
| LD clumping | Custom greedy clumping | sgkit `ld_matrix` + custom clump logic on top | LD matrix computation is the expensive part; clumping logic is simple on top |
| VEP consequence prediction | Direct REST API calls | Existing `EnsemblService.get_variant_consequences` | Rate limiting, retries, caching already handled |
| Variant normalization | Manual left-align/split | cyvcf2 + BCF normalization OR simple numpy-based left-align | Edge cases in multiallelic splitting, indel representation |
| gnomAD frequency lookup | Direct gnomAD API | `VariantAnnotationService._annotate_with_genebe` or VEP colocated_variants | genebe and VEP already return gnomAD AF in annotation results |

**Key insight:** sgkit already provides the computationally expensive operations (LD matrix, GRM, PCA). Our service layer just needs to convert AnnData -> sgkit Dataset, call the sgkit function, and transfer results back. The `_adata_to_sgkit` helper already exists.

## Common Pitfalls

### Pitfall 1: sgkit LD Prune Requires Windowing

**What goes wrong:** Calling `sgkit.ld_prune(ds)` without prior windowing raises `ValueError: Dataset is not windowed`.
**Why it happens:** LD pruning operates within windows for memory efficiency. sgkit requires explicit `window_by_variant` or `window_by_position` call.
**How to avoid:** Always call `sg.window_by_variant(ds, size=500)` or `sg.window_by_position(ds, size=500_000)` before `sg.ld_prune(ds)`.
**Warning signs:** ValueError at runtime mentioning "not windowed".

### Pitfall 2: sgkit Requires variant_contig for Windowing

**What goes wrong:** Calling `window_by_variant` on a Dataset without `variant_contig` variable raises KeyError.
**Why it happens:** The existing `_adata_to_sgkit` method does not create `variant_contig`. Windowing needs it because windows never span contigs.
**How to avoid:** Add `variant_contig` to the Dataset in `_adata_to_sgkit` or in the LD prune method. For single-chromosome data, all zeros. For multi-chromosome, parse from adata.var["CHROM"].
**Warning signs:** KeyError on `variant_contig` during windowing.

### Pitfall 3: sgkit pc_relate Requires PCA First

**What goes wrong:** `sgkit.pc_relate` expects `sample_pca_projection` in the Dataset, which isn't present in raw data.
**Why it happens:** PC-Relate uses PCA projections to adjust for population structure when computing kinship.
**How to avoid:** The `compute_kinship` tool should either (a) run PCA internally before pc_relate, or (b) use `sgkit.genomic_relationship` (VanRaden GRM) which doesn't require PCA.
**Recommendation:** Use `sgkit.genomic_relationship` as primary kinship method (simpler, no PCA prereq). Mention pc_relate in docstring as alternative for large/structured populations.

### Pitfall 4: Ensembl VEP Rate Limits (15 req/sec)

**What goes wrong:** Variant annotation tools that call Ensembl VEP one-at-a-time hit 429 rate limits for variant lists > 100.
**Why it happens:** VEP REST API has a 15 requests/second limit per IP.
**How to avoid:** For bulk operations, use VEP POST endpoint (batch up to 200 variants per request). For single lookups (GEN-14), single GET is fine. The existing `EnsemblService` already has retry+backoff logic.
**Warning signs:** Frequent 429 responses, exponential backoff delays.

### Pitfall 5: GWAS Clumping vs LD Pruning Confusion

**What goes wrong:** Users/LLM confuses LD pruning (pre-analysis variant thinning) with LD clumping (post-GWAS locus identification).
**Why it happens:** Both use LD, but serve different purposes.
**How to avoid:** Clear tool names (`ld_prune` vs `clump_results`), distinct docstrings. LD pruning is a prerequisite step; clumping is a post-GWAS interpretive step.
**Warning signs:** User asking to "clump" before GWAS, or "prune" after GWAS results.

### Pitfall 6: Variant Normalization Complexity

**What goes wrong:** Implementing left-alignment and multiallelic splitting is more complex than it appears (handling indels, reference context, VCF spec compliance).
**Why it happens:** VCF variant representation has many edge cases (padding bases, ambiguous indels, star alleles).
**How to avoid:** For GEN-09 (normalize_variants), use a simple but correct approach: left-trim common prefixes/suffixes, split multiallelic to biallelic rows. Don't attempt full VCF normalization (that's bcftools territory). Document limitations.
**Warning signs:** Complex regex for variant parsing, attempting to handle structural variants.

### Pitfall 7: Child Agent Entry Point Registration

**What goes wrong:** New child agent not discovered at runtime.
**Why it happens:** Entry point not registered in `pyproject.toml`, or `AGENT_CONFIG` defined after heavy imports.
**How to avoid:** (1) Add entry point in pyproject.toml: `variant_analysis_expert = "lobster.agents.genomics.variant_analysis_expert:AGENT_CONFIG"`. (2) Define `AGENT_CONFIG` at module top before any heavy imports. (3) Set `supervisor_accessible=False`.
**Warning signs:** Agent not appearing in `component_registry.list_agents()`.

## Code Examples

### LD Pruning via sgkit (verified from sgkit source)

```python
# Source: sgkit/stats/ld.py lines 409-474
import sgkit as sg

# 1. Convert AnnData to sgkit Dataset (using existing _adata_to_sgkit)
ds = self._adata_to_sgkit(adata, None, None, genotype_layer)

# 2. Compute dosage
ds["call_dosage"] = ds["call_genotype"].sum(dim="ploidy")

# 3. Add variant_contig (required for windowing)
# Parse chromosome to integer index
chroms = adata.var["CHROM"].astype(str).values
unique_chroms = np.unique(chroms)
chrom_to_idx = {c: i for i, c in enumerate(unique_chroms)}
variant_contig = np.array([chrom_to_idx[c] for c in chroms], dtype=int)
ds["variant_contig"] = ("variants", variant_contig)

# 4. Window (REQUIRED before ld_prune)
ds = sg.window_by_variant(ds, size=window_size)

# 5. Prune
ds_pruned = sg.ld_prune(ds, threshold=threshold)

# 6. Get pruned variant indices to subset AnnData
pruned_variant_ids = set(ds_pruned["variant_id"].values)
keep_mask = np.isin(adata.var.index.values, list(pruned_variant_ids))
adata_pruned = adata[:, keep_mask].copy()
```

### Kinship via sgkit GRM (verified from sgkit source)

```python
# Source: sgkit/stats/grm.py lines 81-160
import sgkit as sg

# 1. Convert to sgkit Dataset
ds = self._adata_to_sgkit(adata, None, None, genotype_layer)

# 2. Compute dosage (required for GRM)
ds["call_dosage"] = ds["call_genotype"].sum(dim="ploidy").transpose("variants", "samples")

# 3. Compute GRM (VanRaden estimator)
ds = sg.genomic_relationship(ds, estimator="VanRaden")

# 4. Extract kinship matrix
grm = ds["stat_genomic_relationship"].values  # shape: (n_samples, n_samples)

# 5. Flag related pairs (kinship > threshold, typically 0.125 = 3rd degree)
related_pairs = []
for i in range(grm.shape[0]):
    for j in range(i + 1, grm.shape[1]):
        if grm[i, j] > kinship_threshold:
            related_pairs.append((adata.obs.index[i], adata.obs.index[j], grm[i, j]))

# 6. Store in AnnData
adata.obsm["kinship_matrix"] = grm
adata.uns["related_pairs"] = related_pairs
```

### GWAS Clumping (custom logic on top of sgkit LD matrix)

```python
# Clumping: group GWAS significant variants into independent loci by LD
# Source: Standard GWAS post-processing (Purcell et al. 2007)

# 1. Get significant variants sorted by p-value
sig_mask = adata.var["gwas_pvalue"] < pvalue_threshold
sig_variants = adata.var[sig_mask].sort_values("gwas_pvalue")

# 2. For each index variant (lowest p-value), find LD partners within window
clumps = []
claimed = set()

for idx_var in sig_variants.index:
    if idx_var in claimed:
        continue

    # Get LD partners within clump_kb window
    idx_pos = adata.var.loc[idx_var, "POS"]
    idx_chrom = adata.var.loc[idx_var, "CHROM"]

    # Find nearby variants on same chromosome
    nearby = sig_variants[
        (sig_variants["CHROM"] == idx_chrom) &
        (abs(sig_variants["POS"] - idx_pos) < clump_kb * 1000) &
        (~sig_variants.index.isin(claimed))
    ]

    # Compute LD with index variant (r2 from genotypes)
    # ... simplified; actual implementation uses sgkit ld_matrix for efficiency

    clump_members = [idx_var] + list(nearby.index)
    clumps.append({
        "index_variant": idx_var,
        "n_variants": len(clump_members),
        "members": clump_members,
        "index_pvalue": float(adata.var.loc[idx_var, "gwas_pvalue"]),
    })
    claimed.update(clump_members)
```

### Child Agent AGENT_CONFIG Pattern (verified from annotation_expert.py)

```python
# Source: packages/lobster-transcriptomics/lobster/agents/transcriptomics/annotation_expert.py
from lobster.config.agent_registry import AgentRegistryConfig

AGENT_CONFIG = AgentRegistryConfig(
    name="variant_analysis_expert",
    display_name="Variant Analysis Expert",
    description="Clinical variant interpretation: normalization, VEP consequences, gnomAD frequencies, ClinVar pathogenicity, variant prioritization",
    factory_function="lobster.agents.genomics.variant_analysis_expert.variant_analysis_expert",
    handoff_tool_name=None,       # Not directly accessible from supervisor
    handoff_tool_description=None,
    supervisor_accessible=False,  # Only via genomics_expert parent
    tier_requirement="free",
)
```

### Provenance IR for Load Tools (GEN-07 pattern)

```python
# Pattern for adding IR to load_vcf (same for load_plink)
ir = AnalysisStep(
    operation="cyvcf2.VCF.load",
    tool_name="load_vcf",
    description=f"Load VCF file: {n_samples} samples x {n_variants} variants",
    library="cyvcf2",
    code_template="""# Load VCF file
import cyvcf2
import anndata as ad
import numpy as np

vcf = cyvcf2.VCF({{ file_path | repr }})
# ... parsing logic ...
adata = ad.AnnData(X=genotype_matrix)
""",
    imports=["import cyvcf2", "import anndata as ad", "import numpy as np"],
    parameters={"file_path": file_path, "region": region, "filter_pass": filter_pass, ...},
    parameter_schema={...},
    input_entities=[],
    output_entities=["adata"],
)
data_manager.log_tool_usage(..., ir=ir)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| LD pruning via PLINK subprocess | sgkit.ld_prune (in-process Python) | sgkit 0.7+ | No external binary dependency |
| Kinship via KING binary | sgkit.genomic_relationship (VanRaden GRM) | sgkit 0.7+ | Pure Python, integrates with AnnData pipeline |
| Single-variant VEP lookups | genebe batch annotation + Ensembl VEP REST | Already implemented | ~100x faster for bulk annotation |
| Manual variant normalization | bcftools norm (external) or simplified in-process | Current | For Lobster, simplified in-process is sufficient |

**Deprecated/outdated:**
- `ld_prune=True` parameter on `calculate_pca`: Currently a no-op (logs warning). GEN-01 makes it a standalone tool, and BUG-15 (Phase 7) will address PCA defaults.
- `list_modalities` + `get_modality_info` as separate tools: GEN-04 merges them into `summarize_modality`.

## Open Questions

1. **LD Clumping Complexity**
   - What we know: Standard GWAS clumping uses index-variant + LD-partner grouping within a genomic window.
   - What's unclear: Whether to compute pairwise LD via sgkit `ld_matrix` for the full significant set, or use a simpler position-only window approach. Full LD-based clumping is more correct but much more expensive.
   - Recommendation: Implement position-based clumping first (group variants within `clump_kb` window on same chromosome, index = lowest p-value). Add LD-based refinement as v2 improvement. This matches PLINK's `--clump` behavior for most practical cases.

2. **gnomAD Population-Specific Frequencies**
   - What we know: gnomAD provides population-specific AFs (AFR, AMR, EAS, EUR, SAS). VEP colocated_variants includes some but not all populations.
   - What's unclear: Whether GEN-11 should return only global AF or per-population AFs. The existing VariantAnnotationService only extracts global gnomAD AF.
   - Recommendation: Return global AF by default with optional `population` parameter for per-population lookup. Use VEP colocated_variants for global, fall back to genebe for population-specific.

3. **Variant Normalization Scope**
   - What we know: GEN-09 specifies "left-align indels, split multiallelic". Full VCF normalization is complex (padding bases, reference context).
   - What's unclear: How much normalization is needed in practice. Most incoming VCFs are already normalized by upstream tools.
   - Recommendation: Implement basic normalization (left-trim padding, split multiallelic to biallelic). Skip full reference-aware left-alignment (requires reference genome FASTA). Document this as a limitation.

## Sources

### Primary (HIGH confidence)
- sgkit source code (`/Users/tyo/Omics-OS/lobster/.venv/lib/python3.13/site-packages/sgkit/`) - verified `ld_prune`, `genomic_relationship`, `pc_relate`, `window_by_variant`, `window_by_position` function signatures and requirements
- Existing codebase (`packages/lobster-genomics/`) - verified all existing services, tools, adapters, and entry points
- Existing child agent pattern (`packages/lobster-transcriptomics/lobster/agents/transcriptomics/annotation_expert.py`) - verified AGENT_CONFIG, factory function, delegation pattern
- Graph builder (`lobster/agents/graph.py`) - verified child_agents delegation mechanism

### Secondary (MEDIUM confidence)
- Ensembl VEP REST API documentation (verified through existing EnsemblService implementation and VariantAnnotationService VEP parsing)
- genebe Python package (verified through existing VariantAnnotationService integration)

### Tertiary (LOW confidence)
- GWAS clumping algorithm specifics (based on training data knowledge of PLINK --clump behavior; position-based approach is standard but LD-based variant should be validated against plink2 documentation)

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| GEN-01 | Add `ld_prune` tool — standalone LD pruning | sgkit `ld_prune` + `window_by_variant` verified in source. Requires `variant_contig` and windowing step before calling. Add method to GWASService. |
| GEN-02 | Add `compute_kinship` tool — pairwise kinship matrix | sgkit `genomic_relationship` (VanRaden GRM) verified. Simpler than pc_relate (no PCA prereq). Returns (n_samples, n_samples) matrix stored in `obsm["kinship_matrix"]`. |
| GEN-03 | Add `clump_results` tool — LD-clump GWAS results | Position-based clumping implemented as new GWASService method. Groups significant variants within window by chromosome, uses lowest p-value as index. |
| GEN-04 | Merge `list_modalities` + `get_modality_info` into `summarize_modality` | ModalityManagementService already has both methods. New tool combines: if modality_name given, show detail; if not, show list. Single tool reduces LLM confusion. |
| GEN-05 | Move `predict_variant_consequences` from parent to child | Remove from genomics_expert tools list (it's currently `create_variant_consequence_tool(data_manager)`). Add to variant_analysis_expert tools list instead. |
| GEN-06 | Move `get_ensembl_sequence` from parent to child | Remove from genomics_expert tools list (it's currently `create_sequence_retrieval_tool(data_manager)`). Add to variant_analysis_expert as `retrieve_sequence`. |
| GEN-07 | Add IR (provenance) to `load_vcf` and `load_plink` | Currently missing `ir=` parameter in `log_tool_usage()` calls. Add AnalysisStep creation with code_template for VCF/PLINK loading. Pattern verified from existing QC/GWAS IR creation. |
| GEN-08 | Create variant_analysis_expert child agent | Follow annotation_expert pattern exactly: AGENT_CONFIG at top, supervisor_accessible=False, factory function accepting data_manager + delegation_tools. Register in pyproject.toml entry points. Add `child_agents=["variant_analysis_expert"]` to parent AGENT_CONFIG. |
| GEN-09 | Implement `normalize_variants` tool | New VariantAnnotationService method. Left-trim common prefix/suffix from ref/alt, split multiallelic rows in adata.var to biallelic. Operates on adata.var columns (CHROM, POS, REF, ALT). |
| GEN-10 | Implement `predict_consequences` tool | Relocated from parent (GEN-05). Enhance with batch support. Use existing `create_variant_consequence_tool` factory but adapted for batch annotation on adata modality (not just single-variant lookup). Wraps VEP + genebe. |
| GEN-11 | Implement `query_population_frequencies` tool | New tool wrapping genebe/VEP for gnomAD AF lookup. For single variants: use EnsemblService VEP colocated_variants. For modality: leverage VariantAnnotationService annotate_variants (already extracts gnomad_af). |
| GEN-12 | Implement `query_clinical_databases` tool | New tool wrapping genebe/VEP for ClinVar lookup. Use VariantAnnotationService (already extracts clinvar_significance). For single-variant: use EnsemblService VEP colocated_variants. |
| GEN-13 | Implement `prioritize_variants` tool | New VariantAnnotationService method. Rank variants by composite score: VEP consequence severity (missense > synonymous), population frequency (rare > common), pathogenicity (CADD, SIFT, PolyPhen, ClinVar). Operates on annotated adata.var. |
| GEN-14 | Implement `lookup_variant` tool | New tool for single-variant comprehensive lookup by rsID or coordinates. Combines VEP consequences + gnomAD AF + ClinVar in one call. Uses EnsemblService.get_variant_consequences with full colocated_variants parsing. |
| GEN-15 | Implement `retrieve_sequence` tool | Relocated from parent (GEN-06). Use existing `create_sequence_retrieval_tool` factory. No changes needed to the tool itself, just move to child's tool list. |
| GEN-16 | Implement `summarize_modality` tool | Shared tool used by both parent and child. Combine list_modalities + get_modality_info into single tool. Use ModalityManagementService methods. Create as tool factory function for reuse. |
| DOC-01 | Update genomics_expert prompt for new tools + handoff | Update prompts.py: add new tools (ld_prune, compute_kinship, clump_results, summarize_modality), remove relocated tools (predict_variant_consequences, get_ensembl_sequence), add variant_analysis_expert handoff decision tree. Create new prompt function for variant_analysis_expert. |
</phase_requirements>

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - All libraries already installed and verified in codebase. sgkit function signatures verified from source.
- Architecture: HIGH - Parent-child delegation pattern verified from 2 existing examples (transcriptomics, proteomics). Graph builder mechanism verified.
- Pitfalls: HIGH - sgkit windowing/contig requirements verified from source code. Rate limiting concerns verified from existing EnsemblService.
- Service implementation: MEDIUM - LD clumping algorithm specifics need validation against real data. Variant normalization scope is a design decision.

**Research date:** 2026-02-22
**Valid until:** 2026-03-22 (stable domain, no expected library changes)
