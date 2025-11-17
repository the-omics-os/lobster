# Migration Guides

This document provides comprehensive migration guides for upgrading between Lobster AI versions, covering breaking changes, new features, and recommended upgrade paths.

---

## Table of Contents

- [Migrating to v2.4](#migrating-to-v24)
- [Migrating to v2.3](#migrating-to-v23)
- [Version Feature Matrix](#version-feature-matrix)

---

## Migrating to v2.4

**Release Date:** January 2025
**Status:** Production-ready
**Breaking Changes:** Minor (mostly deprecations)

### Overview

Version 2.4 introduces major architectural improvements focused on content access, protein structure visualization, and download queue management. The release consolidates provider infrastructure and deprecates legacy services.

### Breaking Changes

#### 1. PublicationService Deprecated

**What changed:**
`PublicationService` and `UnifiedContentService` have been deprecated in favor of the new `ContentAccessService` with provider infrastructure.

**Migration path:**

```python
# ❌ OLD (v2.3 and earlier)
from lobster.tools.publication_service import PublicationService

service = PublicationService(data_manager)
result = service.get_publication_content(url)

# ✅ NEW (v2.4+)
from lobster.tools.content_access_service import ContentAccessService

service = ContentAccessService(data_manager)
result = await service.access_content(
    url=url,
    content_type="publication"
)
```

**Why upgrade:**
- Unified API for publications, datasets, and web content
- 5+ specialized providers (PubMed, PMC, GEO, bioRxiv, webpage)
- Better error handling and fallback strategies
- Improved caching (two-tier architecture)

#### 2. DataManagerV2 Parameter Changes

**What changed:**
`workspace_dir` parameter renamed to `workspace_path` for consistency across the codebase.

**Migration path:**

```python
# ❌ OLD (v2.3 and earlier)
data_manager = DataManagerV2(
    workspace_dir=Path("/path/to/workspace")
)

# ✅ NEW (v2.4+)
data_manager = DataManagerV2(
    workspace_path=Path("/path/to/workspace")
)
```

**Note:** The old parameter still works but triggers a deprecation warning. Update code to avoid warnings.

---

### New Features to Adopt

#### 1. ContentAccessService (Provider Architecture)

**What:** Unified service for accessing publications, datasets, and web content with specialized provider infrastructure.

**Why upgrade:**
- Single API for all content types
- 70-80% automatic resolution success rate for DOI/PMID
- Docling-powered PDF parsing with >90% Methods section detection
- Two-tier caching (30-50x speedup on cache hits)
- 5 specialized providers with automatic fallback

**How to migrate:**

```python
from lobster.tools.content_access_service import ContentAccessService

service = ContentAccessService(data_manager)

# Access publication (auto-resolves DOI/PMID to PDF)
pub_result = await service.access_content(
    url="10.1101/2024.08.29.610467",  # Bare DOI auto-detected
    content_type="publication"
)

# Access GEO dataset
geo_result = await service.access_content(
    url="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE123456",
    content_type="dataset"
)

# Access generic webpage
web_result = await service.access_content(
    url="https://example.com/protocol.html",
    content_type="webpage"
)
```

**When to use:**
- Research workflows requiring publication access
- Automated parameter extraction from papers
- Dataset metadata retrieval
- Web-based protocol parsing

**Related documentation:** [37-publication-intelligence-deep-dive.md](37-publication-intelligence-deep-dive.md)

#### 2. Protein Structure Visualization Expert

**What:** Full-featured protein structure analysis with PyMOL visualization, BioPython analysis, and omics data integration.

**Why upgrade:**
- Professional 3D visualizations (publication-ready)
- Link protein structures to gene expression / proteomics data
- RMSD comparisons for structural analysis
- Interactive and batch visualization modes
- Automatic PDB fetching with caching

**How to migrate:**

```python
# No migration needed - this is a new agent
# Access via natural language or direct handoff

# Example: Fetch and visualize
"Visualize protein structure 1AKE with cartoon representation"

# Example: Link to omics data
"Link protein structures to my RNA-seq data for top 50 genes"

# Example: Structural comparison
"Compare structures 1AKE and 4AKE and calculate RMSD"
```

**Tools available:**
- `fetch_protein_structure(pdb_id)` - Download from RCSB PDB
- `visualize_with_pymol(pdb_id, style, color_by)` - Create visualizations
- `analyze_protein_structure(pdb_id, analysis_type)` - Structural analysis
- `compare_structures(pdb_id1, pdb_id2)` - RMSD comparison
- `link_to_expression_data(modality_name)` - Omics integration

**When to use:**
- Visualizing protein structures for marker genes
- Comparing protein conformations (open/closed, mutants)
- Linking structural data to expression changes
- Publication-ready structure figures

**Related documentation:** [40-protein-structure-visualization.md](40-protein-structure-visualization.md)

#### 3. Download Queue System

**What:** JSONL-persisted download queue for robust multi-step data acquisition workflows.

**Why upgrade:**
- Persistent queue survives crashes and restarts
- Multi-agent handoff (research_agent → data_expert)
- Automatic retry logic with exponential backoff
- Queue status tracking (PENDING, IN_PROGRESS, COMPLETED, FAILED)
- Better separation of concerns (discovery vs loading)

**How to migrate:**

```python
# No direct migration - this is internal infrastructure
# Automatically used by research_agent and data_expert

# Example workflow:
# 1. Research agent finds dataset → creates queue entry
# 2. Supervisor detects pending queue → hands off to data_expert
# 3. Data expert processes queue → marks complete/failed
```

**Queue states:**
- `PENDING` - Discovered but not yet downloaded
- `IN_PROGRESS` - Currently downloading
- `COMPLETED` - Successfully loaded
- `FAILED` - Download or loading failed

**When to use:**
Automatically handled for:
- GEO dataset downloads
- SRA file retrieval
- Large multi-file downloads

**Related documentation:** [35-download-queue-system.md](35-download-queue-system.md)

#### 4. Enhanced Two-Tier Caching

**What:** Two-tier caching architecture with workspace-level and agent-level caches for publication content.

**Why upgrade:**
- 30-50x speedup on cached content access
- Workspace-level cache shared across sessions
- Agent-level cache for session-specific data
- Automatic cache invalidation and cleanup
- Reduced API calls to external services

**How to migrate:**

```python
# No migration needed - automatically enabled

# Caching tiers:
# Tier 1: Workspace cache (~/.lobster_ai/workspace_cache/)
#   - Shared across all sessions
#   - Publication PDFs, parsed content
#   - Persistent between runs

# Tier 2: Agent cache (in-memory)
#   - Session-specific
#   - Quick lookups
#   - Cleared on session end
```

**Cache locations:**
- Workspace cache: `workspace_dir/cached_content/`
- Cache index: `workspace_dir/content_cache_index.json`

**Performance:**
- First access (cache miss): 8-15 seconds (PDF download + parsing)
- Subsequent access (cache hit): 0.2-0.5 seconds
- Speedup: **30-50x**

**When to use:**
Automatically used for:
- Publication content access
- GEO metadata retrieval
- Web content parsing

**Related documentation:** [39-two-tier-caching-architecture.md](39-two-tier-caching-architecture.md)

#### 5. Provider Infrastructure

**What:** Modular provider system for content access with automatic fallback strategies.

**Why upgrade:**
- Clean separation of concerns (each provider has one job)
- Easy to add new providers
- Automatic fallback chains (PMC → bioRxiv → Publisher)
- Provider-specific optimizations

**Available providers:**
1. **PubMedProvider** - PubMed search and metadata
2. **PMCProvider** - PubMed Central full-text access
3. **GEOProvider** - GEO dataset discovery and URLs
4. **WebPageProvider** - Generic web content scraping
5. **AbstractProvider** - Quick abstract-only access

**How to migrate:**

```python
# Direct provider access (advanced usage)
from lobster.tools.providers import PMCProvider, GEOProvider

pmc = PMCProvider()
result = pmc.fetch_content("PMC12345678")

geo = GEOProvider()
metadata = geo.get_metadata("GSE123456")
```

**When to use:**
- Most users should use `ContentAccessService` (automatic provider selection)
- Direct provider access for custom workflows
- Extending with new providers

**Related documentation:** [37-publication-intelligence-deep-dive.md#provider-architecture](37-publication-intelligence-deep-dive.md)

---

### Upgrade Steps

#### Step 1: Update Dependencies

```bash
cd lobster
git pull origin main
make clean-install
```

#### Step 2: Update Code (Breaking Changes)

Search for deprecated usage:

```bash
# Find PublicationService usage
grep -r "PublicationService" --include="*.py" .

# Find workspace_dir parameter
grep -r "workspace_dir=" --include="*.py" .
```

Replace with v2.4 equivalents as shown in "Breaking Changes" section above.

#### Step 3: Install PyMOL (Optional)

For protein structure visualization:

```bash
# Automated installation
make install-pymol

# Or manual installation
# macOS: brew install brewsci/bio/pymol
# Linux: sudo apt-get install pymol
```

#### Step 4: Test Workflows

```bash
# Run test suite
make test

# Test new features
lobster chat
> "Access publication 10.1038/s41586-021-12345-6"
> "Visualize protein structure 1AKE"
> "Show download queue status"
```

#### Step 5: Update Notebooks

If using exported notebooks, regenerate with new IR:

```bash
# Export updated pipeline
/pipeline export my_analysis.ipynb

# Notebooks now include:
# - ContentAccessService imports
# - Protein structure tools
# - Download queue status checks
```

---

### Compatibility Notes

**Backward Compatibility:**
- v2.4 is mostly backward compatible with v2.3
- Deprecated services still work but trigger warnings
- Old notebooks continue to run (but consider regenerating)

**Forward Compatibility:**
- v2.3 workspaces work in v2.4 without migration
- Cache format is forward-compatible
- Session files (.json) are compatible

**Cloud vs Local:**
- All v2.4 features work in both modes
- Protein visualization: Limited in cloud (no interactive mode)
- All other features: Full parity

---

### Performance Improvements

| Feature | v2.3 Performance | v2.4 Performance | Improvement |
|---------|------------------|------------------|-------------|
| Publication access (cache hit) | 2-3s | 0.2-0.5s | **4-15x faster** |
| DOI resolution | 70% success | 75% success | +5% success rate |
| Methods section detection | >90% | >90% | Maintained |
| Download retry logic | Manual | Automatic | N/A |
| Provider fallback | 2 strategies | 5 providers | +150% coverage |

---

### Troubleshooting v2.4

#### Issue: ContentAccessService import error

**Symptom:**
```python
ImportError: cannot import name 'ContentAccessService'
```

**Solution:**
```bash
# Ensure v2.4+ is installed
pip show lobster-ai | grep Version

# Reinstall if needed
make clean-install
```

#### Issue: PyMOL not found

**Symptom:**
```
PyMOL not found. Install with: brew install brewsci/bio/pymol
```

**Solution:**
```bash
# Automated installation
make install-pymol

# Or manual installation (see above)

# Or run without execution (generates scripts only)
visualize_with_pymol("1AKE", execute=False)
```

#### Issue: Download queue entries stuck in PENDING

**Symptom:**
Queue shows PENDING entries but data_expert never picks them up.

**Solution:**
```bash
# Check queue status
/workspace queue

# Manual handoff to data_expert if supervisor doesn't detect
"Load the pending dataset GSE123456"

# Clear failed queue entries
/workspace queue clear-failed
```

#### Issue: Cache directory permissions

**Symptom:**
```
PermissionError: [Errno 13] Permission denied: 'workspace_dir/cached_content/'
```

**Solution:**
```bash
# Fix permissions
chmod -R u+w ~/.lobster_ai/workspaces/

# Or specify different workspace
lobster chat --workspace /tmp/my_workspace
```

---

## Migrating to v2.3

**Release Date:** January 2025
**Status:** Production-ready
**Breaking Changes:** Moderate (agent registry refactoring)

### Overview

Version 2.3 introduces structure-aware PDF parsing with Docling, formula-based differential expression analysis, and automatic agent registry discovery. The release significantly improves publication intelligence and experimental design handling.

### Breaking Changes

#### 1. Agent Registry Configuration

**What changed:**
Agent registration now uses `AGENT_REGISTRY` in `config/agent_registry.py` instead of manual graph construction.

**Migration path:**

```python
# ❌ OLD (v2.2 and earlier)
# Manual agent wiring in graph.py
def create_graph():
    singlecell = create_singlecell_expert(...)
    bulk = create_bulk_expert(...)
    # ... manual wiring

# ✅ NEW (v2.3+)
# Registry-based discovery in config/agent_registry.py
AGENT_REGISTRY = {
    "singlecell_expert": AgentConfig(
        name="singlecell_expert",
        factory_function="lobster.agents.singlecell_expert.singlecell_expert",
        # ... auto-discovered
    )
}
```

**Why upgrade:**
- Automatic supervisor handoff tool generation
- Dynamic agent discovery
- Easier to add new agents
- Centralized configuration

**Related documentation:** [36-supervisor-configuration.md](36-supervisor-configuration.md), [09-creating-agents.md](09-creating-agents.md)

#### 2. Handoff Tool Pattern

**What changed:**
Handoff tools are now auto-generated from `AGENT_REGISTRY` instead of manually defined.

**Migration path:**

```python
# ❌ OLD (v2.2 and earlier)
@tool
def handoff_to_singlecell_expert():
    """Manually defined handoff tool"""
    pass

# ✅ NEW (v2.3+)
# Auto-generated from AGENT_REGISTRY
# No manual tool definition needed
# Handoff tools created by create_bioinformatics_graph()
```

**Note:** If you created custom agents, update them to use the registry pattern. See [26-tutorial-custom-agent.md](26-tutorial-custom-agent.md).

---

### New Features to Adopt

#### 1. Docling PDF Parsing

**What:** Structure-aware PDF parsing that intelligently extracts Methods sections, parameter tables, and formulas.

**Why upgrade:**
- >90% Methods section hit rate (vs ~30% with PyPDF2)
- Complete section extraction (no 10K char limit)
- Parameter tables extracted as pandas DataFrames
- Mathematical formulas preserved (LaTeX)
- Document structure awareness (headings, lists, tables)

**How to migrate:**

```python
# Automatic - no migration needed
# Already used by research_agent.extract_paper_methods()

# Example usage:
"Extract computational methods from DOI 10.1101/2024.08.29.610467"

# Returns:
# - Software packages with versions
# - Parameter tables (as DataFrames)
# - QC steps and thresholds
# - Mathematical formulas (LaTeX)
# - Analysis workflow descriptions
```

**Input formats (auto-detected):**
- Bare DOI: `"10.1101/2024.08.29.610467"`
- DOI with prefix: `"DOI:10.1038/s41586-021-12345-6"`
- PMID: `"PMID:12345678"`
- PMC ID: `"PMC12345678"`
- Direct PDF URL

**When to use:**
- Reproducing published analyses
- Parameter extraction for similar datasets
- Method validation and comparison

**Related documentation:** [37-publication-intelligence-deep-dive.md](37-publication-intelligence-deep-dive.md)

#### 2. Two-Tier Publication Access

**What:** Optimized access strategy with quick abstract retrieval (Tier 1) and full PDF parsing (Tier 2).

**Why upgrade:**
- Fast initial assessment (abstract-only, 1-2 seconds)
- Deep dive only when needed (full PDF, 8-15 seconds)
- Reduces unnecessary PDF downloads
- Better user experience (progressive disclosure)

**How to migrate:**

```python
# Tier 1: Quick abstract (fast)
result = research_agent.get_quick_abstract("10.1101/2024.08.29.610467")
# Returns: title, authors, abstract, journal, year

# Tier 2: Full content (slower, more detailed)
result = research_agent.get_publication_overview("10.1101/2024.08.29.610467")
# Returns: full text, methods, results, figures, tables
```

**Decision flow:**
1. User asks for publication info → Quick abstract (Tier 1)
2. User needs specific details (methods, parameters) → Full content (Tier 2)

**When to use:**
- Tier 1: Literature search, relevance checking, quick summaries
- Tier 2: Parameter extraction, method reproduction, detailed analysis

**Related documentation:** [37-publication-intelligence-deep-dive.md#two-tier-access](37-publication-intelligence-deep-dive.md)

#### 3. Formula-Based Differential Expression

**What:** Support for complex experimental designs with formula-based DE using pyDESeq2.

**Why upgrade:**
- Handle complex designs (multi-factor, interactions, covariates)
- Agent-guided formula construction (no R knowledge needed)
- Automatic formula validation
- Support for continuous and categorical variables

**How to migrate:**

```python
# ❌ OLD (v2.2 and earlier)
# Only simple two-group comparisons
"Compare treatment vs control"

# ✅ NEW (v2.3+)
# Complex designs with formulas
"Analyze differential expression with formula: ~ batch + treatment + treatment:timepoint"

# Agent-guided construction
"Set up a differential expression analysis with batch correction"
# Agent asks: "Which columns represent batch effect?"
# Agent asks: "What's your main comparison?"
# Agent suggests: "~ batch + condition"
```

**Supported design patterns:**
- Simple: `~ condition`
- Batch correction: `~ batch + condition`
- Multi-factor: `~ genotype + treatment`
- Interactions: `~ treatment + timepoint + treatment:timepoint`
- Continuous covariates: `~ age + condition`

**When to use:**
- Complex experimental designs (multi-factor, batch effects)
- Time-course experiments
- Studies with technical/biological covariates

**Related documentation:** [32-agent-guided-formula-construction.md](32-agent-guided-formula-construction.md), [24-tutorial-bulk-rnaseq.md](24-tutorial-bulk-rnaseq.md)

#### 4. WorkspaceContentService

**What:** Type-safe caching service for research content (publications, datasets, metadata) in workspace.

**Why upgrade:**
- Pydantic schema validation (prevents corrupt cache)
- Enum-based type safety
- Centralized content management
- Better error handling

**How to migrate:**

```python
# Automatic - used internally by research_agent
# No direct migration needed

# Content types:
# - PUBLICATION (papers, abstracts, PDFs)
# - DATASET (GEO, SRA metadata)
# - SEARCH_RESULTS (PubMed search)
# - METHOD_EXTRACTION (parsed methods)
# - WEBPAGE (generic web content)
```

**Storage location:**
- Workspace cache: `workspace_dir/cached_content/`
- Index file: `workspace_dir/content_cache_index.json`

**When to use:**
Automatically used for:
- Publication caching (Docling outputs)
- Dataset metadata (GEO, SRA)
- Search results persistence

**Related documentation:** [38-workspace-content-service.md](38-workspace-content-service.md)

#### 5. Enhanced FTP Download Handling

**What:** Robust FTP downloads with automatic retry logic and corruption detection.

**Why upgrade:**
- Exponential backoff (2s, 4s, 8s delays)
- Chunked downloads for large files (>70MB)
- Integrity verification (size checks)
- Automatic cleanup of failed downloads
- Clear error messages

**How to migrate:**

```python
# No migration needed - automatic
# Applies to:
# - GEO FTP downloads
# - SRA file retrieval
# - Any FTP-based data access

# Features:
# - 3 retry attempts with backoff
# - Chunked transfer for large files
# - Connection pooling
# - Progress tracking
```

**Performance:**
- Small files (<10MB): Single transfer
- Medium files (10-70MB): Standard transfer with retry
- Large files (>70MB): Chunked transfer (10MB chunks)

**When to use:**
Automatically enabled for:
- GEO supplementary files
- Raw sequencing data (FASTQ)
- Large matrix files

**Related documentation:** [28-troubleshooting.md#ftp-download-failures](28-troubleshooting.md)

#### 6. VDJ/TCR/BCR Data Support

**What:** Fixed duplicate barcode handling and proper VDJ data loading.

**Why upgrade:**
- Correct single-cell immune repertoire analysis
- Proper TCR/BCR data integration
- No more "duplicate barcode" errors

**How to migrate:**

```python
# No migration needed - automatic fix
# Applies to:
# - 10x VDJ data
# - TCR/BCR repertoire data
# - Clonotype analysis

# Example usage:
"Load my 10x VDJ data with gene expression"
```

**Fixed issues:**
- Duplicate barcodes in VDJ data (now properly deduplicated)
- Missing VDJ annotations (now correctly loaded)
- Integration with gene expression (proper alignment)

**When to use:**
- Single-cell immune repertoire studies
- TCR/BCR clonotype tracking
- T-cell/B-cell analysis

**Related documentation:** [28-troubleshooting.md#vdj-data-duplicate-barcode-errors](28-troubleshooting.md)

---

### Upgrade Steps

#### Step 1: Update Installation

```bash
cd lobster
git pull origin main
make clean-install
```

#### Step 2: Update Custom Agents (If Any)

If you created custom agents, update to registry pattern:

```python
# In config/agent_registry.py
AGENT_REGISTRY = {
    "my_custom_agent": AgentConfig(
        name="my_custom_agent",
        display_name="My Custom Agent",
        description="Custom analysis agent",
        factory_function="my_module.my_agent.my_custom_agent",
        handoff_tool_name="handoff_to_my_custom_agent",
        handoff_tool_description="Use when user requests custom analysis"
    )
}
```

See [26-tutorial-custom-agent.md](26-tutorial-custom-agent.md) for complete migration guide.

#### Step 3: Test Publication Access

```bash
lobster chat

# Test Tier 1 (quick abstract)
> "Get abstract for DOI 10.1101/2024.08.29.610467"

# Test Tier 2 (full content)
> "Extract computational methods from that paper"

# Test auto-detection
> "Analyze paper 10.1038/s41586-021-12345-6"  # Bare DOI
> "Get info on PMID:12345678"  # PMID
```

#### Step 4: Test Formula-Based DE

```bash
lobster chat

# Load bulk RNA-seq data
> "Load my Kallisto quantification files from results/"

# Test formula-based DE
> "Set up differential expression with formula: ~ batch + treatment"

# Agent-guided construction
> "I want to compare treatments while accounting for batch effects"
```

#### Step 5: Verify Workspaces

```bash
# v2.3 workspaces are backward compatible
lobster chat --workspace ~/.lobster_ai/workspaces/my_v2.2_workspace

# Check for warnings (should be none)
# Verify data loads correctly
```

---

### Compatibility Notes

**Backward Compatibility:**
- v2.3 workspaces work with v2.2 (but lose new features)
- v2.2 workspaces work with v2.3 (upgrade seamless)

**Breaking:**
- Custom agents require registry migration
- Manual handoff tools no longer work

**Recommended:**
- Regenerate exported notebooks (new IR includes formulas)
- Update custom agent code to registry pattern

---

### Performance Improvements

| Feature | v2.2 Performance | v2.3 Performance | Improvement |
|---------|------------------|------------------|-------------|
| Methods section detection | ~30% success | >90% success | **3x better** |
| Publication parsing | 10K char limit | Full document | **Unlimited** |
| DOI resolution | 60% success | 70-80% success | +10-20% success |
| FTP downloads | No retry | 3 retries + backoff | Robust |
| VDJ data loading | Errors | Success | Fixed |

---

### Troubleshooting v2.3

#### Issue: Agent registry not found

**Symptom:**
```python
KeyError: 'my_custom_agent' not found in AGENT_REGISTRY
```

**Solution:**
Add agent to `config/agent_registry.py`:
```python
AGENT_REGISTRY = {
    "my_custom_agent": AgentConfig(...)
}
```

#### Issue: Docling installation error

**Symptom:**
```
ImportError: No module named 'docling'
```

**Solution:**
```bash
# Reinstall with v2.3+ dependencies
make clean-install

# Or manual install
pip install docling>=1.0.0
```

#### Issue: Formula validation error

**Symptom:**
```
FormulaError: Column 'batch' not found in adata.obs
```

**Solution:**
```python
# Check available columns
print(adata.obs.columns)

# Use correct column name
"Analyze with formula: ~ Batch + Treatment"  # Match exact column names
```

#### Issue: FTP download timeouts

**Symptom:**
```
FTPError: Connection timed out after 3 retries
```

**Solution:**
```bash
# Check network connection
ping ftp.ncbi.nlm.nih.gov

# Try manual download to verify FTP access
# Retry later (NCBI FTP can be slow during peak hours)
```

---

## Version Feature Matrix

Comprehensive comparison of features across versions.

### Feature Availability

| Feature | v2.2 | v2.3 | v2.4 | Notes |
|---------|------|------|------|-------|
| **Publication Intelligence** |
| PyPDF2 parsing | ✅ | ⚠️ Deprecated | ❌ | Replaced by Docling |
| Docling parsing | ❌ | ✅ | ✅ | >90% success rate |
| Two-tier access | ❌ | ✅ | ✅ | Quick abstract + full content |
| DOI auto-detection | ⚠️ Limited | ✅ | ✅ | Bare DOI, PMID, PMC |
| Parameter table extraction | ❌ | ✅ | ✅ | As pandas DataFrames |
| Formula extraction | ❌ | ✅ | ✅ | LaTeX format |
| **Content Access** |
| PublicationService | ✅ | ✅ | ⚠️ Deprecated | Use ContentAccessService |
| UnifiedContentService | ❌ | ✅ | ⚠️ Deprecated | Use ContentAccessService |
| ContentAccessService | ❌ | ❌ | ✅ | Provider infrastructure |
| Provider system | ❌ | ⚠️ Partial | ✅ | 5 specialized providers |
| **Data Management** |
| Basic workspace | ✅ | ✅ | ✅ | H5AD + MuData |
| WorkspaceContentService | ❌ | ✅ | ✅ | Type-safe caching |
| Download queue | ❌ | ❌ | ✅ | JSONL persistence |
| Two-tier caching | ❌ | ⚠️ Basic | ✅ | 30-50x speedup |
| **Analysis Capabilities** |
| Simple DE (two-group) | ✅ | ✅ | ✅ | Basic comparisons |
| Formula-based DE | ❌ | ✅ | ✅ | Complex designs |
| Agent-guided formulas | ❌ | ✅ | ✅ | Interactive construction |
| Protein visualization | ❌ | ❌ | ✅ | PyMOL + BioPython |
| **Infrastructure** |
| Manual agent registry | ✅ | ⚠️ Deprecated | ❌ | Use AGENT_REGISTRY |
| Auto agent discovery | ❌ | ✅ | ✅ | Registry-based |
| FTP retry logic | ❌ | ✅ | ✅ | Exponential backoff |
| VDJ data support | ⚠️ Broken | ✅ | ✅ | Fixed duplicates |
| Chunked FTP downloads | ❌ | ✅ | ✅ | For files >70MB |

**Legend:**
- ✅ Full support
- ⚠️ Partial support / Deprecated
- ❌ Not available

### Recommended Upgrade Path

**Currently on v2.2:**
1. Upgrade to v2.3 first (breaking changes in agent registry)
2. Update custom agents to registry pattern
3. Test formula-based DE workflows
4. Then upgrade to v2.4

**Currently on v2.3:**
1. Direct upgrade to v2.4 (minimal breaking changes)
2. Update ContentAccessService usage
3. Install PyMOL if using protein visualization

**Starting fresh:**
- Install v2.4 directly (latest production release)

### Cloud vs Local Feature Parity

| Feature | Local Mode | Cloud Mode | Notes |
|---------|-----------|------------|-------|
| ContentAccessService | ✅ | ✅ | Full parity |
| WorkspaceContentService | ✅ | ✅ | Full parity |
| Protein visualization (batch) | ✅ | ✅ | Image generation |
| Protein visualization (interactive) | ✅ | ⚠️ Limited | No GUI in cloud |
| Download queue | ✅ | ✅ | Full parity |
| Two-tier caching | ✅ | ✅ | Full parity |
| Formula-based DE | ✅ | ✅ | Full parity |
| Docling parsing | ✅ | ✅ | Full parity |

**Interactive PyMOL in cloud:** Not supported due to GUI requirements. Use batch mode for image generation.

---

## Getting Help

- **Documentation:** [Wiki Home](Home.md)
- **Troubleshooting:** [28-troubleshooting.md](28-troubleshooting.md)
- **FAQ:** [29-faq.md](29-faq.md)
- **GitHub Issues:** [github.com/the-omics-os/lobster/issues](https://github.com/the-omics-os/lobster/issues)

---

**Last Updated:** 2025-11-16
**Document Version:** 1.0
**Maintainer:** Lobster Development Team
