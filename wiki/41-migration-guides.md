# Release Notes & Migration Guides

This document provides comprehensive release notes for Lobster AI versions, covering new features, breaking changes, and recommended upgrade paths for future releases.

---

## Table of Contents

- [v0.2 Release Notes](#v02-release-notes)
- [Future Migrations](#future-migrations)

---

## v0.2 Release Notes

**Release Date:** January 2025
**Status:** First Public Release (Production-ready)
**Breaking Changes:** None (first release)

### Overview

Version 0.2 is the **first public release** of Lobster AI, providing a production-ready bioinformatics analysis platform with AI-powered agents, comprehensive multi-omics support, and professional tooling for computational biology research.

---

### Key Features in v0.2

#### 🔌 Content Intelligence & Publication Access

**ContentAccessService** - Unified publication, dataset, and web content access:
- 5 specialized providers (PubMed, PMC, GEO, bioRxiv, generic web)
- 70-80% automatic DOI/PMID resolution success rate
- Docling-powered PDF parsing with >90% Methods section detection
- Two-tier caching architecture (30-50x speedup on cache hits)
- Smart fallback strategies across providers

**Usage example:**
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
```

**Documentation:** [37-publication-intelligence-deep-dive.md](37-publication-intelligence-deep-dive.md)

---

#### 🧬 Protein Structure Visualization

Full-featured protein structure analysis with PyMOL integration:
- Professional 3D visualizations (publication-ready)
- Link protein structures to gene expression / proteomics data
- RMSD comparisons for structural analysis
- Interactive and batch visualization modes
- Automatic PDB fetching with caching

**Natural language usage:**
```bash
"Visualize protein structure 1AKE with cartoon representation"
"Link protein structures to my RNA-seq data for top 50 genes"
```

**Documentation:** [40-protein-structure-visualization.md](40-protein-structure-visualization.md)

---

#### 📥 Download Queue System

Robust multi-step data acquisition with JSONL persistence:
- Agent handoff pattern for complex downloads
- Persistent queue (survives crashes/restarts)
- Status tracking (PENDING → IN_PROGRESS → COMPLETED/FAILED)
- Automatic retry logic with exponential backoff
- Multi-source dataset support (GEO, SRA, PRIDE, ENA)

**Key benefits:**
- Decouples dataset discovery from loading
- Enables background downloads
- Fault-tolerant with automatic recovery

**Documentation:** [35-download-queue-system.md](35-download-queue-system.md)

---

#### 🔄 Workspace & Data Management

**Workspace Restoration:**
- Seamless session continuity across restarts
- Pattern-based dataset loading (smart memory management)
- Automatic state tracking and recovery
- Enhanced Data Expert Agent with restoration tools

**WorkspaceContentService:**
- Type-safe caching for research content (publications, datasets)
- Structured storage with provenance tracking
- Fast workspace-level access (no global cache pollution)

**Documentation:**
- [31-data-expert-agent-enhancements.md](31-data-expert-agent-enhancements.md)
- [38-workspace-content-service.md](38-workspace-content-service.md)

---

#### 🧪 Formula-Based Differential Expression

Complex experimental designs with R-style formulas:
- pyDESeq2 integration for bulk RNA-seq
- Multi-factor designs (`~ condition + batch + condition:batch`)
- Agent-guided formula construction (interactive)
- Batch effect modeling and correction

**Natural language usage:**
```bash
"Run differential expression with formula '~ treatment + timepoint'"
"Compare conditions accounting for batch effects"
```

**Documentation:** [32-agent-guided-formula-construction.md](32-agent-guided-formula-construction.md)

---

#### 🏗️ Agent Infrastructure

**Agent Registry Auto-Discovery:**
- Dynamic agent configuration and registration
- Modular agent system with zero-config discovery
- Centralized tool routing and delegation

**Enhanced CLI:**
- Arrow navigation and command history
- Professional orange branding
- Rich terminal interface with syntax highlighting
- Optimized startup and processing performance

---

### Installation

```bash
# Install via PyPI (recommended)
pip install lobster-ai

# Configure API keys
cat > .env << 'EOF'
ANTHROPIC_API_KEY=sk-ant-api03-your-key-here
EOF

# Run
lobster chat
```

**Complete installation guide:** [02-installation.md](02-installation.md)

---

### Architecture Highlights

**Multi-Agent System:**
- 8+ specialized AI agents (supervisor, research, data expert, single-cell, bulk RNA-seq, proteomics, etc.)
- LangGraph-based coordination with centralized registry
- Natural language interface with context-aware routing

**Data Management:**
- DataManagerV2 for multi-modal orchestration (H5AD, MuData)
- W3C-PROV compliant provenance tracking
- S3-ready backends for cloud deployment

**Analysis Services:**
- Single-cell RNA-seq: QC, clustering, annotation, trajectory, pseudobulk
- Bulk RNA-seq: pyDESeq2 differential expression, complex designs
- Mass spectrometry proteomics: DDA/DIA workflows, missing value handling
- Affinity proteomics: Olink/antibody arrays, NPX handling

**Documentation:** [18-architecture-overview.md](18-architecture-overview.md)

---

### Feature Availability

All features in v0.2 are available in both **local** and **cloud** deployment modes, with the following exceptions:

| Feature | Local | Cloud |
|---------|:-----:|:-----:|
| Interactive PyMOL visualization | ✅ | ⚠️ |
| Batch image generation | ✅ | ✅ |

> **Note:** Interactive PyMOL requires local GUI support. Cloud mode supports batch image generation only.

---

### Known Limitations

1. **Rate Limits:** Claude API has conservative limits for new accounts. For production, use AWS Bedrock.
2. **Memory:** Large datasets (>10GB) may require cloud deployment for optimal performance.
3. **Windows:** Native installation requires WSL2. Docker is recommended for Windows users.

**Troubleshooting:** [28-troubleshooting.md](28-troubleshooting.md)

---

## Agent Architecture Migration (v0.2 → v0.3)

**Status:** Deprecation phase (v0.2.x) → Removal (v0.3.0)

### Background
In v0.2, we unified transcriptomics analysis:
- **Before:** `singlecell_expert` + `bulk_rnaseq_expert` (2 agents)
- **After:** `transcriptomics_expert` (unified agent)

### Migration for Test Code

#### Old API (deprecated):
```python
from lobster.agents.singlecell_expert import singlecell_expert
agent = singlecell_expert(data_manager)
```

#### New API (v0.2+):
```python
from lobster.agents.transcriptomics.transcriptomics_expert import transcriptomics_expert
agent = transcriptomics_expert(data_manager)
```

### Timeline
- **v0.2.0:** Deprecation warnings added
- **v0.2.x:** Both APIs available (current)
- **v0.3.0:** Old agents removed (Q2 2025)

### User Impact
- **End users:** No action required (supervisor handles routing)
- **Test code:** Update imports to use `transcriptomics_expert`
- **Custom integrations:** Update agent factory references

---

## Future Migrations

This section will be updated as new versions are released. Check back for:
- Breaking changes and deprecation notices
- New feature adoption guides
- Version-specific upgrade paths

For questions or issues, see:
- [GitHub Issues](https://github.com/the-omics-os/lobster/issues)
- [FAQ](29-faq.md)
- [Troubleshooting Guide](28-troubleshooting.md)

---

*Last updated: December 2025 - v0.2 Release*
