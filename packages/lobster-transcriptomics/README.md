# lobster-transcriptomics

Single-cell and bulk RNA-seq analysis agents for the Lobster AI platform.

## Installation

```bash
pip install lobster-transcriptomics
```

## Agents

| Agent | Description |
|-------|-------------|
| `transcriptomics_expert` | Parent orchestrator for single-cell and bulk RNA-seq workflows. Handles QC, clustering, and coordinates sub-agents. |
| `annotation_expert` | Cell type annotation specialist. Automated and manual annotation, debris detection, annotation templates. |
| `de_analysis_expert` | Differential expression specialist. Pseudobulk aggregation, pyDESeq2, pathway enrichment. |

## Services

| Service | Purpose |
|---------|---------|
| EnhancedSingleCellService | Single-cell clustering and analysis workflows |
| PseudobulkService | Aggregate single-cell data to pseudobulk for DE analysis |
| DifferentialFormulaService | Formula-based differential expression with pyDESeq2 |
| QualityService | Comprehensive QC metrics calculation and filtering |
| PreprocessingService | Normalization, log transformation, and preprocessing |
| ManualAnnotationService | Manual cluster annotation with template support |
| AnnotationTemplates | Tissue-specific marker gene templates (PBMC, brain, lung) |
| BulkVisualizationService | Publication-quality plots for bulk RNA-seq results |

## Features

- Auto-detection of single-cell vs bulk RNA-seq data types
- Comprehensive QC metrics including gene counts, mitochondrial %, and ribosomal %
- Flexible filtering with data-type-appropriate default thresholds
- Leiden clustering with multi-resolution support and quality metrics
- UMAP visualization with automatic point size scaling
- Deviance-based highly variable gene selection
- Batch correction for multi-sample experimental designs
- Sub-clustering for heterogeneous cell populations
- Tissue-specific annotation templates for rapid cell type identification
- Pseudobulk aggregation from single-cell to bulk format
- Formula-based DE analysis supporting complex experimental designs
- GO and KEGG pathway enrichment for biological interpretation

## Requirements

- Python 3.12+
- lobster-ai >= 1.0.0

## Documentation

Full documentation: [docs.omics-os.com/docs/agents/transcriptomics](https://docs.omics-os.com/docs/agents/transcriptomics)

## License

MIT
