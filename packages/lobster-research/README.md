# lobster-research

Literature discovery and data acquisition agents for scientific research workflows.

## Installation

```bash
pip install lobster-research
```

## Agents

| Agent | Description |
|-------|-------------|
| `research_agent` | Literature discovery specialist. PubMed/bioRxiv search, GEO/SRA dataset discovery, metadata extraction, publication queue management. |
| `data_expert_agent` | Data operations specialist. Queue-based downloads, modality management, local file loading, workspace orchestration. |

## Services

| Service | Purpose |
|---------|---------|
| ModalityDetectionService | Auto-detect data modality type from file characteristics |

## Features

### Research Agent (Online Operations)
- PubMed literature search with filters and related paper discovery
- bioRxiv and medRxiv preprint search with full-text access
- GEO dataset discovery with organism and platform filtering
- SRA run metadata extraction and download URL generation
- PRIDE proteomics repository integration
- Full-text content extraction from PMC articles
- Methods section parsing for computational parameter discovery
- Publication queue for batch processing of research papers
- Automatic extraction of associated dataset identifiers

### Data Expert Agent (Offline Operations)
- Execute downloads from pre-validated queue entries
- Zero online access boundary for security and reproducibility
- Multi-format file loading (CSV, TSV, H5AD, Excel)
- Modality listing, inspection, and validation
- Download strategy selection (AUTO, H5_FIRST, MATRIX_FIRST)
- Sample concatenation with union or intersection logic
- Failed download retry with exponential backoff
- Custom Python code execution for edge cases

### Platform Support
- 10x Genomics MTX format (matrix, barcodes, features)
- H5AD pre-processed AnnData files
- Kallisto and Salmon bulk RNA-seq quantification
- CSV and TSV generic delimited matrices
- MaxQuant, Olink, and generic proteomics formats

## Architecture

The research and data_expert agents implement a clean boundary pattern:

```
research_agent (ONLINE)              data_expert (OFFLINE)
-- Search literature                 -- Execute downloads
-- Discover datasets                 -- Load local files
-- Extract metadata/URLs             -- Manage modalities
-- Validate metadata                 -- Retry failed downloads
-- Create queue entries              -- Concatenate samples
         |                                    |
         ----------- Queue Entry -------------
                   (PENDING -> IN_PROGRESS -> COMPLETED)
```

## Requirements

- Python 3.12+
- lobster-ai >= 1.0.0

## Documentation

Full documentation: [docs.omics-os.com/docs/agents/research](https://docs.omics-os.com/docs/agents/research)

## License

MIT
