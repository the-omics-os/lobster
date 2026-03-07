# lobster-metadata

Sample metadata management and harmonization for multi-omics datasets.

## Installation

```bash
pip install lobster-metadata
```

## Agents

| Agent | Description |
|-------|-------------|
| `metadata_assistant` | Metadata operations specialist. Sample ID mapping, schema standardization, dataset validation, and disease annotation enrichment. |

## Services

| Service | Purpose |
|---------|---------|
| SampleMappingService | Map sample IDs between datasets using multiple strategies |
| MetadataStandardizationService | Standardize metadata fields to Pydantic schemas |
| MetadataFilteringService | Filter datasets by metadata criteria |
| DiseaseStandardizationService | Standardize disease names to controlled vocabularies |
| DiseaseOntologyService | Map diseases to ontology terms (MONDO, DO) |
| ClinicalMetadataService | Extract and validate clinical metadata fields |
| SampleGroupingService | Group samples by metadata attributes |
| MicrobiomeFilteringService | Microbiome-specific metadata filtering |

## Features

### Sample ID Mapping
- Exact match between matrix and metadata identifiers
- Fuzzy matching with configurable similarity thresholds
- Pattern-based matching using regular expressions
- Metadata-based correlation for complex mapping scenarios

### Metadata Standardization
- Transcriptomics schema (cell type, tissue, organism, disease)
- Proteomics schema (platform, quantification method, normalization)
- Microbiome schema (16S vs shotgun, taxonomic level, diversity)

### Dataset Validation
- Sample count consistency between matrix and metadata
- Condition coverage verification for experimental design
- Control sample identification and validation
- Biological and technical replicate detection
- Platform consistency checks across samples

### Disease Enrichment
Four-phase hierarchy for missing disease annotations:
1. Column re-scan for disease-related field names
2. LLM-based abstract extraction from publication context
3. LLM-based methods section parsing
4. Manual mapping fallback for known datasets

### Multi-Omics Integration
- Cross-modality sample alignment
- Shared sample identification across data types
- Metadata merging with conflict resolution

## Requirements

- Python 3.12+
- lobster-ai >= 1.0.0

## Testing

To run the test suite for this package:

```bash
# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest tests/ -v

# Run specific test categories
pytest tests/services/metadata/ -v  # Service tests only
pytest tests/agents/ -v             # Agent tests only
```

## Documentation

Full documentation: [docs.omics-os.com/docs/agents/metadata](https://docs.omics-os.com/docs/agents/metadata)

## License

AGPL-3.0-or-later
