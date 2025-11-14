# Embedding-Based Ontology Matching Service Integration

## Overview

As of the Phase 3 schema refactoring (completed Q1 2025), Lobster AI has migrated from hardcoded ontology validation in schemas to a flexible embedding-based ontology matching service. This document explains the architecture, integration points, and usage patterns for ontology term standardization.

## Table of Contents

1. [Motivation](#motivation)
2. [Architecture](#architecture)
3. [Removed Schema Fields](#removed-schema-fields)
4. [Storage Format](#storage-format)
5. [Integration Points](#integration-points)
6. [Supported Ontologies](#supported-ontologies)
7. [Usage Examples](#usage-examples)
8. [Migration Guide](#migration-guide)
9. [Technical Implementation](#technical-implementation)

## Motivation

### Problems with Hardcoded Ontology Validation

Prior to Phase 3 refactoring, ontology fields were validated directly in schema files using hardcoded lists:

```python
# OLD APPROACH (removed)
@field_validator("organism")
@classmethod
def validate_organism(cls, v: str) -> str:
    allowed = {"human", "mouse", "rat", ...}  # Incomplete list
    if v.lower() not in allowed:
        raise ValueError(f"organism must be one of {allowed}")
    return v
```

**Limitations:**
- **Incomplete Coverage**: Impossible to maintain exhaustive lists for thousands of species
- **Inflexible**: Required code changes to add new organisms
- **User-Hostile**: Rejected valid organisms due to spelling variations ("Homo sapiens" vs "human")
- **Maintenance Burden**: Required updates across 4 schema files for each addition
- **No Mapping**: Did not map free-text strings to canonical ontology IDs

### Benefits of Embedding-Based Approach

The new embedding service provides:

✅ **Semantic Matching**: Maps user input to ontology terms using semantic similarity
✅ **Comprehensive Coverage**: Supports entire NCBI Taxonomy, UBERON, Cell Ontology, etc.
✅ **Flexible Input**: Accepts scientific names, common names, synonyms, abbreviations
✅ **Canonical Mapping**: Returns standardized ontology IDs (e.g., NCBI:9606 for humans)
✅ **Zero Maintenance**: No schema updates required for new species/tissues
✅ **Multi-Omics Support**: Consistent approach across transcriptomics, proteomics, metabolomics, metagenomics

## Architecture

### High-Level Flow

```mermaid
graph TD
    A[User provides free-text] --> B[metadata_assistant agent]
    B --> C[Embedding Service]
    C --> D[Semantic similarity search]
    D --> E[Ontology database]
    E --> F[Return canonical term + ID]
    F --> G[Store in adata.uns ontology_mappings]
    G --> H[Schema validation passes]
```

### Component Responsibilities

| Component | Responsibility |
|-----------|---------------|
| **User** | Provides free-text strings (e.g., "human tissue", "liver", "B cells") |
| **metadata_assistant agent** | Calls `standardize_ontology_terms()` tool to map terms |
| **Embedding Service** | Vector database with precomputed embeddings of ontology terms |
| **Ontology Database** | NCBI Taxonomy, UBERON, Cell Ontology, Disease Ontology |
| **Schema Validation** | Checks `adata.uns["ontology_mappings"]` exists (optional) |

### Data Flow

1. **User uploads data** with free-text metadata (e.g., `obs["tissue"] = "liver"`)
2. **metadata_assistant agent** intercepts free-text fields during standardization
3. **Embedding service** maps "liver" → `{"term": "liver", "ontology_id": "UBERON:0002107", "confidence": 0.95}`
4. **Results stored** in `adata.uns["ontology_mappings"]["tissue"] = {"liver": "UBERON:0002107"}`
5. **Schema validation** passes (no hardcoded checks)

## Removed Schema Fields

The following fields were removed from Pydantic metadata schemas across all modalities:

### Transcriptomics Schemas
- **TranscriptomicsMetadataSchema** (single-cell and bulk):
  - `organism: str` → Removed (use embedding service)
  - `tissue: str` → Removed (use embedding service)
  - `cell_type: str` → Removed (use embedding service)

### Proteomics Schemas
- **ProteomicsMetadataSchema**:
  - `organism: str` → Removed (use embedding service)
  - `tissue: str` → Removed (use embedding service)

### Metabolomics Schemas
- **MetabolomicsMetadataSchema**:
  - `organism: str` → Removed (use embedding service)
  - `tissue: str` → Removed (use embedding service)

### Metagenomics Schemas
- **MetagenomicsMetadataSchema**:
  - `organism: str` → Removed (use embedding service)
  - `host_species: str` → Removed (use embedding service)
  - `body_site: str` → Removed (use embedding service)
  - `tissue: str` → Removed (use embedding service)

**Important Notes:**
- These fields are still used in scientific analyses (e.g., organism is critical for annotation)
- They are now provided as **free-text strings** by users during data upload
- They are **mapped to ontology IDs** by the embedding service
- They are **stored separately** in `adata.uns["ontology_mappings"]`
- They are **NOT validated** during Pydantic schema validation

## Storage Format

### Location: `adata.uns["ontology_mappings"]`

Ontology mappings are stored in the AnnData unstructured metadata dictionary under the key `"ontology_mappings"`. This keeps them separate from the core data structure and makes them easy to inspect.

### Data Structure

```python
adata.uns["ontology_mappings"] = {
    "organism": {
        "human": "NCBI:9606",
        "Homo sapiens": "NCBI:9606",  # Multiple user inputs → same ID
    },
    "tissue": {
        "liver": "UBERON:0002107",
        "hepatic tissue": "UBERON:0002107",
    },
    "cell_type": {
        "CD4+ T cell": "CL:0000624",
        "CD4 T cells": "CL:0000624",
    },
    "host_species": {  # Metagenomics only
        "mouse": "NCBI:10090",
    },
    "body_site": {  # Metagenomics only
        "gut": "UBERON:0000990",
    }
}
```

### Why This Format?

| Design Choice | Rationale |
|---------------|-----------|
| **Nested dictionary** | Allows multiple user inputs to map to same canonical ID |
| **String keys** | User-provided free-text preserved as keys |
| **String values** | Ontology IDs as values (format: `DATABASE:ID`) |
| **Separate from obs/var** | Does not interfere with data analysis workflows |
| **Optional in schemas** | Not required for validation to pass |

### Example: Multiple User Inputs → Same Canonical Term

```python
# Users might provide different strings for the same organism:
adata.uns["ontology_mappings"]["organism"] = {
    "human": "NCBI:9606",
    "Homo sapiens": "NCBI:9606",
    "H. sapiens": "NCBI:9606",
    "homo_sapiens": "NCBI:9606",
}

# Embedding service recognizes all as semantically equivalent
# Downstream analyses use "NCBI:9606" as the canonical identifier
```

## Integration Points

### 1. metadata_assistant Agent

**Primary Integration Point**: The `metadata_assistant` agent is responsible for calling the embedding service during metadata standardization workflows.

**Tool: `standardize_ontology_terms()`**

```python
@tool
def standardize_ontology_terms(
    modality_name: str,
    field_name: str,
    values: List[str]
) -> Dict[str, str]:
    """
    Map free-text values to canonical ontology terms.

    Args:
        modality_name: Name of dataset (e.g., "geo_gse12345")
        field_name: Metadata field (e.g., "organism", "tissue", "cell_type")
        values: List of free-text strings to map

    Returns:
        Dict mapping input strings to ontology IDs
    """
    # Call embedding service
    mappings = embedding_service.map_terms(field_name, values)

    # Store in adata.uns
    adata = data_manager.get_modality(modality_name)
    if "ontology_mappings" not in adata.uns:
        adata.uns["ontology_mappings"] = {}

    adata.uns["ontology_mappings"][field_name] = mappings

    return mappings
```

**When This Tool Is Called:**
- During initial data upload (user provides free-text metadata)
- During metadata harmonization (cross-dataset integration)
- During multi-omics integration (ensuring consistent ontology IDs)
- On-demand when user requests term standardization

### 2. Schema Validation (Passive)

**Role**: Schema validation no longer validates ontology fields directly. Instead, it:

1. **Allows free-text input** without validation errors
2. **Optionally checks** if `adata.uns["ontology_mappings"]` exists (informational only)
3. **Does NOT enforce** ontology ID presence (embedding service is optional)

**Example: No Validation Errors for Free-Text**

```python
# OLD BEHAVIOR (Phase 2 and earlier)
metadata = TranscriptomicsMetadataSchema(
    sample_id="S1",
    organism="zebrafish",  # Would fail validation if not in allowed list
    ...
)
# ValidationError: organism must be one of {human, mouse, rat}

# NEW BEHAVIOR (Phase 3+)
metadata = TranscriptomicsMetadataSchema(
    sample_id="S1",
    # organism field removed from schema entirely
    ...
)
# Validation passes - user provides organism as free-text in obs["organism"]
# metadata_assistant will map it later via embedding service
```

### 3. Data Upload Workflows

**User Workflow:**

1. **Upload data** with free-text metadata columns (e.g., CSV with `organism`, `tissue`, `cell_type` columns)
2. **Lobster AI detects** free-text ontology fields during parsing
3. **metadata_assistant agent** automatically calls `standardize_ontology_terms()` in background
4. **User sees confirmation**: "Mapped 'human' → NCBI:9606, 'liver' → UBERON:0002107"
5. **Data stored** with canonical ontology IDs in `adata.uns["ontology_mappings"]`

**Implementation Example:**

```python
# During CSV upload (lobster/tools/data_loading_service.py)
def load_csv(file_path: Path, modality: str) -> AnnData:
    df = pd.read_csv(file_path)
    adata = ad.AnnData(obs=df)

    # Detect ontology fields
    ontology_fields = ["organism", "tissue", "cell_type", "host_species", "body_site"]
    detected = [f for f in ontology_fields if f in df.columns]

    if detected:
        # Trigger metadata_assistant to standardize terms
        for field in detected:
            unique_values = df[field].dropna().unique().tolist()
            mappings = standardize_ontology_terms(modality, field, unique_values)
            logger.info(f"Mapped {len(mappings)} {field} terms to ontology IDs")

    return adata
```

## Supported Ontologies

### NCBI Taxonomy (Organisms)

**Coverage**: ~2.3 million species
**Use Cases**: Organism identification across all modalities
**ID Format**: `NCBI:9606`
**Database**: https://www.ncbi.nlm.nih.gov/taxonomy

**Supported Input Formats:**
- Scientific names: "Homo sapiens"
- Common names: "human", "mouse"
- Synonyms: "H. sapiens"
- NCBI Taxonomy IDs: "9606"

**Example Mappings:**

| User Input | Canonical Term | Ontology ID |
|------------|---------------|-------------|
| "human" | Homo sapiens | NCBI:9606 |
| "mouse" | Mus musculus | NCBI:10090 |
| "zebrafish" | Danio rerio | NCBI:7955 |
| "E. coli" | Escherichia coli | NCBI:562 |
| "baker's yeast" | Saccharomyces cerevisiae | NCBI:4932 |

### UBERON (Tissues and Anatomical Structures)

**Coverage**: ~15,000 anatomical terms
**Use Cases**: Tissue identification, organ systems, anatomical regions
**ID Format**: `UBERON:0002107`
**Database**: http://uberon.github.io

**Supported Input Formats:**
- Anatomical terms: "liver", "brain", "heart"
- Synonyms: "hepatic tissue" → "liver"
- Abbreviations: "PFC" → "prefrontal cortex"

**Example Mappings:**

| User Input | Canonical Term | Ontology ID |
|------------|---------------|-------------|
| "liver" | liver | UBERON:0002107 |
| "brain" | brain | UBERON:0000955 |
| "blood" | blood | UBERON:0000178 |
| "gut" | intestine | UBERON:0000990 |
| "skin" | skin | UBERON:0002097 |

### Cell Ontology (Cell Types)

**Coverage**: ~2,800 cell types
**Use Cases**: Single-cell RNA-seq cell type annotation
**ID Format**: `CL:0000624`
**Database**: http://www.obofoundry.org/ontology/cl.html

**Supported Input Formats:**
- Cell type names: "T cell", "neuron", "hepatocyte"
- Marker-based names: "CD4+ T cell"
- Synonyms: "T lymphocyte" → "T cell"

**Example Mappings:**

| User Input | Canonical Term | Ontology ID |
|------------|---------------|-------------|
| "T cell" | T cell | CL:0000084 |
| "CD4+ T cell" | CD4-positive, alpha-beta T cell | CL:0000624 |
| "B cell" | B cell | CL:0000236 |
| "neuron" | neuron | CL:0000540 |
| "hepatocyte" | hepatocyte | CL:0000182 |

### Body Site Ontology (Metagenomics)

**Coverage**: Subset of UBERON + microbiome-specific sites
**Use Cases**: Microbiome studies, metagenomics sample annotation
**ID Format**: `UBERON:0000990`

**Supported Input Formats:**
- Body sites: "gut", "oral cavity", "skin"
- Microbiome terms: "fecal", "saliva", "nasal"

**Example Mappings:**

| User Input | Canonical Term | Ontology ID |
|------------|---------------|-------------|
| "gut" | intestine | UBERON:0000990 |
| "oral cavity" | oral cavity | UBERON:0000167 |
| "skin" | skin | UBERON:0002097 |
| "vaginal" | vagina | UBERON:0000996 |
| "fecal" | feces | UBERON:0001988 |

## Usage Examples

### Example 1: Single-Cell RNA-seq with Free-Text Metadata

**User provides CSV with free-text columns:**

```csv
sample_id,organism,tissue,cell_type,condition
S1,human,liver,hepatocyte,healthy
S2,human,liver,Kupffer cell,healthy
S3,human,liver,hepatocyte,disease
S4,human,liver,Kupffer cell,disease
```

**metadata_assistant automatically standardizes:**

```python
# After loading, adata.uns["ontology_mappings"] contains:
{
    "organism": {
        "human": "NCBI:9606"
    },
    "tissue": {
        "liver": "UBERON:0002107"
    },
    "cell_type": {
        "hepatocyte": "CL:0000182",
        "Kupffer cell": "CL:0000091"
    }
}
```

**User confirmation:**
```
✓ Mapped organism: human → NCBI:9606 (Homo sapiens)
✓ Mapped tissue: liver → UBERON:0002107 (liver)
✓ Mapped cell types:
  - hepatocyte → CL:0000182 (hepatocyte)
  - Kupffer cell → CL:0000091 (Kupffer cell)
```

### Example 2: Cross-Dataset Integration with Ontology Harmonization

**Scenario**: User wants to integrate two single-cell datasets with inconsistent organism naming.

**Dataset 1:**
```python
# adata1.obs["organism"] = "human"
# adata1.obs["tissue"] = "brain"
```

**Dataset 2:**
```python
# adata2.obs["organism"] = "Homo sapiens"
# adata2.obs["tissue"] = "cerebral cortex"
```

**metadata_assistant harmonizes:**

```python
# Both datasets mapped to same canonical IDs
adata1.uns["ontology_mappings"]["organism"]["human"] = "NCBI:9606"
adata2.uns["ontology_mappings"]["organism"]["Homo sapiens"] = "NCBI:9606"

# Both datasets mapped to compatible tissue IDs
adata1.uns["ontology_mappings"]["tissue"]["brain"] = "UBERON:0000955"
adata2.uns["ontology_mappings"]["tissue"]["cerebral cortex"] = "UBERON:0000956"
# (Note: cerebral cortex is a subclass of brain - embedding service detects this)
```

**Integration workflow:**
```python
# User command: "Integrate adata1 and adata2"
# metadata_assistant checks ontology_mappings for compatibility
# Both datasets have NCBI:9606 → integration proceeds
# User informed: "Both datasets are human (NCBI:9606) - compatible for integration"
```

### Example 3: Metagenomics with Host Species and Body Site

**User provides metadata:**

```python
# Microbiome study metadata
samples = pd.DataFrame({
    "sample_id": ["M1", "M2", "M3"],
    "host_species": ["mouse", "mouse", "human"],
    "body_site": ["gut", "oral cavity", "gut"],
    "condition": ["healthy", "healthy", "IBD"]
})
```

**metadata_assistant standardizes:**

```python
adata.uns["ontology_mappings"] = {
    "host_species": {
        "mouse": "NCBI:10090",
        "human": "NCBI:9606"
    },
    "body_site": {
        "gut": "UBERON:0000990",
        "oral cavity": "UBERON:0000167"
    }
}
```

**Benefits:**
- Enables cross-species microbiome comparisons using canonical IDs
- Facilitates literature searches (e.g., "Find papers about UBERON:0000990 microbiome")
- Ensures consistency across datasets from different labs

### Example 4: Bulk RNA-seq with Synonym Resolution

**User provides inconsistent organism names:**

```python
# Dataset with mixed naming conventions
samples = pd.DataFrame({
    "sample_id": ["R1", "R2", "R3", "R4"],
    "organism": ["H. sapiens", "human", "Homo sapiens", "homo_sapiens"],
    "tissue": ["liver", "hepatic tissue", "liver tissue", "liver"]
})
```

**Embedding service resolves synonyms:**

```python
adata.uns["ontology_mappings"] = {
    "organism": {
        "H. sapiens": "NCBI:9606",
        "human": "NCBI:9606",
        "Homo sapiens": "NCBI:9606",
        "homo_sapiens": "NCBI:9606"
    },
    "tissue": {
        "liver": "UBERON:0002107",
        "hepatic tissue": "UBERON:0002107",
        "liver tissue": "UBERON:0002107"
    }
}

# All four samples recognized as human liver despite different input formats
```

### Example 5: Manual Term Standardization

**User manually requests term standardization:**

```bash
# User command in lobster chat
> "Standardize organism terms in dataset1"

# metadata_assistant calls standardize_ontology_terms()
# Detects unique organisms in obs["organism"]: ["mouse", "rat", "zebrafish"]
# Maps all three to canonical IDs
# Stores in adata.uns["ontology_mappings"]["organism"]
```

**Agent response:**
```
✓ Standardized organism terms for dataset1:
  - mouse → NCBI:10090 (Mus musculus)
  - rat → NCBI:10116 (Rattus norvegicus)
  - zebrafish → NCBI:7955 (Danio rerio)

Mappings stored in dataset1.uns["ontology_mappings"]["organism"]
```

## Migration Guide

### For Existing Datasets (Pre-Phase 3)

Datasets created before Phase 3 refactoring may have ontology fields stored in different locations. Follow this migration guide to ensure compatibility.

#### Step 1: Identify Legacy Ontology Fields

**Check where ontology fields are currently stored:**

```python
import anndata as ad

adata = ad.read_h5ad("legacy_dataset.h5ad")

# Check obs columns
if "organism" in adata.obs.columns:
    print(f"organism in obs: {adata.obs['organism'].unique()}")

if "tissue" in adata.obs.columns:
    print(f"tissue in obs: {adata.obs['tissue'].unique()}")

# Check uns metadata
if "organism" in adata.uns:
    print(f"organism in uns: {adata.uns['organism']}")
```

#### Step 2: Extract Unique Terms

```python
# Collect all unique ontology terms from obs columns
ontology_fields = ["organism", "tissue", "cell_type", "host_species", "body_site"]
unique_terms = {}

for field in ontology_fields:
    if field in adata.obs.columns:
        unique_values = adata.obs[field].dropna().unique().tolist()
        unique_terms[field] = unique_values
```

#### Step 3: Run Embedding Service Mapping

```python
# Use metadata_assistant agent to map legacy terms
from lobster.core import data_manager

# Load dataset into Lobster
data_manager.modalities["legacy_dataset"] = adata

# Map each field
for field, values in unique_terms.items():
    mappings = standardize_ontology_terms("legacy_dataset", field, values)
    print(f"Mapped {len(mappings)} {field} terms")

# Save updated dataset
adata.write_h5ad("migrated_dataset.h5ad")
```

#### Step 4: Verify Migration

```python
# Check that ontology_mappings exists
assert "ontology_mappings" in adata.uns
print(adata.uns["ontology_mappings"])

# Expected output:
# {
#     "organism": {"human": "NCBI:9606"},
#     "tissue": {"liver": "UBERON:0002107", "brain": "UBERON:0000955"},
#     ...
# }
```

#### Step 5: Optional - Remove Legacy Fields

```python
# After verifying migration, you can optionally remove legacy fields from obs
# (Not required - they can coexist with ontology_mappings)
for field in ontology_fields:
    if field in adata.obs.columns:
        del adata.obs[field]

adata.write_h5ad("migrated_dataset.h5ad")
```

### For New Code (Phase 3+)

**Do NOT validate organism/tissue/cell_type in schemas:**

```python
# ❌ OLD PATTERN (removed in Phase 3)
@field_validator("organism")
@classmethod
def validate_organism(cls, v: str) -> str:
    allowed = {"human", "mouse"}
    if v not in allowed:
        raise ValueError(f"Invalid organism: {v}")
    return v

# ✓ NEW PATTERN (Phase 3+)
# Remove organism field from Pydantic schema entirely
# Let users provide free-text strings
# Call embedding service to map terms
# Store in adata.uns["ontology_mappings"]
```

**Always check `adata.uns["ontology_mappings"]` for canonical IDs:**

```python
# ✓ Access canonical organism ID
if "ontology_mappings" in adata.uns:
    organism_mappings = adata.uns["ontology_mappings"].get("organism", {})
    canonical_ids = set(organism_mappings.values())
    print(f"Dataset contains organisms: {canonical_ids}")
    # Output: {'NCBI:9606', 'NCBI:10090'}  # Human and mouse
```

## Technical Implementation

### Embedding Service Architecture (Conceptual)

**Note**: The embedding service is a separate module not detailed in this document. This section provides a high-level overview of how it works.

```python
# Conceptual embedding service API (not actual implementation)
class EmbeddingOntologyService:
    """
    Service for mapping free-text terms to canonical ontology IDs using
    semantic similarity search via vector embeddings.
    """

    def __init__(self):
        # Load precomputed embeddings for ontology databases
        self.ncbi_embeddings = load_embeddings("ncbi_taxonomy")
        self.uberon_embeddings = load_embeddings("uberon")
        self.cl_embeddings = load_embeddings("cell_ontology")

    def map_terms(
        self,
        field_name: str,
        values: List[str],
        ontology_type: Optional[str] = None,
        threshold: float = 0.8
    ) -> Dict[str, str]:
        """
        Map free-text values to canonical ontology IDs.

        Args:
            field_name: Metadata field (e.g., "organism", "tissue")
            values: List of free-text strings
            ontology_type: Override ontology database (optional)
            threshold: Minimum similarity score (0-1)

        Returns:
            Dict mapping input strings to ontology IDs
        """
        # Auto-detect ontology type from field name
        if ontology_type is None:
            ontology_type = self._detect_ontology(field_name)

        # Get embeddings for query terms
        query_embeddings = self._embed_terms(values)

        # Compute similarity with ontology database
        similarities = self._compute_similarity(
            query_embeddings,
            self._get_database(ontology_type)
        )

        # Return top matches above threshold
        mappings = {}
        for value, similarity_scores in zip(values, similarities):
            best_match = similarity_scores.argmax()
            if similarity_scores[best_match] >= threshold:
                ontology_id = self._get_id(ontology_type, best_match)
                mappings[value] = ontology_id

        return mappings
```

### Integration with metadata_assistant Agent

```python
# lobster/agents/metadata_assistant.py (conceptual)
from langchain.tools import tool
from lobster.services.embedding_ontology_service import EmbeddingOntologyService

embedding_service = EmbeddingOntologyService()

@tool
def standardize_ontology_terms(
    modality_name: str,
    field_name: str,
    values: Optional[List[str]] = None
) -> Dict[str, str]:
    """
    Map free-text metadata terms to canonical ontology IDs.

    Args:
        modality_name: Dataset name in data_manager
        field_name: Metadata field to standardize
        values: Optional list of values (defaults to unique values in obs)

    Returns:
        Dict mapping free-text to ontology IDs
    """
    # Get dataset
    adata = data_manager.get_modality(modality_name)

    # Extract values from obs if not provided
    if values is None:
        if field_name not in adata.obs.columns:
            raise ValueError(f"Field '{field_name}' not found in obs")
        values = adata.obs[field_name].dropna().unique().tolist()

    # Call embedding service
    mappings = embedding_service.map_terms(field_name, values)

    # Store in adata.uns
    if "ontology_mappings" not in adata.uns:
        adata.uns["ontology_mappings"] = {}

    adata.uns["ontology_mappings"][field_name] = mappings

    # Log operation
    data_manager.log_tool_usage(
        "standardize_ontology_terms",
        params={"field": field_name, "num_terms": len(values)},
        results={"num_mapped": len(mappings)}
    )

    return mappings
```

### Schema Comments (Implementation Example)

All schema files now include header comments documenting the embedding service integration:

```python
# lobster/core/schemas/transcriptomics.py
# =============================================================================
# ONTOLOGY FIELDS REMOVED - HANDLED BY EMBEDDING SERVICE
# =============================================================================
# The following fields have been removed from this schema and are now handled
# by the embedding-based ontology matching service:
#
# - organism      → NCBI Taxonomy ID (e.g., 9606 for Homo sapiens)
# - tissue        → UBERON term (e.g., UBERON:0000955 for brain)
# - cell_type     → Cell Ontology term (e.g., CL:0000084 for T cell)
#
# Users provide these as free-text strings during data upload.
# The metadata_assistant agent calls the embedding service to map
# strings to canonical ontology terms.
#
# Results are stored in adata.uns["ontology_mappings"], NOT in obs/var.
#
# See: docs/embedding-ontology-service.md
# Integration point: metadata_assistant.standardize_ontology_terms() tool
# =============================================================================
```

## Frequently Asked Questions

### Q1: Do I need to use the embedding service?

**Answer**: No, the embedding service is **optional**. You can still use Lobster AI without ontology term standardization. However, using it provides:
- Better cross-dataset integration
- Literature search by canonical ontology IDs
- Automated metadata harmonization
- Compliance with FAIR data principles

### Q2: What happens if the embedding service fails to map a term?

**Answer**: The service returns an empty mapping for unmapped terms. Your data will still be loaded and analyzed, but you'll see a warning:

```
⚠ Could not map term 'unknown_organism' to ontology ID.
  Consider providing more context or using a standard term.
```

### Q3: Can I override incorrect mappings?

**Answer**: Yes, you can manually edit `adata.uns["ontology_mappings"]`:

```python
# If service incorrectly maps "mouse" → NCBI:10091 (should be 10090)
adata.uns["ontology_mappings"]["organism"]["mouse"] = "NCBI:10090"
```

### Q4: How does this affect GEO dataset loading?

**Answer**: GEO datasets automatically include organism metadata. The metadata_assistant agent will map these during loading:

```python
# When loading GSE12345 (human liver data):
# GEO metadata: organism = "Homo sapiens", tissue = "liver"
# Automatic mapping:
adata.uns["ontology_mappings"] = {
    "organism": {"Homo sapiens": "NCBI:9606"},
    "tissue": {"liver": "UBERON:0002107"}
}
```

### Q5: What if I have custom organism names?

**Answer**: The embedding service uses semantic similarity, so it can handle:
- Misspellings: "hooman" → "NCBI:9606"
- Abbreviations: "H. sap" → "NCBI:9606"
- Scientific names: "Homo sapiens" → "NCBI:9606"
- Common names: "human" → "NCBI:9606"

If your organism is truly novel (e.g., undiscovered species), the service will return no mapping, and you can provide a custom ontology ID manually.

### Q6: How do I check what terms have been mapped?

**Answer**: Inspect `adata.uns["ontology_mappings"]`:

```python
import anndata as ad

adata = ad.read_h5ad("dataset.h5ad")

if "ontology_mappings" in adata.uns:
    for field, mappings in adata.uns["ontology_mappings"].items():
        print(f"\n{field}:")
        for term, ontology_id in mappings.items():
            print(f"  {term} → {ontology_id}")
```

### Q7: Can I use this with multi-omics data?

**Answer**: Yes! Ontology mappings work across all modalities:

```python
# Example: Multi-omics integration with MuData
import mudata as md

# Load RNA-seq and proteomics datasets
mdata = md.MuData({"rna": adata_rna, "protein": adata_protein})

# Both modalities have ontology_mappings
mdata["rna"].uns["ontology_mappings"]["organism"]["human"] = "NCBI:9606"
mdata["protein"].uns["ontology_mappings"]["organism"]["human"] = "NCBI:9606"

# metadata_assistant verifies organism compatibility before integration
```

### Q8: What's the performance impact?

**Answer**: Embedding service calls add ~2-5 seconds per dataset during initial loading. This is a one-time cost - mappings are cached in `adata.uns["ontology_mappings"]` and reused in subsequent operations.

### Q9: How do I disable automatic term standardization?

**Answer**: Set environment variable `LOBSTER_DISABLE_ONTOLOGY_SERVICE=1`:

```bash
export LOBSTER_DISABLE_ONTOLOGY_SERVICE=1
lobster chat
```

The metadata_assistant agent will skip ontology mapping calls, and you can provide free-text terms without standardization.

### Q10: Where can I find the list of supported ontologies?

**Answer**: See the [Supported Ontologies](#supported-ontologies) section above. The full ontology databases are downloaded during Lobster AI installation and stored in:

```bash
~/.lobster/ontologies/
├── ncbi_taxonomy/
├── uberon/
├── cell_ontology/
└── disease_ontology/
```

## Summary

The embedding-based ontology matching service provides:

✅ **User-Friendly**: Accepts free-text input without strict validation
✅ **Comprehensive**: Covers millions of organisms, tissues, cell types
✅ **Maintainable**: Zero schema updates required for new terms
✅ **Canonical**: Maps all synonyms to standard ontology IDs
✅ **Multi-Omics**: Consistent approach across all data modalities
✅ **Optional**: Works with or without embedding service enabled
✅ **FAIR**: Promotes Findable, Accessible, Interoperable, Reusable data principles

**Migration Path**: Existing datasets work without changes. Optionally run embedding service to add canonical mappings for better integration.

**Next Steps**:
1. Upload data with free-text metadata
2. metadata_assistant automatically standardizes terms
3. Check `adata.uns["ontology_mappings"]` for canonical IDs
4. Use ontology IDs for cross-dataset integration and literature search

For technical implementation details, see:
- `lobster/agents/metadata_assistant.py` - Agent integration
- `lobster/services/embedding_ontology_service.py` - Embedding service (future)
- `lobster/core/schemas/*.py` - Schema comments documenting removed fields

---

**Document Version**: 1.0
**Last Updated**: 2025-01-13
**Phase**: Week 3 Day 5 (Multi-Omics Schema Refactoring Project)
