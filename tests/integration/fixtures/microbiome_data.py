"""
Fixtures for microbiome harmonization workflow integration tests.

Provides sample RIS files, SRA metadata, and mock DataManagerV2 instances
for testing the complete microbiome harmonization workflow.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from lobster.core.schemas.publication_queue import (
    ExtractionLevel,
    PublicationQueueEntry,
    PublicationStatus,
)


@pytest.fixture
def sample_ris_file() -> Path:
    """Return path to sample RIS file with 3 microbiome publications."""
    fixtures_dir = Path(__file__).parent
    return fixtures_dir / "sample_microbiome_studies.ris"


@pytest.fixture
def sample_sra_metadata() -> List[Dict[str, Any]]:
    """
    Sample SRA metadata with 24 runs, mixed sample types for filtering tests.

    Mix includes:
    - 12 human gut samples (should pass filtering)
    - 4 mouse gut samples (should be filtered out)
    - 4 human stool samples (should pass filtering)
    - 4 environmental samples (should be filtered out)
    """
    metadata = []

    # Human gut samples (12) - should pass
    for i in range(12):
        metadata.append({
            "run_accession": f"SRR10000{i:03d}",
            "sample_accession": f"SRS20000{i:03d}",
            "experiment_accession": f"SRX30000{i:03d}",
            "study_accession": "SRP123456",
            "bioproject_accession": "PRJNA789012",
            "organism": "Homo sapiens",
            "sample_type": "gut microbiome",
            "tissue": "gut",
            "host": "Homo sapiens",
            "library_strategy": "AMPLICON",
            "library_source": "METAGENOMIC",
            "platform": "ILLUMINA",
            "instrument_model": "Illumina MiSeq",
            "reads": 50000 + i * 1000,
            "bases": 7500000 + i * 150000,
            "avg_length": 150,
            "disease": "IBD" if i < 6 else "healthy control",
            "age": 35 + i,
            "sex": "male" if i % 2 == 0 else "female",
        })

    # Mouse gut samples (4) - should be filtered out
    for i in range(4):
        metadata.append({
            "run_accession": f"SRR10001{i:02d}",
            "sample_accession": f"SRS20001{i:02d}",
            "experiment_accession": f"SRX30001{i:02d}",
            "study_accession": "SRP123456",
            "bioproject_accession": "PRJNA789012",
            "organism": "Mus musculus",
            "sample_type": "gut microbiome",
            "tissue": "intestine",
            "host": "Mus musculus",
            "library_strategy": "AMPLICON",
            "library_source": "METAGENOMIC",
            "platform": "ILLUMINA",
            "instrument_model": "Illumina MiSeq",
            "reads": 45000 + i * 1000,
            "bases": 6750000 + i * 150000,
            "avg_length": 150,
            "disease": "control",
            "age": 12 + i,
            "sex": "male" if i % 2 == 0 else "female",
        })

    # Human stool samples (4) - should pass
    for i in range(4):
        metadata.append({
            "run_accession": f"SRR10002{i:02d}",
            "sample_accession": f"SRS20002{i:02d}",
            "experiment_accession": f"SRX30002{i:02d}",
            "study_accession": "SRP123456",
            "bioproject_accession": "PRJNA789012",
            "organism": "human gut metagenome",
            "sample_type": "stool",
            "tissue": "stool",
            "host": "Homo sapiens",
            "library_strategy": "AMPLICON",
            "library_source": "METAGENOMIC",
            "platform": "ILLUMINA",
            "instrument_model": "Illumina MiSeq",
            "reads": 55000 + i * 1000,
            "bases": 8250000 + i * 150000,
            "avg_length": 150,
            "disease": "Crohn's disease" if i < 2 else "healthy",
            "age": 40 + i,
            "sex": "female",
        })

    # Environmental samples (4) - should be filtered out
    for i in range(4):
        metadata.append({
            "run_accession": f"SRR10003{i:02d}",
            "sample_accession": f"SRS20003{i:02d}",
            "experiment_accession": f"SRX30003{i:02d}",
            "study_accession": "SRP123456",
            "bioproject_accession": "PRJNA789012",
            "organism": "soil metagenome",
            "sample_type": "environmental",
            "tissue": "soil",
            "host": None,
            "library_strategy": "WGS",
            "library_source": "METAGENOMIC",
            "platform": "ILLUMINA",
            "instrument_model": "Illumina NovaSeq",
            "reads": 100000 + i * 10000,
            "bases": 30000000 + i * 3000000,
            "avg_length": 300,
            "disease": None,
            "age": None,
            "sex": None,
        })

    return metadata


@pytest.fixture
def expected_filtered_metadata() -> Dict[str, Any]:
    """
    Expected output after filtering sample_sra_metadata.

    Should contain:
    - 16 samples (12 human gut + 4 human stool)
    - Standardized disease terms
    - No mouse or environmental samples
    """
    return {
        "total_samples": 16,
        "filtered_samples": 8,  # 4 mouse + 4 environmental
        "passed_samples": 16,
        "filter_criteria": {
            "organism": "Homo sapiens OR human gut metagenome",
            "sample_type": "gut microbiome OR stool",
        },
        "disease_standardization": {
            "IBD": "inflammatory bowel disease",
            "Crohn's disease": "crohn disease",
            "healthy control": "healthy",
            "healthy": "healthy",
        },
    }


@pytest.fixture
def mock_data_manager(tmp_path):
    """
    Create mock DataManagerV2 for testing without real workspace dependencies.

    Returns:
        MagicMock: Mock DataManagerV2 with necessary methods
    """
    mock = MagicMock()

    # Set workspace path
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    mock.workspace_path = workspace

    # Create metadata directory
    metadata_dir = workspace / "metadata"
    metadata_dir.mkdir()

    # Mock publication queue
    mock.publication_queue = MagicMock()
    mock.publication_queue.get_entry = MagicMock()
    mock.publication_queue.update_status = MagicMock()
    mock.publication_queue.add_entry = MagicMock()

    # Mock provenance tracker
    mock.log_tool_usage = MagicMock()

    return mock


@pytest.fixture
def sample_publication_entry() -> PublicationQueueEntry:
    """
    Sample publication queue entry for microbiome study.

    Returns:
        PublicationQueueEntry: Entry with SRA identifiers
    """
    return PublicationQueueEntry(
        entry_id="pub_microbiome_001",
        pmid="12345678",
        doi="10.1038/nmicrobiol.2023.001",
        title="Gut microbiome analysis in IBD patients using 16S rRNA sequencing",
        authors=["Smith J", "Johnson M", "Chen W"],
        year=2023,
        journal="Nature Microbiology",
        priority=5,
        status=PublicationStatus.PENDING,
        extraction_level=ExtractionLevel.METHODS,
        schema_type="microbiome",
        extracted_identifiers={
            "sra": ["SRP123456"],
            "bioproject": ["PRJNA789012"],
        },
        workspace_metadata_keys=[],  # Will be populated by research_agent
        harmonization_metadata=None,  # Will be populated by metadata_assistant
    )


@pytest.fixture
def sample_harmonization_metadata() -> Dict[str, Any]:
    """
    Sample harmonization metadata from metadata_assistant.

    Returns:
        Dict: Harmonized metadata ready for export
    """
    return {
        "samples": [
            {
                "run_accession": "SRR1000000",
                "sample_accession": "SRS2000000",
                "organism": "Homo sapiens",
                "tissue": "gut",
                "disease_standardized": "inflammatory bowel disease",
                "age": 35,
                "sex": "male",
            },
            {
                "run_accession": "SRR1000001",
                "sample_accession": "SRS2000001",
                "organism": "Homo sapiens",
                "tissue": "gut",
                "disease_standardized": "inflammatory bowel disease",
                "age": 36,
                "sex": "female",
            },
            {
                "run_accession": "SRR1000006",
                "sample_accession": "SRS2000006",
                "organism": "Homo sapiens",
                "tissue": "gut",
                "disease_standardized": "healthy",
                "age": 41,
                "sex": "male",
            },
        ],
        "validation_status": "passed",
        "total_samples": 3,
        "filtered_count": 21,
        "filter_summary": {
            "organism_filtered": 8,
            "sample_type_filtered": 4,
        },
    }
