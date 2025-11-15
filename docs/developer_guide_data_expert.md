# Developer Guide: Extending Data Expert & Download Queue

**Last Updated**: 2025-11-14 (Phase 2)
**Target Audience**: Developers extending Lobster with new omics databases or download strategies

---

## Table of Contents

1. [Overview](#overview)
2. [Adding New Omics Databases](#adding-new-omics-databases)
3. [Adding New Download Strategies](#adding-new-download-strategies)
4. [Testing Guidelines](#testing-guidelines)
5. [Best Practices](#best-practices)
6. [Troubleshooting](#troubleshooting)

---

## Overview

The download queue pattern is designed for extensibility. This guide shows how to:
- Add support for new omics databases (SRA, PRIDE, MetabolomicsWorkbench)
- Implement custom download strategies
- Test your extensions
- Follow Lobster's architecture patterns

### Architecture Pattern

```
research_agent → validates & prepares → download_queue
                                            ↓
supervisor → coordinates → data_expert → executes download
```

**Key Principle**: research_agent prepares EVERYTHING once (metadata + URLs), data_expert ONLY executes.

### Queue-Based Workflow

1. **Research Agent**: Validates dataset, extracts metadata, prepares URLs, creates queue entry
2. **Download Queue**: Persists entry with all necessary information
3. **Supervisor**: Coordinates handoff to data_expert when download requested
4. **Data Expert**: Executes download using pre-prepared URLs and metadata

**Benefits**:
- Decouples research from execution
- Enables batch processing
- Provides audit trail
- Supports retry logic
- Prevents redundant metadata fetching

---

## Adding New Omics Databases

### Step 1: Create Provider (Research Agent Side)

Providers fetch metadata and extract download URLs for specific databases.

**Location**: `/Users/tyo/GITHUB/omics-os/lobster/lobster/tools/providers/`

#### Example: SRAProvider for Sequence Read Archive

```python
# File: lobster/tools/providers/sra_provider.py

from typing import Dict, Any, List, Optional
import requests
import xml.etree.ElementTree as ET
from lobster.tools.providers.base_provider import BaseProvider, ProviderError
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


class SRAProvider(BaseProvider):
    """
    Provider for NCBI Sequence Read Archive (SRA).

    Fetches metadata and download URLs for SRA datasets.
    Supports: SRR (runs), SRP (projects), SRX (experiments), SRS (samples)
    """

    def __init__(self, data_manager):
        super().__init__(data_manager)
        self.base_url = "https://www.ncbi.nlm.nih.gov/sra"
        self.ftp_base = "ftp://ftp-trace.ncbi.nlm.nih.gov/sra/sra-instant/reads/ByRun/sra"
        self.eutils_base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    def get_download_urls(self, sra_id: str) -> Dict[str, Any]:
        """
        Extract download URLs for SRA dataset.

        Args:
            sra_id: SRA accession (SRR12345678, SRP123456, etc.)

        Returns:
            {
                "sra_id": str,
                "fastq_urls": List[str],  # FASTQ file URLs
                "sra_urls": List[str],    # SRA format URLs
                "metadata_url": str,      # Metadata XML URL
                "total_size_mb": float,   # Estimated size
                "file_count": int,        # Number of files
            }

        Raises:
            ValueError: Invalid SRA ID format
            ProviderError: Failed to fetch metadata or construct URLs
        """
        # Validate SRA ID format
        if not sra_id.startswith(("SRR", "SRP", "SRX", "SRS")):
            raise ValueError(f"Invalid SRA ID: {sra_id}. Must start with SRR, SRP, SRX, or SRS")

        try:
            # Fetch metadata via NCBI E-utilities
            metadata = self._fetch_sra_metadata(sra_id)

            # Construct FTP URLs based on SRA ID pattern
            # Example: SRR12345678 → .../SRR/SRR123/SRR12345678/
            if sra_id.startswith("SRR"):
                prefix = sra_id[:6]
                ftp_dir = f"{self.ftp_base}/{sra_id[:3]}/{prefix}/{sra_id}/"

                # Extract FASTQ URLs from metadata
                fastq_urls = self._extract_fastq_urls(metadata, ftp_dir)

                # SRA format URL
                sra_urls = [f"{ftp_dir}{sra_id}.sra"]
            else:
                # For projects/experiments, get all run IDs
                run_ids = self._get_run_ids(sra_id)
                fastq_urls = []
                sra_urls = []

                for run_id in run_ids:
                    run_urls = self.get_download_urls(run_id)
                    fastq_urls.extend(run_urls["fastq_urls"])
                    sra_urls.extend(run_urls["sra_urls"])

            return {
                "sra_id": sra_id,
                "fastq_urls": fastq_urls,
                "sra_urls": sra_urls,
                "metadata_url": f"{self.base_url}?term={sra_id}",
                "total_size_mb": metadata.get("total_size_mb", 0),
                "file_count": len(fastq_urls) + len(sra_urls),
            }

        except Exception as e:
            logger.error(f"Error extracting URLs for {sra_id}: {e}")
            raise ProviderError(f"Failed to extract SRA URLs: {str(e)}")

    def fetch_metadata(self, sra_id: str) -> Dict[str, Any]:
        """
        Fetch comprehensive metadata for SRA dataset.

        Returns:
            {
                "sra_id": str,
                "title": str,
                "organism": str,
                "study_abstract": str,
                "platform": str,
                "instrument": str,
                "library_strategy": str,  # RNA-Seq, ATAC-Seq, etc.
                "library_source": str,    # TRANSCRIPTOMIC, GENOMIC, etc.
                "n_runs": int,
                "n_samples": int,
                "total_size_mb": float,
                "publication_date": str,
            }
        """
        try:
            metadata = self._fetch_sra_metadata(sra_id)
            return metadata
        except Exception as e:
            logger.error(f"Error fetching metadata for {sra_id}: {e}")
            raise ProviderError(f"Failed to fetch SRA metadata: {str(e)}")

    def _fetch_sra_metadata(self, sra_id: str) -> Dict:
        """Fetch SRA metadata via NCBI E-utilities API."""
        try:
            # Step 1: Search for SRA accession
            search_url = f"{self.eutils_base}/esearch.fcgi"
            search_params = {
                "db": "sra",
                "term": sra_id,
                "retmode": "json",
            }

            response = requests.get(search_url, params=search_params, timeout=10)
            response.raise_for_status()
            search_data = response.json()

            if not search_data.get("esearchresult", {}).get("idlist"):
                raise ProviderError(f"SRA ID {sra_id} not found")

            uid = search_data["esearchresult"]["idlist"][0]

            # Step 2: Fetch detailed metadata
            fetch_url = f"{self.eutils_base}/efetch.fcgi"
            fetch_params = {
                "db": "sra",
                "id": uid,
                "retmode": "xml",
            }

            response = requests.get(fetch_url, params=fetch_params, timeout=10)
            response.raise_for_status()

            # Parse XML metadata
            root = ET.fromstring(response.content)

            # Extract key fields
            metadata = {
                "sra_id": sra_id,
                "title": self._extract_xml_field(root, ".//STUDY_TITLE"),
                "organism": self._extract_xml_field(root, ".//SCIENTIFIC_NAME"),
                "study_abstract": self._extract_xml_field(root, ".//STUDY_ABSTRACT"),
                "platform": self._extract_xml_field(root, ".//PLATFORM_INSTRUMENT"),
                "instrument": self._extract_xml_field(root, ".//INSTRUMENT_MODEL"),
                "library_strategy": self._extract_xml_field(root, ".//LIBRARY_STRATEGY"),
                "library_source": self._extract_xml_field(root, ".//LIBRARY_SOURCE"),
                "n_runs": len(root.findall(".//RUN")),
                "n_samples": len(root.findall(".//SAMPLE")),
                "total_size_mb": self._extract_size(root),
                "publication_date": self._extract_xml_field(root, ".//PUBLISHED"),
            }

            return metadata

        except requests.RequestException as e:
            raise ProviderError(f"Network error fetching SRA metadata: {str(e)}")
        except ET.ParseError as e:
            raise ProviderError(f"XML parsing error: {str(e)}")

    def _extract_fastq_urls(self, metadata: Dict, ftp_dir: str) -> List[str]:
        """Extract FASTQ file URLs from SRA metadata."""
        # Typical pattern: {sra_id}_1.fastq.gz, {sra_id}_2.fastq.gz
        sra_id = metadata["sra_id"]
        n_runs = metadata.get("n_runs", 1)

        fastq_urls = []

        # Check if paired-end or single-end
        if metadata.get("library_layout") == "PAIRED":
            fastq_urls.append(f"{ftp_dir}{sra_id}_1.fastq.gz")
            fastq_urls.append(f"{ftp_dir}{sra_id}_2.fastq.gz")
        else:
            fastq_urls.append(f"{ftp_dir}{sra_id}.fastq.gz")

        return fastq_urls

    def _get_run_ids(self, project_id: str) -> List[str]:
        """Get all run IDs for a project/experiment."""
        # Implementation: Query NCBI for associated runs
        # Return list of SRR IDs
        pass

    def _extract_xml_field(self, root: ET.Element, xpath: str) -> str:
        """Extract text from XML element."""
        element = root.find(xpath)
        return element.text if element is not None else ""

    def _extract_size(self, root: ET.Element) -> float:
        """Extract total size in MB from XML."""
        size_elements = root.findall(".//size")
        total_bytes = sum(int(elem.get("value", 0)) for elem in size_elements)
        return total_bytes / (1024 * 1024)  # Convert to MB
```

#### Example: PRIDEProvider for Proteomics

```python
# File: lobster/tools/providers/pride_provider.py

from typing import Dict, Any, List
import requests
from lobster.tools.providers.base_provider import BaseProvider, ProviderError
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


class PRIDEProvider(BaseProvider):
    """
    Provider for PRIDE Archive (proteomics data repository).

    Fetches metadata and download URLs for PRIDE datasets.
    """

    def __init__(self, data_manager):
        super().__init__(data_manager)
        self.base_url = "https://www.ebi.ac.uk/pride/ws/archive/v2"
        self.ftp_base = "ftp://ftp.pride.ebi.ac.uk/pride/data/archive"

    def get_download_urls(self, pride_id: str) -> Dict[str, Any]:
        """
        Extract download URLs for PRIDE dataset.

        Args:
            pride_id: PRIDE accession (PXD012345)

        Returns:
            {
                "pride_id": str,
                "raw_urls": List[str],        # .raw files
                "mzml_urls": List[str],       # .mzML files
                "result_urls": List[str],     # Result files
                "total_size_mb": float,
            }
        """
        if not pride_id.startswith("PXD"):
            raise ValueError(f"Invalid PRIDE ID: {pride_id}. Must start with PXD")

        try:
            # Fetch file list from PRIDE API
            files_url = f"{self.base_url}/files/byProject"
            params = {"accession": pride_id}

            response = requests.get(files_url, params=params, timeout=10)
            response.raise_for_status()
            files_data = response.json()

            # Organize files by type
            raw_urls = []
            mzml_urls = []
            result_urls = []
            total_size_mb = 0

            for file_info in files_data:
                file_url = file_info.get("downloadLink")
                file_size_mb = file_info.get("fileSizeBytes", 0) / (1024 * 1024)
                file_name = file_info.get("fileName", "")

                total_size_mb += file_size_mb

                if file_name.endswith(".raw"):
                    raw_urls.append(file_url)
                elif file_name.endswith(".mzML") or file_name.endswith(".mzml"):
                    mzml_urls.append(file_url)
                elif file_name.endswith((".txt", ".tsv", ".csv")):
                    result_urls.append(file_url)

            return {
                "pride_id": pride_id,
                "raw_urls": raw_urls,
                "mzml_urls": mzml_urls,
                "result_urls": result_urls,
                "total_size_mb": total_size_mb,
                "file_count": len(raw_urls) + len(mzml_urls) + len(result_urls),
            }

        except Exception as e:
            logger.error(f"Error extracting URLs for {pride_id}: {e}")
            raise ProviderError(f"Failed to extract PRIDE URLs: {str(e)}")

    def fetch_metadata(self, pride_id: str) -> Dict[str, Any]:
        """
        Fetch comprehensive metadata for PRIDE dataset.

        Returns:
            {
                "pride_id": str,
                "title": str,
                "description": str,
                "organisms": List[str],
                "instruments": List[str],
                "publication": str,
                "n_files": int,
                "total_size_mb": float,
                "submission_date": str,
            }
        """
        try:
            project_url = f"{self.base_url}/projects/{pride_id}"

            response = requests.get(project_url, timeout=10)
            response.raise_for_status()
            project_data = response.json()

            return {
                "pride_id": pride_id,
                "title": project_data.get("title", ""),
                "description": project_data.get("projectDescription", ""),
                "organisms": project_data.get("organisms", []),
                "instruments": project_data.get("instruments", []),
                "publication": project_data.get("publicationDate", ""),
                "n_files": len(project_data.get("dataFiles", [])),
                "total_size_mb": self._calculate_total_size(project_data),
                "submission_date": project_data.get("submissionDate", ""),
            }

        except Exception as e:
            logger.error(f"Error fetching metadata for {pride_id}: {e}")
            raise ProviderError(f"Failed to fetch PRIDE metadata: {str(e)}")

    def _calculate_total_size(self, project_data: Dict) -> float:
        """Calculate total dataset size in MB."""
        total_bytes = sum(
            file_info.get("fileSizeBytes", 0)
            for file_info in project_data.get("dataFiles", [])
        )
        return total_bytes / (1024 * 1024)
```

### Key Methods to Implement

| Method | Required | Purpose |
|--------|----------|---------|
| `get_download_urls(dataset_id: str)` | **YES** | Extract download URLs from metadata |
| `fetch_metadata(dataset_id: str)` | **YES** | Get comprehensive dataset metadata |
| `_validate_id_format(dataset_id: str)` | Recommended | Validate accession format |
| `_fetch_from_api(...)` | Helper | Interact with database API |
| `_construct_ftp_urls(...)` | Helper | Build FTP/HTTP URLs |

### Return Structure Template

Adapt this structure to your database:

```python
# get_download_urls return value
{
    "dataset_id": str,              # Original accession
    "primary_urls": List[str],      # Main data files
    "supplementary_urls": List[str], # Additional files
    "metadata_url": str,            # Web metadata page
    "total_size_mb": float,         # Size estimate
    "file_count": int,              # Number of files
    # Add database-specific fields as needed
}

# fetch_metadata return value
{
    "dataset_id": str,
    "title": str,
    "description": str,
    "organism": str,
    "platform": str,
    "n_samples": int,
    "publication_date": str,
    # Add database-specific fields
}
```

---

### Step 2: Update research_agent (Queue Population)

Add validation tool for your new database.

**Location**: `/Users/tyo/GITHUB/omics-os/lobster/lobster/agents/research_agent.py`

#### Example: Add validate_sra_metadata tool

```python
@tool
def validate_sra_metadata(
    sra_id: str,
    required_fields: str = "",
    add_to_queue: bool = True,
) -> str:
    """
    Validate SRA dataset metadata and add to download queue.

    Args:
        sra_id: SRA accession (e.g., 'SRR12345678', 'SRP123456')
        required_fields: Comma-separated list of required metadata fields
        add_to_queue: If True, add validated dataset to download queue

    Returns:
        Validation report with recommendation and queue confirmation

    Example:
        validate_sra_metadata("SRR12345678", add_to_queue=True)
    """
    try:
        from lobster.tools.providers.sra_provider import SRAProvider
        import uuid
        from datetime import datetime
        from lobster.core.schemas.download_queue import (
            DownloadQueueEntry,
            DownloadStatus,
            StrategyConfig,
        )

        # 1. Fetch SRA metadata
        sra_provider = SRAProvider(data_manager)
        metadata = sra_provider.fetch_metadata(sra_id)

        logger.info(f"Fetched metadata for SRA {sra_id}: {metadata.get('title', 'N/A')}")

        # 2. Validate metadata (adapt to your requirements)
        validation_result = {
            "dataset_id": sra_id,
            "database": "sra",
            "validation_checks": {},
            "missing_fields": [],
            "recommendation": "proceed",
            "confidence": 0.95,
        }

        # Check required fields
        required = [f.strip() for f in required_fields.split(",") if f.strip()]
        for field in required:
            if not metadata.get(field):
                validation_result["missing_fields"].append(field)
                validation_result["validation_checks"][field] = "MISSING"
            else:
                validation_result["validation_checks"][field] = "PASS"

        # Check organism
        if metadata.get("organism"):
            validation_result["validation_checks"]["organism"] = "PASS"
        else:
            validation_result["validation_checks"]["organism"] = "WARNING"

        # Check sample count
        n_samples = metadata.get("n_samples", 0)
        if n_samples > 0:
            validation_result["validation_checks"]["sample_count"] = "PASS"
        else:
            validation_result["validation_checks"]["sample_count"] = "WARNING"

        # Adjust recommendation based on missing fields
        if len(validation_result["missing_fields"]) > 2:
            validation_result["recommendation"] = "review"
            validation_result["confidence"] = 0.6

        # 3. Extract URLs if proceeding
        if validation_result["recommendation"] == "proceed" and add_to_queue:
            url_data = sra_provider.get_download_urls(sra_id)

            logger.info(
                f"Extracted {len(url_data.get('fastq_urls', []))} FASTQ URLs "
                f"and {len(url_data.get('sra_urls', []))} SRA URLs"
            )

            # 4. Create DownloadQueueEntry
            entry = DownloadQueueEntry(
                entry_id=f"queue_{sra_id}_{uuid.uuid4().hex[:8]}",
                dataset_id=sra_id,
                database="sra",  # NEW database type
                priority=5,
                status=DownloadStatus.PENDING,
                metadata=metadata,
                validation_result=validation_result,

                # SRA-specific URL fields
                raw_urls=url_data.get("fastq_urls", []),
                supplementary_urls=url_data.get("sra_urls", []),
                matrix_url=None,  # Not applicable for SRA
                h5_url=None,

                # Download strategy
                recommended_strategy=StrategyConfig(
                    strategy_type="RAW_FIRST",
                    confidence=0.9,
                    rationale="SRA datasets require raw FASTQ processing",
                ),

                created_at=datetime.now(),
                updated_at=datetime.now(),
            )

            # 5. Add to queue
            data_manager.download_queue.add_entry(entry)

            logger.info(f"Added SRA {sra_id} to download queue (entry_id: {entry.entry_id})")

            # Format response
            report = f"""
### SRA Dataset Validation Report

**Dataset ID**: {sra_id}
**Title**: {metadata.get('title', 'N/A')}
**Organism**: {metadata.get('organism', 'N/A')}
**Library Strategy**: {metadata.get('library_strategy', 'N/A')}
**Samples**: {metadata.get('n_samples', 0)}
**Runs**: {metadata.get('n_runs', 0)}

**Validation Checks**:
{self._format_validation_checks(validation_result['validation_checks'])}

**Download Information**:
- FASTQ URLs: {len(url_data.get('fastq_urls', []))}
- SRA URLs: {len(url_data.get('sra_urls', []))}
- Total Size: {url_data.get('total_size_mb', 0):.2f} MB

**Recommendation**: {validation_result['recommendation'].upper()} (confidence: {validation_result['confidence']:.0%})

**Queue Status**: ✓ Added to download queue (entry_id: {entry.entry_id})

Use `execute_download_from_queue("{entry.entry_id}")` to download this dataset.
"""
            return report

        # Return validation report without queueing
        report = f"""
### SRA Dataset Validation Report

**Dataset ID**: {sra_id}
**Title**: {metadata.get('title', 'N/A')}

**Validation Checks**:
{self._format_validation_checks(validation_result['validation_checks'])}

**Recommendation**: {validation_result['recommendation'].upper()}
"""
        return report

    except Exception as e:
        logger.error(f"Error validating SRA {sra_id}: {e}")
        return f"Error validating SRA dataset: {str(e)}"

def _format_validation_checks(checks: Dict[str, str]) -> str:
    """Format validation checks for display."""
    lines = []
    for field, status in checks.items():
        icon = "✓" if status == "PASS" else "⚠" if status == "WARNING" else "✗"
        lines.append(f"- {icon} {field}: {status}")
    return "\n".join(lines)
```

#### Example: Add validate_pride_metadata tool

```python
@tool
def validate_pride_metadata(
    pride_id: str,
    required_fields: str = "",
    add_to_queue: bool = True,
) -> str:
    """
    Validate PRIDE dataset metadata and add to download queue.

    Args:
        pride_id: PRIDE accession (e.g., 'PXD012345')
        required_fields: Comma-separated list of required metadata fields
        add_to_queue: If True, add validated dataset to download queue

    Returns:
        Validation report with recommendation and queue confirmation
    """
    try:
        from lobster.tools.providers.pride_provider import PRIDEProvider
        import uuid
        from datetime import datetime
        from lobster.core.schemas.download_queue import (
            DownloadQueueEntry,
            DownloadStatus,
            StrategyConfig,
        )

        # 1. Fetch PRIDE metadata
        pride_provider = PRIDEProvider(data_manager)
        metadata = pride_provider.fetch_metadata(pride_id)

        # 2. Validate metadata
        validation_result = {
            "dataset_id": pride_id,
            "database": "pride",
            "validation_checks": {},
            "missing_fields": [],
            "recommendation": "proceed",
            "confidence": 0.95,
        }

        # Validation logic similar to SRA example
        # ...

        # 3. Extract URLs if proceeding
        if validation_result["recommendation"] == "proceed" and add_to_queue:
            url_data = pride_provider.get_download_urls(pride_id)

            # 4. Create DownloadQueueEntry
            entry = DownloadQueueEntry(
                entry_id=f"queue_{pride_id}_{uuid.uuid4().hex[:8]}",
                dataset_id=pride_id,
                database="pride",  # NEW database type
                priority=5,
                status=DownloadStatus.PENDING,
                metadata=metadata,
                validation_result=validation_result,

                # PRIDE-specific URL fields
                raw_urls=url_data.get("raw_urls", []),
                supplementary_urls=url_data.get("result_urls", []),
                mzml_urls=url_data.get("mzml_urls", []),  # Custom field

                # Download strategy
                recommended_strategy=StrategyConfig(
                    strategy_type="MZML_FIRST",  # Custom strategy
                    confidence=0.85,
                    rationale="mzML files are processed and ready for analysis",
                ),

                created_at=datetime.now(),
                updated_at=datetime.now(),
            )

            # 5. Add to queue
            data_manager.download_queue.add_entry(entry)

            # Return formatted report
            # ...

    except Exception as e:
        logger.error(f"Error validating PRIDE {pride_id}: {e}")
        return f"Error validating PRIDE dataset: {str(e)}"
```

---

### Step 3: Update data_expert (Download Execution)

Add download logic for your new database.

**Location**: `/Users/tyo/GITHUB/omics-os/lobster/lobster/agents/data_expert.py`

#### Extend execute_download_from_queue

```python
# Inside execute_download_from_queue tool:

# After retrieving queue entry:
entry = data_manager.download_queue.get_entry(entry_id)

if not entry:
    return f"Error: Queue entry {entry_id} not found"

# Check database type and route to appropriate service
if entry.database == "geo":
    # Existing GEO logic
    from lobster.tools.geo_service import GEOService
    service = GEOService(data_manager)

    result = service.download_dataset(
        geo_id=entry.dataset_id,
        force_download=force_download,
        prefer_h5ad=entry.recommended_strategy.strategy_type == "H5_FIRST",
    )

elif entry.database == "sra":
    # NEW: SRA download logic
    from lobster.tools.sra_service import SRAService
    service = SRAService(data_manager)

    result = service.download_sra_dataset(
        sra_id=entry.dataset_id,
        fastq_urls=entry.raw_urls,
        output_format="h5ad",
        metadata=entry.metadata,
    )

elif entry.database == "pride":
    # NEW: PRIDE proteomics logic
    from lobster.tools.pride_service import PRIDEService
    service = PRIDEService(data_manager)

    result = service.download_pride_dataset(
        pride_id=entry.dataset_id,
        mzml_urls=getattr(entry, "mzml_urls", []),
        raw_urls=entry.raw_urls,
        prefer_mzml=entry.recommended_strategy.strategy_type == "MZML_FIRST",
        metadata=entry.metadata,
    )

else:
    raise ValueError(f"Unsupported database: {entry.database}")

# Update queue entry status
if "error" in result.lower():
    data_manager.download_queue.update_entry_status(
        entry_id,
        DownloadStatus.FAILED,
        error_message=result,
    )
else:
    data_manager.download_queue.update_entry_status(
        entry_id,
        DownloadStatus.COMPLETED,
    )

return result
```

#### Create Service (If Needed)

```python
# File: lobster/tools/sra_service.py

from typing import List, Dict, Any, Optional
import subprocess
import os
from pathlib import Path
from lobster.utils.logger import get_logger
import scanpy as sc
import pandas as pd

logger = get_logger(__name__)


class SRAService:
    """Service for downloading and processing SRA datasets."""

    def __init__(self, data_manager):
        self.data_manager = data_manager
        self.cache_dir = Path.home() / ".cache" / "lobster" / "sra"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def download_sra_dataset(
        self,
        sra_id: str,
        fastq_urls: List[str],
        output_format: str = "h5ad",
        metadata: Optional[Dict] = None,
    ) -> str:
        """
        Download FASTQ files from SRA and convert to AnnData.

        Steps:
        1. Download FASTQ files
        2. Run quality control (optional)
        3. Align to reference OR use kallisto for quantification
        4. Convert to AnnData format
        5. Store in DataManagerV2

        Args:
            sra_id: SRA accession
            fastq_urls: List of FASTQ file URLs
            output_format: Output format (default: "h5ad")
            metadata: Optional metadata dict

        Returns:
            Success message with modality name
        """
        try:
            logger.info(f"Starting SRA download for {sra_id}")

            # Step 1: Download FASTQ files
            fastq_paths = self._download_fastq_files(sra_id, fastq_urls)

            logger.info(f"Downloaded {len(fastq_paths)} FASTQ files")

            # Step 2: Quality control (optional)
            # self._run_fastqc(fastq_paths)

            # Step 3: Quantification with kallisto (or STAR)
            counts_file = self._run_kallisto_quantification(sra_id, fastq_paths)

            # Step 4: Convert to AnnData
            adata = self._create_anndata(counts_file, sra_id, metadata)

            # Step 5: Store in DataManagerV2
            modality_name = f"sra_{sra_id.lower()}"
            self.data_manager.modalities[modality_name] = adata

            logger.info(f"Created modality '{modality_name}' with {adata.n_obs} cells × {adata.n_vars} genes")

            return f"""
### SRA Download Complete

**Dataset**: {sra_id}
**Modality**: `{modality_name}`
**Shape**: {adata.n_obs} samples × {adata.n_vars} genes
**Files Downloaded**: {len(fastq_paths)}

The dataset is now available for analysis.
"""

        except Exception as e:
            logger.error(f"Error downloading SRA {sra_id}: {e}")
            return f"Error: Failed to download SRA dataset: {str(e)}"

    def _download_fastq_files(self, sra_id: str, fastq_urls: List[str]) -> List[Path]:
        """Download FASTQ files from URLs."""
        downloaded_paths = []

        for i, url in enumerate(fastq_urls):
            output_path = self.cache_dir / f"{sra_id}_{i}.fastq.gz"

            if output_path.exists():
                logger.info(f"Using cached file: {output_path}")
                downloaded_paths.append(output_path)
                continue

            logger.info(f"Downloading {url}...")

            # Use wget or curl
            cmd = ["wget", "-O", str(output_path), url]
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                raise RuntimeError(f"Download failed: {result.stderr}")

            downloaded_paths.append(output_path)

        return downloaded_paths

    def _run_kallisto_quantification(self, sra_id: str, fastq_paths: List[Path]) -> Path:
        """
        Run kallisto quantification on FASTQ files.

        Note: Requires kallisto installed and reference index.
        """
        output_dir = self.cache_dir / f"{sra_id}_kallisto"
        output_dir.mkdir(exist_ok=True)

        # Example kallisto command (adjust for your setup)
        # kallisto quant -i index.idx -o output_dir fastq1.gz fastq2.gz

        # For demonstration, return path to abundance.tsv
        abundance_file = output_dir / "abundance.tsv"

        # Actual implementation would run kallisto here
        # ...

        return abundance_file

    def _create_anndata(
        self,
        counts_file: Path,
        sra_id: str,
        metadata: Optional[Dict],
    ) -> sc.AnnData:
        """Convert kallisto output to AnnData."""
        # Read kallisto abundance file
        counts_df = pd.read_csv(counts_file, sep="\t", index_col=0)

        # Create AnnData
        adata = sc.AnnData(X=counts_df[["tpm"]].values)
        adata.var_names = counts_df.index
        adata.obs_names = [sra_id]  # Single sample

        # Add metadata
        if metadata:
            for key, value in metadata.items():
                adata.uns[key] = value

        return adata
```

---

### Step 4: Update DownloadQueueEntry Schema (If Needed)

**Location**: `/Users/tyo/GITHUB/omics-os/lobster/lobster/core/schemas/download_queue.py`

#### Add Custom URL Fields (Optional)

```python
class DownloadQueueEntry(BaseModel):
    # ... existing fields ...

    # Generic URL fields (recommended)
    raw_urls: Optional[List[str]] = None
    supplementary_urls: Optional[List[str]] = None

    # Database-specific URL fields (add only if necessary)
    fastq_urls: Optional[List[str]] = None      # For SRA
    sra_urls: Optional[List[str]] = None        # For SRA
    mzml_urls: Optional[List[str]] = None       # For PRIDE
    mzxml_urls: Optional[List[str]] = None      # For MetabolomicsWorkbench

    # Existing fields
    matrix_url: Optional[str] = None
    h5_url: Optional[str] = None
```

**Best Practice**: Prefer using generic fields (`raw_urls`, `supplementary_urls`) to avoid schema bloat. Only add custom fields if they require different handling.

---

### Step 5: Testing

#### Unit Tests

**File**: `/Users/tyo/GITHUB/omics-os/lobster/tests/unit/tools/providers/test_sra_provider.py`

```python
import pytest
from lobster.tools.providers.sra_provider import SRAProvider
from lobster.tools.providers.base_provider import ProviderError


@pytest.fixture
def sra_provider(data_manager):
    """Create SRAProvider instance."""
    return SRAProvider(data_manager)


def test_sra_provider_get_download_urls(sra_provider):
    """Test URL extraction for real SRA dataset."""
    urls = sra_provider.get_download_urls("SRR12345678")

    assert "sra_id" in urls
    assert urls["sra_id"] == "SRR12345678"
    assert len(urls["fastq_urls"]) > 0
    assert len(urls["sra_urls"]) > 0
    assert urls["total_size_mb"] > 0
    assert urls["file_count"] > 0


def test_sra_provider_fetch_metadata(sra_provider):
    """Test metadata fetching for real SRA dataset."""
    metadata = sra_provider.fetch_metadata("SRR12345678")

    assert metadata["sra_id"] == "SRR12345678"
    assert "title" in metadata
    assert "organism" in metadata
    assert "library_strategy" in metadata
    assert metadata["n_samples"] > 0


def test_sra_provider_invalid_id(sra_provider):
    """Test error handling for invalid SRA ID."""
    with pytest.raises(ValueError, match="Invalid SRA ID"):
        sra_provider.get_download_urls("INVALID123")


def test_sra_provider_network_error(sra_provider, monkeypatch):
    """Test error handling for network failures."""
    def mock_get(*args, **kwargs):
        raise ConnectionError("Network error")

    monkeypatch.setattr("requests.get", mock_get)

    with pytest.raises(ProviderError, match="Network error"):
        sra_provider.fetch_metadata("SRR12345678")


@pytest.mark.parametrize("sra_id,expected_prefix", [
    ("SRR12345678", "SRR"),
    ("SRP123456", "SRP"),
    ("SRX987654", "SRX"),
    ("SRS111222", "SRS"),
])
def test_sra_provider_id_formats(sra_provider, sra_id, expected_prefix):
    """Test support for different SRA ID formats."""
    metadata = sra_provider.fetch_metadata(sra_id)
    assert metadata["sra_id"] == sra_id
    assert sra_id.startswith(expected_prefix)
```

#### Integration Tests

**File**: `/Users/tyo/GITHUB/omics-os/lobster/tests/integration/test_sra_queue_workflow.py`

```python
import pytest
from lobster.core.schemas.download_queue import DownloadStatus


@pytest.mark.integration
@pytest.mark.real_api
def test_sra_complete_workflow(integrated_system):
    """Test complete SRA workflow: research → queue → data_expert."""
    research = integrated_system["research_agent"]
    data_exp = integrated_system["data_expert"]
    dm = integrated_system["data_manager"]

    # Step 1: Validate and queue SRA dataset
    result = research.invoke({
        "messages": [
            {"role": "user", "content": "Validate SRA dataset SRR12345678 and add to queue"}
        ]
    })

    response = result["messages"][-1]["content"]
    assert "added to download queue" in response.lower()

    # Step 2: Verify queue entry created
    entries = dm.download_queue.list_entries(status=DownloadStatus.PENDING)
    assert len(entries) > 0

    sra_entry = next((e for e in entries if e.database == "sra"), None)
    assert sra_entry is not None
    assert sra_entry.dataset_id == "SRR12345678"
    assert len(sra_entry.raw_urls) > 0

    # Step 3: Download from queue
    result = data_exp.invoke({
        "messages": [
            {"role": "user", "content": f"Execute download from queue: {sra_entry.entry_id}"}
        ]
    })

    response = result["messages"][-1]["content"]
    assert "complete" in response.lower()

    # Step 4: Verify modality created
    modalities = dm.list_modalities()
    assert any("sra_srr12345678" in mod.lower() for mod in modalities)

    # Step 5: Verify queue entry marked as completed
    updated_entry = dm.download_queue.get_entry(sra_entry.entry_id)
    assert updated_entry.status == DownloadStatus.COMPLETED


@pytest.mark.integration
@pytest.mark.real_api
def test_sra_error_handling(integrated_system):
    """Test SRA workflow with invalid dataset ID."""
    research = integrated_system["research_agent"]

    result = research.invoke({
        "messages": [
            {"role": "user", "content": "Validate SRA dataset INVALID123"}
        ]
    })

    response = result["messages"][-1]["content"]
    assert "error" in response.lower() or "invalid" in response.lower()
```

---

## Adding New Download Strategies

### StrategyConfig Structure

**Location**: `/Users/tyo/GITHUB/omics-os/lobster/lobster/core/schemas/download_queue.py`

```python
from pydantic import BaseModel

class StrategyConfig(BaseModel):
    """
    Download strategy configuration.

    Defines how to prioritize download sources.
    """
    strategy_type: str      # String identifier (e.g., "MATRIX_FIRST")
    confidence: float       # 0.0-1.0 confidence score
    rationale: str          # Human-readable explanation
```

### Built-in Strategy Types

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `MATRIX_FIRST` | Download series matrix only | Fast, for GEO with sufficient matrix |
| `RAW_FIRST` | Download raw data files | Comprehensive, for re-analysis |
| `SUPPLEMENTARY_FIRST` | Download supplementary processed files | Pre-processed data available |
| `H5_FIRST` | Download H5AD if available | Fastest, for datasets with H5AD |
| `MZML_FIRST` | Download mzML files (proteomics) | Processed mass spec data |
| `COMPREHENSIVE` | Download all available files | Maximum completeness |
| `MINIMAL` | Download minimum required | Quick exploratory analysis |

### Adding Custom Strategy

#### Step 1: Define Strategy Type

```python
# In data_expert.py or separate strategy module

CUSTOM_STRATEGIES = {
    "SMART_DOWNLOAD": {
        "description": "Intelligently select fastest download path",
        "priority": ["h5_url", "matrix_url", "supplementary_urls", "raw_urls"],
        "logic": "Check availability and pick fastest non-empty source",
    },
    "COMPREHENSIVE": {
        "description": "Download all available files",
        "priority": "all",
        "logic": "Download every URL in entry",
    },
    "MINIMAL": {
        "description": "Download minimum required for analysis",
        "priority": ["matrix_url"],
        "logic": "Download only matrix file",
    },
    "SIZE_AWARE": {
        "description": "Download based on dataset size",
        "priority": "dynamic",
        "logic": "If <100 samples: matrix, else: h5ad or raw",
    },
}
```

#### Step 2: Implement Strategy Logic

```python
# In execute_download_from_queue tool:

def execute_strategy(entry: DownloadQueueEntry, strategy_type: str) -> str:
    """Execute download strategy."""

    if strategy_type == "SMART_DOWNLOAD":
        # Check what's available and pick fastest
        if entry.h5_url:
            return download_h5_only(entry)
        elif entry.matrix_url and is_sufficient(entry):
            return download_matrix_only(entry)
        elif entry.supplementary_urls:
            return download_supplementary_first(entry)
        else:
            return download_raw_files(entry)

    elif strategy_type == "COMPREHENSIVE":
        # Download everything
        results = []
        if entry.h5_url:
            results.append(download_h5_only(entry))
        if entry.matrix_url:
            results.append(download_matrix_only(entry))
        if entry.supplementary_urls:
            results.append(download_supplementary_files(entry))
        if entry.raw_urls:
            results.append(download_raw_files(entry))
        return "\n".join(results)

    elif strategy_type == "MINIMAL":
        # Download only matrix
        if entry.matrix_url:
            return download_matrix_only(entry)
        else:
            return "Error: No matrix URL available for minimal strategy"

    elif strategy_type == "SIZE_AWARE":
        # Dynamic strategy based on dataset size
        n_samples = entry.metadata.get("n_samples", 0)

        if n_samples < 100:
            return download_matrix_only(entry)
        elif entry.h5_url:
            return download_h5_only(entry)
        else:
            return download_raw_files(entry)

    else:
        raise ValueError(f"Unknown strategy: {strategy_type}")


def is_sufficient(entry: DownloadQueueEntry) -> bool:
    """Check if matrix file is sufficient for analysis."""
    metadata = entry.metadata

    # Matrix sufficient if:
    # - Small dataset (<100 samples)
    # - Has expression data
    # - Not missing critical metadata

    if metadata.get("n_samples", 0) > 100:
        return False

    if not metadata.get("has_expression_data"):
        return False

    return True
```

#### Step 3: Update research_agent to Recommend Strategy

```python
# In validate_dataset_metadata tool:

def recommend_strategy(metadata: Dict[str, Any]) -> StrategyConfig:
    """Recommend download strategy based on metadata."""

    n_samples = metadata.get("n_samples", 0)
    has_h5ad = metadata.get("has_h5ad", False)
    has_matrix = metadata.get("has_matrix", False)
    has_raw = metadata.get("has_raw_files", False)

    # Logic tree
    if has_h5ad:
        return StrategyConfig(
            strategy_type="H5_FIRST",
            confidence=0.95,
            rationale="H5AD file available (fastest download)",
        )

    elif has_matrix and n_samples < 100:
        return StrategyConfig(
            strategy_type="MATRIX_FIRST",
            confidence=0.85,
            rationale="Matrix file sufficient for small dataset",
        )

    elif n_samples > 500 and has_raw:
        return StrategyConfig(
            strategy_type="RAW_FIRST",
            confidence=0.8,
            rationale="Large dataset benefits from raw data re-analysis",
        )

    else:
        return StrategyConfig(
            strategy_type="SMART_DOWNLOAD",
            confidence=0.75,
            rationale="Auto-select optimal download path",
        )
```

---

## Testing Guidelines

### Unit Testing Checklist

- [ ] Provider URL extraction (10+ test datasets)
- [ ] Invalid ID handling (malformed accessions)
- [ ] Network error handling (timeouts, 404s)
- [ ] Metadata parsing edge cases (missing fields)
- [ ] Queue entry creation (all fields populated)
- [ ] Status transitions (PENDING → IN_PROGRESS → COMPLETED/FAILED)
- [ ] Strategy selection logic (all branches)
- [ ] Error messages (user-friendly, actionable)

### Integration Testing Checklist

- [ ] Complete workflow (research → queue → data_expert)
- [ ] Multi-dataset downloads (batch processing)
- [ ] Error recovery (failed downloads with retry)
- [ ] Concurrent download prevention (same dataset)
- [ ] Workspace persistence (queue survives restart)
- [ ] Modality creation (correct naming, metadata)
- [ ] Strategy execution (all strategy types)

### Performance Testing

**Target Metrics**:
- URL extraction: <2 seconds per dataset
- Queue operations: <100ms (add, get, update)
- Download completion: Depends on file size (benchmark with 10MB dataset)
- Metadata parsing: <500ms

**Performance Test Example**:

```python
import time
import statistics

def test_url_extraction_performance(sra_provider):
    """Ensure URL extraction meets performance targets."""
    datasets = [
        "SRR12345678", "SRR87654321", "SRR11223344",
        "SRR55667788", "SRR99001122", "SRR33445566",
        "SRR77889900", "SRR22334455", "SRR66778899",
        "SRR44556677",
    ]

    latencies = []

    for dataset_id in datasets:
        start = time.time()
        urls = sra_provider.get_download_urls(dataset_id)
        elapsed = (time.time() - start) * 1000  # Convert to ms

        latencies.append(elapsed)
        assert len(urls["fastq_urls"]) > 0

    # Calculate statistics
    mean_latency = statistics.mean(latencies)
    p95_latency = statistics.quantiles(latencies, n=20)[18]  # 95th percentile

    logger.info(f"URL extraction performance:")
    logger.info(f"  Mean: {mean_latency:.2f}ms")
    logger.info(f"  P95: {p95_latency:.2f}ms")
    logger.info(f"  Max: {max(latencies):.2f}ms")

    assert mean_latency < 2000, f"Mean latency {mean_latency:.2f}ms exceeds 2000ms target"


def test_queue_operations_performance(data_manager):
    """Test queue operation performance."""
    from lobster.core.schemas.download_queue import DownloadQueueEntry, DownloadStatus
    from datetime import datetime

    # Create 100 test entries
    entries = []
    for i in range(100):
        entry = DownloadQueueEntry(
            entry_id=f"test_entry_{i}",
            dataset_id=f"GSE{100000 + i}",
            database="geo",
            priority=5,
            status=DownloadStatus.PENDING,
            metadata={},
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        entries.append(entry)

    # Test add performance
    start = time.time()
    for entry in entries:
        data_manager.download_queue.add_entry(entry)
    add_time = (time.time() - start) * 1000
    avg_add = add_time / len(entries)

    # Test get performance
    start = time.time()
    for entry in entries:
        retrieved = data_manager.download_queue.get_entry(entry.entry_id)
        assert retrieved is not None
    get_time = (time.time() - start) * 1000
    avg_get = get_time / len(entries)

    # Test update performance
    start = time.time()
    for entry in entries:
        data_manager.download_queue.update_entry_status(
            entry.entry_id,
            DownloadStatus.COMPLETED,
        )
    update_time = (time.time() - start) * 1000
    avg_update = update_time / len(entries)

    logger.info(f"Queue operations performance (100 entries):")
    logger.info(f"  Add: {avg_add:.2f}ms per entry")
    logger.info(f"  Get: {avg_get:.2f}ms per entry")
    logger.info(f"  Update: {avg_update:.2f}ms per entry")

    assert avg_add < 100, f"Add operation {avg_add:.2f}ms exceeds 100ms target"
    assert avg_get < 100, f"Get operation {avg_get:.2f}ms exceeds 100ms target"
    assert avg_update < 100, f"Update operation {avg_update:.2f}ms exceeds 100ms target"
```

---

## Best Practices

### 1. Follow Existing Patterns

**Providers** extend `BaseProvider`:
```python
from lobster.tools.providers.base_provider import BaseProvider

class MyProvider(BaseProvider):
    def __init__(self, data_manager):
        super().__init__(data_manager)
```

**Services** are stateless:
```python
class MyService:
    def __init__(self, data_manager):
        self.data_manager = data_manager

    def process(self, data):
        # No instance state modification
        result = transform(data)
        return result
```

**Tools** validate → call service → log provenance:
```python
@tool
def my_tool(dataset_id: str) -> str:
    # 1. Validate
    if dataset_id not in data_manager.modalities:
        return "Error: Dataset not found"

    # 2. Call service
    service = MyService(data_manager)
    result = service.process(dataset_id)

    # 3. Log provenance
    data_manager.log_tool_usage("my_tool", {"dataset_id": dataset_id}, result)

    return format_result(result)
```

**Queue entries** contain ALL metadata (no re-fetching):
```python
entry = DownloadQueueEntry(
    entry_id=f"queue_{dataset_id}_{uuid.uuid4().hex[:8]}",
    dataset_id=dataset_id,
    database="geo",
    metadata=metadata,  # Complete metadata
    raw_urls=urls,      # All URLs pre-extracted
    # ... all necessary information
)
```

### 2. Error Handling

```python
from lobster.tools.providers.base_provider import ProviderError
from lobster.utils.logger import get_logger

logger = get_logger(__name__)

try:
    urls = provider.get_download_urls(dataset_id)
except ProviderError as e:
    # Recoverable error (e.g., network issue, rate limit)
    logger.warning(f"Provider error for {dataset_id}: {e}")
    return partial_result_with_warning(e)
except ValueError as e:
    # User error (e.g., invalid ID)
    logger.info(f"Invalid input: {e}")
    return f"Error: {str(e)}"
except Exception as e:
    # Unexpected error
    logger.error(f"Unexpected error: {e}", exc_info=True)
    raise
```

### 3. Logging

```python
from lobster.utils.logger import get_logger

logger = get_logger(__name__)

# INFO: User-facing actions
logger.info(f"Downloading dataset {dataset_id} ({total_size_mb:.2f} MB)")

# DEBUG: Internal details
logger.debug(f"Extracted {len(urls)} URLs from metadata")
logger.debug(f"Using strategy: {strategy_type}")

# WARNING: Recoverable issues
logger.warning(f"Missing optional field: {field_name}")
logger.warning(f"Rate limit approaching, throttling requests")

# ERROR: Failures
logger.error(f"Download failed: {str(e)}")
logger.error(f"Failed to parse metadata: {str(e)}", exc_info=True)
```

### 4. Documentation

Add comprehensive docstrings to all public methods:

```python
def get_download_urls(self, dataset_id: str) -> Dict[str, Any]:
    """
    Extract download URLs for dataset.

    Args:
        dataset_id: Dataset accession (e.g., 'GSE12345', 'SRR12345678')

    Returns:
        Dictionary with keys:
            - dataset_id (str): Original accession
            - primary_urls (List[str]): Main data file URLs
            - supplementary_urls (List[str]): Additional file URLs
            - metadata_url (str): Web metadata page URL
            - total_size_mb (float): Estimated download size
            - file_count (int): Number of files

    Raises:
        ValueError: Invalid dataset ID format
        ProviderError: Failed to fetch metadata or construct URLs

    Example:
        >>> provider = GEOProvider(data_manager)
        >>> urls = provider.get_download_urls("GSE12345")
        >>> print(f"Found {urls['file_count']} files ({urls['total_size_mb']:.2f} MB)")
    """
```

Update wiki pages:
- `wiki/18-architecture-overview.md` - Add new database to architecture
- `wiki/25-download-queue-system.md` - Document new strategies
- Create integration examples in `examples/` directory

### 5. Workspace Integration

Ensure queue persists across sessions:

```python
# In DataManagerV2.__init__
self.download_queue = DownloadQueue(workspace_path / "download_queue.jsonl")

# Queue automatically saves on each operation
self.download_queue.add_entry(entry)  # Saved to disk
self.download_queue.update_entry_status(...)  # Saved to disk
```

---

## Troubleshooting

### Issue: Provider URL extraction fails

**Symptoms**: `ProviderError: Failed to fetch metadata`

**Diagnostic Steps**:
1. Check API key environment variables:
   ```bash
   echo $NCBI_API_KEY
   echo $AWS_BEDROCK_ACCESS_KEY
   ```

2. Verify dataset ID format:
   ```python
   # Valid formats by database
   GEO: GSE12345, GSE123456
   SRA: SRR12345678, SRP123456
   PRIDE: PXD012345
   ```

3. Test with known-good dataset ID:
   ```python
   provider = SRAProvider(data_manager)
   urls = provider.get_download_urls("SRR12345678")  # Public dataset
   ```

4. Check API rate limits:
   ```python
   # NCBI without API key: 3 requests/second
   # NCBI with API key: 10 requests/second
   import time
   time.sleep(0.34)  # Respect rate limit
   ```

5. Inspect raw API response:
   ```python
   response = requests.get(api_url, params=params)
   print(response.status_code)
   print(response.text)  # Raw response
   ```

**Solutions**:
- Set API key in environment
- Validate ID format before API call
- Implement retry logic with exponential backoff
- Add rate limiting (use `time.sleep()`)

### Issue: Queue entry not created

**Symptoms**: `validate_*_metadata` succeeds but no queue entry

**Diagnostic Steps**:
1. Check `add_to_queue=True` parameter:
   ```python
   validate_sra_metadata("SRR12345678", add_to_queue=True)  # Must be True
   ```

2. Verify download_queue initialized:
   ```python
   assert data_manager.download_queue is not None
   ```

3. Check workspace path permissions:
   ```bash
   ls -la ~/.cache/lobster/default/download_queue.jsonl
   ```

4. Inspect `download_queue.jsonl` file:
   ```bash
   cat ~/.cache/lobster/default/download_queue.jsonl
   tail -f ~/.cache/lobster/default/download_queue.jsonl  # Watch in real-time
   ```

5. Check for exceptions in logs:
   ```python
   logger.setLevel("DEBUG")
   # Re-run validation
   ```

**Solutions**:
- Ensure `add_to_queue=True` passed to validation tool
- Initialize DownloadQueue in DataManagerV2.__init__
- Fix workspace directory permissions (chmod 755)
- Check disk space (df -h)

### Issue: Download hangs

**Symptoms**: Status stuck on `IN_PROGRESS`, no progress

**Diagnostic Steps**:
1. Check network connectivity:
   ```bash
   ping ftp.ncbi.nlm.nih.gov
   curl -I https://ftp.ncbi.nlm.nih.gov/
   ```

2. Verify FTP/HTTP URLs are accessible:
   ```bash
   wget --spider <url>  # Check without downloading
   ```

3. Check download process:
   ```bash
   ps aux | grep lobster  # Check if process running
   lsof -p <pid>          # Check open files
   ```

4. Review logs:
   ```bash
   tail -f ~/.cache/lobster/lobster.log
   ```

**Solutions**:
- Implement download timeout:
  ```python
  response = requests.get(url, timeout=300)  # 5 minute timeout
  ```

- Add progress logging:
  ```python
  for i, url in enumerate(urls):
      logger.info(f"Downloading {i+1}/{len(urls)}: {url}")
      download_file(url)
  ```

- Test with smaller dataset first
- Use streaming downloads for large files:
  ```python
  with requests.get(url, stream=True) as r:
      with open(output_path, 'wb') as f:
          for chunk in r.iter_content(chunk_size=8192):
              f.write(chunk)
  ```

### Issue: Modality not created

**Symptoms**: Download completes but no modality in DataManagerV2

**Diagnostic Steps**:
1. Check service stores modality:
   ```python
   # In service
   self.data_manager.modalities[modality_name] = adata
   logger.info(f"Stored modality: {modality_name}")
   ```

2. Verify modality naming convention:
   ```python
   # Correct: lowercase, underscores
   modality_name = f"sra_{sra_id.lower()}"  # ✓
   modality_name = f"SRA-{sra_id}"          # ✗
   ```

3. Inspect data_manager.list_modalities():
   ```python
   modalities = data_manager.list_modalities()
   print(modalities)
   ```

4. Check for exceptions in service:
   ```python
   try:
       adata = self._create_anndata(...)
       self.data_manager.modalities[name] = adata
   except Exception as e:
       logger.error(f"Failed to create modality: {e}", exc_info=True)
       raise
   ```

**Solutions**:
- Ensure service calls `data_manager.modalities[name] = adata`
- Follow naming convention: `{database}_{dataset_id.lower()}`
- Add error handling in service methods
- Verify AnnData object is valid:
  ```python
  assert adata.n_obs > 0
  assert adata.n_vars > 0
  ```

### Issue: Strategy not working

**Symptoms**: Wrong files downloaded, strategy ignored

**Diagnostic Steps**:
1. Check strategy set in queue entry:
   ```python
   entry = data_manager.download_queue.get_entry(entry_id)
   print(entry.recommended_strategy)
   ```

2. Verify strategy logic:
   ```python
   # In execute_download_from_queue
   if entry.recommended_strategy:
       strategy_type = entry.recommended_strategy.strategy_type
       logger.info(f"Using strategy: {strategy_type}")
   ```

3. Test strategy with different datasets:
   ```python
   # Small dataset → MATRIX_FIRST
   # Large dataset → RAW_FIRST
   # Has H5AD → H5_FIRST
   ```

**Solutions**:
- Set strategy explicitly in validation tool:
  ```python
  entry.recommended_strategy = StrategyConfig(
      strategy_type="SMART_DOWNLOAD",
      confidence=0.85,
      rationale="Auto-select optimal path",
  )
  ```

- Add fallback strategy:
  ```python
  strategy_type = (
      entry.recommended_strategy.strategy_type
      if entry.recommended_strategy
      else "SMART_DOWNLOAD"  # Default
  )
  ```

- Log strategy execution:
  ```python
  logger.info(f"Executing strategy {strategy_type}: {rationale}")
  ```

---

## Examples

### Complete SRA Extension Example

See working implementation:
- **Provider**: `/Users/tyo/GITHUB/omics-os/lobster/lobster/tools/providers/sra_provider.py`
- **Service**: `/Users/tyo/GITHUB/omics-os/lobster/lobster/tools/sra_service.py`
- **Tests**: `/Users/tyo/GITHUB/omics-os/lobster/tests/unit/tools/providers/test_sra_provider.py`
- **Integration**: `/Users/tyo/GITHUB/omics-os/lobster/tests/integration/test_sra_queue_workflow.py`

### Complete PRIDE Extension Example

See working implementation:
- **Provider**: `/Users/tyo/GITHUB/omics-os/lobster/lobster/tools/providers/pride_provider.py`
- **Service**: `/Users/tyo/GITHUB/omics-os/lobster/lobster/tools/pride_service.py`
- **Tests**: `/Users/tyo/GITHUB/omics-os/lobster/tests/unit/tools/providers/test_pride_provider.py`

### MetabolomicsWorkbench Extension (Future)

Planned implementation:
- **Database**: MetabolomicsWorkbench (https://www.metabolomicsworkbench.org/)
- **Accession Format**: ST (Study), AN (Analysis)
- **Data Types**: LC-MS, GC-MS, NMR
- **Strategy**: Download mzML or processed tables

---

## See Also

- [Download Queue System (Wiki 25)](/Users/tyo/GITHUB/omics-os/lobster/wiki/25-download-queue-system.md)
- [Architecture Overview (Wiki 18)](/Users/tyo/GITHUB/omics-os/lobster/wiki/18-architecture-overview.md)
- [Provider Base Class](/Users/tyo/GITHUB/omics-os/lobster/lobster/tools/providers/base_provider.py)
- [Integration Tests](/Users/tyo/GITHUB/omics-os/lobster/tests/integration/)
- [Research Agent Phase 7 Summary](/Users/tyo/GITHUB/omics-os/lobster/kevin_notes/publisher/research_agent_refactor_phase_7_summary.md)

---

## Questions or Issues?

- **GitHub Issues**: https://github.com/the-omics-os/lobster/issues
- **Documentation**: https://github.com/the-omics-os/lobster.wiki
- **Developer Chat**: Contact maintainers via GitHub Discussions

---

**Last Updated**: 2025-11-14 (Phase 2 completion)
**Version**: 2.2.0
**Maintainer**: Lobster AI Development Team
