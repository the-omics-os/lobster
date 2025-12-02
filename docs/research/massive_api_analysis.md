# MassIVE Database API Integration Analysis

**Date:** 2025-12-01
**Purpose:** Research MassIVE API capabilities to inform proteomics database integration strategy
**Context:** Lobster currently supports GEO (transcriptomics). Expanding to proteomics databases (PRIDE, MassIVE, etc.)

---

## Executive Summary

**Key Finding:** MassIVE offers **two API layers** with varying maturity:

1. **PROXI API (v0.1)** - Standardized, ProteomeXchange-compliant REST API (RECOMMENDED)
2. **Legacy JSON Endpoints** - Undocumented, servlet-based, less stable

**Recommendation:** Use PROXI API as primary integration point. MassIVE is a **Phase 1 priority** alongside PRIDE due to:
- Standardized API (PROXI) shared with PRIDE
- Large repository (100,000+ datasets)
- ProteomeXchange cross-referencing
- Metabolomics coverage (valuable for multi-omics)

**Integration Complexity:** MEDIUM (similar to PRIDE, simpler than web scraping approaches)

---

## 1. MassIVE Database Overview

### 1.1 Repository Profile

| Attribute | Details |
|-----------|---------|
| **Host Institution** | UC San Diego (CCMS - Center for Computational Mass Spectrometry) |
| **Mission** | Free, community-driven proteomics/metabolomics data exchange |
| **Dataset Count** | 100,000+ datasets (as of Nov 2025) |
| **Data Types** | Proteomics (DDA/DIA), Metabolomics, Lipidomics |
| **ProteomeXchange Member** | Yes (partner since 2014) |
| **Accession Format** | MSV######  (e.g., MSV000100000) |
| **Related Platform** | GNPS (Global Natural Products Social Molecular Networking) |

### 1.2 Dataset Structure

**Typical Dataset Contents:**
- Raw mass spectrometry files (mzML, mzXML, vendor formats)
- Identification results (mzTab, TSV)
- Search parameters and workflows
- Sample metadata (species, instrument, PI, keywords)
- ProteomeXchange cross-references (PXD identifiers when applicable)

**File Access:**
- FTP: `ftp://massive.ucsd.edu/v{volume}/MSV{dataset_number}/`
- HTTP: Via ProteoSAFe web interface
- PROXI API: Metadata only (links to FTP for files)

---

## 2. API Capabilities

### 2.1 PROXI API (Recommended)

**Base URL:** `https://massive.ucsd.edu/ProteoSAFe/proxi/v0.1/`

**Specification:** HUPO-PSI ProteomeXchange API standard (shared with PRIDE, jPOST)

#### Endpoints

| Endpoint | Method | Purpose | Example |
|----------|--------|---------|---------|
| `/datasets` | GET | Search datasets by filters | `?species=human&pageSize=50` |
| `/datasets/{accession}` | GET | Retrieve single dataset metadata | `/datasets/MSV000100000` |

#### Query Parameters (Search)

```python
params = {
    "species": str,           # e.g., "human", "mouse" (NCBI Taxon name)
    "accession": str,         # MSV accession number
    "contact": str,           # PI name or email
    "publication": str,       # PMID or DOI
    "modification": str,      # PTM keyword
    "instrument": str,        # MS instrument type
    "resultType": str,        # "compact" or "full" (default: compact)
    "pageSize": int,          # Results per page (default: 100, max: 500)
    "pageOffset": int         # Pagination offset
}
```

#### Response Format (JSON with CV terms)

```json
{
  "accession": [{"cvLabel": "MS", "accession": "MS:1002487", "value": "MSV000100000"}],
  "title": "Environmental water samples positive",
  "summary": "Water samples collected from city stream...",
  "species": [[
    {"cvLabel": "MS", "accession": "MS:1001467", "value": "33858"},
    {"cvLabel": "MS", "accession": "MS:1001469", "value": "environmental samples"}
  ]],
  "instruments": [{"cvLabel": "MS", "name": "Orbitrap Exploris 240", "accession": "MS:1003094"}],
  "datasetLink": [
    {"cvLabel": "MS", "name": "MassIVE dataset URI", "value": "https://massive.ucsd.edu/..."},
    {"cvLabel": "MS", "name": "Dataset FTP location", "value": "ftp://massive.ucsd.edu/v11/MSV000100000"}
  ],
  "contacts": [[
    {"cvLabel": "MS", "name": "contact name", "value": "Giovana Anceski Bataglion"},
    {"cvLabel": "MS", "name": "contact email", "value": "giovanabataglion@gmail.com"},
    {"cvLabel": "MS", "name": "contact affiliation", "value": "Federal University of Amazonas"}
  ]],
  "publications": [{"cvLabel": "MS", "name": "Dataset with no associated published manuscript"}],
  "keywords": [
    {"cvLabel": "MS", "name": "submitter keyword", "value": "water"},
    {"cvLabel": "MS", "name": "submitter keyword", "value": "DatasetType:Other (environmental)"}
  ]
}
```

**Pros:**
- Standardized across PRIDE, MassIVE, jPOST (code reuse)
- CV-based metadata (ontology compliance)
- Includes FTP download URLs
- Paginated search
- No authentication required for public datasets

**Cons:**
- v0.1 specification (may evolve)
- Limited to metadata (file lists require FTP or separate API)

### 2.2 Legacy JSON Endpoints (Not Recommended)

**Base URL:** `https://massive.ucsd.edu/ProteoSAFe/`

#### Undocumented Endpoints

| Endpoint | Purpose | Status |
|----------|---------|--------|
| `datasets_json.jsp?pageSize=N` | List datasets (JSON) | Working but undocumented |
| `QueryDatasets?query={...}` | Complex search | Requires knowledge of internal schema |
| `result_json.jsp?task={id}&view={view}` | Job results (GNPS workflows) | Not for dataset metadata |
| `MassiveServlet?function=...` | Backend servlet API | Internal, unstable |

**Pros:**
- Detailed dataset lists (file counts, PSM counts, etc.)
- Direct JSON access without CV overhead

**Cons:**
- No official documentation
- Schema subject to change without notice
- Requires reverse-engineering
- Not standardized (MassIVE-specific)

---

## 3. Comparison with PRIDE API

| Feature | MassIVE (PROXI) | PRIDE (REST v2) | Winner |
|---------|----------------|-----------------|--------|
| **API Maturity** | v0.1 (emerging) | v2 (stable) | PRIDE |
| **Documentation** | PROXI spec (shared) | Comprehensive Swagger docs | PRIDE |
| **Standardization** | PROXI (HUPO-PSI) | Custom + PROXI support | TIE |
| **Dataset Count** | 100,000+ | 40,000+ | MassIVE |
| **Data Types** | Proteomics + Metabolomics | Proteomics only | MassIVE |
| **File Access** | FTP (via API links) | FTP + Aspera + API | PRIDE |
| **Search Flexibility** | Basic (species, PI, keywords) | Advanced (PTMs, tissues, diseases) | PRIDE |
| **Rate Limits** | None documented | 10 req/sec (documented) | PRIDE |
| **Response Format** | CV terms (structured) | CV terms + user-friendly fields | PRIDE |
| **ProteomeXchange** | Yes (native) | Yes (native) | TIE |
| **Authentication** | Not required | Not required (public data) | TIE |

**Verdict:** PRIDE has more mature API, but MassIVE adds metabolomics and is equally accessible via PROXI.

---

## 4. ProteomeXchange Central API

**Key Discovery:** ProteomeXchange Central does NOT provide a unified API for querying across repositories.

**What it offers:**
- Individual PXD identifier resolution (redirects to host repository)
- RSS feed for new datasets
- Manual browsing interface

**Implication:** Must integrate with each repository (PRIDE, MassIVE, jPOST) individually. However, PROXI standardization makes this easier.

---

## 5. Integration Architecture Recommendation

### 5.1 Proposed Approach

**Pattern:** Adapt existing GEO integration pattern to PROXI-based providers

```
research_agent (search/validate)
    ↓
DownloadQueue (PENDING entry with MassIVE MSV accession)
    ↓
supervisor (detects queue entry)
    ↓
data_expert (execute download via MassIVEDownloadService)
    ↓
MassIVEProvider (PROXI API for metadata)
    ↓
Download files via FTP (using URLs from PROXI response)
    ↓
MassIVEAdapter (load mzTab/MaxQuant/Spectronaut → AnnData)
    ↓
DataManagerV2 (store as proteomics modality)
```

### 5.2 Recommended Components

#### New Files to Create

```
lobster/tools/providers/massive_provider.py          # PROXI API client
lobster/services/data_access/massive_service.py      # MassIVE download service
lobster/services/data_access/massive_download_service.py  # IDownloadService impl
lobster/core/interfaces/massive_adapter.py           # mzTab/vendor file loading (may reuse existing)
```

#### Modifications to Existing Files

```
lobster/tools/download_orchestrator.py               # Register MassIVEDownloadService
lobster/agents/research_agent.py                     # Add MassIVE URL detection
lobster/config/agent_registry.py                     # Update data_expert capabilities (optional)
```

### 5.3 Implementation Steps (Phased)

**Phase 1: PROXI Provider (1-2 days)**
- [ ] Create `MassIVEProvider` with PROXI API client
- [ ] Implement search (species, keywords, accession)
- [ ] Parse CV-based responses
- [ ] Extract FTP URLs for file download
- [ ] Unit tests with real MassIVE datasets

**Phase 2: Download Service (1-2 days)**
- [ ] Create `MassIVEDownloadService(IDownloadService)`
- [ ] Implement FTP download logic (reuse GEO patterns)
- [ ] Handle multiple file types (mzML, mzTab, vendor formats)
- [ ] Register with `DownloadOrchestrator`
- [ ] Integration tests

**Phase 3: File Parsing (2-3 days)**
- [ ] Identify target file formats (mzTab most common)
- [ ] Create/extend adapter for mzTab → AnnData conversion
- [ ] Handle vendor-specific formats (MaxQuant, Spectronaut, Progenesis)
- [ ] Schema validation (ProteomicsSchema)
- [ ] Parser tests

**Phase 4: Agent Integration (1 day)**
- [ ] Update `research_agent` to detect MSV accessions
- [ ] Update `data_expert` tools (if needed)
- [ ] End-to-end workflow test
- [ ] Documentation

**Total Estimate:** 5-8 days (similar to PRIDE integration effort)

---

## 6. Code Examples

### 6.1 MassIVE Provider (PROXI Client)

```python
from typing import List, Dict, Optional
import requests
from lobster.tools.providers.base_provider import BaseProvider

class MassIVEProvider(BaseProvider):
    """
    Provider for MassIVE database using PROXI API.

    Docs: https://ccms-ucsd.github.io/MassIVEDocumentation/api/
    Spec: https://github.com/HUPO-PSI/proxi-schemas
    """

    BASE_URL = "https://massive.ucsd.edu/ProteoSAFe/proxi/v0.1"

    def search_datasets(
        self,
        species: Optional[str] = None,
        keywords: Optional[str] = None,
        accession: Optional[str] = None,
        page_size: int = 100,
        page_offset: int = 0
    ) -> List[Dict]:
        """
        Search MassIVE datasets using PROXI API.

        Args:
            species: Species name (e.g., "human", "mouse")
            keywords: Keyword search
            accession: MSV accession number
            page_size: Results per page (max 500)
            page_offset: Pagination offset

        Returns:
            List of dataset metadata dictionaries
        """
        params = {
            "resultType": "full",
            "pageSize": page_size,
            "pageOffset": page_offset
        }
        if species:
            params["species"] = species
        if keywords:
            params["keywords"] = keywords
        if accession:
            params["accession"] = accession

        response = requests.get(
            f"{self.BASE_URL}/datasets",
            params=params,
            timeout=30
        )
        response.raise_for_status()
        return response.json()

    def get_dataset(self, accession: str) -> Dict:
        """
        Retrieve single dataset metadata.

        Args:
            accession: MSV accession (e.g., "MSV000100000")

        Returns:
            Dataset metadata with FTP download URLs
        """
        response = requests.get(
            f"{self.BASE_URL}/datasets/{accession}",
            timeout=30
        )
        response.raise_for_status()
        data = response.json()

        # Extract FTP URL from CV terms
        ftp_url = self._extract_cv_value(
            data.get("datasetLink", []),
            "Dataset FTP location"
        )
        return {
            "accession": accession,
            "title": data.get("title", ""),
            "summary": data.get("summary", ""),
            "species": self._extract_species(data),
            "instruments": self._extract_instruments(data),
            "ftp_url": ftp_url,
            "contacts": self._extract_contacts(data),
            "raw_metadata": data
        }

    def _extract_cv_value(self, cv_list: List[Dict], name: str) -> Optional[str]:
        """Extract value from CV term list by name."""
        for item in cv_list:
            if isinstance(item, dict) and item.get("name") == name:
                return item.get("value")
        return None

    def _extract_species(self, data: Dict) -> List[str]:
        """Extract species names from nested CV terms."""
        species_list = []
        for species_group in data.get("species", []):
            for cv_term in species_group:
                if cv_term.get("name") == "taxonomy: scientific name":
                    species_list.append(cv_term.get("value", ""))
        return species_list

    def _extract_instruments(self, data: Dict) -> List[str]:
        """Extract instrument names."""
        return [inst.get("name", "") for inst in data.get("instruments", [])]

    def _extract_contacts(self, data: Dict) -> List[Dict]:
        """Extract contact information."""
        contacts = []
        for contact_group in data.get("contacts", []):
            contact_dict = {}
            for cv_term in contact_group:
                name_key = cv_term.get("name", "")
                if "contact name" in name_key:
                    contact_dict["name"] = cv_term.get("value", "")
                elif "contact email" in name_key:
                    contact_dict["email"] = cv_term.get("value", "")
                elif "contact affiliation" in name_key:
                    contact_dict["affiliation"] = cv_term.get("value", "")
            if contact_dict:
                contacts.append(contact_dict)
        return contacts
```

### 6.2 Download Service Integration

```python
from typing import Tuple, Dict, Any, Optional
from anndata import AnnData
from lobster.core.interfaces.download_service import IDownloadService
from lobster.core.provenance import AnalysisStep
from lobster.tools.providers.massive_provider import MassIVEProvider
import ftplib
import os

class MassIVEDownloadService(IDownloadService):
    """
    Download service for MassIVE datasets.
    Implements IDownloadService interface for DownloadOrchestrator.
    """

    def __init__(self, data_manager):
        self.data_manager = data_manager
        self.provider = MassIVEProvider()

    def supports_database(self, database: str) -> bool:
        """Check if this service supports the given database."""
        return database.lower() in ["massive", "msv"]

    def download_dataset(
        self,
        queue_entry,
        strategy_override: Optional[Dict] = None
    ) -> Tuple[AnnData, Dict[str, Any], AnalysisStep]:
        """
        Download and parse MassIVE dataset.

        Args:
            queue_entry: DownloadQueueEntry with MSV accession
            strategy_override: Optional strategy parameters

        Returns:
            (adata, stats, ir) tuple
        """
        accession = queue_entry.accession

        # Step 1: Get metadata via PROXI API
        metadata = self.provider.get_dataset(accession)
        ftp_url = metadata["ftp_url"]

        # Step 2: Download files via FTP
        local_dir = self._download_ftp_files(ftp_url, accession)

        # Step 3: Identify target files (e.g., mzTab, MaxQuant results)
        target_file = self._find_quantification_file(local_dir)

        # Step 4: Parse to AnnData
        adapter = self._get_adapter(target_file)
        adata = adapter.load(target_file, metadata=metadata)

        # Step 5: Store in DataManagerV2
        modality_name = f"massive_{accession.lower()}"
        self.data_manager.modalities[modality_name] = adata

        # Step 6: Build stats and IR
        stats = {
            "accession": accession,
            "title": metadata["title"],
            "species": metadata["species"],
            "n_obs": adata.n_obs,
            "n_vars": adata.n_vars,
            "source_file": target_file
        }

        ir = AnalysisStep(
            operation="massive.download_dataset",
            tool_name="MassIVEDownloadService",
            description=f"Downloaded MassIVE dataset {accession}",
            library="lobster",
            code_template="# Download via PROXI API + FTP",
            imports=["requests", "ftplib"],
            parameters={"accession": accession},
            input_entities=[],
            output_entities=[{"name": modality_name, "type": "AnnData"}]
        )

        return adata, stats, ir

    def _download_ftp_files(self, ftp_url: str, accession: str) -> str:
        """Download files from FTP URL."""
        # Parse FTP URL: ftp://massive.ucsd.edu/v11/MSV000100000
        parts = ftp_url.replace("ftp://", "").split("/")
        host = parts[0]
        path = "/" + "/".join(parts[1:])

        local_dir = os.path.join(
            self.data_manager.workspace.path,
            "downloads",
            accession
        )
        os.makedirs(local_dir, exist_ok=True)

        # Connect and download (simplified - add error handling)
        ftp = ftplib.FTP(host)
        ftp.login()  # Anonymous FTP
        ftp.cwd(path)

        # Download key files (mzTab, txt, etc.)
        for filename in ftp.nlst():
            if self._is_target_file(filename):
                local_path = os.path.join(local_dir, filename)
                with open(local_path, "wb") as f:
                    ftp.retrbinary(f"RETR {filename}", f.write)

        ftp.quit()
        return local_dir

    def _is_target_file(self, filename: str) -> bool:
        """Check if file is a target quantification file."""
        target_extensions = [
            ".mzTab", ".txt", ".tsv",  # Quantification tables
            ".xlsx",                    # Excel exports
            "proteinGroups.txt",        # MaxQuant
            "peptides.txt",             # MaxQuant
            "_Report.tsv"               # Spectronaut
        ]
        return any(ext in filename for ext in target_extensions)

    def _find_quantification_file(self, directory: str) -> str:
        """Identify main quantification file."""
        # Priority: mzTab > MaxQuant > Spectronaut > generic TSV
        for filename in os.listdir(directory):
            if filename.endswith(".mzTab"):
                return os.path.join(directory, filename)
        # Add more logic for other formats
        raise FileNotFoundError("No quantification file found in MassIVE dataset")

    def _get_adapter(self, filepath: str):
        """Get appropriate adapter for file type."""
        # Return MzTabAdapter, MaxQuantAdapter, etc.
        # This would use existing proteomics adapters
        from lobster.core.interfaces.modality_adapter import get_adapter
        return get_adapter(filepath)

    def validate_strategy_params(self, params: Dict) -> Tuple[bool, Optional[str]]:
        """Validate strategy parameters."""
        return True, None  # MassIVE has no special strategies yet

    def get_supported_strategies(self) -> List[str]:
        """List supported download strategies."""
        return ["default"]  # Add "mztab_only", "maxquant_only", etc.
```

---

## 7. Priority Recommendation

### Phase 1: High Priority (Immediate)
1. **PRIDE** - Most mature API, largest proteomics-only repository
2. **MassIVE** - PROXI standardization, adds metabolomics

### Phase 2: Medium Priority (3-6 months)
3. **jPOST** - Japanese datasets, PROXI support
4. **PeptideAtlas** - SRM/targeted proteomics

### Phase 3: Low Priority (6-12 months)
5. **iProX** - Chinese datasets (language barriers)
6. **Panorama Public** - Skyline-centric (niche use case)

**Rationale for MassIVE Phase 1:**
- PROXI API reduces development effort (code reuse with PRIDE)
- 100,000+ datasets (largest public repository)
- Metabolomics support enables multi-omics workflows (strategic for Omics-OS)
- FTP access is straightforward
- No authentication barriers
- ProteomeXchange integration (cross-referencing with PRIDE)

---

## 8. Potential Challenges

### Technical Challenges

1. **File Format Diversity**
   - MassIVE hosts vendor formats (Thermo .raw, Agilent .d, Waters .raw)
   - Parsing requires msconvert or vendor SDKs (not Python-native)
   - **Mitigation:** Focus on converted formats (mzML, mzTab) initially

2. **Dataset Completeness**
   - Not all datasets have quantification tables
   - Some are "Partial" status (raw files only)
   - **Mitigation:** Filter for "Complete" datasets with quant_analysis field

3. **Metabolomics vs Proteomics**
   - MassIVE has mixed data types (need to distinguish)
   - Keywords field has "DatasetType:Proteomics" vs "DatasetType:Metabolomics"
   - **Mitigation:** Filter by keywords in search

4. **PROXI v0.1 Evolution**
   - Specification may change (currently v0.1)
   - **Mitigation:** Abstract API calls in provider layer (easy to update)

### Operational Challenges

1. **FTP Reliability**
   - Large files (multi-GB), slow downloads
   - **Mitigation:** Add retry logic, resume support, Aspera alternative (if available)

2. **Storage Costs**
   - Raw MS files are huge (10-100 GB per dataset)
   - **Mitigation:** Download quantification tables only (default strategy)

3. **Rate Limits**
   - Not documented for MassIVE
   - **Mitigation:** Implement conservative rate limiting (1 req/sec), monitor

---

## 9. Comparison with Current Lobster Capabilities

### What Lobster Has (GEO Integration)

```python
# Existing pattern
geo_provider.search_geo(query="GSE12345")
geo_download_service.download_dataset(entry)  # Downloads SRA files
geo_parser.parse_to_anndata(files)            # Converts to AnnData
```

### What MassIVE Integration Adds

```python
# New pattern (parallel structure)
massive_provider.search_datasets(species="human", keywords="cancer")
massive_download_service.download_dataset(entry)  # Downloads mzTab/MaxQuant
massive_adapter.parse_to_anndata(files)           # Converts to AnnData
```

**Architectural Fit:** Perfect fit with existing `IDownloadService` + `DownloadOrchestrator` pattern

---

## 10. Success Criteria

**Minimum Viable Integration (Phase 1):**
- [ ] Search MassIVE by accession number (MSV######)
- [ ] Retrieve metadata via PROXI API
- [ ] Download mzTab files via FTP
- [ ] Parse mzTab to AnnData (protein-level quantification)
- [ ] Store as proteomics modality in DataManagerV2
- [ ] End-to-end test with 3 public datasets

**Full Integration (Phase 2):**
- [ ] Search by species, keywords, PI
- [ ] Support MaxQuant output (proteinGroups.txt)
- [ ] Support Spectronaut output (_Report.tsv)
- [ ] Handle missing values (proteomics-specific)
- [ ] Link to publications (PMID/DOI in metadata)
- [ ] Cross-reference with PRIDE (PXD identifiers)

---

## 11. References

### Official Documentation
- MassIVE Homepage: https://massive.ucsd.edu
- MassIVE Documentation: https://ccms-ucsd.github.io/MassIVEDocumentation/
- GNPS Documentation: https://ccms-ucsd.github.io/GNPSDocumentation/
- PROXI Specification: https://github.com/HUPO-PSI/proxi-schemas
- ProteomeXchange: https://www.proteomexchange.org

### API Endpoints
- PROXI Base: `https://massive.ucsd.edu/ProteoSAFe/proxi/v0.1/`
- Dataset JSON: `https://massive.ucsd.edu/ProteoSAFe/datasets_json.jsp`
- FTP Root: `ftp://massive.ucsd.edu/`

### Test Datasets
- MSV000100000 - Environmental metabolomics (small, simple)
- MSV000100058 - Proteomics with quantification (complete)
- MSV000099979 - SILAC-iTRAQ-TAILS (complex proteomics)

---

## 12. Next Steps

**Immediate Actions (This Week):**
1. Create GitHub issue: "Add MassIVE database support via PROXI API"
2. Prototype `MassIVEProvider` (PROXI client)
3. Test with 3 datasets (different formats)
4. Design `MassIVEAdapter` for mzTab parsing

**Short-Term (Next Sprint):**
5. Implement `MassIVEDownloadService`
6. Integrate with `DownloadOrchestrator`
7. Add unit tests + integration tests
8. Update `research_agent` for MSV detection

**Long-Term (Next Quarter):**
9. Add MaxQuant/Spectronaut support
10. Implement metabolomics-specific workflows
11. Cross-link with PRIDE datasets (PXD identifiers)
12. User documentation + example notebooks

---

## Appendix A: API Response Examples

### A.1 PROXI Dataset Response (MSV000100000)

```json
{
  "accession": [{"cvLabel": "MS", "accession": "MS:1002487", "value": "MSV000100000"}],
  "title": "Environmental water samples positive",
  "summary": "Water samples collected from an city stream analyzed under positive ionization mode",
  "species": [[
    {"cvLabel": "MS", "accession": "MS:1001467", "value": "33858"},
    {"cvLabel": "MS", "accession": "MS:1001469", "value": "environmental samples <Bacillariophyta>"}
  ]],
  "instruments": [
    {"cvLabel": "MS", "name": "Orbitrap Exploris 240", "accession": "MS:1003094"}
  ],
  "datasetLink": [
    {"cvLabel": "MS", "name": "MassIVE dataset URI",
     "value": "https://massive.ucsd.edu/ProteoSAFe/QueryMSV?id=MSV000100000"},
    {"cvLabel": "MS", "name": "Dataset FTP location",
     "value": "ftp://massive.ucsd.edu/v11/MSV000100000"}
  ],
  "contacts": [[
    {"cvLabel": "MS", "name": "contact name", "value": "Giovana Anceski Bataglion"},
    {"cvLabel": "MS", "name": "contact email", "value": "giovanabataglion@gmail.com"},
    {"cvLabel": "MS", "name": "contact affiliation", "value": "Federal University of Amazonas"}
  ]],
  "keywords": [
    {"cvLabel": "MS", "name": "submitter keyword", "value": "water"},
    {"cvLabel": "MS", "name": "submitter keyword", "value": "DatasetType:Other (environmental)"}
  ],
  "publications": [
    {"cvLabel": "MS", "name": "Dataset with no associated published manuscript"}
  ],
  "modifications": [
    {"cvLabel": "MS", "name": "No PTMs are included in the dataset", "accession": "MS:1002864"}
  ]
}
```

### A.2 Legacy JSON Endpoint Response (datasets_json.jsp)

```json
{
  "dataset": "MSV000100058",
  "datasetNum": "100058",
  "title": "A self-organizing single-cell morphology circuit optimizes Podophrya collini predatory trap structure",
  "user": "mazurkiewicz",
  "site": "massive.ucsd.edu",
  "flowname": "MASSIVE-COMPLETE",
  "createdMillis": "1764549972000",
  "created": "Nov. 30, 2025, 4:46 PM",
  "description": "Label-free quantification proteomics dataset associated with preprint...",
  "fileCount": "13",
  "fileSizeKB": "19872403",
  "spectra": "0",
  "psms": "315048",
  "peptides": "49376",
  "variants": "65555",
  "proteins": "5018",
  "species": "Podophrya collini",
  "instrument": "Orbitrap Exploris 480",
  "modification": "[M] [15.995] oxidation;[Protein N-term] [42.011] acetyl;...",
  "keywords": "cell biology;DatasetType:Proteomics",
  "pi": [
    {"name": "Amy Weeks", "email": "amweeks@wisc.edu",
     "institution": "University of Wisconsin-Madison", "country": "USA"}
  ],
  "complete": "true",
  "quant_analysis": "Quantification Results",
  "status": "Complete",
  "private": "false",
  "px": "PXD071400",
  "task": "634b29c2800d4074925bf3ac84077634"
}
```

---

**Document Version:** 1.0
**Author:** Claude Code (ultrathink mode)
**Review Status:** Draft for discussion
