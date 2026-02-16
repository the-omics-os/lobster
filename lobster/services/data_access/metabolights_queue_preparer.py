"""
MetaboLights Queue Preparer -- prepares download queue entries for MetaboLights (MTBLS*) datasets.

Uses heuristic strategy recommendation (no LLM needed). Strategy priority:
MAF files (processed) -> mzML -> raw vendor files.

MetaboLights is the EBI repository for metabolomics datasets. Studies contain
ISA-Tab metadata (Investigation, Study, Assay files), Metabolite Assignment Files
(MAF, tab-separated intensity matrices), and raw instrument data (mzML, vendor formats).

API reference: https://www.ebi.ac.uk/metabolights/ws/
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import requests

from lobster.core.interfaces.queue_preparer import IQueuePreparer
from lobster.utils.logger import get_logger

if TYPE_CHECKING:
    from lobster.core.schemas.download_queue import StrategyConfig
    from lobster.core.schemas.download_urls import DownloadUrlResult

logger = get_logger(__name__)

# MetaboLights API base URL
METABOLIGHTS_API_BASE = "https://www.ebi.ac.uk/metabolights/ws/studies"

# MTBLS accession pattern
MTBLS_PATTERN = re.compile(r"^MTBLS\d+$", re.IGNORECASE)

# File extension classification
MAF_EXTENSIONS = (".tsv", ".txt")
MZML_EXTENSIONS = (".mzml", ".mzml.gz")
RAW_EXTENSIONS = (".raw", ".wiff", ".wiff2", ".d", ".mzxml")


class MetaboLightsQueuePreparer(IQueuePreparer):
    """
    Prepares download queue entries for MetaboLights metabolomics datasets.

    MetaboLights datasets follow the ISA-Tab standard with a distinct file
    hierarchy: investigation metadata, study/sample descriptions, assay
    definitions, MAF intensity matrices, and raw spectral data. Strategy
    is determined by which file types are available in the study.
    """

    _session = None

    def supported_databases(self) -> List[str]:
        return ["metabolights"]

    def _get_session(self) -> requests.Session:
        """
        Get or create a requests session for MetaboLights API communication.

        Returns:
            requests.Session: Configured HTTP session
        """
        if self._session is None:
            self._session = requests.Session()
            self._session.headers.update(
                {
                    "Accept": "application/json",
                    "User-Agent": "Lobster-AI/1.0 (MetaboLights Queue Preparer)",
                }
            )
            logger.debug("Created HTTP session for MetaboLightsQueuePreparer")
        return self._session

    def fetch_metadata(
        self, accession: str
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        """
        Fetch MetaboLights study metadata via REST API.

        Retrieves study-level metadata including title, description,
        organism information, and submission details.

        Args:
            accession: MetaboLights accession (e.g., "MTBLS123")

        Returns:
            Tuple of (metadata dict, validation_info or None)

        Raises:
            RuntimeError: If API request fails
        """
        accession = accession.upper()
        if not MTBLS_PATTERN.match(accession):
            raise ValueError(
                f"Invalid MetaboLights accession format: '{accession}'. "
                f"Expected pattern: MTBLS followed by digits (e.g., MTBLS123)"
            )

        session = self._get_session()
        url = f"{METABOLIGHTS_API_BASE}/{accession}"
        logger.info(f"Fetching MetaboLights metadata for {accession}")

        try:
            response = session.get(url, timeout=60)
            response.raise_for_status()
            data = response.json()

            # Normalize metadata from the API response
            metadata = {}

            # Handle nested study object or flat response
            study_data = data.get("content", data)
            if isinstance(study_data, dict):
                metadata["title"] = study_data.get("title", "")
                metadata["description"] = study_data.get("description", "")
                metadata["study_status"] = study_data.get(
                    "studyStatus", study_data.get("status", "")
                )
                metadata["submission_date"] = study_data.get(
                    "submissionDate", ""
                )
                metadata["release_date"] = study_data.get(
                    "publicReleaseDate", study_data.get("releaseDate", "")
                )

                # Organism info
                organisms = study_data.get("organism", [])
                if isinstance(organisms, list):
                    metadata["organisms"] = organisms
                    metadata["_organism_names"] = [
                        (
                            org.get("organismName", str(org))
                            if isinstance(org, dict)
                            else str(org)
                        )
                        for org in organisms
                    ]
                elif isinstance(organisms, str):
                    metadata["organisms"] = [organisms]
                    metadata["_organism_names"] = [organisms]

                # Study design
                metadata["study_design_descriptors"] = study_data.get(
                    "studyDesignDescriptors", []
                )

                # Protocols
                protocols = study_data.get("protocols", [])
                if protocols:
                    metadata["protocol_count"] = len(protocols)
                    metadata["protocol_names"] = [
                        p.get("name", "") for p in protocols if isinstance(p, dict)
                    ]

                # Assays
                assays = study_data.get("assays", [])
                if assays:
                    metadata["assay_count"] = len(assays)
                    metadata["assay_technologies"] = [
                        a.get("technology", "")
                        for a in assays
                        if isinstance(a, dict)
                    ]
                    metadata["assay_platforms"] = [
                        a.get("platform", "")
                        for a in assays
                        if isinstance(a, dict)
                    ]

                # Contacts
                contacts = study_data.get("people", study_data.get("contacts", []))
                if contacts:
                    metadata["contact_count"] = len(contacts)

                # Publications
                publications = study_data.get("publications", [])
                if publications:
                    metadata["publication_count"] = len(publications)
                    metadata["publication_dois"] = [
                        p.get("doi", "")
                        for p in publications
                        if isinstance(p, dict) and p.get("doi")
                    ]

            return metadata, None

        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                raise RuntimeError(
                    f"MetaboLights study {accession} not found (404)"
                )
            raise RuntimeError(
                f"MetaboLights API error for {accession}: {e}"
            )
        except requests.exceptions.RequestException as e:
            raise RuntimeError(
                f"Network error fetching MetaboLights metadata for "
                f"{accession}: {e}"
            )

    def extract_download_urls(self, accession: str) -> "DownloadUrlResult":
        """
        Extract download URLs for a MetaboLights study.

        Fetches the file list from MetaboLights API and categorizes files into
        primary (MAF), raw (mzML/vendor), processed, supplementary, and metadata.

        Args:
            accession: MetaboLights accession (e.g., "MTBLS123")

        Returns:
            DownloadUrlResult with categorized download URLs
        """
        from lobster.core.schemas.download_urls import DownloadFile, DownloadUrlResult

        accession = accession.upper()
        session = self._get_session()
        url = f"{METABOLIGHTS_API_BASE}/{accession}/files"
        logger.info(f"Fetching file list for {accession}")

        try:
            response = session.get(url, timeout=60)
            response.raise_for_status()
            data = response.json()

            # Parse file list from API response
            file_list = []
            if isinstance(data, dict):
                file_list = (
                    data.get("study", [])
                    or data.get("files", [])
                    or data.get("data", [])
                )
                if isinstance(file_list, dict):
                    file_list = file_list.get("files", [])
            elif isinstance(data, list):
                file_list = data

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch file list for {accession}: {e}")
            return DownloadUrlResult(
                accession=accession,
                database="metabolights",
                error=f"Failed to fetch file list: {e}",
            )

        # Categorize files
        primary_files = []  # MAF files (processed intensity matrices)
        raw_files = []  # mzML and vendor raw files
        metadata_files = []  # ISA-Tab metadata files
        supplementary_files = []  # Other files
        total_size = 0

        for f in file_list:
            if isinstance(f, dict):
                filename = f.get("file", f.get("filename", ""))
                file_size = f.get("fileSize", f.get("size"))
                directory = f.get("directory", f.get("relativePath", ""))
            elif isinstance(f, str):
                filename = f
                file_size = None
                directory = ""
            else:
                continue

            if not filename:
                continue

            # Build download URL
            if directory:
                download_url = (
                    f"https://www.ebi.ac.uk/metabolights/{accession}/files/"
                    f"{directory}/{filename}"
                )
            else:
                download_url = (
                    f"https://www.ebi.ac.uk/metabolights/{accession}/files/"
                    f"{filename}"
                )

            size_bytes = int(file_size) if file_size else None
            if size_bytes:
                total_size += size_bytes

            download_file = DownloadFile(
                url=download_url,
                filename=filename,
                size_bytes=size_bytes,
                file_type=self._classify_file(filename),
            )

            # Classify into categories
            file_type = self._classify_file(filename)
            if file_type == "maf":
                primary_files.append(download_file)
            elif file_type in ("mzml", "raw"):
                raw_files.append(download_file)
            elif file_type in ("investigation", "assay", "sample_metadata"):
                metadata_files.append(download_file)
            else:
                supplementary_files.append(download_file)

        # Determine FTP base
        ftp_base = (
            f"ftp://ftp.ebi.ac.uk/pub/databases/metabolights/"
            f"studies/public/{accession}"
        )

        return DownloadUrlResult(
            accession=accession,
            database="metabolights",
            primary_files=primary_files,
            raw_files=raw_files,
            processed_files=[],  # MAF files are in primary_files
            supplementary_files=supplementary_files,
            metadata_files=metadata_files,
            ftp_base=ftp_base,
            total_size_bytes=total_size if total_size > 0 else None,
            recommended_strategy="maf" if primary_files else "raw",
        )

    def recommend_strategy(
        self,
        metadata: Dict[str, Any],
        url_data: "DownloadUrlResult",
        accession: str,
    ) -> "StrategyConfig":
        """
        Recommend MetaboLights download strategy based on available file types.

        Priority: MAF_FIRST > MZML_FIRST > RAW_FIRST.

        MAF files (Metabolite Assignment Files) are strongly preferred because
        they contain the processed intensity matrix ready for downstream analysis.
        mzML and vendor raw files require additional peak detection and alignment.

        Args:
            metadata: Study metadata from fetch_metadata()
            url_data: Download URLs from extract_download_urls()
            accession: MTBLS accession

        Returns:
            StrategyConfig with recommended strategy, confidence, and rationale
        """
        from lobster.core.schemas.download_queue import StrategyConfig

        # Check for MAF files (best case: processed intensity matrices)
        if url_data.primary_files:
            strategy_name = "MAF_FIRST"
            confidence = 0.90
            maf_count = len(url_data.primary_files)
            rationale = (
                f"Metabolite Assignment Files available ({maf_count} MAF "
                f"file{'s' if maf_count > 1 else ''}) with processed "
                f"intensity data"
            )
        # Check for mzML files
        elif url_data.raw_files and any(
            f.filename.lower().endswith(MZML_EXTENSIONS)
            for f in url_data.raw_files
        ):
            strategy_name = "MZML_FIRST"
            confidence = 0.75
            mzml_count = sum(
                1
                for f in url_data.raw_files
                if f.filename.lower().endswith(MZML_EXTENSIONS)
            )
            rationale = (
                f"mzML spectral files available ({mzml_count} files). "
                f"Requires peak detection and alignment before analysis."
            )
        # Check for vendor raw files
        elif url_data.raw_files:
            strategy_name = "RAW_FIRST"
            confidence = 0.55
            raw_count = len(url_data.raw_files)
            rationale = (
                f"Only vendor raw instrument files available ({raw_count} "
                f"files). Requires vendor-specific conversion and processing."
            )
        else:
            strategy_name = "MAF_FIRST"
            confidence = 0.30
            rationale = (
                "No downloadable data files detected. Defaulting to "
                "MAF_FIRST strategy for retry."
            )

        # Estimate timeout based on total file count and size
        total_files = url_data.file_count
        if total_files > 100:
            timeout = 7200  # 2 hours
            max_retries = 5
        elif total_files > 20:
            timeout = 3600  # 1 hour
            max_retries = 3
        else:
            timeout = 1800  # 30 minutes
            max_retries = 3

        return StrategyConfig(
            strategy_name=strategy_name,
            concatenation_strategy="auto",
            confidence=confidence,
            rationale=rationale,
            strategy_params={
                "file_type_priority": strategy_name.replace("_FIRST", "")
                .lower(),
                "include_metadata": True,
            },
            execution_params={
                "timeout": timeout,
                "max_retries": max_retries,
                "verify_checksum": False,  # MetaboLights does not always provide checksums
                "resume_enabled": False,
            },
        )

    def get_available_strategies(self) -> List[str]:
        """
        Get list of MetaboLights download strategies.

        Returns:
            List of strategy names in order of preference:
            - MAF_FIRST: Metabolite Assignment Files (processed, ready to use)
            - MZML_FIRST: mzML spectral data (requires peak detection)
            - RAW_FIRST: Vendor raw files (requires conversion + processing)
        """
        return ["MAF_FIRST", "MZML_FIRST", "RAW_FIRST"]

    def validate_entry(self, accession: str) -> bool:
        """
        Validate that a MetaboLights study exists and is accessible.

        Makes a lightweight HEAD request to the MetaboLights API to check
        study accessibility without downloading full metadata.

        Args:
            accession: MTBLS accession to validate

        Returns:
            bool: True if study is accessible, False otherwise
        """
        accession = accession.upper()

        # Validate format first
        if not MTBLS_PATTERN.match(accession):
            logger.warning(
                f"Invalid MetaboLights accession format: '{accession}'"
            )
            return False

        session = self._get_session()
        url = f"{METABOLIGHTS_API_BASE}/{accession}"

        try:
            response = session.head(url, timeout=30)
            if response.status_code == 200:
                logger.info(f"MetaboLights study {accession} is accessible")
                return True
            elif response.status_code == 404:
                logger.warning(f"MetaboLights study {accession} not found")
                return False
            else:
                # Try GET as fallback (some servers don't support HEAD)
                response = session.get(url, timeout=30)
                return response.status_code == 200
        except requests.exceptions.RequestException as e:
            logger.warning(
                f"Could not validate MetaboLights study {accession}: {e}"
            )
            return False

    # =========================================================================
    # HELPERS
    # =========================================================================

    @staticmethod
    def _classify_file(filename: str) -> str:
        """
        Classify a file by its name and extension into a category.

        MetaboLights files follow the ISA-Tab naming convention:
        - i_*.txt: Investigation files
        - s_*.txt: Study/sample description files
        - a_*.txt: Assay description files
        - m_*.tsv: Metabolite Assignment Files (MAF)

        Args:
            filename: Name of the file

        Returns:
            File category: "maf", "mzml", "raw", "investigation",
                          "assay", "sample_metadata", or "other"
        """
        lower = filename.lower()

        # MAF files (Metabolite Assignment Files)
        if lower.startswith("m_") and lower.endswith(MAF_EXTENSIONS):
            return "maf"

        # Sample metadata files
        if lower.startswith("s_") and lower.endswith(MAF_EXTENSIONS):
            return "sample_metadata"

        # Investigation files (ISA-Tab)
        if lower.startswith("i_") and lower.endswith(MAF_EXTENSIONS):
            return "investigation"

        # Assay files
        if lower.startswith("a_") and lower.endswith(MAF_EXTENSIONS):
            return "assay"

        # mzML files
        if lower.endswith(MZML_EXTENSIONS):
            return "mzml"

        # Vendor raw files
        if lower.endswith(RAW_EXTENSIONS):
            return "raw"

        return "other"
