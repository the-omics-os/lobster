"""
MetaboLights provider implementation for metabolomics dataset search and metadata extraction.

This provider implements search and metadata capabilities for MetaboLights
(https://www.ebi.ac.uk/metabolights), the leading public repository for
metabolomics studies hosted by EMBL-EBI. Supports MTBLS accession-based
discovery, keyword searching, and file listing.

MetaboLights REST API reference:
https://www.ebi.ac.uk/metabolights/ws/api-docs
"""

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from lobster.core.data_manager_v2 import DataManagerV2
from lobster.core.schemas.download_urls import DownloadFile, DownloadUrlResult
from lobster.tools.providers.base_provider import (
    BasePublicationProvider,
    DatasetType,
    ProviderCapability,
    PublicationMetadata,
    PublicationSource,
)
from lobster.utils.logger import get_logger
from lobster.utils.ssl_utils import create_ssl_context

logger = get_logger(__name__)

# Compiled regex for MTBLS accession validation (e.g., MTBLS1, MTBLS12345)
_MTBLS_PATTERN = re.compile(r"^MTBLS\d+$", re.IGNORECASE)


class MetaboLightsProviderConfig(BaseModel):
    """Configuration for MetaboLights provider."""

    # MetaboLights API settings
    base_url: str = "https://www.ebi.ac.uk/metabolights/ws"
    max_results: int = Field(default=100, ge=1, le=500)
    max_retry: int = Field(default=3, ge=1, le=10)
    sleep_time: float = Field(default=0.5, ge=0.1, le=5.0)
    request_timeout: int = Field(default=30, ge=5, le=120)

    # Result processing settings
    include_file_info: bool = True
    cache_results: bool = True


class MetaboLightsProvider(BasePublicationProvider):
    """
    MetaboLights provider for metabolomics dataset search and metadata extraction.

    Implements MetaboLights REST API for:
    - Study search by keywords
    - Study metadata retrieval
    - File listing for download URL extraction
    - Accession validation (MTBLS* pattern)
    """

    def __init__(
        self,
        data_manager: DataManagerV2,
        config: Optional[MetaboLightsProviderConfig] = None,
    ):
        """
        Initialize MetaboLights provider.

        Args:
            data_manager: DataManagerV2 instance for provenance tracking
            config: Optional configuration, uses defaults if not provided
        """
        self.data_manager = data_manager
        self.config = config or MetaboLightsProviderConfig()

        # Initialize cache directory
        self.cache_dir = self.data_manager.cache_dir / "metabolights"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Create SSL context for secure connections
        self.ssl_context = create_ssl_context()

        logger.debug(
            f"Initialized MetaboLights provider with base URL: {self.config.base_url}"
        )

    # =========================================================================
    # PROPERTIES
    # =========================================================================

    @property
    def source(self) -> PublicationSource:
        """Return MetaboLights as the publication source."""
        return PublicationSource.METABOLIGHTS

    @property
    def supported_dataset_types(self) -> List[DatasetType]:
        """Return list of dataset types supported by MetaboLights."""
        return [DatasetType.METABOLIGHTS]

    @property
    def priority(self) -> int:
        """
        Return provider priority for capability-based routing.

        MetaboLights has high priority (10) as the authoritative source
        for metabolomics datasets.

        Returns:
            int: Priority 10 (high priority)
        """
        return 10

    def get_supported_capabilities(self) -> Dict[str, bool]:
        """
        Return capabilities supported by MetaboLights provider.

        MetaboLights excels at metabolomics dataset discovery and metadata
        extraction.

        Returns:
            Dict[str, bool]: Capability support mapping
        """
        return {
            ProviderCapability.SEARCH_LITERATURE: False,
            ProviderCapability.DISCOVER_DATASETS: True,
            ProviderCapability.FIND_LINKED_DATASETS: False,
            ProviderCapability.EXTRACT_METADATA: True,
            ProviderCapability.VALIDATE_METADATA: False,
            ProviderCapability.QUERY_CAPABILITIES: True,
            ProviderCapability.GET_ABSTRACT: False,
            ProviderCapability.GET_FULL_CONTENT: False,
            ProviderCapability.EXTRACT_METHODS: False,
            ProviderCapability.EXTRACT_PDF: False,
            ProviderCapability.INTEGRATE_MULTI_OMICS: False,
        }

    def validate_identifier(self, identifier: str) -> bool:
        """
        Validate MetaboLights accession format (MTBLS followed by digits).

        Args:
            identifier: Potential MetaboLights accession (e.g., "MTBLS1234")

        Returns:
            bool: True if valid MTBLS accession
        """
        return bool(_MTBLS_PATTERN.match(identifier.strip()))

    # =========================================================================
    # CORE API METHODS
    # =========================================================================

    def search_publications(
        self,
        query: str,
        max_results: int = 20,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> str:
        """
        Search MetaboLights studies by keyword.

        Uses the MetaboLights search endpoint to find studies matching the
        given query. Results are formatted as a human-readable summary.

        Args:
            query: Search keyword (e.g., "diabetes", "lipidomics")
            max_results: Maximum number of results to return
            filters: Optional filters (currently unused, reserved for future use)
            **kwargs: Additional parameters

        Returns:
            str: Formatted search results
        """
        try:
            page_size = min(max_results, self.config.max_results)

            url = f"{self.config.base_url}/studies/search"
            params = {"query": query}

            logger.debug(f"Searching MetaboLights for: {query}")
            response_data = self._make_api_request(url, params)

            # The search endpoint returns a dict with study accessions or
            # a list of study objects depending on API version
            studies = self._parse_search_response(response_data)

            # Limit results
            studies = studies[:page_size]

            if not studies:
                return (
                    f"No MetaboLights studies found for query: '{query}'\n\n"
                    "Try different keywords or broader search terms."
                )

            result = (
                f"Found MetaboLights studies for query: '{query}'\n\n"
                f"Showing {len(studies)} studies:\n\n"
            )

            for i, study in enumerate(studies[:10], 1):
                accession = study.get("accession", study.get("id", "Unknown"))
                title = study.get("title", "No title available")
                organism = study.get("organism", "Unknown")
                status = study.get("status", "")

                result += f"{i}. **{accession}** - {title[:100]}\n"
                result += f"   Organism: {organism}\n"

                if status:
                    result += f"   Status: {status}\n"

                submission_date = study.get("submissionDate", "")
                if submission_date:
                    result += f"   Submitted: {submission_date}\n"

                result += f"   URL: https://www.ebi.ac.uk/metabolights/{accession}\n"
                result += "\n"

            if len(studies) > 10:
                result += f"... and {len(studies) - 10} more studies\n"

            return result

        except Exception as e:
            logger.error(f"Error searching MetaboLights studies: {e}")
            return f"Error searching MetaboLights: {str(e)}"

    def find_datasets_from_publication(
        self,
        identifier: str,
        dataset_types: Optional[List[DatasetType]] = None,
        **kwargs,
    ) -> str:
        """
        Find MetaboLights datasets linked to a publication.

        MetaboLights does not natively support direct publication-to-study
        linking like NCBI E-Link. This method searches for the identifier
        as a keyword and returns matching studies.

        Args:
            identifier: Publication identifier (PMID, DOI, or MTBLS accession)
            dataset_types: Expected to include DatasetType.METABOLIGHTS
            **kwargs: Additional parameters

        Returns:
            str: Formatted list of linked datasets
        """
        try:
            # If the identifier is already an MTBLS accession, fetch directly
            if self.validate_identifier(identifier):
                try:
                    study = self._get_study_metadata(identifier)
                    title = study.get("title", "No title")
                    return (
                        f"MetaboLights study found: {identifier}\n\n"
                        f"**{identifier}** - {title}\n"
                        f"URL: https://www.ebi.ac.uk/metabolights/{identifier}\n"
                    )
                except ValueError:
                    return f"MetaboLights study {identifier} not found."

            # Otherwise, search by keyword (DOI/PMID as search term)
            return self.search_publications(identifier, max_results=10)

        except Exception as e:
            logger.error(f"Error finding linked MetaboLights datasets: {e}")
            return f"Error finding linked datasets: {str(e)}"

    def extract_publication_metadata(
        self, identifier: str, **kwargs
    ) -> PublicationMetadata:
        """
        Extract metadata from a MetaboLights study.

        Maps MetaboLights study metadata to the standard PublicationMetadata
        format used across all providers.

        Args:
            identifier: MTBLS accession (e.g., "MTBLS1234")
            **kwargs: Additional parameters

        Returns:
            PublicationMetadata: Standardized metadata
        """
        try:
            study = self._get_study_metadata(identifier)

            # Extract publication references if available
            publications = study.get("publications", [])
            pmid = None
            doi = None
            journal = None

            if publications:
                pub = publications[0] if isinstance(publications, list) else {}
                if isinstance(pub, dict):
                    pmid = pub.get("pubmedId") or pub.get("pmid")
                    doi = pub.get("doi")
                    journal = pub.get("journal") or pub.get("publication")

            # Extract authors from contacts/submitters
            authors = self._extract_authors(study)

            # Extract keywords from study factors and descriptors
            keywords = self._extract_keywords(study)

            return PublicationMetadata(
                uid=identifier,
                title=study.get("title", ""),
                journal=journal,
                published=study.get("releaseDate") or study.get("submissionDate"),
                doi=doi,
                pmid=pmid,
                abstract=study.get("description", ""),
                authors=authors,
                keywords=keywords,
            )

        except Exception as e:
            logger.error(
                f"Error extracting MetaboLights metadata for {identifier}: {e}"
            )
            return PublicationMetadata(
                uid=identifier,
                title=f"MetaboLights Study {identifier}",
                authors=[],
            )

    # =========================================================================
    # METABOLIGHTS-SPECIFIC METHODS
    # =========================================================================

    def get_download_urls(self, accession: str) -> DownloadUrlResult:
        """
        Get download URLs for a MetaboLights study.

        Retrieves the file listing from the study and categorizes files into
        raw data, processed data, and metadata.

        Args:
            accession: MTBLS accession (e.g., "MTBLS1234")

        Returns:
            DownloadUrlResult with categorized file URLs:
              - raw_files: Raw instrument data (mzML, mzXML, nmrML, RAW)
              - processed_files: Processed data matrices, peak lists
              - metadata_files: ISA-Tab metadata files (i_*.txt, s_*.txt, a_*.txt, m_*.tsv)

        Example:
            >>> provider = MetaboLightsProvider(data_manager)
            >>> result = provider.get_download_urls("MTBLS1234")
            >>> print(len(result.raw_files))
            >>> print(len(result.processed_files))
        """
        if not self.validate_identifier(accession):
            return DownloadUrlResult(
                accession=accession,
                database="metabolights",
                error=f"Invalid MetaboLights accession format: {accession}",
            )

        try:
            files = self._get_study_files(accession)

            raw_files: List[DownloadFile] = []
            processed_files: List[DownloadFile] = []
            metadata_files: List[DownloadFile] = []

            ftp_base = f"ftp://ftp.ebi.ac.uk/pub/databases/metabolights/studies/public/{accession}"

            for file_info in files:
                filename = file_info.get("file", file_info.get("fileName", ""))
                if not filename:
                    continue

                # Build download URL
                file_url = f"{ftp_base}/{filename}"

                # Determine file size if available
                size_bytes = file_info.get("fileSize") or file_info.get("size")
                if isinstance(size_bytes, str):
                    try:
                        size_bytes = int(size_bytes)
                    except (ValueError, TypeError):
                        size_bytes = None

                file_type = self._classify_file(filename)

                download_file = DownloadFile(
                    url=file_url,
                    filename=filename,
                    size_bytes=size_bytes,
                    file_type=file_type,
                )

                if file_type == "raw":
                    raw_files.append(download_file)
                elif file_type == "metadata":
                    metadata_files.append(download_file)
                else:
                    processed_files.append(download_file)

            return DownloadUrlResult(
                accession=accession,
                database="metabolights",
                raw_files=raw_files,
                processed_files=processed_files,
                metadata_files=metadata_files,
                ftp_base=ftp_base,
                recommended_strategy="processed" if processed_files else "raw",
            )

        except Exception as e:
            logger.error(f"Error getting download URLs for {accession}: {e}")
            return DownloadUrlResult(
                accession=accession,
                database="metabolights",
                error=str(e),
            )

    # =========================================================================
    # PRIVATE HELPERS
    # =========================================================================

    def _get_study_metadata(self, accession: str) -> Dict[str, Any]:
        """
        Get comprehensive metadata for a MetaboLights study.

        Args:
            accession: MTBLS accession (e.g., "MTBLS1234")

        Returns:
            Dict containing study metadata

        Raises:
            ValueError: If accession not found or API error
        """
        if not self.validate_identifier(accession):
            raise ValueError(f"Invalid MetaboLights accession format: {accession}")

        try:
            url = f"{self.config.base_url}/studies/{accession}"
            study_data = self._make_api_request(url)

            logger.debug(f"Retrieved metadata for MetaboLights study {accession}")
            return study_data

        except Exception as e:
            logger.error(f"Error getting MetaboLights study metadata: {e}")
            raise ValueError(
                f"Failed to retrieve MetaboLights study {accession}: {str(e)}"
            )

    def _get_study_files(self, accession: str) -> List[Dict[str, Any]]:
        """
        Get file list for a MetaboLights study.

        Args:
            accession: MTBLS accession

        Returns:
            List of file dictionaries with name, size, type
        """
        try:
            url = f"{self.config.base_url}/studies/{accession}/files"
            params = {"include_sub_dir": "false"}

            response_data = self._make_api_request(url, params)

            # Response may be a list directly or a dict with a "study" key
            if isinstance(response_data, list):
                return response_data
            elif isinstance(response_data, dict):
                # Try common response wrapper keys
                return (
                    response_data.get("files", [])
                    or response_data.get("study", [])
                    or response_data.get("data", [])
                )
            else:
                logger.warning(
                    f"Unexpected file list response type: {type(response_data)}"
                )
                return []

        except Exception as e:
            logger.error(f"Error getting MetaboLights study files for {accession}: {e}")
            return []

    def _parse_search_response(self, response_data: Any) -> List[Dict[str, Any]]:
        """
        Parse search API response into a normalized list of study dicts.

        The MetaboLights search API may return different formats depending
        on the API version and query type. This method handles known
        response variations.

        Args:
            response_data: Raw JSON response from search endpoint

        Returns:
            List of study dictionaries
        """
        if isinstance(response_data, list):
            # Direct list of study objects or accession strings
            studies = []
            for item in response_data:
                if isinstance(item, dict):
                    studies.append(item)
                elif isinstance(item, str):
                    # Just an accession string; wrap it
                    studies.append({"accession": item})
            return studies

        if isinstance(response_data, dict):
            # Dict wrapper with various possible keys
            for key in ("content", "studies", "data", "_embedded"):
                if key in response_data:
                    inner = response_data[key]
                    if isinstance(inner, list):
                        return inner
                    elif isinstance(inner, dict) and "studies" in inner:
                        return inner["studies"]

            # If dict has accession-like keys, it might be a single study
            if "accession" in response_data:
                return [response_data]

        logger.warning(
            f"Could not parse MetaboLights search response: {type(response_data)}"
        )
        return []

    def _make_api_request(
        self, url: str, params: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Make API request to MetaboLights with retry logic.

        Args:
            url: API endpoint URL
            params: Query parameters

        Returns:
            Parsed JSON response (dict or list)

        Raises:
            ValueError: If request fails after retries
        """
        if params:
            query_string = urllib.parse.urlencode(params)
            full_url = f"{url}?{query_string}"
        else:
            full_url = url

        for attempt in range(self.config.max_retry):
            try:
                logger.debug(f"MetaboLights API request: {full_url}")

                request = urllib.request.Request(
                    full_url,
                    headers={
                        "Accept": "application/json",
                        "User-Agent": "LobsterAI/1.0 (https://lobsterbio.com)",
                    },
                )

                with urllib.request.urlopen(
                    request,
                    timeout=self.config.request_timeout,
                    context=self.ssl_context,
                ) as response:
                    data = response.read().decode("utf-8")
                    return json.loads(data)

            except urllib.error.HTTPError as e:
                if e.code == 404:
                    raise ValueError(f"Resource not found: {url}")
                elif e.code == 429:
                    # Rate limit - exponential backoff
                    wait_time = self.config.sleep_time * (2**attempt)
                    logger.warning(
                        f"Rate limit hit, waiting {wait_time:.1f}s before retry "
                        f"{attempt + 1}/{self.config.max_retry}"
                    )
                    time.sleep(wait_time)
                elif e.code >= 500:
                    # Server error - retry with backoff
                    wait_time = self.config.sleep_time * (attempt + 1)
                    logger.warning(
                        f"Server error {e.code}, retrying in {wait_time:.1f}s "
                        f"(attempt {attempt + 1}/{self.config.max_retry})"
                    )
                    time.sleep(wait_time)
                else:
                    logger.warning(
                        f"HTTP error {e.code}: {e.reason}, "
                        f"attempt {attempt + 1}/{self.config.max_retry}"
                    )
                    if attempt < self.config.max_retry - 1:
                        time.sleep(self.config.sleep_time * (attempt + 1))

            except urllib.error.URLError as e:
                logger.warning(
                    f"URL error: {e.reason}, "
                    f"attempt {attempt + 1}/{self.config.max_retry}"
                )
                if attempt < self.config.max_retry - 1:
                    time.sleep(self.config.sleep_time * (attempt + 1))

            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON response from MetaboLights API: {e}")
                if attempt < self.config.max_retry - 1:
                    time.sleep(self.config.sleep_time)
                else:
                    raise ValueError(
                        f"Invalid JSON response from MetaboLights API: {str(e)}"
                    )

            except Exception as e:
                logger.error(f"Unexpected error in MetaboLights API request: {e}")
                if attempt < self.config.max_retry - 1:
                    time.sleep(self.config.sleep_time)
                else:
                    raise

        raise ValueError(
            f"Failed to retrieve data from MetaboLights after "
            f"{self.config.max_retry} attempts"
        )

    @staticmethod
    def _classify_file(filename: str) -> str:
        """
        Classify a MetaboLights file into a category based on its name/extension.

        Categories:
        - "raw": Raw instrument data (mzML, mzXML, nmrML, .raw, .d, .wiff)
        - "metadata": ISA-Tab files (i_*.txt, s_*.txt, a_*.txt, m_*.tsv)
        - "processed": Everything else (peak lists, matrices, results)

        Args:
            filename: File name to classify

        Returns:
            str: File category ("raw", "metadata", or "processed")
        """
        lower = filename.lower()

        # ISA-Tab metadata files
        if lower.startswith(("i_", "s_", "a_", "m_")) and lower.endswith(
            (".txt", ".tsv")
        ):
            return "metadata"

        # Raw instrument data extensions
        raw_extensions = (
            ".mzml",
            ".mzxml",
            ".nmrml",
            ".raw",
            ".wiff",
            ".wiff2",
            ".d",
            ".cdf",
            ".fid",
            ".ser",
            ".baf",
        )
        if any(lower.endswith(ext) for ext in raw_extensions):
            return "raw"

        # Compressed raw data
        if any(lower.endswith(ext + ".gz") for ext in raw_extensions):
            return "raw"

        return "processed"

    @staticmethod
    def _extract_authors(study: Dict[str, Any]) -> List[str]:
        """
        Extract author/contact names from study metadata.

        MetaboLights stores contacts in the "people" or "contacts" field
        of the investigation.

        Args:
            study: Study metadata dictionary

        Returns:
            List of author name strings
        """
        authors = []

        # Try "people" field (ISA-Tab investigation contacts)
        contacts = study.get("people", study.get("contacts", []))
        if isinstance(contacts, list):
            for contact in contacts:
                if isinstance(contact, dict):
                    first = contact.get("firstName", "")
                    mid = contact.get("midInitials", "")
                    last = contact.get("lastName", "")
                    parts = [p for p in [first, mid, last] if p]
                    name = " ".join(parts).strip()
                    if name and name not in authors:
                        authors.append(name)
                elif isinstance(contact, str):
                    if contact.strip() and contact.strip() not in authors:
                        authors.append(contact.strip())

        # Fall back to submitters if no contacts
        if not authors:
            submitters = study.get("submitters", [])
            if isinstance(submitters, list):
                for sub in submitters:
                    if isinstance(sub, dict):
                        name = f"{sub.get('firstName', '')} {sub.get('lastName', '')}".strip()
                        if name:
                            authors.append(name)
                    elif isinstance(sub, str) and sub.strip():
                        authors.append(sub.strip())

        return authors

    @staticmethod
    def _extract_keywords(study: Dict[str, Any]) -> List[str]:
        """
        Extract keywords from study factors, descriptors, and technology.

        Args:
            study: Study metadata dictionary

        Returns:
            List of keyword strings
        """
        keywords = []

        # Study factors (e.g., "diet", "genotype")
        factors = study.get("factors", study.get("studyFactors", []))
        if isinstance(factors, list):
            for factor in factors:
                if isinstance(factor, dict):
                    name = factor.get("name", factor.get("factorName", ""))
                    if name and name not in keywords:
                        keywords.append(name)
                elif isinstance(factor, str) and factor not in keywords:
                    keywords.append(factor)

        # Study design descriptors
        descriptors = study.get("descriptors", study.get("studyDesignDescriptors", []))
        if isinstance(descriptors, list):
            for desc in descriptors:
                if isinstance(desc, dict):
                    term = desc.get("description", desc.get("annotationValue", ""))
                    if term and term not in keywords:
                        keywords.append(term)
                elif isinstance(desc, str) and desc not in keywords:
                    keywords.append(desc)

        # Organism
        organism = study.get("organism", "")
        if isinstance(organism, str) and organism and organism not in keywords:
            keywords.append(organism)
        elif isinstance(organism, list):
            for org in organism:
                org_str = org if isinstance(org, str) else str(org)
                if org_str and org_str not in keywords:
                    keywords.append(org_str)

        return keywords
