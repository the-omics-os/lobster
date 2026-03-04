"""
GEO metadata fetching, extraction, and validation.

Extracted from geo_service.py as part of Phase 4 GEO Service Decomposition.
Contains 12 methods that handle metadata retrieval from GEO via GEOparse
and NCBI Entrez, metadata extraction from GSE objects, platform validation,
and sample type detection.
"""

import json
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

try:
    import GEOparse
except ImportError:
    GEOparse = None

from lobster.core.exceptions import (
    FeatureNotImplementedError,
    UnsupportedPlatformError,
)
from lobster.services.data_access.geo.constants import (
    PLATFORM_REGISTRY,
    SUPPORTED_KEYWORDS,
    UNSUPPORTED_KEYWORDS,
    PlatformCompatibility,
)
from lobster.services.data_access.geo.helpers import (
    RetryOutcome,
    RetryResult,
)
from lobster.utils.logger import get_logger
from lobster.utils.ssl_utils import create_ssl_context, handle_ssl_error

logger = get_logger(__name__)


class MetadataFetcher:
    """GEO metadata fetching, extraction, and validation.

    Handles all metadata-related operations for GEO datasets including:
    - GSE metadata fetching via GEOparse with SOFT pre-download
    - GDS-to-GSE conversion via NCBI E-utilities
    - Entrez fallback for missing SOFT files
    - Metadata extraction from GEOparse GSE objects
    - Platform compatibility validation
    - Sample type detection (RNA, protein, VDJ, ATAC)
    """

    def __init__(self, service):
        """Initialize with reference to parent GEOService.

        Args:
            service: Parent GEOService instance providing shared state
                     (data_manager, cache_dir, console, geo_downloader, etc.)
        """
        self.service = service

    def fetch_metadata_only(self, geo_id: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Fetch and validate GEO metadata with fallback mechanisms (Scenario 1).

        This function handles both GSE and GDS identifiers, converting GDS to GSE
        when needed, and stores the metadata in data_manager for user review.

        Args:
            geo_id: GEO accession ID (e.g., GSE194247 or GDS5826)

        Returns:
            Tuple[Dict, Dict]: metadata and validation_result
        """
        try:
            logger.info(f"Fetching metadata for GEO ID: {geo_id}")

            # Clean the GEO ID
            clean_geo_id = geo_id.strip().upper()

            # Check if it's a GDS identifier
            if clean_geo_id.startswith("GDS"):
                logger.info(f"Detected GDS identifier: {clean_geo_id}")
                return self._fetch_gds_metadata_and_convert(clean_geo_id)
            elif not clean_geo_id.startswith("GSE"):
                logger.error(
                    f"Invalid GEO ID format: {geo_id}. Must be a GSE or GDS accession (e.g., GSE194247 or GDS5826)."
                )
                return (None, None)

            # Handle GSE identifiers (existing logic)
            return self._fetch_gse_metadata(clean_geo_id)

        except UnsupportedPlatformError:
            # Re-raise platform errors - they should be handled by caller
            raise
        except FeatureNotImplementedError:
            # Re-raise modality errors - they should be handled by caller
            raise
        except Exception as e:
            logger.exception(f"Error fetching metadata for {geo_id}: {e}")
            return (None, None)

    def _fetch_gse_metadata(self, gse_id: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Fetch GSE metadata using GEOparse with retry logic.

        Args:
            gse_id: GSE accession ID

        Returns:
            Tuple[Dict, Dict]: metadata and validation_result
        """
        try:
            logger.debug(f"Downloading SOFT metadata for {gse_id} using GEOparse...")

            # PRE-DOWNLOAD SOFT FILE USING HTTPS TO BYPASS GEOparse's FTP DOWNLOADER
            # GEOparse internally uses FTP which lacks error detection and causes corruption.
            # By pre-downloading with HTTPS, GEOparse finds existing file and skips its FTP download.
            soft_file_path = Path(self.service.cache_dir) / f"{gse_id}_family.soft.gz"
            if not soft_file_path.exists():
                # Construct SOFT file URL components
                gse_num_str = gse_id[3:]  # Remove 'GSE' prefix
                if len(gse_num_str) >= 3:
                    series_folder = f"GSE{gse_num_str[:-3]}nnn"
                else:
                    series_folder = "GSEnnn"

                soft_url = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{series_folder}/{gse_id}/soft/{gse_id}_family.soft.gz"

                logger.debug(f"Pre-downloading SOFT file using HTTPS: {soft_url}")
                try:
                    ssl_context = create_ssl_context()
                    with urllib.request.urlopen(
                        soft_url, context=ssl_context
                    ) as response:
                        with open(soft_file_path, "wb") as f:
                            f.write(response.read())
                    logger.debug(
                        f"Successfully pre-downloaded SOFT file to {soft_file_path}"
                    )
                except Exception as e:
                    error_str = str(e)
                    if "CERTIFICATE_VERIFY_FAILED" in error_str or "SSL" in error_str:
                        handle_ssl_error(e, soft_url, logger)
                        raise Exception(
                            "SSL certificate verification failed when downloading SOFT file. "
                            "See error message above for solutions."
                        )
                    # If pre-download fails, let GEOparse try (will use FTP as fallback)
                    logger.warning(
                        f"Pre-download failed: {e}. GEOparse will attempt download."
                    )

            # Wrap GEOparse call with retry logic for transient network failures
            # Note: GEOparse will find our pre-downloaded SOFT file and skip its FTP download
            result = self.service._retry_with_backoff(
                operation=lambda: GEOparse.get_GEO(
                    geo=gse_id, destdir=str(self.service.cache_dir)
                ),
                operation_name=f"Fetch metadata for {gse_id}",
                max_retries=5,
                is_ftp=False,
            )

            # Check if SOFT file was missing (typed retry result)
            if result.needs_fallback:
                logger.info(
                    f"SOFT file unavailable for {gse_id}, attempting Entrez fallback..."
                )
                try:
                    return self._fetch_gse_metadata_via_entrez(gse_id)
                except Exception as e:
                    logger.error(f"Entrez fallback also failed for {gse_id}: {e}")
                    logger.error(
                        f"Failed to fetch metadata for {gse_id}: "
                        f"SOFT file missing and Entrez fallback failed ({str(e)}). "
                        f"Please check GEO database status or try again later."
                    )
                    return (None, None)

            if not result.succeeded:
                logger.error(
                    f"Failed to fetch metadata for {gse_id} after multiple retry attempts."
                )
                return (None, None)

            gse = result.value
            metadata = self._extract_metadata(gse)
            logger.debug(f"Successfully extracted metadata using GEOparse for {gse_id}")

            if not metadata:
                logger.error(f"No metadata could be extracted for {gse_id}")
                return (None, None)

            # Validate metadata against transcriptomics schema
            validation_result = self._validate_geo_metadata(metadata)

            # Check platform compatibility BEFORE downloading files (Phase 2: Early Validation)
            try:
                is_compatible, compat_message = self._check_platform_compatibility(
                    gse_id, metadata
                )
                logger.info(f"Platform validation for {gse_id}: {compat_message}")
            except UnsupportedPlatformError as e:
                # CHANGE 3: Use helper method for consistent structure
                self.service.data_manager._store_geo_metadata(
                    geo_id=gse_id,
                    metadata=metadata,
                    stored_by="_fetch_gse_metadata (exception)",
                )
                # Enrich with error details via centralized helper
                self.service.data_manager._enrich_geo_metadata(
                    gse_id,
                    validation_result=validation_result,
                    platform_error=str(e),
                    platform_details=e.details,
                    status="unsupported_platform",
                    error_timestamp=datetime.now().isoformat(),
                )
                logger.error(
                    f"Platform validation failed for {gse_id}: {e.details['detected_platforms']}"
                )
                raise

            return metadata, validation_result

        except UnsupportedPlatformError:
            # Re-raise platform errors without catching them
            raise
        except FeatureNotImplementedError:
            # Re-raise modality errors without catching them
            raise
        except Exception as geoparse_error:
            logger.error(f"GEOparse metadata fetch failed: {geoparse_error}")
            logger.error(
                f"Failed to fetch metadata for {gse_id}. GEOparse ({geoparse_error}) failed."
            )
            return (None, None)

    def _fetch_gds_metadata_and_convert(
        self, gds_id: str
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Fetch GDS metadata using NCBI E-utilities and convert to GSE for downstream processing.

        Args:
            gds_id: GDS accession ID (e.g., GDS5826)

        Returns:
            Tuple[Dict, Dict]: Combined metadata and validation_result
        """
        try:
            logger.info(f"Fetching GDS metadata for {gds_id} using NCBI E-utilities...")

            # Extract GDS number from ID
            gds_number = gds_id.replace("GDS", "")

            # Build NCBI E-utilities URL for GDS metadata
            base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
            params = {"db": "gds", "id": gds_number, "retmode": "json"}

            # Construct URL with parameters
            url_params = urllib.parse.urlencode(params)
            url = f"{base_url}?{url_params}"

            logger.debug(f"Fetching GDS metadata from: {url}")

            # Create SSL context for secure connection
            ssl_context = create_ssl_context()

            # Make the request with SSL support
            try:
                response = urllib.request.urlopen(url, context=ssl_context, timeout=30)
                response_data = response.read().decode("utf-8")
            except Exception as e:
                error_str = str(e)
                if "CERTIFICATE_VERIFY_FAILED" in error_str or "SSL" in error_str:
                    handle_ssl_error(e, url, logger)
                    raise Exception(
                        "SSL certificate verification failed when fetching GDS metadata. "
                        "See error message above for solutions."
                    )
                raise

            # Parse JSON response
            gds_data = json.loads(response_data)

            # Extract the GDS record
            if "result" not in gds_data or gds_number not in gds_data["result"]:
                logger.error(f"No GDS record found for {gds_id}")
                return (None, None)

            gds_record = gds_data["result"][gds_number]
            logger.debug(f"Successfully retrieved GDS metadata for {gds_id}")

            # Extract GSE ID from GDS record
            gse_id = gds_record.get("gse", "")
            if not gse_id:
                logger.error(f"No associated GSE found for GDS {gds_id}")
                return (None, None)

            # Ensure GSE has proper format
            if not gse_id.startswith("GSE"):
                gse_id = f"GSE{gse_id}"

            logger.info(f"Found associated GSE: {gse_id} for GDS {gds_id}")

            # Fetch the GSE metadata using existing method
            result = self._fetch_gse_metadata(gse_id)

            # Check if metadata fetch failed (returns (None, None) on error)
            if result[0] is None:
                logger.error(
                    f"Failed to fetch GSE metadata for {gse_id} (from GDS {gds_id})"
                )
                return (None, None)

            gse_metadata, validation_result = result

            # Enhance metadata with GDS information
            enhanced_metadata = self._combine_gds_gse_metadata(
                gds_record, gse_metadata, gds_id, gse_id
            )

            return enhanced_metadata, validation_result

        except UnsupportedPlatformError:
            # Re-raise platform errors - they should be handled by caller
            raise
        except urllib.error.URLError as e:
            logger.error(f"Network error fetching GDS metadata: {e}")
            logger.error(f"Network error fetching GDS metadata for {gds_id}: {str(e)}")
            return (None, None)
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing GDS JSON response: {e}")
            logger.error(f"Error parsing GDS metadata response for {gds_id}: {str(e)}")
            return (None, None)
        except Exception as e:
            logger.error(f"Error fetching GDS metadata for {gds_id}: {e}")
            logger.error(f"Error fetching GDS metadata for {gds_id}: {str(e)}")
            return (None, None)

    def _combine_gds_gse_metadata(
        self,
        gds_record: Dict[str, Any],
        gse_metadata: Dict[str, Any],
        gds_id: str,
        gse_id: str,
    ) -> Dict[str, Any]:
        """
        Combine GDS and GSE metadata into a unified metadata structure.

        Args:
            gds_record: GDS record from NCBI E-utilities
            gse_metadata: GSE metadata from GEOparse
            gds_id: Original GDS identifier
            gse_id: Associated GSE identifier

        Returns:
            Dict: Combined metadata with both GDS and GSE information
        """
        try:
            # Start with GSE metadata as base
            combined_metadata = gse_metadata.copy()

            # Add GDS-specific information
            combined_metadata["gds_info"] = {
                "gds_id": gds_id,
                "gds_title": gds_record.get("title", ""),
                "gds_summary": gds_record.get("summary", ""),
                "gds_type": gds_record.get("gdstype", ""),
                "platform_technology": gds_record.get("ptechtype", ""),
                "value_type": gds_record.get("valtype", ""),
                "sample_info": gds_record.get("ssinfo", ""),
                "subset_info": gds_record.get("subsetinfo", ""),
                "n_samples": gds_record.get("n_samples", 0),
                "platform_taxa": gds_record.get("platformtaxa", ""),
                "samples_taxa": gds_record.get("samplestaxa", ""),
                "ftp_link": gds_record.get("ftplink", ""),
                "associated_gse": gse_id,
            }

            # Update title and summary to include GDS information if different
            gds_title = gds_record.get("title", "")
            if gds_title and gds_title != combined_metadata.get("title", ""):
                combined_metadata["title"] = (
                    f"{gds_title} (GDS: {gds_id}, GSE: {gse_id})"
                )
            else:
                combined_metadata["title"] = (
                    f"{combined_metadata.get('title', '')} (GDS: {gds_id}, GSE: {gse_id})"
                )

            # Add cross-reference information
            combined_metadata["cross_references"] = {
                "original_request": gds_id,
                "gds_accession": gds_id,
                "gse_accession": gse_id,
                "data_source": "GDS_to_GSE_conversion",
            }

            logger.debug(
                f"Successfully combined GDS and GSE metadata for {gds_id} -> {gse_id}"
            )
            return combined_metadata

        except Exception as e:
            logger.error(f"Error combining GDS and GSE metadata: {e}")
            # Return GSE metadata if combination fails
            return gse_metadata

    def _fetch_gse_metadata_via_entrez(
        self, gse_id: str
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Fetch GSE metadata using NCBI Entrez E-utilities (fallback for missing SOFT files).

        This method provides basic metadata (title, summary, organism, platform, sample count)
        when SOFT files are unavailable on the FTP server. Sample-level characteristics
        are NOT available via this fallback method.

        Uses the same Entrez esummary API as GDS fetching, but directly for GSE accessions.

        Args:
            gse_id: GSE accession ID (e.g., GSE233321)

        Returns:
            Tuple[Dict, Dict]: metadata and validation_result

        Raises:
            urllib.error.URLError: Network connection errors
            json.JSONDecodeError: Invalid JSON response
            Exception: Other unexpected errors

        Note:
            Entrez metadata is less complete than SOFT files. Missing:
            - Detailed sample-level characteristics (e.g., treatment groups)
            - Protocol details
            - Some contact information
        """
        try:
            logger.info(
                f"Fetching GSE metadata via Entrez fallback for {gse_id} "
                "(SOFT file unavailable)"
            )

            # Extract GSE number from ID
            gse_number = gse_id.replace("GSE", "")

            # Build NCBI E-utilities URL (same pattern as GDS fetching)
            base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
            params = {"db": "gds", "id": gse_number, "retmode": "json"}

            # Construct URL with parameters
            url_params = urllib.parse.urlencode(params)
            url = f"{base_url}?{url_params}"

            logger.debug(f"Fetching GSE metadata from Entrez: {url}")

            # Create SSL context for secure connection
            ssl_context = create_ssl_context()

            # Make the request with SSL support (timeout: 30s)
            try:
                response = urllib.request.urlopen(url, context=ssl_context, timeout=30)
                response_data = response.read().decode("utf-8")
            except Exception as e:
                error_str = str(e)
                if "CERTIFICATE_VERIFY_FAILED" in error_str or "SSL" in error_str:
                    handle_ssl_error(e, url, logger)
                    raise Exception(
                        "SSL certificate verification failed when fetching GSE metadata via Entrez. "
                        "See error message above for solutions."
                    )
                raise

            # Parse JSON response
            entrez_data = json.loads(response_data)

            # Extract the GSE record
            if "result" not in entrez_data or gse_number not in entrez_data["result"]:
                logger.error(f"No Entrez record found for {gse_id}")
                raise ValueError(f"No Entrez record found for {gse_id}")

            gse_record = entrez_data["result"][gse_number]
            logger.debug(f"Successfully retrieved Entrez metadata for {gse_id}")

            # Convert Entrez format to Lobster metadata format
            metadata = self._convert_entrez_to_lobster_metadata(gse_record, gse_id)

            # Validate metadata against transcriptomics schema
            validation_result = self._validate_geo_metadata(metadata)

            # Add fallback markers and warnings
            metadata["_entrez_fallback"] = True
            metadata["_metadata_source"] = "NCBI Entrez E-utilities (esummary)"
            metadata["_metadata_completeness"] = "partial"
            metadata["_warning"] = (
                "Metadata fetched via Entrez fallback due to missing SOFT file. "
                "Sample-level characteristics and protocol details not available. "
                "Basic information (title, summary, organism, platform, sample count) provided."
            )

            # Check platform compatibility BEFORE downloading files (Phase 2: Early Validation)
            try:
                is_compatible, compat_message = self._check_platform_compatibility(
                    gse_id, metadata
                )
                logger.info(
                    f"Platform validation for {gse_id} (Entrez): {compat_message}"
                )
            except UnsupportedPlatformError as e:
                # CHANGE 3: Use helper method for consistent structure
                self.service.data_manager._store_geo_metadata(
                    geo_id=gse_id,
                    metadata=metadata,
                    stored_by="_fetch_gse_metadata_via_entrez (exception)",
                )
                # Enrich with error details via centralized helper
                self.service.data_manager._enrich_geo_metadata(
                    gse_id,
                    validation_result=validation_result,
                    platform_error=str(e),
                    platform_details=e.details,
                    status="unsupported_platform",
                    error_timestamp=datetime.now().isoformat(),
                )
                logger.error(
                    f"Platform validation failed for {gse_id}: {e.details['detected_platforms']}"
                )
                raise

            logger.info(
                f"Successfully fetched {gse_id} metadata via Entrez fallback "
                f"(~70% complete, missing sample characteristics)"
            )

            return metadata, validation_result

        except UnsupportedPlatformError:
            # Re-raise platform errors without catching them
            raise
        except urllib.error.URLError as e:
            logger.error(f"Network error in Entrez fallback for {gse_id}: {e}")
            raise Exception(
                f"Network error fetching Entrez metadata for {gse_id}: {str(e)}"
            )
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing Entrez JSON response for {gse_id}: {e}")
            raise Exception(
                f"Error parsing Entrez metadata response for {gse_id}: {str(e)}"
            )
        except Exception as e:
            logger.error(f"Error in Entrez fallback for {gse_id}: {e}")
            raise

    def _convert_entrez_to_lobster_metadata(
        self, entrez_record: Dict[str, Any], gse_id: str
    ) -> Dict[str, Any]:
        """
        Convert Entrez esummary record to Lobster metadata format.

        Maps Entrez JSON fields to GEOparse-compatible structure for downstream processing.
        Entrez provides basic dataset information but lacks detailed sample characteristics.

        Args:
            entrez_record: Entrez esummary record (JSON parsed)
            gse_id: GSE accession ID

        Returns:
            Dict: Metadata in Lobster format (compatible with _extract_metadata structure)
        """
        try:
            # Extract platform IDs (can be list or single value)
            platform_ids = entrez_record.get("gpl", [])
            if isinstance(platform_ids, str):
                platform_ids = [platform_ids]
            elif not isinstance(platform_ids, list):
                platform_ids = []

            # Extract PubMed IDs
            pubmed_ids = entrez_record.get("pubmedids", [])
            if isinstance(pubmed_ids, str):
                pubmed_ids = [pubmed_ids]
            elif not isinstance(pubmed_ids, list):
                pubmed_ids = []

            # Build metadata dict compatible with GEOparse structure
            metadata = {
                # Core identifiers
                "geo_accession": gse_id,
                "accession": gse_id,
                # Basic information
                "title": entrez_record.get("title", ""),
                "summary": entrez_record.get("summary", ""),
                "type": entrez_record.get(
                    "gdstype", "Expression profiling by high throughput sequencing"
                ),
                # Organism information
                "taxon": entrez_record.get("taxon", ""),
                "organism": entrez_record.get("taxon", ""),
                # Platform information (as list for consistency with GEOparse)
                "platform_id": platform_ids,
                # Sample information
                "n_samples": entrez_record.get("n_samples", 0),
                "sample_count": entrez_record.get("n_samples", 0),
                # Publication information
                "pubmed_id": pubmed_ids if pubmed_ids else "",
                # Dates
                "submission_date": entrez_record.get("pdat", ""),
                "last_update_date": entrez_record.get("pdat", ""),
                # Platform details (limited from Entrez)
                "platforms": self._extract_platform_info_from_entrez(
                    entrez_record, platform_ids
                ),
                # Sample metadata (empty - not available via Entrez)
                "samples": {},
                "sample_id": [],
                # Supplementary files (not available via Entrez)
                "supplementary_file": [],
                # Status
                "status": "Public",  # Assume public if in GEO
                # FTP link (if available)
                "ftp_link": entrez_record.get("ftplink", ""),
            }

            logger.debug(f"Converted Entrez record to Lobster metadata for {gse_id}")
            return metadata

        except Exception as e:
            logger.error(f"Error converting Entrez metadata to Lobster format: {e}")
            # Return minimal metadata to avoid complete failure
            return {
                "geo_accession": gse_id,
                "title": entrez_record.get("title", "Unknown"),
                "summary": entrez_record.get("summary", ""),
                "organism": entrez_record.get("taxon", "Unknown"),
                "platform_id": [],
                "n_samples": 0,
                "_conversion_error": str(e),
            }

    def _extract_platform_info_from_entrez(
        self, entrez_record: Dict[str, Any], platform_ids: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Extract platform information from Entrez record.

        Entrez provides limited platform information compared to SOFT files.
        This method creates a minimal platform dict compatible with downstream processing.

        Args:
            entrez_record: Entrez esummary record
            platform_ids: List of platform GPL IDs

        Returns:
            Dict: Platform information in GEOparse-compatible format
        """
        platforms = {}

        try:
            # Entrez doesn't provide detailed platform metadata in esummary
            # Create minimal platform entries for compatibility
            platform_organism = entrez_record.get("taxon", "")
            platform_tech = entrez_record.get("ptechtype", "")

            for gpl_id in platform_ids:
                platforms[gpl_id] = {
                    "title": f"Platform {gpl_id}",
                    "organism": platform_organism,
                    "technology": (
                        platform_tech if platform_tech else "high throughput sequencing"
                    ),
                    "_note": "Platform details from Entrez are limited. Full details available on GEO website.",
                }

            logger.debug(
                f"Extracted platform info for {len(platform_ids)} platform(s) from Entrez"
            )

        except Exception as e:
            logger.warning(f"Error extracting platform info from Entrez: {e}")

        return platforms

    def _extract_metadata(self, gse) -> Dict[str, Any]:
        """
        Extract comprehensive metadata from GEOparse GSE object.

        Args:
            gse: GEOparse GSE object

        Returns:
            dict: Extracted metadata
        """
        try:
            metadata = {}

            # Define which fields should remain as lists vs be joined as strings
            # Fields that need to remain as lists for downstream processing
            LIST_FIELDS = {
                "supplementary_file",
                "relation",
                "sample_id",
                "platform_id",
                "platform_taxid",
                "sample_taxid",
            }

            # Fields that should be joined as strings for display/summary
            STRING_FIELDS = {
                "title",
                "summary",
                "overall_design",
                "type",
                "contributor",
                "contact_name",
                "contact_email",
                "contact_phone",
                "contact_department",
                "contact_institute",
                "contact_address",
                "contact_city",
                "contact_zip/postal_code",
                "contact_country",
                "geo_accession",
                "status",
                "submission_date",
                "last_update_date",
                "pubmed_id",
                "web_link",
            }

            # Basic metadata from GEOparse
            if hasattr(gse, "metadata"):
                for key, value in gse.metadata.items():
                    # VALIDATION: Skip None keys from malformed SOFT files
                    if key is None:
                        logger.warning(
                            f"Skipping None metadata key in {gse.accession} series metadata "
                            f"(malformed SOFT file from GEO)"
                        )
                        continue

                    if isinstance(value, list):
                        # Keep file-related and ID fields as lists for downstream processing
                        if key in LIST_FIELDS:
                            metadata[key] = value
                        # Join descriptive/text fields as strings for summary generation
                        elif key in STRING_FIELDS:
                            metadata[key] = ", ".join(value) if value else ""
                        else:
                            # For unknown fields, use a conservative approach:
                            # If it looks like a file/ID field, keep as list; otherwise join
                            if any(
                                keyword in key.lower()
                                for keyword in ["file", "url", "id", "accession"]
                            ):
                                metadata[key] = value
                            else:
                                metadata[key] = ", ".join(value) if value else ""
                    else:
                        metadata[key] = value

            # Platform information - keep as structured dict
            if hasattr(gse, "gpls"):
                platforms = {}
                for gpl_id, gpl in gse.gpls.items():
                    platforms[gpl_id] = {
                        "title": self._safely_extract_metadata_field(gpl, "title"),
                        "organism": self._safely_extract_metadata_field(
                            gpl, "organism"
                        ),
                        "technology": self._safely_extract_metadata_field(
                            gpl, "technology"
                        ),
                    }
                metadata["platforms"] = platforms

            # Sample metadata - keep as structured dict
            if hasattr(gse, "gsms"):
                sample_metadata = {}
                for gsm_id, gsm in gse.gsms.items():
                    sample_meta = {}
                    if hasattr(gsm, "metadata"):
                        for key, value in gsm.metadata.items():
                            # VALIDATION: Skip None keys from malformed SOFT files
                            if key is None:
                                logger.warning(
                                    f"Skipping None metadata key for {gsm_id} "
                                    f"(malformed SOFT file from GEO)"
                                )
                                continue

                            if isinstance(value, list):
                                # For sample-level metadata, preserve lists for characteristics
                                # but join others for display
                                if (
                                    key in ["characteristics_ch1", "supplementary_file"]
                                    or "file" in key.lower()
                                ):
                                    sample_meta[key] = value
                                else:
                                    sample_meta[key] = ", ".join(value) if value else ""
                            else:
                                sample_meta[key] = value
                    sample_metadata[gsm_id] = sample_meta
                metadata["samples"] = sample_metadata

            return metadata

        except Exception as e:
            logger.error(f"Error extracting metadata: {e}")
            return {}

    def _safely_extract_metadata_field(
        self, obj, field_name: str, default: str = ""
    ) -> str:
        """
        Safely extract a metadata field from a GEOparse object.

        Handles the common pattern of checking for metadata attribute,
        extracting the field, and joining all elements if it's a list.

        Args:
            obj: GEOparse object (GSE, GPL, or GSM)
            field_name: Name of the metadata field to extract
            default: Default value if field is not found

        Returns:
            str: Extracted metadata value or default
        """
        try:
            if not hasattr(obj, "metadata"):
                return default

            field_value = obj.metadata.get(field_name, [default])

            # Handle list values by joining all elements
            if isinstance(field_value, list):
                return ", ".join(field_value) if field_value else default

            return str(field_value) if field_value else default

        except (AttributeError, IndexError, KeyError):
            return default

    def _validate_geo_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate GEO metadata against transcriptomics schema.

        Args:
            metadata: Extracted GEO metadata dictionary

        Returns:
            Dict containing validation results and schema alignment
        """
        try:
            from lobster.core.schemas.transcriptomics import TranscriptomicsSchema

            # Get the single-cell schema (covers most GEO datasets)
            schema = TranscriptomicsSchema.get_single_cell_schema()
            uns_schema = schema.get("uns", {}).get("optional", [])

            # Check which metadata fields align with our schema
            schema_aligned = {}
            schema_missing = []
            extra_fields = []

            # Check alignment for each field in metadata
            for field in metadata.keys():
                if field in uns_schema:
                    schema_aligned[field] = metadata[field]
                else:
                    extra_fields.append(field)

            # Check for schema fields not present in metadata
            for schema_field in uns_schema:
                if schema_field not in metadata:
                    schema_missing.append(schema_field)

            # Determine data type based on metadata
            data_type = self.service._determine_data_type_from_metadata(metadata)

            validation_result = {
                "schema_aligned_fields": len(schema_aligned),
                "schema_missing_fields": len(schema_missing),
                "extra_fields_count": len(extra_fields),
                "alignment_percentage": (
                    (len(schema_aligned) / len(uns_schema) * 100) if uns_schema else 0.0
                ),
                "aligned_metadata": schema_aligned,
                "missing_fields": schema_missing[:10],  # Limit for display
                "extra_fields": extra_fields[:10],  # Limit for display
                "predicted_data_type": data_type,
                "validation_status": (
                    "PASS" if len(schema_aligned) > len(schema_missing) else "WARNING"
                ),
            }

            logger.debug(
                f"Metadata validation: {validation_result['alignment_percentage']:.1f}% schema alignment"
            )
            return validation_result

        except Exception as e:
            logger.error(f"Error validating metadata: {e}")
            return {
                "validation_status": "ERROR",
                "error_message": str(e),
                "schema_aligned_fields": 0,
                "alignment_percentage": 0.0,
            }

    def _check_platform_compatibility(
        self, geo_id: str, metadata: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """
        Check if dataset platform is supported before downloading files.

        Performs multi-level platform detection:
        1. Series-level platforms (most common)
        2. Sample-level platforms (for mixed-platform datasets)
        3. Keyword matching for unknown platforms

        Args:
            geo_id: GEO series identifier
            metadata: Parsed SOFT file metadata from _extract_metadata()

        Returns:
            Tuple of (is_compatible, message):
                - is_compatible: True if platform is supported or unknown
                - message: Human-readable explanation

        Raises:
            UnsupportedPlatformError: If platform is explicitly unsupported (microarray)
        """
        # CHANGE 1: Store minimal metadata BEFORE any early returns
        # This guarantees metadata availability for all code paths (including early returns)
        if geo_id not in self.service.data_manager.metadata_store:
            logger.debug(f"Storing minimal metadata for {geo_id} before validation")

            # Use helper method for consistent structure
            self.service.data_manager._store_geo_metadata(
                geo_id=geo_id,
                metadata=metadata,
                stored_by="_check_platform_compatibility (entry)",
            )
            logger.debug(f"Stored minimal metadata for {geo_id} before validation")

        # Extract platform information at series level
        series_platforms = metadata.get("platforms", {})

        # Extract platform information at sample level
        # metadata["samples"] is a dict: {gsm_id: {metadata...}}
        sample_platforms = {}
        samples_dict = metadata.get("samples", {})
        if isinstance(samples_dict, dict):
            for gsm_id, sample_meta in samples_dict.items():
                platform_id = sample_meta.get("platform_id")
                if platform_id:
                    sample_platforms.setdefault(platform_id, []).append(gsm_id)

        # Combine both levels
        all_platforms = {}
        for platform_id, platform_data in series_platforms.items():
            all_platforms[platform_id] = {
                "title": platform_data.get("title", ""),
                "level": "series",
                "samples": sample_platforms.get(platform_id, []),
            }

        # Add sample-only platforms
        for platform_id, samples in sample_platforms.items():
            if platform_id not in all_platforms:
                all_platforms[platform_id] = {
                    "title": f"Platform {platform_id}",
                    "level": "sample",
                    "samples": samples,
                }

        if not all_platforms:
            logger.warning(f"No platform information found for {geo_id}")
            return True, "No platform information available - proceeding with caution"

        # Classify platforms
        unsupported_platforms = []
        supported_platforms = []
        experimental_platforms = []
        unknown_platforms = []

        for platform_id, platform_info in all_platforms.items():
            platform_title = platform_info.get("title", "").lower()

            # Check registry
            if platform_id in PLATFORM_REGISTRY:
                status = PLATFORM_REGISTRY[platform_id]

                if status == PlatformCompatibility.UNSUPPORTED:
                    unsupported_platforms.append((platform_id, platform_info))
                elif status == PlatformCompatibility.SUPPORTED:
                    supported_platforms.append((platform_id, platform_info))
                elif status == PlatformCompatibility.EXPERIMENTAL:
                    experimental_platforms.append((platform_id, platform_info))
            else:
                # Unknown platform - use keyword matching
                if any(kw in platform_title for kw in UNSUPPORTED_KEYWORDS):
                    unsupported_platforms.append((platform_id, platform_info))
                elif any(kw in platform_title for kw in SUPPORTED_KEYWORDS):
                    supported_platforms.append((platform_id, platform_info))
                else:
                    unknown_platforms.append((platform_id, platform_info))

        # Decision logic: Reject if ANY samples use unsupported platforms
        # (unless they also have supported platform data)
        if unsupported_platforms:
            # Check if this is a mixed dataset
            if supported_platforms:
                logger.warning(
                    f"{geo_id} has BOTH supported and unsupported platforms. "
                    f"Will attempt to load supported samples only."
                )
                return (
                    True,
                    "Mixed platform dataset - will filter to supported samples",
                )

            # Pure unsupported dataset - reject
            platform_list = "\n".join(
                [
                    f"  - {pid}: {info['title']} (level: {info['level']}, samples: {len(info.get('samples', []))})"
                    for pid, info in unsupported_platforms
                ]
            )

            raise UnsupportedPlatformError(
                message=f"Dataset {geo_id} uses unsupported platform(s)",
                details={
                    "geo_id": geo_id,
                    "unsupported_platforms": [
                        (pid, info["title"]) for pid, info in unsupported_platforms
                    ],
                    "platform_type": "microarray",
                    "explanation": (
                        "This dataset appears to use microarray platform(s), which are not "
                        "currently supported by Lobster. Lobster is designed for RNA-seq data "
                        "(bulk or single-cell) and proteomics data."
                    ),
                    "detected_platforms": platform_list,
                    "suggestions": [
                        "Search for RNA-seq version of this experiment",
                        "Use RNA-seq platforms: Illumina HiSeq, NextSeq, NovaSeq",
                        "Use single-cell platforms: 10X Chromium, Smart-seq",
                        f"Check if {geo_id} has supplementary RNA-seq files",
                    ],
                },
            )

        # Handle experimental platforms
        if experimental_platforms:
            platform_list = ", ".join([pid for pid, _ in experimental_platforms])
            logger.warning(
                f"{geo_id} uses experimental platform(s): {platform_list}. "
                f"Analysis may require manual validation."
            )
            return True, "Experimental platform detected - proceed with validation"

        # Handle unknown platforms conservatively
        if unknown_platforms and not supported_platforms:
            platform_list = ", ".join([pid for pid, _ in unknown_platforms])
            logger.warning(
                f"{geo_id} has unknown platform(s): {platform_list}. "
                f"Will attempt loading but recommend validation."
            )
            return True, "Unknown platform - will attempt loading"

        # All platforms supported (GPL registry check passed)
        platform_list = ", ".join([pid for pid, _ in supported_platforms])
        logger.info(f"GPL registry check passed for {geo_id}: {platform_list}")

        # === TIER 2: LLM MODALITY DETECTION (Phase 2.1) ===
        logger.info(f"Running LLM modality detection for {geo_id}...")

        # Initialize DataExpertAssistant (lazy initialization pattern)
        # Keep lazy import inside method to avoid circular imports
        from lobster.agents.data_expert.assistant import DataExpertAssistant

        if not hasattr(self.service, "_data_expert_assistant"):
            self.service._data_expert_assistant = DataExpertAssistant()

        # Call LLM to detect modality
        modality_result = self.service._data_expert_assistant.detect_modality(metadata, geo_id)

        if modality_result is None:
            # LLM analysis failed - fall back to permissive mode with warning
            logger.warning(
                f"LLM modality detection failed for {geo_id}. "
                f"Proceeding with permissive mode (may cause issues with multi-omics data)."
            )
            return True, "LLM modality detection unavailable - proceeding with caution"

        # Log detection results
        logger.info(
            f"Modality detected: {modality_result.modality} "
            f"(confidence: {modality_result.confidence:.2f}, "
            f"supported: {modality_result.is_supported})"
        )

        # CHANGE 2: Enrich existing metadata entry instead of "store if missing"
        # This ensures progressive enrichment as validation proceeds
        if hasattr(self.service, "data_manager") and hasattr(
            self.service.data_manager, "metadata_store"
        ):
            # Store modality detection results with enforced nested structure
            modality_detection_info = {
                "modality": modality_result.modality,
                "confidence": modality_result.confidence,
                "detected_signals": modality_result.detected_signals,
                "timestamp": pd.Timestamp.now().isoformat(),
            }

            if geo_id in self.service.data_manager.metadata_store:
                # Update existing entry with validation results (progressive enrichment)
                self.service.data_manager._enrich_geo_metadata(
                    geo_id,
                    modality_detection=modality_detection_info,
                    status="validated",
                    validation_timestamp=datetime.now().isoformat(),
                )
            else:
                # Fallback: Create new entry if somehow missing (shouldn't happen with Change 1)
                logger.warning(
                    f"Metadata for {geo_id} missing during enrichment - creating new entry"
                )
                self.service.data_manager._store_geo_metadata(
                    geo_id=geo_id,
                    metadata=metadata,
                    stored_by="_check_platform_compatibility (fallback)",
                    modality_detection=modality_detection_info,
                )

        # Decision: Handle multi-modal datasets intelligently
        if not modality_result.is_supported:
            # Check if this is a multi-modal dataset by examining sample types
            logger.info(
                f"Detected unsupported modality '{modality_result.modality}', checking for multi-modal composition..."
            )
            sample_types = self._detect_sample_types(metadata)

            # Check if we have any supported modalities
            has_rna = "rna" in sample_types and len(sample_types["rna"]) > 0
            has_unsupported = any(
                modality in sample_types and len(sample_types[modality]) > 0
                for modality in ["protein", "vdj", "atac"]
            )

            if has_rna and has_unsupported:
                # Multi-modal dataset with RNA + unsupported modalities
                logger.info(
                    f"Multi-modal dataset detected: RNA ({len(sample_types.get('rna', []))}) samples + "
                    f"unsupported modalities. Will load RNA samples only."
                )

                # Store multi-modal info in metadata for downstream use
                multimodal_info = {
                    "is_multimodal": True,
                    "sample_types": sample_types,
                    "supported_types": ["rna"],
                    "unsupported_types": [t for t in sample_types.keys() if t != "rna"],
                    "detection_timestamp": pd.Timestamp.now().isoformat(),
                }

                # Update metadata store with multi-modal info
                if geo_id in self.service.data_manager.metadata_store:
                    self.service.data_manager._enrich_geo_metadata(
                        geo_id,
                        multimodal_info=multimodal_info,
                    )

                # Log what will be skipped
                unsupported_summary = ", ".join(
                    [
                        f"{modality}: {len(samples)} samples"
                        for modality, samples in sample_types.items()
                        if modality != "rna"
                    ]
                )
                logger.warning(
                    f"Skipping unsupported modalities in {geo_id}: {unsupported_summary}. "
                    f"Support planned for future releases (v2.6+)."
                )

                return (
                    True,
                    f"Multi-modal dataset - loading RNA samples only ({len(sample_types['rna'])} samples)",
                )

            elif has_rna and not has_unsupported:
                # RNA-only dataset that was misclassified by pre-filter
                logger.info(
                    "Dataset is RNA-only despite modality detection. Proceeding with load."
                )
                return (True, "RNA-only dataset (pre-filter was overly conservative)")

            else:
                # No RNA samples found - truly unsupported
                signals_display = "\n".join(
                    [f"  - {signal}" for signal in modality_result.detected_signals[:5]]
                )
                if len(modality_result.detected_signals) > 5:
                    signals_display += f"\n  ... and {len(modality_result.detected_signals) - 5} more signals"

                raise FeatureNotImplementedError(
                    message=f"Dataset {geo_id} uses unsupported sequencing modality: {modality_result.modality}",
                    details={
                        "geo_id": geo_id,
                        "modality": modality_result.modality,
                        "confidence": modality_result.confidence,
                        "detected_signals": modality_result.detected_signals,
                        "explanation": modality_result.compatibility_reason,
                        "current_workaround": (
                            f"Lobster v2.3 currently supports bulk RNA-seq, 10X single-cell, and Smart-seq2. "
                            f"Support for {modality_result.modality} is planned for future releases."
                        ),
                        "suggestions": modality_result.suggestions,
                        "estimated_implementation": "Planned for Lobster v2.6-v2.8 depending on modality",
                        "detected_signals_formatted": signals_display,
                        "sample_types_detected": (
                            sample_types if sample_types else "No samples classified"
                        ),
                    },
                )

        # Supported modality - return success with modality info
        return (
            True,
            f"Modality compatible: {modality_result.modality} (confidence: {modality_result.confidence:.0%})",
        )

    def _detect_sample_types(self, metadata: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Detect data types for each sample in a GEO dataset.

        This method examines sample-level metadata to classify samples by modality
        (RNA, protein, VDJ, ATAC, etc.). It uses GEO's standardized fields when
        available (library_strategy) and falls back to characteristics_ch1 patterns.

        Args:
            metadata: GEO metadata dict with 'samples' key containing per-sample metadata

        Returns:
            Dict mapping modality names to lists of sample IDs:
                {"rna": ["GSM1", "GSM2"], "protein": ["GSM3"], "vdj": ["GSM4"]}
        """
        sample_types: Dict[str, List[str]] = {}
        samples_dict = metadata.get("samples", {})

        if not samples_dict:
            logger.warning("No samples found in metadata for sample type detection")
            return sample_types

        logger.info(f"Detecting sample types for {len(samples_dict)} samples...")

        for gsm_id, sample_meta in samples_dict.items():
            detected_type = None

            # Strategy 1: Use library_strategy field (most reliable - NCBI controlled vocabulary)
            lib_strategy = sample_meta.get("library_strategy", "")
            if lib_strategy:
                lib_strategy_lower = lib_strategy.lower()
                if "rna-seq" in lib_strategy_lower or "rna seq" in lib_strategy_lower:
                    detected_type = "rna"
                elif "atac" in lib_strategy_lower:
                    detected_type = "atac"
                # Add other strategies as needed

            # Strategy 2: Check characteristics_ch1 field (flexible but less standardized)
            if not detected_type:
                chars = sample_meta.get("characteristics_ch1", [])
                if isinstance(chars, list):
                    chars_text = " ".join([str(c).lower() for c in chars])
                else:
                    chars_text = str(chars).lower()

                # RNA detection
                if any(
                    pattern in chars_text
                    for pattern in [
                        "assay: rna",
                        "assay:rna",
                        "library type: gene expression",
                        "library_type: gene expression",
                        "data type: rna-seq",
                        "datatype: rna-seq",
                        "library type: gex",
                        "library_type: gex",
                    ]
                ):
                    detected_type = "rna"

                # Protein detection (CITE-seq, antibody capture)
                elif any(
                    pattern in chars_text
                    for pattern in [
                        "assay: protein",
                        "assay:protein",
                        "library type: antibody capture",
                        "library_type: antibody capture",
                        "antibody-derived tag",
                        "adt",
                        "cite-seq protein",
                        "citeseq protein",
                    ]
                ):
                    detected_type = "protein"

                # VDJ detection (TCR/BCR sequencing)
                elif any(
                    pattern in chars_text
                    for pattern in [
                        "assay: vdj",
                        "assay:vdj",
                        "library type: vdj",
                        "library_type: vdj",
                        "tcr-seq",
                        "tcr seq",
                        "bcr-seq",
                        "bcr seq",
                        "immune repertoire",
                    ]
                ):
                    detected_type = "vdj"

                # ATAC detection
                elif any(
                    pattern in chars_text
                    for pattern in [
                        "assay: atac",
                        "assay:atac",
                        "library type: atac",
                        "library_type: atac",
                        "chromatin accessibility",
                    ]
                ):
                    detected_type = "atac"

            # Strategy 3: Check sample title for common patterns
            if not detected_type:
                title = sample_meta.get("title", "").lower()
                if any(
                    pattern in title for pattern in ["_rna", "_gex", "_gene_expression"]
                ):
                    detected_type = "rna"
                elif any(
                    pattern in title for pattern in ["_protein", "_adt", "_antibody"]
                ):
                    detected_type = "protein"
                elif any(pattern in title for pattern in ["_vdj", "_tcr", "_bcr"]):
                    detected_type = "vdj"
                elif "_atac" in title:
                    detected_type = "atac"

            # Store result
            if detected_type:
                sample_types.setdefault(detected_type, []).append(gsm_id)
                logger.debug(f"Sample {gsm_id}: detected as '{detected_type}'")
            else:
                # Unknown - default to RNA for backward compatibility
                sample_types.setdefault("rna", []).append(gsm_id)
                logger.debug(f"Sample {gsm_id}: type unclear, defaulting to 'rna'")

        # Log summary
        summary = ", ".join(
            [
                f"{modality}: {len(samples)}"
                for modality, samples in sample_types.items()
            ]
        )
        logger.info(f"Sample type detection complete: {summary}")

        return sample_types
