"""Services for orchestrating publication queue extraction workflows."""

from __future__ import annotations

import json
import re
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional

import time

from lobster.core.data_manager_v2 import DataManagerV2
from lobster.core.schemas.publication_queue import PublicationQueueEntry, PublicationStatus
from lobster.services.data_access.content_access_service import ContentAccessService
from lobster.services.data_access.workspace_content_service import (
    WorkspaceContentService,
    ContentType,
    MetadataContent,
)
from lobster.tools.providers.sra_provider import SRAProvider
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


class PublicationProcessingService:
    """High-level orchestration for publication queue extraction."""

    def __init__(self, data_manager: DataManagerV2) -> None:
        self.data_manager = data_manager
        self.content_service = ContentAccessService(data_manager=data_manager)
        self.workspace_service = WorkspaceContentService(data_manager=data_manager)
        self.sra_provider = SRAProvider(data_manager=data_manager)
        # Lazy-initialized to avoid creating new provider (and Redis connection) per entry
        self._pubmed_provider = None
        self._timing_enabled = False
        self._latest_timings: Dict[str, float] = {}

    @property
    def pubmed_provider(self):
        """
        Lazy-initialized PubMedProvider singleton for this service.

        Avoids creating a new provider (and triggering Redis connection pool
        initialization) for every entry processed. The provider is created
        once and reused for the lifetime of the service.
        """
        if self._pubmed_provider is None:
            from lobster.tools.providers.pubmed_provider import PubMedProvider
            self._pubmed_provider = PubMedProvider(data_manager=self.data_manager)
        return self._pubmed_provider

    def _get_best_source_for_extraction(self, entry: PublicationQueueEntry) -> Optional[str]:
        """
        Get best source for content extraction with PMC-first strategy.

        Priority (identifiers first for PMC Open Access, URLs as fallback):
        1. PMC ID (from NCBI enrichment - guaranteed free open access XML)
        2. PMID (triggers PMC lookup in content_access_service)
        3. PubMed URL (extract PMID for PMC lookup)
        4. DOI (can resolve to PMC via content_access_service)
        5. Fulltext URL (publisher page - often paywalled, fallback only)
        6. PDF URL (direct PDF for Docling extraction)
        7. Metadata URL (webpage scraping - last resort)

        This ordering ensures PMC Open Access is tried BEFORE publisher URLs,
        avoiding 403 Forbidden errors from paywalled publishers (Cell, Elsevier, etc.).

        Args:
            entry: PublicationQueueEntry with URL fields populated from RIS

        Returns:
            Best source identifier/URL for content extraction, or None
        """
        # Priority 1: PMC ID (from NCBI enrichment - guaranteed free open access)
        if entry.pmc_id:
            logger.info(f"Using PMC ID (priority 1, open access): {entry.pmc_id}")
            return entry.pmc_id

        # Priority 2: PMID (triggers automatic PMC lookup in content_access_service)
        if entry.pmid:
            logger.info(f"Using PMID (priority 2, may resolve to PMC): PMID:{entry.pmid}")
            return f"PMID:{entry.pmid}"

        # Priority 3: PubMed URL → extract PMID for PMC lookup
        if entry.pubmed_url:
            pmid = self._extract_pmid_from_url(entry.pubmed_url)
            if pmid:
                logger.info(f"Using PubMed URL (priority 3), extracted PMID:{pmid}")
                return f"PMID:{pmid}"

        # Priority 4: DOI (can resolve to PMC via content_access_service)
        if entry.doi:
            logger.info(f"Using DOI (priority 4): {entry.doi}")
            return entry.doi

        # Priority 5: Fulltext URL (fallback - may be paywalled)
        if entry.fulltext_url:
            logger.info(f"Using fulltext URL (priority 5, fallback): {entry.fulltext_url}")
            return entry.fulltext_url

        # Priority 6: Direct PDF (fallback for Docling extraction)
        if entry.pdf_url:
            logger.info(f"Using PDF URL (priority 6, fallback): {entry.pdf_url}")
            return entry.pdf_url

        # Priority 7: Article/metadata URL (last resort - webpage scraping)
        if entry.metadata_url:
            logger.info(f"Using metadata URL (priority 7, last resort): {entry.metadata_url}")
            return entry.metadata_url

        return None

    def _extract_pmid_from_url(self, pubmed_url: str) -> Optional[str]:
        """
        Extract PMID from a PubMed URL.

        Handles formats:
        - http://www.ncbi.nlm.nih.gov/pubmed/38906102
        - https://pubmed.ncbi.nlm.nih.gov/38906102/

        Args:
            pubmed_url: PubMed URL string

        Returns:
            PMID string or None if extraction fails
        """
        if not pubmed_url:
            return None

        # Pattern for PMID in URL path
        match = re.search(r"/pubmed/(\d+)", pubmed_url.lower())
        if match:
            return match.group(1)

        # Alternative pattern for newer PubMed URLs
        match = re.search(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d+)", pubmed_url.lower())
        if match:
            return match.group(1)

        return None

    def _resolve_identifiers_via_ncbi(self, doi: str) -> Dict[str, str]:
        """
        Resolve DOI to PMID and PMC ID using NCBI ID Converter API.

        This enables E-Link enrichment for publications that only have DOIs
        (common in RIS exports from Crossref and publisher websites).

        API: http://www.ncbi.nlm.nih.gov/pmc/tools/id-converter-api/

        Args:
            doi: Digital Object Identifier (e.g., "10.3389/fendo.2022.970825")

        Returns:
            Dict with keys: 'pmid', 'pmc', 'doi' (empty string if not found)
        """
        import requests
        from lxml import html

        from lobster.config.settings import get_settings

        settings = get_settings()
        ncbi_email = getattr(settings, "NCBI_EMAIL", "lobster@omics-os.com")

        convert_url = (
            f"https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
            f"?tool=lobster&email={ncbi_email}&ids={doi}"
        )

        try:
            response = requests.get(convert_url, timeout=10)
            response.raise_for_status()

            tree = html.fromstring(response.content)
            record = tree.find(".//record")

            if record is None:
                logger.debug(f"No record found for DOI {doi} in NCBI ID Converter")
                return {"pmid": "", "pmc": "", "doi": doi}

            attrib = record.attrib

            # Check for error status (DOI not in PubMed)
            if "status" in attrib and attrib["status"] == "error":
                errmsg = attrib.get("errmsg", "Unknown error")
                logger.debug(f"NCBI ID Converter error for DOI {doi}: {errmsg}")
                return {"pmid": "", "pmc": "", "doi": doi}

            result = {
                "pmid": attrib.get("pmid", ""),
                "pmc": attrib.get("pmcid", ""),
                "doi": attrib.get("doi", doi),
            }

            if result["pmid"]:
                logger.info(f"Resolved DOI {doi} → PMID:{result['pmid']}, PMC:{result['pmc']}")
            else:
                logger.debug(f"DOI {doi} not found in PubMed (may be preprint or non-indexed)")

            return result

        except Exception as e:
            logger.warning(f"NCBI ID Converter failed for DOI {doi}: {e}")
            return {"pmid": "", "pmc": "", "doi": doi}

    # ------------------------------------------------------------------
    # Timing helpers
    # ------------------------------------------------------------------

    @contextmanager
    def _measure_step(self, name: str):
        """Context manager to capture elapsed time for a processing step."""

        if not self._timing_enabled:
            yield
            return

        start = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - start
            self._latest_timings[name] = elapsed
            logger.debug("%s completed in %.2fs", name, elapsed)

    def enable_timing(self, enabled: bool = True) -> None:
        """Enable or disable per-step timing instrumentation."""

        self._timing_enabled = enabled
        if not enabled:
            self._latest_timings = {}

    def get_latest_timings(self) -> Dict[str, float]:
        """Return the most recent per-step timing measurements."""

        return dict(self._latest_timings)

    def _resolve_identifiers(self, entry: PublicationQueueEntry) -> Dict[str, Any]:
        """
        Resolve missing identifiers from DOI before NCBI enrichment.

        This enables E-Link dataset discovery for publications that only have DOIs
        in their RIS data (common with Crossref and publisher exports).

        Priority:
        1. Skip if PMID already present
        2. DOI → NCBI ID Converter → PMID + PMC ID

        Args:
            entry: PublicationQueueEntry with DOI but potentially missing PMID

        Returns:
            Dict with resolution results:
            - resolved_pmid: The resolved PMID (or None)
            - resolved_pmc: The resolved PMC ID (or None)
            - success: Whether resolution found new identifiers
            - skipped: Whether resolution was skipped (PMID already present)
            - error: Error message if failed
        """
        result = {
            "resolved_pmid": None,
            "resolved_pmc": None,
            "success": False,
            "skipped": False,
            "error": None,
        }

        # Check if PMID already available
        pmid = entry.pmid
        if not pmid and entry.pubmed_url:
            pmid = self._extract_pmid_from_url(entry.pubmed_url)

        if pmid:
            result["skipped"] = True
            result["resolved_pmid"] = pmid
            logger.debug(f"Skipping identifier resolution - PMID already present: {pmid}")
            return result

        # Try to resolve from DOI
        if not entry.doi:
            result["error"] = "No DOI available for identifier resolution"
            return result

        logger.info(f"Resolving identifiers for DOI: {entry.doi}")

        resolved = self._resolve_identifiers_via_ncbi(entry.doi)

        if resolved["pmid"]:
            result["resolved_pmid"] = resolved["pmid"]
            result["resolved_pmc"] = resolved["pmc"] if resolved["pmc"] else None
            result["success"] = True

            # Update the entry with resolved identifiers
            entry.pmid = resolved["pmid"]
            if resolved["pmc"] and not entry.pmc_id:
                entry.pmc_id = resolved["pmc"]

            logger.info(
                f"Successfully resolved DOI {entry.doi} → "
                f"PMID:{resolved['pmid']}, PMC:{resolved['pmc'] or 'N/A'}"
            )
        else:
            result["error"] = f"DOI {entry.doi} not found in PubMed (may be preprint or non-indexed)"
            logger.debug(result["error"])

        return result

    def _enrich_from_ncbi(self, entry: PublicationQueueEntry) -> Dict[str, Any]:
        """
        Enrich publication entry with NCBI E-Link data before full content extraction.

        Uses NCBI E-Link API to find linked datasets (GEO, SRA, BioProject, BioSample)
        directly from the PMID, without needing to read the full publication.

        This is faster and more reliable than regex extraction from full text because:
        1. No need to download/parse the full publication
        2. Uses official NCBI database links (authoritative)
        3. Works even for paywalled publications
        4. Discovers datasets that may not be mentioned in the text

        Args:
            entry: PublicationQueueEntry with PMID or pubmed_url

        Returns:
            Dict with enrichment results:
            - pmid: The PMID used
            - linked_datasets: Dict of database → List of accessions
            - pmc_id: PMC ID if available (for full text access)
            - success: Whether enrichment succeeded
            - error: Error message if failed
        """
        result = {
            "pmid": None,
            "linked_datasets": {},
            "pmc_id": None,
            "success": False,
            "error": None,
        }

        # Get PMID from entry or extract from URL
        pmid = entry.pmid
        if not pmid and entry.pubmed_url:
            pmid = self._extract_pmid_from_url(entry.pubmed_url)

        if not pmid:
            result["error"] = "No PMID available for NCBI enrichment"
            return result

        result["pmid"] = pmid
        logger.info(f"Enriching publication from NCBI E-Link: PMID {pmid}")

        try:
            # Use lazy-initialized provider to avoid repeated Redis pool initialization
            provider = self.pubmed_provider

            # Get linked datasets via E-Link
            with self._measure_step("ncbi_enrich:linked_datasets"):
                linked_datasets = provider._find_linked_datasets(pmid)
            result["linked_datasets"] = linked_datasets

            # Also try to get PMC ID for better full text access via E-Link
            try:
                import urllib.request

                with self._measure_step("ncbi_enrich:pmc_lookup"):
                    elink_url = provider.build_ncbi_url(
                        "elink",
                        {
                            "dbfrom": "pubmed",
                            "db": "pmc",
                            "id": pmid,
                            "linkname": "pubmed_pmc",
                            "retmode": "json",
                        },
                    )
                    content = provider._make_ncbi_request(
                        elink_url, f"get PMC ID for {pmid}"
                    )
                    text = content.decode("utf-8")
                    pmc_data = json.loads(text)

                    # Extract PMC ID from response
                    linksets = pmc_data.get("linksets", [])
                    if linksets:
                        linksetdbs = linksets[0].get("linksetdbs", [])
                        if linksetdbs:
                            links = linksetdbs[0].get("links", [])
                            if links:
                                pmc_id = f"PMC{links[0]}"
                                result["pmc_id"] = pmc_id
                                logger.info(f"Found PMC ID: {pmc_id}")
            except Exception as e:
                logger.debug(f"Could not get PMC ID: {e}")

            result["success"] = True

            # Log results
            total_linked = sum(len(v) for v in linked_datasets.values())
            logger.info(
                f"NCBI E-Link enrichment complete: {total_linked} linked datasets found"
            )
            for db, ids in linked_datasets.items():
                if ids:
                    logger.info(f"  {db}: {', '.join(ids[:5])}")

        except Exception as e:
            logger.error(f"NCBI E-Link enrichment failed: {e}")
            result["error"] = str(e)

        return result

    def process_entry(
        self, entry_id: str,
        extraction_tasks: str = "resolve_identifiers,ncbi_enrich,fetch_sra_metadata,metadata,methods,identifiers",
    ) -> str:
        """Process a single publication queue entry."""

        data_manager = self.data_manager

        try:
            try:
                entry = data_manager.publication_queue.get_entry(entry_id)
            except Exception as e:  # pragma: no cover - defensive
                return (
                    "## Error: Publication Queue Entry Not Found\n\n"
                    f"Entry ID '{entry_id}' not found in publication queue.\n\n"
                    f"**Error**: {str(e)}\n\n"
                    "**Tip**: Use get_content_from_workspace(workspace='publication_queue') "
                    "to list available entries."
                )

            tasks = [task.strip().lower() for task in extraction_tasks.split(",")]
            if self._timing_enabled:
                self._latest_timings = {}

            # FIX: Removed intermediate update_status call to prevent O(n²) file rewrites
            # All updates are now collected in-memory and written once at the end

            # In-memory collection for all extracted identifiers (merged at end)
            all_extracted_identifiers = {}
            is_paywalled = False
            paywall_error = None

            response_parts = [
                f"## Processing Publication: {entry.title or entry.entry_id}",
                "",
                f"**Entry ID**: {entry_id}",
                "**Status**: EXTRACTING → Processing",
                "",
            ]

            extracted_data = {}
            identifiers_found = None
            content_result = None
            ncbi_enrichment_result = None
            identifier_resolution_result = None
            workspace_keys: List[str] = []

            # Identifier Resolution (runs FIRST - DOI → PMID before E-Link enrichment)
            # This enables E-Link for RIS files that only have DOIs (Crossref, publishers)
            if "resolve_identifiers" in tasks or "full_text" in tasks:
                with self._measure_step("resolve_identifiers"):
                    try:
                        identifier_resolution_result = self._resolve_identifiers(entry)

                        if identifier_resolution_result["success"]:
                            response_parts.append("✓ Identifier resolution complete:")
                            response_parts.append(
                                f"  - DOI: {entry.doi} → PMID:{identifier_resolution_result['resolved_pmid']}"
                            )
                            if identifier_resolution_result["resolved_pmc"]:
                                response_parts.append(
                                    f"  - PMC ID: {identifier_resolution_result['resolved_pmc']}"
                                )
                            extracted_data["identifier_resolution"] = {
                                "pmid": identifier_resolution_result["resolved_pmid"],
                                "pmc": identifier_resolution_result["resolved_pmc"],
                            }
                        elif identifier_resolution_result["skipped"]:
                            response_parts.append(
                                f"⚠ Identifier resolution skipped: PMID already present ({identifier_resolution_result['resolved_pmid']})"
                            )
                        else:
                            error = identifier_resolution_result.get("error", "Unknown error")
                            response_parts.append(f"⚠ Identifier resolution: {error}")

                    except Exception as e:
                        response_parts.append(f"✗ Identifier resolution failed: {str(e)}")

                response_parts.append("")

            # NCBI E-Link enrichment (runs after identifier resolution)
            # This is fast and reliable - uses official NCBI database links
            if "ncbi_enrich" in tasks or "full_text" in tasks:
                with self._measure_step("ncbi_enrich"):
                    try:
                        ncbi_enrichment_result = self._enrich_from_ncbi(entry)

                        if ncbi_enrichment_result["success"]:
                            linked = ncbi_enrichment_result["linked_datasets"]
                            total_linked = sum(len(v) for v in linked.values())

                            response_parts.append("✓ NCBI E-Link enrichment complete:")
                            response_parts.append(f"  - PMID: {ncbi_enrichment_result['pmid']}")

                            if ncbi_enrichment_result["pmc_id"]:
                                response_parts.append(
                                    f"  - PMC ID: {ncbi_enrichment_result['pmc_id']}"
                                )
                                # Update entry with PMC ID for better full text access
                                if not entry.pmc_id:
                                    entry.pmc_id = ncbi_enrichment_result["pmc_id"]

                            if total_linked > 0:
                                response_parts.append(f"  - Linked datasets: {total_linked}")
                                for db, ids in linked.items():
                                    if ids:
                                        response_parts.append(f"    - {db}: {', '.join(ids[:5])}")
                                        if len(ids) > 5:
                                            response_parts.append(f"      (+{len(ids) - 5} more)")

                                # Merge NCBI-linked datasets with existing identifiers
                                # Convert to lowercase keys to match existing schema
                                ncbi_identifiers = {
                                    "geo": linked.get("GEO", []),
                                    "sra": linked.get("SRA", []),
                                    "bioproject": linked.get("BioProject", []),
                                    "biosample": linked.get("BioSample", []),
                                }

                                # FIX: Collect in-memory instead of writing to disk
                                # Will be merged and written once at the end
                                for key, values in ncbi_identifiers.items():
                                    if values:
                                        if key in all_extracted_identifiers:
                                            all_extracted_identifiers[key].extend(values)
                                        else:
                                            all_extracted_identifiers[key] = list(values)
                                extracted_data["ncbi_enrichment"] = ncbi_identifiers
                            else:
                                response_parts.append("  - No linked datasets found in NCBI")
                        else:
                            error = ncbi_enrichment_result.get("error", "Unknown error")
                            response_parts.append(f"⚠ NCBI E-Link enrichment skipped: {error}")

                    except Exception as e:
                        response_parts.append(f"✗ NCBI E-Link enrichment failed: {str(e)}")

                response_parts.append("")

            # SRA metadata fetching (runs after NCBI enrichment)
            # Fetches detailed sample metadata for BioProject IDs found via E-Link
            # Note: E-Link returns internal SRA link IDs (SRA12345678) which pysradb cannot use
            # Instead, we use BioProject IDs (PRJNA*) which pysradb can resolve to samples
            if "fetch_sra_metadata" in tasks or "full_text" in tasks:
                with self._measure_step("fetch_sra_metadata"):
                    try:
                        # Get BioProject IDs from NCBI enrichment (these are proper accessions)
                        bioproject_ids = extracted_data.get("ncbi_enrichment", {}).get("bioproject", [])

                        if bioproject_ids:
                            response_parts.append(f"✓ Fetching SRA metadata for {len(bioproject_ids)} BioProject(s):")

                            sra_fetch_success = 0
                            sra_fetch_failed = 0
                            sra_workspace_keys = []  # Track SRA workspace keys for handoff

                            for bioproject_id in bioproject_ids:
                                try:
                                    # Get SRAweb instance from provider
                                    sraweb = self.sra_provider._get_sraweb()

                                    # Fetch metadata using pysradb with BioProject ID
                                    df = sraweb.sra_metadata(bioproject_id, detailed=True)

                                    if df is not None and not df.empty:
                                        # Convert DataFrame to dict for storage
                                        metadata_dict = df.to_dict(orient="records")

                                        # Create MetadataContent for workspace storage
                                        metadata_content = MetadataContent(
                                            identifier=f"sra_{bioproject_id}_samples",
                                            content_type="sra_samples",
                                            description=f"SRA sample metadata for BioProject {bioproject_id}",
                                            data={"samples": metadata_dict, "sample_count": len(df)},
                                            related_datasets=[bioproject_id],
                                            source="SRAProvider",
                                            cached_at=datetime.now().isoformat(),
                                        )

                                        # Write to workspace
                                        workspace_path = self.workspace_service.write_content(
                                            metadata_content, ContentType.METADATA
                                        )

                                        # Track workspace key for handoff
                                        sra_workspace_keys.append(f"sra_{bioproject_id}_samples")

                                        sra_fetch_success += 1
                                        response_parts.append(
                                            f"  - {bioproject_id}: {len(df)} sample(s) → {workspace_path}"
                                        )

                                        logger.info(
                                            f"Fetched {len(df)} samples for BioProject {bioproject_id}"
                                        )
                                    else:
                                        response_parts.append(
                                            f"  - {bioproject_id}: No metadata found (may be restricted or invalid)"
                                        )
                                        sra_fetch_failed += 1

                                except Exception as e:
                                    logger.warning(f"Failed to fetch SRA metadata for {bioproject_id}: {e}")
                                    response_parts.append(f"  - {bioproject_id}: ✗ Failed ({str(e)})")
                                    sra_fetch_failed += 1

                            # Summary
                            if sra_fetch_success > 0:
                                response_parts.append(
                                    f"✓ SRA metadata fetch complete: {sra_fetch_success} succeeded, {sra_fetch_failed} failed"
                                )
                            else:
                                response_parts.append(
                                    f"⚠ SRA metadata fetch: All {sra_fetch_failed} ID(s) failed"
                                )

                            # Extend workspace_keys with SRA workspace keys for handoff
                            if sra_workspace_keys:
                                workspace_keys.extend(sra_workspace_keys)

                        else:
                            response_parts.append(
                                "⚠ SRA metadata fetch skipped: No BioProject IDs from NCBI enrichment"
                            )

                    except Exception as e:
                        response_parts.append(f"✗ SRA metadata fetch failed: {str(e)}")
                        logger.error(f"SRA metadata fetch error: {e}")

                response_parts.append("")

            # Metadata extraction
            if "metadata" in tasks or "full_text" in tasks:
                with self._measure_step("metadata"):
                    try:
                        # Use priority-based URL selection (fulltext > pdf > metadata > pubmed > doi)
                        source = self._get_best_source_for_extraction(entry)
                        if source:
                            with self._measure_step("metadata:get_content"):
                                content_result = self.content_service.get_full_content(
                                    source=source,
                                    prefer_webpage=True,
                                    keywords=["abstract", "introduction", "methods"],
                                    max_paragraphs=100,
                                )

                            # Check for paywall error
                            if content_result and content_result.get("error"):
                                error_msg = content_result.get("error", "")
                                if "paywalled" in error_msg.lower():
                                    # FIX: Track paywall status in-memory, write at end
                                    is_paywalled = True
                                    paywall_error = error_msg
                                    response_parts.append(f"⚠ Publication is paywalled: {error_msg}")
                                    response_parts.append("User can manually add content later.")
                                    # Continue with partial extraction
                                else:
                                    response_parts.append(f"✗ Metadata extraction failed: {error_msg}")
                            else:
                                content = content_result.get("content", "") if content_result else ""
                                extracted_data["metadata_extracted"] = bool(content)
                                response_parts.append("✓ Metadata extracted successfully")
                        else:
                            response_parts.append(
                                "⚠ No identifier or URL available for metadata extraction"
                            )
                    except Exception as e:  # pragma: no cover - provider errors
                        response_parts.append(f"✗ Metadata extraction failed: {str(e)}")

            # Methods extraction
            if "methods" in tasks or "full_text" in tasks:
                with self._measure_step("methods"):
                    try:
                        # Use priority-based URL selection (fulltext > pdf > metadata > pubmed > doi)
                        source = self._get_best_source_for_extraction(entry)
                        if source:
                            if not content_result or content_result.get("error"):
                                with self._measure_step("methods:get_content"):
                                    content_result = self.content_service.get_full_content(
                                        source=source
                                    )

                            if content_result and content_result.get("content"):
                                with self._measure_step("methods:extract"):
                                    methods_dict = self.content_service.extract_methods(
                                        content_result
                                    )
                                methods_content = methods_dict.get("methods_text", "")
                            else:
                                methods_content = ""

                            extracted_data["methods_extracted"] = bool(methods_content)
                            if methods_content:
                                response_parts.append("✓ Methods section extracted successfully")
                            else:
                                response_parts.append("⚠ Methods section not found in content")
                        else:
                            response_parts.append(
                                "⚠ No identifier or URL available for methods extraction"
                            )
                    except Exception as e:  # pragma: no cover - provider errors
                        response_parts.append(f"✗ Methods extraction failed: {str(e)}")

            # Identifier extraction
            if "identifiers" in tasks or "full_text" in tasks:
                with self._measure_step("identifiers"):
                    try:
                        # Use priority-based URL selection (fulltext > pdf > metadata > pubmed > doi)
                        source = self._get_best_source_for_extraction(entry)
                        if source:
                            if not content_result or content_result.get("error"):
                                with self._measure_step("identifiers:get_content"):
                                    content_result = self.content_service.get_full_content(
                                        source=source
                                    )

                            full_content = (
                                content_result.get("content", "") if content_result else ""
                            )
                            extracted_data["identifiers_extracted"] = bool(full_content)

                            import re

                            with self._measure_step("identifiers:regex"):
                                # Expanded identifier patterns based on manual inspection
                                # Categories:
                                # 1. NCBI databases (GEO, SRA, BioProject, BioSample)
                                # 2. European databases (ENA/EBI, ArrayExpress, EGA)
                                # 3. Japanese database (DDBJ)
                                # 4. Chinese database (CNGB/NGDC)
                                # 5. Controlled access (dbGaP)
                                # 6. General repositories (Zenodo, Figshare)
                                identifiers_found = {
                                    # GEO: Gene Expression Omnibus (NCBI)
                                    "geo": re.findall(r"GSE\d{4,}|GDS\d{4,}", full_content),
                                    # SRA: Sequence Read Archive (NCBI + ENA + DDBJ)
                                    # S=NCBI SRA, E=ENA, D=DDBJ
                                    "sra": re.findall(
                                        r"[SED]RP\d{6,}|[SED]RX\d{6,}|[SED]RR\d{6,}",
                                        full_content,
                                    ),
                                    # BioProject: NCBI, EBI, DDBJ, CNGB
                                    # PRJNA=NCBI, PRJEB=EBI, PRJDA/PRJDB=DDBJ, PRJCA=CNGB
                                    "bioproject": re.findall(
                                        r"PRJ[A-Z]{2}\d+", full_content
                                    ),
                                    # BioSample: NCBI (SAMN), EBI (SAME), DDBJ (SAMD)
                                    "biosample": re.findall(r"SAM[A-Z]+\d+", full_content),
                                    # ArrayExpress (EBI) - microarray/RNA-seq
                                    "arrayexpress": re.findall(
                                        r"E-[A-Z]+-\d+", full_content
                                    ),
                                    # CNGB/NGDC: Chinese National GeneBank Database
                                    # CRA=raw reads, CNP=project, CRX=experiment, CRR=run
                                    "cngb": re.findall(
                                        r"CRA\d{6,}|CNP\d{7,}|CRX\d{6,}|CRR\d{6,}",
                                        full_content,
                                    ),
                                    # EGA: European Genome-phenome Archive (controlled access)
                                    "ega": re.findall(r"EGAS\d{11}", full_content),
                                    # dbGaP: Database of Genotypes and Phenotypes (controlled)
                                    "dbgap": re.findall(r"phs\d{6}", full_content, re.I),
                                    # Zenodo: General-purpose repository
                                    "zenodo": re.findall(
                                        r"10\.5281/zenodo\.\d+", full_content
                                    ),
                                    # Figshare: General-purpose repository
                                    "figshare": re.findall(
                                        r"10\.6084/m9\.figshare\.\d+", full_content
                                    ),
                                }

                            for key in identifiers_found:
                                identifiers_found[key] = list(set(identifiers_found[key]))

                            # FIX: Collect identifiers in-memory, merge with NCBI identifiers
                            # Will be written once at the end
                            for key, values in identifiers_found.items():
                                if values:
                                    if key in all_extracted_identifiers:
                                        # Merge and deduplicate
                                        all_extracted_identifiers[key] = list(
                                            set(all_extracted_identifiers[key] + values)
                                        )
                                    else:
                                        all_extracted_identifiers[key] = list(values)

                            extracted_data["identifiers"] = identifiers_found

                            total_ids = sum(len(v) for v in identifiers_found.values())
                            if total_ids > 0:
                                response_parts.append(
                                    f"✓ Found {total_ids} dataset identifiers:"
                                )
                                for id_type, id_list in identifiers_found.items():
                                    if id_list:
                                        response_parts.append(
                                            f"  - {id_type.upper()}: {', '.join(id_list[:5])}"
                                        )
                                        if len(id_list) > 5:
                                            response_parts.append(
                                                f"    (+{len(id_list) - 5} more)"
                                            )
                            else:
                                response_parts.append(
                                    "⚠ No dataset identifiers found in publication"
                                )
                        else:
                            response_parts.append(
                                "⚠ No identifier or URL available for identifier extraction"
                            )
                    except Exception as e:  # pragma: no cover - provider errors
                        response_parts.append(f"✗ Identifier extraction failed: {str(e)}")

            # Persist extracted data to workspace
            try:
                from pathlib import Path

                metadata_dir = data_manager.workspace_path / "metadata"
                metadata_dir.mkdir(parents=True, exist_ok=True)

                if extracted_data.get("metadata_extracted") and content_result:
                    metadata_file = metadata_dir / f"{entry_id}_metadata.json"
                    metadata_content = {
                        "content": content_result.get("content", ""),
                        "summary": content_result.get("summary"),
                        "source": entry.pmid or entry.doi or entry.pmc_id,
                        "authors": entry.authors,
                        "year": entry.year,
                        "journal": entry.journal,
                        "extracted_at": datetime.now().isoformat(),
                        "extraction_type": "metadata",
                    }
                    metadata_file.write_text(json.dumps(metadata_content, indent=2))
                    workspace_keys.append(f"{entry_id}_metadata.json")
                    logger.info("Saved metadata to %s", metadata_file)

                if extracted_data.get("methods_extracted") and "methods_content" in locals():
                    methods_file = metadata_dir / f"{entry_id}_methods.json"
                    methods_data = {
                        "methods_text": methods_content,
                        "methods_dict": methods_dict if "methods_dict" in locals() else {},
                        "source": entry.pmid or entry.doi or entry.pmc_id,
                        "extracted_at": datetime.now().isoformat(),
                        "extraction_type": "methods",
                    }
                    methods_file.write_text(json.dumps(methods_data, indent=2))
                    workspace_keys.append(f"{entry_id}_methods.json")
                    logger.info("Saved methods to %s", methods_file)

                if extracted_data.get("identifiers_extracted") and identifiers_found:
                    identifiers_file = metadata_dir / f"{entry_id}_identifiers.json"
                    identifiers_data = {
                        "identifiers": identifiers_found,
                        "source": entry.pmid or entry.doi or entry.pmc_id,
                        "full_content_length": len(full_content)
                        if "full_content" in locals()
                        else 0,
                        "extracted_at": datetime.now().isoformat(),
                        "extraction_type": "identifiers",
                    }
                    identifiers_file.write_text(json.dumps(identifiers_data, indent=2))
                    workspace_keys.append(f"{entry_id}_identifiers.json")
                    logger.info("Saved identifiers to %s", identifiers_file)

                if workspace_keys:
                    response_parts.append(
                        f"✓ Saved {len(workspace_keys)} files to workspace/metadata/"
                    )

            except Exception as e:  # pragma: no cover - workspace errors
                logger.error("Failed to save extracted data to workspace: %s", e)
                response_parts.append(
                    f"⚠ Warning: Workspace persistence failed: {str(e)}"
                )

            # Aggregate all dataset IDs for easy access
            all_dataset_ids = []
            for db_type in ["geo", "sra", "bioproject", "biosample", "arrayexpress", "cngb"]:
                if db_type in all_extracted_identifiers:
                    all_dataset_ids.extend(all_extracted_identifiers[db_type])
            all_dataset_ids = list(set(all_dataset_ids))  # Deduplicate

            # Determine if entry is ready for metadata assistant handoff
            # Conditions: has identifiers AND has datasets AND has workspace metadata
            is_ready_for_handoff = (
                bool(all_extracted_identifiers) and  # Has extracted identifiers
                bool(all_dataset_ids) and            # Has dataset IDs
                bool(workspace_keys)                  # Has workspace metadata files
            )

            # FIX: Determine final status including paywall and handoff readiness
            if is_paywalled:
                final_status = PublicationStatus.PAYWALLED.value
                handoff_status = None  # Don't set handoff for paywalled
            elif is_ready_for_handoff:
                final_status = PublicationStatus.HANDOFF_READY.value
                from lobster.core.schemas.publication_queue import HandoffStatus
                handoff_status = HandoffStatus.READY_FOR_METADATA
            elif extracted_data:
                final_status = PublicationStatus.COMPLETED.value
                handoff_status = None
            else:
                final_status = PublicationStatus.FAILED.value
                handoff_status = None

            # FIX: SINGLE update_status call with ALL collected data
            # This replaces 4 intermediate calls that caused O(n²) file rewrites
            data_manager.publication_queue.update_status(
                entry_id=entry_id,
                status=final_status,
                processed_by="research_agent",
                workspace_metadata_keys=workspace_keys if workspace_keys else None,
                extracted_identifiers=all_extracted_identifiers if all_extracted_identifiers else None,
                dataset_ids=all_dataset_ids if all_dataset_ids else None,
                handoff_status=handoff_status,
                error=paywall_error,
            )

            data_manager.log_tool_usage(
                tool_name="process_publication_entry",
                parameters={
                    "entry_id": entry_id,
                    "extraction_tasks": extraction_tasks,
                    "tasks": tasks,
                    "final_status": final_status,
                    "extracted_identifiers": identifiers_found
                    if "identifiers" in tasks
                    else None,
                    "title": entry.title or "N/A",
                    "pmid": entry.pmid,
                    "doi": entry.doi,
                },
                description=(
                    f"Processed publication entry {entry_id}: {entry.title or 'N/A'} "
                    f"[{final_status}]"
                ),
            )

            response_parts.extend(
                [
                    "",
                    f"**Final Status**: {final_status.upper()}",
                    "",
                    "**Next Steps**:",
                    "- View extracted content: get_content_from_workspace("
                    + f"identifier='{entry_id}', workspace='publication_queue')",
                    "- Process more entries: process_publication_entry('next_entry_id')",
                ]
            )

            return "\n".join(response_parts)

        except Exception as e:  # pragma: no cover - outer safety net
            logger.error("Failed to process publication entry %s: %s", entry_id, e)
            return (
                "## Error Processing Publication Entry\n\n"
                f"Entry ID: {entry_id}\n\n"
                f"**Error**: {str(e)}\n\n"
                "**Tip**: Ensure the entry exists in the publication queue and try again."
            )

    def process_queue_entries(
        self,
        status_filter: str = "pending",
        max_entries: Optional[int] = None,
        extraction_tasks: str = "metadata,methods,identifiers",
    ) -> str:
        """Process multiple publication queue entries in sequence."""

        queue = getattr(self.data_manager, "publication_queue", None)
        if queue is None:
            return "Error: Publication queue is not initialized in DataManagerV2."

        status_enum = None
        if status_filter:
            try:
                status_enum = PublicationStatus(status_filter.lower())
            except ValueError:
                return (
                    "Error: Invalid status filter '"
                    + status_filter
                    + "'."
                )

        entries = queue.list_entries(status=status_enum)
        if not entries:
            scope = status_filter or "any"
            return f"No publication queue entries found with status '{scope}'."

        entries = sorted(entries, key=lambda e: e.created_at)
        if max_entries:
            entries = entries[: max(0, max_entries)]

        summary = [
            "## Publication Queue Processing",
            "",
            f"**Status Filter**: {status_filter or 'any'}",
            f"**Entries Selected**: {len(entries)}",
            f"**Extraction Tasks**: {extraction_tasks}",
            "",
        ]

        for idx, entry in enumerate(entries, start=1):
            summary.append(f"### Entry {idx}: {entry.title or entry.entry_id}")
            summary.append(f"- Entry ID: {entry.entry_id}")
            summary.append(f"- Current Status: {entry.status}")
            result = self.process_entry(
                entry.entry_id, extraction_tasks=extraction_tasks
            )
            summary.append(result)
            summary.append("")

        summary.append(
            f"Processed {len(entries)} publication entries from the queue successfully."
        )
        return "\n".join(summary)
