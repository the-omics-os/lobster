"""
Publication Resolver for automatic PMID/DOI to PDF URL resolution.

This module provides intelligent PDF access resolution with a tiered waterfall strategy:
1. PubMed Central (PMC) - Free full text via NCBI E-utilities
2. bioRxiv/medRxiv - Preprint servers with direct PDF access
3. Publisher Direct - When open access flag is set
4. Helpful suggestions when paywalled

This eliminates the #1 user pain point: manually finding PDF URLs.
"""

import re
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

import requests

from lobster.utils.logger import get_logger

logger = get_logger(__name__)


class PublicationResolutionResult:
    """Result of publication resolution attempt."""

    def __init__(
        self,
        identifier: str,
        pdf_url: Optional[str] = None,
        source: str = "unknown",
        access_type: str = "unknown",
        alternative_urls: Optional[List[str]] = None,
        suggestions: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize resolution result.

        Args:
            identifier: Original identifier (PMID/DOI)
            pdf_url: Direct PDF URL if found
            source: Resolution source ("pmc" | "biorxiv" | "medrxiv" | "publisher" | "paywalled")
            access_type: Access type ("open_access" | "paywalled" | "preprint")
            alternative_urls: List of alternative access URLs
            suggestions: Human-readable guidance for accessing paper
            metadata: Additional metadata about the resolution
        """
        self.identifier = identifier
        self.pdf_url = pdf_url
        self.source = source
        self.access_type = access_type
        self.alternative_urls = alternative_urls or []
        self.suggestions = suggestions or ""
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "identifier": self.identifier,
            "pdf_url": self.pdf_url,
            "source": self.source,
            "access_type": self.access_type,
            "alternative_urls": self.alternative_urls,
            "suggestions": self.suggestions,
            "metadata": self.metadata,
        }

    def is_accessible(self) -> bool:
        """Check if PDF is accessible."""
        return self.pdf_url is not None and self.access_type != "paywalled"


class PublicationResolver:
    """
    Intelligent publication resolver with tiered waterfall strategy.

    Resolution priority:
    1. PubMed Central (PMC) - Best success rate for biomedical papers
    2. bioRxiv/medRxiv - Preprint servers with open access
    3. Publisher Direct - When open access flag is set
    4. Paywalled - Return helpful suggestions
    """

    def __init__(self, timeout: int = 30):
        """
        Initialize resolver.

        Args:
            timeout: Request timeout in seconds (default: 30)
        """
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "Lobster AI Research Tool/1.0 (mailto:support@omics-os.com)"}
        )
        logger.info("Initialized PublicationResolver")

    def resolve(self, identifier: str) -> PublicationResolutionResult:
        """
        Resolve identifier to PDF URL using tiered waterfall strategy.

        Args:
            identifier: PMID, DOI, or publication identifier

        Returns:
            PublicationResolutionResult with access information

        Examples:
            >>> resolver = PublicationResolver()
            >>> result = resolver.resolve("PMID:12345678")
            >>> if result.is_accessible():
            ...     print(f"PDF available at: {result.pdf_url}")
            >>> else:
            ...     print(f"Suggestions: {result.suggestions}")
        """
        logger.info(f"Resolving identifier: {identifier}")

        # Normalize identifier
        identifier = identifier.strip()
        pmid, doi = self._parse_identifier(identifier)

        # Strategy 1: Try PubMed Central (PMC) first
        if pmid:
            result = self._resolve_via_pmc(pmid)
            if result.is_accessible():
                logger.info(f"Resolved via PMC: {result.pdf_url}")
                return result

        # Strategy 2: Try bioRxiv/medRxiv preprints
        if doi:
            result = self._resolve_via_preprint_servers(doi)
            if result.is_accessible():
                logger.info(f"Resolved via preprint server: {result.pdf_url}")
                return result

        # Strategy 3: Try publisher direct (limited support)
        if doi:
            result = self._resolve_via_publisher(doi)
            if result.is_accessible():
                logger.info(f"Resolved via publisher: {result.pdf_url}")
                return result

        # Strategy 4: Generate helpful suggestions for paywalled papers
        logger.info(f"Paper appears paywalled: {identifier}")
        return self._generate_access_suggestions(identifier, pmid, doi)

    def _parse_identifier(self, identifier: str) -> tuple[Optional[str], Optional[str]]:
        """
        Parse identifier to extract PMID and/or DOI.

        Args:
            identifier: Input identifier

        Returns:
            Tuple of (pmid, doi)
        """
        pmid = None
        doi = None

        # Check for PMID
        if identifier.upper().startswith("PMID:"):
            pmid = identifier[5:].strip()
        elif identifier.isdigit() and len(identifier) <= 8:
            pmid = identifier

        # Check for DOI
        if identifier.startswith("10."):
            doi = identifier
        elif "doi.org/" in identifier.lower():
            doi = identifier.split("doi.org/")[-1]

        return pmid, doi

    def _resolve_via_pmc(self, pmid: str) -> PublicationResolutionResult:
        """
        Resolve PMID to PDF via PubMed Central.

        Uses NCBI E-utilities API to check for free full text in PMC.

        Args:
            pmid: PubMed ID

        Returns:
            PublicationResolutionResult
        """
        logger.info(f"Checking PMC for PMID: {pmid}")

        try:
            # Step 1: Use elink to find PMC ID
            elink_url = (
                f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi"
                f"?dbfrom=pubmed&db=pmc&id={pmid}&linkname=pubmed_pmc_refs&retmode=json"
            )

            response = self.session.get(elink_url, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()

            # Extract PMC ID if available
            pmc_id = None
            try:
                linksets = data.get("linksets", [])
                if linksets and len(linksets) > 0:
                    linksetdbs = linksets[0].get("linksetdbs", [])
                    if linksetdbs and len(linksetdbs) > 0:
                        links = linksetdbs[0].get("links", [])
                        if links and len(links) > 0:
                            pmc_id = links[0]
            except (KeyError, IndexError, TypeError) as e:
                logger.debug(f"No PMC link found for PMID {pmid}: {e}")

            if not pmc_id:
                return PublicationResolutionResult(
                    identifier=f"PMID:{pmid}",
                    source="pmc",
                    access_type="not_in_pmc",
                )

            # Step 2: Construct PMC HTML article URL
            # PMC HTML articles have better structure extraction than PDF directory
            # Docling auto-detects format and handles both HTML and PDF intelligently
            pdf_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmc_id}/"

            logger.info(f"Found PMC article: PMC{pmc_id}")

            return PublicationResolutionResult(
                identifier=f"PMID:{pmid}",
                pdf_url=pdf_url,
                source="pmc",
                access_type="open_access",
                alternative_urls=[
                    f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmc_id}/pdf/"  # PDF directory as fallback
                ],
                metadata={"pmc_id": f"PMC{pmc_id}"},
            )

        except requests.exceptions.RequestException as e:
            logger.error(f"Error querying PMC for PMID {pmid}: {e}")
            return PublicationResolutionResult(
                identifier=f"PMID:{pmid}", source="pmc", access_type="error"
            )
        except Exception as e:
            logger.error(f"Unexpected error in PMC resolution: {e}")
            return PublicationResolutionResult(
                identifier=f"PMID:{pmid}", source="pmc", access_type="error"
            )

    def _resolve_via_preprint_servers(self, doi: str) -> PublicationResolutionResult:
        """
        Resolve DOI to PDF via bioRxiv/medRxiv preprint servers.

        Args:
            doi: Digital Object Identifier

        Returns:
            PublicationResolutionResult
        """
        logger.info(f"Checking preprint servers for DOI: {doi}")

        # Check if DOI is from bioRxiv or medRxiv
        if "biorxiv.org" in doi.lower() or doi.startswith("10.1101/"):
            # bioRxiv pattern: https://www.biorxiv.org/content/10.1101/{id}.full.pdf
            pdf_url = f"https://www.biorxiv.org/content/{doi}.full.pdf"

            return PublicationResolutionResult(
                identifier=doi,
                pdf_url=pdf_url,
                source="biorxiv",
                access_type="preprint",
                alternative_urls=[f"https://www.biorxiv.org/content/{doi}"],
                metadata={"server": "biorxiv"},
            )

        elif "medrxiv.org" in doi.lower():
            # medRxiv pattern: https://www.medrxiv.org/content/10.1101/{id}.full.pdf
            pdf_url = f"https://www.medrxiv.org/content/{doi}.full.pdf"

            return PublicationResolutionResult(
                identifier=doi,
                pdf_url=pdf_url,
                source="medrxiv",
                access_type="preprint",
                alternative_urls=[f"https://www.medrxiv.org/content/{doi}"],
                metadata={"server": "medrxiv"},
            )

        # Not a preprint server DOI
        return PublicationResolutionResult(
            identifier=doi, source="preprint", access_type="not_preprint"
        )

    def _resolve_via_publisher(self, doi: str) -> PublicationResolutionResult:
        """
        Resolve DOI to PDF via publisher (limited support for open access).

        This is a fallback strategy with limited success rate.

        Args:
            doi: Digital Object Identifier

        Returns:
            PublicationResolutionResult
        """
        logger.info(f"Checking publisher for DOI: {doi}")

        try:
            # Use CrossRef API to get metadata
            crossref_url = f"https://api.crossref.org/works/{doi}"
            response = self.session.get(crossref_url, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()

            message = data.get("message", {})
            is_open_access = False

            # Check for open access indicators
            license_info = message.get("license", [])
            for license_item in license_info:
                if "open-access" in str(license_item).lower():
                    is_open_access = True
                    break

            if not is_open_access:
                link = message.get("link", [])
                for link_item in link:
                    if link_item.get(
                        "content-type"
                    ) == "application/pdf" and "unixref.org" not in link_item.get(
                        "URL", ""
                    ):
                        # Found a direct PDF link
                        pdf_url = link_item.get("URL")
                        return PublicationResolutionResult(
                            identifier=doi,
                            pdf_url=pdf_url,
                            source="publisher",
                            access_type="open_access",
                            metadata={"publisher": message.get("publisher")},
                        )

            # No direct PDF found
            return PublicationResolutionResult(
                identifier=doi,
                source="publisher",
                access_type="not_open_access",
                metadata={"publisher": message.get("publisher")},
            )

        except requests.exceptions.RequestException as e:
            logger.error(f"Error querying CrossRef for DOI {doi}: {e}")
            return PublicationResolutionResult(
                identifier=doi, source="publisher", access_type="error"
            )
        except Exception as e:
            logger.error(f"Unexpected error in publisher resolution: {e}")
            return PublicationResolutionResult(
                identifier=doi, source="publisher", access_type="error"
            )

    def _generate_access_suggestions(
        self, identifier: str, pmid: Optional[str], doi: Optional[str]
    ) -> PublicationResolutionResult:
        """
        Generate helpful suggestions when paper is paywalled.

        Args:
            identifier: Original identifier
            pmid: PMID if available
            doi: DOI if available

        Returns:
            PublicationResolutionResult with suggestions
        """
        suggestions = []
        alternative_urls = []

        suggestions.append("## Alternative Access Options\n")

        # Suggestion 1: PMC Accepted Manuscript
        if pmid:
            suggestions.append(
                f"1. **PubMed Central Accepted Manuscript**: Check if an accepted manuscript version is available:\n"
                f"   - https://www.ncbi.nlm.nih.gov/pmc/?term={pmid}\n"
            )
            alternative_urls.append(f"https://www.ncbi.nlm.nih.gov/pmc/?term={pmid}")

        # Suggestion 2: bioRxiv/medRxiv search
        if doi or pmid:
            search_term = doi if doi else pmid
            suggestions.append(
                f"2. **Preprint Servers**: Check for preprints on bioRxiv or medRxiv:\n"
                f"   - bioRxiv: https://www.biorxiv.org/search/{search_term}\n"
                f"   - medRxiv: https://www.medrxiv.org/search/{search_term}\n"
            )
            alternative_urls.append(f"https://www.biorxiv.org/search/{search_term}")

        # Suggestion 3: Institutional Access
        suggestions.append(
            "3. **Institutional Access**: If you're affiliated with a university, try:\n"
            "   - Accessing through your institution's library proxy\n"
            "   - Using VPN to connect to institutional network\n"
            "   - Requesting through interlibrary loan\n"
        )

        # Suggestion 4: Author Contact
        if pmid:
            suggestions.append(
                f"4. **Contact Authors**: You can:\n"
                f"   - Email the corresponding author to request a PDF\n"
                f"   - Check author profiles on ResearchGate or Academia.edu\n"
                f"   - PubMed author info: https://pubmed.ncbi.nlm.nih.gov/{pmid}/\n"
            )
            alternative_urls.append(f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/")

        # Suggestion 5: Unpaywall
        if doi:
            suggestions.append(
                f"5. **Unpaywall Service**: Check for legal open access versions:\n"
                f"   - https://unpaywall.org/{doi}\n"
            )
            alternative_urls.append(f"https://unpaywall.org/{doi}")

        suggestions.append(
            "\n💡 **Tip**: Many publishers allow authors to share accepted manuscripts. "
            "Contacting the corresponding author is often successful!"
        )

        return PublicationResolutionResult(
            identifier=identifier,
            pdf_url=None,
            source="paywalled",
            access_type="paywalled",
            alternative_urls=alternative_urls,
            suggestions="\n".join(suggestions),
            metadata={"pmid": pmid, "doi": doi},
        )

    def batch_resolve(
        self, identifiers: List[str], max_batch: int = 5
    ) -> List[PublicationResolutionResult]:
        """
        Resolve multiple identifiers sequentially (conservative approach).

        Args:
            identifiers: List of PMIDs/DOIs to resolve
            max_batch: Maximum batch size (default: 5)

        Returns:
            List of PublicationResolutionResult objects

        Examples:
            >>> resolver = PublicationResolver()
            >>> identifiers = ["PMID:12345678", "10.1038/s41586-021-12345-6"]
            >>> results = resolver.batch_resolve(identifiers)
            >>> for result in results:
            ...     if result.is_accessible():
            ...         print(f"✅ {result.identifier}: {result.pdf_url}")
            ...     else:
            ...         print(f"❌ {result.identifier}: Paywalled")
        """
        logger.info(f"Batch resolving {len(identifiers)} identifiers")

        # Limit batch size
        if len(identifiers) > max_batch:
            logger.warning(
                f"Batch size {len(identifiers)} exceeds max {max_batch}, truncating"
            )
            identifiers = identifiers[:max_batch]

        results = []
        for i, identifier in enumerate(identifiers, 1):
            logger.info(f"Processing {i}/{len(identifiers)}: {identifier}")
            try:
                result = self.resolve(identifier)
                results.append(result)
            except Exception as e:
                logger.error(f"Error resolving {identifier}: {e}")
                results.append(
                    PublicationResolutionResult(
                        identifier=identifier,
                        source="error",
                        access_type="error",
                        suggestions=f"Error during resolution: {str(e)}",
                    )
                )

        logger.info(
            f"Batch resolution complete: {sum(1 for r in results if r.is_accessible())}/{len(results)} accessible"
        )
        return results
