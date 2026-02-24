"""
Ensembl REST API service for genome annotation queries.

Stateless HTTP client for https://rest.ensembl.org providing:
- Gene/transcript lookup by Ensembl ID or gene symbol
- Variant Effect Predictor (VEP) consequence prediction
- Sequence retrieval (genomic, cDNA, CDS, protein)
- Cross-database references (xrefs to UniProt, HGNC, RefSeq, OMIM)

All methods return plain dicts; formatting is handled by tool layers.
Uses in-memory LRU cache (200 entries) per service instance.
Respects Ensembl rate limits via X-RateLimit-Remaining header.
No new dependencies — uses requests (already in pyproject.toml).
"""

import time
from functools import lru_cache
from typing import Any, Dict, List, Optional, Union

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from lobster.utils.logger import get_logger

logger = get_logger(__name__)

# =============================================================================
# Exceptions
# =============================================================================


class EnsemblServiceError(Exception):
    """Base exception for Ensembl service errors."""

    pass


class EnsemblNotFoundError(EnsemblServiceError):
    """Raised when an identifier is not found (404)."""

    pass


class EnsemblRateLimitError(EnsemblServiceError):
    """Raised when rate limit is exceeded (429)."""

    pass


# =============================================================================
# Constants
# =============================================================================

BASE_URL = "https://rest.ensembl.org"

# Common species aliases → Ensembl species names
SPECIES_ALIASES = {
    "human": "homo_sapiens",
    "mouse": "mus_musculus",
    "rat": "rattus_norvegicus",
    "zebrafish": "danio_rerio",
    "fly": "drosophila_melanogaster",
    "worm": "caenorhabditis_elegans",
    "yeast": "saccharomyces_cerevisiae",
    "chicken": "gallus_gallus",
    "pig": "sus_scrofa",
    "dog": "canis_lupus_familiaris",
    # NCBI taxonomy IDs
    "9606": "homo_sapiens",
    "10090": "mus_musculus",
    "10116": "rattus_norvegicus",
    "7955": "danio_rerio",
    "7227": "drosophila_melanogaster",
    "6239": "caenorhabditis_elegans",
    "4932": "saccharomyces_cerevisiae",
    "9031": "gallus_gallus",
    "9823": "sus_scrofa",
    "9615": "canis_lupus_familiaris",
}


# =============================================================================
# Service
# =============================================================================


class EnsemblService:
    """
    Stateless HTTP client for the Ensembl REST API.

    Thread-safe. Each instance maintains its own requests.Session with
    retry logic, an LRU cache for lookups, and rate limit awareness.

    Usage:
        service = EnsemblService()
        gene = service.lookup_gene("ENSG00000141510")
        gene = service.lookup_gene("TP53", species="human")
        vep = service.get_variant_consequences("9:g.22125503G>C", species="human")
        seq = service.get_sequence("ENSG00000141510", seq_type="cdna")
        xrefs = service.get_xrefs("ENSG00000141510")
    """

    def __init__(self, timeout: int = 30):
        self._timeout = timeout
        self._session = self._create_session()

    @staticmethod
    def _create_session() -> requests.Session:
        """Create session with retry logic."""
        session = requests.Session()
        retry = Retry(
            total=3,
            backoff_factor=1.0,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"],
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("https://", adapter)
        session.headers.update(
            {
                "Content-Type": "application/json",
                "User-Agent": "lobster-ai/1.0 (https://lobsterbio.com)",
            }
        )
        return session

    # =========================================================================
    # Public API
    # =========================================================================

    def lookup_gene(
        self,
        identifier: str,
        species: str = "homo_sapiens",
        expand: bool = False,
    ) -> Dict[str, Any]:
        """
        Look up a gene or transcript by Ensembl ID or gene symbol.

        Auto-detects whether identifier is an Ensembl stable ID (ENSG/ENST/ENSP)
        or a gene symbol (e.g. TP53) and uses the appropriate endpoint.

        Args:
            identifier: Ensembl stable ID or gene symbol
            species: Species name, alias, or NCBI taxonomy ID (default "homo_sapiens")
            expand: If True, include transcripts/exons in response

        Returns:
            Dict with gene/transcript metadata including:
            - id, display_name, description, biotype, species
            - assembly_name, seq_region_name, start, end, strand
            - If expand=True: Transcript[] with exon details

        Raises:
            EnsemblNotFoundError: If identifier not found
            EnsemblServiceError: On API errors
        """
        species = self._normalize_species(species)
        identifier = self._strip_version(identifier.strip())

        return self._lookup_gene_cached(identifier, species, expand)

    @lru_cache(maxsize=200)
    def _lookup_gene_cached(
        self, identifier: str, species: str, expand: bool
    ) -> Dict[str, Any]:
        """Cached gene lookup."""
        # Auto-detect: Ensembl stable IDs start with ENS
        if identifier.upper().startswith("ENS"):
            url = f"{BASE_URL}/lookup/id/{identifier}"
            params = {"expand": "1"} if expand else {}
        else:
            # Treat as gene symbol
            url = f"{BASE_URL}/lookup/symbol/{species}/{identifier}"
            params = {"expand": "1"} if expand else {}

        return self._request("GET", url, params=params)

    def get_variant_consequences(
        self,
        notation: str,
        species: str = "homo_sapiens",
        notation_type: str = "hgvs",
    ) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Predict variant consequences using Ensembl VEP.

        Args:
            notation: Variant notation. Supported formats:
                - HGVS: "9:g.22125503G>C" or "ENST00000269305.9:c.817C>T"
                - Region: "9:22125503-22125503:1/C"
                - rsID: "rs1042522"
            species: Species name or alias (default "homo_sapiens")
            notation_type: One of "hgvs" (default), "region", or "id" (for rsIDs)

        Returns:
            List of consequence predictions, each containing:
            - most_severe_consequence, transcript_consequences[]
            - colocated_variants, regulatory_feature_consequences

        Raises:
            EnsemblNotFoundError: If variant not found
            EnsemblServiceError: On API errors
        """
        species = self._normalize_species(species)
        notation = notation.strip()

        if notation_type == "hgvs":
            url = f"{BASE_URL}/vep/{species}/hgvs/{notation}"
        elif notation_type == "region":
            url = f"{BASE_URL}/vep/{species}/region/{notation}"
        elif notation_type == "id":
            url = f"{BASE_URL}/vep/{species}/id/{notation}"
        else:
            raise EnsemblServiceError(
                f"Invalid notation_type: {notation_type}. Use 'hgvs', 'region', or 'id'."
            )

        return self._request("GET", url)

    def get_sequence(
        self,
        ensembl_id: str,
        seq_type: str = "genomic",
    ) -> Dict[str, Any]:
        """
        Retrieve nucleotide or protein sequence for an Ensembl ID.

        Args:
            ensembl_id: Ensembl stable ID (gene, transcript, or protein)
            seq_type: Sequence type — "genomic", "cdna", "cds", or "protein"

        Returns:
            Dict with 'id', 'seq' (the sequence string), 'molecule', 'desc'

        Raises:
            EnsemblNotFoundError: If ID not found
            EnsemblServiceError: On API errors
        """
        ensembl_id = self._strip_version(ensembl_id.strip())
        valid_types = {"genomic", "cdna", "cds", "protein"}
        if seq_type not in valid_types:
            raise EnsemblServiceError(
                f"Invalid seq_type: {seq_type}. Use one of {valid_types}."
            )

        return self._get_sequence_cached(ensembl_id, seq_type)

    @lru_cache(maxsize=200)
    def _get_sequence_cached(self, ensembl_id: str, seq_type: str) -> Dict[str, Any]:
        """Cached sequence retrieval."""
        url = f"{BASE_URL}/sequence/id/{ensembl_id}"
        params = {"type": seq_type}
        return self._request("GET", url, params=params)

    def get_xrefs(
        self,
        ensembl_id: str,
        external_db: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get cross-database references for an Ensembl ID.

        Args:
            ensembl_id: Ensembl stable ID
            external_db: Optional filter by external database name
                        (e.g. "UniProt/SWISSPROT", "HGNC", "RefSeq_mRNA", "MIM_GENE")

        Returns:
            List of xref dicts, each containing:
            - primary_id, display_id, dbname, description, info_type

        Raises:
            EnsemblNotFoundError: If ID not found
            EnsemblServiceError: On API errors
        """
        ensembl_id = self._strip_version(ensembl_id.strip())
        url = f"{BASE_URL}/xrefs/id/{ensembl_id}"
        params = {}
        if external_db:
            params["external_db"] = external_db

        return self._request("GET", url, params=params)

    # =========================================================================
    # Helpers
    # =========================================================================

    @staticmethod
    def _normalize_species(species: str) -> str:
        """Normalize species name from alias or taxonomy ID to Ensembl format."""
        species = species.strip().lower()
        return SPECIES_ALIASES.get(species, species)

    @staticmethod
    def _strip_version(ensembl_id: str) -> str:
        """Strip version suffix (.N) from Ensembl stable IDs for API compatibility.

        Ensembl REST API endpoints reject versioned IDs (e.g. ENSG00000141510.5)
        with a 400 error. This strips the version while preserving the stable ID.
        """
        if ensembl_id.upper().startswith("ENS") and "." in ensembl_id:
            return ensembl_id.split(".")[0]
        return ensembl_id

    # =========================================================================
    # Internal HTTP
    # =========================================================================

    def _request(
        self,
        method: str,
        url: str,
        params: Optional[Dict] = None,
        **kwargs,
    ) -> Any:
        """Make HTTP request with rate limit awareness and error handling."""
        try:
            response = self._session.request(
                method, url, params=params, timeout=self._timeout, **kwargs
            )
            self._check_rate_limit(response)
            self._check_response(response)
            return response.json()
        except requests.exceptions.JSONDecodeError as e:
            raise EnsemblServiceError(f"Invalid JSON response from {url}: {e}")
        except requests.exceptions.ConnectionError as e:
            raise EnsemblServiceError(f"Connection error to Ensembl API: {e}")
        except requests.exceptions.Timeout as e:
            raise EnsemblServiceError(f"Request to Ensembl API timed out: {e}")

    def _check_rate_limit(self, response: requests.Response) -> None:
        """Check and respect Ensembl rate limits via response headers."""
        remaining = response.headers.get("X-RateLimit-Remaining")
        if remaining is not None:
            try:
                remaining_int = int(remaining)
                if remaining_int <= 1:
                    reset = response.headers.get("X-RateLimit-Reset", "1")
                    sleep_time = min(float(reset), 10.0)
                    logger.debug(
                        f"Ensembl rate limit nearly exhausted ({remaining_int} remaining), "
                        f"sleeping {sleep_time}s"
                    )
                    time.sleep(sleep_time)
            except (ValueError, TypeError):
                pass

    @staticmethod
    def _check_response(response: requests.Response) -> None:
        """Check response status and raise appropriate errors."""
        if response.ok:
            return
        if response.status_code == 404:
            raise EnsemblNotFoundError(f"Not found: {response.url}")
        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After", "unknown")
            raise EnsemblRateLimitError(
                f"Rate limit exceeded. Retry after {retry_after}s"
            )
        if response.status_code == 400:
            try:
                detail = response.json().get("error", response.text)
            except Exception:
                detail = response.text
            raise EnsemblServiceError(f"Bad request ({response.status_code}): {detail}")
        response.raise_for_status()
