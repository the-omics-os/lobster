"""
UniProt REST API service for protein annotation queries.

Stateless HTTP client for https://rest.uniprot.org providing:
- Single protein lookup by accession or entry name
- Keyword/field search across UniProtKB
- Cross-database ID mapping (async job-based)

All methods return plain dicts; formatting is handled by tool layers.
Uses in-memory LRU cache (200 entries) per service instance.
No new dependencies — uses requests (already in pyproject.toml).
"""

import time
from functools import lru_cache
from typing import Any, Dict, List, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from lobster.utils.logger import get_logger

logger = get_logger(__name__)

# =============================================================================
# Exceptions
# =============================================================================


class UniProtServiceError(Exception):
    """Base exception for UniProt service errors."""

    pass


class UniProtNotFoundError(UniProtServiceError):
    """Raised when a protein accession is not found (404)."""

    pass


class UniProtRateLimitError(UniProtServiceError):
    """Raised when rate limit is exceeded (429)."""

    pass


# =============================================================================
# Service
# =============================================================================

BASE_URL = "https://rest.uniprot.org"
ID_MAPPING_URL = f"{BASE_URL}/idmapping"


class UniProtService:
    """
    Stateless HTTP client for the UniProt REST API.

    Thread-safe. Each instance maintains its own requests.Session with
    retry logic and an LRU cache for protein lookups.

    Usage:
        service = UniProtService()
        protein = service.get_protein("P04637")
        results = service.search_proteins("TP53 human")
        mappings = service.map_ids("Gene_Name", "UniProtKB", ["TP53", "BRCA1"])
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
                "Accept": "application/json",
                "User-Agent": "lobster-ai/1.0 (https://lobsterbio.com)",
            }
        )
        return session

    # =========================================================================
    # Public API
    # =========================================================================

    def get_protein(self, accession: str) -> Dict[str, Any]:
        """
        Fetch a single protein entry by UniProt accession or entry name.

        Args:
            accession: UniProt accession (e.g. "P04637") or entry name (e.g. "P53_HUMAN")

        Returns:
            Dict with protein metadata including:
            - primaryAccession, uniProtkbId, proteinDescription
            - organism, genes, sequence, features, references, etc.

        Raises:
            UniProtNotFoundError: If accession not found
            UniProtServiceError: On API errors
        """
        return self._get_protein_cached(accession.strip().upper())

    @lru_cache(maxsize=200)
    def _get_protein_cached(self, accession: str) -> Dict[str, Any]:
        """Cached protein lookup."""
        url = f"{BASE_URL}/uniprotkb/{accession}"
        response = self._request("GET", url)
        return response

    def search_proteins(
        self,
        query: str,
        max_results: int = 10,
        organism: Optional[str] = None,
        reviewed: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Search UniProtKB by keyword, gene name, or field query.

        Args:
            query: Search query (e.g. "TP53", "kinase AND organism_id:9606")
            max_results: Maximum results to return (default 10, max 500)
            organism: Optional organism filter (e.g. "9606" for human)
            reviewed: If True, only Swiss-Prot (reviewed); if False, only TrEMBL

        Returns:
            Dict with 'results' list of protein entries

        Raises:
            UniProtServiceError: On API errors
        """
        max_results = min(max_results, 500)

        # Build query with optional filters
        parts = [query]
        if organism:
            parts.append(f"organism_id:{organism}")
        if reviewed is not None:
            parts.append(f"reviewed:{'true' if reviewed else 'false'}")

        full_query = " AND ".join(parts) if len(parts) > 1 else query

        params = {
            "query": full_query,
            "size": max_results,
            "format": "json",
        }
        url = f"{BASE_URL}/uniprotkb/search"
        response = self._request("GET", url, params=params)
        return response

    def map_ids(
        self,
        from_db: str,
        to_db: str,
        ids: List[str],
        poll_interval: float = 2.0,
        max_polls: int = 30,
    ) -> Dict[str, Any]:
        """
        Map identifiers between databases using UniProt ID Mapping.

        This is an asynchronous operation: submit job → poll → fetch results.

        Args:
            from_db: Source database (e.g. "Gene_Name", "Ensembl", "RefSeq_Protein")
            to_db: Target database (e.g. "UniProtKB_AC-ID", "UniProtKB", "PDB")
            ids: List of identifiers to map
            poll_interval: Seconds between poll requests (default 2.0)
            max_polls: Maximum number of poll attempts (default 30)

        Returns:
            Dict with 'results' list of mappings, each containing 'from' and 'to' keys

        Raises:
            UniProtServiceError: On job failure or timeout

        Common from/to database values:
            Gene_Name, UniProtKB_AC-ID, UniProtKB, Ensembl, Ensembl_Genomes,
            RefSeq_Protein, PDB, GeneID, EMBL-GenBank-DDBJ
        """
        if not ids:
            return {"results": []}

        # Step 1: Submit job
        submit_url = f"{ID_MAPPING_URL}/run"
        submit_data = {
            "from": from_db,
            "to": to_db,
            "ids": ",".join(ids),
        }
        submit_response = self._session.post(
            submit_url, data=submit_data, timeout=self._timeout
        )
        self._check_response(submit_response)
        job_id = submit_response.json().get("jobId")
        if not job_id:
            raise UniProtServiceError("No jobId returned from ID mapping submission")

        logger.debug(f"UniProt ID mapping job submitted: {job_id}")

        # Step 2: Poll for completion
        status_url = f"{ID_MAPPING_URL}/status/{job_id}"
        for attempt in range(max_polls):
            status_response = self._session.get(status_url, timeout=self._timeout)

            if status_response.status_code == 303:
                # Job complete — follow redirect
                results_url = status_response.headers.get("Location", "")
                if not results_url:
                    results_url = f"{ID_MAPPING_URL}/results/{job_id}"
                break

            status_data = status_response.json()
            if "jobStatus" in status_data:
                job_status = status_data["jobStatus"]
                if job_status == "RUNNING":
                    time.sleep(poll_interval)
                    continue
                elif job_status == "FINISHED":
                    results_url = f"{ID_MAPPING_URL}/results/{job_id}"
                    break
                else:
                    raise UniProtServiceError(
                        f"ID mapping job failed with status: {job_status}"
                    )
            elif "results" in status_data:
                # Some responses include results directly in status
                return status_data
            else:
                time.sleep(poll_interval)
        else:
            raise UniProtServiceError(
                f"ID mapping job timed out after {max_polls * poll_interval}s"
            )

        # Step 3: Fetch results
        results_response = self._request("GET", results_url, params={"format": "json"})
        return results_response

    # =========================================================================
    # Internal
    # =========================================================================

    def _request(
        self,
        method: str,
        url: str,
        params: Optional[Dict] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Make HTTP request with error handling."""
        try:
            response = self._session.request(
                method, url, params=params, timeout=self._timeout, **kwargs
            )
            self._check_response(response)
            return response.json()
        except requests.exceptions.JSONDecodeError as e:
            raise UniProtServiceError(f"Invalid JSON response from {url}: {e}")
        except requests.exceptions.ConnectionError as e:
            raise UniProtServiceError(f"Connection error to UniProt API: {e}")
        except requests.exceptions.Timeout as e:
            raise UniProtServiceError(f"Request to UniProt API timed out: {e}")

    @staticmethod
    def _check_response(response: requests.Response) -> None:
        """Check response status and raise appropriate errors."""
        if response.status_code == 200 or response.status_code == 303:
            return
        if response.status_code == 404:
            raise UniProtNotFoundError(
                f"Not found: {response.url}"
            )
        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After", "unknown")
            raise UniProtRateLimitError(
                f"Rate limit exceeded. Retry after {retry_after}s"
            )
        if response.status_code == 400:
            try:
                detail = response.json()
            except Exception:
                detail = response.text
            raise UniProtServiceError(
                f"Bad request ({response.status_code}): {detail}"
            )
        response.raise_for_status()
