"""
Abstract base class for biomolecule database providers.

Separate from BasePublicationProvider — biomolecule databases return sequences,
properties, and activity data, not publications with titles/abstracts/DOIs.
"""

import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class BiomoleculeSource(Enum):
    """Supported biomolecule databases."""

    DBAASP = "dbaasp"
    IEDB = "iedb"
    PEPTIPEDIA = "peptipedia"
    PEPTIDE_ATLAS = "peptide_atlas"


class PeptideMetadata(BaseModel):
    """Peptide-specific metadata returned by biomolecule providers."""

    accession: str
    sequence: str
    length: int
    source_db: str
    activities: List[str] = []
    target_organisms: List[str] = []
    molecular_weight: Optional[float] = None
    charge: Optional[float] = None
    source_protein: Optional[str] = None
    source_url: Optional[str] = None


class BaseBiomoleculeProvider(ABC):
    """Base class for biomolecule database providers (peptides, metabolites, etc.)."""

    @property
    @abstractmethod
    def source(self) -> BiomoleculeSource:
        """Return the biomolecule source this provider handles."""
        pass

    @abstractmethod
    def search_sequences(
        self, query: str, max_results: int = 20
    ) -> List[PeptideMetadata]:
        """Search for sequences matching query."""
        pass

    @abstractmethod
    def get_entry(self, accession: str) -> Optional[PeptideMetadata]:
        """Get a single entry by accession."""
        pass

    def get_supported_capabilities(self) -> Dict[str, bool]:
        """Return capabilities supported by this provider."""
        return {
            "search_sequences": True,
            "get_entry": True,
            "search_by_activity": False,
            "search_by_target": False,
            "batch_download": False,
        }

    @property
    def priority(self) -> int:
        """Lower = higher priority. Default 50 (medium)."""
        return 50

    def _make_request(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        max_retry: int = 3,
        sleep_time: float = 0.5,
        timeout: int = 30,
    ) -> Optional[Any]:
        """Make HTTP API request with retry and exponential backoff.

        Subclasses can override max_retry/sleep_time/timeout via their config,
        or call super() with custom values.
        """
        if params:
            url = f"{url}?{urllib.parse.urlencode(params)}"

        source_name = self.source.value.upper()
        for attempt in range(max_retry):
            try:
                req = urllib.request.Request(
                    url, headers={"Accept": "application/json"}
                )
                with urllib.request.urlopen(req, timeout=timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    logger.debug(f"{source_name} resource not found: {url}")
                    return None
                if e.code == 429:
                    wait = sleep_time * (2**attempt)
                    logger.warning(f"{source_name} rate limit, waiting {wait:.1f}s")
                    time.sleep(wait)
                    continue
                logger.warning(f"{source_name} HTTP {e.code} on attempt {attempt + 1}")
            except json.JSONDecodeError:
                logger.warning(
                    f"{source_name} returned non-JSON response on attempt {attempt + 1}"
                )
            except (urllib.error.URLError, TimeoutError) as e:
                logger.warning(
                    f"{source_name} request error on attempt {attempt + 1}: {e}"
                )
                time.sleep(sleep_time * (attempt + 1))

        logger.error(f"{source_name} request failed after {max_retry} attempts: {url}")
        return None

    def format_results(self, peptides: List[PeptideMetadata], query: str) -> str:
        """Format search results as readable text."""
        if not peptides:
            return f"No peptides found for query: {query}"

        lines = [
            f"## {self.source.value.upper()} Search Results",
            f"**Query**: {query}",
            f"**Results**: {len(peptides)} peptides found\n",
        ]

        for i, p in enumerate(peptides, 1):
            lines.append(f"### {i}. {p.accession}")
            lines.append(f"**Sequence**: `{p.sequence}`")
            lines.append(f"**Length**: {p.length} aa")
            if p.molecular_weight:
                lines.append(f"**MW**: {p.molecular_weight:.1f} Da")
            if p.charge is not None:
                lines.append(f"**Charge**: {p.charge:+.1f}")
            if p.activities:
                lines.append(f"**Activities**: {', '.join(p.activities)}")
            if p.target_organisms:
                lines.append(f"**Targets**: {', '.join(p.target_organisms[:5])}")
            if p.source_protein:
                lines.append(f"**Source protein**: {p.source_protein}")
            if p.source_url:
                lines.append(f"**URL**: {p.source_url}")
            lines.append("")

        return "\n".join(lines)
