"""
IEDB (Immune Epitope Database) provider.

REST API v3: https://query-api.iedb.org
Focus: Immune epitopes (2.2M+ entries) with MHC allele and assay data.
"""

import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from lobster.tools.providers.biomolecule_provider import (
    BaseBiomoleculeProvider,
    BiomoleculeSource,
    PeptideMetadata,
)

logger = logging.getLogger(__name__)


class IEDBProviderConfig(BaseModel):
    """Configuration for IEDB provider."""

    base_url: str = "https://query-api.iedb.org"
    max_retry: int = Field(default=3, ge=1, le=10)
    sleep_time: float = Field(default=0.5, ge=0.1, le=5.0)
    timeout: int = Field(default=30, ge=5, le=120)


class IEDBProvider(BaseBiomoleculeProvider):
    """IEDB immune epitope database provider."""

    def __init__(self, config: Optional[IEDBProviderConfig] = None):
        self.config = config or IEDBProviderConfig()

    @property
    def source(self) -> BiomoleculeSource:
        return BiomoleculeSource.IEDB

    @property
    def priority(self) -> int:
        return 10

    def get_supported_capabilities(self) -> Dict[str, bool]:
        return {
            "search_sequences": True,
            "get_entry": True,
            "search_by_activity": False,
            "search_by_target": True,
            "batch_download": False,
        }

    def search_sequences(
        self, query: str, max_results: int = 20
    ) -> List[PeptideMetadata]:
        """Search IEDB for epitopes matching query."""
        url = f"{self.config.base_url}/epitope_search"
        params = {"linear_sequence": query, "limit": min(max_results, 100)}
        data = self._make_request(url, params)
        if not data:
            return []
        epitopes = data if isinstance(data, list) else data.get("results", [])
        return [self._parse_entry(e) for e in epitopes[:max_results]]

    def get_entry(self, accession: str) -> Optional[PeptideMetadata]:
        """Get a single IEDB epitope by ID."""
        epitope_id = accession.replace("IEDB:", "")
        url = f"{self.config.base_url}/epitope/{epitope_id}"
        data = self._make_request(url)
        if not data:
            return None
        return self._parse_entry(data)

    def search_by_organism(
        self, organism: str, max_results: int = 20
    ) -> List[PeptideMetadata]:
        """Search IEDB for epitopes from a specific source organism."""
        url = f"{self.config.base_url}/epitope_search"
        params = {"source_organism": organism, "limit": min(max_results, 100)}
        data = self._make_request(url, params)
        if not data:
            return []
        epitopes = data if isinstance(data, list) else data.get("results", [])
        return [self._parse_entry(e) for e in epitopes[:max_results]]

    def _parse_entry(self, raw: Dict[str, Any]) -> PeptideMetadata:
        """Parse a raw IEDB API response into PeptideMetadata."""
        sequence = raw.get("linear_sequence", raw.get("sequence", ""))
        epitope_id = str(raw.get("epitope_id", raw.get("id", "")))

        activities = []
        assay_type = raw.get("assay_type", raw.get("assay", ""))
        if assay_type:
            activities.append(str(assay_type))
        mhc = raw.get("mhc_allele", raw.get("mhc_restriction", ""))
        if mhc:
            activities.append(f"MHC:{mhc}")

        targets = []
        organism = raw.get("source_organism", raw.get("organism", ""))
        if organism:
            targets.append(str(organism))

        source_protein = raw.get("source_antigen", raw.get("antigen_name"))

        return PeptideMetadata(
            accession=(
                f"IEDB:{epitope_id}"
                if not epitope_id.startswith("IEDB")
                else epitope_id
            ),
            sequence=sequence,
            length=len(sequence),
            source_db="iedb",
            activities=activities,
            target_organisms=targets,
            molecular_weight=raw.get("molecular_weight"),
            charge=raw.get("charge"),
            source_protein=str(source_protein) if source_protein else None,
            source_url=(
                f"https://www.iedb.org/epitope/{epitope_id}" if epitope_id else None
            ),
        )

    def _make_request(
        self, url: str, params: Optional[Dict[str, Any]] = None
    ) -> Optional[Any]:
        """Delegate to base class with config-driven parameters."""
        return super()._make_request(
            url,
            params=params,
            max_retry=self.config.max_retry,
            sleep_time=self.config.sleep_time,
            timeout=self.config.timeout,
        )
