"""
Peptipedia provider — aggregated peptide database.

REST API: https://app.peptipedia.cl/api
Focus: Aggregated search across APD3, BIOPEP, DBAASP, SATPdb (90K+ peptides).
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


class PeptipediaProviderConfig(BaseModel):
    """Configuration for Peptipedia provider."""

    base_url: str = "https://app.peptipedia.cl/api"
    max_retry: int = Field(default=3, ge=1, le=10)
    sleep_time: float = Field(default=1.0, ge=0.1, le=5.0)
    timeout: int = Field(default=30, ge=5, le=120)


class PeptipediaProvider(BaseBiomoleculeProvider):
    """Peptipedia aggregated peptide database provider."""

    def __init__(self, config: Optional[PeptipediaProviderConfig] = None):
        self.config = config or PeptipediaProviderConfig()

    @property
    def source(self) -> BiomoleculeSource:
        return BiomoleculeSource.PEPTIPEDIA

    @property
    def priority(self) -> int:
        return 20

    def get_supported_capabilities(self) -> Dict[str, bool]:
        return {
            "search_sequences": True,
            "get_entry": True,
            "search_by_activity": True,
            "search_by_target": False,
            "batch_download": False,
        }

    def search_sequences(
        self, query: str, max_results: int = 20
    ) -> List[PeptideMetadata]:
        """Search Peptipedia for peptides matching query."""
        url = f"{self.config.base_url}/peptides/search"
        params = {"query": query, "limit": min(max_results, 100)}
        data = self._make_request(url, params)
        if not data:
            return []
        peptides = (
            data
            if isinstance(data, list)
            else data.get("results", data.get("data", []))
        )
        return [self._parse_entry(p) for p in peptides[:max_results]]

    def get_entry(self, accession: str) -> Optional[PeptideMetadata]:
        """Get a single Peptipedia entry by ID."""
        entry_id = accession.replace("PEPTIPEDIA:", "")
        url = f"{self.config.base_url}/peptides/{entry_id}"
        data = self._make_request(url)
        if not data:
            return None
        return self._parse_entry(data)

    def search_by_activity(
        self, activity_type: str, max_results: int = 20
    ) -> List[PeptideMetadata]:
        """Search Peptipedia by biological activity type."""
        url = f"{self.config.base_url}/peptides/search"
        params = {"activity": activity_type, "limit": min(max_results, 100)}
        data = self._make_request(url, params)
        if not data:
            return []
        peptides = (
            data
            if isinstance(data, list)
            else data.get("results", data.get("data", []))
        )
        return [self._parse_entry(p) for p in peptides[:max_results]]

    def _parse_entry(self, raw: Dict[str, Any]) -> PeptideMetadata:
        """Parse a raw Peptipedia response into PeptideMetadata."""
        sequence = raw.get("sequence", raw.get("seq", ""))
        entry_id = str(
            raw.get("id", raw.get("peptipedia_id", raw.get("accession", "")))
        )

        activities = []
        for act in raw.get("activities", raw.get("biological_activity", [])):
            if isinstance(act, str):
                activities.append(act)
            elif isinstance(act, dict):
                activities.append(act.get("name", act.get("type", str(act))))

        targets = []
        for tgt in raw.get("target_organisms", raw.get("targets", [])):
            if isinstance(tgt, str):
                targets.append(tgt)
            elif isinstance(tgt, dict):
                targets.append(tgt.get("name", str(tgt)))

        source_db_name = raw.get("source_database", raw.get("database", "peptipedia"))

        return PeptideMetadata(
            accession=(
                f"PEPTIPEDIA:{entry_id}"
                if not entry_id.startswith("PEPTIPEDIA")
                else entry_id
            ),
            sequence=sequence,
            length=len(sequence),
            source_db=str(source_db_name).lower(),
            activities=activities,
            target_organisms=targets,
            molecular_weight=raw.get("molecular_weight", raw.get("mw")),
            charge=raw.get("charge", raw.get("net_charge")),
            source_protein=raw.get("source_protein", raw.get("parent_protein")),
            source_url=(
                f"https://app.peptipedia.cl/peptide/{entry_id}" if entry_id else None
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
