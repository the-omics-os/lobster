"""
DBAASP (Database of Antimicrobial Activity and Structure of Peptides) provider.

REST API v1: https://dbaasp.org/api/v1
Focus: Antimicrobial peptides (20K+ AMPs) with MIC values and target organisms.
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


class DBAASPProviderConfig(BaseModel):
    """Configuration for DBAASP provider."""

    base_url: str = "https://dbaasp.org/api/v1"
    max_retry: int = Field(default=3, ge=1, le=10)
    sleep_time: float = Field(default=0.5, ge=0.1, le=5.0)
    timeout: int = Field(default=30, ge=5, le=120)


class DBAASPProvider(BaseBiomoleculeProvider):
    """DBAASP v4 antimicrobial peptide database provider."""

    def __init__(self, config: Optional[DBAASPProviderConfig] = None):
        self.config = config or DBAASPProviderConfig()

    @property
    def source(self) -> BiomoleculeSource:
        return BiomoleculeSource.DBAASP

    @property
    def priority(self) -> int:
        return 10

    def get_supported_capabilities(self) -> Dict[str, bool]:
        return {
            "search_sequences": True,
            "get_entry": True,
            "search_by_activity": True,
            "search_by_target": True,
            "batch_download": False,
        }

    def search_sequences(
        self, query: str, max_results: int = 20
    ) -> List[PeptideMetadata]:
        """Search DBAASP for peptides matching query."""
        url = f"{self.config.base_url}/peptides"
        params = {"search": query, "limit": min(max_results, 100)}
        data = self._make_request(url, params)
        if not data:
            return []
        peptides = (
            data
            if isinstance(data, list)
            else data.get("results", data.get("peptides", []))
        )
        return [self._parse_entry(p) for p in peptides[:max_results]]

    def get_entry(self, accession: str) -> Optional[PeptideMetadata]:
        """Get a single DBAASP entry by ID."""
        url = f"{self.config.base_url}/peptides/{accession}"
        data = self._make_request(url)
        if not data:
            return None
        return self._parse_entry(data)

    def search_by_activity(
        self,
        activity_type: str,
        target_organism: Optional[str] = None,
        max_results: int = 20,
    ) -> List[PeptideMetadata]:
        """Search by antimicrobial activity type and optional target organism."""
        url = f"{self.config.base_url}/peptides"
        params: Dict[str, Any] = {
            "activity": activity_type,
            "limit": min(max_results, 100),
        }
        if target_organism:
            params["target"] = target_organism
        data = self._make_request(url, params)
        if not data:
            return []
        peptides = (
            data
            if isinstance(data, list)
            else data.get("results", data.get("peptides", []))
        )
        return [self._parse_entry(p) for p in peptides[:max_results]]

    def _parse_entry(self, raw: Dict[str, Any]) -> PeptideMetadata:
        """Parse a raw DBAASP API response entry into PeptideMetadata."""
        sequence = raw.get("sequence", raw.get("seq", ""))
        accession = str(raw.get("id", raw.get("dbaasp_id", raw.get("accession", ""))))

        activities = []
        for act in raw.get("activities", raw.get("activity", [])):
            if isinstance(act, str):
                activities.append(act)
            elif isinstance(act, dict):
                activities.append(act.get("type", act.get("name", str(act))))

        targets = []
        for tgt in raw.get("target_organisms", raw.get("targets", [])):
            if isinstance(tgt, str):
                targets.append(tgt)
            elif isinstance(tgt, dict):
                targets.append(tgt.get("name", tgt.get("organism", str(tgt))))

        return PeptideMetadata(
            accession=(
                f"DBAASP:{accession}"
                if not accession.startswith("DBAASP")
                else accession
            ),
            sequence=sequence,
            length=len(sequence),
            source_db="dbaasp",
            activities=activities,
            target_organisms=targets,
            molecular_weight=raw.get("molecular_weight", raw.get("mw")),
            charge=raw.get("charge", raw.get("net_charge")),
            source_protein=raw.get("source_protein", raw.get("parent_protein")),
            source_url=f"https://dbaasp.org/peptide/{accession}" if accession else None,
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
