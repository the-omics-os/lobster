"""
Disease Ontology Service - Migration-Stable API

Phase 1: Keyword-based substring matching from JSON config
Phase 2 (Q2 2026): Embedding-based semantic search from ChromaDB

API contract remains stable across both phases - consumers write once,
backend implementation can be swapped without code changes.

Part of the Strangler Fig migration pattern for ontology modernization.
"""

import json
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

from lobster.core.schemas.ontology import (
    DiseaseConcept,
    DiseaseMatch,
    DiseaseOntologyConfig,
)
from lobster.utils.logger import get_logger

if TYPE_CHECKING:
    from lobster.core.schemas.search import OntologyMatch

logger = get_logger(__name__)


class DiseaseOntologyService:
    """
    Centralized disease ontology matching service.

    Phase 1: JSON-backed keyword matching
    Phase 2: ChromaDB-backed embedding search (backend swap)

    Consumer code unchanged between phases.

    Usage:
        # Get singleton instance
        service = DiseaseOntologyService.get_instance()

        # Match disease terms (Phase 2 compatible API)
        matches = service.match_disease("colorectal cancer", k=3, min_confidence=0.7)
        if matches:
            best = matches[0]
            print(f"Matched: {best.name} ({best.disease_id}) with confidence {best.confidence}")

        # Legacy API for backward compatibility during migration
        keywords = service.get_extraction_keywords()  # Dict[str, List[str]]
        variants = service.get_standardization_variants()  # Dict[str, List[str]]
    """

    _instance: Optional["DiseaseOntologyService"] = None

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize from JSON config.

        Args:
            config_path: Path to disease_ontology.json (optional)
                        Defaults to lobster/config/disease_ontology.json
        """
        if config_path is None:
            # Default: Package config (relative to this file)
            config_path = (
                Path(__file__).parent.parent.parent / "config" / "disease_ontology.json"
            )

        self._config_path = config_path
        self._config = self._load_from_json(config_path)

        # Phase 2: Initialize vector backend if configured
        self._vector_service = None
        if self._config.backend == "embeddings":
            try:
                from lobster.services.vector.service import VectorSearchService

                self._vector_service = VectorSearchService()
                logger.info("DiseaseOntologyService using embeddings backend")
            except ImportError:
                logger.warning(
                    "Vector search deps not installed, falling back to keyword matching. "
                    "Install with: pip install 'lobster-ai[vector-search]'"
                )
                # Fall through to keyword path (self._vector_service remains None)

        # Always build keyword index (needed for fallback and legacy APIs)
        self._diseases = self._config.diseases
        self._keyword_index = self._build_keyword_index()

        logger.debug(
            f"DiseaseOntologyService initialized with {len(self._diseases)} diseases "
            f"from {config_path} (backend={self._config.backend})"
        )

    @classmethod
    def get_instance(cls) -> "DiseaseOntologyService":
        """
        Singleton accessor.

        Returns:
            Shared DiseaseOntologyService instance

        Usage:
            service = DiseaseOntologyService.get_instance()
            matches = service.match_disease("CRC")
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """
        Reset singleton instance.

        Used for testing to ensure fresh state between tests.
        """
        cls._instance = None

    def _load_from_json(self, path: Path) -> DiseaseOntologyConfig:
        """
        Load and validate disease ontology from JSON.

        Phase 2: This method will be supplemented with ChromaDB loading
        when backend="embeddings" is configured.

        Args:
            path: Path to JSON config file

        Returns:
            Validated DiseaseOntologyConfig

        Raises:
            FileNotFoundError: If config file not found
            ValidationError: If JSON doesn't match schema
        """
        if not path.exists():
            raise FileNotFoundError(
                f"Disease ontology config not found: {path}. "
                f"Expected at: lobster/config/disease_ontology.json"
            )

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        # Validate with Pydantic
        config = DiseaseOntologyConfig(**data)

        logger.debug(
            f"Loaded {len(config.diseases)} disease concepts from {path} "
            f"(version: {config.version}, backend: {config.backend})"
        )
        return config

    def _build_keyword_index(self) -> Dict[str, DiseaseConcept]:
        """
        Build keyword -> disease mapping for fast lookup.

        Phase 1: In-memory dict
        Phase 2: Replaced by ChromaDB index

        Returns:
            Dict mapping lowercase keyword to DiseaseConcept
        """
        index = {}
        for disease in self._diseases:
            for keyword in disease.keywords:
                index[keyword.lower()] = disease
        return index

    def match_disease(
        self,
        query: str,
        k: int = 3,
        min_confidence: float = 0.7,
    ) -> List[DiseaseMatch]:
        """
        Match disease term to ontology.

        **MIGRATION-STABLE API**: This signature works for both Phase 1 and Phase 2.

        Phase 1 Implementation:
            - Keyword substring matching (case-insensitive)
            - Returns confidence=1.0 (exact match)
            - Returns <=1 result (first match wins)

        Phase 2 Implementation:
            - Embedding semantic search (ChromaDB)
            - Returns confidence=0.0-1.0 (cosine similarity)
            - Returns top-k results (ranked by similarity)

        Args:
            query: Disease term to match (e.g., "colorectal cancer", "UC", "crohns")
            k: Maximum results to return (Phase 1 ignores, Phase 2 uses)
            min_confidence: Minimum confidence threshold (Phase 1 ignores, Phase 2 filters)

        Returns:
            List of DiseaseMatch objects (empty if no match >= min_confidence)

        Examples:
            # Phase 1 behavior:
            >>> service = DiseaseOntologyService.get_instance()
            >>> matches = service.match_disease("colorectal cancer")
            >>> matches[0].disease_id
            'crc'
            >>> matches[0].confidence
            1.0
            >>> matches[0].match_type
            'exact_keyword'

            # Phase 2 behavior (same API, different results):
            >>> matches = service.match_disease("colon tumor", k=3)
            >>> matches[0].disease_id
            'MONDO:0005575'
            >>> matches[0].confidence
            0.89
            >>> matches[0].match_type
            'semantic_embedding'
        """
        # Phase 2: Delegate to vector search if available
        if self._vector_service is not None:
            ontology_matches = self._vector_service.match_ontology(query, "mondo", k=k)
            disease_matches = [
                self._convert_ontology_match(m, query) for m in ontology_matches
            ]
            return [m for m in disease_matches if m.confidence >= min_confidence]

        query_lower = query.lower()
        matches = []

        # Phase 1: Keyword substring matching
        for disease in self._diseases:
            matched_keyword = None
            for keyword in disease.keywords:
                keyword_lower = keyword.lower()
                if keyword_lower in query_lower:
                    matched_keyword = keyword
                    break

            if matched_keyword:
                matches.append(
                    DiseaseMatch(
                        disease_id=disease.id,
                        name=disease.name,
                        confidence=1.0,  # Phase 1: Always exact match
                        match_type="exact_keyword",
                        matched_term=matched_keyword,
                        metadata={
                            "mondo_id": disease.mondo_id,
                            "umls_cui": disease.umls_cui,
                            "mesh_terms": disease.mesh_terms,
                        },
                    )
                )

        # Filter by confidence (Phase 2 will use this, Phase 1 always passes)
        matches = [m for m in matches if m.confidence >= min_confidence]

        # Return top-k (Phase 1 returns <=1, Phase 2 returns k ranked)
        return matches[:k]

    def _convert_ontology_match(
        self, match: "OntologyMatch", original_query: str
    ) -> DiseaseMatch:
        """
        Convert OntologyMatch to DiseaseMatch (SCHM-03 compatibility).

        Maps vector search result fields to the DiseaseMatch API contract,
        preserving the same interface regardless of backend.

        Args:
            match: OntologyMatch from VectorSearchService.match_ontology()
            original_query: The original user query string

        Returns:
            DiseaseMatch with semantic_embedding match_type
        """
        return DiseaseMatch(
            disease_id=match.ontology_id,       # "MONDO:0005575"
            name=match.term,                     # "colorectal carcinoma"
            confidence=match.score,              # 0.0-1.0
            match_type="semantic_embedding",     # Phase 2 indicator
            matched_term=original_query,         # original user query
            metadata=match.metadata,             # extensible metadata
        )

    # =========================================================================
    # LEGACY API (Backward Compatibility During Phase 1 Migration)
    # =========================================================================

    def get_extraction_keywords(self) -> Dict[str, List[str]]:
        """
        Get keywords for disease extraction.

        DEPRECATED: Use match_disease() instead for Phase 2 compatibility.

        Provided for backward compatibility during Phase 1 consumer migration.
        Will be removed in Phase 2 when all consumers use match_disease().

        Returns:
            Dict mapping disease_id -> keyword list
            e.g., {"crc": ["colorectal", "colon cancer", ...]}
        """
        return {disease.id: disease.keywords for disease in self._diseases}

    def get_standardization_variants(self) -> Dict[str, List[str]]:
        """
        Get all keywords for fuzzy matching (DiseaseStandardizationService).

        Used by DiseaseStandardizationService to populate DISEASE_MAPPINGS.
        Phase 2: Consider deprecating in favor of match_disease().

        Returns:
            Dict mapping disease_id -> variant list
            e.g., {"crc": ["colorectal cancer", "colon cancer", ...]}
        """
        return {disease.id: disease.keywords for disease in self._diseases}

    def get_all_disease_ids(self) -> List[str]:
        """
        Get list of valid disease IDs.

        Returns:
            List of disease IDs: ["crc", "uc", "cd", "healthy"]
        """
        return [disease.id for disease in self._diseases]

    def validate_disease_id(self, disease_id: str) -> bool:
        """
        Check if disease_id is valid.

        Args:
            disease_id: ID to validate

        Returns:
            True if valid, False otherwise
        """
        return disease_id in [d.id for d in self._diseases]

    def get_disease_by_id(self, disease_id: str) -> Optional[DiseaseConcept]:
        """
        Get disease concept by ID.

        Args:
            disease_id: Disease ID (e.g., "crc")

        Returns:
            DiseaseConcept if found, None otherwise
        """
        for disease in self._diseases:
            if disease.id == disease_id:
                return disease
        return None
