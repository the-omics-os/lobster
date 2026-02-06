"""
Schema definitions and validation for the modular DataManager architecture.

This module provides schema definitions for different biological data modalities
and flexible validation that supports both strict and permissive modes.
"""

from .download_queue import DownloadQueueEntry, DownloadStatus, StrategyConfig
from .ontology import DiseaseConcept, DiseaseMatch, DiseaseOntologyConfig

# Vector search schemas
from .search import (
    SearchBackend,
    EmbeddingProvider,
    SearchResult,
    SearchResponse,
    OntologyMatch,
    LiteratureMatch,
    VectorSearchConfig,
)

# Clinical trial metadata (RECIST 1.1, timepoints)
from .clinical_schema import (
    RECIST_RESPONSES,
    RESPONDER_GROUP,
    NON_RESPONDER_GROUP,
    RESPONSE_SYNONYMS,
    ClinicalSample,
    parse_timepoint,
    timepoint_to_absolute_day,
    normalize_response,
    classify_response_group,
    is_responder,
    is_non_responder,
)

__all__ = [
    "DownloadQueueEntry",
    "DownloadStatus",
    "StrategyConfig",
    "DiseaseConcept",
    "DiseaseMatch",
    "DiseaseOntologyConfig",
    # Vector search schemas
    "SearchBackend",
    "EmbeddingProvider",
    "SearchResult",
    "SearchResponse",
    "OntologyMatch",
    "LiteratureMatch",
    "VectorSearchConfig",
    # Clinical schema exports
    "RECIST_RESPONSES",
    "RESPONDER_GROUP",
    "NON_RESPONDER_GROUP",
    "RESPONSE_SYNONYMS",
    "ClinicalSample",
    "parse_timepoint",
    "timepoint_to_absolute_day",
    "normalize_response",
    "classify_response_group",
    "is_responder",
    "is_non_responder",
]
