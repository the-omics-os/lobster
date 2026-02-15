"""
Schema definitions and validation for the modular DataManager architecture.

This module provides schema definitions for different biological data modalities
and flexible validation that supports both strict and permissive modes.
"""

# Clinical trial metadata (RECIST 1.1, timepoints)
from .clinical_schema import (
    NON_RESPONDER_GROUP,
    RECIST_RESPONSES,
    RESPONDER_GROUP,
    RESPONSE_SYNONYMS,
    ClinicalSample,
    classify_response_group,
    is_non_responder,
    is_responder,
    normalize_response,
    parse_timepoint,
    timepoint_to_absolute_day,
)
from .download_queue import DownloadQueueEntry, DownloadStatus, StrategyConfig
from .ontology import DiseaseConcept, DiseaseMatch, DiseaseOntologyConfig

__all__ = [
    "DownloadQueueEntry",
    "DownloadStatus",
    "StrategyConfig",
    "DiseaseConcept",
    "DiseaseMatch",
    "DiseaseOntologyConfig",
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
