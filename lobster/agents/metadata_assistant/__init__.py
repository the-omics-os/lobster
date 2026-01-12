"""
Metadata Assistant Agent Module - Sample metadata and harmonization operations.

This agent specializes in sample ID mapping, metadata standardization, and
dataset content validation for multi-omics integration.

Note: This agent uses premium services (disease ontology, microbiome filtering).
Graceful degradation is handled within the agent factory.
"""

# =============================================================================
# AGENT_CONFIG must be defined FIRST (before heavy imports) for entry point loading
# This prevents circular import issues when component_registry loads this module
# =============================================================================
from lobster.config.agent_registry import AgentRegistryConfig

AGENT_CONFIG = AgentRegistryConfig(
    name="metadata_assistant",
    display_name="Metadata Assistant",
    description="Handles cross-dataset metadata operations including sample ID mapping (exact/fuzzy/pattern/metadata strategies), metadata standardization using Pydantic schemas (transcriptomics/proteomics), dataset completeness validation (samples, conditions, controls, duplicates, platform), sample metadata reading in multiple formats, and disease enrichment from publication context when SRA metadata is incomplete. Specialized in metadata harmonization for multi-omics integration and publication queue processing.",
    factory_function="lobster.agents.metadata_assistant.metadata_assistant.metadata_assistant",
    handoff_tool_name="handoff_to_metadata_assistant",
    handoff_tool_description="Assign metadata operations (cross-dataset sample mapping, metadata standardization to Pydantic schemas, dataset validation before download, metadata reading/formatting, publication queue filtering) to the metadata assistant",
)

# =============================================================================
# Heavy imports below (graceful degradation for missing dependencies)
# =============================================================================
try:
    from lobster.agents.metadata_assistant.config import (
        suppress_logs,
        detect_metadata_pattern,
        convert_list_to_dict,
        phase1_column_rescan,
        extract_disease_with_llm,
        phase2_llm_abstract_extraction,
        phase3_llm_methods_extraction,
        phase4_manual_mappings,
    )
    from lobster.agents.metadata_assistant.prompts import create_metadata_assistant_prompt
    from lobster.agents.metadata_assistant.metadata_assistant import metadata_assistant
    METADATA_ASSISTANT_AVAILABLE = True
except ImportError:
    METADATA_ASSISTANT_AVAILABLE = False
    metadata_assistant = None
    create_metadata_assistant_prompt = None
    suppress_logs = None
    detect_metadata_pattern = None
    convert_list_to_dict = None
    phase1_column_rescan = None
    extract_disease_with_llm = None
    phase2_llm_abstract_extraction = None
    phase3_llm_methods_extraction = None
    phase4_manual_mappings = None

__all__ = [
    "AGENT_CONFIG",
    "METADATA_ASSISTANT_AVAILABLE",
    "metadata_assistant",
    "create_metadata_assistant_prompt",
    "suppress_logs",
    "detect_metadata_pattern",
    "convert_list_to_dict",
    "phase1_column_rescan",
    "extract_disease_with_llm",
    "phase2_llm_abstract_extraction",
    "phase3_llm_methods_extraction",
    "phase4_manual_mappings",
]
