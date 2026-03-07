# PEP 420 namespace — agent discovery via entry points
# AGENT_CONFIG stays here (entry point: lobster.agents.metadata_assistant:AGENT_CONFIG)
from lobster.config.agent_registry import AgentRegistryConfig

AGENT_CONFIG = AgentRegistryConfig(
    name="metadata_assistant",
    display_name="Metadata Assistant",
    description="Handles cross-dataset metadata operations including sample ID mapping (exact/fuzzy/pattern/metadata strategies), metadata standardization using Pydantic schemas (transcriptomics/proteomics), dataset completeness validation (samples, conditions, controls, duplicates, platform), sample metadata reading in multiple formats, and disease enrichment from publication context when SRA metadata is incomplete. Specialized in metadata harmonization for multi-omics integration and publication queue processing.",
    factory_function="lobster.agents.metadata_assistant.metadata_assistant.metadata_assistant",
    handoff_tool_name="handoff_to_metadata_assistant",
    handoff_tool_description="Assign metadata operations (cross-dataset sample mapping, metadata standardization to Pydantic schemas, dataset validation before download, metadata reading/formatting, publication queue filtering) to the metadata assistant",
)

__all__ = ["AGENT_CONFIG"]
