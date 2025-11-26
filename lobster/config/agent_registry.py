"""
Centralized Agent Registry for the Lobster system.

This module defines all agents used in the system with their configurations,
making it easy to add new agents without modifying multiple files.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional


@dataclass
class AgentRegistryConfig:
    """Configuration for an agent in the system.

    Attributes:
        supervisor_accessible: Controls whether supervisor can directly handoff to this agent.
            - None (default): Inferred from child_agents relationships. If this agent
              appears in ANY parent's child_agents list, it's NOT supervisor-accessible.
            - True: Explicitly allow supervisor access (override inference).
            - False: Explicitly deny supervisor access (override inference).
    """

    name: str
    display_name: str
    description: str
    factory_function: str  # Module path to the factory function
    handoff_tool_name: Optional[str] = None
    handoff_tool_description: Optional[str] = None
    child_agents: Optional[List[str]] = None  # List of agent names this agent can delegate to
    supervisor_accessible: Optional[bool] = None  # None=infer, True/False=override


# Central registry of all agents in the system
AGENT_REGISTRY: Dict[str, AgentRegistryConfig] = {
    "data_expert_agent": AgentRegistryConfig(
        name="data_expert_agent",
        display_name="Data Expert",
        description="Executes queue-based downloads (ZERO online access), manages modalities with CRUD operations, loads local files via adapter system, retry mechanism with strategy overrides, and workspace orchestration",
        factory_function="lobster.agents.data_expert.data_expert",
        handoff_tool_name="handoff_to_data_expert_agent",
        handoff_tool_description="Assign LOCAL data operations: execute downloads from validated queue entries, load local files via adapters, manage modalities (list/inspect/remove/validate), retry failed downloads. DO NOT delegate online operations (metadata/URL extraction) - those go to research_agent",
        child_agents=["metadata_assistant"],
    ),
    "research_agent": AgentRegistryConfig(
        name="research_agent",
        display_name="Research Agent",
        description="Handles literature discovery, dataset identification, PDF extraction with AUTOMATIC PMID/DOI resolution, computational method analysis, and parameter extraction from publications and queuing datasets for download.",
        factory_function="lobster.agents.research_agent.research_agent",
        handoff_tool_name="handoff_to_research_agent",
        handoff_tool_description="Assign literature search, dataset discovery, method analysis, parameter extraction, and download queue creation to the research agent",
        child_agents=["metadata_assistant"],
    ),
    "singlecell_expert_agent": AgentRegistryConfig(
        name="singlecell_expert_agent",
        display_name="Single-Cell Expert",
        description="Handles single-cell RNA-seq analysis (cluster, QC, filter/normalize, automatic and manual cell annotation, differential expression etc) tasks (excluding visualization)",
        factory_function="lobster.agents.singlecell_expert.singlecell_expert",
        handoff_tool_name="handoff_to_singlecell_expert_agent",
        handoff_tool_description="Assign single-cell RNA-seq analysis tasks to the single-cell expert agent",
    ),
    # "bulk_rnaseq_expert_agent": AgentRegistryConfig(
    #     name="bulk_rnaseq_expert_agent",
    #     display_name="Bulk RNA-seq Expert",
    #     description="Handles bulk RNA-seq analysis tasks (excluding visualization)",
    #     factory_function="lobster.agents.bulk_rnaseq_expert.bulk_rnaseq_expert",
    #     handoff_tool_name="handoff_to_bulk_rnaseq_expert_agent",
    #     handoff_tool_description="Assign bulk RNA-seq analysis tasks to the bulk RNA-seq expert agent",
    # ),
    "metadata_assistant": AgentRegistryConfig(
        name="metadata_assistant",
        display_name="Metadata Assistant",
        description="Handles cross-dataset metadata operations including sample ID mapping (exact/fuzzy/pattern/metadata strategies), metadata standardization using Pydantic schemas (transcriptomics/proteomics), dataset completeness validation (samples, conditions, controls, duplicates, platform), and sample metadata reading in multiple formats. Specialized in metadata harmonization for multi-omics integration.",
        factory_function="lobster.agents.metadata_assistant.metadata_assistant",
        handoff_tool_name="handoff_to_metadata_assistant",
        handoff_tool_description="Assign metadata operations (cross-dataset sample mapping, metadata standardization to Pydantic schemas, dataset validation before download, metadata reading/formatting) to the metadata assistant",
    ),
    "machine_learning_expert_agent": AgentRegistryConfig(
        name="machine_learning_expert_agent",
        display_name="ML Expert",
        description="Handles Machine Learning related tasks like transforming the data in the desired format for downstream tasks",
        factory_function="lobster.agents.machine_learning_expert.machine_learning_expert",
        handoff_tool_name="handoff_to_machine_learning_expert_agent",
        handoff_tool_description="Assign all machine learning related tasks (scVI, classification etc) to the machine learning expert agent",
    ),
    "visualization_expert_agent": AgentRegistryConfig(
        name="visualization_expert_agent",
        display_name="Visualization Expert",
        description="Creates publication-quality visualizations through supervisor-mediated workflows",
        factory_function="lobster.agents.visualization_expert.visualization_expert",
        handoff_tool_name="handoff_to_visualization_expert_agent",
        handoff_tool_description="Delegate visualization tasks to the visualization expert agent",
    ),
    "protein_structure_visualization_expert_agent": AgentRegistryConfig(
        name="protein_structure_visualization_expert_agent",
        display_name="Protein Structure Visualization Expert",
        description="Handles 3D protein structure visualization (PDB structure fetching, ChimeraX visualization, RMSD calculation, secondary structure analysis) and structural analysis using PDB and pymol",
        factory_function="lobster.agents.protein_structure_visualization_expert.protein_structure_visualization_expert",
        handoff_tool_name="handoff_to_protein_structure_visualization_expert_agent",
        handoff_tool_description="Assign protein structure visualization tasks to the protein structure visualization expert agent",
    ),
    # 'ms_proteomics_expert_agent': AgentRegistryConfig(
    #     name='ms_proteomics_expert_agent',
    #     display_name='MS Proteomics Expert',
    #     description='Handles mass spectrometry proteomics data analysis including DDA/DIA workflows with database search artifact removal',
    #     factory_function='lobster.agents.ms_proteomics_expert.ms_proteomics_expert',
    #     handoff_tool_name='handoff_to_ms_proteomics_expert_agent',
    #     handoff_tool_description='Assign mass spectrometry proteomics analysis tasks to the MS proteomics expert agent'
    # ),
    # 'affinity_proteomics_expert_agent': AgentRegistryConfig(
    #     name='affinity_proteomics_expert_agent',
    #     display_name='Affinity Proteomics Expert',
    #     description='Handles affinity proteomics data analysis including Olink and targeted protein panels with antibody validation',
    #     factory_function='lobster.agents.affinity_proteomics_expert.affinity_proteomics_expert',
    #     handoff_tool_name='handoff_to_affinity_proteomics_expert_agent',
    #     handoff_tool_description='Assign affinity proteomics and targeted panel analysis tasks to the affinity proteomics expert agent'
    # ),
    # 'custom_feature_agent': AgentRegistryConfig(
    #     name='custom_feature_agent',
    #     display_name='Custom Feature Agent',
    #     description="""META-AGENT that generates new Lobster components using Claude Code SDK.
    #
    #     Capabilities:
    #     - Generate new agents following Lobster architectural patterns (registry-driven, stateless services)
    #     - Create services with 3-tuple return pattern (AnnData, stats, IR)
    #     - Build providers for external data sources (PubMed, GEO, custom APIs)
    #     - Generate adapters for new file formats (H5AD, CSV, custom formats)
    #     - Create comprehensive test suites (unit, integration)
    #     - Validate integration with registry patterns
    #     - Research best practices via Linkup SDK (GitHub repos, Python packages)
    #     - Generate complete documentation (wiki pages, usage examples)
    #
    #     When to delegate:
    #     - User requests NEW CAPABILITIES or MODALITIES (e.g., metabolomics, metagenomics, spatial)
    #     - Need to ADD NEW DATA SOURCES (e.g., PRIDE, Metabolomics Workbench)
    #     - Request involves GENERATING CODE (not analyzing data)
    #     - Building custom extensions for new analysis types
    #     - Creating providers for external APIs
    #     - Need adapters for specialized file formats
    #
    #     DO NOT delegate for:
    #     - Standard data analysis tasks (use domain experts: singlecell, bulk, proteomics)
    #     - Visualization requests (use visualization_expert)
    #     - Literature search (use research_agent)
    #     - Data loading/download (use data_expert)
    #     - Metadata operations (use metadata_assistant)
    #
    #     Note: This is a CODE GENERATION agent, not a DATA ANALYSIS agent.
    #     Uses Claude Code SDK for file creation and Linkup SDK for research.
    #     """,
    #     factory_function='lobster.agents.custom_feature_agent.custom_feature_agent',
    #     handoff_tool_name='handoff_to_custom_feature_agent',
    #     handoff_tool_description="""Delegate to Custom Feature Agent when user requests:
    #     1. NEW CAPABILITIES: "Add support for metabolomics", "Create metagenomics analysis"
    #     2. NEW DATA SOURCES: "Integrate PRIDE database", "Add Metabolomics Workbench"
    #     3. CODE GENERATION: "Generate an agent for...", "Create a service for..."
    #     4. CUSTOM EXTENSIONS: "Build adapter for X format", "Create provider for Y API"
    #
    #     Key indicators: "add support", "create agent", "generate", "build", "integrate new", "extend with"
    #
    #     DO NOT delegate for standard analysis (clustering, DE, QC, visualization, literature search).
    #     """
    # ),
}


# Additional agent names that might appear in chains but aren't worker agents
# SYSTEM_AGENTS = ['supervisor', 'transcriptomics_expert', 'method_agent', 'clarify_with_user']


def get_all_agent_names() -> list[str]:
    """Get all agent names including system agents."""
    return list(AGENT_REGISTRY.keys())


def get_worker_agents() -> Dict[str, AgentRegistryConfig]:
    """Get only the worker agents (excluding system agents)."""
    return AGENT_REGISTRY.copy()


def get_agent_registry_config(agent_name: str) -> Optional[AgentRegistryConfig]:
    """Get registry configuration for a specific agent."""
    return AGENT_REGISTRY.get(agent_name)


def import_agent_factory(factory_path: str) -> Callable:
    """Dynamically import an agent factory function."""
    module_path, function_name = factory_path.rsplit(".", 1)
    module = __import__(module_path, fromlist=[function_name])
    return getattr(module, function_name)

