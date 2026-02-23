"""
Unified Proteomics Expert Parent Agent for mass spectrometry and affinity proteomics analysis.

This agent serves as the main orchestrator for proteomics analysis, with:
- Shared QC/preprocessing tools (from shared_tools.py) available directly
- Platform-specific tools (peptide mapping for MS, antibody validation for affinity) inline
- Delegation to de_analysis_expert for differential expression, time course, correlation
- Delegation to biomarker_discovery_expert for WGCNA network analysis and survival

The agent auto-detects platform type and applies appropriate defaults.
"""

# Agent configuration for entry point discovery (must be at top, before heavy imports)
from lobster.config.agent_registry import AgentRegistryConfig

AGENT_CONFIG = AgentRegistryConfig(
    name="proteomics_expert",
    display_name="Proteomics Expert",
    description="Mass spectrometry and affinity proteomics analysis: QC, normalization, pattern analysis, delegates to DE and biomarker sub-agents",
    factory_function="lobster.agents.proteomics.proteomics_expert.proteomics_expert",
    handoff_tool_name="handoff_to_proteomics_expert",
    handoff_tool_description="Assign proteomics analysis tasks: MS/affinity QC, normalization, differential expression, network analysis, survival analysis",
    child_agents=["proteomics_de_analysis_expert", "biomarker_discovery_expert"],
    supervisor_accessible=True,
    tier_requirement="free",  # All agents free — commercial value in Omics-OS Cloud
)

# === Heavy imports below ===
from pathlib import Path
from typing import Optional

import numpy as np
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from lobster.agents.proteomics.prompts import create_proteomics_expert_prompt
from lobster.agents.proteomics.shared_tools import create_shared_tools
from lobster.agents.proteomics.state import ProteomicsExpertState
from lobster.config.llm_factory import create_llm
from lobster.config.settings import get_settings
from lobster.core.analysis_ir import AnalysisStep, ParameterSpec
from lobster.core.data_manager_v2 import DataManagerV2
from lobster.services.analysis.proteomics_analysis_service import (
    ProteomicsAnalysisService,
)
from lobster.services.quality.proteomics_preprocessing_service import (
    ProteomicsPreprocessingService,
)
from lobster.services.quality.proteomics_quality_service import ProteomicsQualityService
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


class ProteomicsAgentError(Exception):
    """Base exception for proteomics agent operations."""

    pass


class ModalityNotFoundError(ProteomicsAgentError):
    """Raised when requested modality doesn't exist."""

    pass


def proteomics_expert(
    data_manager: DataManagerV2,
    callback_handler=None,
    agent_name: str = "proteomics_expert",
    delegation_tools: list = None,
    force_platform_type: Optional[str] = None,
    workspace_path: Optional[Path] = None,
    provider_override: Optional[str] = None,
    model_override: Optional[str] = None,
):
    """
    Factory function for unified proteomics expert parent agent.

    This agent handles both mass spectrometry and affinity proteomics analysis.
    It auto-detects platform type from data characteristics and applies
    appropriate defaults. Delegates DE analysis and biomarker discovery
    to specialized sub-agents.

    Args:
        data_manager: DataManagerV2 instance for modality management
        callback_handler: Optional callback handler for LLM interactions
        agent_name: Name identifier for the agent instance
        delegation_tools: List of delegation tools for sub-agents (de_analysis_expert, biomarker_discovery_expert)
        force_platform_type: Override auto-detection ("mass_spec" or "affinity")
        workspace_path: Optional workspace path for LLM operations
        provider_override: Optional LLM provider override
        model_override: Optional model override

    Returns:
        Configured ReAct agent with proteomics analysis capabilities
    """
    settings = get_settings()
    model_params = settings.get_agent_llm_params("proteomics_expert")
    llm = create_llm(
        "proteomics_expert",
        model_params,
        provider_override=provider_override,
        model_override=model_override,
        workspace_path=workspace_path,
    )

    # Normalize callbacks to a flat list (fix double-nesting bug)
    if callback_handler and hasattr(llm, "with_config"):
        callbacks = (
            callback_handler
            if isinstance(callback_handler, list)
            else [callback_handler]
        )
        llm = llm.with_config(callbacks=callbacks)

    # Validate data manager type
    if not isinstance(data_manager, DataManagerV2):
        raise ValueError("ProteomicsExpert requires DataManagerV2 for modular analysis")

    # Initialize stateless services
    preprocessing_service = ProteomicsPreprocessingService()
    quality_service = ProteomicsQualityService()
    analysis_service = ProteomicsAnalysisService()

    # Store forced platform type for tools to access
    _forced_platform_type = force_platform_type

    # =========================================================================
    # GET SHARED TOOLS (QC, filter, normalize, patterns, impute, variable selection, summary)
    # =========================================================================

    shared_tools = create_shared_tools(
        data_manager,
        quality_service,
        preprocessing_service,
        analysis_service,
        force_platform_type=force_platform_type,
    )

    # =========================================================================
    # PLATFORM-SPECIFIC TOOLS (kept inline — simple, MS/affinity-specific)
    # =========================================================================

    @tool
    def add_peptide_mapping(
        modality_name: str,
        peptide_file_path: str,
        save_result: bool = True,
    ) -> str:
        """
        Add peptide-to-protein mapping information to an MS proteomics modality.

        This is specific to mass spectrometry data where peptide evidence
        is critical for protein quantification quality.

        Args:
            modality_name: Name of the MS proteomics modality
            peptide_file_path: Path to CSV file with peptide mapping data
            save_result: Whether to save the updated modality

        Returns:
            str: Peptide mapping results
        """
        # DEPRECATED: Merged into import_proteomics_data (shared_tools.py)
        logger.warning("add_peptide_mapping is deprecated. Use import_proteomics_data which automatically extracts peptide mapping from parser output.")
        return "DEPRECATED: add_peptide_mapping has been merged into import_proteomics_data. " \
               "MaxQuant/DIA-NN/Spectronaut parsers automatically extract peptide counts, " \
               "unique peptides, and sequence coverage during import."

    @tool
    def validate_antibody_specificity(
        modality_name: str,
        cross_reactivity_threshold: float = 0.9,
        save_result: bool = True,
    ) -> str:
        """
        Validate antibody specificity and detect potential cross-reactivity issues.

        This is specific to affinity proteomics where antibody cross-reactivity
        can confound results.

        Args:
            modality_name: Name of the affinity proteomics modality
            cross_reactivity_threshold: Correlation threshold for flagging (default: 0.9)
            save_result: Whether to save the updated modality

        Returns:
            str: Antibody validation results
        """
        try:
            adata = data_manager.get_modality(modality_name)
        except ValueError:
            return f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}"

        try:
            adata_validated = adata.copy()
            X = (
                adata_validated.X.toarray()
                if hasattr(adata_validated.X, "toarray")
                else adata_validated.X
            )

            import pandas as pd

            cross_reactive_pairs = []

            if adata_validated.n_vars > 1:
                df = pd.DataFrame(X, columns=adata_validated.var_names)
                correlation_matrix = df.corr(method='pearson', min_periods=3).fillna(0).values

                for i in range(len(correlation_matrix)):
                    for j in range(i + 1, len(correlation_matrix)):
                        if correlation_matrix[i, j] > cross_reactivity_threshold:
                            protein_i = adata_validated.var_names[i]
                            protein_j = adata_validated.var_names[j]
                            cross_reactive_pairs.append(
                                (protein_i, protein_j, correlation_matrix[i, j])
                            )

            if "cross_reactive" not in adata_validated.var.columns:
                adata_validated.var["cross_reactive"] = False

            for protein_pair in cross_reactive_pairs:
                protein_i, protein_j, _ = protein_pair
                adata_validated.var.loc[protein_i, "cross_reactive"] = True
                adata_validated.var.loc[protein_j, "cross_reactive"] = True

            validated_name = f"{modality_name}_antibody_validated"
            data_manager.store_modality(
                name=validated_name,
                adata=adata_validated,
                parent_name=modality_name,
                step_summary=f"Validated antibody specificity: {len(cross_reactive_pairs)} cross-reactive pairs",
            )

            if save_result:
                save_path = f"{modality_name}_antibody_validated.h5ad"
                data_manager.save_modality(validated_name, save_path)

            ir = AnalysisStep(
                operation="proteomics.affinity.validate_antibody_specificity",
                tool_name="validate_antibody_specificity",
                description="Validate antibody specificity for affinity proteomics",
                library="lobster.agents.proteomics.proteomics_expert",
                code_template="""# Antibody cross-reactivity check (Affinity-specific)
import numpy as np
import pandas as pd

X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
df = pd.DataFrame(X, columns=adata.var_names)
correlation_matrix = df.corr(method='pearson', min_periods=3).fillna(0).values

cross_reactive_pairs = []
for i in range(len(correlation_matrix)):
    for j in range(i + 1, len(correlation_matrix)):
        if correlation_matrix[i, j] > {{ cross_reactivity_threshold }}:
            cross_reactive_pairs.append((i, j, correlation_matrix[i, j]))

adata.var['cross_reactive'] = False
for i, j, _ in cross_reactive_pairs:
    adata.var.iloc[i, adata.var.columns.get_loc('cross_reactive')] = True
    adata.var.iloc[j, adata.var.columns.get_loc('cross_reactive')] = True""",
                imports=["import numpy as np", "import pandas as pd"],
                parameters={
                    "cross_reactivity_threshold": cross_reactivity_threshold,
                    "n_cross_reactive_pairs": len(cross_reactive_pairs),
                },
                parameter_schema={
                    "cross_reactivity_threshold": ParameterSpec(
                        param_type="float",
                        papermill_injectable=True,
                        default_value=0.9,
                        required=False,
                        validation_rule="0 < cross_reactivity_threshold <= 1",
                        description="Correlation threshold for flagging cross-reactive antibodies",
                    ),
                },
                input_entities=["adata"],
                output_entities=["adata_validated"],
            )

            data_manager.log_tool_usage(
                tool_name="validate_antibody_specificity",
                parameters={
                    "modality_name": modality_name,
                    "cross_reactivity_threshold": cross_reactivity_threshold,
                },
                description="Validated antibody specificity for affinity data",
                ir=ir,
            )

            response = f"Successfully validated antibody specificity for '{modality_name}'!\n\n"
            response += "**Antibody Validation Results:**\n"
            response += f"- Total proteins analyzed: {adata_validated.n_vars}\n"
            response += f"- Cross-reactive pairs detected: {len(cross_reactive_pairs)}\n"
            response += f"- Correlation threshold: {cross_reactivity_threshold}\n\n"

            if cross_reactive_pairs:
                response += "**Potential Cross-Reactive Pairs:**\n"
                for protein_i, protein_j, correlation in cross_reactive_pairs[:5]:
                    response += f"- {protein_i} <-> {protein_j}: r={correlation:.3f}\n"
                if len(cross_reactive_pairs) > 5:
                    response += f"- ... and {len(cross_reactive_pairs) - 5} more pairs\n"
                response += "\n**Recommendations:**\n"
                response += "- Review antibody specificity documentation\n"
                response += "- Consider removing highly cross-reactive antibodies\n"
                response += "- Validate results with orthogonal methods\n"
            else:
                response += "No significant cross-reactivity detected.\n"

            response += f"\n**New modality created**: '{validated_name}'"
            if save_result:
                response += f"\n**Saved to**: {save_path}"

            return response

        except Exception as e:
            logger.error(f"Error validating antibody specificity: {e}")
            return f"Error in antibody validation: {str(e)}"

    @tool
    def correct_plate_effects(
        modality_name: str,
        plate_column: str = "plate_id",
        method: str = "combat",
        save_result: bool = True,
    ) -> str:
        """
        Correct batch effects from plate layout in affinity proteomics data.

        This is specific to affinity platforms where multi-plate studies
        require batch correction.

        Args:
            modality_name: Name of the affinity proteomics modality
            plate_column: Column in obs containing plate identifiers
            method: Correction method ("combat", "median_centering")
            save_result: Whether to save the corrected modality

        Returns:
            str: Plate correction results
        """
        try:
            adata = data_manager.get_modality(modality_name)
        except ValueError:
            return f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}"

        try:
            if plate_column not in adata.obs.columns:
                return f"Plate column '{plate_column}' not found. Available columns: {list(adata.obs.columns)}"

            corrected_adata, batch_stats, batch_ir = (
                preprocessing_service.correct_batch_effects(
                    adata,
                    batch_key=plate_column,
                    method=method,
                )
            )

            corrected_name = f"{modality_name}_plate_corrected"
            data_manager.store_modality(
                name=corrected_name,
                adata=corrected_adata,
                parent_name=modality_name,
                step_summary=f"Plate effect corrected using {method}",
            )

            if save_result:
                save_path = f"{modality_name}_plate_corrected.h5ad"
                data_manager.save_modality(corrected_name, save_path)

            data_manager.log_tool_usage(
                tool_name="correct_plate_effects",
                parameters={
                    "modality_name": modality_name,
                    "plate_column": plate_column,
                    "method": method,
                },
                description="Corrected plate effects for affinity data",
                ir=batch_ir,
            )

            response = f"Successfully corrected plate effects in '{modality_name}'!\n\n"
            response += "**Plate Correction Results:**\n"
            response += f"- Method: {method}\n"
            response += f"- Plate column: {plate_column}\n"

            if "n_batches_corrected" in batch_stats:
                response += f"- Plates corrected: {batch_stats['n_batches_corrected']}\n"

            response += f"\n**New modality created**: '{corrected_name}'"
            if save_result:
                response += f"\n**Saved to**: {save_path}"

            response += "\n\n**Affinity Platform Notes:**"
            response += "\n- Plate correction essential for multi-plate studies"
            response += "\n- Verify correction by checking inter-plate correlation"

            return response

        except Exception as e:
            logger.error(f"Error correcting plate effects: {e}")
            return f"Error in plate correction: {str(e)}"

    # =========================================================================
    # COLLECT ALL TOOLS
    # =========================================================================

    # Platform-specific tools (kept inline; add_peptide_mapping deprecated — use import_proteomics_data)
    platform_tools = [
        validate_antibody_specificity, # Affinity-specific
        correct_plate_effects,         # Affinity-specific
    ]

    # Combine: shared tools + platform tools
    direct_tools = shared_tools + platform_tools

    # Add delegation tools if provided (de_analysis_expert, biomarker_discovery_expert)
    tools = direct_tools
    if delegation_tools:
        tools = tools + delegation_tools

    # Create system prompt
    system_prompt = create_proteomics_expert_prompt()

    return create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt,
        name=agent_name,
        state_schema=ProteomicsExpertState,
    )
