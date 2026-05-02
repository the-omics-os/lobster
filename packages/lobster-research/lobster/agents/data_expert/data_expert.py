"""
Data Expert Agent for multi-omics data acquisition, processing, and workspace management.

This agent is responsible for managing all data-related operations using the modular
DataManagerV2 system, including GEO data fetching, local file processing, workspace
restoration, and multi-omics data integration with proper modality handling and
schema validation.
"""

# Agent configuration for entry point discovery (must be at top, before heavy imports)
from lobster.config.agent_registry import AgentRegistryConfig

AGENT_CONFIG = AgentRegistryConfig(
    name="data_expert_agent",
    display_name="Data Expert",
    description=(
        "Service agent for LOCAL data operations ONLY (ZERO online access): "
        "queue-based downloads, file loading via adapters, modality CRUD, "
        "workspace file operations (list, read, write, search, shell execute). "
        "NOT for domain analysis — route QC, statistics, and biological interpretation "
        "to domain experts (transcriptomics, genomics, proteomics, metabolomics)"
    ),
    factory_function="lobster.agents.data_expert.data_expert.data_expert",
    handoff_tool_name="handoff_to_data_expert_agent",
    handoff_tool_description=(
        "Assign LOCAL data operations ONLY: execute downloads from validated queue entries, "
        "load/convert local files via adapters, manage modalities (list/inspect/remove/validate), "
        "retry failed downloads, list/read/write workspace files, extract archives, "
        "debug file format issues. "
        "NEVER use for domain analysis tasks — computing QC metrics, call rates, "
        "statistical tests, normalization, clustering, or biological interpretation "
        "MUST go to the appropriate domain expert (transcriptomics_expert, genomics_expert, "
        "proteomics_expert, metabolomics_expert). Data expert loads data; domain experts analyze it."
    ),
    child_agents=["metadata_assistant"],
)

# === Heavy imports below ===
from pathlib import Path
from typing import List, Optional

import pandas as pd
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

# Agent-specific imports (new modular structure)
from lobster.agents.data_expert.prompts import create_data_expert_prompt
from lobster.agents.data_expert.state import DataExpertState

# Core Lobster imports
from lobster.config.llm_factory import create_llm
from lobster.config.settings import get_settings
from lobster.core.runtime.data_manager import DataManagerV2
from lobster.core.runtime.workspace import WORKSPACE_FOLDER_NAME
from lobster.core.schemas.download_queue import DownloadStatus, ValidationStatus

# Service imports
from lobster.services.execution.registry import (
    CustomCodeExecutionService,
    SDKDelegationError,
    SDKDelegationService,
)

# Tool imports
from lobster.tools.custom_code_tool import create_execute_custom_code_tool
from lobster.tools.download_orchestrator import DownloadOrchestrator
from lobster.tools.filesystem_tools import create_filesystem_tools
from lobster.tools.workspace_tool import create_list_modalities_tool
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


def _normalize_download_strategy_name(strategy_name: Optional[str]) -> Optional[str]:
    """Normalize AUTO/empty strategies to no explicit override."""
    if not strategy_name:
        return None

    normalized = strategy_name.strip().upper()
    if not normalized or normalized == "AUTO":
        return None

    return normalized


def _resolve_filesystem_root(
    data_manager: DataManagerV2,
    workspace_path: Optional[Path],
) -> Optional[Path]:
    """Return the human-facing file root for agent file tools.

    Lobster stores runtime/session state in ``.lobster_workspace``, but chat
    users expect file tools and delegated agents to operate on the enclosing
    worktree/project root. When the resolved workspace points at the storage
    directory, expose its parent as the file root instead.
    """
    candidate = workspace_path or getattr(data_manager, "workspace_path", None)
    if candidate is None:
        return None

    resolved = Path(candidate).resolve()
    if resolved.name == WORKSPACE_FOLDER_NAME and resolved.parent != resolved:
        return resolved.parent
    return resolved


def _resolve_relative_file_argument(file_root: Optional[Path], file_path: str) -> str:
    """Resolve relative user file inputs against the agent file root."""
    file_path_obj = Path(file_path)
    if file_root is None or file_path_obj.is_absolute():
        return file_path
    return str((file_root / file_path_obj).resolve())


def data_expert(
    data_manager: DataManagerV2,
    callback_handler=None,
    agent_name: str = "data_expert_agent",
    delegation_tools: list = None,
    workspace_path: Optional[Path] = None,
    provider_override: Optional[str] = None,
    model_override: Optional[str] = None,
):
    """
    Create a multi-omics data acquisition, processing, and workspace management specialist agent.

    This expert agent serves as the primary interface for all data-related operations in the
    Lobster bioinformatics platform, specializing in:

    - **GEO Data Acquisition**: Fetching, validating, and downloading datasets from NCBI GEO
    - **Local File Processing**: Loading and validating custom data files with automatic format detection
    - **Workspace Management**: Restoring previous sessions and managing dataset persistence
    - **Multi-modal Integration**: Handling transcriptomics, proteomics, and other omics data types
    - **Quality Assurance**: Ensuring data integrity through schema validation and provenance tracking

    Built on the modular DataManagerV2 architecture, this agent provides seamless integration
    with downstream analysis workflows while maintaining professional scientific standards.

    Args:
        data_manager: DataManagerV2 instance for modular data operations
        callback_handler: Optional callback handler for LLM interactions
        agent_name: Name identifier for the agent instance
        delegation_tools: Optional delegation tools for sub-agent handoffs
        workspace_path: Optional workspace path for config resolution

    Returns:
        Configured ReAct agent with comprehensive data management capabilities
    """

    settings = get_settings()
    model_params = settings.get_agent_llm_params("data_expert_agent")
    llm = create_llm(
        "data_expert_agent",
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

    # Initialize modality management service
    from lobster.services.data_management.modality_management_service import (
        ModalityManagementService,
    )

    modality_service = ModalityManagementService(data_manager)

    # Initialize execution services
    custom_code_service = CustomCodeExecutionService(data_manager)

    # Create shared execute_custom_code tool via factory (v2.7+: unified tool)
    execute_custom_code = create_execute_custom_code_tool(
        data_manager=data_manager,
        custom_code_service=custom_code_service,
        agent_name="data_expert",
        post_processor=None,  # data_expert has no special post-processing
    )
    execute_custom_code.metadata = {"categories": ["CODE_EXEC"], "provenance": False}
    execute_custom_code.tags = ["CODE_EXEC"]

    # Try to initialize SDK delegation service (may fail if SDK not available)
    # SDKDelegationService may be None in open-core distribution
    if SDKDelegationService is not None:
        try:
            SDKDelegationService(data_manager)
        except SDKDelegationError as e:
            logger.debug(f"SDK delegation not available: {e}")
        except Exception as e:
            logger.debug(f"SDK delegation initialization failed: {e}")

    # Define tools for data operations
    @tool
    def execute_download_from_queue(
        entry_id: str,
        concatenation_strategy: str = "auto",
        force_download: bool = False,
        strategy_override: str = "",
    ) -> str:
        """Execute download from a queue entry prepared by research_agent.

        Check get_queue_status() first to find entry_id. Entry must be PENDING or FAILED.

        Args:
            entry_id: Queue entry ID (format: queue_GSE12345_abc123)
            concatenation_strategy: "auto" (recommended), "union", or "intersection"
            force_download: Proceed despite validation warnings (default: False)
            strategy_override: Override strategy for retries: "MATRIX_FIRST", "SUPPLEMENTARY_FIRST", "H5_FIRST", "RAW_FIRST"
        """
        try:
            # 1. RETRIEVE QUEUE ENTRY
            if entry_id not in [
                e.entry_id for e in data_manager.download_queue.list_entries()
            ]:
                available = [
                    e.entry_id for e in data_manager.download_queue.list_entries()
                ]
                return (
                    f"Error: Queue entry '{entry_id}' not found. Available: {available}"
                )

            entry = data_manager.download_queue.get_entry(entry_id)

            # Check validation status and warn if issues detected
            if (
                hasattr(entry, "validation_status")
                and entry.validation_status == ValidationStatus.VALIDATED_WITH_WARNINGS
                and not force_download
            ):
                warnings = []
                if entry.validation_result:
                    warnings = entry.validation_result.get("warnings", [])

                # Get strategy info
                strategy_info = ""
                if entry.recommended_strategy:
                    strategy_info = f"""
**Recommended Download Strategy:**
- Strategy: {entry.recommended_strategy.strategy_name}
- Confidence: {entry.recommended_strategy.confidence:.2f}
- Rationale: {entry.recommended_strategy.rationale}
- Concatenation: {entry.recommended_strategy.concatenation_strategy}
"""

                warning_msg = f"""
⚠️ **Dataset has validation warnings but is downloadable**

Entry ID: {entry_id}
Dataset: {entry.dataset_id}

**Validation Warnings:**
{chr(10).join(f"  • {w}" for w in warnings[:5])}
{f"  ... and {len(warnings) - 5} more warnings" if len(warnings) > 5 else ""}
{strategy_info}

**Options:**
1. **Proceed with download**: execute_download_from_queue(entry_id="{entry_id}", force_download=True)
2. **Skip dataset**: Look for alternative datasets with cleaner validation

**Recommendation:** Review warnings above. If warnings are acceptable (e.g., missing optional metadata fields), proceed with force_download=True.
"""
                return warning_msg

            # Verify entry is downloadable
            if entry.status == DownloadStatus.COMPLETED:
                return (
                    f"Entry '{entry_id}' already downloaded as '{entry.modality_name}'"
                )
            elif entry.status == DownloadStatus.IN_PROGRESS:
                return (
                    f"Entry '{entry_id}' is currently being downloaded by another agent"
                )

            # 2. BUILD STRATEGY OVERRIDE (if applicable)
            strategy_override_dict = None
            if (
                entry.recommended_strategy
                or concatenation_strategy != "auto"
                or strategy_override
            ):
                strategy_override_dict = {}
                if entry.recommended_strategy:
                    recommended_strategy_name = _normalize_download_strategy_name(
                        entry.recommended_strategy.strategy_name
                    )
                    if recommended_strategy_name:
                        strategy_override_dict["strategy_name"] = (
                            recommended_strategy_name
                        )
                # Map concatenation_strategy to strategy_params
                if concatenation_strategy != "auto":
                    use_intersecting = concatenation_strategy == "intersection"
                    strategy_override_dict["strategy_params"] = {
                        "use_intersecting_genes_only": use_intersecting
                    }
                # Explicit strategy_override takes precedence
                if strategy_override:
                    explicit_strategy_name = _normalize_download_strategy_name(
                        strategy_override
                    )
                    if explicit_strategy_name:
                        strategy_override_dict["strategy_name"] = explicit_strategy_name
                if not strategy_override_dict:
                    strategy_override_dict = None

            logger.debug(
                f"Starting download for {entry.dataset_id} from queue entry {entry_id}"
            )

            # 3. EXECUTE DOWNLOAD VIA ORCHESTRATOR
            # DownloadOrchestrator routes to the correct service (GEO, PRIDE, SRA, MassIVE)
            # based on the queue entry's database type, handles status management,
            # modality storage, and provenance logging.
            orchestrator = DownloadOrchestrator(data_manager)

            try:
                modality_name, stats = orchestrator.execute_download(
                    entry_id, strategy_override_dict
                )

                result_adata = data_manager.get_modality(modality_name)

                logger.info(f"Download complete: {entry.dataset_id} → {modality_name}")

                # 4. POST-DOWNLOAD VALIDATION GATE
                # Cross-check loaded data characteristics against expected data type
                type_warning = ""
                n_vars = result_adata.n_vars

                # Detect actual modality from data characteristics
                has_proteomics_cols = any(
                    col in result_adata.var.columns
                    for col in [
                        "n_peptides",
                        "sequence_coverage",
                        "protein_group",
                        "peptide_count",
                    ]
                )
                has_transcriptomics_cols = any(
                    col in result_adata.var.columns
                    for col in ["mt", "ribo", "gene_ids", "feature_types"]
                )

                # Feature count heuristics
                likely_proteomics = n_vars < 12000 or has_proteomics_cols
                likely_transcriptomics = n_vars > 15000 or has_transcriptomics_cols

                # Check if expected type (from database or modality name) mismatches
                is_pride_source = entry.database.lower() == "pride"
                modality_lower = modality_name.lower()
                labeled_as_proteomics = (
                    is_pride_source
                    or "proteomics" in modality_lower
                    or "protein" in modality_lower
                )
                labeled_as_transcriptomics = (
                    "transcriptomics" in modality_lower or "rna" in modality_lower
                )

                if (
                    labeled_as_proteomics
                    and likely_transcriptomics
                    and not likely_proteomics
                ):
                    type_warning = (
                        f"\n⚠️ **Data type mismatch detected**: Dataset was labeled as proteomics "
                        f"but loaded data has {n_vars} features, which is characteristic of "
                        f"transcriptomics (RNA-seq typically has 20,000-60,000 genes). "
                        f"This may be an RNA-seq dataset, not proteomics. "
                        f"Verify before running proteomics-specific analysis.\n"
                    )
                elif (
                    labeled_as_transcriptomics
                    and likely_proteomics
                    and not likely_transcriptomics
                ):
                    type_warning = (
                        f"\n⚠️ **Data type mismatch detected**: Dataset was labeled as transcriptomics "
                        f"but loaded data has only {n_vars} features, which may indicate "
                        f"proteomics data (typically 500-10,000 proteins). "
                        f"Verify before running transcriptomics-specific analysis.\n"
                    )

                if type_warning:
                    logger.warning(
                        f"Data type mismatch for {entry.dataset_id}: {type_warning.strip()}"
                    )

                # 5. RETURN SUCCESS REPORT
                strategy_used = stats.get("strategy_used", "auto")
                if strategy_used == "auto":
                    strategy_used = "auto-detected"
                concatenation_used = stats.get(
                    "concatenation_strategy", concatenation_strategy
                )

                response = f"""
✅ **Download completed successfully**

Dataset ID: {entry.dataset_id}
Entry ID: {entry_id}
Database: {entry.database}
Modality Name: {modality_name}
Strategy Used: {strategy_used}

Samples: {result_adata.n_obs}
Features: {result_adata.n_vars}
Concatenation: {concatenation_used}
{type_warning}
You can now analyze this dataset using the appropriate analysis tools.
"""

                data_manager.log_tool_usage(
                    tool_name="execute_download_from_queue",
                    parameters={"entry_id": entry_id, "dataset_id": entry.dataset_id},
                    description=f"Downloaded {entry.dataset_id} → {modality_name}",
                    ir=None,
                )
                return response

            except Exception as download_error:
                logger.error(
                    f"Download failed for {entry.dataset_id}: {download_error}"
                )

                response = f"## Download Failed: {entry.dataset_id}\n\n"
                response += "❌ **Status**: Download failed\n"
                response += f"- **Error**: {download_error}\n"
                response += f"- **Database**: {entry.database}\n"
                response += f"- **Queue entry**: `{entry_id}` (FAILED)\n"
                response += "\n**Troubleshooting**:\n"
                response += f"1. Check dataset availability: {entry.dataset_id}\n"
                response += "2. Verify URLs are accessible\n"
                response += "3. Try a different download strategy:\n"
                response += '   - strategy_override="MATRIX_FIRST"\n'
                response += '   - strategy_override="SUPPLEMENTARY_FIRST"\n'
                response += '   - strategy_override="H5_FIRST"\n'
                response += "4. Try different concatenation mode:\n"
                response += (
                    '   - concatenation_strategy="union" (preserves all genes)\n'
                )
                response += (
                    '   - concatenation_strategy="intersection" (only common genes)\n'
                )
                response += (
                    "5. Review error log: `get_queue_status(status_filter='FAILED')`\n"
                )

                return response

        except Exception as e:
            logger.error(f"Error in execute_download_from_queue: {e}")
            return f"Error processing queue entry '{entry_id}': {str(e)}"

    execute_download_from_queue.metadata = {
        "categories": ["IMPORT"],
        "provenance": True,
    }
    execute_download_from_queue.tags = ["IMPORT"]

    # Use shared tool from workspace_tool.py (shared with supervisor)
    list_available_modalities = create_list_modalities_tool(data_manager)
    list_available_modalities.metadata = {
        "categories": ["UTILITY"],
        "provenance": False,
    }
    list_available_modalities.tags = ["UTILITY"]

    @tool
    def get_modality_details(modality_name: str) -> str:
        """
        Get detailed information about a specific modality.

        Args:
            modality_name: Name of the modality to inspect

        Returns:
            str: Detailed modality information
        """
        try:
            info, stats, ir = modality_service.get_modality_info(modality_name)

            # Log to provenance
            data_manager.log_tool_usage(
                tool_name="get_modality_details",
                parameters={"modality_name": modality_name},
                description=f"Retrieved info for {modality_name}: {info['shape']['n_obs']} obs x {info['shape']['n_vars']} vars",
                ir=ir,
            )

            # Format response
            response = f"## Modality: {info['name']}\n\n"
            response += f"**Shape**: {info['shape']['n_obs']} obs × {info['shape']['n_vars']} vars\n"
            response += f"**Sparse**: {info['is_sparse']}\n\n"

            response += "**Observation Columns**:\n"
            response += f"  {', '.join(info['obs_columns'][:10])}\n"
            if len(info["obs_columns"]) > 10:
                response += f"  ... and {len(info['obs_columns']) - 10} more\n"

            response += "\n**Variable Columns**:\n"
            response += f"  {', '.join(info['var_columns'][:10])}\n"
            if len(info["var_columns"]) > 10:
                response += f"  ... and {len(info['var_columns']) - 10} more\n"

            if info["layers"]:
                response += f"\n**Layers**: {', '.join(info['layers'])}\n"
            if info["obsm_keys"]:
                response += f"**Obsm Keys**: {', '.join(info['obsm_keys'])}\n"
            if info["varm_keys"]:
                response += f"**Varm Keys**: {', '.join(info['varm_keys'])}\n"
            if info["uns_keys"]:
                response += f"**Uns Keys**: {', '.join(info['uns_keys'])}\n"

            if info["quality_metrics"]:
                response += f"\n**Quality Metrics**: {len(info['quality_metrics'])} metrics available\n"

            return response

        except Exception as e:
            logger.error(
                f"Error getting modality details for '{modality_name}': {e}",
                exc_info=True,
            )
            try:
                available = data_manager.list_modalities()
            except Exception:
                available = "(unavailable)"
            if isinstance(available, list) and modality_name not in available:
                return (
                    f"Error: Modality '{modality_name}' not found. "
                    f"Available modalities: {available[:15] if available else '(none)'}"
                )
            return (
                f"Error getting modality details for '{modality_name}': {str(e)}\n"
                f"The modality exists but inspection failed. "
                f"Try list_available_modalities() for a summary."
            )

    get_modality_details.metadata = {"categories": ["UTILITY"], "provenance": False}
    get_modality_details.tags = ["UTILITY"]

    @tool
    def remove_modality(modality_name: str) -> str:
        """
        Remove a modality from memory using the modality management service.

        Args:
            modality_name: Name of modality to remove

        Returns:
            str: Status of removal operation
        """
        try:
            success, stats, ir = modality_service.remove_modality(modality_name)

            # Log to provenance
            data_manager.log_tool_usage(
                tool_name="remove_modality",
                parameters={"modality_name": modality_name},
                description=f"Removed modality {stats['removed_modality']}: {stats['shape']['n_obs']} obs x {stats['shape']['n_vars']} vars",
                ir=ir,
            )

            response = f"## Removed Modality: {stats['removed_modality']}\n\n"
            response += f"**Shape**: {stats['shape']['n_obs']} obs × {stats['shape']['n_vars']} vars\n"
            response += f"**Remaining modalities**: {stats['remaining_modalities']}\n"

            return response

        except Exception as e:
            logger.error(f"Error removing modality {modality_name}: {e}")
            return f"Error removing modality: {str(e)}"

    remove_modality.metadata = {"categories": ["UTILITY"], "provenance": False}
    remove_modality.tags = ["UTILITY"]

    @tool
    def validate_modality_compatibility(modality_names: List[str]) -> str:
        """
        Validate compatibility between multiple modalities for integration.

        Checks observation/variable overlap, batch effects, and provides recommendations.

        Args:
            modality_names: List of modality names to validate for compatibility

        Returns:
            str: Compatibility validation report with recommendations
        """
        try:
            validation, stats, ir = modality_service.validate_compatibility(
                modality_names
            )

            # Log to provenance
            data_manager.log_tool_usage(
                tool_name="validate_modality_compatibility",
                parameters={"modality_names": modality_names},
                description=f"Validated compatibility of {len(modality_names)} modalities: {'compatible' if validation['compatible'] else 'issues detected'}",
                ir=ir,
            )

            # Format response
            response = "## Modality Compatibility Report\n\n"
            response += f"**Status**: {'✅ Compatible' if validation['compatible'] else '⚠️ Issues Detected'}\n"
            response += f"**Modalities**: {', '.join(validation['modalities'])}\n\n"

            response += "**Overlap Analysis**:\n"
            response += (
                f"  - Shared observations: {validation['shared_observations']}\n"
            )
            response += f"  - Observation overlap: {validation['observation_overlap_rate']:.1%}\n"
            response += f"  - Shared variables: {validation['shared_variables']}\n"
            response += (
                f"  - Variable overlap: {validation['variable_overlap_rate']:.1%}\n"
            )

            if validation["batch_columns"]:
                response += (
                    f"\n**Batch Columns**: {', '.join(validation['batch_columns'])}\n"
                )

            if validation["issues"]:
                response += "\n**Issues**:\n"
                for issue in validation["issues"]:
                    response += f"  - {issue}\n"

            response += "\n**Recommendations**:\n"
            for rec in validation["recommendations"]:
                response += f"  - {rec}\n"

            return response

        except Exception as e:
            logger.error(f"Error validating compatibility: {e}")
            return f"Error validating compatibility: {str(e)}"

    validate_modality_compatibility.metadata = {
        "categories": ["QUALITY"],
        "provenance": True,
    }
    validate_modality_compatibility.tags = ["QUALITY"]

    @tool
    def load_modality(
        modality_name: str,
        file_path: str,
        adapter: str,
        dataset_type: str = "custom",
    ) -> str:
        """Load a local data file as a modality. Use list_files/glob_files FIRST to verify the file.

        Args:
            modality_name: Name for the new modality
            file_path: Path to data file (relative to workspace or absolute)
            adapter: Adapter name (call get_adapter_info() to see options)
            dataset_type: Source type ("custom", "geo", "local")
        """
        try:
            resolved_file_path = _resolve_relative_file_argument(
                fs_workspace, file_path
            )
            adata, stats, ir = modality_service.load_modality(
                modality_name=modality_name,
                file_path=resolved_file_path,
                adapter=adapter,
                dataset_type=dataset_type,
                validate=True,
            )

            # Log to provenance
            data_manager.log_tool_usage(
                tool_name="load_modality",
                parameters={
                    "modality_name": modality_name,
                    "file_path": resolved_file_path,
                    "adapter": adapter,
                    "dataset_type": dataset_type,
                },
                description=f"Loaded modality {stats['modality_name']}: {stats['shape']['n_obs']} obs x {stats['shape']['n_vars']} vars via {stats['adapter']}",
                ir=ir,
            )

            # Format response
            response = f"## Loaded Modality: {stats['modality_name']}\n\n"
            response += f"**Shape**: {stats['shape']['n_obs']} obs × {stats['shape']['n_vars']} vars\n"
            response += f"**Adapter**: {stats['adapter']}\n"
            response += f"**File**: {stats['file_path']}\n"
            response += f"**Dataset Type**: {stats['dataset_type']}\n"
            response += f"**Validation**: {stats['validation_status']}\n"
            response += f"**Quality Metrics**: {stats['quality_metrics_count']} metrics calculated\n"
            response += (
                f"\nThe modality '{modality_name}' is now available for analysis.\n"
            )

            return response

        except Exception as e:
            logger.error(f"Error loading modality {modality_name}: {e}", exc_info=True)
            try:
                available = data_manager.list_modalities()
            except Exception:
                available = "(unavailable)"
            return (
                f"Error loading modality: {str(e)}\n"
                f"Parameters: file_path='{resolved_file_path if 'resolved_file_path' in locals() else file_path}', adapter='{adapter}', dataset_type='{dataset_type}'\n"
                f"Existing modalities: {available[:15] if isinstance(available, list) else available if available else '(none)'}\n"
                f"Hints: verify file exists with list_files or glob_files. "
                f"Check adapter name with get_adapter_info()."
            )

    load_modality.metadata = {"categories": ["IMPORT"], "provenance": True}
    load_modality.tags = ["IMPORT"]

    @tool
    def create_mudata_from_modalities(
        modality_names: List[str], output_name: str = "multimodal_analysis"
    ) -> str:
        """
        Create a MuData object from multiple loaded modalities for integrated analysis.

        Args:
            modality_names: List of modality names to combine
            output_name: Name for the output file

        Returns:
            str: Status of MuData creation
        """
        try:
            # Check that all modalities exist
            available_modalities = data_manager.list_modalities()
            missing = [
                name for name in modality_names if name not in available_modalities
            ]
            if missing:
                return f"Modalities not found: {missing}. Available: {available_modalities}"

            # Create MuData object
            mdata = data_manager.to_mudata(modalities=modality_names)

            # Save the MuData object
            mudata_path = f"{output_name}.h5mu"
            data_manager.save_mudata(mudata_path, modalities=modality_names)

            data_manager.log_tool_usage(
                tool_name="create_mudata_from_modalities",
                parameters={
                    "modality_names": modality_names,
                    "output_name": output_name,
                },
                description=f"Created MuData from {len(modality_names)} modalities → {mudata_path}",
                ir=None,
            )
            return f"""Successfully created MuData from {len(modality_names)} modalities.

Combined modalities: {", ".join(modality_names)}
Global shape: {mdata.n_obs} obs across {len(mdata.mod)} modalities
Saved to: {mudata_path}
Ready for integrated multi-omics analysis

The MuData object contains all selected modalities and is ready for cross-modal analysis."""

        except Exception as e:
            logger.error(f"Error creating MuData: {e}")
            return f"Error creating MuData: {str(e)}"

    create_mudata_from_modalities.metadata = {
        "categories": ["PREPROCESS"],
        "provenance": True,
    }
    create_mudata_from_modalities.tags = ["PREPROCESS"]

    @tool
    def get_adapter_info() -> str:
        """List all registered data adapters and their supported file formats.

        Call before load_modality() to determine which adapter to use.
        """
        try:
            adapter_info = data_manager.get_adapter_info()

            response = "Available Data Adapters:\n\n"

            for adapter_name, info in adapter_info.items():
                response += f"**{adapter_name}**:\n"
                response += f"  - Modality: {info['modality_name']}\n"
                response += (
                    f"  - Supported formats: {', '.join(info['supported_formats'])}\n"
                )
                response += f"  - Schema: {info['schema']['modality']}\n"
                response += "\n"

            response += "\nUse these adapter names when loading data with load_modality_from_file or upload_data_file."

            return response

        except Exception as e:
            logger.error(f"Error getting adapter info: {e}")
            return f"Error getting adapter info: {str(e)}"

    get_adapter_info.metadata = {"categories": ["UTILITY"], "provenance": False}
    get_adapter_info.tags = ["UTILITY"]

    @tool
    def concatenate_samples(
        sample_modalities: List[str] = None,
        output_modality_name: str = None,
        geo_id: str = None,
        use_intersecting_genes_only: bool = True,
        save_to_file: bool = True,
    ) -> str:
        """Concatenate multiple sample modalities into one combined modality.

        Use after SAMPLES_FIRST download creates multiple per-sample modalities.

        Args:
            sample_modalities: Modality names to merge (None = auto-detect from geo_id)
            output_modality_name: Output name (None = auto-generate)
            geo_id: GEO ID for auto-detection (e.g. "GSE12345")
            use_intersecting_genes_only: True=common genes only, False=all genes (zero-fill)
            save_to_file: Save concatenated result (default: True)
        """
        try:
            # Import the ConcatenationService
            from lobster.services.data_management.concatenation_service import (
                ConcatenationService,
            )

            # Initialize the concatenation service
            concat_service = ConcatenationService(data_manager)

            # Auto-detect sample modalities if not provided
            if sample_modalities is None:
                if geo_id is None:
                    return "Either provide sample_modalities list or geo_id for auto-detection"

                clean_geo_id = geo_id.strip().upper()
                sample_modalities = concat_service.auto_detect_samples(
                    f"geo_{clean_geo_id.lower()}"
                )

                if not sample_modalities:
                    return f"No sample modalities found for {clean_geo_id}"

                logger.info(
                    f"Auto-detected {len(sample_modalities)} samples for {clean_geo_id}"
                )

            # Generate output name if not provided
            if output_modality_name is None:
                if geo_id:
                    output_modality_name = f"geo_{geo_id.lower()}_concatenated"
                else:
                    prefix = (
                        sample_modalities[0].rsplit("_sample_", 1)[0]
                        if "_sample_" in sample_modalities[0]
                        else sample_modalities[0].split("_")[0]
                    )
                    output_modality_name = f"{prefix}_concatenated"

            # Check if output modality already exists
            if output_modality_name in data_manager.list_modalities():
                return f"Modality '{output_modality_name}' already exists. Use remove_modality first or choose a different name."

            # Use ConcatenationService for the actual concatenation
            concatenated_adata, statistics, ir = (
                concat_service.concatenate_from_modalities(
                    modality_names=sample_modalities,
                    output_name=output_modality_name if save_to_file else None,
                    use_intersecting_genes_only=use_intersecting_genes_only,
                    batch_key="batch",
                )
            )

            # Add concatenation metadata for provenance tracking
            concatenated_adata.uns["concatenation_metadata"] = {
                "dataset_type": "concatenated_samples",
                "source_modalities": sample_modalities,
                "processing_date": pd.Timestamp.now().isoformat(),
                "concatenation_strategy": statistics.get(
                    "strategy_used", "smart_sparse"
                ),
                "concatenation_info": statistics,
            }

            # Store the concatenated result in DataManager (following tool pattern)
            data_manager.store_modality(
                name=output_modality_name,
                adata=concatenated_adata,
                step_summary=f"Concatenated {len(sample_modalities)} samples",
            )

            # Log the concatenation operation for provenance
            data_manager.log_tool_usage(
                tool_name="concatenate_samples",
                parameters={
                    "sample_modalities": sample_modalities,
                    "output_modality_name": output_modality_name,
                    "use_intersecting_genes_only": use_intersecting_genes_only,
                    "save_to_file": save_to_file,
                },
                description=f"Concatenated {len(sample_modalities)} samples into modality '{output_modality_name}'",
                ir=ir,
            )

            # Format results for user display
            if save_to_file:
                return f"""Successfully concatenated {statistics["n_samples"]} samples using ConcatenationService.

Output modality: '{output_modality_name}'
Shape: {statistics["final_shape"][0]} obs × {statistics["final_shape"][1]} vars
Join type: {statistics["join_type"]}
Strategy: {statistics["strategy_used"]}
Processing time: {statistics.get("processing_time_seconds", 0):.2f}s
Saved and stored as modality for analysis

The concatenated dataset is now available as modality '{output_modality_name}' for analysis."""
            else:
                return f"""Concatenation preview (not saved):

Shape: {statistics["final_shape"][0]} obs × {statistics["final_shape"][1]} vars
Join type: {statistics["join_type"]}
Strategy: {statistics["strategy_used"]}

To save, run again with save_to_file=True"""

        except Exception as e:
            logger.error(f"Error concatenating samples: {e}", exc_info=True)
            try:
                available = data_manager.list_modalities()
            except Exception:
                available = "(unavailable)"
            samples_str = (
                str(sample_modalities)
                if sample_modalities
                else "(auto-detect failed or not provided)"
            )
            output_str = (
                output_modality_name if output_modality_name else "(not yet determined)"
            )
            return (
                f"Error concatenating samples: {str(e)}\n"
                f"Parameters: sample_modalities={samples_str}, "
                f"output_name='{output_str}', geo_id={geo_id}\n"
                f"Available modalities: {available[:15] if isinstance(available, list) else available if available else '(none)'}\n"
                f"Hint: verify sample modalities exist with list_available_modalities()."
            )

    concatenate_samples.metadata = {"categories": ["PREPROCESS"], "provenance": True}
    concatenate_samples.tags = ["PREPROCESS"]

    @tool
    def get_queue_status(
        status_filter: str = None,
        dataset_id_filter: str = None,
    ) -> str:
        """Show download queue status with optional filtering.

        Args:
            status_filter: "PENDING", "IN_PROGRESS", "COMPLETED", "FAILED", or "all" (default: all)
            dataset_id_filter: Filter by dataset ID (e.g. "GSE12345")
        """
        try:
            # Get all queue entries
            all_entries = data_manager.download_queue.list_entries()

            if not all_entries:
                return "## Download Queue Status\n\n📭 Queue is empty - no pending downloads"

            # Apply filters
            filtered_entries = all_entries

            if status_filter and status_filter.upper() != "ALL":
                try:
                    filter_status = DownloadStatus[status_filter.upper()]
                    filtered_entries = [
                        e for e in filtered_entries if e.status == filter_status
                    ]
                except KeyError:
                    return f"Invalid status filter '{status_filter}'. Valid options: PENDING, IN_PROGRESS, COMPLETED, FAILED, all"

            if dataset_id_filter:
                dataset_pattern = dataset_id_filter.upper()
                filtered_entries = [
                    e
                    for e in filtered_entries
                    if dataset_pattern in e.dataset_id.upper()
                ]

            # Group entries by status for better readability
            status_groups = {
                DownloadStatus.PENDING: [],
                DownloadStatus.IN_PROGRESS: [],
                DownloadStatus.COMPLETED: [],
                DownloadStatus.FAILED: [],
            }

            for entry in filtered_entries:
                status_groups[entry.status].append(entry)

            # Build response
            response = "## Download Queue Status\n\n"

            # Summary counts
            response += "**Summary**:\n"
            response += f"- Total entries: {len(all_entries)}\n"
            if status_filter or dataset_id_filter:
                response += f"- Filtered entries: {len(filtered_entries)}\n"
            response += f"- Pending: {len(status_groups[DownloadStatus.PENDING])}\n"
            response += (
                f"- In Progress: {len(status_groups[DownloadStatus.IN_PROGRESS])}\n"
            )
            response += f"- Completed: {len(status_groups[DownloadStatus.COMPLETED])}\n"
            response += f"- Failed: {len(status_groups[DownloadStatus.FAILED])}\n\n"

            if not filtered_entries:
                response += "No entries match the specified filters.\n"
                return response

            # Detailed entries by status
            for status in [
                DownloadStatus.PENDING,
                DownloadStatus.IN_PROGRESS,
                DownloadStatus.COMPLETED,
                DownloadStatus.FAILED,
            ]:
                entries = status_groups[status]
                if not entries:
                    continue

                # Status section header with emoji
                status_emoji = {
                    DownloadStatus.PENDING: "⏳",
                    DownloadStatus.IN_PROGRESS: "🔄",
                    DownloadStatus.COMPLETED: "✅",
                    DownloadStatus.FAILED: "❌",
                }
                response += (
                    f"### {status_emoji[status]} {status.value} ({len(entries)})\n\n"
                )

                # Table header
                response += (
                    "| Entry ID | Dataset ID | Database | Priority | Modality |\n"
                )
                response += (
                    "|----------|------------|----------|----------|----------|\n"
                )

                for entry in entries:
                    modality_display = entry.modality_name or "-"
                    response += f"| `{entry.entry_id}` | {entry.dataset_id} | {entry.database} | {entry.priority} | {modality_display} |\n"

                # Show error details for failed entries
                if status == DownloadStatus.FAILED:
                    response += "\n**Error Details**:\n"
                    for entry in entries:
                        if entry.error_log:
                            response += f"- `{entry.entry_id}`: {entry.error_log[-1]}\n"

                response += "\n"

            # Log tool usage
            data_manager.log_tool_usage(
                tool_name="get_queue_status",
                parameters={
                    "status_filter": status_filter,
                    "dataset_id_filter": dataset_id_filter,
                    "total_entries": len(all_entries),
                    "filtered_entries": len(filtered_entries),
                },
                description=f"Retrieved queue status: {len(filtered_entries)} entries",
            )

            return response

        except Exception as e:
            logger.error(f"Error getting queue status: {e}")
            return f"Error getting queue status: {str(e)}"

    get_queue_status.metadata = {"categories": ["UTILITY"], "provenance": False}
    get_queue_status.tags = ["UTILITY"]

    # execute_custom_code is created via factory at line ~96 (v2.7+: unified tool)
    # See create_execute_custom_code_tool() in lobster/tools/custom_code_tool.py

    # @tool
    # def delegate_complex_reasoning( #TODO DEACTIVATED FOR NOW
    #     task: str,
    #     context: Optional[str] = None,
    #     persist: bool = False
    # ) -> str:
    #     """
    #     Delegate complex multi-step reasoning to Claude Agent SDK sub-agent.

    #     **Use this tool when you need:**
    #     - Multi-step analysis planning
    #     - Complex troubleshooting ("Why does my data look wrong?")
    #     - Integration strategy recommendations
    #     - Experimental design reasoning

    #     The sub-agent has READ-ONLY access to:
    #     - List available modalities
    #     - Inspect modality details (shape, columns, quality metrics)
    #     - List workspace files

    #     Args:
    #         task: Clear description of the reasoning task
    #         context: Optional additional context about the situation
    #         persist: If True, save reasoning to provenance/notebook export

    #     Returns:
    #         Formatted reasoning result from SDK sub-agent

    #     Example:
    #         >>> delegate_complex_reasoning(
    #         ...     task="Why do I have 15 clusters when the paper reports 7?",
    #         ...     context="Dataset: geo_gse12345 with 5000 cells, paper had 3000 cells",
    #         ...     persist=False
    #         ... )
    #     """
    #     if not sdk_available:
    #         return "❌ SDK delegation not available. Claude Agent SDK is not installed or not accessible."

    #     try:
    #         reasoning_result, stats, ir = sdk_delegation_service.delegate(
    #             task=task,
    #             context=context,
    #             persist=persist,
    #             description=f"SDK Reasoning: {task[:100]}"
    #         )

    #         # Log to data manager
    #         data_manager.log_tool_usage(
    #             tool_name="delegate_complex_reasoning",
    #             parameters={'task': task[:200], 'persist': persist},
    #             description=f"SDK delegation: {task[:100]}",
    #             ir=ir
    #         )

    #         # Format response
    #         response = f"## SDK Reasoning Result\n\n"
    #         response += f"**Task**: {task}\n\n"
    #         response += f"**Reasoning**:\n{reasoning_result}\n\n"

    #         if persist:
    #             response += "\n📝 This reasoning was saved to provenance.\n"
    #         else:
    #             response += "\n💨 This reasoning was ephemeral (not saved).\n"

    #         return response

    #     except SDKDelegationError as e:
    #         logger.error(f"SDK delegation failed: {e}")
    #         return f"❌ SDK delegation failed: {str(e)}"

    #     except Exception as e:
    #         logger.error(f"Unexpected error in delegate_complex_reasoning: {e}")
    #         return f"❌ Unexpected error: {str(e)}"

    # Create filesystem tools for file-level operations (DeepAgent-inspired)
    fs_workspace = _resolve_filesystem_root(data_manager, workspace_path)
    filesystem_tools = create_filesystem_tools(workspace_path=fs_workspace) if fs_workspace else []

    base_tools = [
        # CORE (3 tools)
        execute_download_from_queue,
        concatenate_samples,
        get_queue_status,
        # MODALITY MANAGEMENT (ModalityManagementService)
        list_available_modalities,
        get_modality_details,
        load_modality,
        remove_modality,
        validate_modality_compatibility,
        # HELPER
        get_adapter_info,
        # ADVANCED (Execution & Reasoning)
        execute_custom_code,
        # delegate_complex_reasoning, #TODO needs further security validation
        # FILESYSTEM (DeepAgent-inspired file-level tools)
        *filesystem_tools,
    ]
    # create_mudata_from_modalities: Combine modalities into MuData for integrated analysis

    tools = base_tools

    # Use prompt from prompts.py (extracted for modularity)
    system_prompt = create_data_expert_prompt()

    # Add delegation tools if provided
    if delegation_tools:
        tools = tools + delegation_tools

    return create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt,
        name=agent_name,
        state_schema=DataExpertState,
    )
