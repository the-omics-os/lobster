"""
Differential Expression Analysis Sub-Agent for specialized DE workflows.

This sub-agent handles all differential expression analysis tools including:
- Pseudobulk aggregation from single-cell data
- Formula-based experimental design
- pyDESeq2 differential expression analysis
- Iterative DE analysis with comparison
- Pathway enrichment analysis

CRITICAL SCIENTIFIC FIXES:
1. DESeq2 requires RAW INTEGER COUNTS from adata.raw.X (not normalized adata.X)
2. Minimum replicate threshold changed from 2 to 3 for stable variance estimation
3. Warning when any condition has fewer than 4 replicates (low statistical power)
"""

# Agent configuration for entry point discovery (must be at top, before heavy imports)
from lobster.config.agent_registry import AgentRegistryConfig

AGENT_CONFIG = AgentRegistryConfig(
    name="de_analysis_expert",
    display_name="DE Analysis Expert",
    description="Differential expression sub-agent: pseudobulk, pyDESeq2, formula-based DE, pathway enrichment",
    factory_function="lobster.agents.transcriptomics.de_analysis_expert.de_analysis_expert",
    handoff_tool_name=None,  # Not directly accessible
    handoff_tool_description=None,
    supervisor_accessible=False,  # Only via transcriptomics_expert
)

# === Heavy imports below ===
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from lobster.agents.transcriptomics.prompts import create_de_analysis_expert_prompt
from lobster.agents.transcriptomics.state import DEAnalysisExpertState
from lobster.config.llm_factory import create_llm
from lobster.config.settings import get_settings
from lobster.core import (
    AggregationError,
    DesignMatrixError,
    FormulaError,
    InsufficientCellsError,
    PseudobulkError,
)
from lobster.core.data_manager_v2 import DataManagerV2
from lobster.services.analysis.bulk_rnaseq_service import (
    BulkRNASeqError,
    BulkRNASeqService,
)
from lobster.services.analysis.differential_formula_service import (
    DifferentialFormulaService,
)
from lobster.services.analysis.pseudobulk_service import PseudobulkService
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


class DEAnalysisError(Exception):
    """Base exception for differential expression analysis errors."""

    pass


class ModalityNotFoundError(DEAnalysisError):
    """Raised when requested modality doesn't exist."""

    pass


class InsufficientReplicatesError(DEAnalysisError):
    """Raised when there are insufficient replicates for stable variance estimation."""

    pass


def de_analysis_expert(
    data_manager: DataManagerV2,
    callback_handler=None,
    agent_name: str = "de_analysis_expert",
    delegation_tools: List = None,
    workspace_path: Optional[Path] = None,
    provider_override: Optional[str] = None,
    model_override: Optional[str] = None,
):
    """
    Create differential expression analysis sub-agent.

    This agent handles all DE-related tasks for transcriptomics workflows.

    Args:
        data_manager: DataManagerV2 instance for modality management
        callback_handler: Optional LangChain callback handler
        agent_name: Name for the agent
        delegation_tools: List of delegation tools from parent agent

    Returns:
        LangGraph react agent configured for DE analysis
    """
    settings = get_settings()
    model_params = settings.get_agent_llm_params("de_analysis_expert")
    llm = create_llm(
        "de_analysis_expert",
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

    # Initialize stateless services
    pseudobulk_service = PseudobulkService()
    bulk_rnaseq_service = BulkRNASeqService(data_manager=data_manager)
    formula_service = DifferentialFormulaService()

    analysis_results = {"summary": "", "details": {}}

    # -------------------------
    # HELPER FUNCTIONS
    # -------------------------
    def _extract_raw_counts(adata) -> Tuple[pd.DataFrame, bool]:
        """
        Extract raw counts from AnnData, preferring adata.raw.X.

        CRITICAL: DESeq2 requires raw integer counts, not normalized data.

        Args:
            adata: AnnData object

        Returns:
            Tuple of (count_matrix DataFrame, used_raw_flag)
        """
        used_raw = False

        # CRITICAL FIX: Use raw counts for DESeq2
        if adata.raw is not None:
            raw_data = adata.raw.X
            if hasattr(raw_data, "toarray"):
                raw_data = raw_data.toarray()
            count_matrix = pd.DataFrame(
                raw_data.T, index=adata.raw.var_names, columns=adata.obs_names
            )
            used_raw = True
            logger.info("Using adata.raw.X for count matrix (recommended for DESeq2)")
        else:
            # Fallback to adata.X with warning
            logger.warning(
                "No adata.raw found - using adata.X which may be normalized. "
                "DESeq2 requires raw counts for accurate results."
            )
            data = adata.X
            if hasattr(data, "toarray"):
                data = data.toarray()
            count_matrix = pd.DataFrame(
                data.T, index=adata.var_names, columns=adata.obs_names
            )

        return count_matrix, used_raw

    def _validate_replicate_counts(
        metadata: pd.DataFrame,
        groupby: str,
        min_replicates: int = 3,  # SCIENTIFIC FIX: Changed from 2 to 3
    ) -> Dict[str, Any]:
        """
        Validate replicate counts per condition.

        Args:
            metadata: Sample metadata DataFrame
            groupby: Column name for grouping
            min_replicates: Minimum required replicates (default: 3)

        Returns:
            Dict with validation results
        """
        group_counts = metadata[groupby].value_counts().to_dict()

        validation = {
            "valid": True,
            "group_counts": group_counts,
            "warnings": [],
            "errors": [],
        }

        for group, count in group_counts.items():
            if count < min_replicates:
                validation["valid"] = False
                validation["errors"].append(
                    f"Group '{group}' has only {count} replicates "
                    f"(minimum {min_replicates} required for stable variance estimation)"
                )
            # SCIENTIFIC FIX: Add warning when n < 4
            elif count < 4:
                validation["warnings"].append(
                    f"Group '{group}' has {count} replicates. "
                    f"Statistical power may be limited. 4+ replicates recommended."
                )

        return validation

    # -------------------------
    # PSEUDOBULK TOOLS
    # -------------------------
    @tool
    def create_pseudobulk_matrix(
        modality_name: str,
        sample_col: str,
        celltype_col: str,
        layer: str = None,
        min_cells: int = 10,
        aggregation_method: str = "sum",
        min_genes: int = 200,
        filter_zeros: bool = True,
        save_result: bool = True,
    ) -> str:
        """
        Aggregate single-cell data to pseudobulk matrix for differential expression analysis.

        IMPORTANT: This tool extracts RAW COUNTS from adata.raw.X for DESeq2 compatibility.

        Args:
            modality_name: Name of single-cell modality to aggregate
            sample_col: Column containing sample identifiers
            celltype_col: Column containing cell type identifiers
            layer: Layer to use for aggregation (default: None, uses raw counts)
            min_cells: Minimum cells per sample-celltype combination
            aggregation_method: Aggregation method ('sum' for DESeq2, 'mean', 'median')
            min_genes: Minimum genes detected per pseudobulk sample
            filter_zeros: Remove genes with all zeros after aggregation
            save_result: Whether to save the pseudobulk modality
        """
        try:
            # Validate modality exists
            if modality_name not in data_manager.list_modalities():
                raise ModalityNotFoundError(
                    f"Modality '{modality_name}' not found. "
                    f"Available: {data_manager.list_modalities()}"
                )

            # Get the single-cell modality
            adata = data_manager.get_modality(modality_name)
            logger.info(
                f"Creating pseudobulk matrix from '{modality_name}': "
                f"{adata.n_obs} cells x {adata.n_vars} genes"
            )

            # Validate required columns exist
            if sample_col not in adata.obs.columns:
                available_cols = list(adata.obs.columns)[:10]
                raise PseudobulkError(
                    f"Sample column '{sample_col}' not found. "
                    f"Available columns: {available_cols}..."
                )

            if celltype_col not in adata.obs.columns:
                available_cols = list(adata.obs.columns)[:10]
                raise PseudobulkError(
                    f"Cell type column '{celltype_col}' not found. "
                    f"Available columns: {available_cols}..."
                )

            # Use pseudobulk service (it uses raw counts internally)
            pseudobulk_adata = pseudobulk_service.aggregate_to_pseudobulk(
                adata=adata,
                sample_col=sample_col,
                celltype_col=celltype_col,
                layer=layer,
                min_cells=min_cells,
                aggregation_method=aggregation_method,
                min_genes=min_genes,
                filter_zeros=filter_zeros,
            )

            # Save as new modality
            pseudobulk_modality_name = f"{modality_name}_pseudobulk"
            data_manager.store_modality(
                name=pseudobulk_modality_name,
                adata=pseudobulk_adata,
                parent_name=modality_name,
                step_summary=f"Created pseudobulk: {pseudobulk_adata.n_obs} samples x {pseudobulk_adata.n_vars} genes",
            )

            # Save to file if requested
            if save_result:
                save_path = f"{modality_name}_pseudobulk.h5ad"
                data_manager.save_modality(pseudobulk_modality_name, save_path)

            # Get aggregation statistics
            agg_stats = pseudobulk_adata.uns.get("aggregation_stats", {})

            # Log the operation
            data_manager.log_tool_usage(
                tool_name="create_pseudobulk_matrix",
                parameters={
                    "modality_name": modality_name,
                    "sample_col": sample_col,
                    "celltype_col": celltype_col,
                    "aggregation_method": aggregation_method,
                    "min_cells": min_cells,
                },
                description=f"Created pseudobulk matrix: {pseudobulk_adata.n_obs} samples x {pseudobulk_adata.n_vars} genes",
            )

            # Format response
            response = f"""Pseudobulk matrix created from single-cell data '{modality_name}'!

**Aggregation Results:**
- Original: {adata.n_obs:,} single cells -> {pseudobulk_adata.n_obs} pseudobulk samples
- Genes retained: {pseudobulk_adata.n_vars:,} / {adata.n_vars:,} ({pseudobulk_adata.n_vars / adata.n_vars * 100:.1f}%)
- Aggregation method: {aggregation_method}
- Min cells threshold: {min_cells}

**Sample & Cell Type Distribution:**
- Unique samples: {agg_stats.get("n_samples", "N/A")}
- Cell types: {agg_stats.get("n_cell_types", "N/A")}
- Total cells aggregated: {agg_stats.get("total_cells_aggregated", "N/A"):,}
- Mean cells per pseudobulk: {agg_stats.get("mean_cells_per_pseudobulk", 0):.1f}

**New modality created**: '{pseudobulk_modality_name}'"""

            if save_result:
                response += f"\n**Saved to**: {save_path}"

            response += "\n\nNext step: Use 'prepare_de_design' to set up statistical design for DE analysis."

            analysis_results["details"]["pseudobulk_aggregation"] = response
            return response

        except (
            PseudobulkError,
            AggregationError,
            InsufficientCellsError,
            ModalityNotFoundError,
        ) as e:
            logger.error(f"Error creating pseudobulk matrix: {e}")
            return f"Error creating pseudobulk matrix: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error in pseudobulk creation: {e}")
            return f"Unexpected error: {str(e)}"

    @tool
    def prepare_de_design(
        modality_name: str,
        formula: str,
        contrast: List[str],
        reference_condition: str = None,
    ) -> str:
        """
        Prepare design matrix and validate experimental design for differential expression analysis.

        CRITICAL: Validates minimum replicate requirements (3+ per condition).

        Args:
            modality_name: Name of pseudobulk modality
            formula: R-style formula (e.g., "~condition + batch")
            contrast: Contrast specification [factor, level1, level2]
            reference_condition: Reference level for main condition
        """
        try:
            # Validate modality exists
            if modality_name not in data_manager.list_modalities():
                raise ModalityNotFoundError(
                    f"Modality '{modality_name}' not found. "
                    f"Available: {data_manager.list_modalities()}"
                )

            # Get the pseudobulk modality
            adata = data_manager.get_modality(modality_name)
            logger.info(
                f"Preparing DE design for '{modality_name}': "
                f"{adata.n_obs} samples x {adata.n_vars} genes"
            )

            # SCIENTIFIC FIX: Validate with min_replicates=3
            design_validation = bulk_rnaseq_service.validate_experimental_design(
                metadata=adata.obs, formula=formula, min_replicates=3
            )

            if not design_validation["valid"]:
                error_msg = "; ".join(design_validation["errors"])
                raise DesignMatrixError(f"Invalid experimental design: {error_msg}")

            # Additional replicate validation
            condition_col = contrast[0]
            replicate_validation = _validate_replicate_counts(
                adata.obs, condition_col, min_replicates=3
            )

            if not replicate_validation["valid"]:
                error_msg = "; ".join(replicate_validation["errors"])
                raise InsufficientReplicatesError(error_msg)

            # Create design matrix
            design_result = bulk_rnaseq_service.create_formula_design(
                metadata=adata.obs,
                condition_col=contrast[0],
                reference_condition=reference_condition,
            )

            # Store design information in modality
            adata.uns["formula_design"] = {
                "formula": formula,
                "contrast": contrast,
                "design_matrix_info": design_result,
                "validation_results": design_validation,
            }

            # Update modality with design info
            data_manager.store_modality(
                name=modality_name,
                adata=adata,
                step_summary=f"Prepared DE design: formula={formula}",
            )

            # Log the operation
            data_manager.log_tool_usage(
                tool_name="prepare_de_design",
                parameters={
                    "modality_name": modality_name,
                    "formula": formula,
                    "contrast": contrast,
                },
                description=f"Prepared DE design for {adata.n_obs} pseudobulk samples",
            )

            # Format response with warnings
            response = f"""Differential expression design prepared for '{modality_name}'!

**Experimental Design:**
- Formula: {formula}
- Contrast: {contrast[1]} vs {contrast[2]} in {contrast[0]}
- Design matrix: {design_result["design_matrix"].shape[0]} samples x {design_result["design_matrix"].shape[1]} coefficients
- Matrix rank: {design_result["rank"]} (full rank: {"Yes" if design_result["rank"] == design_result["n_coefficients"] else "No"})

**Design Validation:**
- Valid: {"Yes" if design_validation["valid"] else "No"}
- Warnings: {len(design_validation["warnings"])} ({", ".join(design_validation["warnings"][:2]) if design_validation["warnings"] else "None"})

**Replicate Counts:**"""

            for group, count in replicate_validation["group_counts"].items():
                status = (
                    "OK"
                    if count >= 4
                    else "LOW POWER"
                    if count >= 3
                    else "INSUFFICIENT"
                )
                response += f"\n- {group}: {count} replicates ({status})"

            if replicate_validation["warnings"]:
                response += "\n\n**Warnings:**"
                for warning in replicate_validation["warnings"]:
                    response += f"\n- {warning}"

            response += (
                "\n\n**Design information stored in**: adata.uns['formula_design']"
            )
            response += "\n\nNext step: Run 'run_differential_expression' to perform pyDESeq2 analysis."

            analysis_results["details"]["de_design"] = response
            return response

        except (
            DesignMatrixError,
            FormulaError,
            ModalityNotFoundError,
            InsufficientReplicatesError,
        ) as e:
            logger.error(f"Error preparing DE design: {e}")
            return f"Error preparing differential expression design: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error in DE design preparation: {e}")
            return f"Unexpected error: {str(e)}"

    @tool
    def run_differential_expression(
        modality_name: str,
        groupby: str,
        group1: str,
        group2: str,
        method: str = "deseq2",
        alpha: float = 0.05,
        shrink_lfc: bool = True,
        save_result: bool = True,
    ) -> str:
        """
        Run differential expression analysis (simple 2-group comparison).

        Works for both pseudobulk (from single-cell) and direct bulk RNA-seq data.
        Auto-detects whether the data is pseudobulk (via adata.uns['pseudobulk_design']
        or adata.uns['formula_design']) and routes to the appropriate analysis method.

        CRITICAL: Uses raw counts from adata.raw.X for DESeq2 accuracy.
        Validates minimum replicate requirements (3+ per group).

        For complex multi-factor designs with covariates, use run_de_with_formula instead.

        Args:
            modality_name: Name of the bulk RNA-seq or pseudobulk modality
            groupby: Column name for grouping (e.g., 'condition', 'treatment')
            group1: First group for comparison (e.g., 'control')
            group2: Second group for comparison (e.g., 'treatment')
            method: Analysis method ('deseq2', 'wilcoxon', 't_test')
            alpha: Significance threshold for adjusted p-values
            shrink_lfc: Whether to apply log fold change shrinkage (DESeq2 only)
            save_result: Whether to save the results
        """
        try:
            # Ensure group IDs are strings (cluster IDs from Leiden are often integers)
            group1 = str(group1)
            group2 = str(group2)

            # Validate modality exists
            if modality_name not in data_manager.list_modalities():
                raise ModalityNotFoundError(
                    f"Modality '{modality_name}' not found. "
                    f"Available: {data_manager.list_modalities()}"
                )

            # Get the modality
            adata = data_manager.get_modality(modality_name)
            logger.info(
                f"Running DE analysis on '{modality_name}': "
                f"{adata.n_obs} samples x {adata.n_vars} genes"
            )

            # Validate groupby column exists
            if groupby not in adata.obs.columns:
                available_columns = [
                    col
                    for col in adata.obs.columns
                    if col.lower() in ["condition", "treatment", "group", "batch"]
                ]
                return f"Grouping column '{groupby}' not found. Available experimental design columns: {available_columns}"

            # Check if groups exist - match type of column values
            available_groups = list(adata.obs[groupby].unique())
            available_groups_str = [str(g) for g in available_groups]

            if group1 not in available_groups and group1 in available_groups_str:
                adata.obs[groupby] = adata.obs[groupby].astype(str)
                available_groups = list(adata.obs[groupby].unique())
            elif group1 not in available_groups:
                return f"Group '{group1}' not found in column '{groupby}'. Available groups: {available_groups}"
            if group2 not in available_groups:
                return f"Group '{group2}' not found in column '{groupby}'. Available groups: {available_groups}"

            # Validate replicate counts
            replicate_validation = _validate_replicate_counts(
                adata.obs, groupby, min_replicates=3
            )

            if not replicate_validation["valid"]:
                error_msg = "; ".join(replicate_validation["errors"])
                return f"**Insufficient replicates**: {error_msg}\n\nMinimum 3 replicates per group required for stable variance estimation."

            # Get group counts for response
            group1_count = (adata.obs[groupby] == group1).sum()
            group2_count = (adata.obs[groupby] == group2).sum()

            # Auto-detect pseudobulk vs direct bulk
            is_pseudobulk = (
                "pseudobulk_design" in adata.uns
                or "is_pseudobulk" in adata.uns
                or "formula_design" in adata.uns
            )

            if is_pseudobulk and method == "deseq2":
                # Pseudobulk path: use pyDESeq2 via formula design
                formula = f"~{groupby}"
                contrast = [groupby, group2, group1]

                # CRITICAL: Extract raw counts for DESeq2
                count_matrix, used_raw = _extract_raw_counts(adata)

                raw_warning = ""
                if not used_raw:
                    raw_warning = (
                        "\n\n**WARNING**: Using adata.X instead of adata.raw.X. "
                        "DESeq2 requires raw counts for accurate results."
                    )

                results_df, analysis_stats = (
                    bulk_rnaseq_service.run_pydeseq2_from_pseudobulk(
                        pseudobulk_adata=adata,
                        formula=formula,
                        contrast=contrast,
                        alpha=alpha,
                        shrink_lfc=shrink_lfc,
                        n_cpus=1,
                    )
                )

                # Store results
                contrast_name = f"{groupby}_{group2}_vs_{group1}"
                adata.uns[f"de_results_{contrast_name}"] = {
                    "results_df": results_df,
                    "analysis_stats": analysis_stats,
                    "parameters": {
                        "alpha": alpha,
                        "shrink_lfc": shrink_lfc,
                        "formula": formula,
                        "contrast": contrast,
                        "used_raw_counts": used_raw,
                        "method": "pydeseq2",
                    },
                }

                data_manager.store_modality(
                    name=modality_name,
                    adata=adata,
                    step_summary=f"DE analysis complete: {contrast_name}",
                )

                if save_result:
                    results_path = f"{modality_name}_de_results.csv"
                    results_df.to_csv(results_path)
                    save_path = f"{modality_name}_with_de_results.h5ad"
                    data_manager.save_modality(modality_name, save_path)

                # Create IR for provenance
                from lobster.core.provenance import AnalysisStep

                ir = AnalysisStep(
                    operation="pydeseq2.differential_expression",
                    tool_name="run_differential_expression",
                    description=f"pyDESeq2 DE: {group2} vs {group1} in {groupby}",
                    library="pydeseq2",
                    parameters={
                        "modality_name": modality_name,
                        "groupby": groupby,
                        "group1": group1,
                        "group2": group2,
                        "alpha": alpha,
                        "shrink_lfc": shrink_lfc,
                    },
                    code_template=(
                        "from pydeseq2.dds import DeseqDataSet\n"
                        "from pydeseq2.ds import DeseqStats\n"
                        "dds = DeseqDataSet(adata=adata, design_factors='{{ groupby }}')\n"
                        "dds.deseq2()\n"
                        "stat_res = DeseqStats(dds, contrast=['{{ groupby }}', '{{ group2 }}', '{{ group1 }}'])\n"
                        "stat_res.summary()\n"
                        "results_df = stat_res.results_df"
                    ),
                    imports=["pydeseq2"],
                )

                data_manager.log_tool_usage(
                    tool_name="run_differential_expression",
                    parameters={
                        "modality_name": modality_name,
                        "groupby": groupby,
                        "group1": group1,
                        "group2": group2,
                        "method": "pydeseq2",
                        "alpha": alpha,
                        "shrink_lfc": shrink_lfc,
                    },
                    description=f"pyDESeq2 analysis: {analysis_stats['n_significant_genes']} significant genes found",
                    ir=ir,
                )

                response = f"""## Differential Expression Analysis Complete for '{modality_name}'

**pyDESeq2 Analysis Results (pseudobulk detected):**
- Contrast: {group2} vs {group1} in {groupby}
- Genes tested: {analysis_stats["n_genes_tested"]:,}
- Significant genes: {analysis_stats["n_significant_genes"]:,} (alpha={alpha})
- Upregulated: {analysis_stats["n_upregulated"]:,}
- Downregulated: {analysis_stats["n_downregulated"]:,}

**Top Differentially Expressed Genes:**
**Upregulated ({group2} > {group1}):**
{chr(10).join([f"- {gene}" for gene in analysis_stats["top_upregulated"][:5]])}

**Downregulated ({group2} < {group1}):**
{chr(10).join([f"- {gene}" for gene in analysis_stats["top_downregulated"][:5]])}

**Analysis Parameters:**
- Method: pyDESeq2 (pseudobulk)
- LFC shrinkage: {"Yes" if shrink_lfc else "No"}
- Significance threshold: {alpha}
- Used raw counts: {"Yes" if used_raw else "No (WARNING)"}{raw_warning}

**Results stored in**: adata.uns['de_results_{contrast_name}']"""

                if save_result:
                    response += f"\n**Saved to**: {results_path} & {save_path}"

            else:
                # Direct bulk / simple 2-group DE path
                method_map = {"deseq2": "deseq2_like", "wilcoxon": "wilcoxon", "t_test": "t_test"}
                internal_method = method_map.get(method, method)

                adata_de, de_stats, ir = (
                    bulk_rnaseq_service.run_differential_expression_analysis(
                        adata=adata,
                        groupby=groupby,
                        group1=group1,
                        group2=group2,
                        method=internal_method,
                    )
                )

                # Save as new modality
                de_modality_name = f"{modality_name}_de_{group1}_vs_{group2}"
                data_manager.store_modality(
                    name=de_modality_name,
                    adata=adata_de,
                    parent_name=modality_name,
                    step_summary=f"DE analysis: {group1} vs {group2}",
                )

                if save_result:
                    save_path = f"{modality_name}_de_{group1}_vs_{group2}.h5ad"
                    data_manager.save_modality(de_modality_name, save_path)

                data_manager.log_tool_usage(
                    tool_name="run_differential_expression",
                    parameters={
                        "modality_name": modality_name,
                        "groupby": groupby,
                        "group1": group1,
                        "group2": group2,
                        "method": internal_method,
                    },
                    description=f"DE analysis: {de_stats['n_significant_genes']} significant genes found",
                    ir=ir,
                )

                analysis_stats = de_stats

                response = f"""## Differential Expression Analysis Complete for '{modality_name}'

**Analysis Results:**
- Comparison: {group1} ({group1_count} samples) vs {group2} ({group2_count} samples)
- Method: {de_stats['method']}
- Genes tested: {de_stats['n_genes_tested']:,}
- Significant genes (padj < {alpha}): {de_stats['n_significant_genes']:,}

**Differential Expression Summary:**
- Upregulated in {group2}: {de_stats['n_upregulated']} genes
- Downregulated in {group2}: {de_stats['n_downregulated']} genes

**Top Upregulated Genes:**
{chr(10).join([f"- {gene}" for gene in de_stats["top_upregulated"][:5]])}

**Top Downregulated Genes:**
{chr(10).join([f"- {gene}" for gene in de_stats["top_downregulated"][:5]])}

**New modality created**: '{de_modality_name}'"""

                if save_result:
                    response += f"\n**Saved to**: {save_path}"

                response += f"\n**Access detailed results**: adata.uns['{de_stats['de_results_key']}']\n"

            # Add replicate warnings if any
            if replicate_validation["warnings"]:
                response += "\n\n**Statistical Power Warnings:**\n"
                for warning in replicate_validation["warnings"]:
                    response += f"- {warning}\n"

            response += "\n\nNext steps: Visualize with volcano plots, use filter_de_results to filter, or run_pathway_enrichment for pathway analysis."

            analysis_results["details"]["differential_expression"] = response
            return response

        except (BulkRNASeqError, PseudobulkError, ModalityNotFoundError) as e:
            logger.error(f"Error in differential expression analysis: {e}")
            return f"Error in differential expression analysis: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error in differential expression: {e}")
            return f"Unexpected error: {str(e)}"

    @tool
    def validate_experimental_design(
        modality_name: str,
        formula: str,
    ) -> str:
        """
        Validate experimental design for statistical power and balance.

        CRITICAL: Requires minimum 3 replicates per condition (changed from 2).
        Warns when any condition has fewer than 4 replicates.

        Args:
            modality_name: Name of bulk RNA-seq or pseudobulk modality
            formula: R-style formula to validate
        """
        try:
            # Validate modality exists
            if modality_name not in data_manager.list_modalities():
                raise ModalityNotFoundError(
                    f"Modality '{modality_name}' not found. "
                    f"Available: {data_manager.list_modalities()}"
                )

            # Get modality
            adata = data_manager.get_modality(modality_name)
            metadata = adata.obs

            # SCIENTIFIC FIX: Validate design with min_replicates=3
            validation_result = formula_service.validate_experimental_design(
                metadata, formula, min_replicates=3
            )

            # Format response
            response = f"## Experimental Design Validation: `{formula}`\n\n"
            response += f"**Modality**: `{modality_name}`\n"
            response += f"**Samples**: {len(metadata)}\n\n"

            # Overall status
            if validation_result["valid"]:
                response += "**Overall Status**: PASSED - Design is valid for DESeq2 analysis\n\n"
            else:
                response += (
                    "**Overall Status**: FAILED - Design has issues (see below)\n\n"
                )

            # Design summary
            if validation_result.get("design_summary"):
                response += "### Group Balance\n"
                for factor, counts in validation_result["design_summary"].items():
                    response += f"**{factor}**:\n"
                    for level, count in counts.items():
                        # SCIENTIFIC FIX: Indicate status based on replicate count
                        if count < 3:
                            status = "INSUFFICIENT"
                        elif count < 4:
                            status = "LOW POWER"
                        else:
                            status = "OK"
                        response += f"  - {level}: {count} samples ({status})\n"
                response += "\n"

            # Warnings
            if validation_result.get("warnings"):
                response += "### Warnings\n"
                for warning in validation_result["warnings"]:
                    response += f"- {warning}\n"
                response += "\n"

            # Errors
            if validation_result.get("errors"):
                response += "### Errors\n"
                for error in validation_result["errors"]:
                    response += f"- {error}\n"
                response += "\n"

            # Replicate requirements note
            response += "### Replicate Requirements\n"
            response += "- **Minimum required**: 3 replicates per condition (for stable variance estimation)\n"
            response += "- **Recommended**: 4+ replicates per condition (for adequate statistical power)\n"
            response += "- **Ideal**: 6+ replicates per condition (for publication-quality results)\n\n"

            if validation_result["valid"]:
                response += "**Conclusion**: Design is ready for pyDESeq2 analysis\n"
            else:
                response += (
                    "**Conclusion**: Please address issues before running analysis\n"
                )

            return response

        except ModalityNotFoundError as e:
            logger.error(f"Modality not found error: {e}")
            return f"Error: {str(e)}"
        except Exception as e:
            logger.error(f"Error validating experimental design: {e}")
            return f"Error: {str(e)}"

    @tool
    def suggest_de_formula(
        modality_name: str,
        groupby: str = None,
        covariates: Optional[List[str]] = None,
        include_interaction: bool = False,
    ) -> str:
        """
        Analyze metadata, suggest DE formula, construct it, and validate the design.

        Combines metadata analysis, formula construction, and validation in one tool.
        Examines the data to suggest appropriate formulas, then builds and validates
        the chosen formula. Returns ready-to-use formula for run_de_with_formula.

        Args:
            modality_name: Name of pseudobulk or bulk RNA-seq modality
            groupby: Main variable of interest (auto-detected if None)
            covariates: Optional list of covariate variables to include
            include_interaction: Whether to include interaction terms
        """
        try:
            # Validate modality exists
            if modality_name not in data_manager.list_modalities():
                return f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}"

            # Get the data
            adata = data_manager.get_modality(modality_name)
            metadata = adata.obs
            n_samples = len(metadata)

            # Step 1: Analyze metadata columns
            variable_analysis = {}
            for col in metadata.columns:
                if col.startswith("_") or col in ["n_cells", "total_counts"]:
                    continue

                series = metadata[col]
                if pd.api.types.is_numeric_dtype(series):
                    var_type = "continuous"
                else:
                    var_type = "categorical"

                unique_vals = len(series.unique())
                missing = series.isna().sum()

                variable_analysis[col] = {
                    "type": var_type,
                    "unique_values": unique_vals,
                    "missing_count": missing,
                    "sample_values": list(series.unique())[:5],
                }

            categorical_vars = [
                col
                for col, info in variable_analysis.items()
                if info["type"] == "categorical"
                and info["unique_values"] > 1
                and info["unique_values"] < n_samples / 2
            ]
            continuous_vars = [
                col
                for col, info in variable_analysis.items()
                if info["type"] == "continuous"
                and info["missing_count"] < n_samples / 2
            ]

            # Auto-detect groupby if not provided
            if groupby is None:
                for col in categorical_vars:
                    if variable_analysis[col]["unique_values"] == 2:
                        groupby = col
                        break
                if groupby is None and categorical_vars:
                    groupby = categorical_vars[0]

            if groupby is None:
                return (
                    "**No suitable grouping variable found.**\n"
                    "Please ensure your data has at least one categorical variable with 2+ levels.\n"
                    f"Available variables: {list(variable_analysis.keys())}"
                )

            # Validate groupby exists
            if groupby not in metadata.columns:
                available_vars = [col for col in metadata.columns if not col.startswith("_")]
                return f"Variable '{groupby}' not found. Available: {available_vars}"

            # Validate covariates if provided
            if covariates:
                missing_covariates = [c for c in covariates if c not in metadata.columns]
                if missing_covariates:
                    return f"Covariates not found: {missing_covariates}"

            # Step 2: Construct formula
            formula_terms = [groupby]
            if covariates:
                formula_terms.extend(covariates)

            if include_interaction and covariates:
                interaction_term = f"{groupby}*{covariates[0]}"
                formula = f"~{interaction_term}"
                if len(covariates) > 1:
                    formula += f" + {' + '.join(covariates[1:])}"
            else:
                formula = f"~{' + '.join(formula_terms)}"

            # Step 3: Validate formula
            validation_warnings = []
            validation_errors = []

            # Check all terms exist
            for term in formula_terms:
                if term not in metadata.columns:
                    validation_errors.append(f"Variable '{term}' not found in metadata")

            # Check for confounding (single-level factors)
            for term in formula_terms:
                if term in variable_analysis and variable_analysis[term]["unique_values"] < 2:
                    validation_errors.append(f"Variable '{term}' has only 1 level - cannot be used in formula")

            # Check replicate counts
            replicate_validation = _validate_replicate_counts(
                metadata, groupby, min_replicates=3
            )
            if not replicate_validation["valid"]:
                for err in replicate_validation["errors"]:
                    validation_errors.append(err)
            if replicate_validation["warnings"]:
                for warn in replicate_validation["warnings"]:
                    validation_warnings.append(warn)

            # Validate with formula service if no errors
            formula_valid = len(validation_errors) == 0
            design_result = None
            if formula_valid:
                try:
                    formula_components = formula_service.parse_formula(formula, metadata)
                    design_result = formula_service.construct_design_matrix(
                        formula_components, metadata
                    )
                    svc_validation = formula_service.validate_experimental_design(
                        metadata, formula, min_replicates=3
                    )
                    if not svc_validation["valid"]:
                        for err in svc_validation.get("errors", []):
                            validation_errors.append(err)
                        formula_valid = False
                    for warn in svc_validation.get("warnings", []):
                        if warn not in validation_warnings:
                            validation_warnings.append(warn)
                except (FormulaError, DesignMatrixError) as e:
                    validation_errors.append(f"Formula validation error: {str(e)}")
                    formula_valid = False

            # Generate alternative suggestions
            suggestions = []
            batch_vars = [
                col for col in categorical_vars
                if col.lower() in ["batch", "sample", "donor", "patient", "subject"]
                or (variable_analysis[col]["unique_values"] > 2 and variable_analysis[col]["unique_values"] <= 6)
            ]

            if batch_vars and not covariates:
                primary_batch = batch_vars[0]
                suggestions.append({
                    "formula": f"~{groupby} + {primary_batch}",
                    "description": f"Add batch correction for {primary_batch}",
                    "min_samples": 8,
                })

            if len(batch_vars) > 1 or continuous_vars:
                extra_covs = batch_vars[:2] + continuous_vars[:1]
                all_terms = [groupby] + extra_covs
                suggestions.append({
                    "formula": f"~{' + '.join(all_terms)}",
                    "description": f"Multi-factor model with {len(extra_covs)} covariates",
                    "min_samples": max(12, len(all_terms) * 4),
                })

            # Store formula in modality
            adata.uns["constructed_formula"] = {
                "formula": formula,
                "main_variable": groupby,
                "covariates": covariates,
                "include_interactions": include_interaction,
                "validated": formula_valid,
            }
            data_manager.store_modality(
                name=modality_name,
                adata=adata,
                step_summary=f"Formula suggested and validated: {formula}",
            )

            # Create IR
            from lobster.core.provenance import AnalysisStep

            ir = AnalysisStep(
                operation="de_formula.suggest_and_validate",
                tool_name="suggest_de_formula",
                description=f"Analyzed metadata and constructed formula: {formula}",
                library="lobster",
                parameters={
                    "modality_name": modality_name,
                    "groupby": groupby,
                    "covariates": covariates,
                    "include_interaction": include_interaction,
                    "formula": formula,
                },
                code_template=(
                    "# Formula: {{ formula }}\n"
                    "# Main variable: {{ groupby }}\n"
                    "# Covariates: {{ covariates }}"
                ),
                imports=[],
            )

            data_manager.log_tool_usage(
                tool_name="suggest_de_formula",
                parameters={
                    "modality_name": modality_name,
                    "groupby": groupby,
                    "covariates": covariates,
                    "formula": formula,
                },
                description=f"Suggested and validated formula: {formula}",
                ir=ir,
            )

            # Build response
            response = f"## DE Formula Analysis for '{modality_name}'\n\n"

            response += "**Metadata Summary:**\n"
            response += f"- Samples: {n_samples}\n"
            response += f"- Categorical variables: {len(categorical_vars)}\n"
            response += f"- Continuous variables: {len(continuous_vars)}\n\n"

            response += "**Key Variables:**\n"
            for col, info in list(variable_analysis.items())[:6]:
                if col in categorical_vars + continuous_vars:
                    response += f"- **{col}**: {info['type']}, {info['unique_values']} levels"
                    if info["type"] == "categorical":
                        response += f" ({', '.join(map(str, info['sample_values']))})"
                    response += "\n"
            response += "\n"

            response += f"**Constructed Formula**: `{formula}`\n"
            response += f"- Main variable: {groupby}\n"
            if covariates:
                response += f"- Covariates: {', '.join(covariates)}\n"
            if include_interaction:
                response += f"- Interaction: {groupby} x {covariates[0] if covariates else 'N/A'}\n"
            response += "\n"

            if design_result:
                response += "**Design Matrix:**\n"
                response += f"- Dimensions: {design_result['design_matrix'].shape[0]} samples x {design_result['design_matrix'].shape[1]} coefficients\n"
                response += f"- Matrix rank: {design_result['rank']} (full rank: {'Yes' if design_result['rank'] == design_result['n_coefficients'] else 'WARNING'})\n\n"

            response += "**Validation:**\n"
            if formula_valid:
                response += "- Status: PASSED\n"
            else:
                response += "- Status: FAILED\n"

            if validation_errors:
                for err in validation_errors:
                    response += f"- ERROR: {err}\n"
            if validation_warnings:
                for warn in validation_warnings:
                    response += f"- WARNING: {warn}\n"

            response += "\n**Replicate Counts:**\n"
            for group, count in replicate_validation["group_counts"].items():
                status = "OK" if count >= 4 else "LOW POWER" if count >= 3 else "INSUFFICIENT"
                response += f"- {group}: {count} replicates ({status})\n"

            if suggestions:
                response += "\n**Alternative Formulas:**\n"
                for i, s in enumerate(suggestions, 1):
                    response += f"  {i}. `{s['formula']}` - {s['description']} (min {s['min_samples']} samples)\n"

            if formula_valid:
                response += f"\n**Next step**: Use `run_de_with_formula` with formula='{formula}' to execute the analysis."
            else:
                response += "\n**Action needed**: Address validation errors before running DE analysis."

            return response

        except Exception as e:
            logger.error(f"Error in suggest_de_formula: {e}")
            return f"Error analyzing design for formula suggestions: {str(e)}"

    @tool
    def construct_de_formula_interactive(
        pseudobulk_modality: str,
        main_variable: str,
        covariates: Optional[List[str]] = None,
        include_interactions: bool = False,
        validate_design: bool = True,
    ) -> str:
        """
        DEPRECATED: Use suggest_de_formula instead.

        This tool is deprecated and will be removed in a future version.

        Args:
            pseudobulk_modality: Name of pseudobulk modality
            main_variable: Primary variable of interest
            covariates: List of covariate variables
            include_interactions: Whether to include interaction terms
            validate_design: Whether to validate the design
        """
        logger.warning("construct_de_formula_interactive is deprecated. Use suggest_de_formula instead.")
        return "DEPRECATED: Use suggest_de_formula for formula construction and validation."

    @tool
    def run_de_with_formula(
        pseudobulk_modality: str,
        formula: Optional[str] = None,
        contrast: Optional[List[str]] = None,
        reference_levels: Optional[dict] = None,
        alpha: float = 0.05,
        lfc_threshold: float = 0.0,
        save_results: bool = True,
    ) -> str:
        """
        Run DE with a custom formula for complex multi-factor designs (covariates, interactions, batch correction).

        Uses pyDESeq2 for analysis with RAW COUNTS from adata.raw.X.
        For simple 2-group comparisons, use run_differential_expression instead.

        Args:
            pseudobulk_modality: Name of pseudobulk modality
            formula: R-style formula (uses stored formula if None)
            contrast: Contrast specification [factor, level1, level2]
            reference_levels: Reference levels for categorical variables
            alpha: Significance threshold for adjusted p-values
            lfc_threshold: Log fold change threshold
            save_results: Whether to save results to files
        """
        try:
            # Validate modality exists
            if pseudobulk_modality not in data_manager.list_modalities():
                return f"Modality '{pseudobulk_modality}' not found. Available: {data_manager.list_modalities()}"

            # Get the pseudobulk data
            adata = data_manager.get_modality(pseudobulk_modality)

            # Use stored formula if none provided
            if formula is None:
                if "constructed_formula" in adata.uns:
                    formula = adata.uns["constructed_formula"]["formula"]
                    stored_info = adata.uns["constructed_formula"]
                    response_prefix = (
                        "Using stored formula from interactive construction:\n"
                    )
                else:
                    return "No formula provided and no stored formula found. Use `suggest_de_formula` first or provide a formula."
            else:
                response_prefix = "Using provided formula:\n"
                stored_info = None

            # Auto-detect contrast if not provided
            if contrast is None and stored_info:
                main_var = stored_info["main_variable"]
                levels = list(adata.obs[main_var].unique())
                if len(levels) == 2:
                    contrast = [main_var, str(levels[1]), str(levels[0])]
                    response_prefix += (
                        f"Auto-detected contrast: {contrast[1]} vs {contrast[2]}\n"
                    )
                else:
                    return f"Multiple levels found for {main_var}: {levels}. Please specify contrast as [factor, level1, level2]."
            elif contrast is None:
                return "No contrast specified. Please provide contrast as [factor, level1, level2]."

            logger.info(
                f"Running DE analysis on '{pseudobulk_modality}' with formula: {formula}"
            )

            # SCIENTIFIC FIX: Validate design with min_replicates=3
            design_validation = bulk_rnaseq_service.validate_experimental_design(
                metadata=adata.obs, formula=formula, min_replicates=3
            )

            if not design_validation["valid"]:
                error_msgs = "; ".join(design_validation["errors"])
                return f"**Invalid experimental design**: {error_msgs}\n\nUse `suggest_de_formula` to debug the design."

            # Create design matrix
            condition_col = contrast[0]
            reference_condition = (
                reference_levels.get(condition_col) if reference_levels else None
            )

            design_result = bulk_rnaseq_service.create_formula_design(
                metadata=adata.obs,
                condition_col=condition_col,
                reference_condition=reference_condition,
            )

            # Store design information
            adata.uns["de_formula_design"] = {
                "formula": formula,
                "contrast": contrast,
                "design_matrix_info": design_result,
                "validation_results": design_validation,
                "reference_levels": reference_levels,
            }

            # Run pyDESeq2 analysis
            results_df, analysis_stats = (
                bulk_rnaseq_service.run_pydeseq2_from_pseudobulk(
                    pseudobulk_adata=adata,
                    formula=formula,
                    contrast=contrast,
                    alpha=alpha,
                    shrink_lfc=True,
                    n_cpus=1,
                )
            )

            # Filter by LFC threshold if specified
            if lfc_threshold > 0:
                significant_mask = (results_df["padj"] < alpha) & (
                    abs(results_df["log2FoldChange"]) >= lfc_threshold
                )
                n_lfc_filtered = significant_mask.sum()
            else:
                n_lfc_filtered = analysis_stats["n_significant_genes"]

            # Store results in modality
            contrast_name = f"{contrast[0]}_{contrast[1]}_vs_{contrast[2]}"
            adata.uns[f"de_results_formula_{contrast_name}"] = {
                "results_df": results_df,
                "analysis_stats": analysis_stats,
                "parameters": {
                    "formula": formula,
                    "contrast": contrast,
                    "alpha": alpha,
                    "lfc_threshold": lfc_threshold,
                    "reference_levels": reference_levels,
                },
            }

            # Update modality
            data_manager.store_modality(
                name=pseudobulk_modality,
                adata=adata,
                step_summary=f"DE with formula complete: {contrast_name}",
            )

            # Save results if requested
            if save_results:
                results_path = f"{pseudobulk_modality}_formula_de_results.csv"
                results_df.to_csv(results_path)

                modality_path = f"{pseudobulk_modality}_with_formula_results.h5ad"
                data_manager.save_modality(pseudobulk_modality, modality_path)

            # Format response
            response = "## Differential Expression Analysis Complete\n\n"
            response += response_prefix
            response += f"**Formula**: `{formula}`\n"
            response += (
                f"**Contrast**: {contrast[1]} vs {contrast[2]} (in {contrast[0]})\n\n"
            )

            response += "**Results Summary**:\n"
            response += f"- Genes tested: {analysis_stats['n_genes_tested']:,}\n"
            response += f"- Significant genes (FDR < {alpha}): {analysis_stats['n_significant_genes']:,}\n"
            if lfc_threshold > 0:
                response += (
                    f"- Significant + |LFC| >= {lfc_threshold}: {n_lfc_filtered:,}\n"
                )
            response += f"- Upregulated ({contrast[1]} > {contrast[2]}): {analysis_stats['n_upregulated']:,}\n"
            response += f"- Downregulated ({contrast[1]} < {contrast[2]}): {analysis_stats['n_downregulated']:,}\n\n"

            response += "**Top Differentially Expressed Genes**:\n"
            response += "**Most Upregulated**:\n"
            for gene in analysis_stats["top_upregulated"][:5]:
                gene_data = results_df.loc[gene]
                response += f"- {gene}: LFC = {gene_data['log2FoldChange']:.2f}, FDR = {gene_data['padj']:.2e}\n"

            response += "\n**Most Downregulated**:\n"
            for gene in analysis_stats["top_downregulated"][:5]:
                gene_data = results_df.loc[gene]
                response += f"- {gene}: LFC = {gene_data['log2FoldChange']:.2f}, FDR = {gene_data['padj']:.2e}\n"

            response += "\n**Results Storage**:\n"
            response += (
                f"- Stored in: adata.uns['de_results_formula_{contrast_name}']\n"
            )
            if save_results:
                response += f"- CSV file: {results_path}\n"
                response += f"- H5AD file: {modality_path}\n"

            response += "\n**Next steps**: Use `iterate_de_analysis` to try different formulas or `compare_de_iterations` to compare results."

            # Log the operation
            data_manager.log_tool_usage(
                tool_name="run_de_with_formula",
                parameters={
                    "pseudobulk_modality": pseudobulk_modality,
                    "formula": formula,
                    "contrast": contrast,
                    "alpha": alpha,
                    "lfc_threshold": lfc_threshold,
                },
                description=f"Formula-based DE analysis: {analysis_stats['n_significant_genes']} significant genes",
            )

            return response

        except Exception as e:
            logger.error(f"Error in formula-based DE analysis: {e}")
            return f"Error running differential expression with formula: {str(e)}"

    # NOTE: run_differential_expression_analysis has been merged into run_differential_expression above.
    # The merged tool auto-detects pseudobulk vs direct bulk data.

    @tool
    def iterate_de_analysis(
        pseudobulk_modality: str,
        new_formula: Optional[str] = None,
        new_contrast: Optional[List[str]] = None,
        filter_criteria: Optional[dict] = None,
        compare_to_previous: bool = True,
        iteration_name: Optional[str] = None,
    ) -> str:
        """
        Support iterative analysis with formula/filter changes.

        Enables trying different formulas or filters, tracking iterations,
        and comparing results between analyses.

        Args:
            pseudobulk_modality: Name of pseudobulk modality
            new_formula: New formula to try (if None, modify existing)
            new_contrast: New contrast to test
            filter_criteria: Additional filtering criteria (e.g., {'min_lfc': 0.5})
            compare_to_previous: Whether to compare with previous iteration
            iteration_name: Custom name for this iteration
        """
        try:
            # Validate modality exists
            if pseudobulk_modality not in data_manager.list_modalities():
                return f"Modality '{pseudobulk_modality}' not found. Available: {data_manager.list_modalities()}"

            # Get the pseudobulk data
            adata = data_manager.get_modality(pseudobulk_modality)

            # Initialize iteration tracking if not exists
            if "de_iterations" not in adata.uns:
                adata.uns["de_iterations"] = {"iterations": [], "current_iteration": 0}

            iteration_tracker = adata.uns["de_iterations"]
            current_iter = iteration_tracker["current_iteration"] + 1

            # Determine iteration name
            if iteration_name is None:
                iteration_name = f"iteration_{current_iter}"

            # Get previous results for comparison
            previous_results = None
            previous_iteration = None
            if compare_to_previous and iteration_tracker["iterations"]:
                previous_iteration = iteration_tracker["iterations"][-1]
                prev_key = f"de_results_formula_{previous_iteration['contrast_name']}"
                if prev_key in adata.uns:
                    previous_results = adata.uns[prev_key]["results_df"]

            # Use existing formula/contrast if not provided
            if new_formula is None or new_contrast is None:
                if "de_formula_design" in adata.uns:
                    if new_formula is None:
                        new_formula = adata.uns["de_formula_design"]["formula"]
                    if new_contrast is None:
                        new_contrast = adata.uns["de_formula_design"]["contrast"]
                else:
                    return "No previous formula/contrast found and none provided. Run a DE analysis first."

            logger.info(
                f"Starting DE iteration '{iteration_name}' on '{pseudobulk_modality}'"
            )

            # Run the analysis with new parameters
            run_result = run_de_with_formula(
                pseudobulk_modality=pseudobulk_modality,
                formula=new_formula,
                contrast=new_contrast,
                alpha=0.05,
                lfc_threshold=(
                    filter_criteria.get("min_lfc", 0.0) if filter_criteria else 0.0
                ),
                save_results=False,
            )

            if "Error" in run_result:
                return f"Error in iteration '{iteration_name}': {run_result}"

            # Get current results
            contrast_name = f"{new_contrast[0]}_{new_contrast[1]}_vs_{new_contrast[2]}"
            current_key = f"de_results_formula_{contrast_name}"

            if current_key not in adata.uns:
                return "Results not found after analysis. Analysis may have failed."

            current_results = adata.uns[current_key]["results_df"]
            current_stats = adata.uns[current_key]["analysis_stats"]

            # Store iteration information
            iteration_info = {
                "name": iteration_name,
                "formula": new_formula,
                "contrast": new_contrast,
                "contrast_name": contrast_name,
                "n_significant": current_stats["n_significant_genes"],
                "timestamp": pd.Timestamp.now().isoformat(),
                "filter_criteria": filter_criteria or {},
            }

            # Compare with previous if requested
            comparison_results = None
            if compare_to_previous and previous_results is not None:
                current_sig = set(current_results[current_results["padj"] < 0.05].index)
                previous_sig = set(
                    previous_results[previous_results["padj"] < 0.05].index
                )

                overlap = len(current_sig & previous_sig)
                current_only = len(current_sig - previous_sig)
                previous_only = len(previous_sig - current_sig)

                common_genes = list(current_sig & previous_sig)
                if len(common_genes) > 3:
                    current_lfc = current_results.loc[common_genes, "log2FoldChange"]
                    previous_lfc = previous_results.loc[common_genes, "log2FoldChange"]
                    correlation = current_lfc.corr(previous_lfc)
                else:
                    correlation = None

                comparison_results = {
                    "overlap_genes": overlap,
                    "current_only": current_only,
                    "previous_only": previous_only,
                    "correlation": correlation,
                }

                iteration_info["comparison"] = comparison_results

            # Update iteration tracking
            iteration_tracker["iterations"].append(iteration_info)
            iteration_tracker["current_iteration"] = current_iter
            adata.uns["de_iterations"] = iteration_tracker

            # Update modality
            data_manager.store_modality(
                name=pseudobulk_modality,
                adata=adata,
                step_summary=f"DE iteration '{iteration_name}' complete",
            )

            # Format response
            response = f"## DE Analysis Iteration '{iteration_name}' Complete\n\n"
            response += f"**Formula**: `{new_formula}`\n"
            response += f"**Contrast**: {new_contrast[1]} vs {new_contrast[2]} (in {new_contrast[0]})\n\n"

            response += "**Current Results**:\n"
            response += (
                f"- Significant genes: {current_stats['n_significant_genes']:,}\n"
            )
            response += f"- Upregulated: {current_stats['n_upregulated']:,}\n"
            response += f"- Downregulated: {current_stats['n_downregulated']:,}\n"

            if comparison_results:
                response += "\n**Comparison with Previous Iteration**:\n"
                response += f"- Overlapping significant genes: {comparison_results['overlap_genes']:,}\n"
                response += (
                    f"- New in current: {comparison_results['current_only']:,}\n"
                )
                response += (
                    f"- Lost from previous: {comparison_results['previous_only']:,}\n"
                )
                if comparison_results["correlation"] is not None:
                    response += f"- Fold change correlation: {comparison_results['correlation']:.3f}\n"

            response += "\n**Iteration Summary**:\n"
            response += f"- Total iterations: {len(iteration_tracker['iterations'])}\n"
            response += f"- Current iteration: {current_iter}\n"

            response += f"\n**Results stored in**: adata.uns['de_results_formula_{contrast_name}']\n"
            response += "**Iteration tracking**: adata.uns['de_iterations']\n"

            response += "\n**Next steps**: Use `compare_de_iterations` to compare all iterations or continue iterating with different parameters."

            # Log the operation
            data_manager.log_tool_usage(
                tool_name="iterate_de_analysis",
                parameters={
                    "pseudobulk_modality": pseudobulk_modality,
                    "iteration_name": iteration_name,
                    "formula": new_formula,
                    "contrast": new_contrast,
                    "compare_to_previous": compare_to_previous,
                },
                description=f"DE iteration {current_iter}: {current_stats['n_significant_genes']} significant genes",
            )

            return response

        except Exception as e:
            logger.error(f"Error in DE iteration: {e}")
            return f"Error in DE analysis iteration: {str(e)}"

    @tool
    def compare_de_iterations(
        pseudobulk_modality: str,
        iteration_1: Optional[str] = None,
        iteration_2: Optional[str] = None,
        show_overlap: bool = True,
        show_unique: bool = True,
        save_comparison: bool = True,
    ) -> str:
        """
        Compare results between different DE analysis iterations.

        Shows overlapping and unique DEGs, correlation of fold changes, and helps
        users understand impact of formula changes.

        Args:
            pseudobulk_modality: Name of pseudobulk modality
            iteration_1: Name of first iteration (latest if None)
            iteration_2: Name of second iteration (second latest if None)
            show_overlap: Whether to show overlapping genes
            show_unique: Whether to show unique genes per iteration
            save_comparison: Whether to save comparison results
        """
        try:
            # Validate modality exists
            if pseudobulk_modality not in data_manager.list_modalities():
                return f"Modality '{pseudobulk_modality}' not found. Available: {data_manager.list_modalities()}"

            # Get the pseudobulk data
            adata = data_manager.get_modality(pseudobulk_modality)

            # Check iteration tracking exists
            if "de_iterations" not in adata.uns:
                return "No iteration tracking found. Run `iterate_de_analysis` first to create iterations."

            iteration_tracker = adata.uns["de_iterations"]
            iterations = iteration_tracker["iterations"]

            if len(iterations) < 2:
                return f"Only {len(iterations)} iteration(s) available. Need at least 2 for comparison."

            # Select iterations to compare
            if iteration_1 is None:
                iter1_info = iterations[-1]
            else:
                iter1_info = next(
                    (i for i in iterations if i["name"] == iteration_1), None
                )
                if not iter1_info:
                    available = [i["name"] for i in iterations]
                    return (
                        f"Iteration '{iteration_1}' not found. Available: {available}"
                    )

            if iteration_2 is None:
                iter2_info = iterations[-2] if len(iterations) >= 2 else iterations[0]
            else:
                iter2_info = next(
                    (i for i in iterations if i["name"] == iteration_2), None
                )
                if not iter2_info:
                    available = [i["name"] for i in iterations]
                    return (
                        f"Iteration '{iteration_2}' not found. Available: {available}"
                    )

            # Get results DataFrames
            iter1_key = f"de_results_formula_{iter1_info['contrast_name']}"
            iter2_key = f"de_results_formula_{iter2_info['contrast_name']}"

            if iter1_key not in adata.uns or iter2_key not in adata.uns:
                return "Results not found for one or both iterations."

            results1 = adata.uns[iter1_key]["results_df"]
            results2 = adata.uns[iter2_key]["results_df"]

            # Get significant genes (FDR < 0.05)
            sig1 = set(results1[results1["padj"] < 0.05].index)
            sig2 = set(results2[results2["padj"] < 0.05].index)

            # Calculate overlaps
            overlap = sig1 & sig2
            unique1 = sig1 - sig2
            unique2 = sig2 - sig1

            # Calculate fold change correlation for overlapping genes
            if len(overlap) > 3:
                overlap_genes = list(overlap)
                lfc1 = results1.loc[overlap_genes, "log2FoldChange"]
                lfc2 = results2.loc[overlap_genes, "log2FoldChange"]
                correlation = lfc1.corr(lfc2)
            else:
                correlation = None

            # Format response
            response = "## DE Iteration Comparison\n\n"
            response += "**Comparing:**\n"
            response += (
                f"- Iteration 1: '{iter1_info['name']}' - {iter1_info['formula']}\n"
            )
            response += (
                f"- Iteration 2: '{iter2_info['name']}' - {iter2_info['formula']}\n\n"
            )

            response += "**Results Summary:**\n"
            response += f"- Iteration 1 significant genes: {len(sig1):,}\n"
            response += f"- Iteration 2 significant genes: {len(sig2):,}\n"
            response += f"- Overlapping genes: {len(overlap):,} ({len(overlap) / max(len(sig1), len(sig2)) * 100:.1f}%)\n"
            response += f"- Unique to iteration 1: {len(unique1):,}\n"
            response += f"- Unique to iteration 2: {len(unique2):,}\n"

            if correlation is not None:
                response += f"- Fold change correlation: {correlation:.3f}\n"

            if show_overlap and len(overlap) > 0:
                response += "\n**Top Overlapping Genes:**\n"
                overlap_df = results1.loc[list(overlap)]
                overlap_df = overlap_df.reindex(overlap_df["padj"].sort_values().index)

                for gene in list(overlap_df.index)[:10]:
                    lfc1 = results1.loc[gene, "log2FoldChange"]
                    lfc2 = results2.loc[gene, "log2FoldChange"]
                    response += f"- {gene}: LFC1={lfc1:.2f}, LFC2={lfc2:.2f}\n"

            if show_unique and (len(unique1) > 0 or len(unique2) > 0):
                response += "\n**Unique Significant Genes:**\n"

                if len(unique1) > 0:
                    response += (
                        f"**Only in '{iter1_info['name']}'** ({len(unique1)} genes):\n"
                    )
                    unique1_sorted = results1.loc[list(unique1)].sort_values("padj")
                    for gene in unique1_sorted.index[:8]:
                        lfc = results1.loc[gene, "log2FoldChange"]
                        fdr = results1.loc[gene, "padj"]
                        response += f"- {gene}: LFC={lfc:.2f}, FDR={fdr:.2e}\n"

                if len(unique2) > 0:
                    response += f"\n**Only in '{iter2_info['name']}'** ({len(unique2)} genes):\n"
                    unique2_sorted = results2.loc[list(unique2)].sort_values("padj")
                    for gene in unique2_sorted.index[:8]:
                        lfc = results2.loc[gene, "log2FoldChange"]
                        fdr = results2.loc[gene, "padj"]
                        response += f"- {gene}: LFC={lfc:.2f}, FDR={fdr:.2e}\n"

            # Analysis interpretation
            response += "\n**Interpretation:**\n"
            if correlation is not None:
                if correlation > 0.8:
                    response += f"- High correlation ({correlation:.3f}) suggests similar biological effects\n"
                elif correlation > 0.5:
                    response += f"- Moderate correlation ({correlation:.3f}) - some consistency but notable differences\n"
                else:
                    response += f"- Low correlation ({correlation:.3f}) - formulas capture different effects\n"

            overlap_percent = len(overlap) / max(len(sig1), len(sig2)) * 100
            if overlap_percent > 70:
                response += f"- High overlap ({overlap_percent:.1f}%) - formulas yield similar gene sets\n"
            elif overlap_percent > 40:
                response += f"- Moderate overlap ({overlap_percent:.1f}%) - some formula-specific effects\n"
            else:
                response += f"- Low overlap ({overlap_percent:.1f}%) - formulas capture different biology\n"

            # Save comparison if requested
            if save_comparison:
                comparison_data = {
                    "iteration_1": iter1_info,
                    "iteration_2": iter2_info,
                    "overlap_genes": list(overlap),
                    "unique_to_1": list(unique1),
                    "unique_to_2": list(unique2),
                    "correlation": correlation,
                    "summary_stats": {
                        "n_sig_1": len(sig1),
                        "n_sig_2": len(sig2),
                        "n_overlap": len(overlap),
                        "overlap_percent": overlap_percent,
                    },
                }

                comparison_key = (
                    f"iteration_comparison_{iter1_info['name']}_vs_{iter2_info['name']}"
                )
                if "iteration_comparisons" not in adata.uns:
                    adata.uns["iteration_comparisons"] = {}
                adata.uns["iteration_comparisons"][comparison_key] = comparison_data

                data_manager.store_modality(
                    name=pseudobulk_modality,
                    adata=adata,
                    step_summary=f"Compared iterations: {iter1_info['name']} vs {iter2_info['name']}",
                )
                response += f"\n**Comparison saved**: adata.uns['iteration_comparisons']['{comparison_key}']\n"

            response += "\n**Next steps**: Choose the most appropriate formula based on biological interpretation and statistical robustness."

            # Log the operation
            data_manager.log_tool_usage(
                tool_name="compare_de_iterations",
                parameters={
                    "pseudobulk_modality": pseudobulk_modality,
                    "iteration_1": iter1_info["name"],
                    "iteration_2": iter2_info["name"],
                    "show_overlap": show_overlap,
                    "show_unique": show_unique,
                },
                description=f"Compared iterations: {len(overlap)} overlapping, {len(unique1)}+{len(unique2)} unique genes",
            )

            return response

        except Exception as e:
            logger.error(f"Error comparing DE iterations: {e}")
            return f"Error comparing DE iterations: {str(e)}"

    @tool
    def run_pathway_enrichment(
        gene_list: List[str],
        analysis_type: str = "GO",
        modality_name: str = None,
        save_result: bool = True,
    ) -> str:
        """
        Run pathway enrichment analysis on gene lists from differential expression results.

        Args:
            gene_list: List of genes for enrichment analysis
            analysis_type: Type of analysis ("GO" or "KEGG")
            modality_name: Optional modality name to extract genes from DE results
            save_result: Whether to save enrichment results
        """
        try:
            # If modality name provided, extract significant genes from it
            if modality_name and modality_name in data_manager.list_modalities():
                adata = data_manager.get_modality(modality_name)

                # Look for DE results in uns
                de_keys = [
                    key for key in adata.uns.keys() if key.startswith("de_results")
                ]
                if de_keys:
                    de_results = adata.uns[de_keys[0]]
                    if isinstance(de_results, dict) and "results_df" in de_results:
                        de_df = de_results["results_df"]
                        if "padj" in de_df.columns:
                            significant_genes = de_df[
                                de_df["padj"] < 0.05
                            ].index.tolist()
                            if significant_genes:
                                gene_list = significant_genes[:500]
                                logger.info(
                                    f"Extracted {len(gene_list)} significant genes from {modality_name}"
                                )

            if not gene_list or len(gene_list) == 0:
                return "No genes provided for enrichment analysis. Please provide a gene list or run differential expression analysis first."

            logger.info(f"Running pathway enrichment on {len(gene_list)} genes")

            # Use bulk service for pathway enrichment
            enrichment_df, enrichment_stats, ir = (
                bulk_rnaseq_service.run_pathway_enrichment(
                    gene_list=gene_list, analysis_type=analysis_type
                )
            )

            # Log the operation with IR for provenance tracking
            data_manager.log_tool_usage(
                tool_name="run_pathway_enrichment",
                parameters={
                    "gene_list_size": len(gene_list),
                    "analysis_type": analysis_type,
                    "modality_name": modality_name,
                },
                description=f"{analysis_type} enrichment: {enrichment_stats['n_significant_terms']} significant terms",
                ir=ir,
            )

            # Format response
            response = f"## {analysis_type} Pathway Enrichment Analysis Complete\n\n"
            response += "**Enrichment Results:**\n"
            response += f"- Genes analyzed: {enrichment_stats['n_genes_input']:,}\n"
            response += f"- Database: {enrichment_stats['enrichment_database']}\n"
            response += f"- Terms found: {enrichment_stats['n_terms_total']}\n"
            response += f"- Significant terms (p.adj < 0.05): {enrichment_stats['n_significant_terms']}\n\n"

            response += "**Top Enriched Pathways:**\n"
            for term in enrichment_stats["top_terms"][:8]:
                response += f"- {term}\n"

            if len(enrichment_stats["top_terms"]) > 8:
                remaining = len(enrichment_stats["top_terms"]) - 8
                response += f"... and {remaining} more pathways\n"

            response += "\nPathway enrichment reveals biological processes and pathways associated with differential expression."

            analysis_results["details"]["pathway_enrichment"] = response
            return response

        except BulkRNASeqError as e:
            logger.error(f"Error in pathway enrichment: {e}")
            return f"Error running pathway enrichment: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error in pathway enrichment: {e}")
            return f"Unexpected error: {str(e)}"

    # -------------------------
    # TOOL REGISTRY
    # -------------------------
    base_tools = [
        # Pseudobulk preparation
        create_pseudobulk_matrix,
        prepare_de_design,
        # Design validation
        validate_experimental_design,
        # Formula tools
        suggest_de_formula,
        # DE analysis (2 clear tools)
        run_differential_expression,
        run_de_with_formula,
        # Result tools
        filter_de_results,
        export_de_results,
        # Iteration tools
        iterate_de_analysis,
        compare_de_iterations,
        # Pathway analysis
        run_pathway_enrichment,
    ]

    tools = base_tools + (delegation_tools or [])

    # -------------------------
    # CREATE AGENT
    # -------------------------
    system_prompt = create_de_analysis_expert_prompt()

    return create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt,
        name=agent_name,
        state_schema=DEAnalysisExpertState,
    )
