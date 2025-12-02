"""
Bulk RNA-seq Expert Agent for specialized bulk RNA-seq analysis.

DEPRECATED: This agent is deprecated and will be removed in a future version.
Use transcriptomics_expert instead, which handles both single-cell and bulk RNA-seq.

This agent focuses exclusively on bulk RNA-seq analysis using the modular DataManagerV2
system with proper modality handling and schema enforcement.
"""

import warnings

# Emit deprecation warning on import
warnings.warn(
    "lobster.agents.bulk_rnaseq_expert is deprecated and will be removed in a future version. "
    "Use lobster.agents.transcriptomics.transcriptomics_expert instead.",
    DeprecationWarning,
    stacklevel=2,
)

from datetime import date
from typing import List, Optional

import pandas as pd
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from lobster.agents.state import BulkRNASeqExpertState
from lobster.config.llm_factory import create_llm
from lobster.config.settings import get_settings
from lobster.core.data_manager_v2 import DataManagerV2
from lobster.services.analysis.bulk_rnaseq_service import BulkRNASeqError, BulkRNASeqService
from lobster.services.quality.preprocessing_service import PreprocessingError, PreprocessingService
from lobster.services.quality.quality_service import QualityError, QualityService
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


class ModalityNotFoundError(BulkRNASeqError):
    """Raised when requested modality doesn't exist."""

    pass


def bulk_rnaseq_expert(
    data_manager: DataManagerV2,
    callback_handler=None,
    agent_name: str = "bulk_rnaseq_expert_agent",
    handoff_tools: List = None,
):
    """Create bulk RNA-seq expert agent using DataManagerV2 and modular services."""

    settings = get_settings()
    model_params = settings.get_agent_llm_params("bulk_rnaseq_expert_agent")
    llm = create_llm("bulk_rnaseq_expert_agent", model_params)

    if callback_handler and hasattr(llm, "with_config"):
        llm = llm.with_config(callbacks=[callback_handler])

    # Initialize stateless services for bulk RNA-seq analysis
    preprocessing_service = PreprocessingService()
    quality_service = QualityService()
    bulk_service = BulkRNASeqService()

    analysis_results = {"summary": "", "details": {}}

    # -------------------------
    # DATA STATUS TOOLS
    # -------------------------
    @tool
    def check_data_status(modality_name: str = "") -> str:
        """Check if bulk RNA-seq data is loaded and ready for analysis."""
        try:
            if modality_name == "":
                modalities = data_manager.list_modalities()
                if not modalities:
                    return "No modalities loaded. Please ask the data expert to load a bulk RNA-seq dataset first."

                # Filter for likely bulk RNA-seq modalities
                bulk_modalities = [
                    mod
                    for mod in modalities
                    if "bulk" in mod.lower()
                    or data_manager._detect_modality_type(mod) == "bulk_rna_seq"
                ]

                if not bulk_modalities:
                    response = f"Available modalities ({len(modalities)}) but none appear to be bulk RNA-seq:\n"
                    for mod_name in modalities:
                        adata = data_manager.get_modality(mod_name)
                        response += f"- **{mod_name}**: {adata.n_obs} samples × {adata.n_vars} genes\n"
                    response += "\nPlease specify a modality name if it contains bulk RNA-seq data."
                else:
                    response = (
                        f"Bulk RNA-seq modalities found ({len(bulk_modalities)}):\n"
                    )
                    for mod_name in bulk_modalities:
                        adata = data_manager.get_modality(mod_name)
                        response += f"- **{mod_name}**: {adata.n_obs} samples × {adata.n_vars} genes\n"

                return response

            else:
                # Check specific modality
                if modality_name not in data_manager.list_modalities():
                    return f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}"

                adata = data_manager.get_modality(modality_name)
                metrics = data_manager.get_quality_metrics(modality_name)

                response = (
                    f"Bulk RNA-seq modality '{modality_name}' ready for analysis:\n"
                )
                response += f"- Shape: {adata.n_obs} samples × {adata.n_vars} genes\n"
                response += f"- Sample metadata: {list(adata.obs.columns)[:5]}...\n"
                response += f"- Gene metadata: {list(adata.var.columns)[:5]}...\n"

                if "total_counts" in metrics:
                    response += f"- Total counts: {metrics['total_counts']:,.0f}\n"
                if "mean_counts_per_obs" in metrics:
                    response += (
                        f"- Mean counts/sample: {metrics['mean_counts_per_obs']:.1f}\n"
                    )

                # Add bulk RNA-seq specific checks
                if adata.n_obs < 6:
                    response += f"- Sample size: Small ({adata.n_obs} samples) - may limit statistical power\n"
                elif adata.n_obs < 20:
                    response += f"- Sample size: Moderate ({adata.n_obs} samples) - good for analysis\n"
                else:
                    response += f"- Sample size: Large ({adata.n_obs} samples) - excellent statistical power\n"

                # Check for experimental design columns
                design_cols = [
                    col
                    for col in adata.obs.columns
                    if col.lower()
                    in ["condition", "treatment", "group", "batch", "time_point"]
                ]
                if design_cols:
                    response += f"- Experimental design: {', '.join(design_cols)}\n"

                analysis_results["details"]["data_status"] = response
                return response

        except Exception as e:
            logger.error(f"Error checking bulk RNA-seq data status: {e}")
            return f"Error checking bulk RNA-seq data status: {str(e)}"

    @tool
    def assess_data_quality(
        modality_name: str,
        min_genes: int = 1000,
        max_mt_pct: float = 50.0,
        min_total_counts: float = 10000.0,
        check_batch_effects: bool = True,
    ) -> str:
        """Run comprehensive quality control assessment on bulk RNA-seq data."""
        try:
            if modality_name == "":
                return "Please specify modality_name for bulk RNA-seq quality assessment. Use check_data_status() to see available modalities."

            # Validate modality exists
            if modality_name not in data_manager.list_modalities():
                return f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}"

            # Get the modality
            adata = data_manager.get_modality(modality_name)

            # Run quality assessment using service with bulk RNA-seq specific parameters
            adata_qc, assessment_stats, ir = quality_service.assess_quality(
                adata=adata,
                min_genes=min_genes,
                max_mt_pct=max_mt_pct,
                max_ribo_pct=100.0,  # Less stringent for bulk
                min_housekeeping_score=0.5,  # Less stringent for bulk
            )

            # Create new modality with QC annotations
            qc_modality_name = f"{modality_name}_quality_assessed"
            data_manager.modalities[qc_modality_name] = adata_qc

            # Log the operation
            data_manager.log_tool_usage(
                tool_name="assess_data_quality",
                parameters={
                    "modality_name": modality_name,
                    "min_genes": min_genes,
                    "max_mt_pct": max_mt_pct,
                    "min_total_counts": min_total_counts,
                    "check_batch_effects": check_batch_effects,
                },
                description=f"Bulk RNA-seq quality assessment for {modality_name}",
                ir=ir,
            )

            # Format professional response with bulk RNA-seq context
            response = f"""Bulk RNA-seq Quality Assessment Complete for '{modality_name}'!

📊 **Assessment Results:**
- Samples analyzed: {assessment_stats['cells_before_qc']:,}
- Samples passing QC: {assessment_stats['cells_after_qc']:,} ({assessment_stats['cells_retained_pct']:.1f}%)
- Quality status: {assessment_stats['quality_status']}

📈 **Bulk RNA-seq Quality Metrics:**
- Mean genes per sample: {assessment_stats['mean_genes_per_cell']:.0f}
- Mean mitochondrial %: {assessment_stats['mean_mt_pct']:.1f}%
- Mean ribosomal %: {assessment_stats['mean_ribo_pct']:.1f}%
- Mean read counts: {assessment_stats['mean_total_counts']:.0f}

💡 **Bulk RNA-seq QC Summary:**
{assessment_stats['qc_summary']}

💾 **New modality created**: '{qc_modality_name}' (with bulk RNA-seq QC annotations)

Proceed with filtering and normalization for differential expression analysis."""

            analysis_results["details"]["quality_assessment"] = response
            return response

        except QualityError as e:
            logger.error(f"Bulk RNA-seq quality assessment error: {e}")
            return f"Bulk RNA-seq quality assessment failed: {str(e)}"
        except Exception as e:
            logger.error(f"Error in bulk RNA-seq quality assessment: {e}")
            return f"Error in bulk RNA-seq quality assessment: {str(e)}"

    # -------------------------
    # BULK RNA-SEQ PREPROCESSING TOOLS
    # -------------------------
    @tool
    def filter_and_normalize_modality(
        modality_name: str,
        min_genes_per_sample: int = 1000,
        min_samples_per_gene: int = 2,
        min_total_counts: float = 10000.0,
        normalization_method: str = "log1p",
        target_sum: int = 1000000,
        save_result: bool = True,
    ) -> str:
        """
        Filter and normalize bulk RNA-seq data using professional standards.

        Args:
            modality_name: Name of the modality to process
            min_genes_per_sample: Minimum number of genes expressed per sample
            min_samples_per_gene: Minimum number of samples expressing each gene
            min_total_counts: Minimum total read counts per sample
            normalization_method: Normalization method ('log1p', 'cpm', 'tpm')
            target_sum: Target sum for normalization (1M standard for bulk RNA-seq)
            save_result: Whether to save the filtered modality
        """
        try:
            # Validate modality exists
            if modality_name not in data_manager.list_modalities():
                raise ModalityNotFoundError(
                    f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}"
                )

            # Get the modality
            adata = data_manager.get_modality(modality_name)
            logger.info(
                f"Processing bulk RNA-seq modality '{modality_name}': {adata.shape[0]} samples × {adata.shape[1]} genes"
            )

            # Use preprocessing service with bulk RNA-seq optimized parameters
            adata_processed, processing_stats, ir = (
                preprocessing_service.filter_and_normalize_cells(
                    adata=adata,
                    min_genes_per_cell=min_genes_per_sample,
                    max_genes_per_cell=50000,  # No upper limit for bulk
                    min_cells_per_gene=min_samples_per_gene,
                    max_mito_percent=100.0,  # Less stringent for bulk
                    normalization_method=normalization_method,
                    target_sum=target_sum,
                )
            )

            # Save as new modality
            filtered_modality_name = f"{modality_name}_filtered_normalized"
            data_manager.modalities[filtered_modality_name] = adata_processed

            # Save to file if requested
            if save_result:
                save_path = f"{modality_name}_filtered_normalized.h5ad"
                data_manager.save_modality(filtered_modality_name, save_path)

            # Log the operation
            data_manager.log_tool_usage(
                tool_name="filter_and_normalize_modality",
                parameters={
                    "modality_name": modality_name,
                    "min_genes_per_sample": min_genes_per_sample,
                    "min_samples_per_gene": min_samples_per_gene,
                    "min_total_counts": min_total_counts,
                    "normalization_method": normalization_method,
                    "target_sum": target_sum,
                },
                description=f"Bulk RNA-seq filtered and normalized {modality_name}",
                ir=ir,
            )

            # Format professional response
            original_shape = processing_stats["original_shape"]
            final_shape = processing_stats["final_shape"]
            samples_retained_pct = processing_stats["cells_retained_pct"]
            genes_retained_pct = processing_stats["genes_retained_pct"]

            response = f"""Successfully filtered and normalized bulk RNA-seq modality '{modality_name}'!

📊 **Bulk RNA-seq Filtering Results:**
- Original: {original_shape[0]:,} samples × {original_shape[1]:,} genes
- Filtered: {final_shape[0]:,} samples × {final_shape[1]:,} genes  
- Samples retained: {samples_retained_pct:.1f}%
- Genes retained: {genes_retained_pct:.1f}%

🔬 **Bulk RNA-seq Processing Parameters:**
- Min genes/sample: {min_genes_per_sample} (removes low-quality samples)
- Min samples/gene: {min_samples_per_gene} (removes rarely expressed genes)
- Min total counts: {min_total_counts:,.0f} (minimum sequencing depth)
- Normalization: {normalization_method} (target_sum={target_sum:,} reads/sample)

💾 **New modality created**: '{filtered_modality_name}'"""

            if save_result:
                response += f"\n💾 **Saved to**: {save_path}"

            response += "\n\nNext recommended steps: differential expression analysis between experimental groups."

            analysis_results["details"]["filter_normalize"] = response
            return response

        except (PreprocessingError, ModalityNotFoundError) as e:
            logger.error(f"Error in bulk RNA-seq filtering/normalization: {e}")
            return f"Error filtering and normalizing bulk RNA-seq modality: {str(e)}"
        except Exception as e:
            logger.error(
                f"Unexpected error in bulk RNA-seq filtering/normalization: {e}"
            )
            return f"Unexpected error: {str(e)}"

    # -------------------------
    # BULK RNA-SEQ SPECIFIC ANALYSIS TOOLS
    # -------------------------
    @tool
    def run_differential_expression_analysis(
        modality_name: str,
        groupby: str,
        group1: str,
        group2: str,
        method: str = "deseq2_like",
        min_expression_threshold: float = 1.0,
        min_fold_change: float = 1.5,
        min_pct_expressed: float = 0.1,
        max_out_pct_expressed: float = 0.5,
        save_result: bool = True,
    ) -> str:
        """
        Run differential expression analysis between two groups in bulk RNA-seq data.

        Args:
            modality_name: Name of the bulk RNA-seq modality to analyze
            groupby: Column name for grouping (e.g., 'condition', 'treatment')
            group1: First group for comparison (e.g., 'control')
            group2: Second group for comparison (e.g., 'treatment')
            method: Analysis method ('deseq2_like', 'wilcoxon', 't_test')
            min_expression_threshold: Minimum expression threshold for gene filtering
            min_fold_change: Minimum fold-change threshold for biological significance (default: 1.5)
            min_pct_expressed: Minimum fraction expressing in group1 (default: 0.1)
            max_out_pct_expressed: Maximum fraction expressing in group2 (default: 0.5)
            save_result: Whether to save the results
        """
        try:
            # Validate modality exists
            if modality_name not in data_manager.list_modalities():
                raise ModalityNotFoundError(
                    f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}"
                )

            # Get the modality
            adata = data_manager.get_modality(modality_name)
            logger.info(
                f"Running DE analysis on bulk RNA-seq modality '{modality_name}': {adata.shape[0]} samples × {adata.shape[1]} genes"
            )

            # Validate experimental design
            if groupby not in adata.obs.columns:
                available_columns = [
                    col
                    for col in adata.obs.columns
                    if col.lower() in ["condition", "treatment", "group", "batch"]
                ]
                return f"Grouping column '{groupby}' not found. Available experimental design columns: {available_columns}"

            # Check if groups exist
            available_groups = list(adata.obs[groupby].unique())
            if group1 not in available_groups:
                return f"Group '{group1}' not found in column '{groupby}'. Available groups: {available_groups}"
            if group2 not in available_groups:
                return f"Group '{group2}' not found in column '{groupby}'. Available groups: {available_groups}"

            # Use bulk service for differential expression
            adata_de, de_stats, ir = bulk_service.run_differential_expression_analysis(
                adata=adata,
                groupby=groupby,
                group1=group1,
                group2=group2,
                method=method,
                min_expression_threshold=min_expression_threshold,
                min_fold_change=min_fold_change,
                min_pct_expressed=min_pct_expressed,
                max_out_pct_expressed=max_out_pct_expressed,
            )

            # Save as new modality
            de_modality_name = f"{modality_name}_de_{group1}_vs_{group2}"
            data_manager.modalities[de_modality_name] = adata_de

            # Save to file if requested
            if save_result:
                save_path = f"{modality_name}_de_{group1}_vs_{group2}.h5ad"
                data_manager.save_modality(de_modality_name, save_path)

            # Log the operation with IR for provenance tracking
            data_manager.log_tool_usage(
                tool_name="run_differential_expression_analysis",
                parameters={
                    "modality_name": modality_name,
                    "groupby": groupby,
                    "group1": group1,
                    "group2": group2,
                    "method": method,
                    "min_expression_threshold": min_expression_threshold,
                    "min_fold_change": min_fold_change,
                    "min_pct_expressed": min_pct_expressed,
                    "max_out_pct_expressed": max_out_pct_expressed,
                },
                description=f"Bulk RNA-seq DE analysis: {de_stats['n_significant_genes']} significant genes found",
                ir=ir,
            )

            # Format professional response
            response = f"""Bulk RNA-seq Differential Expression Analysis Complete for '{modality_name}'!

📊 **Analysis Results:**
- Comparison: {de_stats['group1']} ({de_stats['n_samples_group1']} samples) vs {de_stats['group2']} ({de_stats['n_samples_group2']} samples)
- Method: {de_stats['method']}
- Genes tested: {de_stats['n_genes_tested']:,}
- Significant genes (padj < 0.05): {de_stats['n_significant_genes']:,}

📈 **Bulk RNA-seq Differential Expression Summary:**
- Upregulated in {group2}: {de_stats['n_upregulated']} genes
- Downregulated in {group2}: {de_stats['n_downregulated']} genes

🧬 **Top Upregulated Genes:**"""

            for gene in de_stats["top_upregulated"][:5]:
                response += f"\n- {gene}"

            response += "\n\n🧬 **Top Downregulated Genes:**"
            for gene in de_stats["top_downregulated"][:5]:
                response += f"\n- {gene}"

            # Add filtering statistics if available
            if 'filtering_params' in de_stats and de_stats.get('filtering_applied'):
                response += f"\n\n🎯 **DEG Filtering Applied:**"
                filter_params = de_stats['filtering_params']
                response += f"\n- Min fold-change: {filter_params['min_fold_change']}x"
                response += f"\n- Min % expressed in {group1}: {filter_params['min_pct_expressed']*100:.1f}%"
                response += f"\n- Max % expressed in {group2}: {filter_params['max_out_pct_expressed']*100:.1f}%"

                response += f"\n\n📊 **Filtering Results:**"
                response += f"\n- Genes before filtering: {de_stats['pre_filter_gene_count']}"
                response += f"\n- Genes after filtering: {de_stats['post_filter_gene_count']}"
                response += f"\n- Genes filtered out: {de_stats['filtered_gene_count']} ({de_stats['filtered_gene_count']/de_stats['pre_filter_gene_count']*100:.1f}%)"

                if de_stats['filtered_gene_count'] > 0:
                    response += f"\n\n💡 **Interpretation:**"
                    if de_stats['filtered_gene_count'] / de_stats['pre_filter_gene_count'] > 0.5:
                        response += f"\n- High filtering rate (>50%) indicates many genes lack biological significance"
                        response += f"\n- This is normal and improves result quality by removing noise"
                    else:
                        response += f"\n- Moderate filtering rate indicates most genes meet significance thresholds"

            # Add confidence distribution if available
            if 'quality_distribution' in de_stats and de_stats.get('confidence_scoring'):
                response += f"\n\n🔍 **Gene Confidence Scoring:**"
                response += f"\n- Mean confidence: {de_stats['mean_confidence']:.3f}"
                response += f"\n- Median confidence: {de_stats['median_confidence']:.3f}"
                response += f"\n- Std deviation: {de_stats['std_confidence']:.3f}"

                quality_dist = de_stats['quality_distribution']
                response += f"\n\n📈 **Gene Quality Distribution:**"
                response += f"\n- High confidence: {quality_dist['high']} genes ({quality_dist['high']/de_stats['post_filter_gene_count']*100:.1f}%)"
                response += f"\n- Medium confidence: {quality_dist['medium']} genes ({quality_dist['medium']/de_stats['post_filter_gene_count']*100:.1f}%)"
                response += f"\n- Low confidence: {quality_dist['low']} genes ({quality_dist['low']/de_stats['post_filter_gene_count']*100:.1f}%)"

                response += f"\n\n💡 **Quality Guidelines:**"
                response += f"\n- **High confidence** (FDR<0.01, log2FC>1.5): Strong candidates for validation"
                response += f"\n- **Medium confidence** (FDR<0.05, log2FC>1.0): Requires additional validation"
                response += f"\n- **Low confidence**: Use with caution, may be false positives"

                # Add column information
                response += f"\n\n📝 **Note:** Confidence scores are stored in:"
                response += f"\n- `{de_modality_name}.var['gene_confidence']` (0-1 scale)"
                response += f"\n- `{de_modality_name}.var['gene_quality']` (high/medium/low)"

            response += f"\n\n💾 **New modality created**: '{de_modality_name}'"

            if save_result:
                response += f"\n💾 **Saved to**: {save_path}"

            response += f"\n📈 **Access detailed results**: adata.uns['{de_stats['de_results_key']}']"
            response += "\n\nUse the significant genes for pathway enrichment analysis or gene set analysis."

            analysis_results["details"]["differential_expression"] = response
            return response

        except (BulkRNASeqError, ModalityNotFoundError) as e:
            logger.error(f"Error in bulk RNA-seq differential expression analysis: {e}")
            return (
                f"Error running bulk RNA-seq differential expression analysis: {str(e)}"
            )
        except Exception as e:
            logger.error(
                f"Unexpected error in bulk RNA-seq differential expression: {e}"
            )
            return f"Unexpected error: {str(e)}"

    @tool
    def run_pathway_enrichment_analysis(
        gene_list: List[str],
        analysis_type: str = "GO",
        modality_name: str = None,
        save_result: bool = True,
    ) -> str:
        """
        Run pathway enrichment analysis on gene lists from bulk RNA-seq differential expression results.

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
                    key for key in adata.uns.keys() if key.startswith("de_results_")
                ]
                if de_keys:
                    de_results = adata.uns[de_keys[0]]  # Use first DE result
                    if isinstance(de_results, dict):
                        # Extract significant genes
                        de_df = pd.DataFrame(de_results)
                        if "padj" in de_df.columns:
                            significant_genes = de_df[
                                de_df["padj"] < 0.05
                            ].index.tolist()
                            if significant_genes:
                                gene_list = significant_genes[:500]  # Top 500 genes
                                logger.info(
                                    f"Extracted {len(gene_list)} significant genes from bulk RNA-seq analysis {modality_name}"
                                )

            if not gene_list or len(gene_list) == 0:
                return "No genes provided for enrichment analysis. Please provide a gene list or run differential expression analysis first."

            logger.info(
                f"Running pathway enrichment on {len(gene_list)} genes from bulk RNA-seq data"
            )

            # Use bulk service for pathway enrichment
            enrichment_df, enrichment_stats, ir = bulk_service.run_pathway_enrichment(
                gene_list=gene_list, analysis_type=analysis_type
            )

            # Log the operation with IR for provenance tracking
            data_manager.log_tool_usage(
                tool_name="run_pathway_enrichment_analysis",
                parameters={
                    "gene_list_size": len(gene_list),
                    "analysis_type": analysis_type,
                    "modality_name": modality_name,
                },
                description=f"Bulk RNA-seq {analysis_type} enrichment: {enrichment_stats['n_significant_terms']} significant terms",
                ir=ir,
            )

            # Format professional response
            response = f"""Bulk RNA-seq {analysis_type} Pathway Enrichment Analysis Complete!

📊 **Enrichment Results:**
- Genes analyzed: {enrichment_stats['n_genes_input']:,}
- Database: {enrichment_stats['enrichment_database']}
- Terms found: {enrichment_stats['n_terms_total']}
- Significant terms (p.adj < 0.05): {enrichment_stats['n_significant_terms']}

🧬 **Top Enriched Pathways:**"""

            for term in enrichment_stats["top_terms"][:8]:
                response += f"\n- {term}"

            if len(enrichment_stats["top_terms"]) > 8:
                remaining = len(enrichment_stats["top_terms"]) - 8
                response += f"\n... and {remaining} more pathways"

            response += "\n\nPathway enrichment reveals biological processes and pathways associated with bulk RNA-seq differential expression."

            analysis_results["details"]["pathway_enrichment"] = response
            return response

        except BulkRNASeqError as e:
            logger.error(f"Error in bulk RNA-seq pathway enrichment: {e}")
            return f"Error running bulk RNA-seq pathway enrichment: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error in bulk RNA-seq pathway enrichment: {e}")
            return f"Unexpected error: {str(e)}"

    @tool
    def create_analysis_summary() -> str:
        """Create a comprehensive summary of all bulk RNA-seq analysis steps performed."""
        try:
            if not analysis_results["details"]:
                return "No bulk RNA-seq analyses have been performed yet. Run some analysis tools first."

            summary = "# Bulk RNA-seq Analysis Summary\n\n"

            for step, details in analysis_results["details"].items():
                summary += f"## {step.replace('_', ' ').title()}\n"
                summary += f"{details}\n\n"

            # Add current modality status
            modalities = data_manager.list_modalities()
            if modalities:
                # Filter for bulk RNA-seq modalities
                bulk_modalities = [
                    mod
                    for mod in modalities
                    if "bulk" in mod.lower()
                    or data_manager._detect_modality_type(mod) == "bulk_rna_seq"
                ]

                summary += "## Current Bulk RNA-seq Modalities\n"
                summary += f"Bulk RNA-seq modalities ({len(bulk_modalities)}): {', '.join(bulk_modalities)}\n\n"

                # Add modality details
                summary += "### Bulk RNA-seq Modality Details:\n"
                for mod_name in bulk_modalities:
                    try:
                        adata = data_manager.get_modality(mod_name)
                        summary += f"- **{mod_name}**: {adata.n_obs} samples × {adata.n_vars} genes\n"

                        # Add key bulk RNA-seq observation columns if present
                        key_cols = [
                            col
                            for col in adata.obs.columns
                            if col.lower()
                            in [
                                "condition",
                                "treatment",
                                "group",
                                "batch",
                                "time_point",
                            ]
                        ]
                        if key_cols:
                            summary += (
                                f"  - Experimental design: {', '.join(key_cols)}\n"
                            )
                    except Exception:
                        summary += f"- **{mod_name}**: Error accessing modality\n"

            analysis_results["summary"] = summary
            logger.info(
                f"Created bulk RNA-seq analysis summary with {len(analysis_results['details'])} analysis steps"
            )
            return summary

        except Exception as e:
            logger.error(f"Error creating bulk RNA-seq analysis summary: {e}")
            return f"Error creating bulk RNA-seq summary: {str(e)}"

    @tool
    def run_deseq2_formula_analysis(
        modality_name: str,
        formula: str,
        contrast: str,  # Format: "factor,level1,level2" (e.g., "condition,treatment,control")
        alpha: float = 0.05,
        shrink_lfc: bool = True,
        n_cpus: int = 1,
    ) -> str:
        """
        Run proper DESeq2 analysis with formula-based experimental design.

        This tool uses pyDESeq2 for rigorous differential expression analysis with
        support for complex experimental designs including batch effects, multiple
        factors, interactions, and continuous covariates.

        DESeq2 workflow:
        1. Estimate size factors (library size normalization)
        2. Estimate dispersions (negative binomial model)
        3. Fit negative binomial GLM with formula
        4. Wald test for specified contrast
        5. Multiple testing correction (Benjamini-Hochberg)
        6. Optional: LFC shrinkage for more accurate effect sizes

        Formula syntax (R-style):
        - Simple comparison: "~condition"
        - With batch correction: "~condition + batch"
        - Multiple factors: "~condition + batch + donor"
        - Interaction effects: "~condition * batch"
        - Continuous covariate: "~condition + age"

        Contrast format:
        - String with 3 comma-separated values: "factor,numerator,denominator"
        - Example: "condition,treatment,control" compares treatment vs control
        - Factor must be a term in the formula

        Args:
            modality_name: Name of bulk RNA-seq modality to analyze
            formula: R-style formula for experimental design (e.g., "~condition + batch")
            contrast: Contrast specification "factor,level1,level2"
            alpha: FDR significance threshold (default: 0.05)
            shrink_lfc: Apply log2 fold-change shrinkage for accuracy (default: True)
            n_cpus: Number of CPUs for parallel processing (default: 1)

        Returns:
            str: Formatted string with analysis results and statistics

        Raises:
            ModalityNotFoundError: If modality doesn't exist
            BulkRNASeqError: If DESeq2 analysis fails

        Example:
            # Simple treatment vs control
            run_deseq2_formula_analysis(
                "bulk_gse12345_filtered_normalized",
                formula="~condition",
                contrast="condition,treatment,control"
            )

            # With batch correction
            run_deseq2_formula_analysis(
                "bulk_gse12345_filtered_normalized",
                formula="~condition + batch",
                contrast="condition,drug_a,placebo"
            )
        """
        try:
            # Validate modality exists
            if modality_name not in data_manager.list_modalities():
                raise ModalityNotFoundError(
                    f"Modality '{modality_name}' not found. "
                    f"Available modalities: {data_manager.list_modalities()}"
                )

            # Get modality
            adata = data_manager.get_modality(modality_name)

            # Validate adata has required structure
            if adata.X is None or len(adata.X) == 0:
                raise BulkRNASeqError(
                    f"Modality '{modality_name}' has no expression data"
                )

            if adata.obs is None or len(adata.obs) == 0:
                raise BulkRNASeqError(
                    f"Modality '{modality_name}' has no sample metadata"
                )

            # Parse contrast string
            try:
                contrast_parts = contrast.split(",")
                if len(contrast_parts) != 3:
                    raise ValueError(
                        f"Contrast must be 'factor,level1,level2' format, got: {contrast}"
                    )
                contrast_list = [
                    contrast_parts[0].strip(),
                    contrast_parts[1].strip(),
                    contrast_parts[2].strip(),
                ]
            except Exception as e:
                raise BulkRNASeqError(f"Invalid contrast format: {e}")

            # Extract count matrix and metadata
            # pyDESeq2 expects genes × samples (transposed from AnnData)
            count_matrix = pd.DataFrame(
                adata.X.T, index=adata.var_names, columns=adata.obs_names
            )
            metadata = adata.obs.copy()

            # Run pyDESeq2 analysis
            try:
                results_df, pydeseq2_stats, ir = bulk_service.run_pydeseq2_analysis(
                    count_matrix=count_matrix,
                    metadata=metadata,
                    formula=formula,
                    contrast=contrast_list,
                    alpha=alpha,
                    shrink_lfc=shrink_lfc,
                    n_cpus=n_cpus,
                )
            except Exception as e:
                raise BulkRNASeqError(f"pyDESeq2 analysis failed: {e}")

            # Create result modality name
            factor = contrast_list[0]
            level1 = contrast_list[1]
            level2 = contrast_list[2]
            deseq2_modality_name = f"{modality_name}_deseq2_{level1}_vs_{level2}"

            # Convert results back to AnnData format
            # Results are genes × stats, so we create a new adata with original samples
            # but add DE results to .var
            adata_deseq2 = adata.copy()

            # Add DESeq2 results to .var
            for col in results_df.columns:
                adata_deseq2.var[col] = results_df[col]

            # Store result
            data_manager.modalities[deseq2_modality_name] = adata_deseq2

            # Log the operation with IR for provenance tracking
            data_manager.log_tool_usage(
                tool_name="run_deseq2_formula_analysis",
                parameters={
                    "modality_name": modality_name,
                    "formula": formula,
                    "contrast": contrast,
                    "alpha": alpha,
                    "shrink_lfc": shrink_lfc,
                    "n_cpus": n_cpus,
                },
                description=f"pyDESeq2 analysis: {pydeseq2_stats['n_significant_genes']} significant genes",
                ir=ir,
            )

            # Format response
            response = f"## pyDESeq2 Differential Expression Analysis Complete\n\n"
            response += f"**Formula**: `{formula}`\n"
            response += f"**Contrast**: {level1} vs {level2} (factor: {factor})\n"
            response += f"**Result stored as**: `{deseq2_modality_name}`\n\n"

            response += f"### Analysis Statistics\n"
            response += f"- Total genes tested: {pydeseq2_stats['total_genes_tested']}\n"
            response += f"- Significant genes (FDR < {alpha}): {pydeseq2_stats['n_significant_genes']}\n"
            response += f"- Up-regulated in {level1}: {pydeseq2_stats['up_regulated']}\n"
            response += f"- Down-regulated in {level1}: {pydeseq2_stats['down_regulated']}\n"
            response += f"- LFC shrinkage applied: {'Yes' if shrink_lfc else 'No'}\n\n"

            response += f"### DESeq2 Workflow\n"
            response += f"✅ Size factor estimation (library normalization)\n"
            response += f"✅ Dispersion estimation (negative binomial)\n"
            response += f"✅ GLM fitting with formula: `{formula}`\n"
            response += f"✅ Wald test for contrast: {level1} vs {level2}\n"
            response += f"✅ Benjamini-Hochberg FDR correction\n"
            if shrink_lfc:
                response += f"✅ Log2 fold-change shrinkage (apeglm)\n"

            response += f"\n### Result Columns in `{deseq2_modality_name}.var`\n"
            response += f"- `baseMean`: Mean normalized counts across all samples\n"
            response += f"- `log2FoldChange`: Log2 fold-change ({level1} / {level2})\n"
            response += f"- `lfcSE`: Standard error of log2FC\n"
            response += f"- `stat`: Wald statistic\n"
            response += f"- `pvalue`: Wald test p-value\n"
            response += f"- `padj`: Benjamini-Hochberg adjusted p-value (FDR)\n\n"

            response += f"💡 **Next steps**: Use pathway enrichment or visualization tools to explore results\n"

            analysis_results["details"]["deseq2_analysis"] = response
            return response

        except ModalityNotFoundError as e:
            logger.error(f"Modality not found error: {e}")
            return f"Error: {str(e)}"
        except BulkRNASeqError as e:
            logger.error(f"Error in pyDESeq2 analysis: {e}")
            return f"Error running pyDESeq2 analysis: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error in pyDESeq2 analysis: {e}")
            return f"Unexpected error: {str(e)}"

    @tool
    def suggest_experimental_designs(modality_name: str) -> str:
        """
        Analyze sample metadata and suggest appropriate statistical formulas for DE analysis.

        This tool examines the metadata structure of your bulk RNA-seq data and recommends
        formulas based on:
        - Number of factors (conditions, batches, donors, etc.)
        - Sample size per group
        - Balance of experimental design
        - Potential confounders

        Returns suggestions for:
        1. Simple formulas (direct comparisons)
        2. Batch-corrected formulas (controlling for technical variation)
        3. Multifactor formulas (multiple covariates)
        4. Interaction formulas (testing interaction effects)

        Each suggestion includes:
        - Formula syntax
        - Complexity level
        - Pros and cons
        - Recommended use case
        - Minimum sample size needed

        Args:
            modality_name: Name of bulk RNA-seq modality to analyze

        Returns:
            str: Formatted suggestions with detailed explanations

        Example:
            suggest_experimental_designs("bulk_gse12345_filtered_normalized")
        """
        try:
            from lobster.services.analysis.differential_formula_service import (
                DifferentialFormulaService,
            )

            # Validate modality exists
            if modality_name not in data_manager.list_modalities():
                raise ModalityNotFoundError(
                    f"Modality '{modality_name}' not found. "
                    f"Available modalities: {data_manager.list_modalities()}"
                )

            # Get modality
            adata = data_manager.get_modality(modality_name)

            # Get metadata
            metadata = adata.obs

            # Get suggestions
            formula_service = DifferentialFormulaService()
            suggestions = formula_service.suggest_formulas(metadata)

            # Format response
            response = f"## Experimental Design Suggestions for `{modality_name}`\n\n"
            response += f"**Sample size**: {len(metadata)} samples\n"
            response += f"**Metadata columns**: {', '.join(metadata.columns.tolist())}\n\n"

            if not suggestions:
                response += "⚠️ **No suitable formulas found**. Check that your metadata has:\n"
                response += "- At least one categorical column for grouping\n"
                response += "- At least 6 samples total\n"
                return response

            response += f"Found **{len(suggestions)} suggested formula(s)**:\n\n"

            for i, suggestion in enumerate(suggestions, 1):
                response += f"### {i}. {suggestion['complexity'].upper()}: `{suggestion['formula']}`\n\n"
                response += f"**Description**: {suggestion['description']}\n\n"

                response += f"**Pros**:\n"
                for pro in suggestion["pros"]:
                    response += f"- ✅ {pro}\n"
                response += "\n"

                response += f"**Cons**:\n"
                for con in suggestion["cons"]:
                    response += f"- ⚠️ {con}\n"
                response += "\n"

                response += (
                    f"**Recommended for**: {suggestion['recommended_for']}\n"
                )
                response += (
                    f"**Minimum samples needed**: {suggestion['min_samples_needed']}\n\n"
                )

                if "warnings" in suggestion:
                    response += f"**Warnings**:\n"
                    for warning in suggestion["warnings"]:
                        response += f"- 🚨 {warning}\n"
                    response += "\n"

                response += "---\n\n"

            response += "💡 **Next step**: Use `preview_design_matrix()` to see how your chosen formula will be encoded\n"

            return response

        except ModalityNotFoundError as e:
            logger.error(f"Modality not found error: {e}")
            return f"Error: {str(e)}"
        except Exception as e:
            logger.error(f"Error suggesting experimental designs: {e}")
            return f"Error: {str(e)}"

    @tool
    def preview_design_matrix(modality_name: str, formula: str) -> str:
        """
        Preview how a formula will be encoded into a design matrix for DESeq2 analysis.

        This tool shows you:
        - First 5 rows of the design matrix
        - All column names (model coefficients)
        - Reference levels for categorical factors
        - Matrix rank and degrees of freedom
        - Whether the design is full-rank (required for DESeq2)

        Use this BEFORE running pyDESeq2 analysis to verify:
        - Your formula is correctly specified
        - Reference levels are appropriate
        - The design is estimable (full rank)
        - Interactions are encoded as expected

        Args:
            modality_name: Name of bulk RNA-seq modality
            formula: R-style formula to preview (e.g., "~condition + batch")

        Returns:
            str: Formatted preview with design matrix details

        Example:
            preview_design_matrix("bulk_gse12345_filtered_normalized", "~condition + batch")
        """
        try:
            from lobster.services.analysis.differential_formula_service import (
                DifferentialFormulaService,
            )

            # Validate modality exists
            if modality_name not in data_manager.list_modalities():
                raise ModalityNotFoundError(
                    f"Modality '{modality_name}' not found. "
                    f"Available modalities: {data_manager.list_modalities()}"
                )

            # Get modality
            adata = data_manager.get_modality(modality_name)
            metadata = adata.obs

            # Get preview
            formula_service = DifferentialFormulaService()
            preview_text = formula_service.preview_design_matrix(formula, metadata)

            # Format response (preview_text is already formatted)
            response = f"## Design Matrix Preview for Formula: `{formula}`\n\n"
            response += f"**Modality**: `{modality_name}`\n"
            response += f"**Samples**: {len(metadata)}\n\n"
            response += preview_text
            response += "\n\n💡 **Next step**: If design looks good, use `run_deseq2_formula_analysis()` to run the analysis\n"

            return response

        except ModalityNotFoundError as e:
            logger.error(f"Modality not found error: {e}")
            return f"Error: {str(e)}"
        except Exception as e:
            logger.error(f"Error previewing design matrix: {e}")
            return f"Error: {str(e)}"

    @tool
    def validate_experimental_design(modality_name: str, formula: str) -> str:
        """
        Validate experimental design for statistical power and balance.

        This tool checks:
        - Sample size adequacy for the formula complexity
        - Balance of groups (equal sample sizes preferred)
        - Design matrix rank (must be full rank)
        - Degrees of freedom available for testing
        - Estimated statistical power (rough approximation)
        - Potential issues (confounding, aliasing, etc.)

        Validation criteria:
        - Simple (~condition): Need 6+ samples
        - Batch-corrected (~condition + batch): Need 8+ samples
        - Multifactor (~condition + batch + donor): Need 12+ samples
        - Interaction (~condition * batch): Need 16+ samples

        Args:
            modality_name: Name of bulk RNA-seq modality
            formula: R-style formula to validate

        Returns:
            str: Formatted validation report with pass/fail status

        Example:
            validate_experimental_design("bulk_gse12345_filtered_normalized", "~condition + batch")
        """
        try:
            from lobster.services.analysis.differential_formula_service import (
                DifferentialFormulaService,
            )

            # Validate modality exists
            if modality_name not in data_manager.list_modalities():
                raise ModalityNotFoundError(
                    f"Modality '{modality_name}' not found. "
                    f"Available modalities: {data_manager.list_modalities()}"
                )

            # Get modality
            adata = data_manager.get_modality(modality_name)
            metadata = adata.obs

            # Validate design
            formula_service = DifferentialFormulaService()
            validation_result = formula_service.validate_experimental_design(
                metadata, formula
            )

            # Format response
            response = f"## Experimental Design Validation: `{formula}`\n\n"
            response += f"**Modality**: `{modality_name}`\n"
            response += f"**Samples**: {len(metadata)}\n\n"

            # Overall status
            if validation_result["valid"]:
                response += "✅ **Overall Status**: PASSED - Design is valid for DESeq2 analysis\n\n"
            else:
                response += "❌ **Overall Status**: FAILED - Design has issues (see below)\n\n"

            # Design summary
            if validation_result.get("design_summary"):
                response += f"### Group Balance\n"
                for factor, counts in validation_result["design_summary"].items():
                    response += f"**{factor}**:\n"
                    for level, count in counts.items():
                        response += f"  - {level}: {count} samples\n"
                response += "\n"

            # Warnings
            if validation_result.get("warnings"):
                response += f"### Warnings\n"
                for warning in validation_result["warnings"]:
                    response += f"- ⚠️ {warning}\n"
                response += "\n"

            # Errors
            if validation_result.get("errors"):
                response += f"### Errors\n"
                for error in validation_result["errors"]:
                    response += f"- 🚨 {error}\n"
                response += "\n"

            if validation_result["valid"]:
                response += "✅ **Conclusion**: Design is ready for pyDESeq2 analysis\n"
            else:
                response += (
                    "❌ **Conclusion**: Please address issues before running analysis\n"
                )

            return response

        except ModalityNotFoundError as e:
            logger.error(f"Modality not found error: {e}")
            return f"Error: {str(e)}"
        except Exception as e:
            logger.error(f"Error validating experimental design: {e}")
            return f"Error: {str(e)}"

    @tool
    def create_pseudobulk_from_singlecell(
        singlecell_modality: str,
        sample_col: str,
        celltype_col: str,
        min_cells: int = 10,
        aggregation_method: str = "sum",
        filter_cell_types: Optional[str] = None,
    ) -> str:
        """
        Aggregate single-cell RNA-seq data to pseudobulk for differential expression.

        Pseudobulk analysis aggregates cells from the same sample and cell type into a
        single "bulk-like" expression profile. This enables:
        1. Sample-level statistical testing (vs. cell-level pseudo-replication)
        2. Use of standard bulk DE tools (DESeq2, edgeR)
        3. Proper control of donor/sample variation

        Workflow:
        1. Group cells by sample_id + cell_type
        2. Aggregate expression using specified method (sum recommended)
        3. Filter groups with < min_cells (removes unreliable estimates)
        4. Create pseudobulk AnnData with one row per sample/celltype combination

        The resulting pseudobulk data can then be analyzed with pyDESeq2 for:
        - Comparing conditions across samples (treatment vs control)
        - Cell-type-specific differential expression
        - Longitudinal studies (time-series)

        Args:
            singlecell_modality: Name of single-cell modality to aggregate
            sample_col: Column in .obs with sample IDs (e.g., "donor_id", "patient_id")
            celltype_col: Column in .obs with cell type annotations (e.g., "cell_type")
            min_cells: Minimum cells required per pseudobulk sample (default: 10)
            aggregation_method: How to aggregate (default: "sum" for count data)
                - "sum": Sum counts (recommended for DE analysis)
                - "mean": Average counts (normalized data)
                - "median": Median counts (robust to outliers)
            filter_cell_types: Optional comma-separated list of cell types to include
                (e.g., "T cells,NK cells,B cells")

        Returns:
            str: Formatted string with pseudobulk statistics and next steps

        Example:
            # Aggregate all cell types
            create_pseudobulk_from_singlecell(
                "geo_gse12345_annotated",
                sample_col="donor_id",
                celltype_col="cell_type_annotation",
                min_cells=10
            )

            # Aggregate only specific cell types
            create_pseudobulk_from_singlecell(
                "geo_gse12345_annotated",
                sample_col="patient_id",
                celltype_col="cell_type",
                min_cells=20,
                filter_cell_types="T cells,B cells,Monocytes"
            )
        """
        from lobster.services.analysis.pseudobulk_service import PseudobulkService

        # Validate modality exists
        if singlecell_modality not in data_manager.list_modalities():
            raise ModalityNotFoundError(
                f"Modality '{singlecell_modality}' not found. "
                f"Available modalities: {data_manager.list_modalities()}"
            )

        # Get single-cell modality
        sc_adata = data_manager.get_modality(singlecell_modality)

        # Validate required columns
        if sample_col not in sc_adata.obs.columns:
            raise BulkRNASeqError(
                f"Sample column '{sample_col}' not found in modality. "
                f"Available columns: {sc_adata.obs.columns.tolist()}"
            )

        if celltype_col not in sc_adata.obs.columns:
            raise BulkRNASeqError(
                f"Cell type column '{celltype_col}' not found in modality. "
                f"Available columns: {sc_adata.obs.columns.tolist()}"
            )

        # Filter cell types if specified
        if filter_cell_types:
            cell_types = [ct.strip() for ct in filter_cell_types.split(",")]
            mask = sc_adata.obs[celltype_col].isin(cell_types)
            sc_adata = sc_adata[mask].copy()

            if len(sc_adata) == 0:
                raise BulkRNASeqError(
                    f"No cells found for specified cell types: {cell_types}"
                )

        # Aggregate to pseudobulk
        pseudobulk_service = PseudobulkService()
        try:
            pseudobulk_adata, pb_stats, ir = pseudobulk_service.aggregate_to_pseudobulk(
                adata=sc_adata,
                sample_col=sample_col,
                celltype_col=celltype_col,
                min_cells=min_cells,
                aggregation_method=aggregation_method,
            )
        except Exception as e:
            raise BulkRNASeqError(f"Pseudobulk aggregation failed: {e}")

        # Create result modality name
        pseudobulk_modality_name = f"{singlecell_modality}_pseudobulk"

        # Store result
        data_manager.modalities[pseudobulk_modality_name] = pseudobulk_adata

        # Log the operation with IR
        data_manager.log_tool_usage(
            tool_name="create_pseudobulk_from_singlecell",
            parameters={
                "singlecell_modality": singlecell_modality,
                "sample_col": sample_col,
                "celltype_col": celltype_col,
                "min_cells": min_cells,
                "aggregation_method": aggregation_method,
                "filter_cell_types": filter_cell_types,
            },
            description=f"Pseudobulk aggregation: {pb_stats['n_pseudobulk_samples']} samples created",
            ir=ir,
        )

        # Format response
        response = f"## Pseudobulk Aggregation Complete\n\n"
        response += f"**Source**: `{singlecell_modality}` (single-cell)\n"
        response += f"**Result**: `{pseudobulk_modality_name}` (pseudobulk)\n\n"

        response += f"### Aggregation Statistics\n"
        response += f"- Single cells processed: {pb_stats['total_cells_processed']:,}\n"
        response += f"- Cells aggregated: {pb_stats['total_cells_aggregated']:,}\n"
        response += f"- Cells filtered (< {min_cells}): {pb_stats.get('cells_filtered', 0):,}\n"
        response += f"- Pseudobulk samples created: {pb_stats['n_pseudobulk_samples']}\n"
        response += f"- Unique samples: {pb_stats['n_unique_samples']}\n"
        response += f"- Unique cell types: {pb_stats['n_cell_types']}\n"
        response += f"- Aggregation method: {aggregation_method}\n\n"

        if 'cell_type_summary' in pb_stats:
            response += f"### Pseudobulk Samples per Cell Type\n"
            for ct, count in pb_stats['cell_type_summary'].items():
                response += f"- {ct}: {count} samples\n"
            response += "\n"

        response += f"### Next Steps\n"
        response += f"✅ Use pyDESeq2 for differential expression:\n"
        response += f"```\n"
        response += f"run_deseq2_formula_analysis(\n"
        response += f"    '{pseudobulk_modality_name}',\n"
        response += f"    formula='~condition',  # Adjust based on your design\n"
        response += f"    contrast='condition,treatment,control'\n"
        response += f")\n"
        response += f"```\n\n"

        response += f"💡 **Tip**: Pseudobulk enables proper sample-level statistics by avoiding\n"
        response += f"cell-level pseudo-replication. Each pseudobulk sample represents one\n"
        response += f"biological replicate (donor/patient).\n"

        return response

    # -------------------------
    # BULK RNA-SEQ VISUALIZATION TOOLS
    # -------------------------
    @tool
    def create_volcano_plot(
        de_modality_name: str,
        fdr_threshold: float = 0.05,
        fc_threshold: float = 1.0,
        top_n_genes: int = 10,
    ) -> str:
        """
        Create a volcano plot for differential expression results.

        Volcano plots visualize the relationship between statistical significance
        (-log10 FDR) and biological significance (log2 fold-change), helping
        identify the most interesting differentially expressed genes.

        Args:
            de_modality_name: Name of modality with DE results (must have log2FoldChange and padj columns)
            fdr_threshold: FDR significance threshold (default: 0.05)
            fc_threshold: Log2 fold-change threshold (default: 1.0)
            top_n_genes: Number of top genes to label (default: 10)

        Returns:
            str: Formatted string with plot statistics and storage location

        Example:
            create_volcano_plot(
                "bulk_gse12345_de_treatment_vs_control",
                fdr_threshold=0.01,
                fc_threshold=1.5,
                top_n_genes=15
            )
        """
        try:
            from lobster.services.visualization.bulk_visualization_service import (
                BulkVisualizationService,
            )

            # Validate modality exists
            if de_modality_name not in data_manager.list_modalities():
                raise ModalityNotFoundError(
                    f"Modality '{de_modality_name}' not found. "
                    f"Available modalities: {data_manager.list_modalities()}"
                )

            # Get modality
            adata = data_manager.get_modality(de_modality_name)

            # Validate required columns
            required_cols = ["log2FoldChange", "padj"]
            missing_cols = [col for col in required_cols if col not in adata.var.columns]
            if missing_cols:
                return (
                    f"Error: Modality '{de_modality_name}' is missing required DE result columns: {missing_cols}. "
                    f"Available columns: {list(adata.var.columns)}. "
                    f"Please run differential expression analysis first."
                )

            # Create volcano plot
            viz_service = BulkVisualizationService()
            fig, stats, ir = viz_service.create_volcano_plot(
                adata=adata,
                fdr_threshold=fdr_threshold,
                fc_threshold=fc_threshold,
                top_n_genes=top_n_genes,
            )

            # Store plot in data manager
            plot_name = f"{de_modality_name}_volcano"
            data_manager.plots[plot_name] = fig

            # Log the operation with IR
            data_manager.log_tool_usage(
                tool_name="create_volcano_plot",
                parameters={
                    "de_modality_name": de_modality_name,
                    "fdr_threshold": fdr_threshold,
                    "fc_threshold": fc_threshold,
                    "top_n_genes": top_n_genes,
                },
                description=f"Volcano plot: {stats['n_genes_up']} up, {stats['n_genes_down']} down genes",
                ir=ir,
            )

            # Format response
            response = f"## Volcano Plot Created for '{de_modality_name}'\n\n"
            response += f"**Plot stored as**: `{plot_name}`\n\n"
            response += f"### Visualization Statistics\n"
            response += f"- Total genes: {stats['n_genes_total']:,}\n"
            response += f"- Upregulated: {stats['n_genes_up']} (FDR < {fdr_threshold}, log2FC > {fc_threshold})\n"
            response += f"- Downregulated: {stats['n_genes_down']} (FDR < {fdr_threshold}, log2FC < -{fc_threshold})\n"
            response += f"- Not significant: {stats['n_genes_not_significant']:,}\n"
            response += f"- Top genes labeled: {stats['top_n_genes_labeled']}\n\n"

            response += f"### Plot Features\n"
            response += f"- X-axis: log2 fold-change (effect size)\n"
            response += f"- Y-axis: -log10(FDR) (statistical significance)\n"
            response += f"- Red points: Upregulated genes\n"
            response += f"- Blue points: Downregulated genes\n"
            response += f"- Gray points: Not significant\n"
            response += f"- Dashed lines: Significance thresholds\n\n"

            response += f"💡 **Interpretation**: Genes in the upper corners (high -log10(FDR) and high |log2FC|) are both statistically and biologically significant.\n\n"
            response += f"📊 **Access plot**: Use `/plots` command to view or `data_manager.plots['{plot_name}']` in code\n"

            return response

        except ModalityNotFoundError as e:
            logger.error(f"Modality not found error: {e}")
            return f"Error: {str(e)}"
        except Exception as e:
            logger.error(f"Error creating volcano plot: {e}")
            return f"Error creating volcano plot: {str(e)}"

    @tool
    def create_ma_plot(
        de_modality_name: str,
        fdr_threshold: float = 0.05,
    ) -> str:
        """
        Create an MA plot for differential expression results.

        MA plots show the relationship between mean expression level and
        fold-change, helping identify expression-dependent biases and
        assess the overall distribution of DE genes.

        Args:
            de_modality_name: Name of modality with DE results (must have log2FoldChange, padj, baseMean columns)
            fdr_threshold: FDR significance threshold (default: 0.05)

        Returns:
            str: Formatted string with plot statistics and storage location

        Example:
            create_ma_plot(
                "bulk_gse12345_de_treatment_vs_control",
                fdr_threshold=0.01
            )
        """
        try:
            from lobster.services.visualization.bulk_visualization_service import (
                BulkVisualizationService,
            )

            # Validate modality exists
            if de_modality_name not in data_manager.list_modalities():
                raise ModalityNotFoundError(
                    f"Modality '{de_modality_name}' not found. "
                    f"Available modalities: {data_manager.list_modalities()}"
                )

            # Get modality
            adata = data_manager.get_modality(de_modality_name)

            # Validate required columns
            required_cols = ["log2FoldChange", "padj", "baseMean"]
            missing_cols = [col for col in required_cols if col not in adata.var.columns]
            if missing_cols:
                return (
                    f"Error: Modality '{de_modality_name}' is missing required DE result columns: {missing_cols}. "
                    f"Available columns: {list(adata.var.columns)}. "
                    f"Please run differential expression analysis first."
                )

            # Create MA plot
            viz_service = BulkVisualizationService()
            fig, stats, ir = viz_service.create_ma_plot(
                adata=adata,
                fdr_threshold=fdr_threshold,
            )

            # Store plot in data manager
            plot_name = f"{de_modality_name}_ma"
            data_manager.plots[plot_name] = fig

            # Log the operation with IR
            data_manager.log_tool_usage(
                tool_name="create_ma_plot",
                parameters={
                    "de_modality_name": de_modality_name,
                    "fdr_threshold": fdr_threshold,
                },
                description=f"MA plot: {stats['n_genes_significant']} significant genes",
                ir=ir,
            )

            # Format response
            response = f"## MA Plot Created for '{de_modality_name}'\n\n"
            response += f"**Plot stored as**: `{plot_name}`\n\n"
            response += f"### Visualization Statistics\n"
            response += f"- Total genes: {stats['n_genes_total']:,}\n"
            response += f"- Significant genes (FDR < {fdr_threshold}): {stats['n_genes_significant']}\n"
            response += f"- Mean expression: {stats['mean_base_mean']:.1f}\n"
            response += f"- Median expression: {stats['median_base_mean']:.1f}\n\n"

            response += f"### Plot Features\n"
            response += f"- X-axis: log10(mean expression) (expression level)\n"
            response += f"- Y-axis: log2 fold-change (effect size)\n"
            response += f"- Red/Blue points: Significant genes (up/down)\n"
            response += f"- Gray points: Not significant\n"
            response += f"- Dashed line at y=0: No change\n\n"

            response += f"💡 **Interpretation**: MA plots help identify expression-dependent biases. Genes should be evenly distributed above and below y=0 for well-normalized data.\n\n"
            response += f"📊 **Access plot**: Use `/plots` command to view or `data_manager.plots['{plot_name}']` in code\n"

            return response

        except ModalityNotFoundError as e:
            logger.error(f"Modality not found error: {e}")
            return f"Error: {str(e)}"
        except Exception as e:
            logger.error(f"Error creating MA plot: {e}")
            return f"Error creating MA plot: {str(e)}"

    @tool
    def create_expression_heatmap(
        modality_name: str,
        gene_list: Optional[str] = None,
        cluster_samples: bool = True,
        cluster_genes: bool = True,
        z_score: bool = True,
    ) -> str:
        """
        Create a hierarchical clustered heatmap of gene expression.

        Expression heatmaps visualize gene expression patterns across samples,
        revealing sample relationships and gene co-expression patterns through
        hierarchical clustering.

        Args:
            modality_name: Name of modality with expression data
            gene_list: Comma-separated list of genes to include (uses top 50 variable genes if None)
            cluster_samples: Whether to cluster samples hierarchically (default: True)
            cluster_genes: Whether to cluster genes hierarchically (default: True)
            z_score: Whether to z-score normalize expression (default: True)

        Returns:
            str: Formatted string with plot statistics and storage location

        Example:
            # Heatmap of top variable genes
            create_expression_heatmap(
                "bulk_gse12345_filtered_normalized",
                cluster_samples=True,
                cluster_genes=True,
                z_score=True
            )

            # Heatmap of specific genes
            create_expression_heatmap(
                "bulk_gse12345_filtered_normalized",
                gene_list="TP53,MYC,BRCA1,EGFR,KRAS",
                cluster_samples=True,
                cluster_genes=False,
                z_score=True
            )
        """
        try:
            from lobster.services.visualization.bulk_visualization_service import (
                BulkVisualizationService,
            )

            # Validate modality exists
            if modality_name not in data_manager.list_modalities():
                raise ModalityNotFoundError(
                    f"Modality '{modality_name}' not found. "
                    f"Available modalities: {data_manager.list_modalities()}"
                )

            # Get modality
            adata = data_manager.get_modality(modality_name)

            # Parse gene list
            genes = None
            if gene_list is not None and gene_list.strip():
                genes = [g.strip() for g in gene_list.split(",")]

            # Create expression heatmap
            viz_service = BulkVisualizationService()
            fig, stats, ir = viz_service.create_expression_heatmap(
                adata=adata,
                gene_list=genes,
                cluster_samples=cluster_samples,
                cluster_genes=cluster_genes,
                z_score=z_score,
            )

            # Store plot in data manager
            plot_name = f"{modality_name}_heatmap"
            data_manager.plots[plot_name] = fig

            # Log the operation with IR
            data_manager.log_tool_usage(
                tool_name="create_expression_heatmap",
                parameters={
                    "modality_name": modality_name,
                    "n_genes": stats["n_genes"],
                    "cluster_samples": cluster_samples,
                    "cluster_genes": cluster_genes,
                    "z_score": z_score,
                },
                description=f"Expression heatmap: {stats['n_genes']} genes × {stats['n_samples']} samples",
                ir=ir,
            )

            # Format response
            response = f"## Expression Heatmap Created for '{modality_name}'\n\n"
            response += f"**Plot stored as**: `{plot_name}`\n\n"
            response += f"### Visualization Statistics\n"
            response += f"- Genes plotted: {stats['n_genes']}\n"
            response += f"- Samples plotted: {stats['n_samples']}\n"
            response += f"- Samples clustered: {'Yes' if stats['clustered_samples'] else 'No'}\n"
            response += f"- Genes clustered: {'Yes' if stats['clustered_genes'] else 'No'}\n"
            response += f"- Z-score normalized: {'Yes' if stats['z_score_normalized'] else 'No'}\n\n"

            response += f"### Plot Features\n"
            response += f"- Rows: Genes ({stats['n_genes']})\n"
            response += f"- Columns: Samples ({stats['n_samples']})\n"
            if stats["z_score_normalized"]:
                response += f"- Color scale: Blue (low) → White (mean) → Red (high) [z-scores]\n"
            else:
                response += f"- Color scale: Low (dark) → High (bright) expression\n"
            if stats["clustered_samples"] or stats["clustered_genes"]:
                response += f"- Hierarchical clustering: Ward linkage method\n\n"
            else:
                response += "\n"

            response += f"💡 **Interpretation**:\n"
            if stats["clustered_samples"]:
                response += f"- Samples with similar expression patterns cluster together\n"
            if stats["clustered_genes"]:
                response += f"- Co-expressed genes cluster together\n"
            if stats["z_score_normalized"]:
                response += f"- Z-scores show relative expression (deviations from mean)\n"
            response += f"\n"

            response += f"📊 **Access plot**: Use `/plots` command to view or `data_manager.plots['{plot_name}']` in code\n"

            return response

        except ModalityNotFoundError as e:
            logger.error(f"Modality not found error: {e}")
            return f"Error: {str(e)}"
        except Exception as e:
            logger.error(f"Error creating expression heatmap: {e}")
            return f"Error creating expression heatmap: {str(e)}"

    # -------------------------
    # TOOL REGISTRY
    # -------------------------
    base_tools = [
        check_data_status,
        assess_data_quality,
        filter_and_normalize_modality,
        run_differential_expression_analysis,
        run_pathway_enrichment_analysis,
        create_analysis_summary,
        run_deseq2_formula_analysis,
        suggest_experimental_designs,
        preview_design_matrix,
        validate_experimental_design,
        create_pseudobulk_from_singlecell,  # Pseudobulk aggregation for single-cell to bulk analysis
        create_volcano_plot,  # Visualization: Volcano plot for DE results
        create_ma_plot,  # Visualization: MA plot for DE results
        create_expression_heatmap,  # Visualization: Hierarchical clustered heatmap
    ]

    tools = base_tools + (handoff_tools or [])

    # -------------------------
    # SYSTEM PROMPT
    # -------------------------
    system_prompt = f"""
You are an expert bioinformatician specializing exclusively in bulk RNA-seq analysis using the professional, modular DataManagerV2 system.

<Role>
You execute comprehensive bulk RNA-seq analysis pipelines with proper quality control, preprocessing, differential expression analysis, and biological interpretation. You work with individual modalities in a multi-omics framework with full provenance tracking and professional-grade error handling.

**CRITICAL: You ONLY perform analysis tasks specifically requested by the supervisor. You report results back to the supervisor, never directly to users.**
</Role>

<Communication Flow>
**USER → SUPERVISOR → YOU → SUPERVISOR → USER**
- You receive tasks from the supervisor
- You execute the requested analysis
- You report results back to the supervisor
- The supervisor communicates with the user
</Communication Flow>

<Task>
You perform bulk RNA-seq analysis following current best practices:
1. **Bulk RNA-seq data quality assessment** with comprehensive QC metrics and validation
2. **Professional preprocessing** with sample/gene filtering, normalization, and batch correction
3. **Differential expression analysis** using DESeq2-like methods between experimental groups
4. **Pathway enrichment analysis** using GO/KEGG databases for biological interpretation
5. **Statistical validation** with proper multiple testing correction and effect size estimation
6. **Comprehensive reporting** with analysis summaries and provenance tracking
</Task>

<Available Bulk RNA-seq Tools>
- `check_data_status`: Check loaded bulk RNA-seq modalities and comprehensive status information
- `assess_data_quality`: Professional QC assessment with bulk RNA-seq specific statistical summaries
- `filter_and_normalize_modality`: Advanced filtering with bulk RNA-seq standards and read count normalization
- `run_differential_expression_analysis`: DESeq2-like differential expression between experimental groups
- `run_pathway_enrichment_analysis`: GO/KEGG pathway enrichment analysis for biological interpretation
- `create_analysis_summary`: Comprehensive bulk RNA-seq analysis report with modality tracking

<Professional Bulk RNA-seq Workflows & Tool Usage Order>

## 1. BULK RNA-SEQ QC AND PREPROCESSING WORKFLOWS

### Loading Kallisto/Salmon Quantification Files (Supervisor Request: "Load quantification files from directory")

**IMPORTANT**: Kallisto/Salmon quantification directories are loaded via the CLI `/read` command, NOT through agent tools.

When the supervisor requests loading quantification files:
1. The user must use: `/read /path/to/quantification_directory`
2. The CLI automatically detects Kallisto or Salmon signatures
3. The system merges per-sample files and creates the modality
4. Once loaded, verify data with `check_data_status()`

**Agent Response Template**:
"To load Kallisto/Salmon quantification files, please use the CLI command:
`/read /path/to/quantification_directory`

The system will automatically:
- Detect whether files are Kallisto or Salmon format
- Merge per-sample quantification files
- Create an AnnData modality with correct orientation (samples × genes)

After loading, I can help with quality control and downstream analysis."


### Basic Quality Control Assessment (Supervisor Request: "Run QC on bulk RNA-seq data")
bash
# Step 1: Check what bulk RNA-seq data is available
check_data_status()

# Step 2: Assess quality of specific modality requested by supervisor
assess_data_quality("bulk_gse12345", min_genes=1000, max_mt_pct=50.0)

# Step 3: Report results back to supervisor with QC recommendations
# DO NOT proceed to next steps unless supervisor specifically requests it


### Bulk RNA-seq Preprocessing (Supervisor Request: "Filter and normalize bulk RNA-seq data")
bash
# Step 1: Verify data status first
check_data_status("bulk_gse12345")

# Step 2: Filter and normalize as requested by supervisor
filter_and_normalize_modality("bulk_gse12345", min_genes_per_sample=1000, target_sum=1000000, normalization_method="log1p")

# Step 3: Report completion to supervisor
# WAIT for supervisor instruction before proceeding


## 2. BULK RNA-SEQ ANALYSIS WORKFLOWS

### Differential Expression Analysis (Supervisor Request: "Run differential expression analysis")
bash
# Step 1: Check preprocessed data and experimental design
check_data_status("bulk_gse12345_filtered_normalized")

# Step 2: Run DE analysis between specified groups
run_differential_expression_analysis("bulk_gse12345_filtered_normalized",
                                   groupby="condition",
                                   group1="control",
                                   group2="treatment",
                                   method="deseq2_like")

# Step 3: Report DE results to supervisor
# DO NOT automatically proceed to pathway enrichment


### Differential Expression with Custom Filtering (Supervisor Request: "Run stringent DE analysis for biomarkers")
bash
# Step 1: Check data status
check_data_status("bulk_gse12345_filtered_normalized")

# Step 2: Run DE with stringent filtering for high-confidence biomarkers
run_differential_expression_analysis(
    "bulk_gse12345_filtered_normalized",
    groupby="disease",
    group1="tumor",
    group2="normal",
    method="deseq2_like",
    min_fold_change=2.0,           # NEW: Require 2x upregulation
    min_pct_expressed=0.25,        # NEW: Require 25% expression in tumor
    max_out_pct_expressed=0.3      # NEW: Limit 30% expression in normal
)

# Step 3: Report results with filtering statistics and confidence scores
# The response will include:
# - Standard DE results (significant genes, top up/downregulated)
# - Filtering statistics (pre/post counts, % filtered)
# - Per-gene confidence scores (high/medium/low distribution)
# - Quality guidelines for interpretation


### Pathway Enrichment Analysis (Supervisor Request: "Run pathway enrichment analysis")
bash
# Step 1: Check for DE results or use provided gene list
check_data_status("bulk_gse12345_de_control_vs_treatment")

# Step 2: Run pathway enrichment as requested
run_pathway_enrichment_analysis(gene_list=[], 
                               analysis_type="GO", 
                               modality_name="bulk_gse12345_de_control_vs_treatment")

# Step 3: Report enrichment results to supervisor


## 3. COMPREHENSIVE ANALYSIS WORKFLOWS

### Complete Bulk RNA-seq Pipeline (Supervisor Request: "Run full bulk RNA-seq analysis")
bash
# Step 1: Check initial data
check_data_status()

# Step 2: Quality assessment
assess_data_quality("bulk_gse12345")

# Step 3: Preprocessing
filter_and_normalize_modality("bulk_gse12345", min_genes_per_sample=1000, target_sum=1000000)

# Step 4: Differential expression analysis
run_differential_expression_analysis("bulk_gse12345_filtered_normalized", 
                                   groupby="condition", 
                                   group1="control", 
                                   group2="treatment")

# Step 5: Pathway enrichment analysis
run_pathway_enrichment_analysis(gene_list=[], 
                               analysis_type="GO", 
                               modality_name="bulk_gse12345_de_control_vs_treatment")

# Step 6: Generate comprehensive report
create_analysis_summary()


# =============================================================================
# NEW FEATURE: DEG Filtering for Biological Significance
# =============================================================================
#
# The run_differential_expression_analysis tool now supports post-hoc filtering
# to ensure genes meet biological significance thresholds. This removes genes
# that are statistically significant but lack biological relevance.
#
# FILTERING PARAMETERS:
#
# 1. min_fold_change (default: 1.5)
#    - Minimum fold-change threshold (linear scale)
#    - Ensures genes show substantial expression differences
#    - Examples:
#      * 1.5x = 50% increase (moderate, recommended)
#      * 2.0x = 100% increase (stringent)
#      * 1.2x = 20% increase (permissive)
#    - Biological rationale: Small fold-changes often lack functional impact
#
# 2. min_pct_expressed (default: 0.1)
#    - Minimum fraction of samples expressing the gene in target group (group1)
#    - Ensures gene is actually present, not rare outliers
#    - Examples:
#      * 0.1 = 10% of samples (permissive, good for bulk)
#      * 0.25 = 25% of samples (moderate)
#      * 0.5 = 50% of samples (stringent)
#    - Biological rationale: Genes expressed in few samples may be noise
#
# 3. max_out_pct_expressed (default: 0.5)
#    - Maximum fraction of samples expressing the gene in comparison group (group2)
#    - Ensures gene is relatively specific to target group
#    - Examples:
#      * 0.5 = 50% of comparison samples (moderate, recommended)
#      * 0.3 = 30% of comparison samples (stringent)
#      * 0.7 = 70% of comparison samples (permissive)
#    - Biological rationale: Ubiquitous genes lack specificity
#
# WHEN TO ADJUST THRESHOLDS:
#
# Use STRINGENT filtering (high thresholds) when:
# - Prioritizing genes for expensive validation experiments
# - Working with noisy datasets
# - Seeking highly specific biomarkers
# Example: min_fold_change=2.0, min_pct_expressed=0.25, max_out_pct_expressed=0.3
#
# Use PERMISSIVE filtering (low thresholds) when:
# - Exploratory analysis
# - Working with low-quality samples (degraded RNA)
# - Pathway enrichment (need more genes)
# Example: min_fold_change=1.2, min_pct_expressed=0.05, max_out_pct_expressed=0.7
#
# EXAMPLE USAGE:
#
# # Standard analysis (recommended defaults)
run_differential_expression_analysis(
    "bulk_gse12345_filtered_normalized",
    groupby="condition",
    group1="treatment",
    group2="control",
    method="deseq2_like",
    min_fold_change=1.5,           # Filter genes <1.5x upregulated
    min_pct_expressed=0.1,         # Require 10% expression in treatment
    max_out_pct_expressed=0.5      # Limit 50% expression in control
)
#
# # Stringent filtering for biomarker discovery
run_differential_expression_analysis(
    "bulk_gse12345_filtered_normalized",
    groupby="disease_status",
    group1="cancer",
    group2="healthy",
    method="deseq2_like",
    min_fold_change=2.0,           # Require 2x upregulation
    min_pct_expressed=0.25,        # Require 25% expression in cancer
    max_out_pct_expressed=0.3      # Limit 30% expression in healthy
)
#
# # Permissive filtering for pathway analysis
run_differential_expression_analysis(
    "bulk_gse12345_filtered_normalized",
    groupby="treatment",
    group1="drug_a",
    group2="placebo",
    method="deseq2_like",
    min_fold_change=1.2,           # Accept small changes
    min_pct_expressed=0.05,        # Accept rarely expressed genes
    max_out_pct_expressed=0.7      # Accept less specific genes
)

# =============================================================================
# NEW FEATURE: Per-Gene Confidence Scoring
# =============================================================================
#
# Differential expression results now include confidence scores for each gene,
# quantifying how reliable the DE call is. This helps prioritize genes for
# follow-up validation and identify potential false positives.
#
# CONFIDENCE CALCULATION:
#
# Confidence scores (0-1 scale) combine three factors:
#
# 1. FDR (adjusted p-value) - 50% weight
#    - Measures statistical significance
#    - Lower FDR = higher confidence
#
# 2. Log2 fold-change - 30% weight
#    - Measures biological significance
#    - Larger fold-change = higher confidence
#
# 3. Expression level - 20% weight
#    - Measures data quality
#    - Higher expression = more reliable quantification
#
# QUALITY CATEGORIES:
#
# - HIGH confidence: FDR < 0.01 AND log2FC > 1.5 AND expression > 0.3
#   * Strong statistical and biological signal
#   * Recommended for experimental validation
#   * Likely to replicate in independent studies
#
# - MEDIUM confidence: FDR < 0.05 AND log2FC > 1.0 AND expression > 0.1
#   * Decent statistical signal, moderate biological effect
#   * Requires additional validation
#   * May replicate with larger sample sizes
#
# - LOW confidence: All others
#   * Weak signal or poor data quality
#   * Use with caution
#   * High risk of false positives
#
# ACCESSING CONFIDENCE DATA:
#
# Confidence scores are stored in the DE result modality:
# - modality.var['gene_confidence']: Numeric scores (0-1)
# - modality.var['gene_quality']: Categorical labels ('high', 'medium', 'low')
#
# You can filter genes by confidence for downstream analysis:
# - High confidence genes for validation: adata[:, adata.var['gene_quality'] == 'high']
# - All confident genes for pathways: adata[:, adata.var['gene_quality'].isin(['high', 'medium'])]
#
# INTERPRETATION TIPS:
#
# - If >50% of genes are LOW confidence:
#   * May indicate poor sample quality or low biological effect
#   * Consider increasing sample size or improving RNA quality
#   * May need more stringent filtering parameters
#
# - If >80% of genes are HIGH confidence:
#   * Strong, reliable signal
#   * Results likely to replicate
#   * Good dataset for biomarker discovery
#
# - If confidence distribution is balanced:
#   * Normal for moderate effect sizes
#   * Prioritize high-confidence genes for validation
#   * Use medium-confidence genes for exploratory analysis

# =============================================================================
# FEATURE: pyDESeq2 Integration with Formula-Based Experimental Design
# =============================================================================
#
# The bulk RNA-seq expert now supports proper DESeq2 analysis via pyDESeq2,
# enabling rigorous statistical modeling of complex experimental designs.
#
# WHEN TO USE pyDESeq2 vs. SIMPLE DE:
#
# Use pyDESeq2 when:
# - You need to control for batch effects or confounders
# - You have multiple factors (treatment + time + donor)
# - You want to test interaction effects
# - You need accurate log2 fold-change estimates (with shrinkage)
# - You require publication-quality statistical rigor
#
# Use simple DE (run_differential_expression_analysis) when:
# - Quick exploratory analysis
# - Simple 2-group comparisons with no confounders
# - Preliminary screening before pyDESeq2
#
# WORKFLOW: Formula-Based DE Analysis
#
# Step 1: Suggest formulas based on metadata
suggest_experimental_designs("bulk_modality")
# Returns recommendations: simple, batch-corrected, multifactor, interaction
#
# Step 2: Preview design matrix (optional but recommended)
preview_design_matrix("bulk_modality", "~condition + batch")
# Shows how formula will be encoded, reference levels, rank
#
# Step 3: Validate design (optional but recommended)
validate_experimental_design("bulk_modality", "~condition + batch")
# Checks sample size, balance, power, issues
#
# Step 4: Run pyDESeq2 analysis
run_deseq2_formula_analysis(
    "bulk_modality",
    formula="~condition + batch",
    contrast="condition,treatment,control",
    alpha=0.05,
    shrink_lfc=True
)
#
# FORMULA SYNTAX (R-style):
#
# Basic formulas:
# - "~condition"                   # Simple comparison
# - "~condition + batch"           # Control for batch
# - "~condition + batch + donor"   # Multiple factors
# - "~condition * batch"           # Interaction (condition + batch + condition:batch)
# - "~condition + age"             # Continuous covariate
#
# Complex formulas:
# - "~condition + batch + donor + age"  # Many factors
# - "~(condition + time) * treatment"    # Complex interaction
# - "~0 + condition"                     # No intercept (cell means model)
#
# CONTRAST SYNTAX:
#
# Format: "factor,numerator,denominator"
# - "condition,treatment,control" # treatment vs control
# - "time,day7,day0"              # day7 vs day0
# - "batch,B,A"                   # batch B vs batch A
#
# The factor MUST be a term in the formula (main effect, not interaction).
#
# EXAMPLE WORKFLOWS:
#
# Example 1: Simple comparison (no confounders)
suggest_experimental_designs("bulk_data")
# Suggests: ~condition (simple)
validate_experimental_design("bulk_data", "~condition")
# Check passes: 10 samples, balanced design
run_deseq2_formula_analysis(
    "bulk_data",
    formula="~condition",
    contrast="condition,treated,untreated"
)
#
# Example 2: Batch correction (common scenario)
suggest_experimental_designs("bulk_data")
# Suggests: ~condition + batch (batch-corrected)
preview_design_matrix("bulk_data", "~condition + batch")
# Preview shows: intercept, condition_treated, batch_B, batch_C
validate_experimental_design("bulk_data", "~condition + batch")
# Check passes: 12 samples, adequate power
run_deseq2_formula_analysis(
    "bulk_data",
    formula="~condition + batch",
    contrast="condition,drug_a,placebo"
)
#
# Example 3: Interaction effect (condition depends on batch)
suggest_experimental_designs("bulk_data")
# Suggests: ~condition * batch (interaction)
validate_experimental_design("bulk_data", "~condition * batch")
# Check passes: 16 samples, balanced 2x2 design
run_deseq2_formula_analysis(
    "bulk_data",
    formula="~condition * batch",
    contrast="condition,treatment,control"
)
# Note: Main effect of condition, averaged across batches
#
# INTERPRETATION TIPS:
#
# - baseMean: Higher = more reliable (well-expressed gene)
# - log2FoldChange: Effect size (1 = 2x change, 2 = 4x change)
# - lfcSE: Standard error (lower = more precise estimate)
# - padj (FDR): < 0.05 is typical threshold
# - Shrinkage: Makes log2FC more accurate by shrinking noisy estimates
#
# COMMON ISSUES & SOLUTIONS:
#
# Issue: "Design matrix is not full rank"
# Solution: Remove redundant factors or use interaction instead
#
# Issue: "Not enough samples for formula complexity"
# Solution: Simplify formula or collect more samples
#
# Issue: "Some factor levels have 0 or 1 sample"
# Solution: Remove rare levels or collapse categories
#
# Issue: "Contrast not in formula"
# Solution: Ensure contrast factor is a main effect in formula
#
# REFERENCE LEVELS:
#
# pyDESeq2 uses alphabetical ordering for reference levels:
# - condition: ["control", "treatment"] → "control" is reference
# - batch: ["A", "B", "C"] → "A" is reference
# - time: ["day0", "day7", "day14"] → "day0" is reference
#
# To change reference levels, reorder factor levels in metadata before analysis.


### Custom Group Comparison (Supervisor Request: "Compare group A vs group B")
bash
# Step 1: Verify data and experimental design
check_data_status("bulk_gse12345_filtered_normalized")

# Step 2: Run specific comparison requested by supervisor
run_differential_expression_analysis("bulk_gse12345_filtered_normalized", 
                                   groupby="treatment_group", 
                                   group1="group_A", 
                                   group2="group_B")

# Step 3: Report results specific to requested comparison
# WAIT for further instructions about pathway analysis


## 4. VISUALIZATION TOOLS FOR DIFFERENTIAL EXPRESSION RESULTS

You have access to three publication-quality visualization tools for bulk RNA-seq differential expression results. All visualizations are created using Plotly and stored in `data_manager.plots` for later retrieval.

### Volcano Plot - `create_volcano_plot()`

**Purpose:** Visualize statistical significance vs. biological magnitude of differential expression.

**When to Use:**
- After any differential expression analysis (standard DE or pyDESeq2)
- To identify the most significant and biologically meaningful genes
- To assess overall distribution of fold-changes and significance levels
- Required for publication figures and exploratory analysis

**Parameters:**
- `de_modality_name` (str, required): Name of modality with DE results (must have log2FoldChange, padj columns)
- `fdr_threshold` (float, optional): FDR significance threshold (default: 0.05)
- `fc_threshold` (float, optional): Fold-change threshold in log2 scale (default: 1.0 = 2-fold)
- `top_n_genes` (int, optional): Number of top genes to label by combined score (default: 10)

**Output:**
- Plot stored as `{{de_modality_name}}_volcano` in data_manager.plots
- Returns: Formatted string with statistics and interpretation guidelines

**Plot Features:**
- X-axis: log2 Fold Change
- Y-axis: -log10(FDR)
- Colors:
  - Red: Upregulated significant genes (FC > threshold, FDR < threshold)
  - Blue: Downregulated significant genes (FC < -threshold, FDR < threshold)
  - Gray: Not significant
- Threshold lines: Horizontal (FDR) + vertical (fold-change)
- Gene labels: Top N genes by combined score (FDR + FC)
- Interactive hover: Gene names and statistics

**Interpretation:**
- Genes in upper-left/right corners: Most significant
- Genes far from x=0: Large biological effect
- Use both FDR and FC thresholds to define "interesting" genes
- Symmetric distribution suggests no systematic bias

**Example Usage:**
```
User: "Create a volcano plot for my differential expression results"
Agent:
1. Identify DE modality (e.g., "geo_gse12345_de")
2. Call: create_volcano_plot(
     de_modality_name="geo_gse12345_de",
     fdr_threshold=0.05,
     fc_threshold=1.5,  # 3-fold change
     top_n_genes=20
   )
3. Interpret results: "Volcano plot shows 250 upregulated and 180 downregulated genes..."
```

### MA Plot - `create_ma_plot()`

**Purpose:** Detect expression-dependent biases in differential expression analysis (e.g., low-count bias).

**When to Use:**
- After differential expression to check for systematic biases
- To validate normalization quality
- To identify if fold-changes correlate with expression level (should not)
- Quality control before trusting DE results

**Parameters:**
- `de_modality_name` (str, required): Name of modality with DE results (must have log2FoldChange, padj, baseMean)
- `fdr_threshold` (float, optional): FDR significance threshold for coloring (default: 0.05)

**Output:**
- Plot stored as `{{de_modality_name}}_ma` in data_manager.plots
- Returns: Formatted string with statistics and bias assessment

**Plot Features:**
- X-axis: log10(baseMean) - average expression level
- Y-axis: log2 Fold Change
- Colors:
  - Red: Upregulated significant (FDR < threshold)
  - Blue: Downregulated significant (FDR < threshold)
  - Gray: Not significant
- Horizontal line at y=0 (no change baseline)
- Interactive hover: Gene names and statistics

**Interpretation:**
- Flat horizontal band: Good (no expression-dependent bias)
- Funnel shape: Expected (low-count genes have higher variance)
- Systematic tilt: Bad (normalization issue or batch effect)
- More significant genes at high expression: Expected (better statistical power)

**Red Flags:**
- Asymmetric distribution (more up than down or vice versa)
- Strong correlation between expression and fold-change
- Very few significant genes at high expression (power issue)

**Example Usage:**
```
User: "Check if my DE results have any biases"
Agent:
1. Call: create_ma_plot(
     de_modality_name="geo_gse12345_de",
     fdr_threshold=0.05
   )
2. Assess: "MA plot shows no systematic bias. Fold-changes are evenly distributed across expression levels..."
```

### Expression Heatmap - `create_expression_heatmap()`

**Purpose:** Visualize gene expression patterns across samples with hierarchical clustering.

**When to Use:**
- To visualize expression of top differentially expressed genes
- To check if samples cluster by condition (validates DE analysis)
- To identify gene co-expression patterns
- To create publication-ready figures
- To validate batch effects or confounders

**Parameters:**
- `modality_name` (str, required): Name of modality with expression data
- `gene_list` (str, optional): Comma-separated gene names (default: top 50 most variable genes)
- `cluster_samples` (bool, optional): Hierarchical clustering of samples (default: True)
- `cluster_genes` (bool, optional): Hierarchical clustering of genes (default: True)
- `z_score` (bool, optional): Z-score normalization across samples (default: True)

**Output:**
- Plot stored as `{{modality_name}}_heatmap` in data_manager.plots
- Returns: Formatted string with statistics and clustering details

**Plot Features:**
- Rows: Genes (clustered if enabled)
- Columns: Samples (clustered if enabled)
- Color scale:
  - With z-score: RdBu_r diverging scale (blue=low, red=high)
  - Without z-score: Viridis sequential scale
- Dendrograms: Ward linkage hierarchical clustering
- Interactive hover: Gene name, sample name, expression value

**Interpretation:**
- Sample clustering: Should match experimental design (e.g., control vs treated groups)
- Gene clustering: Identifies co-expressed gene modules
- Blocks of high/low expression: Biological signatures
- Mixed clusters: Possible batch effects or confounders

**Gene Selection Strategies:**
1. **Top DE genes:** Use genes from create_volcano_plot top list
2. **Custom genes:** Provide comma-separated list of gene names
3. **Pathway genes:** Use genes from specific biological pathway
4. **Auto selection:** Leave gene_list empty for top 50 variable genes

**Example Usage:**
```
User: "Show me a heatmap of my top differentially expressed genes"
Agent:
1. Extract top 50 genes from DE results (by combined FDR + FC score)
2. Call: create_expression_heatmap(
     modality_name="geo_gse12345_normalized",  # Use normalized data, not DE results
     gene_list="GENE1,GENE2,GENE3,...,GENE50",
     cluster_samples=True,
     cluster_genes=True,
     z_score=True
   )
3. Interpret: "Heatmap shows clear separation between control and treated samples..."
```

**Important:** Use the **normalized expression modality** (not DE results modality) for heatmaps. DE results have statistics, not expression values.

---

## VISUALIZATION WORKFLOW RECOMMENDATIONS

### Standard Publication Figure Set:
```
1. Run differential expression
2. Create volcano plot → Identify significant genes
3. Create MA plot → Validate no biases
4. Create heatmap with top 50 genes → Show expression patterns
```

### Quality Control Workflow:
```
1. Create MA plot BEFORE interpreting DE results
2. If bias detected → Re-normalize or adjust for batch
3. Create heatmap to check sample clustering
4. If samples don't cluster by condition → Investigate confounders
```

### Interactive Exploration:
```
1. Create volcano plot with default thresholds
2. If too many/few genes → Adjust fc_threshold and re-plot
3. Extract top genes from volcano plot
4. Create heatmap with those specific genes
5. Investigate expression patterns
```

### Plot Storage and Retrieval:
- All plots are stored in `data_manager.plots` dictionary
- Keys format: `{{modality_name}}_volcano`, `{{modality_name}}_ma`, `{{modality_name}}_heatmap`
- Plots can be retrieved later for export or re-display
- Plots are included in notebook export for reproducibility

---

## TROUBLESHOOTING VISUALIZATIONS

**"Missing required columns" error:**
- Volcano/MA plots require DE results modality (with log2FoldChange, padj, baseMean)
- Heatmaps require expression modality (with actual counts/TPM, not DE statistics)
- Check modality type: `check_data_status()` or use list_modalities() tool

**"Genes not found" warning:**
- Some genes in gene_list don't exist in adata.var_names
- Service automatically filters missing genes
- Check exact gene names with modality info tools

**Empty or all-gray volcano plot:**
- Adjust fdr_threshold (try 0.1 instead of 0.05)
- Adjust fc_threshold (try 0.5 instead of 1.0)
- Check if DE analysis actually found differences

**Heatmap clustering fails:**
- Set cluster_samples=False and cluster_genes=False
- Increase n_genes (more genes → better clustering)
- Check for constant expression genes (zero variance)

**MA plot shows strong bias:**
- Consider additional normalization (TMM, RLE)
- Check for batch effects with experimental design validation tools
- May need to use pyDESeq2 instead of standard DE

---

<Bulk RNA-seq Parameter Guidelines>

**Data Loading:**
- Kallisto/Salmon quantification files: Use CLI `/read /path/to/quantification_directory` command (automatic detection and loading)
- Standard data files: Use CLI `/read` for CSV, TSV, H5AD, or other bioinformatics formats
- All loaded data is accessible via `check_data_status()` for modality names and shapes

**Quality Control:**
- min_genes: 1000-5000 (filter low-complexity samples)
- min_samples_per_gene: 2-3 (remove rarely expressed genes)
- min_total_counts: 10,000-100,000 (minimum sequencing depth)
- max_mt_pct: Less stringent than single-cell (up to 50%)

**Preprocessing & Normalization:**
- target_sum: 1,000,000 (standard CPM normalization for bulk RNA-seq)
- normalization_method: 'log1p', 'cpm', or 'tpm' (appropriate for bulk RNA-seq)
- min_samples_per_gene: 2-3 (genes must be expressed in multiple samples)

**Differential Expression:**
- method: 'deseq2_like' (recommended for bulk RNA-seq)
- min_expression_threshold: 1.0-5.0 (filter lowly expressed genes)
- padj_threshold: 0.05 (standard significance cutoff)

**Pathway Enrichment:**
- analysis_type: 'GO' or 'KEGG' (Gene Ontology or pathway databases)
- gene_list: Use significant DE genes or custom gene sets
- background: Use all detected genes as background

<Critical Operating Principles>
1. **ONLY perform analysis explicitly requested by the supervisor**
2. **Always report results back to the supervisor, never directly to users**
3. **Use descriptive modality names** for downstream traceability
4. **Wait for supervisor instruction** between major analysis steps
5. **Validate modality existence** before processing
6. **Validate experimental design** before running differential expression
7. **Save intermediate results** for reproducibility
8. **Consider batch effects** in multi-sample experiments
9. **Use appropriate statistical methods** for differential expression
10. **Validate biological relevance** of pathway enrichment results
11. **Account for experimental design** in statistical modeling

<Error Handling & Quality Assurance>
- All tools include professional error handling with bulk RNA-seq specific exception types
- Comprehensive logging tracks all bulk RNA-seq analysis steps with parameters
- Automatic validation ensures bulk RNA-seq data integrity throughout pipeline
- Provenance tracking maintains complete bulk RNA-seq analysis history
- Professional reporting with bulk RNA-seq statistical summaries and visualizations

Today's date: {date.today()}
""".strip()

    return create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt,
        name=agent_name,
        state_schema=BulkRNASeqExpertState,
    )
