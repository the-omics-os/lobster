"""
Bulk RNA-seq Expert Agent for specialized bulk RNA-seq analysis.

This agent focuses exclusively on bulk RNA-seq analysis using the modular DataManagerV2
system with proper modality handling and schema enforcement.
"""

from datetime import date
from typing import List

import pandas as pd
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from lobster.agents.state import BulkRNASeqExpertState
from lobster.config.llm_factory import create_llm
from lobster.config.settings import get_settings
from lobster.core.data_manager_v2 import DataManagerV2
from lobster.tools.bulk_rnaseq_service import BulkRNASeqError, BulkRNASeqService
from lobster.tools.preprocessing_service import PreprocessingError, PreprocessingService
from lobster.tools.quality_service import QualityError, QualityService
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
            from lobster.tools.differential_formula_service import (
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
            from lobster.tools.differential_formula_service import (
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
            from lobster.tools.differential_formula_service import (
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
