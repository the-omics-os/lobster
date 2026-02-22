"""
Genomics Expert Agent for WGS and SNP array analysis.

This agent handles genomics data loading, quality control, filtering, and
advanced analysis for whole genome sequencing (VCF) and SNP array (PLINK) data.

Features:
- Phase 1: Data loading (VCF, PLINK), QC metrics, sample/variant filtering
- Phase 2: GWAS (linear/logistic regression), PCA (population structure), variant annotation
"""

# Agent configuration for entry point discovery (must be at top, before heavy imports)
from lobster.config.agent_registry import AgentRegistryConfig

AGENT_CONFIG = AgentRegistryConfig(
    name="genomics_expert",
    display_name="Genomics Expert",
    description="WGS and SNP array analysis: VCF/PLINK loading, QC (call rate, MAF, HWE), GWAS, variant annotation",
    factory_function="lobster.agents.genomics.genomics_expert.genomics_expert",
    handoff_tool_name="handoff_to_genomics_expert",
    handoff_tool_description="Assign genomics analysis tasks: WGS/SNP array QC, GWAS, variant annotation, genotype filtering",
    supervisor_accessible=True,
    tier_requirement="free",  # All agents free — commercial value in Omics-OS Cloud
)

# === Heavy imports below ===
from pathlib import Path
from typing import Optional

from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from lobster.agents.genomics.prompts import create_genomics_expert_prompt
from lobster.config.llm_factory import create_llm
from lobster.config.settings import get_settings
from lobster.core.adapters.genomics.plink_adapter import PLINKAdapter
from lobster.core.adapters.genomics.vcf_adapter import VCFAdapter
from lobster.core.data_manager_v2 import DataManagerV2
from lobster.services.quality.genomics_quality_service import GenomicsQualityService
from lobster.services.analysis.gwas_service import GWASService
from lobster.services.analysis.variant_annotation_service import (
    VariantAnnotationService,
)
from lobster.tools.knowledgebase_tools import (
    create_variant_consequence_tool,
    create_sequence_retrieval_tool,
)
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


class GenomicsAgentError(Exception):
    """Base exception for genomics agent operations."""

    pass


class ModalityNotFoundError(GenomicsAgentError):
    """Raised when requested modality doesn't exist."""

    pass


def genomics_expert(
    data_manager: DataManagerV2,
    callback_handler=None,
    agent_name: str = "genomics_expert",
    delegation_tools: list = None,
    workspace_path: Optional[Path] = None,
    provider_override: Optional[str] = None,
    model_override: Optional[str] = None,
):
    """
    Factory function for genomics expert agent.

    This agent handles WGS (VCF) and SNP array (PLINK) data loading,
    quality control, and filtering.

    Args:
        data_manager: DataManagerV2 instance for modality management
        callback_handler: Optional callback handler for LLM interactions
        agent_name: Name identifier for the agent instance
        delegation_tools: List of delegation tools for sub-agents (not used in Phase 1)
        workspace_path: Optional workspace path for LLM operations

    Returns:
        Configured ReAct agent with genomics analysis capabilities
    """
    settings = get_settings()
    model_params = settings.get_agent_llm_params("genomics_expert")
    llm = create_llm(
        "genomics_expert",
        model_params,
        provider_override=provider_override,
        model_override=model_override,
        workspace_path=workspace_path,
    )

    # Normalize callbacks to a flat list
    if callback_handler and hasattr(llm, "with_config"):
        callbacks = (
            callback_handler
            if isinstance(callback_handler, list)
            else [callback_handler]
        )
        llm = llm.with_config(callbacks=callbacks)

    # Initialize services
    qc_service = GenomicsQualityService()
    gwas_service = GWASService()
    annotation_service = VariantAnnotationService()

    # =========================================================================
    # DATA LOADING TOOLS
    # =========================================================================

    @tool
    def load_vcf(
        file_path: str,
        modality_name: str,
        region: Optional[str] = None,
        samples: Optional[str] = None,
        filter_pass: bool = True,
        max_variants: Optional[int] = 100000,
    ) -> str:
        """
        Load VCF file for whole genome sequencing data.

        Args:
            file_path: Path to VCF file (.vcf, .vcf.gz, .bcf)
            modality_name: Name for the loaded modality (e.g., "wgs_study1")
            region: Optional genomic region (e.g., "chr1:1000-2000")
            samples: Optional comma-separated sample IDs (e.g., "Sample1,Sample2")
            filter_pass: Only load PASS variants (default True)
            max_variants: Maximum number of variants to load (default 100,000; None = unlimited)

        Returns:
            str: Summary of loaded data with dimensions and QC statistics

        Example:
            load_vcf("/data/ukbb.vcf.gz", "ukbb_chr1", filter_pass=True, max_variants=50000)
        """
        try:
            logger.info(f"Loading VCF file: {file_path} as '{modality_name}'")

            # Parse samples if provided
            sample_list = None
            if samples:
                sample_list = [s.strip() for s in samples.split(",")]

            # Initialize VCF adapter
            adapter = VCFAdapter(strict_validation=False)

            # Load VCF with optional filtering
            adata = adapter.from_source(
                source=file_path,
                region=region,
                samples=sample_list,
                filter_pass=filter_pass,
                max_variants=max_variants,
            )

            # Store in data manager
            n_samples = adata.n_obs
            n_variants = adata.n_vars
            data_manager.store_modality(
                name=modality_name,
                adata=adata,
                step_summary=f"Loaded VCF: {n_samples} samples x {n_variants} variants",
            )

            # Get basic stats
            has_gt = "GT" in adata.layers

            # Log operation (no IR for loading in Phase 1, will add in Phase 2)
            data_manager.log_tool_usage(
                tool_name="load_vcf",
                parameters={
                    "file_path": file_path,
                    "modality_name": modality_name,
                    "region": region,
                    "samples": sample_list,
                    "filter_pass": filter_pass,
                    "max_variants": max_variants,
                },
                description=f"Loaded VCF file: {n_samples} samples x {n_variants} variants",
            )

            # Check if variants were truncated
            truncation_note = ""
            if max_variants is not None:
                truncation_note = f"\n- Max variants limit: {max_variants:,} (variants may be truncated)"

            response = f"""Successfully loaded VCF file: '{modality_name}'

**Data Summary:**
- Samples: {n_samples:,}
- Variants: {n_variants:,}{truncation_note}
- Genotype layer: {"Yes" if has_gt else "No"}
- Source file: {file_path}
- Region filter: {region if region else "None (whole genome)"}
- PASS filter: {"Yes" if filter_pass else "No"}

**Sample metadata (adata.obs)**:
- Columns: {list(adata.obs.columns)}

**Variant metadata (adata.var)**:
- Columns: {list(adata.var.columns)}

**Next steps**: Run assess_quality("{modality_name}") to calculate QC metrics
"""
            return response

        except FileNotFoundError as e:
            logger.error(f"VCF file not found: {e}")
            return f"Error: VCF file not found at {file_path}. Please check the path."
        except Exception as e:
            logger.error(f"Error loading VCF: {e}")
            return f"Error loading VCF file: {str(e)}"

    @tool
    def load_plink(
        file_path: str,
        modality_name: str,
        maf_min: Optional[float] = None,
    ) -> str:
        """
        Load PLINK files for SNP array data.

        Args:
            file_path: Path to .bed file or prefix (e.g., "/data/study.bed" or "/data/study")
            modality_name: Name for the loaded modality (e.g., "gwas_diabetes")
            maf_min: Optional minimum MAF threshold for pre-filtering (e.g., 0.01)

        Returns:
            str: Summary of loaded data with dimensions and QC statistics

        Example:
            load_plink("/data/gwas.bed", "gwas_diabetes", maf_min=0.01)
        """
        try:
            logger.info(f"Loading PLINK file: {file_path} as '{modality_name}'")

            # Initialize PLINK adapter
            adapter = PLINKAdapter(strict_validation=False)

            # Load PLINK with optional MAF filtering
            adata = adapter.from_source(
                source=file_path,
                maf_min=maf_min,
            )

            # Store in data manager
            n_individuals = adata.n_obs
            n_snps = adata.n_vars
            data_manager.store_modality(
                name=modality_name,
                adata=adata,
                step_summary=f"Loaded PLINK: {n_individuals} individuals x {n_snps} SNPs",
            )

            # Get basic stats
            has_gt = "GT" in adata.layers

            # Calculate mean MAF if available
            mean_maf = None
            if "maf" in adata.var.columns:
                mean_maf = adata.var["maf"].mean()

            # Log operation
            data_manager.log_tool_usage(
                tool_name="load_plink",
                parameters={
                    "file_path": file_path,
                    "modality_name": modality_name,
                    "maf_min": maf_min,
                },
                description=f"Loaded PLINK file: {n_individuals} individuals x {n_snps} SNPs",
            )

            response = f"""Successfully loaded PLINK file: '{modality_name}'

**Data Summary:**
- Individuals: {n_individuals:,}
- SNPs: {n_snps:,}
- Genotype layer: {"Yes" if has_gt else "No"}
- Source file: {file_path}
- MAF filter: {f">= {maf_min}" if maf_min else "None"}
- Mean MAF: {f"{mean_maf:.4f}" if mean_maf else "Not calculated"}

**Individual metadata (adata.obs)**:
- Columns: {list(adata.obs.columns)}

**SNP metadata (adata.var)**:
- Columns: {list(adata.var.columns)}

**Next steps**: Run assess_quality("{modality_name}") to calculate QC metrics
"""
            return response

        except FileNotFoundError as e:
            logger.error(f"PLINK files not found: {e}")
            return f"Error: PLINK files not found at {file_path}. Ensure .bed, .bim, and .fam files exist."
        except Exception as e:
            logger.error(f"Error loading PLINK: {e}")
            return f"Error loading PLINK file: {str(e)}"

    # =========================================================================
    # QUALITY CONTROL TOOLS
    # =========================================================================

    @tool
    def assess_quality(
        modality_name: str,
        min_call_rate: float = 0.95,
        min_maf: float = 0.01,
        hwe_pvalue: float = 1e-10,
    ) -> str:
        """
        Assess quality control metrics for genomics data.

        Calculates per-sample and per-variant QC metrics including:
        - Call rate (proportion of non-missing genotypes)
        - Minor allele frequency (MAF)
        - Hardy-Weinberg equilibrium (HWE) p-value
        - Heterozygosity rate and z-scores

        Args:
            modality_name: Name of modality to assess
            min_call_rate: Minimum call rate threshold (default: 0.95)
            min_maf: Minimum MAF threshold (default: 0.01)
            hwe_pvalue: Minimum HWE p-value (default: 1e-10 for WGS, use 1e-6 for SNP arrays)

        Returns:
            str: Detailed QC assessment report

        Example:
            assess_quality("wgs_study1", min_call_rate=0.95, min_maf=0.01, hwe_pvalue=1e-10)
        """
        try:
            # Validate modality exists
            if modality_name not in data_manager.list_modalities():
                raise ModalityNotFoundError(
                    f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}"
                )

            # Get modality
            adata = data_manager.get_modality(modality_name)
            logger.info(
                f"Assessing quality for '{modality_name}': {adata.shape[0]} samples x {adata.shape[1]} variants"
            )

            # Run QC assessment
            adata_qc, stats, ir = qc_service.assess_quality(
                adata=adata,
                min_call_rate=min_call_rate,
                min_maf=min_maf,
                hwe_pvalue=hwe_pvalue,
            )

            # Save as new modality
            qc_modality_name = f"{modality_name}_qc"
            data_manager.store_modality(
                name=qc_modality_name,
                adata=adata_qc,
                parent_name=modality_name,
                step_summary=f"QC assessed: {stats['n_variants_pass_qc']}/{stats['n_variants']} variants pass",
            )

            # Log operation with IR
            data_manager.log_tool_usage(
                tool_name="assess_quality",
                parameters={
                    "modality_name": modality_name,
                    "min_call_rate": min_call_rate,
                    "min_maf": min_maf,
                    "hwe_pvalue": hwe_pvalue,
                },
                description=f"Quality assessment: {stats['n_variants_pass_qc']}/{stats['n_variants']} variants pass QC",
                ir=ir,
            )

            # Format response
            response = f"""Quality assessment completed: '{qc_modality_name}'

**Overall Statistics:**
- Samples: {stats["n_samples"]:,}
- Variants: {stats["n_variants"]:,}
- Variants passing QC: {stats["n_variants_pass_qc"]:,} ({stats["variants_pass_pct"]:.1f}%)
- Variants failing QC: {stats["variants_fail_qc"]:,}

**Sample-Level Metrics:**
- Mean call rate: {stats["sample_metrics"]["mean_call_rate"]:.4f}
- Median call rate: {stats["sample_metrics"]["median_call_rate"]:.4f}
- Mean heterozygosity: {stats["sample_metrics"]["mean_heterozygosity"]:.4f}
- Median heterozygosity: {stats["sample_metrics"]["median_heterozygosity"]:.4f}

**Variant-Level Metrics:**
- Mean call rate: {stats["variant_metrics"]["mean_call_rate"]:.4f}
- Median call rate: {stats["variant_metrics"]["median_call_rate"]:.4f}
- Mean MAF: {stats["variant_metrics"]["mean_maf"]:.4f}
- Median MAF: {stats["variant_metrics"]["median_maf"]:.4f}

**QC Failures:**
- Low call rate: {stats["variant_metrics"]["n_variants_low_call_rate"]:,} variants
- Low MAF: {stats["variant_metrics"]["n_variants_low_maf"]:,} variants
- HWE failure: {stats["variant_metrics"]["n_variants_hwe_fail"]:,} variants

**QC Thresholds Used:**
- Min call rate: {stats["min_call_rate"]}
- Min MAF: {stats["min_maf"]}
- HWE p-value: {stats["hwe_pvalue"]}

**New modality created**: '{qc_modality_name}'
**Next steps**:
1. Filter samples: filter_samples("{qc_modality_name}")
2. Then filter variants: filter_variants("{qc_modality_name}_samples_filtered")
"""
            return response

        except ModalityNotFoundError as e:
            logger.error(f"Error in quality assessment: {e}")
            return f"Error: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error in quality assessment: {e}")
            return f"Unexpected error: {str(e)}"

    @tool
    def filter_samples(
        modality_name: str,
        min_call_rate: float = 0.95,
        het_sd_threshold: float = 3.0,
    ) -> str:
        """
        Filter samples based on QC metrics.

        Removes samples with:
        - Low call rate (< min_call_rate)
        - Extreme heterozygosity (|het_z_score| > het_sd_threshold)

        Args:
            modality_name: Name of modality with QC metrics (from assess_quality)
            min_call_rate: Minimum call rate threshold (default: 0.95)
            het_sd_threshold: Heterozygosity z-score threshold in SD (default: 3.0)

        Returns:
            str: Filtering summary with removal statistics

        Example:
            filter_samples("wgs_study1_qc", min_call_rate=0.95, het_sd_threshold=3.0)
        """
        try:
            # Validate modality exists
            if modality_name not in data_manager.list_modalities():
                raise ModalityNotFoundError(
                    f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}"
                )

            # Get modality
            adata = data_manager.get_modality(modality_name)
            logger.info(f"Filtering samples in '{modality_name}'")

            # Run sample filtering
            adata_filtered, stats, ir = qc_service.filter_samples(
                adata=adata,
                min_call_rate=min_call_rate,
                het_sd_threshold=het_sd_threshold,
            )

            # Save as new modality
            filtered_modality_name = f"{modality_name}_samples_filtered"
            data_manager.store_modality(
                name=filtered_modality_name,
                adata=adata_filtered,
                parent_name=modality_name,
                step_summary=f"Filtered samples: {stats['samples_after']}/{stats['samples_before']} retained",
            )

            # Log operation with IR
            data_manager.log_tool_usage(
                tool_name="filter_samples",
                parameters={
                    "modality_name": modality_name,
                    "min_call_rate": min_call_rate,
                    "het_sd_threshold": het_sd_threshold,
                },
                description=f"Sample filtering: {stats['samples_after']}/{stats['samples_before']} samples retained",
                ir=ir,
            )

            # Format response
            response = f"""Sample filtering completed: '{filtered_modality_name}'

**Filtering Results:**
- Samples before: {stats["samples_before"]:,}
- Samples after: {stats["samples_after"]:,}
- Samples removed: {stats["samples_removed"]:,}
- Retention rate: {stats["samples_retained_pct"]:.1f}%

**Removal Reasons:**
- Low call rate (< {stats["min_call_rate"]}): {stats["removal_reasons"]["low_call_rate"]} samples
- Extreme heterozygosity (|z| > {stats["het_sd_threshold"]}): {stats["removal_reasons"]["extreme_heterozygosity"]} samples

**Filtering Thresholds:**
- Min call rate: {stats["min_call_rate"]}
- Het z-score threshold: +/-{stats["het_sd_threshold"]} SD

**New modality created**: '{filtered_modality_name}'
**Next steps**: Filter variants with filter_variants("{filtered_modality_name}")
"""
            return response

        except ModalityNotFoundError as e:
            logger.error(f"Error in sample filtering: {e}")
            return f"Error: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error in sample filtering: {e}")
            return f"Unexpected error: {str(e)}"

    @tool
    def filter_variants(
        modality_name: str,
        min_call_rate: float = 0.99,
        min_maf: float = 0.01,
        min_hwe_p: float = 1e-10,
    ) -> str:
        """
        Filter variants based on QC metrics.

        Removes variants with:
        - Low call rate (< min_call_rate)
        - Low minor allele frequency (< min_maf)
        - Hardy-Weinberg disequilibrium (hwe_p < min_hwe_p)

        Args:
            modality_name: Name of modality with QC metrics
            min_call_rate: Minimum call rate threshold (default: 0.99)
            min_maf: Minimum MAF threshold (default: 0.01)
            min_hwe_p: Minimum HWE p-value (default: 1e-10 for WGS, 1e-6 for SNP arrays)

        Returns:
            str: Filtering summary with removal statistics

        Example:
            filter_variants("wgs_study1_qc_samples_filtered", min_call_rate=0.99, min_maf=0.01)
        """
        try:
            # Validate modality exists
            if modality_name not in data_manager.list_modalities():
                raise ModalityNotFoundError(
                    f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}"
                )

            # Get modality
            adata = data_manager.get_modality(modality_name)
            logger.info(f"Filtering variants in '{modality_name}'")

            # Run variant filtering
            adata_filtered, stats, ir = qc_service.filter_variants(
                adata=adata,
                min_call_rate=min_call_rate,
                min_maf=min_maf,
                min_hwe_p=min_hwe_p,
            )

            # Save as new modality
            filtered_modality_name = f"{modality_name}_variants_filtered"
            data_manager.store_modality(
                name=filtered_modality_name,
                adata=adata_filtered,
                parent_name=modality_name,
                step_summary=f"Filtered variants: {stats['variants_after']}/{stats['variants_before']} retained",
            )

            # Log operation with IR
            data_manager.log_tool_usage(
                tool_name="filter_variants",
                parameters={
                    "modality_name": modality_name,
                    "min_call_rate": min_call_rate,
                    "min_maf": min_maf,
                    "min_hwe_p": min_hwe_p,
                },
                description=f"Variant filtering: {stats['variants_after']}/{stats['variants_before']} variants retained",
                ir=ir,
            )

            # Format response
            response = f"""Variant filtering completed: '{filtered_modality_name}'

**Filtering Results:**
- Variants before: {stats["variants_before"]:,}
- Variants after: {stats["variants_after"]:,}
- Variants removed: {stats["variants_removed"]:,}
- Retention rate: {stats["variants_retained_pct"]:.1f}%

**Removal Reasons:**
- Low call rate (< {stats["min_call_rate"]}): {stats["removal_reasons"]["low_call_rate"]} variants
- Low MAF (< {stats["min_maf"]}): {stats["removal_reasons"]["low_maf"]} variants
- HWE failure (p < {stats["min_hwe_p"]}): {stats["removal_reasons"]["hwe_fail"]} variants

**Filtering Thresholds:**
- Min call rate: {stats["min_call_rate"]}
- Min MAF: {stats["min_maf"]}
- Min HWE p-value: {stats["min_hwe_p"]}

**New modality created**: '{filtered_modality_name}'

**Quality-controlled genomics data is ready!**
This dataset can now be used for downstream analysis (GWAS, annotation - coming in Phase 2).
"""
            return response

        except ModalityNotFoundError as e:
            logger.error(f"Error in variant filtering: {e}")
            return f"Error: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error in variant filtering: {e}")
            return f"Unexpected error: {str(e)}"

    # =========================================================================
    # GWAS & ADVANCED ANALYSIS TOOLS (Phase 2)
    # =========================================================================

    @tool
    def run_gwas(
        modality_name: str,
        phenotype: str,
        covariates: Optional[str] = None,
        model: str = "linear",
        pvalue_threshold: float = 5e-8,
    ) -> str:
        """
        Run genome-wide association study (GWAS).

        Tests association between genotypes and a phenotype using linear or logistic regression.
        Requires QC-filtered data with phenotype and covariates in adata.obs.

        Args:
            modality_name: Name of QC-filtered modality
            phenotype: Column name in adata.obs containing phenotype values
            covariates: Comma-separated covariate column names (e.g., "age,sex,PC1,PC2")
            model: "linear" (continuous phenotype) or "logistic" (binary phenotype)
            pvalue_threshold: Significance threshold (default: 5e-8 genome-wide)

        Returns:
            str: GWAS summary with significant hits and Lambda GC

        Example:
            run_gwas("wgs_study1_qc_filtered", "height", covariates="age,sex", model="linear")
        """
        try:
            # Validate modality exists
            if modality_name not in data_manager.list_modalities():
                raise ModalityNotFoundError(
                    f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}"
                )

            # Get modality
            adata = data_manager.get_modality(modality_name)
            logger.info(
                f"Running GWAS on '{modality_name}': phenotype={phenotype}, model={model}"
            )

            # Parse covariates
            covariate_list = None
            if covariates:
                covariate_list = [c.strip() for c in covariates.split(",")]

            # Run GWAS
            adata_gwas, stats, ir = gwas_service.run_gwas(
                adata=adata,
                phenotype=phenotype,
                covariates=covariate_list,
                model=model,
                pvalue_threshold=pvalue_threshold,
            )

            # Save as new modality
            gwas_modality_name = f"{modality_name}_gwas"
            data_manager.store_modality(
                name=gwas_modality_name,
                adata=adata_gwas,
                parent_name=modality_name,
                step_summary=f"GWAS: {stats.get('n_significant', 0)} significant hits",
            )

            # Log operation with IR
            data_manager.log_tool_usage(
                tool_name="run_gwas",
                parameters={
                    "modality_name": modality_name,
                    "phenotype": phenotype,
                    "covariates": covariate_list,
                    "model": model,
                    "pvalue_threshold": pvalue_threshold,
                },
                description=f"GWAS: {stats['n_variants_significant']}/{stats['n_variants_tested']} significant variants",
                ir=ir,
            )

            # Format response
            response = f"""GWAS completed: '{gwas_modality_name}'

**GWAS Summary:**
- Phenotype: {stats["phenotype"]}
- Model: {stats["model"]}
- Covariates: {", ".join(stats["covariates"]) if stats["covariates"] else "None"}
- Variants tested: {stats["n_variants_tested"]:,}
- Significant variants (p < {stats["pvalue_threshold"]}): {stats["n_variants_significant"]:,}

**Genomic Inflation:**
- Lambda GC: {stats["lambda_gc"]:.3f}
- Interpretation: {stats["lambda_gc_interpretation"]}"""

            # Add warning if high Lambda GC
            if stats["lambda_gc"] > 1.1:
                response += f"""

Warning: **Population Stratification Detected**
Lambda GC > 1.1 suggests uncorrected population structure.
**Recommendation**: Run calculate_pca("{modality_name}") and re-run GWAS with PC1-PC10 as covariates."""

            # Add top hits if any
            if stats["n_variants_significant"] > 0 and stats.get("top_variants"):
                response += "\n\n**Top Significant Variants:**\n"
                for i, var in enumerate(stats["top_variants"][:5], 1):
                    response += f"\n{i}. {var['variant_id']}: beta={var['beta']:.4f}, p={var['pvalue']:.2e}"

            response += f"\n\n**New modality created**: '{gwas_modality_name}'"
            return response

        except ModalityNotFoundError as e:
            logger.error(f"Error in GWAS: {e}")
            return f"Error: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error in GWAS: {e}")
            return f"Unexpected error: {str(e)}"

    @tool
    def calculate_pca(
        modality_name: str,
        n_components: int = 10,
        ld_prune: bool = False,
    ) -> str:
        """
        Calculate principal components for population structure analysis.

        PCA identifies population stratification in genetic data. Use PC1-PC10 as
        covariates in GWAS to correct for population structure.

        Note: LD pruning is available but currently disabled by default due to sgkit
        configuration complexity. PCA without LD pruning is effective for detecting
        broad population structure (ancestry-level stratification).

        Args:
            modality_name: Name of QC-filtered modality
            n_components: Number of principal components to compute (default: 10)
            ld_prune: Whether to perform LD pruning before PCA (default: False)

        Returns:
            str: PCA summary with variance explained

        Example:
            calculate_pca("wgs_study1_qc_filtered", n_components=10, ld_prune=False)
        """
        try:
            # Validate modality exists
            if modality_name not in data_manager.list_modalities():
                raise ModalityNotFoundError(
                    f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}"
                )

            # Get modality
            adata = data_manager.get_modality(modality_name)
            logger.info(
                f"Calculating PCA on '{modality_name}': n_components={n_components}"
            )

            # Run PCA
            adata_pca, stats, ir = gwas_service.calculate_pca(
                adata=adata,
                n_components=n_components,
                ld_prune=ld_prune,
            )

            # Save as new modality
            pca_modality_name = f"{modality_name}_pca"
            data_manager.store_modality(
                name=pca_modality_name,
                adata=adata_pca,
                parent_name=modality_name,
                step_summary=f"PCA: {n_components} components computed",
            )

            # Log operation with IR
            data_manager.log_tool_usage(
                tool_name="calculate_pca",
                parameters={
                    "modality_name": modality_name,
                    "n_components": n_components,
                    "ld_prune": ld_prune,
                },
                description=f"PCA: {n_components} components, PC1 explains {stats['variance_explained_pc1']:.1%}",
                ir=ir,
            )

            # Format response
            response = f"""PCA completed: '{pca_modality_name}'

**PCA Summary:**
- Components computed: {stats["n_components"]}
- Samples: {stats["n_samples"]:,}
- Variants used: {stats["n_variants_used"]:,}"""

            if stats["ld_pruned"]:
                response += (
                    f"\n- Variants before LD pruning: {stats['n_variants_original']:,}"
                )
            else:
                response += "\n- LD pruning: Not performed"

            response += f"""

**Variance Explained:**
- PC1: {stats["variance_explained_pc1"]:.2%}
- Top 5 PCs: {stats["variance_explained_top5"]:.2%}
- All {n_components} PCs: {sum(stats["variance_explained_per_pc"]):.2%}

**Next Steps:**
1. Re-run GWAS with PCs as covariates:
   run_gwas("{modality_name}", "phenotype", covariates="age,sex,PC1,PC2,PC3,PC4,PC5")
2. This corrects for population stratification (reduces Lambda GC)

**New modality created**: '{pca_modality_name}'
**PCA results stored in**: adata.obsm['X_pca']"""

            return response

        except ModalityNotFoundError as e:
            logger.error(f"Error in PCA: {e}")
            return f"Error: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error in PCA: {e}")
            return f"Unexpected error: {str(e)}"

    @tool
    def annotate_variants(
        modality_name: str,
        annotation_source: str = "ensembl_vep",
        genome_build: str = "GRCh38",
    ) -> str:
        """
        Annotate variants with gene names and functional consequences.

        Adds gene annotations (gene_symbol, gene_id, consequence, biotype) to adata.var.
        Useful for interpreting GWAS results.

        Args:
            modality_name: Name of modality with GWAS results
            annotation_source: "ensembl_vep" (Ensembl VEP REST API) or "genebe" (if installed)
            genome_build: "GRCh38" or "GRCh37"

        Returns:
            str: Annotation summary with gene coverage

        Example:
            annotate_variants("wgs_study1_qc_filtered_gwas", annotation_source="ensembl")
        """
        try:
            # Validate modality exists
            if modality_name not in data_manager.list_modalities():
                raise ModalityNotFoundError(
                    f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}"
                )

            # Get modality
            adata = data_manager.get_modality(modality_name)
            logger.info(
                f"Annotating variants in '{modality_name}': source={annotation_source}"
            )

            # Run annotation
            adata_annotated, stats, ir = annotation_service.annotate_variants(
                adata=adata,
                annotation_source=annotation_source,
                genome_build=genome_build,
            )

            # Save as new modality
            annotated_modality_name = f"{modality_name}_annotated"
            data_manager.store_modality(
                name=annotated_modality_name,
                adata=adata_annotated,
                parent_name=modality_name,
                step_summary=f"Annotated: {stats['n_variants_annotated']}/{stats['n_variants']} variants",
            )

            # Log operation with IR
            data_manager.log_tool_usage(
                tool_name="annotate_variants",
                parameters={
                    "modality_name": modality_name,
                    "annotation_source": annotation_source,
                    "genome_build": genome_build,
                },
                description=f"Annotation: {stats['n_variants_annotated']}/{stats['n_variants']} variants annotated",
                ir=ir,
            )

            # Format response
            response = f"""Variant annotation completed: '{annotated_modality_name}'

**Annotation Summary:**
- Annotation source: {stats["annotation_source"]}
- Genome build: {stats["genome_build"]}
- Variants: {stats["n_variants"]:,}
- Variants annotated: {stats["n_variants_annotated"]:,} ({stats["n_variants_annotated"] / stats["n_variants"]:.1%})

**Annotations Added to adata.var:**
- gene_symbol: Gene name (e.g., BRCA1)
- gene_id: Ensembl ID (e.g., ENSG00000012048)
- consequence: Variant effect (e.g., missense_variant)
- biotype: Gene type (e.g., protein_coding)

**New modality created**: '{annotated_modality_name}'

**Next Steps**: Filter significant variants by gene of interest or functional consequence."""

            return response

        except ModalityNotFoundError as e:
            logger.error(f"Error in annotation: {e}")
            return f"Error: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error in annotation: {e}")
            return f"Unexpected error: {str(e)}"

    # =========================================================================
    # HELPER TOOLS
    # =========================================================================

    @tool
    def list_modalities() -> str:
        """
        List all loaded genomics modalities.

        Returns:
            str: List of modality names with data types
        """
        try:
            all_modalities = data_manager.list_modalities()

            if not all_modalities:
                return "No modalities loaded yet. Use load_vcf() or load_plink() to load data."

            # Filter for genomics modalities
            genomics_modalities = []
            for mod_name in all_modalities:
                adata = data_manager.get_modality(mod_name)
                if adata.uns.get("data_type") == "genomics":
                    modality_type = adata.uns.get("modality", "unknown")
                    n_obs = adata.n_obs
                    n_vars = adata.n_vars
                    genomics_modalities.append(
                        f"  - {mod_name} ({modality_type}): {n_obs:,} samples x {n_vars:,} variants"
                    )

            if not genomics_modalities:
                return f"No genomics modalities found. Total modalities: {len(all_modalities)}"

            response = f"""**Loaded Genomics Modalities** ({len(genomics_modalities)} total):

{chr(10).join(genomics_modalities)}

Use get_modality_info("modality_name") for detailed information.
"""
            return response

        except Exception as e:
            logger.error(f"Error listing modalities: {e}")
            return f"Error listing modalities: {str(e)}"

    @tool
    def get_modality_info(modality_name: str) -> str:
        """
        Get detailed information about a genomics modality.

        Args:
            modality_name: Name of modality to inspect

        Returns:
            str: Detailed modality information including dimensions, data type, and QC status
        """
        try:
            # Validate modality exists
            if modality_name not in data_manager.list_modalities():
                return f"Modality '{modality_name}' not found. Use list_modalities() to see available modalities."

            # Get modality
            adata = data_manager.get_modality(modality_name)

            # Extract metadata
            n_obs = adata.n_obs
            n_vars = adata.n_vars
            data_type = adata.uns.get("data_type", "unknown")
            modality = adata.uns.get("modality", "unknown")
            source_file = adata.uns.get("source_file", "N/A")

            # Check for QC columns
            has_qc = "call_rate" in adata.obs.columns and "qc_pass" in adata.var.columns

            # Format response
            response = f"""**Modality Information: '{modality_name}'**

**Basic Info:**
- Data type: {data_type}
- Modality: {modality}
- Dimensions: {n_obs:,} samples x {n_vars:,} variants
- Source file: {source_file}

**Sample Metadata (adata.obs):**
- Columns: {list(adata.obs.columns)}
- Has QC metrics: {"Yes" if has_qc else "No"}

**Variant Metadata (adata.var):**
- Columns: {list(adata.var.columns)}
- Has QC pass flag: {"Yes" if "qc_pass" in adata.var.columns else "No"}

**Layers:**
- Available: {list(adata.layers.keys()) if adata.layers else "None"}
"""
            # Add QC statistics if available
            if has_qc:
                mean_sample_call_rate = adata.obs["call_rate"].mean()
                mean_variant_call_rate = adata.var["call_rate"].mean()
                n_variants_pass = (
                    adata.var["qc_pass"].sum() if "qc_pass" in adata.var.columns else 0
                )

                response += f"""
**QC Statistics:**
- Mean sample call rate: {mean_sample_call_rate:.4f}
- Mean variant call rate: {mean_variant_call_rate:.4f}
- Variants passing QC: {n_variants_pass:,}/{n_vars:,}
"""

            return response

        except Exception as e:
            logger.error(f"Error getting modality info: {e}")
            return f"Error: {str(e)}"

    # =========================================================================
    # COLLECT ALL TOOLS
    # =========================================================================

    # Knowledgebase tools (VEP + sequence retrieval via Ensembl)
    predict_variant_consequences = create_variant_consequence_tool(data_manager)
    get_ensembl_sequence = create_sequence_retrieval_tool(data_manager)

    # Core genomics tools
    genomics_tools = [
        # Phase 1: Data loading & QC
        load_vcf,
        load_plink,
        assess_quality,
        filter_samples,
        filter_variants,
        # Phase 2: GWAS & advanced analysis
        run_gwas,
        calculate_pca,
        annotate_variants,
        # Knowledgebase: variant consequences & sequences
        predict_variant_consequences,
        get_ensembl_sequence,
        # Helper tools
        list_modalities,
        get_modality_info,
    ]

    # Add delegation tools if provided (not used in Phase 1)
    tools = genomics_tools
    if delegation_tools:
        tools = tools + delegation_tools

    # Create system prompt
    system_prompt = create_genomics_expert_prompt()

    return create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt,
        name=agent_name,
    )
