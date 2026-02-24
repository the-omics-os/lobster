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
    description="WGS and SNP array analysis: VCF/PLINK loading, QC (call rate, MAF, HWE), GWAS, LD pruning, kinship, variant annotation",
    factory_function="lobster.agents.genomics.genomics_expert.genomics_expert",
    handoff_tool_name="handoff_to_genomics_expert",
    handoff_tool_description="Assign genomics analysis tasks: WGS/SNP array QC, GWAS, LD pruning, kinship matrix, clumping, variant annotation. Can hand off to variant_analysis_expert for clinical interpretation.",
    supervisor_accessible=True,
    tier_requirement="free",  # All agents free — commercial value in Omics-OS Cloud
    child_agents=["variant_analysis_expert"],  # Clinical variant interpretation child
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
from lobster.core.analysis_ir import AnalysisStep, ParameterSpec
from lobster.core.data_manager_v2 import DataManagerV2
from lobster.services.analysis.gwas_service import GWASService
from lobster.services.analysis.variant_annotation_service import (
    VariantAnnotationService,
)
from lobster.services.quality.genomics_quality_service import GenomicsQualityService
from lobster.tools.knowledgebase_tools import (
    create_summarize_modality_tool,
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

            # Create provenance IR for notebook export
            ir = AnalysisStep(
                operation="cyvcf2.VCF.load",
                tool_name="load_vcf",
                description=f"Load VCF file: {n_samples} samples x {n_variants} variants",
                library="cyvcf2",
                code_template="""# Load VCF file
import cyvcf2
import anndata as ad
import numpy as np
from lobster.core.adapters.genomics.vcf_adapter import VCFAdapter

adapter = VCFAdapter(strict_validation=False)
adata = adapter.from_source(
    source={{ file_path | repr }},
    region={{ region | repr }},
    samples={{ samples | repr }},
    filter_pass={{ filter_pass }},
    max_variants={{ max_variants }},
)""",
                imports=["import cyvcf2", "import anndata as ad", "import numpy as np"],
                parameters={
                    "file_path": file_path,
                    "region": region,
                    "samples": sample_list,
                    "filter_pass": filter_pass,
                    "max_variants": max_variants,
                },
                parameter_schema={
                    "file_path": ParameterSpec(
                        param_type="str",
                        papermill_injectable=True,
                        default_value=file_path,
                        required=True,
                        description="Path to VCF file",
                    ),
                    "modality_name": ParameterSpec(
                        param_type="str",
                        papermill_injectable=True,
                        default_value=modality_name,
                        required=True,
                        description="Modality name",
                    ),
                },
                input_entities=[],
                output_entities=["adata"],
            )

            # Log operation with IR
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
                ir=ir,
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

            # Create provenance IR for notebook export
            ir = AnalysisStep(
                operation="bed_reader.open_bed",
                tool_name="load_plink",
                description=f"Load PLINK file: {n_individuals} individuals x {n_snps} SNPs",
                library="bed-reader",
                code_template="""# Load PLINK file
from lobster.core.adapters.genomics.plink_adapter import PLINKAdapter

adapter = PLINKAdapter(strict_validation=False)
adata = adapter.from_source(
    source={{ file_path | repr }},
    maf_min={{ maf_min | repr }},
)""",
                imports=[
                    "from lobster.core.adapters.genomics.plink_adapter import PLINKAdapter"
                ],
                parameters={
                    "file_path": file_path,
                    "maf_min": maf_min,
                },
                parameter_schema={
                    "file_path": ParameterSpec(
                        param_type="str",
                        papermill_injectable=True,
                        default_value=file_path,
                        required=True,
                        description="Path to PLINK .bed file",
                    ),
                    "modality_name": ParameterSpec(
                        param_type="str",
                        papermill_injectable=True,
                        default_value=modality_name,
                        required=True,
                        description="Modality name",
                    ),
                },
                input_entities=[],
                output_entities=["adata"],
            )

            # Log operation with IR
            data_manager.log_tool_usage(
                tool_name="load_plink",
                parameters={
                    "file_path": file_path,
                    "modality_name": modality_name,
                    "maf_min": maf_min,
                },
                description=f"Loaded PLINK file: {n_individuals} individuals x {n_snps} SNPs",
                ir=ir,
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
    # GWAS PIPELINE TOOLS (NEW)
    # =========================================================================

    @tool
    def ld_prune(
        modality_name: str,
        threshold: float = 0.2,
        window_size: int = 500,
        genotype_layer: str = "GT",
    ) -> str:
        """
        LD-prune variants to remove redundant SNPs in linkage disequilibrium.

        Run BEFORE PCA or GWAS to ensure variants are approximately independent.
        Removes highly correlated variants using an r-squared threshold within sliding windows.

        Args:
            modality_name: Name of QC-filtered modality
            threshold: r-squared threshold — variant pairs above this are pruned (default 0.2)
            window_size: Number of variants per window (default 500)
            genotype_layer: Layer containing genotypes (default "GT")

        Returns:
            Summary with pruning statistics and new modality name

        Example:
            ld_prune("wgs_study1_filtered", threshold=0.2, window_size=500)
        """
        try:
            # Validate modality exists
            if modality_name not in data_manager.list_modalities():
                raise ModalityNotFoundError(
                    f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}"
                )

            adata = data_manager.get_modality(modality_name)
            logger.info(
                f"LD pruning '{modality_name}': threshold={threshold}, window={window_size}"
            )

            # Run LD pruning
            adata_pruned, stats, ir = gwas_service.ld_prune_variants(
                adata=adata,
                threshold=threshold,
                window_size=window_size,
                genotype_layer=genotype_layer,
            )

            # Save as new modality
            pruned_modality_name = f"{modality_name}_ld_pruned"
            data_manager.store_modality(
                name=pruned_modality_name,
                adata=adata_pruned,
                parent_name=modality_name,
                step_summary=f"LD pruned: {stats['n_variants_after']}/{stats['n_variants_before']} variants retained",
            )

            # Log operation with IR
            data_manager.log_tool_usage(
                tool_name="ld_prune",
                parameters={
                    "modality_name": modality_name,
                    "threshold": threshold,
                    "window_size": window_size,
                    "genotype_layer": genotype_layer,
                },
                description=f"LD pruning: {stats['n_variants_after']}/{stats['n_variants_before']} variants retained (r²>{threshold})",
                ir=ir,
            )

            response = f"""LD pruning completed: '{pruned_modality_name}'

**Pruning Summary:**
- Variants before: {stats['n_variants_before']:,}
- Variants after: {stats['n_variants_after']:,}
- Variants removed: {stats['n_variants_removed']:,}
- Retention rate: {stats['n_variants_after'] / stats['n_variants_before']:.1%}

**Parameters:**
- r² threshold: {threshold}
- Window size: {window_size} variants
- Genotype layer: {genotype_layer}

**New modality created**: '{pruned_modality_name}'
**Next steps**: Use this LD-pruned data for PCA (calculate_pca) or kinship analysis (compute_kinship).
"""
            return response

        except ModalityNotFoundError as e:
            logger.error(f"Error in LD pruning: {e}")
            return f"Error: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error in LD pruning: {e}")
            return f"Unexpected error: {str(e)}"

    @tool
    def compute_kinship(
        modality_name: str,
        kinship_threshold: float = 0.125,
        genotype_layer: str = "GT",
    ) -> str:
        """
        Compute pairwise kinship matrix to detect related individuals.

        Uses VanRaden's GRM (Genomic Relationship Matrix) method. Flags pairs
        with kinship coefficient above threshold (0.125 = 3rd degree relatives).
        Related individuals should be removed before GWAS to avoid inflated statistics.

        Args:
            modality_name: Name of QC-filtered modality
            kinship_threshold: Kinship coefficient threshold for related pairs (default 0.125)
            genotype_layer: Layer containing genotypes (default "GT")

        Returns:
            Summary with kinship statistics and related pair count
        """
        try:
            # Validate modality exists
            if modality_name not in data_manager.list_modalities():
                raise ModalityNotFoundError(
                    f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}"
                )

            adata = data_manager.get_modality(modality_name)
            logger.info(
                f"Computing kinship for '{modality_name}': threshold={kinship_threshold}"
            )

            # Run kinship computation
            adata_kinship, stats, ir = gwas_service.compute_kinship(
                adata=adata,
                kinship_threshold=kinship_threshold,
                genotype_layer=genotype_layer,
            )

            # Save as new modality
            kinship_modality_name = f"{modality_name}_kinship"
            data_manager.store_modality(
                name=kinship_modality_name,
                adata=adata_kinship,
                parent_name=modality_name,
                step_summary=f"Kinship: {stats['n_related_pairs']} related pairs found",
            )

            # Log operation with IR
            data_manager.log_tool_usage(
                tool_name="compute_kinship",
                parameters={
                    "modality_name": modality_name,
                    "kinship_threshold": kinship_threshold,
                    "genotype_layer": genotype_layer,
                },
                description=f"Kinship analysis: {stats['n_related_pairs']} related pairs (threshold={kinship_threshold})",
                ir=ir,
            )

            # Format related pairs info
            related_info = ""
            if stats.get("related_pairs") and len(stats["related_pairs"]) > 0:
                related_info = "\n**Related Pairs (above threshold):**\n"
                for i, pair in enumerate(stats["related_pairs"][:10], 1):
                    related_info += f"  {i}. {pair['sample_1']} <-> {pair['sample_2']}: kinship={pair['kinship']:.4f}\n"
                if len(stats["related_pairs"]) > 10:
                    related_info += (
                        f"  ... and {len(stats['related_pairs']) - 10} more pairs\n"
                    )

            response = f"""Kinship analysis completed: '{kinship_modality_name}'

**Kinship Summary:**
- Samples: {stats['n_samples']:,}
- Sample pairs evaluated: {stats['n_pairs']:,}
- Related pairs (kinship > {kinship_threshold}): {stats['n_related_pairs']:,}
- Mean kinship: {stats['mean_kinship']:.4f}
- Max kinship: {stats['max_kinship']:.4f}
{related_info}
**Kinship matrix stored in**: adata.obsm['kinship']
**New modality created**: '{kinship_modality_name}'

**Next steps**:
- If related pairs found, consider removing one from each pair before GWAS
- Proceed to GWAS: run_gwas("{modality_name}", "phenotype")
"""
            return response

        except ModalityNotFoundError as e:
            logger.error(f"Error in kinship computation: {e}")
            return f"Error: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error in kinship computation: {e}")
            return f"Unexpected error: {str(e)}"

    @tool
    def clump_results(
        modality_name: str,
        pvalue_threshold: float = 5e-8,
        clump_kb: int = 250,
        pvalue_col: str = "gwas_pvalue",
    ) -> str:
        """
        Clump GWAS results into independent genomic loci.

        Groups significant GWAS variants into loci based on genomic proximity.
        Each clump has an index variant (lowest p-value) and member variants within
        the clumping window on the same chromosome. Run AFTER run_gwas().

        Args:
            modality_name: Name of modality with GWAS results (must have pvalue_col in var)
            pvalue_threshold: Significance threshold for variants to include (default 5e-8)
            clump_kb: Clumping window in kilobases (default 250)
            pvalue_col: Column in adata.var containing GWAS p-values (default "gwas_pvalue")

        Returns:
            Summary of clumped loci with index variants

        Example:
            clump_results("wgs_filtered_gwas", pvalue_threshold=5e-8, clump_kb=250)
        """
        try:
            # Validate modality exists
            if modality_name not in data_manager.list_modalities():
                raise ModalityNotFoundError(
                    f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}"
                )

            adata = data_manager.get_modality(modality_name)
            logger.info(
                f"Clumping GWAS results in '{modality_name}': p<{pvalue_threshold}, window={clump_kb}kb"
            )

            # Run clumping
            adata_clumped, stats, ir = gwas_service.clump_gwas_results(
                adata=adata,
                pvalue_threshold=pvalue_threshold,
                clump_kb=clump_kb,
                pvalue_col=pvalue_col,
            )

            # Save as new modality
            clumped_modality_name = f"{modality_name}_clumped"
            data_manager.store_modality(
                name=clumped_modality_name,
                adata=adata_clumped,
                parent_name=modality_name,
                step_summary=f"Clumped: {stats['n_clumps']} independent loci from {stats['n_significant_variants']} significant variants",
            )

            # Log operation with IR
            data_manager.log_tool_usage(
                tool_name="clump_results",
                parameters={
                    "modality_name": modality_name,
                    "pvalue_threshold": pvalue_threshold,
                    "clump_kb": clump_kb,
                    "pvalue_col": pvalue_col,
                },
                description=f"GWAS clumping: {stats['n_clumps']} loci from {stats['n_significant_variants']} significant variants",
                ir=ir,
            )

            # Format top clumps
            clump_info = ""
            if stats.get("clumps") and len(stats["clumps"]) > 0:
                clump_info = "\n**Top Clumped Loci (index variants):**\n"
                for i, clump in enumerate(stats["clumps"][:10], 1):
                    clump_info += (
                        f"  {i}. {clump['index_variant']}: "
                        f"p={clump['index_pvalue']:.2e}, "
                        f"{clump['n_members']} member variant(s)\n"
                    )
                if len(stats["clumps"]) > 10:
                    clump_info += f"  ... and {len(stats['clumps']) - 10} more loci\n"

            # Suggest handoff for clinical interpretation
            handoff_note = ""
            if stats["n_clumps"] > 0:
                handoff_note = """
**Clinical Interpretation**: For clinical variant interpretation (VEP consequences,
gnomAD population frequencies, ClinVar pathogenicity), consider handing off to
variant_analysis_expert."""

            response = f"""GWAS clumping completed: '{clumped_modality_name}'

**Clumping Summary:**
- Significant variants (p < {pvalue_threshold}): {stats['n_significant_variants']:,}
- Independent loci (clumps): {stats['n_clumps']:,}
- Clumping window: {clump_kb} kb
- P-value column: {pvalue_col}
{clump_info}
**Clump assignments stored in**: adata.var['clump_id'], adata.var['is_index_variant']
**New modality created**: '{clumped_modality_name}'
{handoff_note}"""
            return response

        except ModalityNotFoundError as e:
            logger.error(f"Error in clumping: {e}")
            return f"Error: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error in clumping: {e}")
            return f"Unexpected error: {str(e)}"

    # =========================================================================
    # COLLECT ALL TOOLS
    # =========================================================================

    # Shared helper tool (merges list_modalities + get_modality_info)
    summarize_modality = create_summarize_modality_tool(data_manager)

    # Core genomics tools
    genomics_tools = [
        # Data loading
        load_vcf,
        load_plink,
        # Quality control
        assess_quality,
        filter_samples,
        filter_variants,
        # GWAS pipeline
        ld_prune,
        compute_kinship,
        run_gwas,
        calculate_pca,
        clump_results,
        # Annotation
        annotate_variants,
        # Helpers
        summarize_modality,
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
