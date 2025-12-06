"""
Unified Proteomics Expert Agent for mass spectrometry and affinity proteomics analysis.

This agent serves as the main orchestrator for proteomics analysis, supporting:
- Mass spectrometry (DDA/DIA) with MNAR missing value patterns, peptide mapping
- Affinity platforms (Olink, SomaScan) with plate effects, antibody validation

The agent auto-detects platform type and applies appropriate defaults.
"""

from datetime import date
from typing import List, Optional, Union

import numpy as np
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from lobster.agents.proteomics.platform_config import (
    PLATFORM_CONFIGS,
    PlatformConfig,
    detect_platform_type,
    get_platform_config,
)
from lobster.agents.proteomics.state import ProteomicsExpertState
from lobster.config.llm_factory import create_llm
from lobster.config.settings import get_settings
from lobster.core.analysis_ir import AnalysisStep, ParameterSpec
from lobster.core.data_manager_v2 import DataManagerV2
from lobster.services.analysis.proteomics_analysis_service import (
    ProteomicsAnalysisService,
)
from lobster.services.analysis.proteomics_differential_service import (
    ProteomicsDifferentialService,
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


def create_proteomics_prompt() -> str:
    """
    Create the system prompt for the unified proteomics expert agent.

    The prompt explains:
    - Agent handles both mass spectrometry AND affinity proteomics
    - Auto-detection of platform type from data characteristics
    - Platform-specific defaults and considerations
    - Available tools and when to use each
    """
    return f"""<Identity_And_Role>
You are the Proteomics Expert: a unified agent specializing in BOTH mass spectrometry (DDA/DIA)
AND affinity-based (Olink, SomaScan, Luminex) proteomics analysis in Lobster AI's multi-agent architecture.
You work under the supervisor and execute comprehensive proteomics workflows.

<Core_Capabilities>
- Quality control and preprocessing for both MS and affinity proteomics data
- Platform-specific normalization (median/log2 for MS, quantile for affinity)
- Missing value handling appropriate to platform (MNAR for MS, imputation for affinity)
- Pattern analysis with dimensionality reduction and clustering
- Differential protein expression analysis
- Platform-specific validation (peptide mapping for MS, antibody specificity for affinity)
</Core_Capabilities>
</Identity_And_Role>

<Platform_Auto_Detection>
You automatically detect the proteomics platform type from data characteristics:

**Mass Spectrometry Detection Signals:**
- High missing values (30-70%) - MNAR pattern
- MS-specific columns: n_peptides, sequence_coverage, is_contaminant, is_reverse
- Large protein count (>3000 discovery proteomics)
- Platform hints: maxquant, spectronaut, dda, dia

**Affinity Platform Detection Signals:**
- Low missing values (<30%) - MAR pattern
- Affinity-specific columns: antibody_id, panel_type, npx_value, plate_id
- Small protein count (<200 targeted panels)
- Platform hints: olink, somascan, luminex, antibody

**Platform-Specific Defaults Applied:**

| Parameter | Mass Spectrometry | Affinity |
|-----------|------------------|----------|
| Max missing/sample | 70% | 30% |
| Max missing/protein | 80% | 50% |
| CV threshold | 50% | 30% |
| Normalization | median + log2 | quantile (no log) |
| Missing handling | preserve (MNAR) | impute KNN (MAR) |
| Fold change cutoff | 1.5x | 1.2x |
| PCA components | 15 | 10 |

</Platform_Auto_Detection>

<Your_Tools>

## Shared Tools (Platform-Aware):

1. **check_proteomics_status** - Check loaded modalities and detect platform type
   - Shows data characteristics and platform classification
   - Reports missing value patterns and quality indicators

2. **assess_proteomics_quality** - Run QC with platform-appropriate metrics
   - MS: contaminants, reverse hits, peptide counts, MNAR patterns
   - Affinity: CV, plate effects, antibody performance

3. **filter_proteomics_data** - Filter with platform-specific criteria
   - MS: peptide requirements, contaminant removal, high missing tolerance
   - Affinity: stricter missing thresholds, CV filtering, antibody QC

4. **normalize_proteomics_data** - Platform-appropriate normalization
   - MS: median normalization + log2 transformation, preserve MNAR
   - Affinity: quantile normalization, plate correction, KNN imputation

5. **analyze_proteomics_patterns** - Dimensionality reduction and clustering
   - PCA with platform-optimized components
   - Sample clustering for group discovery

6. **find_differential_proteins** - Statistical comparison between groups
   - MS: limma-moderated t-test, higher fold change threshold
   - Affinity: standard t-test, lower fold change threshold

7. **create_proteomics_summary** - Generate comprehensive analysis report

## Mass Spectrometry-Specific Tools:

8. **add_peptide_mapping** - Add peptide-to-protein mapping information
   - Critical for MS data quality assessment
   - Enables peptide-level filtering

## Affinity Platform-Specific Tools:

9. **validate_antibody_specificity** - Check for cross-reactive antibodies
   - Identifies highly correlated protein pairs
   - Flags potential cross-reactivity issues

10. **correct_plate_effects** - Correct batch effects from plate layout
    - Essential for multi-plate affinity studies
    - Uses ComBat or similar methods

</Your_Tools>

<Standard_Workflows>

## Mass Spectrometry Workflow

```
1. check_proteomics_status()                    # Verify MS detection
2. assess_proteomics_quality("modality")        # QC with MS metrics
3. filter_proteomics_data("modality_assessed")  # Remove contaminants, low peptides
4. normalize_proteomics_data("modality_filtered")  # Median + log2
5. analyze_proteomics_patterns("modality_normalized")  # PCA/clustering
6. find_differential_proteins("modality_analyzed", group_column)  # DE analysis
7. add_peptide_mapping("modality_de", peptide_file)  # Optional: peptide evidence
8. create_proteomics_summary()                  # Final report
```

## Affinity Proteomics Workflow

```
1. check_proteomics_status()                    # Verify affinity detection
2. assess_proteomics_quality("modality")        # QC with CV, plate metrics
3. filter_proteomics_data("modality_assessed")  # Remove failed antibodies
4. correct_plate_effects("modality_filtered")   # If multi-plate
5. normalize_proteomics_data("modality_corrected")  # Quantile + impute
6. validate_antibody_specificity("modality_normalized")  # Cross-reactivity
7. analyze_proteomics_patterns("modality_validated")  # PCA/clustering
8. find_differential_proteins("modality_analyzed", group_column)  # DE analysis
9. create_proteomics_summary()                  # Final report
```

</Standard_Workflows>

<Platform_Considerations>

**Mass Spectrometry (MNAR - Missing Not At Random):**
- Missing values reflect true biological absence (below detection limit)
- Do NOT aggressively impute - preserve missingness information
- Use filtering over imputation (require minimum observations)
- Peptide evidence crucial for protein quantification reliability
- Remove contaminants (keratin, albumin, trypsin) and reverse hits
- Account for protein groups and shared peptides
- Higher variability expected (CV 30-50%)

**Affinity Platforms (MAR - Missing At Random):**
- Missing values often technical failures, not biological
- Imputation appropriate (KNN, median)
- Stricter QC thresholds (CV <30%)
- Plate effects common and must be corrected
- Antibody cross-reactivity can confound results
- Smaller targeted panels - interpret with selection bias awareness
- Lower fold changes meaningful (1.2-1.5x)

</Platform_Considerations>

<Communication_Style>
Professional, structured markdown with clear sections. Report:
- Platform detection results and confidence
- QC metrics appropriate to detected platform
- Filtering and normalization statistics
- Clear platform-specific recommendations

When reporting results, clearly indicate:
1. Detected platform type and detection signals
2. Platform-specific parameters applied
3. Any platform-specific warnings or considerations
</Communication_Style>

<Important_Rules>
1. **ONLY perform analysis explicitly requested by the supervisor**
2. **Always report results back to the supervisor, never directly to users**
3. **Auto-detect platform type** and apply appropriate defaults
4. **Respect MNAR patterns** in MS data - don't over-impute
5. **Correct plate effects** in affinity data before analysis
6. **Validate modality existence** before any operation
7. **Log all operations** with proper provenance tracking (ir parameter)
8. **Use descriptive modality names** following the pattern: base_operation (e.g., olink_panel_normalized)
</Important_Rules>

Today's date: {date.today()}
"""


def proteomics_expert(
    data_manager: DataManagerV2,
    callback_handler=None,
    agent_name: str = "proteomics_expert",
    delegation_tools: list = None,
    force_platform_type: Optional[str] = None,
):
    """
    Factory function for unified proteomics expert agent.

    This agent handles both mass spectrometry and affinity proteomics analysis.
    It auto-detects platform type from data characteristics and applies
    appropriate defaults.

    Args:
        data_manager: DataManagerV2 instance for modality management
        callback_handler: Optional callback handler for LLM interactions
        agent_name: Name identifier for the agent instance
        delegation_tools: List of delegation tools for sub-agents
        force_platform_type: Override auto-detection ("mass_spec" or "affinity")

    Returns:
        Configured ReAct agent with proteomics analysis capabilities
    """
    settings = get_settings()
    model_params = settings.get_agent_llm_params("proteomics_expert")
    llm = create_llm("proteomics_expert", model_params)

    # Normalize callbacks to a flat list (fix double-nesting bug)
    if callback_handler and hasattr(llm, "with_config"):
        callbacks = callback_handler if isinstance(callback_handler, list) else [callback_handler]
        llm = llm.with_config(callbacks=callbacks)

    # Validate data manager type
    if not isinstance(data_manager, DataManagerV2):
        raise ValueError("ProteomicsExpert requires DataManagerV2 for modular analysis")

    # Initialize stateless services
    preprocessing_service = ProteomicsPreprocessingService()
    quality_service = ProteomicsQualityService()
    analysis_service = ProteomicsAnalysisService()
    differential_service = ProteomicsDifferentialService()

    # Analysis results storage
    analysis_results = {"summary": "", "details": {}}

    # Store forced platform type for tools to access
    _forced_platform_type = force_platform_type

    # =========================================================================
    # HELPER FUNCTIONS
    # =========================================================================

    def _get_platform_for_modality(modality_name: str) -> PlatformConfig:
        """Get platform config for a modality, using detection or forced type."""
        if _forced_platform_type:
            return get_platform_config(_forced_platform_type)

        try:
            adata = data_manager.get_modality(modality_name)
            detected_type = detect_platform_type(adata)
            return get_platform_config(detected_type)
        except ValueError:
            # Default to mass_spec if modality not found
            return get_platform_config("mass_spec")

    # =========================================================================
    # SHARED PLATFORM-AWARE TOOLS
    # =========================================================================

    @tool
    def check_proteomics_status(modality_name: str = "") -> str:
        """
        Check status of proteomics modalities and detect platform type.

        Args:
            modality_name: Specific modality to check (empty for all proteomics modalities)

        Returns:
            str: Status report with platform detection and data characteristics
        """
        try:
            if modality_name == "":
                # Show all modalities with proteomics focus
                modalities = data_manager.list_modalities()
                proteomics_terms = [
                    "proteomics",
                    "protein",
                    "ms",
                    "mass_spec",
                    "olink",
                    "soma",
                    "affinity",
                    "panel",
                ]
                proteomics_modalities = [
                    m
                    for m in modalities
                    if any(term in m.lower() for term in proteomics_terms)
                ]

                if not proteomics_modalities:
                    response = f"No proteomics modalities found. Available modalities: {modalities}\n"
                    response += "Ask the data_expert to load proteomics data using appropriate adapter."
                    return response

                response = f"Proteomics modalities ({len(proteomics_modalities)}):\n\n"
                for mod_name in proteomics_modalities:
                    adata = data_manager.get_modality(mod_name)
                    platform_config = _get_platform_for_modality(mod_name)
                    metrics = data_manager.get_quality_metrics(mod_name)

                    response += f"**{mod_name}**\n"
                    response += f"- Platform: {platform_config.display_name}\n"
                    response += (
                        f"- Shape: {adata.n_obs} samples x {adata.n_vars} proteins\n"
                    )
                    if "missing_value_percentage" in metrics:
                        response += f"- Missing values: {metrics['missing_value_percentage']:.1f}%\n"
                    response += "\n"

                return response

            else:
                # Check specific modality
                try:
                    adata = data_manager.get_modality(modality_name)
                    platform_config = _get_platform_for_modality(modality_name)
                    metrics = data_manager.get_quality_metrics(modality_name)

                    # Calculate missing rate
                    X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
                    missing_rate = np.isnan(X).sum() / X.size if X.size > 0 else 0

                    response = f"Proteomics modality '{modality_name}' status:\n\n"
                    response += f"**Platform Detection:**\n"
                    response += f"- Detected platform: {platform_config.display_name}\n"
                    response += f"- Platform type: {platform_config.platform_type}\n\n"

                    response += f"**Data Characteristics:**\n"
                    response += (
                        f"- Shape: {adata.n_obs} samples x {adata.n_vars} proteins\n"
                    )
                    response += f"- Missing values: {missing_rate*100:.1f}%\n"
                    expected_range = platform_config.expected_missing_rate_range
                    response += f"- Expected range for platform: {expected_range[0]*100:.0f}-{expected_range[1]*100:.0f}%\n"

                    # Platform-specific metadata check
                    if platform_config.platform_type == "mass_spec":
                        ms_cols = [
                            "n_peptides",
                            "n_unique_peptides",
                            "sequence_coverage",
                        ]
                        present_cols = [
                            col for col in ms_cols if col in adata.var.columns
                        ]
                        if present_cols:
                            response += f"- MS metadata available: {present_cols}\n"
                    else:
                        affinity_cols = ["antibody_id", "panel_type", "plate_id"]
                        present_cols = [
                            col for col in affinity_cols if col in adata.var.columns
                        ]
                        if present_cols:
                            response += (
                                f"- Affinity metadata available: {present_cols}\n"
                            )

                    # Show key columns
                    obs_cols = list(adata.obs.columns)[:5]
                    var_cols = list(adata.var.columns)[:5]
                    response += f"\n**Metadata:**\n"
                    response += f"- Sample columns: {obs_cols}...\n"
                    response += f"- Protein columns: {var_cols}...\n"

                    analysis_results["details"]["status"] = response
                    return response

                except ValueError:
                    return f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}"

        except Exception as e:
            logger.error(f"Error checking proteomics status: {e}")
            return f"Error checking status: {str(e)}"

    @tool
    def assess_proteomics_quality(
        modality_name: str,
        platform_type: str = "auto",
    ) -> str:
        """
        Run comprehensive quality assessment with platform-appropriate metrics.

        Args:
            modality_name: Name of the proteomics modality to assess
            platform_type: Platform type ("mass_spec", "affinity", or "auto" for detection)

        Returns:
            str: Quality assessment report with platform-specific metrics
        """
        try:
            adata = data_manager.get_modality(modality_name)
        except ValueError:
            return f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}"

        try:
            # Determine platform
            if platform_type == "auto":
                platform_config = _get_platform_for_modality(modality_name)
            else:
                platform_config = get_platform_config(platform_type)

            # Run quality assessments using service (returns 3-tuples)
            processed_adata, missing_stats, missing_ir = (
                quality_service.assess_missing_value_patterns(
                    adata,
                    sample_threshold=platform_config.max_missing_per_sample,
                    protein_threshold=platform_config.max_missing_per_protein,
                )
            )

            cv_adata, cv_stats, cv_ir = quality_service.assess_coefficient_variation(
                processed_adata,
                cv_threshold=platform_config.cv_threshold / 100,  # Convert to decimal
            )

            contam_adata, contam_stats, contam_ir = quality_service.detect_contaminants(
                cv_adata
            )
            final_adata, range_stats, range_ir = quality_service.evaluate_dynamic_range(
                contam_adata
            )

            # Update the modality with quality assessment results
            assessed_name = f"{modality_name}_quality_assessed"
            data_manager.modalities[assessed_name] = final_adata

            # Log with IR
            combined_stats = {
                **missing_stats,
                **cv_stats,
                **contam_stats,
                **range_stats,
            }
            data_manager.log_tool_usage(
                tool_name="assess_proteomics_quality",
                parameters={
                    "modality_name": modality_name,
                    "platform_type": platform_config.platform_type,
                },
                description=f"Quality assessment for {platform_config.display_name} data",
                ir=missing_ir,  # Use the first IR as representative
            )

            # Generate response
            response = f"Proteomics Quality Assessment for '{modality_name}':\n\n"
            response += f"**Platform:** {platform_config.display_name}\n\n"

            response += "**Dataset Characteristics:**\n"
            response += f"- Samples: {final_adata.n_obs}\n"
            response += f"- Proteins: {final_adata.n_vars}\n"

            # Missing values
            if "missing_value_percentage" in combined_stats:
                mv_pct = combined_stats["missing_value_percentage"]
                expected = platform_config.expected_missing_rate_range
                status = (
                    "OK"
                    if expected[0] * 100 <= mv_pct <= expected[1] * 100
                    else "CHECK"
                )
                response += f"- Missing values: {mv_pct:.1f}% [{status}] (expected: {expected[0]*100:.0f}-{expected[1]*100:.0f}%)\n"

            # CV metrics
            if "median_cv" in combined_stats:
                cv_val = combined_stats["median_cv"]
                cv_status = "OK" if cv_val <= platform_config.cv_threshold else "HIGH"
                response += f"- Median CV: {cv_val:.1f}% [{cv_status}] (threshold: {platform_config.cv_threshold}%)\n"

            # Platform-specific metrics
            if platform_config.platform_type == "mass_spec":
                if "contaminant_proteins" in combined_stats:
                    response += f"- Contaminant proteins: {combined_stats['contaminant_proteins']}\n"
                if "reverse_hits" in combined_stats:
                    response += (
                        f"- Reverse database hits: {combined_stats['reverse_hits']}\n"
                    )
            else:
                if "high_cv_proteins" in combined_stats:
                    response += (
                        f"- High CV proteins: {combined_stats['high_cv_proteins']}\n"
                    )

            # Dynamic range
            if "dynamic_range_log10" in combined_stats:
                response += f"- Dynamic range: {combined_stats['dynamic_range_log10']:.1f} log10 units\n"

            # Recommendations
            response += f"\n**Platform-Specific Recommendations:**\n"
            if platform_config.platform_type == "mass_spec":
                if combined_stats.get("contaminant_proteins", 0) > 0:
                    response += (
                        "- Remove contaminant proteins before downstream analysis\n"
                    )
                if combined_stats.get("reverse_hits", 0) > 0:
                    response += "- Remove reverse database hits (search artifacts)\n"
                response += "- Consider peptide count requirements for reliable quantification\n"
            else:
                if combined_stats.get("missing_value_percentage", 0) > 30:
                    response += "- High missing values unusual for affinity - check assay quality\n"
                if combined_stats.get("median_cv", 0) > 30:
                    response += (
                        "- High CVs suggest technical issues - check sample handling\n"
                    )
                response += "- Check for plate effects and correct if needed\n"

            response += f"\n**New modality created**: '{assessed_name}'"

            analysis_results["details"]["quality_assessment"] = response
            return response

        except Exception as e:
            logger.error(f"Error in proteomics quality assessment: {e}")
            return f"Error in quality assessment: {str(e)}"

    @tool
    def filter_proteomics_data(
        modality_name: str,
        platform_type: str = "auto",
        max_missing_per_sample: float = None,
        max_missing_per_protein: float = None,
        save_result: bool = True,
    ) -> str:
        """
        Filter proteomics data with platform-specific quality criteria.

        Args:
            modality_name: Name of the proteomics modality to filter
            platform_type: Platform type ("mass_spec", "affinity", or "auto")
            max_missing_per_sample: Override default missing threshold per sample
            max_missing_per_protein: Override default missing threshold per protein
            save_result: Whether to save the filtered modality

        Returns:
            str: Filtering report with statistics
        """
        try:
            adata = data_manager.get_modality(modality_name)
            original_shape = adata.shape
        except ValueError:
            return f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}"

        try:
            # Determine platform and defaults
            if platform_type == "auto":
                platform_config = _get_platform_for_modality(modality_name)
            else:
                platform_config = get_platform_config(platform_type)

            # Use platform defaults if not overridden
            max_sample = (
                max_missing_per_sample or platform_config.max_missing_per_sample
            )
            max_protein = (
                max_missing_per_protein or platform_config.max_missing_per_protein
            )

            # Create working copy
            adata_filtered = adata.copy()
            X = (
                adata_filtered.X.toarray()
                if hasattr(adata_filtered.X, "toarray")
                else adata_filtered.X
            )

            # Step 1: Filter based on missing values
            sample_missing_rate = np.isnan(X).sum(axis=1) / X.shape[1]
            protein_missing_rate = np.isnan(X).sum(axis=0) / X.shape[0]

            sample_filter = sample_missing_rate <= max_sample
            adata_filtered = adata_filtered[sample_filter, :].copy()

            # Recalculate after sample filtering
            X = (
                adata_filtered.X.toarray()
                if hasattr(adata_filtered.X, "toarray")
                else adata_filtered.X
            )
            protein_missing_rate = np.isnan(X).sum(axis=0) / X.shape[0]
            protein_filter = protein_missing_rate <= max_protein
            adata_filtered = adata_filtered[:, protein_filter].copy()

            # Step 2: Platform-specific filtering
            if platform_config.platform_type == "mass_spec":
                # MS: Filter by peptide count
                if "n_peptides" in adata_filtered.var.columns:
                    min_peptides = platform_config.platform_specific.get(
                        "min_peptides_per_protein", 2
                    )
                    peptide_filter = adata_filtered.var["n_peptides"] >= min_peptides
                    adata_filtered = adata_filtered[:, peptide_filter].copy()

                # MS: Remove contaminants
                if platform_config.platform_specific.get("remove_contaminants", True):
                    if "is_contaminant" in adata_filtered.var.columns:
                        adata_filtered = adata_filtered[
                            :, ~adata_filtered.var["is_contaminant"]
                        ].copy()

                # MS: Remove reverse hits
                if platform_config.platform_specific.get("remove_reverse_hits", True):
                    if "is_reverse" in adata_filtered.var.columns:
                        adata_filtered = adata_filtered[
                            :, ~adata_filtered.var["is_reverse"]
                        ].copy()
            else:
                # Affinity: Filter by CV
                if "cv" in adata_filtered.var.columns:
                    cv_filter = adata_filtered.var["cv"] <= platform_config.cv_threshold
                    adata_filtered = adata_filtered[:, cv_filter].copy()

                # Affinity: Remove failed antibodies
                if "antibody_quality" in adata_filtered.var.columns:
                    quality_filter = adata_filtered.var["antibody_quality"] != "failed"
                    adata_filtered = adata_filtered[:, quality_filter].copy()

            # Update modality
            filtered_name = f"{modality_name}_filtered"
            data_manager.modalities[filtered_name] = adata_filtered

            # Save if requested
            if save_result:
                save_path = f"{modality_name}_filtered.h5ad"
                data_manager.save_modality(filtered_name, save_path)

            # Create IR for provenance tracking
            ir = AnalysisStep(
                operation="proteomics.filtering.filter_data",
                tool_name="filter_proteomics_data",
                description=f"Filter proteomics data for {platform_config.display_name}",
                library="lobster.agents.proteomics.proteomics_expert",
                code_template="""# Proteomics data filtering
import numpy as np

# Filter based on missing values
X = adata.X.copy() if not hasattr(adata.X, 'toarray') else adata.X.toarray()
sample_missing_rate = np.isnan(X).sum(axis=1) / X.shape[1]
protein_missing_rate = np.isnan(X).sum(axis=0) / X.shape[0]

sample_filter = sample_missing_rate <= {{ max_missing_per_sample }}
adata_filtered = adata[sample_filter, :].copy()

X = adata_filtered.X.copy() if not hasattr(adata_filtered.X, 'toarray') else adata_filtered.X.toarray()
protein_missing_rate = np.isnan(X).sum(axis=0) / X.shape[0]
protein_filter = protein_missing_rate <= {{ max_missing_per_protein }}
adata_filtered = adata_filtered[:, protein_filter].copy()""",
                imports=["import numpy as np"],
                parameters={
                    "max_missing_per_sample": max_sample,
                    "max_missing_per_protein": max_protein,
                    "platform_type": platform_config.platform_type,
                },
                parameter_schema={
                    "max_missing_per_sample": ParameterSpec(
                        param_type="float",
                        papermill_injectable=True,
                        default_value=0.7,
                        required=False,
                        validation_rule="0 < max_missing_per_sample <= 1",
                        description="Maximum fraction of missing values per sample",
                    ),
                    "max_missing_per_protein": ParameterSpec(
                        param_type="float",
                        papermill_injectable=True,
                        default_value=0.8,
                        required=False,
                        validation_rule="0 < max_missing_per_protein <= 1",
                        description="Maximum fraction of missing values per protein",
                    ),
                },
                input_entities=["adata"],
                output_entities=["adata_filtered"],
            )

            # Log operation
            data_manager.log_tool_usage(
                tool_name="filter_proteomics_data",
                parameters={
                    "modality_name": modality_name,
                    "platform_type": platform_config.platform_type,
                    "max_missing_per_sample": max_sample,
                    "max_missing_per_protein": max_protein,
                },
                description=f"Filtered {platform_config.display_name} data",
                ir=ir,
            )

            # Generate summary
            samples_removed = original_shape[0] - adata_filtered.n_obs
            proteins_removed = original_shape[1] - adata_filtered.n_vars

            response = (
                f"Successfully filtered proteomics modality '{modality_name}'!\n\n"
            )
            response += f"**Platform:** {platform_config.display_name}\n\n"
            response += f"**Filtering Results:**\n"
            response += f"- Original shape: {original_shape[0]} samples x {original_shape[1]} proteins\n"
            response += f"- Filtered shape: {adata_filtered.n_obs} samples x {adata_filtered.n_vars} proteins\n"
            response += f"- Samples removed: {samples_removed} ({samples_removed/original_shape[0]*100:.1f}%)\n"
            response += f"- Proteins removed: {proteins_removed} ({proteins_removed/original_shape[1]*100:.1f}%)\n\n"

            response += (
                f"**Filtering Parameters ({platform_config.display_name} defaults):**\n"
            )
            response += f"- Max missing per sample: {max_sample*100:.0f}%\n"
            response += f"- Max missing per protein: {max_protein*100:.0f}%\n"

            response += f"\n**New modality created**: '{filtered_name}'"
            if save_result:
                response += f"\n**Saved to**: {save_path}"

            analysis_results["details"]["filtering"] = response
            return response

        except Exception as e:
            logger.error(f"Error filtering proteomics data: {e}")
            return f"Error in filtering: {str(e)}"

    @tool
    def normalize_proteomics_data(
        modality_name: str,
        platform_type: str = "auto",
        normalization_method: str = None,
        log_transform: bool = None,
        handle_missing: str = None,
        save_result: bool = True,
    ) -> str:
        """
        Normalize proteomics data using platform-appropriate methods.

        Args:
            modality_name: Name of the proteomics modality to normalize
            platform_type: Platform type ("mass_spec", "affinity", or "auto")
            normalization_method: Override default method (median, quantile, vsn)
            log_transform: Override log transformation setting
            handle_missing: Override imputation method (keep, impute_knn, impute_min)
            save_result: Whether to save the normalized modality

        Returns:
            str: Normalization report with processing details
        """
        try:
            adata = data_manager.get_modality(modality_name)
        except ValueError:
            return f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}"

        try:
            # Determine platform and defaults
            if platform_type == "auto":
                platform_config = _get_platform_for_modality(modality_name)
            else:
                platform_config = get_platform_config(platform_type)

            # Use platform defaults if not overridden
            norm_method = normalization_method or platform_config.default_normalization
            do_log = (
                log_transform
                if log_transform is not None
                else platform_config.log_transform
            )
            impute_method = handle_missing or platform_config.default_imputation

            # Step 1: Handle missing values if requested
            impute_stats = {}
            if impute_method == "impute_knn":
                processed_adata, impute_stats, impute_ir = (
                    preprocessing_service.impute_missing_values(adata, method="knn")
                )
            elif impute_method == "impute_min":
                processed_adata, impute_stats, impute_ir = (
                    preprocessing_service.impute_missing_values(
                        adata, method="min_prob"
                    )
                )
            else:
                processed_adata = adata.copy()
                impute_stats = {
                    "imputation_method": "none",
                    "imputation_applied": False,
                }

            # Step 2: Normalize
            normalized_adata, norm_stats, norm_ir = (
                preprocessing_service.normalize_intensities(
                    processed_adata,
                    method=norm_method,
                    log_transform=do_log,
                )
            )

            # Update modality
            normalized_name = f"{modality_name}_normalized"
            data_manager.modalities[normalized_name] = normalized_adata

            # Save if requested
            if save_result:
                save_path = f"{modality_name}_normalized.h5ad"
                data_manager.save_modality(normalized_name, save_path)

            # Log operation
            combined_stats = {**impute_stats, **norm_stats}
            data_manager.log_tool_usage(
                tool_name="normalize_proteomics_data",
                parameters={
                    "modality_name": modality_name,
                    "platform_type": platform_config.platform_type,
                    "normalization_method": norm_method,
                    "log_transform": do_log,
                    "handle_missing": impute_method,
                },
                description=f"Normalized {platform_config.display_name} data",
                ir=norm_ir,
            )

            # Generate response
            response = (
                f"Successfully normalized proteomics modality '{modality_name}'!\n\n"
            )
            response += f"**Platform:** {platform_config.display_name}\n\n"
            response += f"**Normalization Settings:**\n"
            response += f"- Method: {norm_method}\n"
            response += f"- Log transformation: {do_log}\n"
            response += f"- Missing value handling: {impute_method}\n\n"

            response += f"**Processing Details:**\n"
            if combined_stats.get("imputation_applied", False):
                response += f"- Imputation applied: {combined_stats.get('imputation_method', 'unknown')}\n"
                if "n_imputed_values" in combined_stats:
                    response += (
                        f"- Values imputed: {combined_stats['n_imputed_values']}\n"
                    )
            else:
                if platform_config.platform_type == "mass_spec":
                    response += "- Preserved missing values (MNAR pattern preserved)\n"
                else:
                    response += "- No imputation applied\n"

            if combined_stats.get("log_transform_applied", False):
                response += "- Log2 transformation applied\n"

            response += f"\n**New modality created**: '{normalized_name}'"
            if save_result:
                response += f"\n**Saved to**: {save_path}"

            analysis_results["details"]["normalization"] = response
            return response

        except Exception as e:
            logger.error(f"Error normalizing proteomics data: {e}")
            return f"Error in normalization: {str(e)}"

    @tool
    def analyze_proteomics_patterns(
        modality_name: str,
        platform_type: str = "auto",
        analysis_type: str = "pca_clustering",
        n_components: int = None,
        n_clusters: int = 4,
        save_result: bool = True,
    ) -> str:
        """
        Perform pattern analysis with dimensionality reduction and clustering.

        Args:
            modality_name: Name of the proteomics modality to analyze
            platform_type: Platform type ("mass_spec", "affinity", or "auto")
            analysis_type: Type of analysis ("pca_clustering", "pca_only")
            n_components: Number of PCA components (uses platform default if None)
            n_clusters: Number of clusters for sample grouping
            save_result: Whether to save results

        Returns:
            str: Analysis report with PCA and clustering results
        """
        try:
            adata = data_manager.get_modality(modality_name).copy()
        except ValueError:
            return f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}"

        try:
            # Determine platform and defaults
            if platform_type == "auto":
                platform_config = _get_platform_for_modality(modality_name)
            else:
                platform_config = get_platform_config(platform_type)

            n_pcs = n_components or platform_config.default_n_pca_components

            # Perform dimensionality reduction
            pca_adata, pca_stats, pca_ir = (
                analysis_service.perform_dimensionality_reduction(
                    adata, method="pca", n_components=n_pcs
                )
            )

            # Perform clustering if requested
            if analysis_type == "pca_clustering":
                clustered_adata, cluster_stats, cluster_ir = (
                    analysis_service.perform_clustering_analysis(
                        pca_adata, clustering_method="kmeans", n_clusters=n_clusters
                    )
                )
                final_adata = clustered_adata
                combined_stats = {**pca_stats, **cluster_stats}
                ir = cluster_ir
            else:
                final_adata = pca_adata
                combined_stats = pca_stats
                ir = pca_ir

            # Update modality
            analyzed_name = f"{modality_name}_analyzed"
            data_manager.modalities[analyzed_name] = final_adata

            # Save if requested
            if save_result:
                save_path = f"{modality_name}_analyzed.h5ad"
                data_manager.save_modality(analyzed_name, save_path)

            # Log operation
            data_manager.log_tool_usage(
                tool_name="analyze_proteomics_patterns",
                parameters={
                    "modality_name": modality_name,
                    "platform_type": platform_config.platform_type,
                    "analysis_type": analysis_type,
                    "n_components": n_pcs,
                    "n_clusters": n_clusters,
                },
                description=f"Pattern analysis for {platform_config.display_name} data",
                ir=ir,
            )

            # Generate response
            response = (
                f"Successfully analyzed proteomics patterns in '{modality_name}'!\n\n"
            )
            response += f"**Platform:** {platform_config.display_name}\n\n"
            response += f"**PCA Results:**\n"
            response += f"- Components computed: {n_pcs}\n"

            if "explained_variance_ratio" in combined_stats:
                ev_ratio = combined_stats["explained_variance_ratio"][:3]
                response += f"- Explained variance (PC1-PC3): {[f'{x*100:.1f}%' for x in ev_ratio]}\n"

            if "components_for_90_variance" in combined_stats:
                response += f"- Components for 90% variance: {combined_stats['components_for_90_variance']}\n"

            if analysis_type == "pca_clustering":
                response += f"\n**Clustering Results:**\n"
                if "n_clusters_found" in combined_stats:
                    response += f"- Clusters: {combined_stats['n_clusters_found']}\n"
                if "silhouette_score" in combined_stats:
                    response += f"- Silhouette score: {combined_stats['silhouette_score']:.3f}\n"

            response += f"\n**New modality created**: '{analyzed_name}'"
            if save_result:
                response += f"\n**Saved to**: {save_path}"

            analysis_results["details"]["pattern_analysis"] = response
            return response

        except Exception as e:
            logger.error(f"Error analyzing proteomics patterns: {e}")
            return f"Error in pattern analysis: {str(e)}"

    @tool
    def find_differential_proteins(
        modality_name: str,
        group_column: str,
        platform_type: str = "auto",
        method: str = None,
        fdr_threshold: float = 0.05,
        fold_change_threshold: float = None,
    ) -> str:
        """
        Find differentially expressed proteins between groups.

        Args:
            modality_name: Name of the proteomics modality
            group_column: Column in obs containing group labels
            platform_type: Platform type ("mass_spec", "affinity", or "auto")
            method: Statistical method (default: platform-specific)
            fdr_threshold: FDR threshold for significance
            fold_change_threshold: Minimum fold change (uses platform default if None)

        Returns:
            str: Differential expression results
        """
        try:
            adata = data_manager.get_modality(modality_name).copy()
        except ValueError:
            return f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}"

        try:
            if group_column not in adata.obs.columns:
                return f"Group column '{group_column}' not found. Available columns: {list(adata.obs.columns)}"

            # Determine platform and defaults
            if platform_type == "auto":
                platform_config = _get_platform_for_modality(modality_name)
            else:
                platform_config = get_platform_config(platform_type)

            fc_threshold = (
                fold_change_threshold or platform_config.default_fold_change_threshold
            )
            test_method = method or (
                "t_test" if platform_config.platform_type == "affinity" else "t_test"
            )

            # Perform differential expression
            de_adata, de_stats, de_ir = (
                differential_service.perform_differential_expression(
                    adata,
                    group_column=group_column,
                    test_method=test_method,
                    fdr_threshold=fdr_threshold,
                    fold_change_threshold=fc_threshold,
                )
            )

            # Update modality
            de_name = f"{modality_name}_de_analysis"
            data_manager.modalities[de_name] = de_adata

            # Log operation
            data_manager.log_tool_usage(
                tool_name="find_differential_proteins",
                parameters={
                    "modality_name": modality_name,
                    "group_column": group_column,
                    "platform_type": platform_config.platform_type,
                    "method": test_method,
                    "fdr_threshold": fdr_threshold,
                    "fold_change_threshold": fc_threshold,
                },
                description=f"Differential expression for {platform_config.display_name} data",
                ir=de_ir,
            )

            # Generate response
            response = (
                f"Successfully found differential proteins in '{modality_name}'!\n\n"
            )
            response += f"**Platform:** {platform_config.display_name}\n\n"
            response += f"**Analysis Parameters:**\n"
            response += f"- Method: {test_method}\n"
            response += f"- FDR threshold: {fdr_threshold}\n"
            response += f"- Fold change threshold: {fc_threshold}\n\n"

            response += f"**Results:**\n"
            if "n_comparisons" in de_stats:
                response += f"- Comparisons performed: {de_stats['n_comparisons']}\n"
            if "total_tests" in de_stats:
                response += f"- Total tests: {de_stats['total_tests']}\n"
            if "n_significant" in de_stats:
                response += f"- Significant proteins: {de_stats['n_significant']}\n"

            # Show top significant proteins
            if (
                "top_significant_proteins" in de_stats
                and de_stats["top_significant_proteins"]
            ):
                response += "\n**Top Significant Proteins:**\n"
                for protein_info in de_stats["top_significant_proteins"][:5]:
                    response += f"- {protein_info['protein']}: log2FC={protein_info['log2_fold_change']:.2f}, FDR={protein_info['p_adjusted']:.2e}\n"

            response += f"\n**Results stored in modality**: '{de_name}'"
            response += "\n**Access results via**: adata.uns['differential_analysis']"

            analysis_results["details"]["differential_analysis"] = response
            return response

        except Exception as e:
            logger.error(f"Error in differential protein analysis: {e}")
            return f"Error finding differential proteins: {str(e)}"

    @tool
    def create_proteomics_summary() -> str:
        """
        Create comprehensive summary of all proteomics analysis steps performed.

        Returns:
            str: Complete analysis summary
        """
        if not analysis_results["details"]:
            return "No proteomics analyses have been performed yet. Run some analysis tools first."

        summary = "# Proteomics Analysis Summary\n\n"

        for step, details in analysis_results["details"].items():
            summary += f"## {step.replace('_', ' ').title()}\n"
            summary += f"{details}\n\n"

        # Add current modality status
        modalities = data_manager.list_modalities()
        proteomics_terms = ["proteomics", "protein", "ms", "olink", "soma", "affinity"]
        proteomics_modalities = [
            m for m in modalities if any(term in m.lower() for term in proteomics_terms)
        ]
        summary += "## Current Proteomics Modalities\n"
        summary += f"Proteomics modalities: {', '.join(proteomics_modalities)}\n\n"

        analysis_results["summary"] = summary
        return summary

    # =========================================================================
    # MASS SPECTROMETRY-SPECIFIC TOOLS
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
        try:
            adata = data_manager.get_modality(modality_name)
        except ValueError:
            return f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}"

        try:
            # Get proteomics adapter
            adapter_instance = None
            for adapter_name, adapter in data_manager.adapters.items():
                if "proteomics" in adapter_name:
                    adapter_instance = adapter
                    break

            if not adapter_instance:
                return "No proteomics adapter available for peptide mapping"

            # Add peptide mapping
            adata_with_peptides = adapter_instance.add_peptide_mapping(
                adata, peptide_file_path
            )

            # Update modality
            peptide_name = f"{modality_name}_with_peptides"
            data_manager.modalities[peptide_name] = adata_with_peptides

            # Save if requested
            if save_result:
                save_path = f"{modality_name}_with_peptides.h5ad"
                data_manager.save_modality(peptide_name, save_path)

            # Get peptide info
            peptide_info = adata_with_peptides.uns.get("peptide_to_protein", {})

            # Create IR for provenance tracking
            ir = AnalysisStep(
                operation="proteomics.ms.add_peptide_mapping",
                tool_name="add_peptide_mapping",
                description="Add peptide-to-protein mapping for MS proteomics",
                library="lobster.agents.proteomics.proteomics_expert",
                code_template="""# Peptide mapping (MS-specific)
import pandas as pd

# Load peptide mapping from file
peptide_df = pd.read_csv({{ peptide_file_path | tojson }})

# Expected columns: peptide_sequence, protein_id, n_proteins (shared)
# Add peptide counts to protein annotations
peptide_counts = peptide_df.groupby('protein_id').size().to_dict()
adata.var['n_peptides'] = [peptide_counts.get(p, 0) for p in adata.var_names]

# Store full mapping in uns
adata.uns['peptide_to_protein'] = peptide_df.to_dict('records')""",
                imports=["import pandas as pd"],
                parameters={
                    "peptide_file_path": peptide_file_path,
                    "n_peptides_mapped": peptide_info.get("n_peptides", 0),
                    "n_proteins_with_peptides": peptide_info.get("n_proteins", 0),
                },
                parameter_schema={
                    "peptide_file_path": ParameterSpec(
                        param_type="str",
                        papermill_injectable=True,
                        default_value="",
                        required=True,
                        description="Path to CSV file with peptide-protein mapping",
                    ),
                },
                input_entities=["adata", "peptide_file"],
                output_entities=["adata_with_peptides"],
            )

            # Log operation
            data_manager.log_tool_usage(
                tool_name="add_peptide_mapping",
                parameters={
                    "modality_name": modality_name,
                    "peptide_file_path": peptide_file_path,
                },
                description="Added peptide-to-protein mapping for MS data",
                ir=ir,
            )

            response = f"Successfully added peptide mapping to MS modality '{modality_name}'!\n\n"
            response += f"**Peptide Mapping Results:**\n"
            response += (
                f"- Peptides mapped: {peptide_info.get('n_peptides', 'Unknown')}\n"
            )
            response += f"- Proteins with peptides: {peptide_info.get('n_proteins', 'Unknown')}\n"
            response += f"- Mapping file: {peptide_file_path}\n\n"

            response += f"**Updated Protein Metadata:**\n"
            response += "- Added: n_peptides, n_unique_peptides, sequence_coverage\n"
            response += "- Full mapping stored in: uns['peptide_to_protein']\n"

            response += f"\n**New modality created**: '{peptide_name}'"
            if save_result:
                response += f"\n**Saved to**: {save_path}"

            response += "\n\n**MS-Specific Notes:**"
            response += "\n- Use peptide counts for protein filtering (recommend >= 2)"
            response += "\n- Unique peptides provide more reliable quantification"

            analysis_results["details"]["peptide_mapping"] = response
            return response

        except Exception as e:
            logger.error(f"Error adding peptide mapping: {e}")
            return f"Error adding peptide mapping: {str(e)}"

    # =========================================================================
    # AFFINITY PLATFORM-SPECIFIC TOOLS
    # =========================================================================

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
            # Create working copy
            adata_validated = adata.copy()
            X = (
                adata_validated.X.toarray()
                if hasattr(adata_validated.X, "toarray")
                else adata_validated.X
            )

            # Handle NaN for correlation
            X_filled = np.nan_to_num(X, nan=np.nanmean(X))

            # Check for cross-reactivity patterns via correlation
            cross_reactive_pairs = []

            if adata_validated.n_vars > 1:
                correlation_matrix = np.corrcoef(X_filled.T)

                for i in range(len(correlation_matrix)):
                    for j in range(i + 1, len(correlation_matrix)):
                        if correlation_matrix[i, j] > cross_reactivity_threshold:
                            protein_i = adata_validated.var_names[i]
                            protein_j = adata_validated.var_names[j]
                            cross_reactive_pairs.append(
                                (protein_i, protein_j, correlation_matrix[i, j])
                            )

            # Flag cross-reactive antibodies
            if "cross_reactive" not in adata_validated.var.columns:
                adata_validated.var["cross_reactive"] = False

            for protein_pair in cross_reactive_pairs:
                protein_i, protein_j, _ = protein_pair
                adata_validated.var.loc[protein_i, "cross_reactive"] = True
                adata_validated.var.loc[protein_j, "cross_reactive"] = True

            # Update modality
            validated_name = f"{modality_name}_antibody_validated"
            data_manager.modalities[validated_name] = adata_validated

            # Save if requested
            if save_result:
                save_path = f"{modality_name}_antibody_validated.h5ad"
                data_manager.save_modality(validated_name, save_path)

            # Create IR for provenance tracking
            ir = AnalysisStep(
                operation="proteomics.affinity.validate_antibody_specificity",
                tool_name="validate_antibody_specificity",
                description="Validate antibody specificity for affinity proteomics",
                library="lobster.agents.proteomics.proteomics_expert",
                code_template="""# Antibody cross-reactivity check (Affinity-specific)
import numpy as np

# Calculate correlation matrix between protein measurements
X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
X_filled = np.nan_to_num(X, nan=np.nanmean(X))
correlation_matrix = np.corrcoef(X_filled.T)

# Identify highly correlated pairs (potential cross-reactivity)
cross_reactive_pairs = []
for i in range(len(correlation_matrix)):
    for j in range(i + 1, len(correlation_matrix)):
        if correlation_matrix[i, j] > {{ cross_reactivity_threshold }}:
            cross_reactive_pairs.append((i, j, correlation_matrix[i, j]))

# Flag cross-reactive antibodies
adata.var['cross_reactive'] = False
for i, j, _ in cross_reactive_pairs:
    adata.var.iloc[i, adata.var.columns.get_loc('cross_reactive')] = True
    adata.var.iloc[j, adata.var.columns.get_loc('cross_reactive')] = True""",
                imports=["import numpy as np"],
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

            # Log operation
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
            response += f"**Antibody Validation Results:**\n"
            response += f"- Total proteins analyzed: {adata_validated.n_vars}\n"
            response += (
                f"- Cross-reactive pairs detected: {len(cross_reactive_pairs)}\n"
            )
            response += f"- Correlation threshold: {cross_reactivity_threshold}\n\n"

            if cross_reactive_pairs:
                response += "**Potential Cross-Reactive Pairs:**\n"
                for protein_i, protein_j, correlation in cross_reactive_pairs[:5]:
                    response += f"- {protein_i} <-> {protein_j}: r={correlation:.3f}\n"

                if len(cross_reactive_pairs) > 5:
                    response += (
                        f"- ... and {len(cross_reactive_pairs) - 5} more pairs\n"
                    )

                response += "\n**Recommendations:**\n"
                response += "- Review antibody specificity documentation\n"
                response += "- Consider removing highly cross-reactive antibodies\n"
                response += "- Validate results with orthogonal methods\n"
            else:
                response += "No significant cross-reactivity detected.\n"

            response += f"\n**New modality created**: '{validated_name}'"
            if save_result:
                response += f"\n**Saved to**: {save_path}"

            analysis_results["details"]["antibody_validation"] = response
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

            # Use preprocessing service for batch correction
            corrected_adata, batch_stats, batch_ir = (
                preprocessing_service.correct_batch_effects(
                    adata,
                    batch_key=plate_column,
                    method=method,
                )
            )

            # Update modality
            corrected_name = f"{modality_name}_plate_corrected"
            data_manager.modalities[corrected_name] = corrected_adata

            # Save if requested
            if save_result:
                save_path = f"{modality_name}_plate_corrected.h5ad"
                data_manager.save_modality(corrected_name, save_path)

            # Log operation
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
            response += f"**Plate Correction Results:**\n"
            response += f"- Method: {method}\n"
            response += f"- Plate column: {plate_column}\n"

            if "n_batches_corrected" in batch_stats:
                response += (
                    f"- Plates corrected: {batch_stats['n_batches_corrected']}\n"
                )

            response += f"\n**New modality created**: '{corrected_name}'"
            if save_result:
                response += f"\n**Saved to**: {save_path}"

            response += "\n\n**Affinity Platform Notes:**"
            response += "\n- Plate correction essential for multi-plate studies"
            response += "\n- Verify correction by checking inter-plate correlation"

            analysis_results["details"]["plate_correction"] = response
            return response

        except Exception as e:
            logger.error(f"Error correcting plate effects: {e}")
            return f"Error in plate correction: {str(e)}"

    # =========================================================================
    # COLLECT ALL TOOLS
    # =========================================================================

    # Shared platform-aware tools
    shared_tools = [
        check_proteomics_status,
        assess_proteomics_quality,
        filter_proteomics_data,
        normalize_proteomics_data,
        analyze_proteomics_patterns,
        find_differential_proteins,
        create_proteomics_summary,
    ]

    # MS-specific tools
    ms_tools = [
        add_peptide_mapping,
    ]

    # Affinity-specific tools
    affinity_tools = [
        validate_antibody_specificity,
        correct_plate_effects,
    ]

    # Combine all direct tools
    direct_tools = shared_tools + ms_tools + affinity_tools

    # Add delegation tools if provided
    tools = direct_tools
    if delegation_tools:
        tools = tools + delegation_tools

    # Create system prompt
    system_prompt = create_proteomics_prompt()

    return create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt,
        name=agent_name,
        state_schema=ProteomicsExpertState,
    )
