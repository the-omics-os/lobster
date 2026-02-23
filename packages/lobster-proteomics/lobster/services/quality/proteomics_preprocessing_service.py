"""
Proteomics preprocessing service for missing value imputation, normalization, batch correction,
PTM site import, peptide-to-protein summarization, and PTM-to-protein normalization.

This service implements professional-grade preprocessing methods specifically designed for
proteomics data including MNAR imputation, proteomics-specific normalization methods,
batch correction techniques suitable for mass spectrometry data, and PTM analysis workflows.

All methods return 3-tuples (AnnData, Dict, AnalysisStep) for provenance tracking and
reproducible notebook export via /pipeline export.
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import anndata
import numpy as np
import pandas as pd
from scipy.stats import linregress, rankdata
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer, SimpleImputer

from lobster.core.analysis_ir import AnalysisStep, ParameterSpec
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


class ProteomicsPreprocessingError(Exception):
    """Base exception for proteomics preprocessing operations."""

    pass


class ProteomicsPreprocessingService:
    """
    Advanced preprocessing service for proteomics data.

    This stateless service provides methods for missing value imputation, normalization,
    and batch correction following best practices from proteomics analysis pipelines.
    Handles the unique challenges of proteomics data including high missing value rates,
    intensity-dependent noise, and batch effects.
    """

    def __init__(self):
        """
        Initialize the proteomics preprocessing service.

        This service is stateless and doesn't require a data manager instance.
        """
        logger.debug("Initializing stateless ProteomicsPreprocessingService")
        logger.debug("ProteomicsPreprocessingService initialized successfully")

    def _create_ir_impute_missing_values(
        self,
        method: str,
        knn_neighbors: int,
        min_prob_percentile: float,
        mnar_width: float,
        mnar_downshift: float,
    ) -> AnalysisStep:
        """Create IR for missing value imputation."""
        return AnalysisStep(
            operation="proteomics.preprocessing.impute_missing_values",
            tool_name="impute_missing_values",
            description="Impute missing values in proteomics data using method-specific approaches (KNN, min_prob, MNAR, mixed)",
            library="lobster.services.quality.proteomics_preprocessing_service",
            code_template="""# Missing value imputation
from lobster.services.quality.proteomics_preprocessing_service import ProteomicsPreprocessingService

service = ProteomicsPreprocessingService()
adata_imputed, stats, _ = service.impute_missing_values(
    adata,
    method={{ method | tojson }},
    knn_neighbors={{ knn_neighbors }},
    min_prob_percentile={{ min_prob_percentile }},
    mnar_width={{ mnar_width }},
    mnar_downshift={{ mnar_downshift }}
)
print(f"Imputed {stats['original_missing_count']} values using {method}")""",
            imports=[
                "from lobster.services.quality.proteomics_preprocessing_service import ProteomicsPreprocessingService"
            ],
            parameters={
                "method": method,
                "knn_neighbors": knn_neighbors,
                "min_prob_percentile": min_prob_percentile,
                "mnar_width": mnar_width,
                "mnar_downshift": mnar_downshift,
            },
            parameter_schema={
                "method": ParameterSpec(
                    param_type="str",
                    papermill_injectable=True,
                    default_value="mixed",
                    required=False,
                    validation_rule="method in ['knn', 'min_prob', 'mnar', 'mixed']",
                    description="Imputation method: knn, min_prob, mnar, or mixed",
                ),
                "knn_neighbors": ParameterSpec(
                    param_type="int",
                    papermill_injectable=True,
                    default_value=5,
                    required=False,
                    validation_rule="knn_neighbors > 0",
                    description="Number of neighbors for KNN imputation",
                ),
                "min_prob_percentile": ParameterSpec(
                    param_type="float",
                    papermill_injectable=True,
                    default_value=2.5,
                    required=False,
                    validation_rule="0 < min_prob_percentile < 100",
                    description="Percentile for minimum probability imputation",
                ),
                "mnar_width": ParameterSpec(
                    param_type="float",
                    papermill_injectable=True,
                    default_value=0.3,
                    required=False,
                    validation_rule="mnar_width > 0",
                    description="Width parameter for MNAR distribution",
                ),
                "mnar_downshift": ParameterSpec(
                    param_type="float",
                    papermill_injectable=True,
                    default_value=1.8,
                    required=False,
                    validation_rule="mnar_downshift > 0",
                    description="Downshift parameter for MNAR distribution",
                ),
            },
            input_entities=["adata"],
            output_entities=["adata_imputed"],
        )

    def _create_ir_normalize_intensities(
        self,
        method: str,
        log_transform: bool,
        pseudocount_strategy: str,
        reference_sample: Optional[str],
    ) -> AnalysisStep:
        """Create IR for intensity normalization."""
        return AnalysisStep(
            operation="proteomics.preprocessing.normalize_intensities",
            tool_name="normalize_intensities",
            description="Normalize proteomics intensity data using method-specific approaches (median, quantile, VSN, total_sum)",
            library="lobster.services.quality.proteomics_preprocessing_service",
            code_template="""# Intensity normalization
from lobster.services.quality.proteomics_preprocessing_service import ProteomicsPreprocessingService

service = ProteomicsPreprocessingService()
adata_norm, stats, _ = service.normalize_intensities(
    adata,
    method={{ method | tojson }},
    log_transform={{ log_transform | tojson }},
    pseudocount_strategy={{ pseudocount_strategy | tojson }},
    reference_sample={{ reference_sample | tojson }}
)
print(f"Normalized using {method}, log_transform={log_transform}")""",
            imports=[
                "from lobster.services.quality.proteomics_preprocessing_service import ProteomicsPreprocessingService"
            ],
            parameters={
                "method": method,
                "log_transform": log_transform,
                "pseudocount_strategy": pseudocount_strategy,
                "reference_sample": reference_sample,
            },
            parameter_schema={
                "method": ParameterSpec(
                    param_type="str",
                    papermill_injectable=True,
                    default_value="median",
                    required=False,
                    validation_rule="method in ['median', 'quantile', 'vsn', 'total_sum']",
                    description="Normalization method: median, quantile, vsn, or total_sum",
                ),
                "log_transform": ParameterSpec(
                    param_type="bool",
                    papermill_injectable=True,
                    default_value=True,
                    required=False,
                    description="Whether to apply log2 transformation",
                ),
                "pseudocount_strategy": ParameterSpec(
                    param_type="str",
                    papermill_injectable=True,
                    default_value="adaptive",
                    required=False,
                    validation_rule="pseudocount_strategy in ['adaptive', 'fixed', 'min_observed']",
                    description="Strategy for pseudocount in log transformation",
                ),
                "reference_sample": ParameterSpec(
                    param_type="Optional[str]",
                    papermill_injectable=True,
                    default_value=None,
                    required=False,
                    description="Reference sample for normalization (if applicable)",
                ),
            },
            input_entities=["adata"],
            output_entities=["adata_norm"],
        )

    def _create_ir_correct_batch_effects(
        self,
        batch_key: str,
        method: str,
        n_pcs: int,
        reference_batch: Optional[str],
    ) -> AnalysisStep:
        """Create IR for batch effect correction."""
        return AnalysisStep(
            operation="proteomics.preprocessing.correct_batch_effects",
            tool_name="correct_batch_effects",
            description="Correct for batch effects in proteomics data using method-specific approaches (combat, median_centering, reference_based)",
            library="lobster.services.quality.proteomics_preprocessing_service",
            code_template="""# Batch effect correction
from lobster.services.quality.proteomics_preprocessing_service import ProteomicsPreprocessingService

service = ProteomicsPreprocessingService()
adata_corrected, stats, _ = service.correct_batch_effects(
    adata,
    batch_key={{ batch_key | tojson }},
    method={{ method | tojson }},
    n_pcs={{ n_pcs }},
    reference_batch={{ reference_batch | tojson }}
)
print(f"Batch correction applied: {method}, batches: {stats['n_batches']}")""",
            imports=[
                "from lobster.services.quality.proteomics_preprocessing_service import ProteomicsPreprocessingService"
            ],
            parameters={
                "batch_key": batch_key,
                "method": method,
                "n_pcs": n_pcs,
                "reference_batch": reference_batch,
            },
            parameter_schema={
                "batch_key": ParameterSpec(
                    param_type="str",
                    papermill_injectable=True,
                    default_value="batch",
                    required=True,
                    description="Column in obs containing batch information",
                ),
                "method": ParameterSpec(
                    param_type="str",
                    papermill_injectable=True,
                    default_value="combat",
                    required=False,
                    validation_rule="method in ['combat', 'median_centering', 'reference_based']",
                    description="Batch correction method: combat, median_centering, or reference_based",
                ),
                "n_pcs": ParameterSpec(
                    param_type="int",
                    papermill_injectable=True,
                    default_value=50,
                    required=False,
                    validation_rule="n_pcs > 0",
                    description="Number of principal components for analysis",
                ),
                "reference_batch": ParameterSpec(
                    param_type="Optional[str]",
                    papermill_injectable=True,
                    default_value=None,
                    required=False,
                    description="Reference batch for correction (if applicable)",
                ),
            },
            input_entities=["adata"],
            output_entities=["adata_corrected"],
        )

    def _create_ir_import_ptm_site_data(
        self,
        file_path: str,
        ptm_type: str,
        localization_threshold: float,
        filter_contaminants: bool,
        filter_reverse: bool,
        intensity_type: str,
    ) -> AnalysisStep:
        """Create IR for PTM site data import."""
        return AnalysisStep(
            operation="proteomics.preprocessing.import_ptm_site_data",
            tool_name="import_ptm_site_data",
            description="Import PTM site-level quantification from MaxQuant-style output files, filter by localization probability, and construct site-level AnnData",
            library="lobster.services.quality.proteomics_preprocessing_service",
            code_template="""# Import PTM site-level data
from lobster.services.quality.proteomics_preprocessing_service import ProteomicsPreprocessingService

service = ProteomicsPreprocessingService()
adata_ptm, stats, _ = service.import_ptm_site_data(
    file_path={{ file_path | tojson }},
    ptm_type={{ ptm_type | tojson }},
    localization_threshold={{ localization_threshold }},
    filter_contaminants={{ filter_contaminants | tojson }},
    filter_reverse={{ filter_reverse | tojson }},
    intensity_type={{ intensity_type | tojson }}
)
print(f"Imported {stats['n_sites_after_filter']} {ptm_type} sites from {stats['n_samples']} samples")""",
            imports=[
                "from lobster.services.quality.proteomics_preprocessing_service import ProteomicsPreprocessingService"
            ],
            parameters={
                "file_path": file_path,
                "ptm_type": ptm_type,
                "localization_threshold": localization_threshold,
                "filter_contaminants": filter_contaminants,
                "filter_reverse": filter_reverse,
                "intensity_type": intensity_type,
            },
            parameter_schema={
                "file_path": ParameterSpec(
                    param_type="str",
                    papermill_injectable=True,
                    default_value=None,
                    required=True,
                    description="Path to PTM site file (tab-delimited MaxQuant output)",
                ),
                "ptm_type": ParameterSpec(
                    param_type="str",
                    papermill_injectable=True,
                    default_value="phospho",
                    required=False,
                    validation_rule="ptm_type in ['phospho', 'acetyl', 'ubiquitin']",
                    description="Type of PTM: phospho, acetyl, or ubiquitin",
                ),
                "localization_threshold": ParameterSpec(
                    param_type="float",
                    papermill_injectable=True,
                    default_value=0.75,
                    required=False,
                    validation_rule="0 <= localization_threshold <= 1",
                    description="Minimum localization probability for class I sites",
                ),
                "filter_contaminants": ParameterSpec(
                    param_type="bool",
                    papermill_injectable=True,
                    default_value=True,
                    required=False,
                    description="Remove contaminant proteins",
                ),
                "filter_reverse": ParameterSpec(
                    param_type="bool",
                    papermill_injectable=True,
                    default_value=True,
                    required=False,
                    description="Remove reverse database hits",
                ),
                "intensity_type": ParameterSpec(
                    param_type="str",
                    papermill_injectable=True,
                    default_value="auto",
                    required=False,
                    validation_rule="intensity_type in ['auto', 'lfq', 'intensity']",
                    description="Intensity column type: auto, lfq, or intensity",
                ),
            },
            input_entities=["file_path"],
            output_entities=["adata_ptm"],
        )

    def import_ptm_site_data(
        self,
        file_path: str,
        ptm_type: str = "phospho",
        localization_threshold: float = 0.75,
        filter_contaminants: bool = True,
        filter_reverse: bool = True,
        intensity_type: str = "auto",
    ) -> Tuple[anndata.AnnData, Dict[str, Any], AnalysisStep]:
        """
        Parse PTM site-level quantification data into AnnData.

        Reads tab-delimited MaxQuant site-level output files and constructs
        site-level AnnData where var_names are site IDs (gene_residuePosition format).

        Args:
            file_path: Path to PTM site file (tab-delimited)
            ptm_type: Type of PTM ("phospho", "acetyl", "ubiquitin")
            localization_threshold: Minimum localization probability for class I sites (default 0.75)
            filter_contaminants: Remove contaminant proteins
            filter_reverse: Remove reverse database hits
            intensity_type: "auto" detects LFQ/Intensity columns, or specify "lfq", "intensity"

        Returns:
            Tuple of (AnnData, stats_dict, AnalysisStep)

        Raises:
            ProteomicsPreprocessingError: If PTM site import fails
        """
        try:
            logger.info(f"Starting PTM site data import from {file_path} (type: {ptm_type})")

            # 1. Read the file
            df = pd.read_csv(file_path, sep="\t", low_memory=False)
            n_sites_total = len(df)
            logger.info(f"Read {n_sites_total} rows from {Path(file_path).name}")

            # 2. Identify key columns (case-insensitive matching)
            col_map = self._identify_ptm_columns(df)

            # 3. Filter by localization probability
            n_before_loc = len(df)
            if col_map["localization_prob"] is not None:
                df = df[df[col_map["localization_prob"]] >= localization_threshold].copy()
                logger.info(
                    f"Localization filter ({localization_threshold}): "
                    f"{n_before_loc} -> {len(df)} sites"
                )
            else:
                logger.warning(
                    "No localization probability column found, skipping filter"
                )

            # 4. Filter contaminants
            n_contaminants_removed = 0
            if filter_contaminants and col_map["contaminant"] is not None:
                contaminant_mask = df[col_map["contaminant"]].fillna("").astype(str).str.strip() == "+"
                n_contaminants_removed = contaminant_mask.sum()
                df = df[~contaminant_mask].copy()
                logger.info(f"Removed {n_contaminants_removed} contaminant sites")

            # 5. Filter reverse hits
            n_reverse_removed = 0
            if filter_reverse and col_map["reverse"] is not None:
                reverse_mask = df[col_map["reverse"]].fillna("").astype(str).str.strip() == "+"
                n_reverse_removed = reverse_mask.sum()
                df = df[~reverse_mask].copy()
                logger.info(f"Removed {n_reverse_removed} reverse hits")

            # 6. Auto-detect intensity columns
            intensity_cols, sample_names = self._detect_intensity_columns(
                df, intensity_type
            )
            if not intensity_cols:
                raise ProteomicsPreprocessingError(
                    "No intensity columns found. Expected columns matching "
                    "'Intensity <sample>' or 'LFQ intensity <sample>'"
                )
            logger.info(
                f"Found {len(intensity_cols)} intensity columns "
                f"(type: {'LFQ' if 'LFQ' in intensity_cols[0] else 'raw'})"
            )

            # 7. Construct site IDs
            site_ids = self._construct_site_ids(df, col_map)

            # Handle duplicate site IDs by appending a suffix
            seen = {}
            unique_site_ids: List[str] = []
            for sid in site_ids:
                if sid in seen:
                    seen[sid] += 1
                    unique_site_ids.append(f"{sid}_dup{seen[sid]}")
                else:
                    seen[sid] = 0
                    unique_site_ids.append(sid)
            site_ids = unique_site_ids

            # 8. Build intensity matrix (sites as rows in MaxQuant -> transpose to samples x sites)
            intensity_matrix = df[intensity_cols].values.astype(np.float64)
            # Transpose: MaxQuant has sites as rows, we want samples (columns) as obs
            X = intensity_matrix.T  # shape: (n_samples, n_sites)

            # 9. Replace 0 values with NaN (MaxQuant convention: 0 = undetected)
            X[X == 0] = np.nan

            # 10. Build var metadata
            var_data = {"site_id": site_ids, "ptm_type": ptm_type}
            if col_map["gene_names"] is not None:
                var_data["gene"] = df[col_map["gene_names"]].fillna("unknown").values
            if col_map["position"] is not None:
                var_data["position"] = df[col_map["position"]].values
            if col_map["amino_acid"] is not None:
                var_data["amino_acid"] = df[col_map["amino_acid"]].values
            if col_map["localization_prob"] is not None:
                var_data["localization_prob"] = df[col_map["localization_prob"]].values
            if col_map["multiplicity"] is not None:
                var_data["multiplicity"] = df[col_map["multiplicity"]].values

            var_df = pd.DataFrame(var_data, index=site_ids)

            # 11. Build obs metadata
            obs_df = pd.DataFrame(index=sample_names)
            obs_df.index.name = "sample"

            # 12. Build AnnData
            adata = anndata.AnnData(
                X=X,
                obs=obs_df,
                var=var_df,
            )
            adata.uns["ptm_type"] = ptm_type
            adata.uns["localization_threshold"] = localization_threshold
            adata.uns["source_file"] = Path(file_path).name

            # 13. Calculate stats
            total_values = X.size
            missing_count = np.isnan(X).sum()
            missing_percentage = (
                (missing_count / total_values) * 100 if total_values > 0 else 0
            )

            stats = {
                "n_sites_total": n_sites_total,
                "n_sites_after_filter": adata.n_vars,
                "n_samples": adata.n_obs,
                "n_contaminants_removed": n_contaminants_removed,
                "n_reverse_removed": n_reverse_removed,
                "missing_percentage": float(missing_percentage),
                "ptm_type": ptm_type,
                "intensity_type": "lfq" if "LFQ" in intensity_cols[0] else "intensity",
                "analysis_type": "ptm_site_import",
            }

            logger.info(
                f"PTM site import complete: {adata.n_obs} samples x {adata.n_vars} sites "
                f"({missing_percentage:.1f}% missing)"
            )

            # 14. Create IR
            ir = self._create_ir_import_ptm_site_data(
                file_path,
                ptm_type,
                localization_threshold,
                filter_contaminants,
                filter_reverse,
                intensity_type,
            )
            return adata, stats, ir

        except Exception as e:
            logger.exception(f"Error in PTM site data import: {e}")
            raise ProteomicsPreprocessingError(
                f"PTM site data import failed: {str(e)}"
            )

    def _identify_ptm_columns(self, df: pd.DataFrame) -> Dict[str, Optional[str]]:
        """Identify key PTM columns in a MaxQuant-style site file."""
        columns = df.columns.tolist()
        col_lower = {c.lower(): c for c in columns}

        def find_col(*candidates: str) -> Optional[str]:
            for candidate in candidates:
                candidate_lower = candidate.lower()
                if candidate_lower in col_lower:
                    return col_lower[candidate_lower]
                # Partial match
                for cl, orig in col_lower.items():
                    if candidate_lower in cl:
                        return orig
            return None

        return {
            "localization_prob": find_col(
                "Localization prob", "localization_probability", "Localization probability"
            ),
            "gene_names": find_col("Gene names", "Gene.names", "Proteins", "Leading proteins"),
            "position": find_col("Position", "Positions within proteins"),
            "amino_acid": find_col("Amino acid", "Amino.acid"),
            "contaminant": find_col("Potential contaminant", "Contaminant"),
            "reverse": find_col("Reverse"),
            "multiplicity": find_col("Multiplicity"),
        }

    def _detect_intensity_columns(
        self, df: pd.DataFrame, intensity_type: str
    ) -> Tuple[List[str], List[str]]:
        """Detect intensity columns and extract sample names."""
        columns = df.columns.tolist()

        # Find LFQ intensity columns
        lfq_pattern = re.compile(r"^LFQ [Ii]ntensity\s+(.+)$")
        lfq_cols = []
        lfq_samples = []
        for col in columns:
            m = lfq_pattern.match(col)
            if m:
                lfq_cols.append(col)
                lfq_samples.append(m.group(1).strip())

        # Find raw intensity columns (exclude "Intensity" alone and columns with
        # internal qualifiers like "Intensity __", and exclude LFQ intensity columns)
        raw_pattern = re.compile(r"^Intensity\s+(.+)$")
        raw_cols = []
        raw_samples = []
        for col in columns:
            if col.lower().startswith("lfq"):
                continue
            m = raw_pattern.match(col)
            if m:
                sample = m.group(1).strip()
                # Skip internal MaxQuant columns like "Intensity L", "Intensity H" in SILAC
                # unless they look like actual sample names (more than 1 character)
                raw_cols.append(col)
                raw_samples.append(sample)

        if intensity_type == "lfq":
            return lfq_cols, lfq_samples
        elif intensity_type == "intensity":
            return raw_cols, raw_samples
        else:  # auto: prefer LFQ if available
            if lfq_cols:
                return lfq_cols, lfq_samples
            return raw_cols, raw_samples

    def _construct_site_ids(
        self, df: pd.DataFrame, col_map: Dict[str, Optional[str]]
    ) -> List[str]:
        """Construct unique site identifiers in gene_residuePosition format."""
        site_ids = []
        for _, row in df.iterrows():
            # Get gene name
            gene = "unknown"
            if col_map["gene_names"] is not None:
                gene_val = row[col_map["gene_names"]]
                if pd.notna(gene_val) and str(gene_val).strip():
                    # Take first gene if multiple separated by ;
                    gene = str(gene_val).split(";")[0].strip()

            # Get amino acid and position
            aa = ""
            if col_map["amino_acid"] is not None:
                aa_val = row[col_map["amino_acid"]]
                if pd.notna(aa_val):
                    aa = str(aa_val).strip()

            pos = ""
            if col_map["position"] is not None:
                pos_val = row[col_map["position"]]
                if pd.notna(pos_val):
                    # Take first position if multiple separated by ;
                    pos = str(int(float(str(pos_val).split(";")[0].strip())))

            # Construct site ID
            site_id = f"{gene}_{aa}{pos}"

            # Handle multiplicity for multiply modified sites
            if col_map["multiplicity"] is not None:
                mult_val = row[col_map["multiplicity"]]
                if pd.notna(mult_val) and int(float(mult_val)) > 1:
                    site_id = f"{site_id}_m{int(float(mult_val))}"

            site_ids.append(site_id)
        return site_ids

    def impute_missing_values(
        self,
        adata: anndata.AnnData,
        method: str = "mixed",
        knn_neighbors: int = 5,
        min_prob_percentile: float = 2.5,
        mnar_width: float = 0.3,
        mnar_downshift: float = 1.8,
    ) -> Tuple[anndata.AnnData, Dict[str, Any], AnalysisStep]:
        """
        Impute missing values using proteomics-appropriate methods.

        Args:
            adata: AnnData object with proteomics data
            method: Method ('knn', 'min_prob', 'mnar', 'mixed')
            knn_neighbors: Number of neighbors for KNN imputation
            min_prob_percentile: Percentile for minimum probability imputation
            mnar_width: Width parameter for MNAR distribution
            mnar_downshift: Downshift parameter for MNAR distribution

        Returns:
            Tuple[anndata.AnnData, Dict[str, Any], AnalysisStep]: Imputed AnnData,
                processing stats, and IR for notebook export

        Raises:
            ProteomicsPreprocessingError: If imputation fails
        """
        try:
            logger.info(f"Starting missing value imputation with method: {method}")

            # Create working copy
            adata_imputed = adata.copy()
            original_shape = adata_imputed.shape
            logger.info(
                f"Input data shape: {original_shape[0]} samples × {original_shape[1]} proteins"
            )

            # Store original data for comparison
            if adata_imputed.raw is None:
                adata_imputed.raw = adata_imputed.copy()

            # Check for missing values
            X = adata_imputed.X.copy()
            if not np.isnan(X).any():
                logger.info("No missing values detected, skipping imputation")
                ir = self._create_ir_impute_missing_values(
                    method,
                    knn_neighbors,
                    min_prob_percentile,
                    mnar_width,
                    mnar_downshift,
                )
                return (
                    adata_imputed,
                    {
                        "method": method,
                        "missing_values_found": False,
                        "imputation_performed": False,
                        "analysis_type": "missing_value_imputation",
                    },
                    ir,
                )

            # Calculate missing value statistics
            total_missing = np.isnan(X).sum()
            total_values = X.size
            missing_percentage = (total_missing / total_values) * 100

            logger.info(
                f"Missing values: {total_missing:,} ({missing_percentage:.1f}%)"
            )

            # Apply imputation method
            if method == "knn":
                X_imputed = self._knn_imputation(X, knn_neighbors)
            elif method == "min_prob":
                X_imputed = self._min_prob_imputation(X, min_prob_percentile)
            elif method == "mnar":
                X_imputed = self._mnar_imputation(X, mnar_width, mnar_downshift)
            elif method == "mixed":
                X_imputed = self._mixed_imputation(
                    X, knn_neighbors, min_prob_percentile, mnar_width, mnar_downshift
                )
            else:
                raise ProteomicsPreprocessingError(
                    f"Unknown imputation method: {method}"
                )

            # Update the data
            adata_imputed.X = X_imputed

            # Calculate imputation statistics
            imputation_stats = {
                "method": method,
                "missing_values_found": True,
                "imputation_performed": True,
                "original_missing_count": int(total_missing),
                "original_missing_percentage": float(missing_percentage),
                "remaining_missing_count": int(np.isnan(X_imputed).sum()),
                "proteins_processed": adata_imputed.n_vars,
                "samples_processed": adata_imputed.n_obs,
                "knn_neighbors": knn_neighbors if method in ["knn", "mixed"] else None,
                "min_prob_percentile": (
                    min_prob_percentile if method in ["min_prob", "mixed"] else None
                ),
                "analysis_type": "missing_value_imputation",
            }

            logger.info(
                f"Imputation completed: {total_missing:,} → {np.isnan(X_imputed).sum():,} missing values"
            )

            # Create IR for provenance tracking
            ir = self._create_ir_impute_missing_values(
                method, knn_neighbors, min_prob_percentile, mnar_width, mnar_downshift
            )
            return adata_imputed, imputation_stats, ir

        except Exception as e:
            logger.exception(f"Error in missing value imputation: {e}")
            raise ProteomicsPreprocessingError(
                f"Missing value imputation failed: {str(e)}"
            )

    def normalize_intensities(
        self,
        adata: anndata.AnnData,
        method: str = "median",
        log_transform: bool = True,
        pseudocount_strategy: str = "adaptive",
        reference_sample: Optional[str] = None,
    ) -> Tuple[anndata.AnnData, Dict[str, Any], AnalysisStep]:
        """
        Normalize proteomics intensity data using appropriate methods.

        Args:
            adata: AnnData object with proteomics data
            method: Normalization method ('median', 'quantile', 'vsn', 'total_sum')
            log_transform: Whether to apply log2 transformation
            pseudocount_strategy: Strategy for pseudocount ('adaptive', 'fixed', 'min_observed')
            reference_sample: Reference sample for normalization (if applicable)

        Returns:
            Tuple[anndata.AnnData, Dict[str, Any], AnalysisStep]: Normalized AnnData,
                processing stats, and IR for notebook export

        Raises:
            ProteomicsPreprocessingError: If normalization fails
        """
        try:
            logger.info(f"Starting intensity normalization with method: {method}")

            # Create working copy
            adata_norm = adata.copy()
            original_shape = adata_norm.shape
            logger.info(
                f"Input data shape: {original_shape[0]} samples × {original_shape[1]} proteins"
            )

            # Store raw data if not already stored
            if adata_norm.raw is None:
                adata_norm.raw = adata_norm.copy()

            X = adata_norm.X.copy()

            # Check for negative values
            if np.any(X < 0):
                logger.warning(
                    "Negative values detected in data - may indicate pre-processed data"
                )

            # Apply normalization
            if method == "median":
                X_norm = self._median_normalization(X)
            elif method == "quantile":
                X_norm = self._quantile_normalization(X)
            elif method == "vsn":
                X_norm = self._vsn_normalization(X)
            elif method == "total_sum":
                X_norm = self._total_sum_normalization(X)
            else:
                raise ProteomicsPreprocessingError(
                    f"Unknown normalization method: {method}"
                )

            # Apply log transformation if requested
            log_stats = {}
            if log_transform:
                X_norm, log_stats = self._apply_log_transformation(
                    X_norm, pseudocount_strategy
                )

            # Update the data
            adata_norm.X = X_norm

            # Store in layers
            adata_norm.layers["normalized"] = X_norm
            if log_transform:
                adata_norm.layers["log2_normalized"] = X_norm

            # Calculate normalization statistics
            normalization_stats = {
                "method": method,
                "log_transform": log_transform,
                "pseudocount_strategy": pseudocount_strategy if log_transform else None,
                "samples_processed": adata_norm.n_obs,
                "proteins_processed": adata_norm.n_vars,
                "median_intensity_before": float(np.nanmedian(adata_norm.raw.X)),
                "median_intensity_after": float(np.nanmedian(X_norm)),
                "analysis_type": "intensity_normalization",
                **log_stats,
            }

            logger.info(f"Normalization completed: {method} method applied")

            # Create IR for provenance tracking
            ir = self._create_ir_normalize_intensities(
                method, log_transform, pseudocount_strategy, reference_sample
            )
            return adata_norm, normalization_stats, ir

        except Exception as e:
            logger.exception(f"Error in intensity normalization: {e}")
            raise ProteomicsPreprocessingError(
                f"Intensity normalization failed: {str(e)}"
            )

    def correct_batch_effects(
        self,
        adata: anndata.AnnData,
        batch_key: str,
        method: str = "combat",
        n_pcs: int = 50,
        reference_batch: Optional[str] = None,
    ) -> Tuple[anndata.AnnData, Dict[str, Any], AnalysisStep]:
        """
        Correct for batch effects in proteomics data.

        Args:
            adata: AnnData object with proteomics data
            batch_key: Column in obs containing batch information
            method: Batch correction method ('combat', 'median_centering', 'reference_based')
            n_pcs: Number of principal components for analysis
            reference_batch: Reference batch for correction (if applicable)

        Returns:
            Tuple[anndata.AnnData, Dict[str, Any], AnalysisStep]: Batch-corrected AnnData,
                processing stats, and IR for notebook export

        Raises:
            ProteomicsPreprocessingError: If batch correction fails
        """
        try:
            logger.info(f"Starting batch correction with method: {method}")

            # Create working copy
            adata_corrected = adata.copy()
            original_shape = adata_corrected.shape
            logger.info(
                f"Input data shape: {original_shape[0]} samples × {original_shape[1]} proteins"
            )

            # Check batch information
            if batch_key not in adata_corrected.obs.columns:
                raise ProteomicsPreprocessingError(
                    f"Batch key '{batch_key}' not found in obs"
                )

            batch_counts = adata_corrected.obs[batch_key].value_counts().to_dict()
            n_batches = len(batch_counts)
            logger.info(f"Found {n_batches} batches: {batch_counts}")

            if n_batches < 2:
                logger.warning("Less than 2 batches found, skipping batch correction")
                ir = self._create_ir_correct_batch_effects(
                    batch_key, method, n_pcs, reference_batch
                )
                return (
                    adata_corrected,
                    {
                        "method": method,
                        "batch_correction_performed": False,
                        "n_batches": n_batches,
                        "analysis_type": "batch_correction",
                    },
                    ir,
                )

            # Store original data
            if adata_corrected.raw is None:
                adata_corrected.raw = adata_corrected.copy()

            X = adata_corrected.X.copy()

            # Apply batch correction method
            if method == "combat":
                X_corrected = self._combat_correction(X, adata_corrected.obs[batch_key])
            elif method == "median_centering":
                X_corrected = self._median_centering_correction(
                    X, adata_corrected.obs[batch_key]
                )
            elif method == "reference_based":
                X_corrected = self._reference_based_correction(
                    X, adata_corrected.obs[batch_key], reference_batch
                )
            else:
                raise ProteomicsPreprocessingError(
                    f"Unknown batch correction method: {method}"
                )

            # Update the data
            adata_corrected.X = X_corrected
            adata_corrected.layers["batch_corrected"] = X_corrected

            # Calculate batch correction statistics
            batch_stats = self._calculate_batch_correction_stats(
                adata_corrected.raw.X,
                X_corrected,
                adata_corrected.obs[batch_key],
                n_pcs,
            )

            correction_stats = {
                "method": method,
                "batch_correction_performed": True,
                "batch_key": batch_key,
                "n_batches": n_batches,
                "batch_counts": batch_counts,
                "reference_batch": reference_batch,
                "samples_processed": adata_corrected.n_obs,
                "proteins_processed": adata_corrected.n_vars,
                "analysis_type": "batch_correction",
                **batch_stats,
            }

            logger.info(f"Batch correction completed: {method} method applied")

            # Create IR for provenance tracking
            ir = self._create_ir_correct_batch_effects(
                batch_key, method, n_pcs, reference_batch
            )
            return adata_corrected, correction_stats, ir

        except Exception as e:
            logger.exception(f"Error in batch correction: {e}")
            raise ProteomicsPreprocessingError(f"Batch correction failed: {str(e)}")

    # Helper methods for missing value imputation
    def _knn_imputation(self, X: np.ndarray, n_neighbors: int) -> np.ndarray:
        """Apply KNN imputation."""
        logger.info(f"Applying KNN imputation with {n_neighbors} neighbors")
        imputer = KNNImputer(n_neighbors=n_neighbors)
        return imputer.fit_transform(X)

    def _min_prob_imputation(self, X: np.ndarray, percentile: float) -> np.ndarray:
        """Apply minimum probability imputation."""
        logger.info(
            f"Applying minimum probability imputation at {percentile}th percentile"
        )
        X_imputed = X.copy()

        for i in range(X.shape[1]):  # For each protein
            protein_values = X[:, i]
            observed_values = protein_values[~np.isnan(protein_values)]

            if len(observed_values) > 0:
                # Use percentile of observed values as imputation value
                impute_value = np.percentile(observed_values, percentile)
                X_imputed[np.isnan(protein_values), i] = impute_value

        return X_imputed

    def _mnar_imputation(
        self, X: np.ndarray, width: float, downshift: float
    ) -> np.ndarray:
        """Apply MNAR (Missing Not At Random) imputation."""
        logger.info(f"Applying MNAR imputation (width={width}, downshift={downshift})")
        X_imputed = X.copy()

        for i in range(X.shape[1]):  # For each protein
            protein_values = X[:, i]
            observed_values = protein_values[~np.isnan(protein_values)]

            if len(observed_values) > 0:
                # Create left-truncated normal distribution
                mean_obs = np.mean(observed_values)
                std_obs = np.std(observed_values)

                # Parameters for MNAR distribution
                mnar_mean = mean_obs - downshift * std_obs
                mnar_std = width * std_obs

                # Generate random values from truncated normal
                n_missing = np.isnan(protein_values).sum()
                if n_missing > 0:
                    impute_values = np.random.normal(mnar_mean, mnar_std, n_missing)
                    # Ensure values are below the minimum observed value
                    min_obs = np.min(observed_values)
                    impute_values = np.minimum(impute_values, min_obs - 0.1 * std_obs)
                    X_imputed[np.isnan(protein_values), i] = impute_values

        return X_imputed

    def _mixed_imputation(
        self,
        X: np.ndarray,
        knn_neighbors: int,
        min_prob_percentile: float,
        mnar_width: float,
        mnar_downshift: float,
        mcar_threshold: float = 0.4,
    ) -> np.ndarray:
        """Apply mixed imputation strategy based on missing value patterns."""
        logger.info("Applying mixed imputation strategy")
        X_imputed = X.copy()

        # Calculate missing percentages per protein
        missing_per_protein = np.isnan(X).sum(axis=0) / X.shape[0]

        for i in range(X.shape[1]):
            protein_values = X[:, i]
            missing_pct = missing_per_protein[i]

            if np.isnan(protein_values).any():
                if missing_pct < mcar_threshold:
                    # Low missing rate: use KNN (assume MCAR)
                    protein_data = X[:, [i]]
                    imputer = KNNImputer(n_neighbors=min(knn_neighbors, X.shape[0] - 1))
                    X_imputed[:, [i]] = imputer.fit_transform(protein_data)
                else:
                    # High missing rate: use MNAR approach
                    observed_values = protein_values[~np.isnan(protein_values)]
                    if len(observed_values) > 0:
                        mean_obs = np.mean(observed_values)
                        std_obs = np.std(observed_values)
                        mnar_mean = mean_obs - mnar_downshift * std_obs
                        mnar_std = mnar_width * std_obs

                        n_missing = np.isnan(protein_values).sum()
                        impute_values = np.random.normal(mnar_mean, mnar_std, n_missing)
                        X_imputed[np.isnan(protein_values), i] = impute_values

        return X_imputed

    # Helper methods for normalization
    def _median_normalization(self, X: np.ndarray) -> np.ndarray:
        """Apply median normalization."""
        logger.info("Applying median normalization")
        sample_medians = np.nanmedian(X, axis=1)
        global_median = np.nanmedian(sample_medians)

        # Avoid division by zero
        sample_medians[sample_medians == 0] = 1
        normalization_factors = global_median / sample_medians

        return X * normalization_factors[:, np.newaxis]

    def _quantile_normalization(self, X: np.ndarray) -> np.ndarray:
        """Apply quantile normalization."""
        logger.info("Applying quantile normalization")

        # Handle missing values by working with ranks
        X_norm = X.copy()

        # Get ranks for each sample (handling NaN)
        ranks = np.zeros_like(X)
        for i in range(X.shape[0]):
            sample_data = X[i, :]
            valid_mask = ~np.isnan(sample_data)
            if valid_mask.sum() > 0:
                ranks[i, valid_mask] = rankdata(sample_data[valid_mask])

        # Calculate mean values at each rank
        max_rank = int(np.nanmax(ranks))
        mean_values = np.zeros(max_rank + 1)

        for rank in range(1, max_rank + 1):
            rank_values = []
            for i in range(X.shape[0]):
                rank_positions = ranks[i, :] == rank
                if rank_positions.any():
                    rank_values.extend(X[i, rank_positions])
            if rank_values:
                mean_values[rank] = np.mean(rank_values)

        # Apply quantile normalization
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                if not np.isnan(X[i, j]):
                    rank = int(ranks[i, j])
                    X_norm[i, j] = mean_values[rank]

        return X_norm

    def _vsn_normalization(self, X: np.ndarray) -> np.ndarray:
        """Apply variance stabilizing normalization (VSN-like).

        WARNING: Simplified VSN approximation using arcsinh(x/2) transform.
        This does NOT perform true VSN with maximum likelihood parameter estimation.
        For publication-grade analysis requiring true VSN, use R's vsn package:
            BiocManager::install("vsn")
            vsn::justvsn(data_matrix)

        The arcsinh transform provides similar variance stabilization properties
        but without the sample-specific calibration that true VSN provides.
        """
        logger.info("Applying VSN-like normalization")

        # Simple VSN approximation: asinh transformation with scaling
        X_positive = np.maximum(X, 1e-8)  # Avoid log of zero
        return np.arcsinh(X_positive / 2)

    def _total_sum_normalization(self, X: np.ndarray) -> np.ndarray:
        """Apply total sum normalization."""
        logger.info("Applying total sum normalization")
        sample_sums = np.nansum(X, axis=1)
        sample_sums[sample_sums == 0] = 1  # Avoid division by zero
        return (X / sample_sums[:, np.newaxis]) * 1e6  # Scale to 1M

    def _apply_log_transformation(
        self, X: np.ndarray, strategy: str
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Apply log2 transformation with appropriate pseudocount."""
        if strategy == "adaptive":
            # Use fraction of minimum positive value
            min_positive = np.nanmin(X[X > 0]) if np.any(X > 0) else 1.0
            pseudocount = min_positive * 0.1
        elif strategy == "fixed":
            pseudocount = 1.0
        elif strategy == "min_observed":
            pseudocount = np.nanmin(X[X > 0]) if np.any(X > 0) else 1.0
        else:
            pseudocount = 1.0

        X_log = np.log2(X + pseudocount)

        log_stats = {
            "pseudocount": float(pseudocount),
            "pseudocount_strategy": strategy,
            "median_after_log": float(np.nanmedian(X_log)),
        }

        return X_log, log_stats

    # Helper methods for batch correction
    def _combat_correction(self, X: np.ndarray, batch_labels: pd.Series) -> np.ndarray:
        """Apply ComBat-like batch correction."""
        logger.info("Applying ComBat-like batch correction")

        X_corrected = X.copy()
        unique_batches = batch_labels.unique()

        # Calculate overall mean for each protein
        overall_means = np.nanmean(X, axis=0)

        for batch in unique_batches:
            batch_mask = batch_labels == batch
            batch_data = X[batch_mask, :]

            # Calculate batch-specific means and standard deviations
            batch_means = np.nanmean(batch_data, axis=0)
            batch_stds = np.nanstd(batch_data, axis=0)
            batch_stds[batch_stds == 0] = 1  # Avoid division by zero

            # Apply correction: standardize within batch, then scale to overall distribution
            for i in np.where(batch_mask)[0]:
                X_corrected[i, :] = ((X[i, :] - batch_means) / batch_stds) * np.nanstd(
                    X, axis=0
                ) + overall_means

        return X_corrected

    def _median_centering_correction(
        self, X: np.ndarray, batch_labels: pd.Series
    ) -> np.ndarray:
        """Apply median centering batch correction."""
        logger.info("Applying median centering batch correction")

        X_corrected = X.copy()
        unique_batches = batch_labels.unique()
        overall_median = np.nanmedian(X, axis=0)

        for batch in unique_batches:
            batch_mask = batch_labels == batch
            batch_data = X[batch_mask, :]
            batch_median = np.nanmedian(batch_data, axis=0)

            # Apply median centering
            correction = overall_median - batch_median
            X_corrected[batch_mask, :] = batch_data + correction

        return X_corrected

    def _reference_based_correction(
        self, X: np.ndarray, batch_labels: pd.Series, reference_batch: Optional[str]
    ) -> np.ndarray:
        """Apply reference-based batch correction."""
        logger.info(
            f"Applying reference-based batch correction (reference: {reference_batch})"
        )

        X_corrected = X.copy()
        unique_batches = batch_labels.unique()

        # Determine reference batch
        if reference_batch is None or reference_batch not in unique_batches:
            # Use batch with most samples as reference
            batch_counts = batch_labels.value_counts()
            reference_batch = batch_counts.index[0]
            logger.info(
                f"Using batch with most samples as reference: {reference_batch}"
            )

        # Get reference batch statistics
        ref_mask = batch_labels == reference_batch
        ref_data = X[ref_mask, :]
        ref_median = np.nanmedian(ref_data, axis=0)

        # Correct other batches to match reference
        for batch in unique_batches:
            if batch != reference_batch:
                batch_mask = batch_labels == batch
                batch_data = X[batch_mask, :]
                batch_median = np.nanmedian(batch_data, axis=0)

                correction = ref_median - batch_median
                X_corrected[batch_mask, :] = batch_data + correction

        return X_corrected

    def _calculate_batch_correction_stats(
        self,
        X_before: np.ndarray,
        X_after: np.ndarray,
        batch_labels: pd.Series,
        n_pcs: int,
    ) -> Dict[str, Any]:
        """Calculate statistics to assess batch correction effectiveness."""

        # PCA before and after correction
        try:
            # Handle missing values for PCA
            imputer = SimpleImputer(strategy="mean")
            X_before_pca = imputer.fit_transform(X_before)
            X_after_pca = imputer.fit_transform(X_after)

            pca = PCA(n_components=min(n_pcs, X_before_pca.shape[1]))

            # PCA before correction
            pca.fit_transform(X_before_pca)
            var_explained_before = pca.explained_variance_ratio_[:3].sum()

            # PCA after correction
            pca.fit_transform(X_after_pca)
            var_explained_after = pca.explained_variance_ratio_[:3].sum()

            stats = {
                "pca_variance_before": float(var_explained_before),
                "pca_variance_after": float(var_explained_after),
                "correction_effectiveness": (
                    "improved"
                    if var_explained_after > var_explained_before
                    else "mixed"
                ),
            }

        except Exception as e:
            logger.warning(f"Could not calculate PCA statistics: {e}")
            stats = {
                "pca_variance_before": None,
                "pca_variance_after": None,
                "correction_effectiveness": "unknown",
            }

        return stats
