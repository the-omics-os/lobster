"""
Metabolomics data adapter with schema enforcement and multi-platform support.

This module provides the MetabolomicsAdapter that handles loading,
validation, and preprocessing of metabolomics data including LC-MS,
GC-MS, and NMR platforms with appropriate schema enforcement.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import anndata
import numpy as np
import pandas as pd

from lobster.core.adapters.base import BaseAdapter
from lobster.core.interfaces.validator import ValidationResult
from lobster.core.schemas.metabolomics import MetabolomicsSchema

logger = logging.getLogger(__name__)


class MetabolomicsAdapter(BaseAdapter):
    """
    Adapter for metabolomics data with schema enforcement.

    This adapter handles loading and validation of metabolomics data across
    LC-MS, GC-MS, and NMR platforms with appropriate schema validation,
    metabolite metadata standardization, and quality metric computation.
    """

    def __init__(
        self,
        data_type: str = "lc_ms",
        strict_validation: bool = False,
        handle_missing_values: str = "keep",
    ):
        """
        Initialize the metabolomics adapter.

        Args:
            data_type: Type of data ('lc_ms', 'gc_ms', or 'nmr')
            strict_validation: Whether to use strict validation
            handle_missing_values: How to handle missing values ('keep', 'fill_zero', 'drop')
        """
        super().__init__(name="MetabolomicsAdapter")

        if data_type not in ["lc_ms", "gc_ms", "nmr"]:
            raise ValueError(
                f"Unknown data_type: {data_type}. Must be 'lc_ms', 'gc_ms', or 'nmr'"
            )

        self.data_type = data_type
        self.strict_validation = strict_validation
        self.handle_missing_values = handle_missing_values

        # Create validator for metabolomics data
        self.validator = MetabolomicsSchema.create_validator(strict=strict_validation)

        # Get QC thresholds
        self.qc_thresholds = MetabolomicsSchema.get_recommended_qc_thresholds()

    def from_source(
        self, source: Union[str, Path, pd.DataFrame], **kwargs
    ) -> anndata.AnnData:
        """
        Convert source data to AnnData with metabolomics schema.

        Args:
            source: Data source (file path, DataFrame, or AnnData)
            **kwargs: Additional parameters:
                - transpose: Whether to transpose matrix (default: False)
                - metabolite_id_col: Column name for metabolite identifiers
                - sample_metadata: Additional sample metadata DataFrame
                - metabolite_metadata: Additional metabolite metadata DataFrame
                - intensity_columns: List of columns containing intensity data
                - missing_value_indicators: List of values to treat as missing

        Returns:
            anndata.AnnData: Loaded and validated data

        Raises:
            ValueError: If source data is invalid
            FileNotFoundError: If source file doesn't exist
        """
        self._log_operation("loading", source=str(source), data_type=self.data_type)

        try:
            # Handle different source types
            if isinstance(source, anndata.AnnData):
                adata = source.copy()
            elif isinstance(source, pd.DataFrame):
                adata = self._create_anndata_from_dataframe(source, **kwargs)
            elif isinstance(source, (str, Path)):
                adata = self._load_from_file(source, **kwargs)
            else:
                raise TypeError(f"Unsupported source type: {type(source)}")

            # Handle missing values according to strategy
            adata = self._handle_missing_values(adata)

            # Add basic metadata
            adata = self._add_basic_metadata(adata, source)

            # Apply metabolomics-specific preprocessing
            adata = self.preprocess_data(adata, **kwargs)

            # Add provenance information
            adata = self.add_provenance(
                adata,
                source_info={
                    "source": str(source),
                    "data_type": self.data_type,
                    "source_type": type(source).__name__,
                },
                processing_params=kwargs,
            )

            self.logger.info(
                f"Loaded metabolomics data: {adata.n_obs} obs x {adata.n_vars} vars"
            )
            return adata

        except Exception as e:
            self.logger.error(f"Failed to load metabolomics data from {source}: {e}")
            raise

    def _load_from_file(self, path: Union[str, Path], **kwargs) -> anndata.AnnData:
        """Load data from file with format detection."""
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        format_type = self.detect_format(path)

        if format_type == "h5ad":
            return self._load_h5ad_data(path)
        elif format_type in ["csv", "tsv", "txt"]:
            return self._load_csv_metabolomics_data(path, **kwargs)
        elif format_type in ["xlsx", "xls", "excel"]:
            return self._load_excel_metabolomics_data(path, **kwargs)
        else:
            raise ValueError(f"Unsupported file format for metabolomics: {format_type}")

    def _load_csv_metabolomics_data(
        self, path: Union[str, Path], **kwargs
    ) -> anndata.AnnData:
        """Load metabolomics data from CSV/TSV with proper handling."""
        # Extract parameters
        transpose = kwargs.get(
            "transpose", False
        )  # Samples as rows by default for metabolomics
        metabolite_id_col = kwargs.get("metabolite_id_col", None)
        intensity_columns = kwargs.get("intensity_columns", None)
        missing_value_indicators = kwargs.get(
            "missing_value_indicators", ["", "NA", "NaN", "NULL"]
        )

        # Load the data
        df = self._load_csv_data(
            path,
            index_col=0 if metabolite_id_col is None else metabolite_id_col,
            na_values=missing_value_indicators,
            **{
                k: v
                for k, v in kwargs.items()
                if k
                not in [
                    "transpose",
                    "metabolite_id_col",
                    "intensity_columns",
                    "missing_value_indicators",
                ]
            },
        )

        # Handle intensity columns selection
        if intensity_columns is not None:
            # Use only specified intensity columns
            metadata_cols = [col for col in df.columns if col not in intensity_columns]
            intensity_df = df[intensity_columns]
            metadata_df = df[metadata_cols] if metadata_cols else None
        else:
            # Try to auto-detect intensity vs metadata columns
            intensity_df, metadata_df = self._separate_intensity_metadata_columns(df)

        # Create metabolite metadata from non-intensity columns
        var_metadata = None
        if metadata_df is not None and len(metadata_df.columns) > 0:
            var_metadata = metadata_df.copy()

            # Standardize common metabolomics metadata column names
            var_metadata = self._standardize_metabolomics_metadata(var_metadata)

        # Create AnnData
        adata = self._create_anndata_from_dataframe(
            intensity_df, var_metadata=var_metadata, transpose=transpose
        )

        return adata

    def _load_excel_metabolomics_data(
        self, path: Union[str, Path], **kwargs
    ) -> anndata.AnnData:
        """Load metabolomics data from Excel file."""
        transpose = kwargs.get("transpose", False)
        sheet_name = kwargs.get("sheet_name", 0)

        df = self._load_excel_data(path, sheet_name=sheet_name, index_col=0)

        # Separate intensity and metadata columns
        intensity_df, metadata_df = self._separate_intensity_metadata_columns(df)

        return self._create_anndata_from_dataframe(
            intensity_df, var_metadata=metadata_df, transpose=transpose
        )

    def _separate_intensity_metadata_columns(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Separate intensity data from metabolite metadata columns.

        Args:
            df: Input DataFrame

        Returns:
            tuple: (intensity_df, metadata_df)
        """
        # Common metabolomics metadata column patterns
        metadata_patterns = [
            "metabolite",
            "compound",
            "mz",
            "m/z",
            "retention_time",
            "rt",
            "adduct",
            "formula",
            "hmdb",
            "kegg",
            "chebi",
            "inchi",
            "smiles",
            "pathway",
            "name",
            "class",
            "subclass",
            "pubchem",
            "identification",
            "annotation",
        ]

        # Identify metadata columns
        metadata_cols = []
        for col in df.columns:
            col_lower = str(col).lower()
            if any(pattern in col_lower for pattern in metadata_patterns):
                metadata_cols.append(col)
            elif df[col].dtype == "object" and not self._is_numeric_string_column(
                df[col]
            ):
                # Non-numeric string columns are likely metadata
                metadata_cols.append(col)

        # Intensity columns are the remaining ones
        intensity_cols = [col for col in df.columns if col not in metadata_cols]

        if not intensity_cols:
            raise ValueError("No intensity columns detected in the data")

        intensity_df = df[intensity_cols].copy()
        metadata_df = df[metadata_cols].copy() if metadata_cols else None

        # Convert intensity columns to numeric, handling missing values
        for col in intensity_cols:
            intensity_df[col] = pd.to_numeric(intensity_df[col], errors="coerce")

        return intensity_df, metadata_df

    def _is_numeric_string_column(self, series: pd.Series) -> bool:
        """Check if a string column contains numeric values."""
        if series.dtype != "object":
            return False

        # Try to convert a sample to numeric
        sample = series.dropna().head(10)
        if len(sample) == 0:
            return False

        try:
            pd.to_numeric(sample, errors="raise")
            return True
        except (ValueError, TypeError):
            return False

    def _standardize_metabolomics_metadata(
        self, metadata_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Standardize metabolomics metadata column names."""

        # Common column name mappings
        column_mappings = {
            # Metabolite identifiers
            "compound": "metabolite_name",
            "compound_name": "metabolite_name",
            "metabolite": "metabolite_name",
            "metabolite_name": "metabolite_name",
            "name": "metabolite_name",
            "metabolite_id": "metabolite_id",
            "id": "metabolite_id",
            # Chemical identifiers
            "hmdb": "hmdb_id",
            "hmdb_id": "hmdb_id",
            "chebi": "chebi_id",
            "chebi_id": "chebi_id",
            "kegg": "kegg_id",
            "kegg_id": "kegg_id",
            "kegg_compound": "kegg_id",
            "pubchem": "pubchem_cid",
            "pubchem_cid": "pubchem_cid",
            "inchi": "inchi",
            "inchikey": "inchikey",
            "smiles": "smiles",
            # Chemical properties
            "formula": "chemical_formula",
            "chemical_formula": "chemical_formula",
            "molecular_formula": "chemical_formula",
            "molecular_weight": "molecular_weight",
            "mw": "molecular_weight",
            "exact_mass": "monoisotopic_mass",
            "monoisotopic_mass": "monoisotopic_mass",
            # MS-specific
            "mz": "mz",
            "m/z": "mz",
            "mass_to_charge": "mz",
            "retention_time": "retention_time",
            "rt": "retention_time",
            "retention_index": "retention_index",
            "ri": "retention_index",
            "adduct": "adduct",
            "ion_mode": "adduct",
            # Quality flags
            "identification_level": "identification_level",
            "msi_level": "identification_level",
            "is_identified": "is_identified",
            "is_internal_standard": "is_internal_standard",
            "is_qc_compound": "is_qc_compound",
            # Biological annotations
            "pathway": "pathways",
            "pathways": "pathways",
            "class": "class",
            "chemical_class": "class",
            "subclass": "subclass",
        }

        # Apply mappings
        renamed_df = metadata_df.copy()
        for old_name, new_name in column_mappings.items():
            matching_cols = [
                col
                for col in renamed_df.columns
                if old_name.lower() in str(col).lower()
            ]
            if matching_cols:
                # Use the first matching column
                renamed_df = renamed_df.rename(columns={matching_cols[0]: new_name})

        return renamed_df

    def _handle_missing_values(self, adata: anndata.AnnData) -> anndata.AnnData:
        """Handle missing values according to the specified strategy."""

        if self.handle_missing_values == "keep":
            # Keep missing values as NaN
            pass
        elif self.handle_missing_values == "fill_zero":
            # Replace NaN with zeros
            if hasattr(adata.X, "isnan"):
                adata.X = np.nan_to_num(adata.X, nan=0.0)
        elif self.handle_missing_values == "drop":
            # Remove observations/variables with too many missing values
            adata = self._drop_high_missing_features(adata)

        return adata

    def _drop_high_missing_features(
        self,
        adata: anndata.AnnData,
        max_missing_obs: float = 0.8,
        max_missing_vars: float = 0.9,
    ) -> anndata.AnnData:
        """Drop observations and variables with high missing value rates."""

        if not hasattr(adata.X, "isnan"):
            return adata

        original_shape = adata.shape

        # Calculate missing rates
        obs_missing_rate = (
            np.array(np.isnan(adata.X).sum(axis=1)).flatten() / adata.n_vars
        )
        vars_missing_rate = (
            np.array(np.isnan(adata.X).sum(axis=0)).flatten() / adata.n_obs
        )

        # Filter observations
        obs_keep = obs_missing_rate <= max_missing_obs
        if obs_keep.sum() < adata.n_obs:
            adata = adata[obs_keep, :].copy()
            self.logger.info(
                f"Removed {(~obs_keep).sum()} observations with >{max_missing_obs * 100}% missing values"
            )

        # Filter variables
        vars_keep = vars_missing_rate <= max_missing_vars
        if vars_keep.sum() < adata.n_vars:
            adata = adata[:, vars_keep].copy()
            self.logger.info(
                f"Removed {(~vars_keep).sum()} metabolites with >{max_missing_vars * 100}% missing values"
            )

        self.logger.info(f"Shape after filtering: {adata.shape} (was {original_shape})")

        return adata

    def validate(self, adata: anndata.AnnData, strict: bool = None) -> ValidationResult:
        """
        Validate AnnData against metabolomics schema.

        Args:
            adata: AnnData object to validate
            strict: Override default strict setting

        Returns:
            ValidationResult: Validation results
        """
        if strict is None:
            strict = self.strict_validation

        # Use the configured validator
        result = self.validator.validate(adata, strict=strict)

        # Add basic structural validation
        basic_result = self._validate_basic_structure(adata)
        result = result.merge(basic_result)

        return result

    def get_schema(self) -> Dict[str, Any]:
        """
        Return the expected schema for this modality.

        Returns:
            Dict[str, Any]: Schema definition
        """
        return MetabolomicsSchema.get_metabolomics_schema()

    def get_supported_formats(self) -> List[str]:
        """
        Get list of supported input formats.

        Returns:
            List[str]: List of supported file extensions
        """
        return ["csv", "tsv", "txt", "xlsx", "xls", "h5ad", "mzML"]

    def preprocess_data(self, adata: anndata.AnnData, **kwargs) -> anndata.AnnData:
        """
        Apply metabolomics-specific preprocessing steps.

        Args:
            adata: Input AnnData object
            **kwargs: Preprocessing parameters

        Returns:
            anndata.AnnData: Preprocessed data object
        """
        # Apply base preprocessing
        adata = super().preprocess_data(adata, **kwargs)

        # Add metabolomics-specific metadata
        adata = self._add_metabolomics_metadata(adata)

        return adata

    def _add_metabolomics_metadata(self, adata: anndata.AnnData) -> anndata.AnnData:
        """Add metabolomics-specific metadata to obs and var."""

        # ---------------------------------------------------------------
        # Per-sample metrics (obs)
        # ---------------------------------------------------------------

        if "n_metabolites" not in adata.obs.columns:
            # Count detected metabolites (non-NaN, non-zero values)
            if hasattr(adata.X, "isnan"):
                adata.obs["n_metabolites"] = np.array(
                    (~np.isnan(adata.X)).sum(axis=1)
                ).flatten()
            else:
                adata.obs["n_metabolites"] = np.array(
                    (adata.X > 0).sum(axis=1)
                ).flatten()

        if "total_intensity" not in adata.obs.columns:
            # Sum of all intensities (excluding NaN)
            adata.obs["total_intensity"] = np.nansum(adata.X, axis=1)

        if "median_intensity" not in adata.obs.columns:
            # Median intensity per sample
            adata.obs["median_intensity"] = np.nanmedian(adata.X, axis=1)

        if "pct_missing" not in adata.obs.columns and hasattr(adata.X, "isnan"):
            # Percentage of missing values per sample
            adata.obs["pct_missing"] = (
                np.array(np.isnan(adata.X).sum(axis=1)).flatten() / adata.n_vars * 100
            )

        # ---------------------------------------------------------------
        # Per-metabolite metrics (var)
        # ---------------------------------------------------------------

        if "n_samples" not in adata.var.columns:
            # Count samples with detection
            if hasattr(adata.X, "isnan"):
                adata.var["n_samples"] = np.array(
                    (~np.isnan(adata.X)).sum(axis=0)
                ).flatten()
            else:
                adata.var["n_samples"] = np.array((adata.X > 0).sum(axis=0)).flatten()

        if "mean_intensity" not in adata.var.columns:
            # Mean intensity per metabolite
            adata.var["mean_intensity"] = np.nanmean(adata.X, axis=0)

        if "prevalence" not in adata.var.columns:
            # Prevalence: proportion of samples with detection (non-NaN, non-zero)
            if hasattr(adata.X, "isnan"):
                detected = (~np.isnan(adata.X)) & (adata.X > 0)
                adata.var["prevalence"] = (
                    np.array(detected.sum(axis=0)).flatten() / adata.n_obs
                )
            else:
                adata.var["prevalence"] = (
                    np.array((adata.X > 0).sum(axis=0)).flatten() / adata.n_obs
                )

        if "cv" not in adata.var.columns:
            # Coefficient of variation (%)
            means = np.nanmean(adata.X, axis=0)
            stds = np.nanstd(adata.X, axis=0)
            cv = stds / means * 100
            adata.var["cv"] = np.nan_to_num(cv, nan=0.0, posinf=0.0, neginf=0.0)

        if "pct_missing" not in adata.var.columns and hasattr(adata.X, "isnan"):
            # Percentage of missing values per metabolite
            adata.var["pct_missing"] = (
                np.array(np.isnan(adata.X).sum(axis=0)).flatten() / adata.n_obs * 100
            )

        return adata

    def get_quality_metrics(self, adata: anndata.AnnData) -> Dict[str, Any]:
        """
        Calculate metabolomics-specific quality metrics.

        Args:
            adata: AnnData object to analyze

        Returns:
            Dict[str, Any]: Quality metrics dictionary
        """
        metrics = super().get_quality_metrics(adata)

        # Add metabolomics-specific metrics
        if hasattr(adata.X, "isnan"):
            total_values = adata.X.size
            missing_values = np.isnan(adata.X).sum()
            metrics["missing_value_percentage"] = float(
                (missing_values / total_values) * 100
            )

        if "n_metabolites" in adata.obs.columns:
            metrics["mean_metabolites_per_sample"] = float(
                adata.obs["n_metabolites"].mean()
            )
            metrics["median_metabolites_per_sample"] = float(
                adata.obs["n_metabolites"].median()
            )

        if "n_samples" in adata.var.columns:
            metrics["mean_samples_per_metabolite"] = float(
                adata.var["n_samples"].mean()
            )

        if "prevalence" in adata.var.columns:
            metrics["mean_prevalence"] = float(adata.var["prevalence"].mean())
            metrics["low_prevalence_metabolites"] = int(
                (adata.var["prevalence"] < 0.1).sum()
            )

        if "cv" in adata.var.columns:
            metrics["median_cv"] = float(adata.var["cv"].median())
            metrics["high_cv_metabolites"] = int((adata.var["cv"] > 50).sum())

        # Data type info
        metrics["data_type"] = self.data_type

        return metrics

    def detect_data_type(self, adata: anndata.AnnData) -> str:
        """
        Detect whether data is LC-MS, GC-MS, or NMR metabolomics.

        Uses heuristics based on metadata fields and data characteristics
        to determine the most likely analytical platform.

        Args:
            adata: AnnData object to analyze

        Returns:
            str: Detected data type ('lc_ms', 'gc_ms', or 'nmr')
        """
        # Check for platform-specific metadata indicators

        # LC-MS indicators
        lc_ms_indicators = [
            "retention_time",
            "mz",
            "adduct",
            "ionization_mode",
        ]
        lc_ms_score = sum(
            1 for indicator in lc_ms_indicators if indicator in adata.var.columns
        )

        # GC-MS indicators
        gc_ms_indicators = [
            "retention_index",
            "derivatization",
        ]
        gc_ms_score = sum(
            1 for indicator in gc_ms_indicators if indicator in adata.var.columns
        )
        # Check obs-level derivatization too
        if "derivatization" in adata.obs.columns:
            gc_ms_score += 1

        # NMR indicators
        nmr_indicators = [
            "chemical_shift",
            "ppm",
            "peak_width",
            "multiplicity",
        ]
        nmr_score = sum(
            1 for indicator in nmr_indicators if indicator in adata.var.columns
        )

        # Check uns-level platform information
        if "platform" in adata.obs.columns:
            platform_values = adata.obs["platform"].dropna().unique()
            for val in platform_values:
                val_lower = str(val).lower().replace("-", "_").replace(" ", "_")
                if "lc_ms" in val_lower:
                    lc_ms_score += 2
                elif "gc_ms" in val_lower:
                    gc_ms_score += 2
                elif "nmr" in val_lower:
                    nmr_score += 2

        # Data characteristics heuristic
        n_vars = adata.n_vars

        # NMR typically has many spectral bins (thousands)
        if nmr_score > lc_ms_score and nmr_score > gc_ms_score:
            return "nmr"

        # GC-MS typically has fewer features than LC-MS
        if gc_ms_score > lc_ms_score:
            return "gc_ms"

        # LC-MS is the most common platform
        if lc_ms_score > 0:
            return "lc_ms"

        # Fallback heuristic based on feature count
        if n_vars > 5000:
            # Very high feature count suggests NMR spectral bins
            return "nmr"
        elif n_vars < 200:
            # Lower feature count could be targeted GC-MS
            return "gc_ms"

        # Default to current setting if unclear
        return self.data_type


# =============================================================================
# Factory functions for entry-point registration
# =============================================================================


def create_lc_ms_adapter() -> MetabolomicsAdapter:
    """Create an LC-MS metabolomics adapter."""
    return MetabolomicsAdapter(data_type="lc_ms")


def create_gc_ms_adapter() -> MetabolomicsAdapter:
    """Create a GC-MS metabolomics adapter."""
    return MetabolomicsAdapter(data_type="gc_ms")


def create_nmr_adapter() -> MetabolomicsAdapter:
    """Create an NMR metabolomics adapter."""
    return MetabolomicsAdapter(data_type="nmr")
