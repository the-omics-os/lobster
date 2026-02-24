"""
Luminex MFI parser for affinity proteomics data.

This parser handles Luminex multiplex immunoassay data, supporting
Bio-Plex Manager CSV export format and general MFI (Median Fluorescence
Intensity) data files.

Supported formats:
- Bio-Plex Manager CSV export (long or wide format)
- General Luminex CSV/XLSX with MFI values

MFI Scale: Linear median fluorescence intensity
Typical range: 0 to 30,000+ MFI
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import anndata
import numpy as np
import pandas as pd

from lobster.services.data_access.proteomics_parsers.base_parser import (
    FileValidationError,
    ParsingError,
    ProteomicsParser,
)
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


class LuminexParser(ProteomicsParser):
    """
    Parser for Luminex MFI data (Bio-Plex Manager CSV export format).

    Luminex is a bead-based multiplex immunoassay platform for targeted
    protein quantification. Data is reported as MFI (Median Fluorescence
    Intensity) values.

    Key Features:
    - Auto-detects long vs wide CSV format
    - Supports Bio-Plex Manager export with metadata header rows
    - Flexible column mapping for non-standard formats
    - Handles multiple value types (Median, Mean, Net MFI, Count)
    - Supports both CSV and XLSX input

    AnnData Structure:
    - X: MFI matrix (samples x analytes), linear scale
    - obs: sample metadata (sample IDs, any available sample info)
    - var: analyte metadata (analyte names as index)
    - uns: parser metadata, MFI type, platform info
    """

    # Indicators for Luminex format detection
    LUMINEX_INDICATORS = [
        "Median",
        "MFI",
        "Bio-Plex",
        "Luminex",
        "MAGPIX",
        "Net MFI",
        "Mean",
        "xMAP",
        "FLEXMAP",
    ]

    def __init__(self):
        """Initialize Luminex parser."""
        super().__init__(name="Luminex", version="1.0.0")

    def get_supported_formats(self) -> List[str]:
        """Return supported file extensions for Luminex data."""
        return [".csv", ".xlsx"]

    def validate_file(self, file_path: str) -> bool:
        """
        Validate if file is Luminex MFI format.

        Checks for:
        1. Correct file extension (.csv, .xlsx)
        2. Luminex indicators in header or metadata rows

        Args:
            file_path: Path to file

        Returns:
            bool: True if valid Luminex file
        """
        try:
            path = self._validate_file_exists(file_path)

            if not self._validate_extension(file_path):
                return False

            # Read first few rows to check for Luminex indicators
            if path.suffix.lower() == ".xlsx":
                try:
                    df_header = pd.read_excel(file_path, nrows=20, header=None)
                except Exception:
                    return False
            else:
                try:
                    df_header = pd.read_csv(file_path, nrows=20, header=None)
                except Exception:
                    return False

            # Check all cells in first 20 rows for Luminex indicators
            text_content = df_header.to_string()
            for indicator in self.LUMINEX_INDICATORS:
                if indicator.lower() in text_content.lower():
                    logger.debug(
                        f"Valid Luminex file detected: {path.name} (found '{indicator}')"
                    )
                    return True

            logger.debug(f"File does not appear to be Luminex format: {path.name}")
            return False

        except Exception as e:
            logger.debug(f"Luminex file validation failed: {e}")
            return False

    def parse(
        self,
        file_path: str,
        analyte_column: str = "Analyte",
        value_column: str = "Median",
        sample_column: str = "Sample",
        skip_metadata_rows: bool = True,
        **kwargs,
    ) -> Tuple[anndata.AnnData, Dict[str, Any]]:
        """
        Parse Luminex MFI file to AnnData.

        Auto-detects whether data is in long format (one row per measurement)
        or wide format (samples as rows, analytes as columns).

        Args:
            file_path: Path to Luminex file
            analyte_column: Column name for analyte identifiers (long format)
            value_column: Column name for MFI values ("Median", "Mean", "Net MFI")
            sample_column: Column name for sample identifiers (long format)
            skip_metadata_rows: Skip Bio-Plex metadata rows at top of file

        Returns:
            Tuple[AnnData, Dict]: (parsed data, statistics)
        """
        logger.info(f"Parsing Luminex MFI file: {file_path}")

        if not self.validate_file(file_path):
            raise FileValidationError(f"Invalid Luminex format: {file_path}")

        try:
            path = Path(file_path)

            # Read raw data, handling metadata rows
            df = self._read_luminex_file(path, skip_metadata_rows)

            if df.empty:
                raise ParsingError("No data found in Luminex file")

            logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")

            # Detect format: long vs wide
            is_long = self._detect_long_format(
                df, analyte_column, value_column, sample_column
            )

            if is_long:
                adata, stats = self._parse_long_format(
                    df, analyte_column, value_column, sample_column
                )
            else:
                adata, stats = self._parse_wide_format(df)

            # Add parser metadata
            adata.uns["parser"] = {
                "name": self.name,
                "version": self.version,
                "source_file": path.name,
                "format": "long" if is_long else "wide",
                "intensity_type": "mfi",
                "platform": "luminex",
            }
            adata.uns["platform"] = "luminex"
            adata.uns["mfi_type"] = value_column

            logger.info(f"Successfully parsed Luminex file: {adata.shape}")
            return adata, stats

        except FileValidationError:
            raise
        except Exception as e:
            logger.error(f"Failed to parse Luminex file: {e}")
            raise ParsingError(f"Failed to parse Luminex file: {e}") from e

    # =========================================================================
    # FILE READING
    # =========================================================================

    def _read_luminex_file(self, path: Path, skip_metadata_rows: bool) -> pd.DataFrame:
        """
        Read Luminex file, handling metadata header rows.

        Bio-Plex Manager exports often have metadata rows before the actual
        data table. This method detects and skips them.
        """
        if path.suffix.lower() == ".xlsx":
            df = pd.read_excel(path, header=None)
        else:
            df = pd.read_csv(path, header=None)

        if not skip_metadata_rows:
            # Use first row as header
            df.columns = df.iloc[0].astype(str)
            df = df.iloc[1:].reset_index(drop=True)
            return df

        # Find the header row by looking for the row with the most non-null,
        # non-numeric values (column names)
        header_row_idx = self._find_header_row(df)

        if header_row_idx is not None:
            # Set that row as header
            df.columns = df.iloc[header_row_idx].astype(str)
            df = df.iloc[header_row_idx + 1 :].reset_index(drop=True)
        else:
            # Fall back to first row
            df.columns = df.iloc[0].astype(str)
            df = df.iloc[1:].reset_index(drop=True)

        # Clean column names
        df.columns = [str(c).strip() for c in df.columns]

        return df

    def _find_header_row(self, df: pd.DataFrame) -> Optional[int]:
        """
        Find the header row in a Bio-Plex export.

        The header row typically contains column names like 'Sample', 'Analyte',
        'Median', etc. Metadata rows above it contain instrument info.
        """
        for idx in range(min(30, len(df))):
            row = df.iloc[idx]
            row_str = " ".join(str(v) for v in row if pd.notna(v))

            # Check if this row looks like a header
            header_indicators = [
                "sample",
                "analyte",
                "median",
                "mean",
                "mfi",
                "concentration",
                "standard",
                "net mfi",
            ]
            matches = sum(1 for ind in header_indicators if ind in row_str.lower())
            if matches >= 2:
                return idx

        return None

    # =========================================================================
    # FORMAT DETECTION
    # =========================================================================

    def _detect_long_format(
        self,
        df: pd.DataFrame,
        analyte_column: str,
        value_column: str,
        sample_column: str,
    ) -> bool:
        """
        Detect if data is in long format (one row per sample-analyte measurement).

        Long format indicators:
        - Has analyte_column AND value_column AND sample_column
        - Relatively few unique columns but many rows

        Wide format indicators:
        - Many columns (each is an analyte)
        - Each row is a sample
        """
        columns_lower = {c.lower(): c for c in df.columns}

        has_analyte = analyte_column.lower() in columns_lower
        has_value = value_column.lower() in columns_lower
        has_sample = sample_column.lower() in columns_lower

        # If all three key columns present, it's long format
        if has_analyte and has_value and has_sample:
            return True

        # Check for alternative column names
        alt_analyte_names = ["analyte", "assay", "target", "protein", "bead region"]
        alt_value_names = ["median", "mean", "net mfi", "mfi", "fi"]
        alt_sample_names = ["sample", "sample id", "sampleid", "well", "location"]

        has_alt_analyte = any(name in columns_lower for name in alt_analyte_names)
        has_alt_value = any(name in columns_lower for name in alt_value_names)
        has_alt_sample = any(name in columns_lower for name in alt_sample_names)

        if has_alt_analyte and has_alt_value and has_alt_sample:
            return True

        # Heuristic: wide format has many columns (>10) that look like analyte names
        if len(df.columns) > 10:
            # Check if most columns are numeric (wide format with analyte columns)
            numeric_cols = 0
            for col in df.columns:
                try:
                    pd.to_numeric(df[col], errors="raise")
                    numeric_cols += 1
                except (ValueError, TypeError):
                    pass

            # If most columns are numeric, likely wide format
            if numeric_cols > len(df.columns) * 0.5:
                return False

        return False

    # =========================================================================
    # LONG FORMAT PARSING
    # =========================================================================

    def _parse_long_format(
        self,
        df: pd.DataFrame,
        analyte_column: str,
        value_column: str,
        sample_column: str,
    ) -> Tuple[anndata.AnnData, Dict[str, Any]]:
        """Parse long-format Luminex data (one row per measurement)."""
        logger.info("Parsing long-format Luminex data")

        # Resolve actual column names (case-insensitive)
        actual_analyte = self._find_column(df, analyte_column)
        actual_value = self._find_column(df, value_column)
        actual_sample = self._find_column(df, sample_column)

        if not actual_analyte or not actual_value or not actual_sample:
            raise ParsingError(
                f"Required columns not found. Need: analyte ({actual_analyte}), "
                f"value ({actual_value}), sample ({actual_sample}). "
                f"Available: {list(df.columns)}"
            )

        # Convert value column to numeric
        df[actual_value] = pd.to_numeric(df[actual_value], errors="coerce")

        # Get unique samples and analytes
        samples = df[actual_sample].unique()
        analytes = df[actual_analyte].unique()

        # Remove NaN from sample/analyte lists
        samples = [s for s in samples if pd.notna(s)]
        analytes = [a for a in analytes if pd.notna(a)]

        n_samples = len(samples)
        n_analytes = len(analytes)

        logger.info(f"Long format: {n_samples} samples x {n_analytes} analytes")

        # Pivot to wide format
        pivot_df = df.pivot_table(
            index=actual_sample,
            columns=actual_analyte,
            values=actual_value,
            aggfunc="first",  # Take first value if duplicates
        )

        # Reindex to ensure consistent order
        pivot_df = pivot_df.reindex(index=samples, columns=analytes)

        mfi_matrix = pivot_df.values.astype(np.float32)
        sample_order = [str(s) for s in pivot_df.index.tolist()]
        analyte_order = [str(a) for a in pivot_df.columns.tolist()]

        # Build obs (sample metadata)
        obs_data = {"sample_id": sample_order}

        # Add additional sample-level columns if available
        sample_meta_cols = [
            c
            for c in df.columns
            if c not in [actual_analyte, actual_value] and c != actual_sample
        ]
        if sample_meta_cols:
            sample_meta = df.groupby(actual_sample).first().reindex(samples)
            for col in sample_meta_cols[:10]:  # Limit to 10 metadata columns
                if col in sample_meta.columns:
                    obs_data[col.lower().replace(" ", "_")] = (
                        sample_meta[col].astype(str).values
                    )

        obs_df = pd.DataFrame(obs_data, index=sample_order)

        # Build var (analyte metadata)
        var_df = pd.DataFrame({"analyte": analyte_order}, index=analyte_order)
        var_df.index.name = None

        # Create AnnData
        adata = anndata.AnnData(X=mfi_matrix, obs=obs_df, var=var_df)

        # Calculate statistics
        missing_count = np.isnan(mfi_matrix).sum()
        total_values = mfi_matrix.size
        missing_pct = (missing_count / total_values) * 100 if total_values > 0 else 0

        valid_values = mfi_matrix[~np.isnan(mfi_matrix)]
        mfi_range = (
            (float(np.min(valid_values)), float(np.max(valid_values)))
            if len(valid_values) > 0
            else (0.0, 0.0)
        )

        stats = self._create_base_stats(
            n_samples=n_samples,
            n_proteins=n_analytes,
            n_proteins_raw=n_analytes,
            n_samples_raw=n_samples,
            missing_percentage=missing_pct,
            format="long",
            parser="Luminex",
            mfi_type=value_column,
            n_missing=int(missing_count),
            missing_rate=round(missing_pct / 100, 4),
            mfi_range=mfi_range,
        )

        return adata, stats

    # =========================================================================
    # WIDE FORMAT PARSING
    # =========================================================================

    def _parse_wide_format(
        self, df: pd.DataFrame
    ) -> Tuple[anndata.AnnData, Dict[str, Any]]:
        """Parse wide-format Luminex data (samples as rows, analytes as columns)."""
        logger.info("Parsing wide-format Luminex data")

        # First column should be sample IDs
        # Identify which columns are numeric (analyte data) vs metadata
        numeric_cols = []
        meta_cols = []

        for col in df.columns:
            try:
                pd.to_numeric(df[col], errors="raise")
                numeric_cols.append(col)
            except (ValueError, TypeError):
                meta_cols.append(col)

        if not numeric_cols:
            raise ParsingError(
                "No numeric columns found for MFI values in wide format. "
                f"Available columns: {list(df.columns)}"
            )

        # Use first meta column as sample ID
        if meta_cols:
            sample_col = meta_cols[0]
            sample_ids = df[sample_col].astype(str).tolist()
        else:
            sample_ids = [f"Sample_{i + 1}" for i in range(len(df))]

        analyte_names = numeric_cols
        mfi_matrix = df[numeric_cols].values.astype(np.float32)

        n_samples = len(sample_ids)
        n_analytes = len(analyte_names)

        logger.info(f"Wide format: {n_samples} samples x {n_analytes} analytes")

        # Build obs
        obs_data = {"sample_id": sample_ids}
        for col in meta_cols[1:5]:  # Add up to 4 additional metadata columns
            obs_data[col.lower().replace(" ", "_")] = df[col].astype(str).tolist()

        obs_df = pd.DataFrame(obs_data, index=sample_ids)

        # Build var
        var_df = pd.DataFrame({"analyte": analyte_names}, index=analyte_names)
        var_df.index.name = None

        # Create AnnData
        adata = anndata.AnnData(X=mfi_matrix, obs=obs_df, var=var_df)

        # Calculate statistics
        missing_count = np.isnan(mfi_matrix).sum()
        total_values = mfi_matrix.size
        missing_pct = (missing_count / total_values) * 100 if total_values > 0 else 0

        valid_values = mfi_matrix[~np.isnan(mfi_matrix)]
        mfi_range = (
            (float(np.min(valid_values)), float(np.max(valid_values)))
            if len(valid_values) > 0
            else (0.0, 0.0)
        )

        stats = self._create_base_stats(
            n_samples=n_samples,
            n_proteins=n_analytes,
            n_proteins_raw=n_analytes,
            n_samples_raw=n_samples,
            missing_percentage=missing_pct,
            format="wide",
            parser="Luminex",
            mfi_type="auto_detected",
            n_missing=int(missing_count),
            missing_rate=round(missing_pct / 100, 4),
            mfi_range=mfi_range,
        )

        return adata, stats

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    @staticmethod
    def _find_column(df: pd.DataFrame, target: str) -> Optional[str]:
        """Find column by case-insensitive matching."""
        target_lower = target.lower()
        for col in df.columns:
            if col.lower() == target_lower:
                return col

        # Try partial matching for common alternatives
        alt_names = {
            "analyte": ["analyte", "assay", "target", "protein", "bead region"],
            "median": ["median", "mean", "net mfi", "mfi", "fi"],
            "sample": ["sample", "sample id", "sampleid", "well", "location"],
        }

        if target_lower in alt_names:
            for alt in alt_names[target_lower]:
                for col in df.columns:
                    if col.lower() == alt:
                        return col

        return None
