"""
Spectronaut output parser for DIA proteomics data.

This module provides a production-ready parser for Spectronaut report files. Spectronaut
is a commercial software package from Biognosys AG for analyzing DIA (Data-Independent
Acquisition) mass spectrometry data with industry-leading performance.

Key features:
- Supports both long-format and matrix-format Spectronaut exports
- Automatic format detection based on column structure
- Q-value based filtering at protein group level
- Log2 transformation with configurable pseudocount
- Protein group handling (semicolon-separated IDs)
- Gene symbol indexing for biologist-friendly output

Reference:
    Bruderer, R., Bernhardt, O. M., Gandhi, T., Miber, S. M., Selevsek, N.,
    Reiter, L., et al. (2015). Extending the limits of quantitative proteome
    profiling with data-independent acquisition and application to
    acetaminophen-treated three-dimensional liver microtissues.
    Molecular & Cellular Proteomics, 14(5), 1400-1410.

Example usage:
    >>> from lobster.services.data_access.proteomics_parsers import SpectronautParser
    >>> parser = SpectronautParser()
    >>> adata, stats = parser.parse(
    ...     "spectronaut_report.tsv",
    ...     qvalue_threshold=0.01,
    ...     log_transform=True
    ... )
    >>> print(f"Loaded {adata.n_obs} samples x {adata.n_vars} proteins")
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import anndata
import numpy as np
import pandas as pd

from lobster.core.analysis_ir import AnalysisStep
from lobster.services.data_access.proteomics_parsers.base_parser import (
    FileValidationError,
    ParsingError,
    ProteomicsParser,
)
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


class SpectronautParser(ProteomicsParser):
    """
    Parser for Spectronaut DIA-MS report files.

    Spectronaut is a leading commercial DIA analysis software from Biognosys.
    This parser handles both common export formats:
    1. Long format: One row per precursor/protein per sample
    2. Matrix format: Proteins as rows, samples as columns

    Key Spectronaut Column Patterns:
        Long Format:
        - R.FileName or Run: Sample identifier (strip .raw/.d/.wiff extensions)
        - PG.ProteinGroups: Protein IDs (semicolon-separated for protein groups)
        - PG.Genes: Gene symbols (semicolon-separated)
        - PG.Quantity: Protein group abundance (LINEAR scale, NOT log2)
        - PG.Qvalue: Statistical confidence (FDR at protein group level)
        - EG.Qvalue: Precursor-level Q-value (optional)

        Matrix Format:
        - First column: Protein identifiers (PG.ProteinGroups or similar)
        - Gene column: PG.Genes or Genes
        - Remaining columns: Sample names with abundance values

    AnnData Output Structure:
        - X: intensity matrix (samples x proteins), NaN for missing values
        - obs: sample metadata (sample_name derived from column headers/R.FileName)
        - var: protein metadata including:
            - protein_groups: Full semicolon-separated protein group string
            - protein_id: First/representative protein ID
            - gene_symbols: Full semicolon-separated gene symbol string
            - n_precursors: Number of precursors (long format only)
            - mean_q_value: Mean Q-value for the protein group
            - is_contaminant: Whether protein is a contaminant
            - is_reverse: Whether protein is a reverse/decoy hit
        - uns: parsing metadata (format type, Q-value threshold, etc.)

    Attributes:
        LONG_FORMAT_COLUMNS: Required columns for long format
        MATRIX_FORMAT_INDICATORS: Columns that indicate matrix format
        CONTAMINANT_PATTERNS: Patterns for identifying contaminant proteins
        REVERSE_PATTERNS: Patterns for identifying decoy/reverse hits
    """

    # Required columns for long format Spectronaut report
    LONG_FORMAT_COLUMNS = {
        "PG.ProteinGroups",  # Protein group identifier
    }

    # One of these must be present for long format
    SAMPLE_COLUMNS = ["R.FileName", "Run", "File.Name"]

    # One of these should be present for quantity
    QUANTITY_COLUMNS = ["PG.Quantity", "PG.Normalised", "PG.NormalizedQuantity"]

    # Columns that indicate this is a Spectronaut file
    SPECTRONAUT_INDICATORS = {
        "PG.ProteinGroups",
        "PG.Genes",
        "PG.Quantity",
        "PG.Qvalue",
        "EG.Qvalue",
        "R.FileName",
    }

    # Patterns for identifying contaminant proteins
    CONTAMINANT_PATTERNS = [
        r"^CON__",
        r"^CONT__",
        r"^contaminant",
        r"KERATIN",
        r"TRYP_",
        r"_CONTAMINANT",
    ]

    # Patterns for identifying reverse/decoy hits
    REVERSE_PATTERNS = [
        r"^REV__",
        r"^REVERSE_",
        r"^rev_",
        r"_REVERSE",
        r"_rev$",
        r"^decoy",
    ]

    def __init__(self):
        """Initialize Spectronaut parser."""
        super().__init__(name="Spectronaut", version="1.0.0")

    def get_supported_formats(self) -> List[str]:
        """
        Return list of supported file extensions.

        Returns:
            List[str]: Supported extensions for Spectronaut output
        """
        return [".tsv", ".txt", ".csv", ".xls", ".xlsx"]

    def validate_file(self, file_path: str) -> bool:
        """
        Check if file is valid Spectronaut report format.

        Validates:
        1. File exists and has correct extension
        2. Contains Spectronaut-specific columns (PG.* patterns)
        3. Contains either long-format or matrix-format structure

        Args:
            file_path: Path to the file to validate

        Returns:
            bool: True if file is valid Spectronaut format
        """
        try:
            path = self._validate_file_exists(file_path)

            if not self._validate_extension(file_path):
                logger.debug(f"File extension not supported: {path.suffix}")
                return False

            # Read header only
            df_header = self._read_header(file_path)
            columns = set(df_header.columns)

            # Check for Spectronaut-specific column patterns
            spectronaut_cols = [
                col for col in columns if col.startswith(("PG.", "EG.", "R.", "FG."))
            ]

            if spectronaut_cols:
                logger.debug(
                    f"Valid Spectronaut file: found {len(spectronaut_cols)} Spectronaut columns"
                )
                return True

            # Check for matrix format with PG.ProteinGroups-like first column
            first_col = df_header.columns[0] if len(df_header.columns) > 0 else ""
            if "protein" in first_col.lower() and "group" in first_col.lower():
                logger.debug("Valid Spectronaut matrix format file")
                return True

            logger.debug("No Spectronaut-specific columns found")
            return False

        except Exception as e:
            logger.debug(f"File validation failed: {e}")
            return False

    def _read_header(self, file_path: str) -> pd.DataFrame:
        """
        Read file header to detect format.

        Args:
            file_path: Path to the file

        Returns:
            pd.DataFrame: DataFrame with just the header row
        """
        path = Path(file_path)

        if path.suffix.lower() in [".xlsx", ".xls"]:
            return pd.read_excel(file_path, nrows=0)
        elif path.suffix.lower() == ".csv":
            return pd.read_csv(file_path, nrows=0)
        else:
            # Try tab-separated first
            try:
                df = pd.read_csv(file_path, sep="\t", nrows=0)
                if len(df.columns) > 1:
                    return df
            except Exception:
                pass
            # Fall back to comma-separated
            return pd.read_csv(file_path, nrows=0)

    def _detect_format(self, df: pd.DataFrame) -> str:
        """
        Detect whether data is in long or matrix format.

        Long format: Multiple rows per protein (one per sample)
        Matrix format: One row per protein, samples as columns

        Args:
            df: Input DataFrame

        Returns:
            str: "long" or "matrix"
        """
        columns = set(df.columns)

        # Check for long format indicators
        has_sample_col = any(col in columns for col in self.SAMPLE_COLUMNS)
        has_pg_columns = any(col.startswith("PG.") for col in columns)

        if has_sample_col and has_pg_columns:
            # Verify it's actually long format by checking for repeated proteins
            if "PG.ProteinGroups" in columns:
                n_unique_proteins = df["PG.ProteinGroups"].nunique()
                if (
                    n_unique_proteins < len(df) * 0.8
                ):  # Significantly fewer unique proteins
                    logger.info("Detected Spectronaut long format")
                    return "long"

        # Default to matrix if first column looks like protein identifiers
        # and other columns look like sample names
        first_col = df.columns[0]
        if "protein" in first_col.lower() or first_col.startswith("PG."):
            # Check if other columns look like numeric data
            numeric_cols = df.iloc[:, 1:].select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                logger.info("Detected Spectronaut matrix format")
                return "matrix"

        # Default to long format
        logger.info("Defaulting to Spectronaut long format")
        return "long"

    def _read_file(self, file_path: str) -> pd.DataFrame:
        """
        Read Spectronaut file with appropriate separator detection.

        Args:
            file_path: Path to the file

        Returns:
            pd.DataFrame: Loaded DataFrame
        """
        path = Path(file_path)

        if path.suffix.lower() in [".xlsx", ".xls"]:
            return pd.read_excel(file_path)
        elif path.suffix.lower() == ".csv":
            return pd.read_csv(file_path, low_memory=False)
        else:
            # Try tab-separated first (most common for Spectronaut)
            try:
                df = pd.read_csv(file_path, sep="\t", low_memory=False)
                if len(df.columns) > 1:
                    return df
            except Exception:
                pass
            # Fall back to comma-separated
            return pd.read_csv(file_path, low_memory=False)

    def parse(
        self,
        file_path: str,
        qvalue_threshold: float = 0.01,
        quantity_column: str = "auto",
        log_transform: bool = True,
        pseudocount: float = 1.0,
        use_genes_as_index: bool = True,
        filter_contaminants: bool = True,
        filter_reverse: bool = True,
        aggregation_method: str = "sum",
    ) -> Tuple[anndata.AnnData, Dict[str, Any], AnalysisStep]:
        """
        Parse Spectronaut report file into AnnData with provenance tracking.

        Args:
            file_path: Path to Spectronaut report file (.tsv, .csv, .xlsx)
            qvalue_threshold: Maximum Q-value for protein group filtering
                (default 0.01 = 1% FDR, Biognosys standard)
            quantity_column: Quantification column to use:
                - "auto": Automatically detect (prefer PG.Quantity)
                - "PG.Quantity": Raw quantity values
                - "PG.Normalised": Normalized quantity values
            log_transform: Apply log2 transformation to intensities.
                Spectronaut reports LINEAR scale values, so log2
                transformation is typically needed for analysis.
            pseudocount: Value to add before log2 transformation to
                handle zeros (default: 1.0)
            use_genes_as_index: Use gene symbols as var index if available.
                More biologist-friendly than protein IDs.
            filter_contaminants: Remove proteins marked as contaminants
            filter_reverse: Remove reverse/decoy database hits
            aggregation_method: Method to aggregate precursors to protein level
                (for long format only): "sum", "mean", "median", "max"

        Returns:
            Tuple containing:
                - anndata.AnnData: Parsed proteomics data
                - Dict[str, Any]: Parsing statistics
                - AnalysisStep: Provenance IR for notebook export

        Raises:
            FileValidationError: If file format is invalid
            ParsingError: If parsing fails
            FileNotFoundError: If file does not exist
        """
        logger.info(f"Parsing Spectronaut file: {file_path}")

        # Validate file
        if not self.validate_file(file_path):
            raise FileValidationError(
                f"Invalid Spectronaut report format: {file_path}. "
                "Expected columns like PG.ProteinGroups, PG.Quantity, R.FileName"
            )

        try:
            # Read file
            df = self._read_file(file_path)
            n_rows_raw = len(df)
            logger.info(f"Loaded {n_rows_raw} rows from file")

            # Detect format
            format_type = self._detect_format(df)

            # Parse based on format
            if format_type == "long":
                adata, filter_stats = self._parse_long_format(
                    df,
                    qvalue_threshold=qvalue_threshold,
                    quantity_column=quantity_column,
                    aggregation_method=aggregation_method,
                )
            else:
                adata, filter_stats = self._parse_matrix_format(
                    df, qvalue_threshold=qvalue_threshold
                )

            n_proteins_raw = adata.n_vars

            # Apply log2 transformation if requested
            if log_transform:
                # Add pseudocount before log transformation
                adata.X = np.log2(adata.X + pseudocount)
                logger.info(f"Applied log2 transformation (pseudocount={pseudocount})")

            # Flag contaminants and reverse hits
            adata = self._flag_special_proteins(adata)

            # Apply filters
            if filter_contaminants and "is_contaminant" in adata.var.columns:
                mask = ~adata.var["is_contaminant"]
                n_removed = (~mask).sum()
                filter_stats["n_contaminants"] = int(n_removed)
                adata = adata[:, mask].copy()
                if n_removed > 0:
                    logger.info(f"Removed {n_removed} contaminants")

            if filter_reverse and "is_reverse" in adata.var.columns:
                mask = ~adata.var["is_reverse"]
                n_removed = (~mask).sum()
                filter_stats["n_reverse"] = int(n_removed)
                adata = adata[:, mask].copy()
                if n_removed > 0:
                    logger.info(f"Removed {n_removed} reverse hits")

            # Set gene symbols as index if requested
            if use_genes_as_index and "gene_symbols" in adata.var.columns:
                adata = self._set_gene_index(adata)

            # Store metadata
            adata.uns["parser"] = {
                "name": self.name,
                "version": self.version,
                "source_file": str(Path(file_path).name),
                "format_type": format_type,
                "qvalue_threshold": qvalue_threshold,
                "log_transformed": log_transform,
                "pseudocount": pseudocount if log_transform else None,
            }

            # Calculate missing value statistics
            total_values = adata.X.size
            missing_values = np.isnan(adata.X).sum()
            missing_percentage = (
                (missing_values / total_values) * 100 if total_values > 0 else 0
            )

            # Build statistics
            stats = self._create_base_stats(
                n_samples=adata.n_obs,
                n_proteins=adata.n_vars,
                n_proteins_raw=n_proteins_raw,
                missing_percentage=missing_percentage,
                format_type=format_type,
                qvalue_threshold=qvalue_threshold,
                log_transformed=log_transform,
                n_rows_raw=n_rows_raw,
                **filter_stats,
            )

            logger.info(
                f"Parsing complete: {adata.n_obs} samples x {adata.n_vars} proteins "
                f"({missing_percentage:.1f}% missing)"
            )

            # Create provenance IR for notebook export
            ir = self._create_ir(
                file_path=file_path,
                qvalue_threshold=qvalue_threshold,
                quantity_column=quantity_column,
                log_transform=log_transform,
                pseudocount=pseudocount,
                use_genes_as_index=use_genes_as_index,
                filter_contaminants=filter_contaminants,
                filter_reverse=filter_reverse,
                aggregation_method=aggregation_method,
                format_type=format_type,
            )

            return adata, stats, ir

        except FileValidationError:
            raise
        except Exception as e:
            logger.error(f"Failed to parse Spectronaut file: {e}")
            raise ParsingError(f"Failed to parse Spectronaut file: {e}") from e

    def _create_ir(
        self,
        file_path: str,
        qvalue_threshold: float,
        quantity_column: str,
        log_transform: bool,
        pseudocount: float,
        use_genes_as_index: bool,
        filter_contaminants: bool,
        filter_reverse: bool,
        aggregation_method: str,
        format_type: str,
    ) -> AnalysisStep:
        """
        Create AnalysisStep IR for provenance tracking and notebook export.

        This method generates the intermediate representation (IR) that enables:
        - W3C-PROV compliant provenance tracking
        - Reproducible Jupyter notebook generation via /pipeline export
        - Parameter documentation for audit trails

        Args:
            file_path: Path to the parsed Spectronaut file
            qvalue_threshold: Q-value threshold used for filtering
            quantity_column: Quantity column used (or "auto")
            log_transform: Whether log2 transformation was applied
            pseudocount: Pseudocount used for log transformation
            use_genes_as_index: Whether gene symbols were used as index
            filter_contaminants: Whether contaminants were filtered
            filter_reverse: Whether reverse hits were filtered
            aggregation_method: Aggregation method used (long format)
            format_type: Detected format type ("long" or "matrix")

        Returns:
            AnalysisStep: Provenance IR for this parsing operation
        """
        # Build parameter dictionary for IR
        parameters = {
            "qvalue_threshold": qvalue_threshold,
            "quantity_column": quantity_column,
            "log_transform": log_transform,
            "pseudocount": pseudocount,
            "use_genes_as_index": use_genes_as_index,
            "filter_contaminants": filter_contaminants,
            "filter_reverse": filter_reverse,
            "aggregation_method": aggregation_method,
        }

        # Generate reproducible code template
        code_template = """# Parse Spectronaut DIA-MS proteomics data
from lobster.services.data_access.proteomics_parsers import SpectronautParser

parser = SpectronautParser()
adata, stats, ir = parser.parse(
    file_path="{{ file_path }}",
    qvalue_threshold={{ qvalue_threshold }},
    quantity_column="{{ quantity_column }}",
    log_transform={{ log_transform }},
    pseudocount={{ pseudocount }},
    use_genes_as_index={{ use_genes_as_index }},
    filter_contaminants={{ filter_contaminants }},
    filter_reverse={{ filter_reverse }},
    aggregation_method="{{ aggregation_method }}",
)

print(f"Loaded {adata.n_obs} samples x {adata.n_vars} proteins")
print(f"Format: {{ format_type }}, Missing: {stats['missing_percentage']:.1f}%")
"""

        # Build comprehensive parameter schema
        parameter_schema = {
            "file_path": {
                "type": "str",
                "description": "Path to Spectronaut report file (.tsv, .csv, .xlsx)",
            },
            "qvalue_threshold": {
                "type": "float",
                "default": 0.01,
                "description": "Maximum Q-value for protein group filtering (1% FDR)",
            },
            "quantity_column": {
                "type": "str",
                "default": "auto",
                "description": "Quantification column: 'auto', 'PG.Quantity', 'PG.Normalised'",
            },
            "log_transform": {
                "type": "bool",
                "default": True,
                "description": "Apply log2 transformation to intensities",
            },
            "pseudocount": {
                "type": "float",
                "default": 1.0,
                "description": "Value added before log2 transformation to handle zeros",
            },
            "use_genes_as_index": {
                "type": "bool",
                "default": True,
                "description": "Use gene symbols as var index if available",
            },
            "filter_contaminants": {
                "type": "bool",
                "default": True,
                "description": "Remove proteins marked as contaminants (CON__)",
            },
            "filter_reverse": {
                "type": "bool",
                "default": True,
                "description": "Remove reverse/decoy database hits (REV__)",
            },
            "aggregation_method": {
                "type": "str",
                "default": "sum",
                "description": "Aggregation method for long format: sum, mean, median, max",
            },
        }

        return AnalysisStep(
            operation="spectronaut.parse",
            tool_name="SpectronautParser.parse",
            description=(
                f"Parse Spectronaut {format_type} format DIA-MS proteomics data "
                f"with Q-value threshold {qvalue_threshold} (FDR {qvalue_threshold * 100:.0f}%)"
            ),
            library="lobster",
            code_template=code_template,
            imports=[
                "from lobster.services.data_access.proteomics_parsers import SpectronautParser"
            ],
            parameters={
                "file_path": str(file_path),
                "format_type": format_type,
                **parameters,
            },
            parameter_schema=parameter_schema,
            input_entities=[
                {
                    "name": Path(file_path).name,
                    "type": "spectronaut_report",
                    "format": format_type,
                }
            ],
            output_entities=[
                {
                    "name": "adata",
                    "type": "AnnData",
                    "description": "Parsed proteomics data",
                },
                {"name": "stats", "type": "dict", "description": "Parsing statistics"},
            ],
        )

    def _parse_long_format(
        self,
        df: pd.DataFrame,
        qvalue_threshold: float,
        quantity_column: str,
        aggregation_method: str,
    ) -> Tuple[anndata.AnnData, Dict[str, int]]:
        """
        Parse long-format Spectronaut report.

        Args:
            df: Input DataFrame in long format
            qvalue_threshold: Maximum Q-value threshold
            quantity_column: Quantity column to use
            aggregation_method: Aggregation method for protein-level rollup

        Returns:
            Tuple of (AnnData, filter statistics dict)
        """
        filter_stats = {
            "n_qvalue_filtered": 0,
            "n_contaminants": 0,
            "n_reverse": 0,
        }

        # Detect sample column
        sample_col = self._detect_sample_column(df)
        logger.info(f"Using sample column: {sample_col}")

        # Detect quantity column
        detected_quantity = self._detect_quantity_column(df, quantity_column)
        logger.info(f"Using quantity column: {detected_quantity}")

        # Apply Q-value filter if column exists
        # Always filter to ensure invalid Q-values (>1.0) are removed
        n_before = len(df)
        if "PG.Qvalue" in df.columns:
            # Filter both by threshold AND validity (Q-values must be 0-1)
            df = df[
                (df["PG.Qvalue"] <= qvalue_threshold) & (df["PG.Qvalue"] >= 0)
            ].copy()
            filter_stats["n_qvalue_filtered"] = n_before - len(df)
            if filter_stats["n_qvalue_filtered"] > 0:
                logger.info(
                    f"Filtered {filter_stats['n_qvalue_filtered']} rows by PG.Qvalue > {qvalue_threshold} or invalid"
                )

        if len(df) == 0:
            raise ParsingError(
                f"No data remaining after Q-value filtering (threshold={qvalue_threshold}). "
                "Consider increasing qvalue_threshold."
            )

        # Clean sample names (remove file extensions)
        df = df.copy()
        df["_sample_clean"] = df[sample_col].apply(self._clean_sample_name)

        # Handle duplicate sample names BEFORE groupby to preserve distinct samples
        # (e.g., Sample1.raw and Sample1.d should remain separate)
        unique_raw_names = df[sample_col].unique().tolist()
        cleaned_to_unique = {}
        for raw_name in unique_raw_names:
            clean_name = self._clean_sample_name(raw_name)
            if clean_name not in cleaned_to_unique:
                cleaned_to_unique[clean_name] = []
            cleaned_to_unique[clean_name].append(raw_name)

        # Create mapping of raw names to unique cleaned names
        raw_to_unique_clean = {}
        for clean_name, raw_names in cleaned_to_unique.items():
            if len(raw_names) == 1:
                raw_to_unique_clean[raw_names[0]] = clean_name
            else:
                # Multiple raw names clean to same name - add suffix
                for i, raw_name in enumerate(raw_names):
                    if i == 0:
                        raw_to_unique_clean[raw_name] = clean_name
                    else:
                        raw_to_unique_clean[raw_name] = f"{clean_name}_{i}"

        df["_sample_clean"] = df[sample_col].map(raw_to_unique_clean)

        # Aggregate to protein level and pivot to matrix
        intensity_matrix, protein_metadata, samples = self._pivot_long_to_matrix(
            df,
            sample_column="_sample_clean",
            protein_column="PG.ProteinGroups",
            quantity_column=detected_quantity,
            aggregation_method=aggregation_method,
        )

        # Build sample metadata (obs)
        obs_df = pd.DataFrame(
            {"sample_name": samples},
            index=samples,
        )

        # Create AnnData (samples x proteins)
        adata = anndata.AnnData(
            X=intensity_matrix.astype(np.float32),
            obs=obs_df,
            var=protein_metadata,
        )

        return adata, filter_stats

    def _parse_matrix_format(
        self, df: pd.DataFrame, qvalue_threshold: float
    ) -> Tuple[anndata.AnnData, Dict[str, int]]:
        """
        Parse matrix-format Spectronaut report.

        Args:
            df: Input DataFrame in matrix format
            qvalue_threshold: Maximum Q-value threshold

        Returns:
            Tuple of (AnnData, filter statistics dict)
        """
        filter_stats = {
            "n_qvalue_filtered": 0,
            "n_contaminants": 0,
            "n_reverse": 0,
        }

        # Identify protein ID column (usually first column)
        protein_col = self._detect_protein_column(df)

        # Identify gene column if present
        gene_col = self._detect_gene_column(df)

        # Identify Q-value column if present
        qval_col = None
        for col in ["PG.Qvalue", "Qvalue", "Q.Value"]:
            if col in df.columns:
                qval_col = col
                break

        # Identify metadata vs intensity columns
        metadata_cols = [protein_col]
        if gene_col:
            metadata_cols.append(gene_col)
        if qval_col:
            metadata_cols.append(qval_col)

        # Remaining numeric columns are intensity columns
        intensity_cols = []
        for col in df.columns:
            if col not in metadata_cols:
                # Check if column is numeric
                if pd.api.types.is_numeric_dtype(df[col]):
                    intensity_cols.append(col)

        if not intensity_cols:
            raise ParsingError(
                "No intensity columns detected in matrix format. "
                "Expected numeric columns representing sample intensities."
            )

        # Apply Q-value filter if column exists
        # Always filter to ensure invalid Q-values (>1.0) are removed
        n_before = len(df)
        if qval_col:
            # Filter both by threshold AND validity (Q-values must be 0-1)
            df = df[(df[qval_col] <= qvalue_threshold) & (df[qval_col] >= 0)].copy()
            filter_stats["n_qvalue_filtered"] = n_before - len(df)
            if filter_stats["n_qvalue_filtered"] > 0:
                logger.info(
                    f"Filtered {filter_stats['n_qvalue_filtered']} proteins by {qval_col} > {qvalue_threshold} or invalid"
                )

        if len(df) == 0:
            raise ParsingError(
                f"No data remaining after Q-value filtering (threshold={qvalue_threshold}). "
                "Consider increasing qvalue_threshold."
            )

        # Extract intensity matrix (transpose: samples x proteins)
        intensity_matrix = df[intensity_cols].values.T

        # Replace 0 with NaN (Spectronaut may use 0 for missing)
        intensity_matrix = np.where(intensity_matrix == 0, np.nan, intensity_matrix)

        # Build protein metadata
        var_data = {
            "protein_groups": df[protein_col].fillna("").astype(str).values,
        }

        # Extract first protein ID as representative (consistent logic with long format)
        var_data["protein_id"] = pd.Series(var_data["protein_groups"]).apply(
            lambda x: (
                str(x).split(";")[0].strip()
                if pd.notna(x) and str(x).strip()
                else "UNKNOWN"
            )
        )

        if gene_col:
            var_data["gene_symbols"] = df[gene_col].fillna("").astype(str).values

        if qval_col:
            var_data["mean_q_value"] = df[qval_col].values

        var_df = pd.DataFrame(var_data)
        var_df.index = var_data["protein_id"].values

        # Handle duplicate indices
        var_df = self._handle_duplicate_index(var_df)

        # Clean sample names
        sample_names = [self._clean_sample_name(col) for col in intensity_cols]

        # Handle duplicate sample names (e.g., multiple "unknown" from NaN)
        sample_names = self._make_unique_sample_names(sample_names)

        # Build sample metadata (obs)
        obs_df = pd.DataFrame(
            {"sample_name": sample_names},
            index=sample_names,
        )

        # Create AnnData
        adata = anndata.AnnData(
            X=intensity_matrix.astype(np.float32),
            obs=obs_df,
            var=var_df,
        )

        return adata, filter_stats

    def _detect_sample_column(self, df: pd.DataFrame) -> str:
        """
        Detect the sample identifier column.

        Args:
            df: Input DataFrame

        Returns:
            str: Sample column name
        """
        for col in self.SAMPLE_COLUMNS:
            if col in df.columns:
                return col

        raise ParsingError(
            f"No sample column found. Expected one of: {self.SAMPLE_COLUMNS}. "
            f"Available columns: {df.columns.tolist()[:10]}..."
        )

    def _detect_quantity_column(self, df: pd.DataFrame, quantity_column: str) -> str:
        """
        Detect and validate quantity column.

        Args:
            df: Input DataFrame
            quantity_column: Requested quantity column or "auto"

        Returns:
            str: Selected quantity column name
        """
        if quantity_column == "auto":
            for col in self.QUANTITY_COLUMNS:
                if col in df.columns:
                    return col
            # Fall back to any column with "Quantity" in the name
            for col in df.columns:
                if "quantity" in col.lower() or "normalised" in col.lower():
                    return col
            raise ParsingError(
                f"No quantity column found. Expected one of: {self.QUANTITY_COLUMNS}. "
                f"Available columns: {df.columns.tolist()[:10]}..."
            )
        else:
            if quantity_column not in df.columns:
                raise ParsingError(
                    f"Quantity column '{quantity_column}' not found. "
                    f"Available columns: {[c for c in df.columns if 'quant' in c.lower()]}"
                )
            return quantity_column

    def _detect_protein_column(self, df: pd.DataFrame) -> str:
        """
        Detect protein identifier column for matrix format.

        Args:
            df: Input DataFrame

        Returns:
            str: Protein column name
        """
        # Check for known column names
        for col in [
            "PG.ProteinGroups",
            "ProteinGroups",
            "Protein.Groups",
            "ProteinGroup",
        ]:
            if col in df.columns:
                return col

        # Check first column
        first_col = df.columns[0]
        if "protein" in first_col.lower():
            return first_col

        raise ParsingError(
            "Could not detect protein identifier column. "
            "Expected column like 'PG.ProteinGroups' or similar."
        )

    def _detect_gene_column(self, df: pd.DataFrame) -> Optional[str]:
        """
        Detect gene symbol column if present.

        Args:
            df: Input DataFrame

        Returns:
            Optional[str]: Gene column name or None
        """
        for col in ["PG.Genes", "Genes", "Gene.Names", "GeneNames"]:
            if col in df.columns:
                return col
        return None

    def _clean_sample_name(self, name: str) -> str:
        """
        Clean sample name by removing file extensions and paths.

        Args:
            name: Raw sample name/file name

        Returns:
            str: Cleaned sample name
        """
        if pd.isna(name):
            return "unknown"

        name = str(name)

        # Remove common file extensions
        extensions = [".raw", ".d", ".wiff", ".wiff2", ".mzML", ".mzXML"]
        for ext in extensions:
            if name.lower().endswith(ext.lower()):
                name = name[: -len(ext)]

        # Remove path separators and keep just the filename
        if "/" in name:
            name = name.split("/")[-1]
        if "\\" in name:
            name = name.split("\\")[-1]

        return name.strip()

    def _make_unique_sample_names(self, names: List[str]) -> List[str]:
        """
        Ensure all sample names are unique by appending suffix to duplicates.

        Args:
            names: List of sample names (may contain duplicates)

        Returns:
            List[str]: List of unique sample names
        """
        seen = {}
        unique_names = []
        for name in names:
            if name in seen:
                seen[name] += 1
                unique_names.append(f"{name}_{seen[name]}")
            else:
                seen[name] = 0
                unique_names.append(name)
        return unique_names

    def _pivot_long_to_matrix(
        self,
        df: pd.DataFrame,
        sample_column: str,
        protein_column: str,
        quantity_column: str,
        aggregation_method: str,
    ) -> Tuple[np.ndarray, pd.DataFrame, List[str]]:
        """
        Pivot long-format data to samples x proteins matrix.

        Args:
            df: Input DataFrame in long format
            sample_column: Column containing sample identifiers
            protein_column: Column containing protein identifiers
            quantity_column: Column containing quantification values
            aggregation_method: How to aggregate precursors to protein level

        Returns:
            Tuple of (intensity matrix, protein metadata DataFrame, sample list)
        """
        # Aggregation functions
        agg_funcs = {
            "sum": "sum",
            "mean": "mean",
            "median": "median",
            "max": "max",
        }

        if aggregation_method not in agg_funcs:
            raise ParsingError(
                f"Unknown aggregation method: {aggregation_method}. "
                f"Valid options: {list(agg_funcs.keys())}"
            )

        # Group by sample and protein, aggregate quantity
        grouped = df.groupby([sample_column, protein_column])
        intensity_agg = grouped[quantity_column].agg(agg_funcs[aggregation_method])

        # Pivot to wide format
        intensity_pivot = intensity_agg.unstack(level=protein_column)
        intensity_matrix = intensity_pivot.values

        # Replace 0 with NaN (Spectronaut may use 0 for missing) - consistent with matrix format
        intensity_matrix = np.where(intensity_matrix == 0, np.nan, intensity_matrix)

        # Get sample and protein order
        samples = intensity_pivot.index.tolist()
        samples = self._make_unique_sample_names(samples)  # Handle duplicates
        proteins = intensity_pivot.columns.tolist()

        # Build protein metadata
        meta_cols = []
        if "PG.Genes" in df.columns:
            meta_cols.append("PG.Genes")
        if "PG.Qvalue" in df.columns:
            meta_cols.append("PG.Qvalue")

        # Get first occurrence of each protein's metadata
        if meta_cols:
            protein_meta_df = (
                df.groupby(protein_column)[meta_cols].first().reindex(proteins)
            )
        else:
            protein_meta_df = pd.DataFrame(index=proteins)

        # Add protein groups column
        protein_meta_df["protein_groups"] = proteins

        # Extract first protein ID as representative
        protein_meta_df["protein_id"] = protein_meta_df["protein_groups"].apply(
            lambda x: str(x).split(";")[0] if pd.notna(x) else "UNKNOWN"
        )

        # Rename columns to standard names
        if "PG.Genes" in protein_meta_df.columns:
            protein_meta_df = protein_meta_df.rename(
                columns={"PG.Genes": "gene_symbols"}
            )

        if "PG.Qvalue" in protein_meta_df.columns:
            protein_meta_df = protein_meta_df.rename(
                columns={"PG.Qvalue": "mean_q_value"}
            )

        # Add aggregation statistics
        precursor_counts = df.groupby(protein_column).size().reindex(proteins)
        protein_meta_df["n_precursors"] = precursor_counts.values

        # Set protein_id as index
        protein_meta_df = protein_meta_df.reset_index(drop=True)
        protein_meta_df.index = protein_meta_df["protein_id"].values

        # Handle duplicate indices
        protein_meta_df = self._handle_duplicate_index(protein_meta_df)

        return intensity_matrix, protein_meta_df, samples

    def _handle_duplicate_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle duplicate indices by appending suffix.

        Args:
            df: DataFrame with potentially duplicate index

        Returns:
            pd.DataFrame: DataFrame with unique index
        """
        if not df.index.duplicated().any():
            return df

        counts = {}
        new_index = []
        for idx in df.index:
            idx_str = str(idx)
            if idx_str in counts:
                counts[idx_str] += 1
                new_index.append(f"{idx_str}_{counts[idx_str]}")
            else:
                counts[idx_str] = 0
                new_index.append(idx_str)

        df = df.copy()
        df.index = new_index
        return df

    def _flag_special_proteins(self, adata: anndata.AnnData) -> anndata.AnnData:
        """
        Flag contaminant and reverse proteins.

        Args:
            adata: AnnData object

        Returns:
            anndata.AnnData: AnnData with is_contaminant and is_reverse flags
        """
        # Get protein identifiers to check
        if "protein_groups" in adata.var.columns:
            protein_ids = adata.var["protein_groups"].astype(str)
        elif "protein_id" in adata.var.columns:
            protein_ids = adata.var["protein_id"].astype(str)
        else:
            protein_ids = pd.Series(adata.var_names)

        # Flag contaminants
        contaminant_pattern = "|".join(self.CONTAMINANT_PATTERNS)
        adata.var["is_contaminant"] = protein_ids.str.contains(
            contaminant_pattern, case=False, na=False, regex=True
        )

        # Flag reverse/decoy hits
        reverse_pattern = "|".join(self.REVERSE_PATTERNS)
        adata.var["is_reverse"] = protein_ids.str.contains(
            reverse_pattern, case=False, na=False, regex=True
        )

        n_contaminants = adata.var["is_contaminant"].sum()
        n_reverse = adata.var["is_reverse"].sum()

        if n_contaminants > 0:
            logger.info(f"Flagged {n_contaminants} potential contaminants")
        if n_reverse > 0:
            logger.info(f"Flagged {n_reverse} potential reverse/decoy hits")

        return adata

    def _set_gene_index(self, adata: anndata.AnnData) -> anndata.AnnData:
        """
        Set gene symbols as var index.

        Args:
            adata: AnnData object with gene_symbols column

        Returns:
            anndata.AnnData: AnnData with gene-based index
        """
        genes = adata.var["gene_symbols"].fillna("").astype(str)

        # Use protein_id as fallback for empty/missing genes
        if "protein_id" in adata.var.columns:
            protein_ids = adata.var["protein_id"].astype(str)
        else:
            protein_ids = pd.Series(adata.var_names)

        genes = genes.where(genes.str.strip() != "", protein_ids)

        # Take first gene if multiple (semicolon-separated)
        genes = genes.apply(lambda x: x.split(";")[0].strip() if x else "UNKNOWN")

        # Handle duplicates
        if genes.duplicated().any():
            counts = {}
            new_index = []
            for gene in genes:
                if gene in counts:
                    counts[gene] += 1
                    new_index.append(f"{gene}_{counts[gene]}")
                else:
                    counts[gene] = 0
                    new_index.append(gene)
            genes = pd.Series(new_index)

        adata.var.index = genes.values
        return adata

    def get_column_mapping(self) -> Dict[str, str]:
        """
        Return mapping of Spectronaut columns to standardized names.

        Useful for understanding the column transformations.

        Returns:
            Dict mapping Spectronaut column names to standardized names
        """
        return {
            "PG.ProteinGroups": "protein_groups",
            "PG.Genes": "gene_symbols",
            "PG.Quantity": "quantity",
            "PG.Normalised": "normalised_quantity",
            "PG.Qvalue": "q_value",
            "R.FileName": "sample_name",
            "Run": "sample_name",
            "EG.Qvalue": "precursor_q_value",
            "EG.PrecursorId": "precursor_id",
            "FG.MS2Quantity": "ms2_quantity",
        }
