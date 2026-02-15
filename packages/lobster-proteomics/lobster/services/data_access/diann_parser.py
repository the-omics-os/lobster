"""
DIA-NN output parser for proteomics data.

This module provides a production-ready parser for DIA-NN report files. DIA-NN is a
universal software for data-independent acquisition (DIA) proteomics data processing.
It is known for its speed, accuracy, and library-free search capabilities.

Key features:
- Supports both report.tsv and report.parquet formats
- Handles long-format to matrix pivoting
- Multiple quantification options (Precursor.Normalised, Precursor.Quantity)
- Protein group aggregation with configurable methods
- Q-value based filtering at precursor and protein group levels

Reference:
    Demichev, V., Messner, C. B., Vernardis, S. I., Lilley, K. S., & Ralser, M. (2020).
    DIA-NN: neural networks and interference correction enable deep proteome coverage
    in high throughput. Nature Methods, 17(1), 41-44.

Example usage:
    >>> from lobster.services.data_access.proteomics_parsers import DIANNParser
    >>> parser = DIANNParser()
    >>> adata, stats = parser.parse(
    ...     "report.tsv",
    ...     quantity_column="Precursor.Normalised",
    ...     protein_group_q_value=0.01
    ... )
    >>> print(f"Loaded {adata.n_obs} samples x {adata.n_vars} proteins")
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple

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


class DIANNParser(ProteomicsParser):
    """
    Parser for DIA-NN report.tsv and report.parquet output files.

    DIA-NN outputs data in long format where each row represents a precursor
    identification in a specific run. This parser pivots the data to create
    a samples x proteins matrix suitable for downstream analysis.

    Key DIA-NN Columns:
        - Identifiers: "Protein.Group", "Protein.Ids", "Protein.Names", "Genes"
        - Quantification: "Precursor.Quantity", "Precursor.Normalised"
        - Sample: "Run", "File.Name"
        - Quality: "Q.Value", "PG.Q.Value", "Lib.Q.Value", "Global.Q.Value"
        - Precursor info: "Modified.Sequence", "Stripped.Sequence", "Precursor.Charge"

    AnnData Output Structure:
        - X: intensity matrix (samples x proteins), NaN for missing values
        - obs: sample metadata (run names, file paths)
        - var: protein metadata including:
            - protein_group: DIA-NN protein group identifier
            - protein_ids: UniProt IDs
            - protein_names: Protein names
            - genes: Gene symbols
            - n_precursors: Number of precursors per protein
            - n_peptides: Number of unique peptides
            - mean_q_value: Average Q-value across precursors
        - uns: parsing metadata (quantity column used, Q-value thresholds)

    Attributes:
        REQUIRED_COLUMNS: Minimum columns required in DIA-NN report
        QUANTITY_COLUMNS: Available quantification columns
        AGGREGATION_METHODS: Methods for aggregating precursor to protein level
    """

    # Required columns for valid DIA-NN report
    REQUIRED_COLUMNS = {
        "Run",
        "Protein.Group",
    }

    # Quantification columns in order of preference
    QUANTITY_COLUMNS = [
        "Precursor.Normalised",
        "Precursor.Quantity",
        "PG.Normalised",
        "PG.Quantity",
    ]

    # Aggregation methods for precursor -> protein
    AGGREGATION_METHODS = ["sum", "mean", "median", "max"]

    def __init__(self):
        """Initialize DIA-NN parser."""
        super().__init__(name="DIA-NN", version="1.0.0")

    def get_supported_formats(self) -> List[str]:
        """
        Return list of supported file extensions.

        Returns:
            List[str]: Supported extensions for DIA-NN output
        """
        return [".tsv", ".parquet", ".txt"]

    def validate_file(self, file_path: str) -> bool:
        """
        Check if file is valid DIA-NN report format.

        Validates:
        1. File exists and has correct extension
        2. Contains required columns (Run, Protein.Group)
        3. Contains at least one quantification column

        Args:
            file_path: Path to the file to validate

        Returns:
            bool: True if file is valid DIA-NN report format
        """
        try:
            path = self._validate_file_exists(file_path)

            if not self._validate_extension(file_path):
                logger.debug(f"File extension not supported: {path.suffix}")
                return False

            # Read header based on file type
            if path.suffix.lower() == ".parquet":
                try:
                    df_header = pd.read_parquet(file_path).head(0)
                except Exception as e:
                    logger.debug(f"Failed to read parquet file: {e}")
                    return False
            else:
                df_header = pd.read_csv(file_path, sep="\t", nrows=0)

            columns = set(df_header.columns)

            # Check for required columns
            if not self.REQUIRED_COLUMNS.issubset(columns):
                missing = self.REQUIRED_COLUMNS - columns
                logger.debug(f"Missing required columns: {missing}")
                return False

            # Check for at least one quantification column
            has_quantity = any(col in columns for col in self.QUANTITY_COLUMNS)
            if not has_quantity:
                logger.debug("No quantification columns found")
                return False

            logger.debug("Valid DIA-NN report file")
            return True

        except Exception as e:
            logger.debug(f"File validation failed: {e}")
            return False

    def parse(
        self,
        file_path: str,
        quantity_column: str = "auto",
        aggregation_method: str = "sum",
        precursor_q_value: float = 0.01,
        protein_group_q_value: float = 0.01,
        min_precursors: int = 0,
        min_peptides: int = 0,
        use_genes_as_index: bool = True,
        log_transform: bool = False,
    ) -> Tuple[anndata.AnnData, Dict[str, Any]]:
        """
        Parse DIA-NN report file into AnnData.

        Args:
            file_path: Path to report.tsv or report.parquet file
            quantity_column: Quantification column to use:
                - "auto": Automatically detect (prefer Precursor.Normalised)
                - "Precursor.Normalised": Normalized precursor intensity
                - "Precursor.Quantity": Raw precursor intensity
                - "PG.Normalised": Pre-aggregated protein group normalized
                - "PG.Quantity": Pre-aggregated protein group quantity
            aggregation_method: Method to aggregate precursors to protein level:
                - "sum": Sum of precursor intensities (recommended for label-free)
                - "mean": Mean of precursor intensities
                - "median": Median of precursor intensities
                - "max": Maximum precursor intensity
            precursor_q_value: Maximum Q-value for precursor filtering (0.01 = 1% FDR)
            protein_group_q_value: Maximum Q-value for protein group filtering
            min_precursors: Minimum precursors per protein (0 = no filter)
            min_peptides: Minimum peptides per protein (0 = no filter)
            use_genes_as_index: Use gene names as var index if available
            log_transform: Apply log2 transformation to intensities

        Returns:
            Tuple containing:
                - anndata.AnnData: Parsed proteomics data
                - Dict[str, Any]: Parsing statistics

        Raises:
            FileValidationError: If file format is invalid
            ParsingError: If parsing fails
            FileNotFoundError: If file does not exist
        """
        logger.info(f"Parsing DIA-NN file: {file_path}")

        # Validate file
        if not self.validate_file(file_path):
            raise FileValidationError(f"Invalid DIA-NN report format: {file_path}")

        try:
            # Read file based on format
            path = Path(file_path)
            if path.suffix.lower() == ".parquet":
                df = pd.read_parquet(file_path)
                logger.info(f"Loaded parquet file with {len(df)} rows")
            else:
                df = pd.read_csv(file_path, sep="\t", low_memory=False)
                logger.info(f"Loaded TSV file with {len(df)} rows")

            n_rows_raw = len(df)

            # Determine quantity column
            detected_quantity = self._detect_quantity_column(df, quantity_column)
            logger.info(f"Using quantity column: {detected_quantity}")

            # Apply Q-value filters
            df, filter_stats = self._apply_qvalue_filters(
                df, precursor_q_value, protein_group_q_value
            )

            # Extract unique runs/samples
            sample_column = self._detect_sample_column(df)
            samples = df[sample_column].unique().tolist()
            logger.info(f"Found {len(samples)} samples")

            # Aggregate to protein level and pivot to matrix
            intensity_matrix, protein_metadata = self._pivot_to_matrix(
                df,
                sample_column=sample_column,
                protein_column="Protein.Group",
                quantity_column=detected_quantity,
                aggregation_method=aggregation_method,
            )

            n_proteins_raw = len(protein_metadata)

            # Apply protein-level filters
            if min_precursors > 0 and "n_precursors" in protein_metadata.columns:
                mask = protein_metadata["n_precursors"] >= min_precursors
                intensity_matrix = intensity_matrix[:, mask]
                protein_metadata = protein_metadata[mask].copy()
                filter_stats["n_low_precursors"] = (~mask).sum()
                logger.info(
                    f"Removed {filter_stats['n_low_precursors']} proteins with "
                    f"<{min_precursors} precursors"
                )

            if min_peptides > 0 and "n_peptides" in protein_metadata.columns:
                mask = protein_metadata["n_peptides"] >= min_peptides
                intensity_matrix = intensity_matrix[:, mask]
                protein_metadata = protein_metadata[mask].copy()
                filter_stats["n_low_peptides"] = (~mask).sum()
                logger.info(
                    f"Removed {filter_stats['n_low_peptides']} proteins with "
                    f"<{min_peptides} peptides"
                )

            # Apply log2 transformation if requested
            if log_transform:
                with np.errstate(divide="ignore", invalid="ignore"):
                    intensity_matrix = np.log2(intensity_matrix)
                logger.info("Applied log2 transformation")

            # Build sample metadata (obs)
            obs_df = pd.DataFrame(
                {"sample_name": samples},
                index=samples,
            )

            # Set protein index
            if use_genes_as_index and "genes" in protein_metadata.columns:
                protein_index = self._create_unique_gene_index(protein_metadata)
            else:
                protein_index = protein_metadata["protein_group"].values

            protein_metadata = protein_metadata.copy()
            protein_metadata.index = protein_index

            # Create AnnData (samples x proteins)
            adata = anndata.AnnData(
                X=intensity_matrix.astype(np.float32),
                obs=obs_df,
                var=protein_metadata,
            )

            # Store metadata
            adata.uns["parser"] = {
                "name": self.name,
                "version": self.version,
                "source_file": str(path.name),
                "quantity_column": detected_quantity,
                "aggregation_method": aggregation_method,
                "precursor_q_value_threshold": precursor_q_value,
                "protein_group_q_value_threshold": protein_group_q_value,
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
                quantity_column=detected_quantity,
                aggregation_method=aggregation_method,
                log_transformed=log_transform,
                n_rows_raw=n_rows_raw,
                **filter_stats,
            )

            logger.info(
                f"Parsing complete: {adata.n_obs} samples x {adata.n_vars} proteins "
                f"({missing_percentage:.1f}% missing)"
            )

            return adata, stats

        except FileValidationError:
            raise
        except Exception as e:
            logger.error(f"Failed to parse DIA-NN file: {e}")
            raise ParsingError(f"Failed to parse DIA-NN file: {e}") from e

    def _detect_quantity_column(
        self,
        df: pd.DataFrame,
        quantity_column: str,
    ) -> str:
        """
        Detect and validate quantity column.

        Args:
            df: Input DataFrame
            quantity_column: Requested quantity column or "auto"

        Returns:
            str: Selected quantity column name

        Raises:
            ParsingError: If requested column not found
        """
        columns = df.columns.tolist()

        if quantity_column == "auto":
            # Try columns in order of preference
            for col in self.QUANTITY_COLUMNS:
                if col in columns:
                    return col
            raise ParsingError(f"No valid quantity column found. Available: {columns}")
        else:
            if quantity_column not in columns:
                available_qty = [
                    c
                    for c in columns
                    if "quantity" in c.lower() or "normalised" in c.lower()
                ]
                raise ParsingError(
                    f"Quantity column '{quantity_column}' not found. "
                    f"Available quantity-related columns: {available_qty}"
                )
            return quantity_column

    def _detect_sample_column(self, df: pd.DataFrame) -> str:
        """
        Detect the sample/run identifier column.

        Args:
            df: Input DataFrame

        Returns:
            str: Sample column name
        """
        # Prefer "Run" over "File.Name"
        if "Run" in df.columns:
            return "Run"
        elif "File.Name" in df.columns:
            return "File.Name"
        else:
            raise ParsingError("No sample column found (Run or File.Name)")

    def _apply_qvalue_filters(
        self,
        df: pd.DataFrame,
        precursor_q_value: float,
        protein_group_q_value: float,
    ) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """
        Apply Q-value based filtering.

        Args:
            df: Input DataFrame
            precursor_q_value: Maximum precursor Q-value
            protein_group_q_value: Maximum protein group Q-value

        Returns:
            Tuple of (filtered DataFrame, filter statistics dict)
        """
        stats = {
            "n_precursor_q_filtered": 0,
            "n_protein_q_filtered": 0,
        }

        n_before = len(df)

        # Filter by precursor Q-value
        if "Q.Value" in df.columns and precursor_q_value < 1.0:
            mask = df["Q.Value"] <= precursor_q_value
            df = df[mask].copy()
            stats["n_precursor_q_filtered"] = n_before - len(df)
            logger.info(
                f"Filtered {stats['n_precursor_q_filtered']} rows by precursor Q-value > {precursor_q_value}"
            )
            n_before = len(df)

        # Filter by protein group Q-value
        if "PG.Q.Value" in df.columns and protein_group_q_value < 1.0:
            mask = df["PG.Q.Value"] <= protein_group_q_value
            df = df[mask].copy()
            stats["n_protein_q_filtered"] = n_before - len(df)
            logger.info(
                f"Filtered {stats['n_protein_q_filtered']} rows by PG Q-value > {protein_group_q_value}"
            )

        return df, stats

    def _pivot_to_matrix(
        self,
        df: pd.DataFrame,
        sample_column: str,
        protein_column: str,
        quantity_column: str,
        aggregation_method: str,
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Pivot long-format data to samples x proteins matrix.

        Args:
            df: Input DataFrame in long format
            sample_column: Column containing sample identifiers
            protein_column: Column containing protein identifiers
            quantity_column: Column containing quantification values
            aggregation_method: How to aggregate precursors to protein level

        Returns:
            Tuple of (intensity matrix [samples x proteins], protein metadata DataFrame)
        """
        # Aggregate precursors to protein level per sample
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

        # Get sample order and protein order
        proteins = intensity_pivot.columns.tolist()

        # Build protein metadata
        protein_meta_cols = ["Protein.Ids", "Protein.Names", "Genes"]
        meta_available = [col for col in protein_meta_cols if col in df.columns]

        # Get first occurrence of each protein's metadata
        protein_meta_df = (
            df.groupby(protein_column)[meta_available].first().reindex(proteins)
        )

        # Add aggregation statistics
        precursor_counts = df.groupby(protein_column).size().reindex(proteins)
        protein_meta_df["n_precursors"] = precursor_counts.values

        # Count unique peptides if available
        if "Stripped.Sequence" in df.columns:
            peptide_counts = (
                df.groupby(protein_column)["Stripped.Sequence"]
                .nunique()
                .reindex(proteins)
            )
            protein_meta_df["n_peptides"] = peptide_counts.values
        elif "Modified.Sequence" in df.columns:
            peptide_counts = (
                df.groupby(protein_column)["Modified.Sequence"]
                .nunique()
                .reindex(proteins)
            )
            protein_meta_df["n_peptides"] = peptide_counts.values

        # Mean Q-value per protein
        if "Q.Value" in df.columns:
            mean_qval = df.groupby(protein_column)["Q.Value"].mean().reindex(proteins)
            protein_meta_df["mean_q_value"] = mean_qval.values

        # Standardize column names
        protein_meta_df = protein_meta_df.reset_index()
        protein_meta_df.columns = [
            col.lower().replace(".", "_").replace(" ", "_")
            for col in protein_meta_df.columns
        ]

        # Rename protein_group column
        if "protein_group" not in protein_meta_df.columns:
            protein_meta_df = protein_meta_df.rename(
                columns={protein_meta_df.columns[0]: "protein_group"}
            )

        return intensity_matrix, protein_meta_df

    def _create_unique_gene_index(self, protein_metadata: pd.DataFrame) -> np.ndarray:
        """
        Create unique gene-based index, handling duplicates and missing values.

        Args:
            protein_metadata: Protein metadata DataFrame with 'genes' column

        Returns:
            np.ndarray: Unique gene-based index
        """
        genes = protein_metadata["genes"].fillna("").astype(str)

        # Use protein_group as fallback for empty genes
        protein_groups = protein_metadata["protein_group"].astype(str)
        genes = genes.where(genes != "", protein_groups)

        # Take first gene if multiple
        genes = genes.apply(lambda x: x.split(";")[0].strip() if x else "UNKNOWN")

        # Handle duplicates by appending suffix
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

        return genes.values

    def get_column_mapping(self) -> Dict[str, str]:
        """
        Return mapping of DIA-NN columns to standardized names.

        Useful for understanding the column transformations.

        Returns:
            Dict mapping DIA-NN column names to standardized names
        """
        return {
            "Protein.Group": "protein_group",
            "Protein.Ids": "protein_ids",
            "Protein.Names": "protein_names",
            "Genes": "genes",
            "Run": "sample_name",
            "File.Name": "file_name",
            "Q.Value": "q_value",
            "PG.Q.Value": "pg_q_value",
            "Precursor.Quantity": "precursor_quantity",
            "Precursor.Normalised": "precursor_normalised",
            "Modified.Sequence": "modified_sequence",
            "Stripped.Sequence": "stripped_sequence",
        }
