"""
MaxQuant output parser for proteomics data.

This module provides a production-ready parser for MaxQuant proteinGroups.txt output files.
MaxQuant is one of the most widely used proteomics search engines for DDA (Data-Dependent
Acquisition) mass spectrometry data.

Key features:
- Supports both LFQ intensity and raw intensity columns
- Automatic detection of intensity type
- Contaminant (CON_) and reverse hit (REV_) filtering
- Missing value handling (MaxQuant uses 0 for missing)
- Extraction of comprehensive protein annotations

Reference:
    Cox, J., & Mann, M. (2008). MaxQuant enables high peptide identification rates,
    individualized p.p.b.-range mass accuracies and proteome-wide protein quantification.
    Nature biotechnology, 26(12), 1367-1372.

Example usage:
    >>> from lobster.services.data_access.proteomics_parsers import MaxQuantParser
    >>> parser = MaxQuantParser()
    >>> adata, stats = parser.parse(
    ...     "proteinGroups.txt",
    ...     intensity_type="lfq",
    ...     filter_contaminants=True,
    ...     filter_reverse=True
    ... )
    >>> print(f"Loaded {adata.n_obs} samples x {adata.n_vars} proteins")
"""

import re
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


class MaxQuantParser(ProteomicsParser):
    """
    Parser for MaxQuant proteinGroups.txt output files.

    MaxQuant is a quantitative proteomics software package designed for analyzing
    large mass spectrometric data sets. This parser handles the proteinGroups.txt
    output which contains protein-level quantification.

    Key MaxQuant Column Groups:
        - Intensity columns: "Intensity XXX" (raw) or "LFQ intensity XXX" (normalized)
        - Protein identifiers: "Protein IDs", "Majority protein IDs", "Gene names"
        - Quality metrics: "Peptides", "Unique peptides", "Sequence coverage [%]"
        - Flags: "Potential contaminant", "Reverse", "Only identified by site"

    AnnData Output Structure:
        - X: intensity matrix (samples x proteins), NaN for missing values
        - obs: sample metadata (sample_name derived from column headers)
        - var: protein metadata including:
            - protein_ids: Semicolon-separated UniProt IDs
            - majority_protein_ids: Protein IDs with most peptides
            - gene_names: Gene symbols
            - peptide_count: Total peptides mapped
            - unique_peptides: Unique peptides mapped
            - sequence_coverage: Coverage percentage
            - score: MaxQuant protein score
            - is_contaminant: Original contaminant flag
            - is_reverse: Original reverse hit flag
        - uns: parsing metadata (intensity_type, MaxQuant version if available)

    Attributes:
        REQUIRED_COLUMNS: Minimum columns required in proteinGroups.txt
        CONTAMINANT_PREFIX: Prefix for contaminant proteins
        REVERSE_PREFIX: Prefix for reverse/decoy proteins
    """

    # Required columns for valid proteinGroups.txt
    REQUIRED_COLUMNS = {
        "Protein IDs",
    }

    # Optional but commonly present columns
    EXPECTED_COLUMNS = {
        "Protein IDs",
        "Majority protein IDs",
        "Gene names",
        "Fasta headers",
        "Peptides",
        "Unique peptides",
        "Sequence coverage [%]",
        "Score",
        "Potential contaminant",
        "Reverse",
        "Only identified by site",
    }

    # Prefixes for special protein groups
    CONTAMINANT_PREFIX = "CON__"
    REVERSE_PREFIX = "REV__"

    def __init__(self):
        """Initialize MaxQuant parser."""
        super().__init__(name="MaxQuant", version="1.0.0")

    def get_supported_formats(self) -> List[str]:
        """
        Return list of supported file extensions.

        Returns:
            List[str]: Supported extensions for MaxQuant output
        """
        return [".txt", ".tsv"]

    def validate_file(self, file_path: str) -> bool:
        """
        Check if file is valid MaxQuant proteinGroups.txt format.

        Validates:
        1. File exists and has correct extension
        2. File is tab-delimited
        3. Contains required columns (Protein IDs at minimum)
        4. Contains intensity columns (LFQ or raw)

        Args:
            file_path: Path to the file to validate

        Returns:
            bool: True if file is valid MaxQuant proteinGroups format
        """
        try:
            path = self._validate_file_exists(file_path)

            if not self._validate_extension(file_path):
                logger.debug(f"File extension not supported: {path.suffix}")
                return False

            # Read header only
            df_header = pd.read_csv(file_path, sep="\t", nrows=0)
            columns = set(df_header.columns)

            # Check for required columns
            if not self.REQUIRED_COLUMNS.issubset(columns):
                missing = self.REQUIRED_COLUMNS - columns
                logger.debug(f"Missing required columns: {missing}")
                return False

            # Check for intensity columns (either LFQ or raw)
            has_lfq = any(col.startswith("LFQ intensity ") for col in columns)
            has_raw = any(
                col.startswith("Intensity ") and not col.startswith("Intensity ")
                for col in columns
            )
            # More permissive check for intensity columns
            intensity_cols = [
                col
                for col in columns
                if "intensity" in col.lower() and col.lower() != "intensity"
            ]

            if not intensity_cols:
                logger.debug("No intensity columns found")
                return False

            logger.debug(
                f"Valid MaxQuant file: {len(intensity_cols)} intensity columns found"
            )
            return True

        except Exception as e:
            logger.debug(f"File validation failed: {e}")
            return False

    def parse(
        self,
        file_path: str,
        intensity_type: str = "auto",
        filter_contaminants: bool = True,
        filter_reverse: bool = True,
        filter_only_site: bool = True,
        min_peptides: int = 0,
        min_unique_peptides: int = 0,
        log_transform: bool = False,
    ) -> Tuple[anndata.AnnData, Dict[str, Any]]:
        """
        Parse MaxQuant proteinGroups.txt into AnnData.

        Args:
            file_path: Path to proteinGroups.txt file
            intensity_type: Type of intensity to extract:
                - "auto": Automatically detect (prefer LFQ if available)
                - "lfq": LFQ intensity columns only
                - "raw": Raw intensity columns only
            filter_contaminants: Remove proteins marked as contaminants (CON_)
            filter_reverse: Remove reverse/decoy hits (REV_)
            filter_only_site: Remove proteins identified only by modification site
            min_peptides: Minimum total peptides required (0 = no filter)
            min_unique_peptides: Minimum unique peptides required (0 = no filter)
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
        logger.info(f"Parsing MaxQuant file: {file_path}")

        # Validate file
        if not self.validate_file(file_path):
            raise FileValidationError(
                f"Invalid MaxQuant proteinGroups.txt format: {file_path}"
            )

        try:
            # Read full file
            df = pd.read_csv(file_path, sep="\t", low_memory=False)
            n_proteins_raw = len(df)
            logger.info(f"Loaded {n_proteins_raw} protein groups")

            # Determine intensity type and get columns
            intensity_cols, detected_type = self._detect_intensity_columns(
                df, intensity_type
            )
            sample_names = self._extract_sample_names(intensity_cols, detected_type)

            logger.info(
                f"Using {detected_type} intensities from {len(sample_names)} samples"
            )

            # Extract intensity matrix
            intensity_matrix = df[
                intensity_cols
            ].values.T  # Transpose: samples x proteins

            # Replace 0 with NaN (MaxQuant uses 0 for missing values)
            intensity_matrix = np.where(intensity_matrix == 0, np.nan, intensity_matrix)

            # Apply log2 transformation if requested
            if log_transform:
                with np.errstate(divide="ignore"):
                    intensity_matrix = np.log2(intensity_matrix)
                logger.info("Applied log2 transformation")

            # Build protein metadata (var)
            var_df = self._build_var_dataframe(df)

            # Build sample metadata (obs)
            obs_df = pd.DataFrame(
                {"sample_name": sample_names},
                index=sample_names,
            )

            # Create AnnData (samples x proteins)
            adata = anndata.AnnData(
                X=intensity_matrix.astype(np.float32),
                obs=obs_df,
                var=var_df,
            )

            # Store metadata
            adata.uns["parser"] = {
                "name": self.name,
                "version": self.version,
                "source_file": str(Path(file_path).name),
                "intensity_type": detected_type,
            }

            # Track filtering statistics
            filter_stats = {
                "n_contaminants": 0,
                "n_reverse": 0,
                "n_only_site": 0,
                "n_low_peptides": 0,
                "n_low_unique_peptides": 0,
            }

            # Apply filters
            if filter_contaminants:
                mask = ~adata.var["is_contaminant"]
                n_removed = (~mask).sum()
                filter_stats["n_contaminants"] = int(n_removed)
                adata = adata[:, mask].copy()
                logger.info(f"Removed {n_removed} contaminants")

            if filter_reverse:
                mask = ~adata.var["is_reverse"]
                n_removed = (~mask).sum()
                filter_stats["n_reverse"] = int(n_removed)
                adata = adata[:, mask].copy()
                logger.info(f"Removed {n_removed} reverse hits")

            if filter_only_site and "only_identified_by_site" in adata.var.columns:
                mask = ~adata.var["only_identified_by_site"]
                n_removed = (~mask).sum()
                filter_stats["n_only_site"] = int(n_removed)
                adata = adata[:, mask].copy()
                logger.info(f"Removed {n_removed} only-by-site proteins")

            if min_peptides > 0 and "peptide_count" in adata.var.columns:
                mask = adata.var["peptide_count"] >= min_peptides
                n_removed = (~mask).sum()
                filter_stats["n_low_peptides"] = int(n_removed)
                adata = adata[:, mask].copy()
                logger.info(
                    f"Removed {n_removed} proteins with <{min_peptides} peptides"
                )

            if min_unique_peptides > 0 and "unique_peptides" in adata.var.columns:
                mask = adata.var["unique_peptides"] >= min_unique_peptides
                n_removed = (~mask).sum()
                filter_stats["n_low_unique_peptides"] = int(n_removed)
                adata = adata[:, mask].copy()
                logger.info(
                    f"Removed {n_removed} proteins with <{min_unique_peptides} unique peptides"
                )

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
                intensity_type=detected_type,
                log_transformed=log_transform,
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
            logger.error(f"Failed to parse MaxQuant file: {e}")
            raise ParsingError(f"Failed to parse MaxQuant file: {e}") from e

    def _detect_intensity_columns(
        self,
        df: pd.DataFrame,
        intensity_type: str,
    ) -> Tuple[List[str], str]:
        """
        Detect and return intensity columns based on type preference.

        Args:
            df: Input DataFrame
            intensity_type: Requested intensity type ("auto", "lfq", "raw")

        Returns:
            Tuple of (list of column names, detected type string)

        Raises:
            ParsingError: If no valid intensity columns found
        """
        columns = df.columns.tolist()

        # Find LFQ intensity columns
        lfq_pattern = re.compile(r"^LFQ intensity (.+)$")
        lfq_cols = [col for col in columns if lfq_pattern.match(col)]

        # Find raw intensity columns (excluding LFQ and total Intensity)
        raw_pattern = re.compile(r"^Intensity (.+)$")
        raw_cols = [
            col
            for col in columns
            if raw_pattern.match(col)
            and not col.startswith("LFQ")
            and col != "Intensity"  # Exclude total intensity column
        ]

        if intensity_type == "auto":
            # Prefer LFQ if available
            if lfq_cols:
                return lfq_cols, "lfq"
            elif raw_cols:
                return raw_cols, "raw"
            else:
                raise ParsingError("No intensity columns found in file")

        elif intensity_type == "lfq":
            if not lfq_cols:
                raise ParsingError("No LFQ intensity columns found")
            return lfq_cols, "lfq"

        elif intensity_type == "raw":
            if not raw_cols:
                raise ParsingError("No raw intensity columns found")
            return raw_cols, "raw"

        else:
            raise ParsingError(f"Unknown intensity_type: {intensity_type}")

    def _extract_sample_names(
        self,
        intensity_cols: List[str],
        intensity_type: str,
    ) -> List[str]:
        """
        Extract sample names from intensity column headers.

        Args:
            intensity_cols: List of intensity column names
            intensity_type: Type of intensity ("lfq" or "raw")

        Returns:
            List of sample names
        """
        if intensity_type == "lfq":
            prefix = "LFQ intensity "
        else:
            prefix = "Intensity "

        return [col.replace(prefix, "") for col in intensity_cols]

    def _build_var_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build protein metadata DataFrame for AnnData.var.

        Args:
            df: Original MaxQuant DataFrame

        Returns:
            pd.DataFrame: Protein metadata with standardized columns
        """
        var_data = {}

        # Core identifiers (required)
        var_data["protein_ids"] = df["Protein IDs"].fillna("").astype(str)

        # Create index from Protein IDs (use first ID if multiple)
        protein_index = var_data["protein_ids"].apply(
            lambda x: x.split(";")[0] if x else "UNKNOWN"
        )

        # Handle duplicate indices by appending suffix
        if protein_index.duplicated().any():
            counts = {}
            new_index = []
            for idx in protein_index:
                if idx in counts:
                    counts[idx] += 1
                    new_index.append(f"{idx}_{counts[idx]}")
                else:
                    counts[idx] = 0
                    new_index.append(idx)
            protein_index = pd.Series(new_index)

        # Optional identifiers
        if "Majority protein IDs" in df.columns:
            var_data["majority_protein_ids"] = (
                df["Majority protein IDs"].fillna("").astype(str)
            )

        if "Gene names" in df.columns:
            var_data["gene_names"] = df["Gene names"].fillna("").astype(str)

        if "Fasta headers" in df.columns:
            var_data["fasta_headers"] = df["Fasta headers"].fillna("").astype(str)

        # Quality metrics
        if "Peptides" in df.columns:
            var_data["peptide_count"] = (
                pd.to_numeric(df["Peptides"], errors="coerce").fillna(0).astype(int)
            )

        if "Unique peptides" in df.columns:
            var_data["unique_peptides"] = (
                pd.to_numeric(df["Unique peptides"], errors="coerce")
                .fillna(0)
                .astype(int)
            )

        if "Sequence coverage [%]" in df.columns:
            var_data["sequence_coverage"] = pd.to_numeric(
                df["Sequence coverage [%]"], errors="coerce"
            ).fillna(0.0)

        if "Score" in df.columns:
            var_data["score"] = pd.to_numeric(df["Score"], errors="coerce").fillna(0.0)

        if "Mol. weight [kDa]" in df.columns:
            var_data["molecular_weight_kda"] = pd.to_numeric(
                df["Mol. weight [kDa]"], errors="coerce"
            ).fillna(0.0)

        # Quality flags
        if "Potential contaminant" in df.columns:
            var_data["is_contaminant"] = (
                df["Potential contaminant"].fillna("").str.strip() == "+"
            )
        else:
            # Check protein IDs for contaminant prefix
            var_data["is_contaminant"] = var_data["protein_ids"].str.contains(
                self.CONTAMINANT_PREFIX, case=False, na=False
            )

        if "Reverse" in df.columns:
            var_data["is_reverse"] = df["Reverse"].fillna("").str.strip() == "+"
        else:
            # Check protein IDs for reverse prefix
            var_data["is_reverse"] = var_data["protein_ids"].str.contains(
                self.REVERSE_PREFIX, case=False, na=False
            )

        if "Only identified by site" in df.columns:
            var_data["only_identified_by_site"] = (
                df["Only identified by site"].fillna("").str.strip() == "+"
            )

        var_df = pd.DataFrame(var_data)
        var_df.index = protein_index.values

        return var_df

    def get_column_mapping(self) -> Dict[str, str]:
        """
        Return mapping of MaxQuant columns to standardized names.

        Useful for understanding the column transformations.

        Returns:
            Dict mapping MaxQuant column names to standardized names
        """
        return {
            "Protein IDs": "protein_ids",
            "Majority protein IDs": "majority_protein_ids",
            "Gene names": "gene_names",
            "Fasta headers": "fasta_headers",
            "Peptides": "peptide_count",
            "Unique peptides": "unique_peptides",
            "Sequence coverage [%]": "sequence_coverage",
            "Score": "score",
            "Mol. weight [kDa]": "molecular_weight_kda",
            "Potential contaminant": "is_contaminant",
            "Reverse": "is_reverse",
            "Only identified by site": "only_identified_by_site",
        }
