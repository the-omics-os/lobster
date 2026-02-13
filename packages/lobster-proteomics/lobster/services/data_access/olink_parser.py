"""
Olink NPX parser for affinity proteomics data.

This parser handles Olink proximity extension assay (PEA) data, supporting
multiple export formats (CSV long format and Excel wide format) with NPX
(Normalized Protein eXpression) quantification.

File format specifications adapted from official OlinkRPackage:
https://github.com/Olink-Proteomics/OlinkRPackage
License: AGPL-3.0 (patterns referenced, implementation independent)

Supported formats:
- Explore platforms (384, HT, 3072): CSV/TSV long format
- Target platforms (48, 96): Excel wide format
- Flex platforms: Excel wide format
- Reveal platforms: CSV long format

NPX Scale: Log2-based relative protein quantification
Typical range: -5 to +15 NPX units
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


class OlinkParser(ProteomicsParser):
    """
    Parser for Olink NPX output files (affinity proteomics).

    Olink uses proximity extension assay (PEA) technology for multiplexed
    targeted protein quantification. Data is provided as NPX (Normalized
    Protein eXpression) values on a log2 scale.

    Key Features:
    - Supports long format (Explore CSV) and wide format (Target Excel)
    - NPX value validation (typical range -5 to +15)
    - QC warning detection and filtering
    - LOD (Limit of Detection) handling
    - Multi-panel support
    - Plate effect detection

    File Format Detection:
    - Long format: Has "NPX" column (one row per sample-protein)
    - Wide format: Has multiple "Sample_X" columns

    AnnData Structure:
    - X: NPX matrix (samples x proteins), already log2-transformed
    - obs: sample_id, plate_id, qc_warning, panel (if single)
    - var: uniprot_id, assay, olink_id, panel, lod, missing_freq
    - uns: parser metadata, panel info, normalization method
    """

    # Core required columns for Olink format detection
    REQUIRED_COLUMNS_LONG = {"SampleID", "Assay", "NPX"}
    REQUIRED_COLUMNS_WIDE = {"Assay"}  # Wide format has samples as columns

    # Alternative column name variations
    SAMPLE_ID_VARIANTS = ["SampleID", "Sample ID", "Sample_ID", "sample_id"]
    UNIPROT_VARIANTS = ["UniProt", "Uniprot", "UNIPROT"]
    OLINK_ID_VARIANTS = ["OlinkID", "Olink ID", "OlinkId"]

    def __init__(self):
        """Initialize Olink parser."""
        super().__init__(name="Olink", version="1.0.0")

    def get_supported_formats(self) -> List[str]:
        """Return supported file extensions for Olink data."""
        return [".xlsx", ".xls", ".csv", ".tsv"]

    def validate_file(self, file_path: str) -> bool:
        """
        Validate if file is Olink format.

        Checks for:
        1. Correct file extension
        2. Required columns present
        3. NPX data columns exist

        Args:
            file_path: Path to file

        Returns:
            bool: True if valid Olink file
        """
        try:
            path = self._validate_file_exists(file_path)

            if not self._validate_extension(file_path):
                return False

            # Read header to check columns
            if path.suffix.lower() in [".xlsx", ".xls"]:
                df_header = pd.read_excel(file_path, nrows=5)
            else:
                # Try comma-separated first, then tab
                try:
                    df_header = pd.read_csv(file_path, nrows=5)
                except:
                    try:
                        df_header = pd.read_csv(file_path, sep="\t", nrows=5)
                    except:
                        return False

            columns = set(df_header.columns)

            # Check for Assay column (required in all Olink formats)
            if "Assay" not in columns:
                return False

            # Check for NPX data (long format has NPX column, wide has sample columns)
            has_npx_col = "NPX" in columns
            has_sample_id = any(col in columns for col in self.SAMPLE_ID_VARIANTS)
            has_sample_cols = any(
                col.startswith("Sample") for col in columns
            )  # Wide format

            if not (has_npx_col or (has_sample_id and has_sample_cols)):
                return False

            logger.debug(f"Valid Olink file detected: {path.name}")
            return True

        except Exception as e:
            logger.debug(f"Olink file validation failed: {e}")
            return False

    def parse(
        self,
        file_path: str,
        filter_qc_warnings: bool = False,
        min_samples_detected: int = 0,
        lod_handling: str = "keep",
        use_gene_names: bool = True,
        **kwargs,
    ) -> Tuple[anndata.AnnData, Dict[str, Any]]:
        """
        Parse Olink NPX file to AnnData.

        Args:
            file_path: Path to Olink file
            filter_qc_warnings: Remove samples/proteins with QC warnings (default: False)
            min_samples_detected: Minimum samples with non-missing NPX per protein
            lod_handling: LOD strategy ("keep", "flag", "filter")
            use_gene_names: Use Assay column as var index instead of UniProt

        Returns:
            Tuple[AnnData, Dict]: (parsed data, statistics)
        """
        logger.info(f"Parsing Olink file: {file_path}")

        if not self.validate_file(file_path):
            raise FileValidationError(f"Invalid Olink format: {file_path}")

        try:
            # Read file
            path = Path(file_path)
            if path.suffix.lower() in [".xlsx", ".xls"]:
                df = pd.read_excel(file_path)
            else:
                try:
                    df = pd.read_csv(file_path)
                except:
                    df = pd.read_csv(file_path, sep="\t")

            logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")

            # Detect format (long vs wide)
            is_long_format = "NPX" in df.columns and self._find_column(
                df, self.SAMPLE_ID_VARIANTS
            )

            if is_long_format:
                adata, stats = self._parse_long_format(
                    df,
                    filter_qc_warnings=filter_qc_warnings,
                    min_samples_detected=min_samples_detected,
                    lod_handling=lod_handling,
                    use_gene_names=use_gene_names,
                )
            else:
                # Wide format (Target/Flex) - future implementation
                raise NotImplementedError(
                    "Wide format (Excel Target/Flex) parsing not yet implemented. "
                    "Currently supported: Long format (Explore CSV)"
                )

            # Add parser metadata
            adata.uns["parser"] = {
                "name": self.name,
                "version": self.version,
                "source_file": path.name,
                "format": "long" if is_long_format else "wide",
                "intensity_type": "npx",
                "platform": "olink",
            }

            # Add platform detection signal for proteomics_expert
            adata.uns["platform"] = "olink"

            logger.info(f"Successfully parsed Olink file: {adata.shape}")
            return adata, stats

        except FileValidationError:
            raise
        except Exception as e:
            logger.error(f"Failed to parse Olink file: {e}")
            raise ParsingError(f"Failed to parse Olink file: {e}") from e

    # =========================================================================
    # PARSING LOGIC - LONG FORMAT (EXPLORE CSV)
    # =========================================================================

    def _parse_long_format(
        self,
        df: pd.DataFrame,
        filter_qc_warnings: bool,
        min_samples_detected: int,
        lod_handling: str,
        use_gene_names: bool,
    ) -> Tuple[anndata.AnnData, Dict[str, Any]]:
        """
        Parse long-format Olink data (Explore platforms).

        Long format structure: One row per sample-protein measurement
        Example: 29,441 rows = 158 samples × 1,104 assays (with panel info)

        Args:
            df: Loaded DataFrame
            filter_qc_warnings: Remove QC-failed samples/proteins
            min_samples_detected: Minimum samples per protein
            lod_handling: LOD strategy
            use_gene_names: Use Assay as var index

        Returns:
            Tuple of (AnnData, stats)
        """
        logger.info("Parsing long-format Olink data (Explore)")

        # Find column names (handle variants)
        sample_col = self._find_column(df, self.SAMPLE_ID_VARIANTS, required=True)
        uniprot_col = self._find_column(df, self.UNIPROT_VARIANTS)
        olink_id_col = self._find_column(df, self.OLINK_ID_VARIANTS)

        # Extract unique samples and proteins
        samples = df[sample_col].unique()
        proteins = df["Assay"].unique()

        n_samples_raw = len(samples)
        n_proteins_raw = len(proteins)

        logger.info(
            f"Raw dimensions: {n_samples_raw} samples × {n_proteins_raw} proteins"
        )

        # Apply QC filtering if requested
        if filter_qc_warnings and "QC_Warning" in df.columns:
            df_filtered, qc_stats = self._filter_qc_warnings(df, sample_col)
        else:
            df_filtered = df
            qc_stats = {"samples_removed_qc": 0, "measurements_removed_qc": 0}

        # Pivot long → wide (samples × proteins)
        npx_matrix, sample_order, protein_order = self._pivot_npx_matrix(
            df_filtered, sample_col
        )

        # Build obs (sample metadata)
        obs_df = self._build_obs_dataframe(df_filtered, sample_col, sample_order)

        # Build var (protein metadata)
        var_df = self._build_var_dataframe(
            df_filtered, protein_order, uniprot_col, olink_id_col, use_gene_names
        )

        # Filter proteins by minimum detection
        if min_samples_detected > 0:
            npx_matrix, var_df, filter_stats = self._filter_by_detection(
                npx_matrix, var_df, min_samples_detected
            )
        else:
            filter_stats = {"proteins_removed_min_detection": 0}

        # Handle LOD
        if lod_handling != "keep":
            npx_matrix, var_df, lod_stats = self._handle_lod(
                npx_matrix, var_df, lod_handling
            )
        else:
            lod_stats = {"proteins_removed_lod": 0}

        # Create AnnData
        adata = anndata.AnnData(X=npx_matrix.astype(np.float32), obs=obs_df, var=var_df)

        # Store raw panel info
        if "Panel" in df.columns:
            panels = df["Panel"].unique().tolist()
            adata.uns["panels"] = panels
            adata.uns["n_panels"] = len(panels)

        # Calculate statistics
        missing_pct = (np.isnan(npx_matrix).sum() / npx_matrix.size) * 100

        stats = self._create_base_stats(
            n_samples=adata.n_obs,
            n_proteins=adata.n_vars,
            n_proteins_raw=n_proteins_raw,
            n_samples_raw=n_samples_raw,
            missing_percentage=missing_pct,
            format="long",
            parser="Olink",
            **qc_stats,
            **filter_stats,
            **lod_stats,
        )

        return adata, stats

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _find_column(
        self, df: pd.DataFrame, variants: List[str], required: bool = False
    ) -> Optional[str]:
        """Find column name from list of variants."""
        for variant in variants:
            if variant in df.columns:
                return variant

        if required:
            raise ParsingError(f"Required column not found. Tried: {variants}")

        return None

    def _filter_qc_warnings(
        self, df: pd.DataFrame, sample_col: str
    ) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """Filter samples/measurements with QC warnings."""
        n_before = len(df)
        samples_before = df[sample_col].nunique()

        # Remove rows with QC warnings
        if "QC_Warning" in df.columns:
            mask = df["QC_Warning"].str.lower() == "pass"
            df_filtered = df[mask].copy()
        else:
            df_filtered = df.copy()

        n_removed = n_before - len(df_filtered)
        samples_after = df_filtered[sample_col].nunique()
        samples_removed = samples_before - samples_after

        logger.info(
            f"QC filtering: removed {n_removed} measurements, {samples_removed} samples"
        )

        return df_filtered, {
            "samples_removed_qc": samples_removed,
            "measurements_removed_qc": n_removed,
        }

    def _pivot_npx_matrix(
        self, df: pd.DataFrame, sample_col: str
    ) -> Tuple[np.ndarray, List[str], List[str]]:
        """
        Pivot long-format data to wide NPX matrix.

        Args:
            df: Long-format DataFrame
            sample_col: Sample ID column name

        Returns:
            Tuple of (npx_matrix, sample_order, protein_order)
        """
        logger.info("Pivoting long format to wide matrix...")

        # Pivot: samples as rows, proteins as columns
        pivot_df = df.pivot(index=sample_col, columns="Assay", values="NPX")

        npx_matrix = pivot_df.values
        sample_order = pivot_df.index.tolist()
        protein_order = pivot_df.columns.tolist()

        logger.info(
            f"Pivoted to {len(sample_order)} samples × {len(protein_order)} proteins"
        )

        return npx_matrix, sample_order, protein_order

    def _build_obs_dataframe(
        self, df: pd.DataFrame, sample_col: str, sample_order: List[str]
    ) -> pd.DataFrame:
        """Build sample metadata DataFrame for AnnData.obs."""
        # Get first occurrence of each sample's metadata
        sample_meta = df.groupby(sample_col).first().reindex(sample_order)

        obs_data = {"sample_id": sample_order}

        # Add plate information if available
        if "PlateID" in df.columns:
            obs_data["plate_id"] = sample_meta["PlateID"].values

        # Add well position if available
        if "WellPosition" in df.columns:
            obs_data["well_position"] = sample_meta["WellPosition"].values

        # Add QC warning status
        if "QC_Warning" in df.columns:
            obs_data["qc_warning"] = sample_meta["QC_Warning"].values

        # Add panel info (if single panel, store here; if multi-panel, in uns)
        if "Panel" in df.columns:
            panels = df["Panel"].unique()
            if len(panels) == 1:
                obs_data["panel"] = panels[0]

        # Add custom metadata columns (Subject, Treatment, Time, Site, etc.)
        custom_cols = [
            "Subject",
            "Treatment",
            "Site",
            "Time",
            "Timepoint",
            "Condition",
            "Batch",
        ]
        for col in custom_cols:
            if col in df.columns:
                obs_data[col] = sample_meta[col].values

        obs_df = pd.DataFrame(obs_data)
        obs_df.index = sample_order

        return obs_df

    def _build_var_dataframe(
        self,
        df: pd.DataFrame,
        protein_order: List[str],
        uniprot_col: Optional[str],
        olink_id_col: Optional[str],
        use_gene_names: bool,
    ) -> pd.DataFrame:
        """Build protein metadata DataFrame for AnnData.var."""
        # Get first occurrence of each protein's metadata
        protein_meta = df.groupby("Assay").first().reindex(protein_order)

        var_data = {"assay": protein_order}

        # Add UniProt IDs
        if uniprot_col:
            var_data["uniprot_id"] = (
                protein_meta[uniprot_col].fillna("").astype(str).values
            )

        # Add Olink IDs
        if olink_id_col:
            var_data["olink_id"] = (
                protein_meta[olink_id_col].fillna("").astype(str).values
            )

        # Add panel information
        if "Panel" in df.columns:
            var_data["panel"] = protein_meta["Panel"].fillna("").astype(str).values

        # Add LOD (Limit of Detection)
        if "LOD" in df.columns:
            var_data["lod"] = pd.to_numeric(protein_meta["LOD"], errors="coerce").values

        # Add missing frequency
        if "MissingFreq" in df.columns:
            var_data["missing_freq"] = pd.to_numeric(
                protein_meta["MissingFreq"], errors="coerce"
            ).values

        # Add panel version
        if "Panel_Version" in df.columns:
            var_data["panel_version"] = (
                protein_meta["Panel_Version"].fillna("").astype(str).values
            )

        var_df = pd.DataFrame(var_data)

        # Set index (gene names or UniProt IDs)
        if use_gene_names:
            var_df.index = protein_order  # Assay names (gene symbols)
        elif uniprot_col:
            var_df.index = var_data["uniprot_id"]
        else:
            var_df.index = protein_order

        return var_df

    def _filter_by_detection(
        self, npx_matrix: np.ndarray, var_df: pd.DataFrame, min_samples: int
    ) -> Tuple[np.ndarray, pd.DataFrame, Dict[str, int]]:
        """Filter proteins with insufficient detection."""
        # Count non-missing values per protein
        detected_per_protein = (~np.isnan(npx_matrix)).sum(axis=0)

        # Filter mask
        mask = detected_per_protein >= min_samples
        n_removed = (~mask).sum()

        if n_removed > 0:
            npx_matrix = npx_matrix[:, mask]
            var_df = var_df[mask].copy()
            logger.info(
                f"Filtered {n_removed} proteins with <{min_samples} samples detected"
            )

        return npx_matrix, var_df, {"proteins_removed_min_detection": int(n_removed)}

    def _handle_lod(
        self, npx_matrix: np.ndarray, var_df: pd.DataFrame, strategy: str
    ) -> Tuple[np.ndarray, pd.DataFrame, Dict[str, int]]:
        """
        Handle values below LOD (Limit of Detection).

        Args:
            npx_matrix: NPX matrix
            var_df: Protein metadata with 'lod' column
            strategy: "keep", "flag", "filter"

        Returns:
            Tuple of (matrix, var_df, stats)
        """
        stats = {"proteins_removed_lod": 0}

        if "lod" not in var_df.columns:
            logger.warning("No LOD column found, skipping LOD handling")
            return npx_matrix, var_df, stats

        lod_values = var_df["lod"].values

        if strategy == "flag":
            # Add boolean flag to var
            below_lod_pct = []
            for i, lod in enumerate(lod_values):
                if not np.isnan(lod):
                    below_count = (npx_matrix[:, i] < lod).sum()
                    below_lod_pct.append(below_count / npx_matrix.shape[0])
                else:
                    below_lod_pct.append(0.0)

            var_df["below_lod_pct"] = below_lod_pct

        elif strategy == "filter":
            # Remove proteins with >50% values below LOD
            keep_mask = []
            for i, lod in enumerate(lod_values):
                if not np.isnan(lod):
                    below_pct = (npx_matrix[:, i] < lod).sum() / npx_matrix.shape[0]
                    keep_mask.append(below_pct <= 0.5)
                else:
                    keep_mask.append(True)

            keep_mask = np.array(keep_mask)
            n_removed = (~keep_mask).sum()

            if n_removed > 0:
                npx_matrix = npx_matrix[:, keep_mask]
                var_df = var_df[keep_mask].copy()
                stats["proteins_removed_lod"] = int(n_removed)
                logger.info(f"Filtered {n_removed} proteins with >50% below LOD")

        return npx_matrix, var_df, stats
