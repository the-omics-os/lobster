"""
SomaScan ADAT parser for affinity proteomics data.

This parser handles SomaLogic SomaScan data in ADAT format, supporting
multiple panel sizes (1.3k, 5k, 7k, 11k) with RFU (Relative Fluorescence
Unit) quantification.

File format specifications adapted from SomaDataIO R package structure:
https://github.com/SomaLogic/SomaDataIO

Supported formats:
- ADAT files with structured header blocks (^HEADER, ^COL_DATA, ^ROW_DATA, ^TABLE_BEGIN)
- Tab-separated text despite .adat extension

RFU Scale: Linear relative fluorescence units
Typical range: 100 to 100,000+ RFU
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


class SomaScanParser(ProteomicsParser):
    """
    Parser for SomaScan ADAT output files (affinity proteomics).

    SomaLogic uses SOMAmer (Slow Off-rate Modified Aptamer) technology for
    multiplexed protein quantification. Data is provided as RFU (Relative
    Fluorescence Units) on a linear scale.

    Key Features:
    - Parses structured ADAT header blocks (^HEADER, ^COL_DATA, ^ROW_DATA)
    - Supports multiple SomaScan panel sizes (1.3k, 5k, 7k, 11k)
    - Handles missing/zero RFU values (0 indicates failed measurement)
    - Extracts assay version and analyte metadata (SeqId, Target, UniProt)
    - Fallback for older ADAT exports without proper block headers

    AnnData Structure:
    - X: RFU matrix (samples x analytes), linear scale
    - obs: sample metadata from COL_DATA columns (SampleId, PlateId, etc.)
    - var: analyte metadata from ROW_DATA (SeqId, Target, UniProt, Type)
    - uns: parser metadata, assay version, platform info
    """

    # ADAT block markers
    BLOCK_HEADER = "^HEADER"
    BLOCK_COL_DATA = "^COL_DATA"
    BLOCK_ROW_DATA = "^ROW_DATA"
    BLOCK_TABLE_BEGIN = "^TABLE_BEGIN"
    BLOCK_TABLE_END = "^TABLE_END"

    def __init__(self):
        """Initialize SomaScan parser."""
        super().__init__(name="SomaScan", version="1.0.0")

    def get_supported_formats(self) -> List[str]:
        """Return supported file extensions for SomaScan data."""
        return [".adat"]

    def validate_file(self, file_path: str) -> bool:
        """
        Validate if file is SomaScan ADAT format.

        Checks for:
        1. Correct file extension (.adat)
        2. ADAT header signature (^HEADER or tab-separated with analyte-like columns)

        Args:
            file_path: Path to file

        Returns:
            bool: True if valid SomaScan ADAT file
        """
        try:
            path = self._validate_file_exists(file_path)

            if not self._validate_extension(file_path):
                return False

            # Check for ADAT header signature
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                first_line = f.readline().strip()
                # Standard ADAT starts with ^HEADER
                if first_line.startswith(self.BLOCK_HEADER):
                    return True

                # Some older exports may have !Header_Version
                if "!Header_Version" in first_line or "!StudyMatrix" in first_line:
                    return True

                # Fallback: check if it looks like a tab-separated file with seq IDs
                # (older ADAT exports may lack block markers)
                if "\t" in first_line:
                    # Read a few more lines to check for SeqId patterns
                    for _ in range(10):
                        line = f.readline()
                        if "SeqId" in line or "seq." in line.lower():
                            return True

            logger.debug(f"File does not appear to be ADAT format: {path.name}")
            return False

        except Exception as e:
            logger.debug(f"SomaScan file validation failed: {e}")
            return False

    def parse(
        self,
        file_path: str,
        replace_zero_with_nan: bool = True,
        use_target_names: bool = True,
        **kwargs,
    ) -> Tuple[anndata.AnnData, Dict[str, Any]]:
        """
        Parse SomaScan ADAT file to AnnData.

        Args:
            file_path: Path to ADAT file
            replace_zero_with_nan: Replace 0.0 RFU with NaN (0 indicates failed measurement)
            use_target_names: Use Target (gene symbol) as var index instead of SeqId

        Returns:
            Tuple[AnnData, Dict]: (parsed data, statistics)
        """
        logger.info(f"Parsing SomaScan ADAT file: {file_path}")

        if not self.validate_file(file_path):
            raise FileValidationError(f"Invalid SomaScan ADAT format: {file_path}")

        try:
            path = Path(file_path)

            # Read file lines
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()

            # Check if this is a structured ADAT with block markers
            first_line = lines[0].strip() if lines else ""
            if first_line.startswith(self.BLOCK_HEADER) or "!Header_Version" in first_line:
                adata, stats = self._parse_structured_adat(
                    lines, path, replace_zero_with_nan, use_target_names
                )
            else:
                # Fallback: treat as tab-separated with first column as sample IDs
                adata, stats = self._parse_flat_adat(
                    path, replace_zero_with_nan, use_target_names
                )

            # Add parser metadata
            adata.uns["parser"] = {
                "name": self.name,
                "version": self.version,
                "source_file": path.name,
                "format": "adat",
                "intensity_type": "rfu",
                "platform": "somascan",
            }
            adata.uns["platform"] = "somascan"

            logger.info(f"Successfully parsed SomaScan file: {adata.shape}")
            return adata, stats

        except FileValidationError:
            raise
        except Exception as e:
            logger.error(f"Failed to parse SomaScan ADAT file: {e}")
            raise ParsingError(f"Failed to parse SomaScan ADAT file: {e}") from e

    # =========================================================================
    # STRUCTURED ADAT PARSING (with ^HEADER / ^COL_DATA / ^ROW_DATA blocks)
    # =========================================================================

    def _parse_structured_adat(
        self,
        lines: List[str],
        path: Path,
        replace_zero_with_nan: bool,
        use_target_names: bool,
    ) -> Tuple[anndata.AnnData, Dict[str, Any]]:
        """Parse ADAT with structured block markers."""
        logger.info("Parsing structured ADAT format")

        # 1. Parse header blocks
        header_data, col_data_fields, row_data, table_start_idx = (
            self._parse_header_blocks(lines)
        )

        # 2. Read data matrix starting from TABLE_BEGIN
        data_lines = []
        for i in range(table_start_idx, len(lines)):
            line = lines[i].strip()
            if line.startswith(self.BLOCK_TABLE_END) or not line:
                break
            data_lines.append(line)

        if not data_lines:
            raise ParsingError("No data found after ^TABLE_BEGIN marker")

        # First data line contains column headers
        header_line = data_lines[0]
        col_headers = header_line.split("\t")

        # Identify sample metadata columns vs analyte columns
        # Sample metadata columns are the COL_DATA fields
        n_meta_cols = len(col_data_fields) if col_data_fields else 0

        # If we have row_data, analyte columns start after metadata columns
        analyte_headers = col_headers[n_meta_cols:] if n_meta_cols > 0 else col_headers[1:]
        meta_headers = col_headers[:n_meta_cols] if n_meta_cols > 0 else [col_headers[0]]

        # Parse data rows
        sample_data = []
        sample_meta = []
        for line in data_lines[1:]:
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) <= n_meta_cols:
                continue

            meta_values = parts[:n_meta_cols] if n_meta_cols > 0 else [parts[0]]
            rfu_values = parts[n_meta_cols:] if n_meta_cols > 0 else parts[1:]

            # Convert RFU values to float
            rfu_row = []
            for v in rfu_values:
                try:
                    val = float(v.strip()) if v.strip() else np.nan
                except (ValueError, TypeError):
                    val = np.nan
                rfu_row.append(val)

            sample_data.append(rfu_row)
            sample_meta.append(meta_values)

        if not sample_data:
            raise ParsingError("No sample data rows found in ADAT file")

        # 3. Build matrices
        # Ensure all rows have the same number of analyte columns
        n_analytes = len(analyte_headers)
        rfu_matrix = np.full((len(sample_data), n_analytes), np.nan, dtype=np.float32)
        for i, row in enumerate(sample_data):
            n_cols = min(len(row), n_analytes)
            rfu_matrix[i, :n_cols] = row[:n_cols]

        # Replace zeros with NaN (0 RFU indicates failed measurement)
        if replace_zero_with_nan:
            zero_count = np.sum(rfu_matrix == 0.0)
            rfu_matrix[rfu_matrix == 0.0] = np.nan
        else:
            zero_count = 0

        # 4. Build obs (sample metadata)
        obs_df = pd.DataFrame(sample_meta, columns=meta_headers[:len(sample_meta[0])])
        # Use first column as sample ID index
        sample_id_col = meta_headers[0] if meta_headers else "sample_id"
        obs_df.index = obs_df.iloc[:, 0].astype(str)
        obs_df.index.name = None

        # 5. Build var (analyte metadata)
        var_data = {"analyte_id": analyte_headers}
        if row_data:
            # row_data is a dict of {field_name: [values per analyte]}
            for field_name, values in row_data.items():
                if len(values) == n_analytes:
                    var_data[field_name.lower()] = values

        var_df = pd.DataFrame(var_data)

        # Set index
        if use_target_names and "target" in var_df.columns:
            # Use Target (gene symbol) as index, handle duplicates
            targets = var_df["target"].tolist()
            targets = self._make_unique_index(targets)
            var_df.index = targets
        else:
            var_df.index = analyte_headers
        var_df.index.name = None

        # 6. Create AnnData
        adata = anndata.AnnData(X=rfu_matrix, obs=obs_df, var=var_df)

        # Store header metadata
        if header_data:
            for key, value in header_data.items():
                adata.uns[key.lower().replace("!", "")] = value

        assay_version = header_data.get("!AssayVersion", header_data.get("!StudyMatrix", "unknown"))
        adata.uns["assay_version"] = assay_version

        # 7. Calculate statistics
        missing_count = np.isnan(rfu_matrix).sum()
        total_values = rfu_matrix.size
        missing_pct = (missing_count / total_values) * 100 if total_values > 0 else 0

        stats = self._create_base_stats(
            n_samples=adata.n_obs,
            n_proteins=adata.n_vars,
            n_proteins_raw=n_analytes,
            n_samples_raw=len(sample_data),
            missing_percentage=missing_pct,
            format="structured_adat",
            parser="SomaScan",
            assay_version=str(assay_version),
            n_missing=int(missing_count),
            n_zero_replaced=int(zero_count) if replace_zero_with_nan else 0,
        )

        return adata, stats

    def _parse_header_blocks(
        self, lines: List[str]
    ) -> Tuple[Dict[str, str], List[str], Dict[str, List[str]], int]:
        """
        Parse ADAT header blocks to extract metadata.

        Returns:
            Tuple of:
            - header_data: key-value pairs from ^HEADER block
            - col_data_fields: list of sample metadata column names
            - row_data: dict of {field_name: [values per analyte]}
            - table_start_idx: line index where data matrix begins
        """
        header_data: Dict[str, str] = {}
        col_data_fields: List[str] = []
        row_data: Dict[str, List[str]] = {}
        table_start_idx = 0

        current_block: Optional[str] = None
        i = 0

        while i < len(lines):
            line = lines[i].strip()

            # Block markers
            if line.startswith("^"):
                if line.startswith(self.BLOCK_HEADER):
                    current_block = "HEADER"
                elif line.startswith(self.BLOCK_COL_DATA):
                    current_block = "COL_DATA"
                elif line.startswith(self.BLOCK_ROW_DATA):
                    current_block = "ROW_DATA"
                elif line.startswith(self.BLOCK_TABLE_BEGIN):
                    table_start_idx = i + 1
                    break
                i += 1
                continue

            # Parse content based on current block
            if current_block == "HEADER":
                if line.startswith("!") and "\t" in line:
                    parts = line.split("\t", 1)
                    header_data[parts[0]] = parts[1].strip() if len(parts) > 1 else ""
                elif "=" in line:
                    key, _, value = line.partition("=")
                    header_data[key.strip()] = value.strip()

            elif current_block == "COL_DATA":
                if line and not line.startswith("!"):
                    # COL_DATA lists sample metadata column names
                    col_data_fields.append(line.split("\t")[0] if "\t" in line else line)
                elif line.startswith("!"):
                    parts = line.split("\t", 1)
                    col_data_fields.append(parts[0].lstrip("!"))

            elif current_block == "ROW_DATA":
                if line and "\t" in line:
                    parts = line.split("\t")
                    field_name = parts[0].lstrip("!")
                    row_data[field_name] = parts[1:]

            i += 1

        return header_data, col_data_fields, row_data, table_start_idx

    # =========================================================================
    # FLAT ADAT PARSING (fallback for older exports)
    # =========================================================================

    def _parse_flat_adat(
        self,
        path: Path,
        replace_zero_with_nan: bool,
        use_target_names: bool,
    ) -> Tuple[anndata.AnnData, Dict[str, Any]]:
        """Parse ADAT as flat tab-separated file (fallback for older exports)."""
        logger.info("Parsing flat ADAT format (fallback)")

        # Read as tab-separated
        df = pd.read_csv(path, sep="\t", index_col=0)

        # First column is sample IDs, remaining are analyte measurements
        sample_ids = df.index.astype(str).tolist()
        analyte_names = df.columns.tolist()

        rfu_matrix = df.values.astype(np.float32)

        # Replace zeros with NaN
        if replace_zero_with_nan:
            zero_count = np.sum(rfu_matrix == 0.0)
            rfu_matrix[rfu_matrix == 0.0] = np.nan
        else:
            zero_count = 0

        # Build obs
        obs_df = pd.DataFrame({"sample_id": sample_ids}, index=sample_ids)

        # Build var
        var_df = pd.DataFrame({"analyte_id": analyte_names}, index=analyte_names)

        adata = anndata.AnnData(X=rfu_matrix, obs=obs_df, var=var_df)
        adata.uns["assay_version"] = "unknown"

        missing_count = np.isnan(rfu_matrix).sum()
        total_values = rfu_matrix.size
        missing_pct = (missing_count / total_values) * 100 if total_values > 0 else 0

        stats = self._create_base_stats(
            n_samples=adata.n_obs,
            n_proteins=adata.n_vars,
            n_proteins_raw=len(analyte_names),
            n_samples_raw=len(sample_ids),
            missing_percentage=missing_pct,
            format="flat_adat",
            parser="SomaScan",
            assay_version="unknown",
            n_missing=int(missing_count),
            n_zero_replaced=int(zero_count) if replace_zero_with_nan else 0,
        )

        return adata, stats

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    @staticmethod
    def _make_unique_index(names: List[str]) -> List[str]:
        """Make index names unique by appending suffixes for duplicates."""
        seen: Dict[str, int] = {}
        unique_names = []
        for name in names:
            if name in seen:
                seen[name] += 1
                unique_names.append(f"{name}_{seen[name]}")
            else:
                seen[name] = 0
                unique_names.append(name)
        return unique_names
