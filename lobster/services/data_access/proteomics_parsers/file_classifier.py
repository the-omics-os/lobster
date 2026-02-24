"""
Lightweight file classifier for proteomics data files.

Reads a file ONCE and returns a FileClassification with format, delimiter,
orientation, dimensions, confidence, and human-readable diagnostics.
Used by get_parser_for_file() to route to the correct parser or adapter path.
"""

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from lobster.utils.logger import get_logger

logger = get_logger(__name__)

# Vendor column signatures (lowercase keys checked against column names)
_VENDOR_SIGNATURES = {
    "maxquant": {
        "required": ["protein ids"],
        "optional": ["lfq intensity", "intensity ", "iq intensity"],
    },
    "diann": {
        "required": ["protein.group"],
        "optional": ["run", "precursor.quantity", "pg.quantity"],
    },
    "spectronaut": {
        "required": [],
        "optional": ["pg.quantity", "eg.quantity"],
        "prefix": "pg.",
    },
    "olink_npx": {
        "required": ["assay", "npx"],
        "optional": ["olink", "sampleid", "panel"],
    },
    "luminex": {
        "required": [],
        "optional": ["mfi", "luminex", "bio-plex", "bioplex", "median fi"],
    },
}


@dataclass
class FileClassification:
    """Result of classifying a proteomics data file."""

    format: str  # "maxquant", "diann", "spectronaut", "olink_npx", "somascan_adat", "luminex", "generic_matrix", "unknown"
    delimiter: str = ","
    orientation: str = "unknown"  # "features_as_rows", "features_as_cols", "unknown"
    n_rows: int = 0
    n_cols: int = 0
    first_col_type: str = "unknown"  # "identifiers", "numeric", "mixed"
    numeric_col_ratio: float = 0.0
    confidence: float = 0.0
    diagnostics: str = ""
    header_cols: list = field(default_factory=list)


class FileClassifier:
    """Classifies proteomics files by reading them once."""

    @staticmethod
    def classify(file_path: str) -> FileClassification:
        """
        Classify a proteomics data file.

        Reads only the header + first ~20 rows to determine the file format,
        orientation, and dimensions without fully parsing.

        Args:
            file_path: Path to the data file.

        Returns:
            FileClassification with detected metadata.
        """
        path = Path(file_path)
        if not path.exists():
            return FileClassification(
                format="unknown",
                diagnostics=f"File not found: {path}",
            )

        ext = path.suffix.lower()

        # --- Extension-based shortcuts ---
        if ext == ".adat":
            return FileClassification(
                format="somascan_adat",
                confidence=0.95,
                diagnostics=f"SomaScan ADAT file detected from extension ({path.name})",
            )
        if ext == ".npx":
            return FileClassification(
                format="olink_npx",
                confidence=0.95,
                diagnostics=f"Olink NPX file detected from extension ({path.name})",
            )
        if ext == ".h5ad":
            return FileClassification(
                format="unknown",
                confidence=0.0,
                diagnostics="H5AD files should be loaded directly via load_modality, not through import tools.",
            )

        # --- Content-based classification for CSV/TSV/TXT ---
        if ext not in (".csv", ".tsv", ".txt", ".tab"):
            return FileClassification(
                format="unknown",
                diagnostics=f"Unsupported extension '{ext}'. Expected .csv, .tsv, .txt, .adat, or .npx.",
            )

        return FileClassifier._classify_delimited(path)

    @staticmethod
    def _classify_delimited(path: Path) -> FileClassification:
        """Classify a delimited text file by inspecting header + sample rows."""
        try:
            with open(path, "r", errors="replace") as f:
                # Read up to 25 lines for classification
                lines = []
                for _ in range(25):
                    line = f.readline()
                    if not line:
                        break
                    lines.append(line)

            if not lines:
                return FileClassification(
                    format="unknown", diagnostics="File is empty."
                )

            # Detect delimiter
            delimiter = FileClassifier._detect_delimiter(lines[0])

            # Parse header
            header_cols = next(csv.reader([lines[0]], delimiter=delimiter))
            header_cols = [c.strip() for c in header_cols]
            header_lower = [c.lower() for c in header_cols]
            n_cols = len(header_cols)

            # Count total rows (fast: count newlines)
            n_rows = FileClassifier._count_rows(path) - 1  # subtract header

            # --- Check vendor signatures ---
            vendor_format = FileClassifier._match_vendor(header_lower)
            if vendor_format:
                return FileClassification(
                    format=vendor_format,
                    delimiter=delimiter,
                    n_rows=n_rows,
                    n_cols=n_cols,
                    confidence=0.85,
                    diagnostics=f"Detected {vendor_format} format from column signatures in {path.name}",
                    header_cols=header_cols,
                )

            # --- Generic matrix detection ---
            # Parse a few data rows to inspect types
            data_rows = []
            for line in lines[1:]:
                row = next(csv.reader([line], delimiter=delimiter))
                if len(row) >= 2:
                    data_rows.append(row)

            if not data_rows:
                return FileClassification(
                    format="unknown",
                    delimiter=delimiter,
                    n_rows=n_rows,
                    n_cols=n_cols,
                    diagnostics=f"File has header ({n_cols} columns) but no data rows.",
                    header_cols=header_cols,
                )

            # Check first column type across data rows
            first_col_values = [r[0].strip() for r in data_rows if r]
            first_col_numeric = sum(
                1 for v in first_col_values if FileClassifier._is_numeric(v)
            )
            first_col_type = (
                "numeric"
                if first_col_numeric > len(first_col_values) * 0.8
                else "identifiers"
            )

            # Check remaining columns for numeric content
            numeric_cols = 0
            for col_idx in range(1, min(n_cols, len(data_rows[0]))):
                col_vals = [
                    r[col_idx].strip()
                    for r in data_rows
                    if col_idx < len(r) and r[col_idx].strip()
                ]
                if col_vals and all(
                    FileClassifier._is_numeric(v) or v.upper() in ("", "NA", "NAN", "NULL", "INF", "-INF")
                    for v in col_vals
                ):
                    numeric_cols += 1

            # Ratio of numeric columns (excluding first col)
            non_first_cols = max(n_cols - 1, 1)
            numeric_col_ratio = numeric_cols / non_first_cols

            # Decide: is this a generic expression matrix?
            is_generic = first_col_type == "identifiers" and numeric_col_ratio >= 0.7

            if is_generic:
                # Auto-detect orientation
                # If n_rows >> n_cols (3x+), features are likely rows
                if n_rows > n_cols * 3:
                    orientation = "features_as_rows"
                elif n_cols > n_rows * 3:
                    orientation = "features_as_cols"
                else:
                    # Ambiguous — check if header looks like sample names or gene/protein IDs
                    orientation = FileClassifier._guess_orientation(
                        header_cols[1:], first_col_values, n_rows, n_cols
                    )

                return FileClassification(
                    format="generic_matrix",
                    delimiter=delimiter,
                    orientation=orientation,
                    n_rows=n_rows,
                    n_cols=n_cols,
                    first_col_type=first_col_type,
                    numeric_col_ratio=numeric_col_ratio,
                    confidence=0.70,
                    diagnostics=(
                        f"Generic expression matrix: {n_rows} rows x {n_cols} cols "
                        f"(orientation: {orientation}). "
                        f"First column contains {first_col_type}, "
                        f"{numeric_col_ratio:.0%} of remaining columns are numeric."
                    ),
                    header_cols=header_cols,
                )

            # Unknown: provide diagnostics
            return FileClassification(
                format="unknown",
                delimiter=delimiter,
                n_rows=n_rows,
                n_cols=n_cols,
                first_col_type=first_col_type,
                numeric_col_ratio=numeric_col_ratio,
                diagnostics=(
                    f"Could not classify '{path.name}': {n_rows} rows x {n_cols} cols. "
                    f"First col: {first_col_type}, numeric ratio: {numeric_col_ratio:.0%}. "
                    f"First 5 columns: {header_cols[:5]}"
                ),
                header_cols=header_cols,
            )

        except Exception as e:
            logger.warning(f"FileClassifier error on {path}: {e}")
            return FileClassification(
                format="unknown",
                diagnostics=f"Error reading file: {e}",
            )

    @staticmethod
    def _detect_delimiter(first_line: str) -> str:
        """Detect delimiter from the first line."""
        tab_count = first_line.count("\t")
        comma_count = first_line.count(",")
        if tab_count > comma_count and tab_count >= 2:
            return "\t"
        return ","

    @staticmethod
    def _count_rows(path: Path) -> int:
        """Fast line count."""
        count = 0
        with open(path, "rb") as f:
            for _ in f:
                count += 1
        return count

    @staticmethod
    def _match_vendor(header_lower: list) -> Optional[str]:
        """Match column names against vendor signatures."""
        header_joined = " ".join(header_lower)

        for vendor, sigs in _VENDOR_SIGNATURES.items():
            # Check required columns
            required_match = all(
                any(req in col for col in header_lower) for req in sigs["required"]
            )
            if not required_match and sigs["required"]:
                continue

            # Check optional columns (need at least 1 match if no required)
            optional_match = sum(
                1
                for opt in sigs.get("optional", [])
                if any(opt in col for col in header_lower)
            )

            # Check prefix pattern
            prefix = sigs.get("prefix")
            prefix_match = (
                sum(1 for col in header_lower if col.startswith(prefix))
                if prefix
                else 0
            )

            if sigs["required"] and required_match:
                # Required cols found — high confidence
                if optional_match > 0 or not sigs.get("optional"):
                    return vendor
            elif not sigs["required"]:
                # No required cols — need strong optional/prefix signal
                if optional_match >= 2 or prefix_match >= 3:
                    return vendor

        return None

    @staticmethod
    def _is_numeric(value: str) -> bool:
        """Check if a string value is numeric."""
        if not value:
            return False
        try:
            float(value)
            return True
        except ValueError:
            return False

    @staticmethod
    def _guess_orientation(
        col_names: list, row_ids: list, n_rows: int, n_cols: int
    ) -> str:
        """
        Guess orientation when row/col ratio is ambiguous.

        Heuristic: if column headers look like sample IDs (short, alphanumeric,
        contain digits) and row IDs look like gene/protein names (contain letters,
        longer), features are rows.
        """
        # Check if column names look like sample IDs
        avg_col_len = sum(len(c) for c in col_names[:20]) / max(len(col_names[:20]), 1)
        avg_row_len = sum(len(r) for r in row_ids[:20]) / max(len(row_ids[:20]), 1)

        # Sample IDs tend to be shorter; gene/protein names tend to be longer
        if avg_row_len > avg_col_len * 1.5 and n_rows > n_cols:
            return "features_as_rows"
        if avg_col_len > avg_row_len * 1.5 and n_cols > n_rows:
            return "features_as_cols"

        # Default: assume standard matrix orientation (samples=rows, features=cols)
        # unless there are way more rows
        if n_rows > n_cols:
            return "features_as_rows"
        return "features_as_cols"
