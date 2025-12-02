"""
Abstract base class for proteomics data parsers.

This module defines the interface for all proteomics output parsers (MaxQuant, DIA-NN,
Spectronaut, etc.). Each parser converts vendor-specific output formats into standardized
AnnData objects suitable for downstream analysis.

The parser pattern follows the service architecture where:
- Parsers are stateless (no data manager dependency required)
- All methods return consistent data structures
- Validation is built-in to ensure data quality

Example usage:
    >>> from lobster.services.data_access.proteomics_parsers import MaxQuantParser
    >>> parser = MaxQuantParser()
    >>> if parser.validate_file("proteinGroups.txt"):
    ...     adata, stats = parser.parse("proteinGroups.txt")
    ...     print(f"Loaded {adata.n_obs} samples, {adata.n_vars} proteins")
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import anndata

from lobster.utils.logger import get_logger

logger = get_logger(__name__)


class ProteomicsParserError(Exception):
    """Base exception for proteomics parsing operations."""

    pass


class FileValidationError(ProteomicsParserError):
    """Raised when file validation fails."""

    pass


class ParsingError(ProteomicsParserError):
    """Raised when parsing a file fails."""

    pass


class ProteomicsParser(ABC):
    """
    Abstract base class for proteomics output file parsers.

    This class defines the interface that all proteomics parsers must implement.
    Parsers are responsible for converting vendor-specific output formats
    (MaxQuant, DIA-NN, Spectronaut, etc.) into standardized AnnData objects.

    AnnData Structure Convention:
        - X matrix: samples (rows/obs) x proteins (columns/var)
        - obs: sample-level metadata (sample names, conditions, batches)
        - var: protein-level metadata (IDs, gene names, sequence coverage)
        - uns: analysis-wide metadata (software version, parameters, etc.)

    Subclasses must implement:
        - parse(): Main parsing logic
        - validate_file(): Format-specific file validation
        - get_supported_formats(): List of supported file extensions

    Attributes:
        name: Human-readable parser name
        version: Parser version for provenance tracking
    """

    def __init__(self, name: str = "base", version: str = "1.0.0"):
        """
        Initialize the proteomics parser.

        Args:
            name: Human-readable parser name for logging/provenance
            version: Parser version string
        """
        self.name = name
        self.version = version
        logger.debug(f"Initialized {self.name} parser v{self.version}")

    @abstractmethod
    def parse(
        self,
        file_path: str,
        **kwargs,
    ) -> Tuple[anndata.AnnData, Dict[str, Any]]:
        """
        Parse proteomics output file into AnnData with statistics.

        This is the main entry point for converting vendor-specific formats
        into standardized AnnData objects. Implementations should handle
        all format-specific quirks and produce consistent output.

        Args:
            file_path: Path to the proteomics output file
            **kwargs: Parser-specific options (e.g., intensity_type, filter_contaminants)

        Returns:
            Tuple containing:
                - anndata.AnnData: Parsed data with samples as obs, proteins as var
                - Dict[str, Any]: Parsing statistics including:
                    - n_samples: Number of samples
                    - n_proteins: Number of proteins (after filtering)
                    - n_proteins_raw: Number of proteins before filtering
                    - n_contaminants: Number of contaminants removed
                    - n_reverse_hits: Number of reverse hits removed
                    - missing_percentage: Overall missing value percentage
                    - intensity_type: Type of intensity values used
                    - parser_name: Name of parser used
                    - parser_version: Version of parser

        Raises:
            FileValidationError: If file format is invalid
            ParsingError: If parsing fails
            FileNotFoundError: If file does not exist

        Example:
            >>> parser = MaxQuantParser()
            >>> adata, stats = parser.parse(
            ...     "proteinGroups.txt",
            ...     intensity_type="lfq",
            ...     filter_contaminants=True
            ... )
        """
        pass

    @abstractmethod
    def validate_file(self, file_path: str) -> bool:
        """
        Check if file is valid format for this parser.

        Performs format-specific validation to determine if a file can be
        parsed by this parser. This should check file existence, extension,
        and ideally validate header columns.

        Args:
            file_path: Path to the file to validate

        Returns:
            bool: True if file is valid for this parser, False otherwise

        Example:
            >>> parser = MaxQuantParser()
            >>> parser.validate_file("proteinGroups.txt")
            True
            >>> parser.validate_file("random_file.csv")
            False
        """
        pass

    @abstractmethod
    def get_supported_formats(self) -> List[str]:
        """
        Return list of supported file extensions.

        Returns:
            List[str]: File extensions including the dot (e.g., [".txt", ".tsv"])

        Example:
            >>> parser = DIANNParser()
            >>> parser.get_supported_formats()
            ['.tsv', '.parquet']
        """
        pass

    def _validate_file_exists(self, file_path: str) -> Path:
        """
        Validate that file exists and return Path object.

        Args:
            file_path: Path to the file

        Returns:
            Path: Validated Path object

        Raises:
            FileNotFoundError: If file does not exist
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        if not path.is_file():
            raise FileValidationError(f"Path is not a file: {file_path}")
        return path

    def _validate_extension(self, file_path: str) -> bool:
        """
        Check if file extension is supported.

        Args:
            file_path: Path to the file

        Returns:
            bool: True if extension is supported
        """
        path = Path(file_path)
        return path.suffix.lower() in [ext.lower() for ext in self.get_supported_formats()]

    def _create_base_stats(
        self,
        n_samples: int,
        n_proteins: int,
        n_proteins_raw: int,
        missing_percentage: float,
        **extra_stats,
    ) -> Dict[str, Any]:
        """
        Create standardized statistics dictionary.

        Args:
            n_samples: Number of samples parsed
            n_proteins: Number of proteins after filtering
            n_proteins_raw: Number of proteins before filtering
            missing_percentage: Overall missing value percentage
            **extra_stats: Additional parser-specific statistics

        Returns:
            Dict[str, Any]: Standardized statistics dictionary
        """
        stats = {
            "n_samples": n_samples,
            "n_proteins": n_proteins,
            "n_proteins_raw": n_proteins_raw,
            "n_proteins_filtered": n_proteins_raw - n_proteins,
            "missing_percentage": round(missing_percentage, 2),
            "parser_name": self.name,
            "parser_version": self.version,
        }
        stats.update(extra_stats)
        return stats

    def __repr__(self) -> str:
        """String representation for debugging."""
        formats = ", ".join(self.get_supported_formats())
        return f"{self.__class__.__name__}(name='{self.name}', formats=[{formats}])"
