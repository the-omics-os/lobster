"""
Proteomics data parsers for common mass spectrometry output formats.

This module provides production-ready parsers for converting vendor-specific
proteomics output formats into standardized AnnData objects suitable for
downstream analysis with Lobster's proteomics services.

Supported Formats:
    - MaxQuant proteinGroups.txt (DDA mass spectrometry)
    - DIA-NN report.tsv/report.parquet (DIA mass spectrometry)
    - Olink NPX files .csv/.xlsx (affinity proteomics)

Parser Design:
    All parsers inherit from ProteomicsParser and implement:
    - parse(): Main parsing method returning (AnnData, stats) tuple
    - validate_file(): Format-specific file validation
    - get_supported_formats(): List of supported file extensions

AnnData Convention:
    - X matrix: samples (rows/obs) x proteins (columns/var)
    - obs: sample-level metadata
    - var: protein-level metadata
    - uns: analysis-wide metadata

Example usage:
    >>> from lobster.services.data_access.proteomics_parsers import MaxQuantParser, DIANNParser, OlinkParser
    >>>
    >>> # Parse MaxQuant output
    >>> mq_parser = MaxQuantParser()
    >>> if mq_parser.validate_file("proteinGroups.txt"):
    ...     adata, stats = mq_parser.parse("proteinGroups.txt", intensity_type="lfq")
    ...     print(f"MaxQuant: {adata.n_obs} samples, {adata.n_vars} proteins")
    >>>
    >>> # Parse DIA-NN output
    >>> diann_parser = DIANNParser()
    >>> if diann_parser.validate_file("report.tsv"):
    ...     adata, stats = diann_parser.parse("report.tsv", quantity_column="Precursor.Normalised")
    ...     print(f"DIA-NN: {adata.n_obs} samples, {adata.n_vars} proteins")
    >>>
    >>> # Parse Olink NPX output
    >>> olink_parser = OlinkParser()
    >>> if olink_parser.validate_file("olink_npx_data.csv"):
    ...     adata, stats = olink_parser.parse("olink_npx_data.csv")
    ...     print(f"Olink: {adata.n_obs} samples, {adata.n_vars} proteins")

Auto-detection:
    >>> from lobster.services.data_access.proteomics_parsers import get_parser_for_file
    >>> parser = get_parser_for_file("proteinGroups.txt")  # Returns MaxQuantParser
    >>> parser = get_parser_for_file("report.parquet")     # Returns DIANNParser
    >>> parser = get_parser_for_file("olink_npx.csv")      # Returns OlinkParser
"""

from lobster.services.data_access.proteomics_parsers.base_parser import (
    FileValidationError,
    ParsingError,
    ProteomicsParser,
    ProteomicsParserError,
)
from lobster.services.data_access.proteomics_parsers.diann_parser import DIANNParser
from lobster.services.data_access.proteomics_parsers.maxquant_parser import (
    MaxQuantParser,
)
from lobster.services.data_access.proteomics_parsers.olink_parser import OlinkParser

__all__ = [
    # Base classes and exceptions
    "ProteomicsParser",
    "ProteomicsParserError",
    "FileValidationError",
    "ParsingError",
    # Parser implementations
    "MaxQuantParser",
    "DIANNParser",
    "OlinkParser",
    # Utility functions
    "get_parser_for_file",
    "get_available_parsers",
]


def get_available_parsers() -> dict:
    """
    Get dictionary of all available proteomics parsers.

    Returns:
        Dict[str, ProteomicsParser]: Mapping of parser names to parser classes

    Example:
        >>> parsers = get_available_parsers()
        >>> print(list(parsers.keys()))
        ['maxquant', 'diann', 'olink']
    """
    return {
        "maxquant": MaxQuantParser,
        "diann": DIANNParser,
        "olink": OlinkParser,
    }


def get_parser_for_file(file_path: str) -> ProteomicsParser:
    """
    Auto-detect and return appropriate parser for a proteomics file.

    Tries each available parser in sequence and returns the first one
    that successfully validates the file.

    Args:
        file_path: Path to the proteomics output file

    Returns:
        ProteomicsParser: Appropriate parser instance for the file

    Raises:
        ValueError: If no parser can handle the file

    Example:
        >>> parser = get_parser_for_file("proteinGroups.txt")
        >>> isinstance(parser, MaxQuantParser)
        True
        >>> parser = get_parser_for_file("report.parquet")
        >>> isinstance(parser, DIANNParser)
        True
    """
    # Try parsers in order of specificity
    parser_classes = [
        MaxQuantParser,  # Try MaxQuant first (most specific - proteinGroups.txt)
        OlinkParser,  # Olink second (specific - NPX/Assay/UniProt columns)
        DIANNParser,  # DIA-NN last (more general column names)
    ]

    for parser_class in parser_classes:
        parser = parser_class()
        if parser.validate_file(file_path):
            return parser

    # If no parser validated, raise error with helpful message
    available = ", ".join(get_available_parsers().keys())
    raise ValueError(
        f"No parser found for file: {file_path}. "
        f"Available parsers: {available}. "
        "Please check that the file is a valid proteomics output format."
    )
