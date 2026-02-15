"""
Proteomics parsers namespace package.

Re-exports parser classes from lobster-proteomics package and provides
the get_parser_for_file() utility for automatic parser selection.

When lobster-proteomics is not installed, parsers will not be available
and get_parser_for_file() returns None.
"""

from pathlib import Path
from typing import Optional

from lobster.services.data_access.proteomics_parsers.base_parser import (
    FileValidationError,
    ParsingError,
    ProteomicsParser,
    ProteomicsParserError,
)
from lobster.utils.logger import get_logger

logger = get_logger(__name__)

# Lazy re-exports: attempt to import concrete parsers from lobster-proteomics package.
# These will be None if lobster-proteomics is not installed.
MaxQuantParser = None
DIANNParser = None
SpectronautParser = None
OlinkParser = None

try:
    from lobster.services.data_access.maxquant_parser import (
        MaxQuantParser,  # type: ignore[assignment]
    )
except ImportError:
    pass

try:
    from lobster.services.data_access.diann_parser import (
        DIANNParser,  # type: ignore[assignment]
    )
except ImportError:
    pass

try:
    from lobster.services.data_access.spectronaut_parser import (
        SpectronautParser,  # type: ignore[assignment]
    )
except ImportError:
    pass

try:
    from lobster.services.data_access.olink_parser import (
        OlinkParser,  # type: ignore[assignment]
    )
except ImportError:
    pass


def get_parser_for_file(file_path: str) -> Optional[ProteomicsParser]:
    """
    Auto-detect and return the appropriate parser for a proteomics file.

    Iterates through available parsers and returns the first one whose
    validate_file() returns True for the given file.

    Args:
        file_path: Path to the proteomics output file.

    Returns:
        An instantiated parser if one matches, or None if no parser
        can handle the file (or lobster-proteomics is not installed).
    """
    # Build list of available parser classes (skip None entries)
    available_parsers = [
        cls
        for cls in (MaxQuantParser, DIANNParser, SpectronautParser, OlinkParser)
        if cls is not None
    ]

    if not available_parsers:
        logger.warning(
            "No proteomics parsers available. "
            "Install lobster-proteomics: pip install lobster-proteomics"
        )
        return None

    for parser_cls in available_parsers:
        try:
            parser = parser_cls()
            if parser.validate_file(file_path):
                logger.debug(
                    f"Selected {parser_cls.__name__} for {Path(file_path).name}"
                )
                return parser
        except Exception as e:
            logger.debug(
                f"{parser_cls.__name__} validation failed for {file_path}: {e}"
            )
            continue

    logger.warning(f"No parser found for file: {Path(file_path).name}")
    return None


__all__ = [
    "ProteomicsParser",
    "ProteomicsParserError",
    "FileValidationError",
    "ParsingError",
    "MaxQuantParser",
    "DIANNParser",
    "SpectronautParser",
    "OlinkParser",
    "get_parser_for_file",
]
