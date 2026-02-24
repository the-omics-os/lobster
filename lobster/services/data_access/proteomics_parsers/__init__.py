"""
Proteomics parsers namespace package.

Re-exports parser classes from lobster-proteomics package and provides
the get_parser_for_file() utility for automatic parser selection.

Uses a detect-then-route pattern: FileClassifier inspects the file once
and returns a classification, which is used to route to the correct parser
or signal a generic matrix fallback.

When lobster-proteomics is not installed, parsers will not be available
and get_parser_for_file() returns (None, classification).
"""

from pathlib import Path
from typing import Optional, Tuple

from lobster.services.data_access.proteomics_parsers.base_parser import (
    FileValidationError,
    ParsingError,
    ProteomicsParser,
    ProteomicsParserError,
)
from lobster.services.data_access.proteomics_parsers.file_classifier import (
    FileClassification,
    FileClassifier,
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

# Map classifier format names to parser classes
_FORMAT_TO_PARSER = {
    "maxquant": lambda: MaxQuantParser,
    "diann": lambda: DIANNParser,
    "spectronaut": lambda: SpectronautParser,
    "olink_npx": lambda: OlinkParser,
}


def get_parser_for_file(
    file_path: str,
) -> Tuple[Optional[ProteomicsParser], FileClassification]:
    """
    Classify a proteomics file and return the appropriate parser.

    Uses FileClassifier to inspect the file once, then routes to the
    correct parser. Returns both the parser and the classification so
    callers get useful context even when no vendor parser matches.

    Args:
        file_path: Path to the proteomics output file.

    Returns:
        Tuple of (parser_instance_or_None, FileClassification).
        - If a vendor parser matches: (parser, classification)
        - If generic matrix detected: (None, classification) with format="generic_matrix"
        - If unknown: (None, classification) with diagnostics
    """
    classification = FileClassifier.classify(file_path)

    # Try to get a vendor parser based on classification
    parser_factory = _FORMAT_TO_PARSER.get(classification.format)
    if parser_factory:
        parser_cls = parser_factory()
        if parser_cls is not None:
            try:
                parser = parser_cls()
                logger.debug(
                    f"Selected {parser_cls.__name__} for {Path(file_path).name} "
                    f"(confidence: {classification.confidence:.0%})"
                )
                return parser, classification
            except Exception as e:
                logger.debug(f"Failed to instantiate {parser_cls.__name__}: {e}")

    # For generic_matrix, somascan_adat (handled by SomaScan parser externally),
    # or unknown — return None with classification diagnostics
    if classification.format not in ("generic_matrix", "unknown"):
        logger.warning(
            f"Classified as {classification.format} but no parser available for "
            f"{Path(file_path).name}. Install the appropriate package."
        )
    elif classification.format == "unknown":
        logger.warning(
            f"Could not classify file: {Path(file_path).name}. "
            f"{classification.diagnostics}"
        )

    return None, classification


__all__ = [
    "ProteomicsParser",
    "ProteomicsParserError",
    "FileValidationError",
    "ParsingError",
    "FileClassification",
    "FileClassifier",
    "MaxQuantParser",
    "DIANNParser",
    "SpectronautParser",
    "OlinkParser",
    "get_parser_for_file",
]
