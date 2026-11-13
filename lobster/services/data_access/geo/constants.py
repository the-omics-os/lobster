"""
GEO service constants, enums, and dataclasses.

Contains all shared constants used across the GEO service modules:
- Enums for data sources, types, and compatibility status
- Dataclasses for download strategy and results
- Platform registry for early validation
- Keyword patterns for platform detection
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

import pandas as pd


class GEODataSource(Enum):
    """Enumeration of data source types for GEO downloads."""

    GEOPARSE = "geoparse"
    SOFT_FILE = "soft_file"
    SUPPLEMENTARY = "supplementary"
    TAR_ARCHIVE = "tar_archive"
    SAMPLE_MATRICES = "sample_matrices"


class GEODataType(Enum):
    """Enumeration of data types for GEO datasets."""

    SINGLE_CELL = "single_cell"
    BULK = "bulk"
    MIXED = "mixed"


@dataclass
class DownloadStrategy:
    """Configuration for GEO download preferences and fallback options."""

    prefer_geoparse: bool = True
    allow_fallback: bool = True
    max_retries: int = 3
    timeout_seconds: int = 300
    prefer_supplementary: bool = False
    force_tar_extraction: bool = False


@dataclass
class GEOResult:
    """Result wrapper containing data, metadata, and processing information."""

    data: Optional[pd.DataFrame] = None
    metadata: Dict[str, Any] = None
    source: GEODataSource = GEODataSource.GEOPARSE
    processing_info: Dict[str, Any] = None
    success: bool = False
    error_message: Optional[str] = None


class GEOServiceError(Exception):
    """Custom exception for GEO service errors."""

    pass


class GEOFallbackError(Exception):
    """Custom exception for fallback mechanism failures."""

    pass


class PlatformCompatibility(Enum):
    """Platform support status for early validation."""

    SUPPORTED = "supported"  # RNA-seq, fully supported
    UNSUPPORTED = "unsupported"  # Microarrays, clear rejection
    EXPERIMENTAL = "experimental"  # Partial support, warning only
    UNKNOWN = "unknown"  # Not in registry, conservative approach


# Comprehensive Platform Registry for Early Validation
# This registry enables rejecting unsupported platforms BEFORE downloading files
PLATFORM_REGISTRY: Dict[str, PlatformCompatibility] = {
    # === SUPPORTED RNA-SEQ PLATFORMS ===
    # Illumina RNA-seq platforms
    "GPL16791": PlatformCompatibility.SUPPORTED,  # Illumina HiSeq 2500
    "GPL18573": PlatformCompatibility.SUPPORTED,  # Illumina NextSeq 500
    "GPL20301": PlatformCompatibility.SUPPORTED,  # Illumina HiSeq 4000
    "GPL20795": PlatformCompatibility.SUPPORTED,  # Illumina HiSeq X Ten (human)
    "GPL21290": PlatformCompatibility.SUPPORTED,  # Illumina HiSeq 3000
    "GPL24676": PlatformCompatibility.SUPPORTED,  # Illumina NovaSeq 6000
    "GPL13112": PlatformCompatibility.SUPPORTED,  # Illumina HiSeq 2000 (mouse)
    "GPL11154": PlatformCompatibility.SUPPORTED,  # Illumina HiSeq 2000 (human)
    "GPL10999": PlatformCompatibility.SUPPORTED,  # Illumina Genome Analyzer IIx
    "GPL9115": PlatformCompatibility.SUPPORTED,  # Illumina Genome Analyzer II
    "GPL9052": PlatformCompatibility.SUPPORTED,  # Illumina Genome Analyzer
    # Single-cell platforms
    "GPL24247": PlatformCompatibility.SUPPORTED,  # 10X Chromium (NovaSeq)
    "GPL26966": PlatformCompatibility.SUPPORTED,  # 10X Chromium (HiSeq X)
    "GPL21103": PlatformCompatibility.SUPPORTED,  # Illumina HiSeq 2500 (single-cell)
    "GPL19057": PlatformCompatibility.SUPPORTED,  # Illumina NextSeq 500 (single-cell)
    # === UNSUPPORTED MICROARRAY PLATFORMS ===
    # Illumina DNA methylation arrays (see METHYLATION_PLATFORMS)
    "GPL8490": PlatformCompatibility.UNSUPPORTED,  # HumanMethylation27 BeadChip
    "GPL13534": PlatformCompatibility.UNSUPPORTED,  # HumanMethylation450 BeadChip
    "GPL21145": PlatformCompatibility.UNSUPPORTED,  # Infinium MethylationEPIC
    # Affymetrix arrays
    "GPL570": PlatformCompatibility.UNSUPPORTED,  # Affymetrix U133 Plus 2.0
    "GPL96": PlatformCompatibility.UNSUPPORTED,  # Affymetrix U133A
    "GPL97": PlatformCompatibility.UNSUPPORTED,  # Affymetrix U133B
    "GPL571": PlatformCompatibility.UNSUPPORTED,  # Affymetrix U133 A 2.0
    "GPL1352": PlatformCompatibility.UNSUPPORTED,  # Affymetrix U133 A2
    "GPL6244": PlatformCompatibility.UNSUPPORTED,  # Affymetrix Gene 1.0 ST
    "GPL6246": PlatformCompatibility.UNSUPPORTED,  # Affymetrix Mouse Gene 1.0 ST
    "GPL6247": PlatformCompatibility.UNSUPPORTED,  # Affymetrix Rat Gene 1.0 ST
    "GPL91": PlatformCompatibility.UNSUPPORTED,  # Affymetrix Mu11KsubA
    "GPL92": PlatformCompatibility.UNSUPPORTED,  # Affymetrix Mu11KsubB
    "GPL339": PlatformCompatibility.UNSUPPORTED,  # Affymetrix MOE430A
    "GPL340": PlatformCompatibility.UNSUPPORTED,  # Affymetrix MOE430B
    "GPL8321": PlatformCompatibility.UNSUPPORTED,  # Affymetrix Mouse Genome 430A 2.0
    "GPL1261": PlatformCompatibility.UNSUPPORTED,  # Affymetrix Mouse Genome 430 2.0
    # Illumina BeadArray (microarray, NOT RNA-seq)
    "GPL10558": PlatformCompatibility.UNSUPPORTED,  # Illumina HumanHT-12 V4.0
    "GPL6947": PlatformCompatibility.UNSUPPORTED,  # Illumina HumanHT-12 V3.0
    "GPL6883": PlatformCompatibility.UNSUPPORTED,  # Illumina HumanRef-8 V3.0
    "GPL6887": PlatformCompatibility.UNSUPPORTED,  # Illumina MouseRef-8 V2.0
    "GPL6885": PlatformCompatibility.UNSUPPORTED,  # Illumina MouseRef-8 V1.1
    "GPL6102": PlatformCompatibility.UNSUPPORTED,  # Illumina human-6 V2.0
    "GPL6104": PlatformCompatibility.UNSUPPORTED,  # Illumina mouse Ref-8 V2.0
    # Agilent microarrays
    "GPL6480": PlatformCompatibility.UNSUPPORTED,  # Agilent-014850
    "GPL13497": PlatformCompatibility.UNSUPPORTED,  # Agilent-026652
    "GPL17077": PlatformCompatibility.UNSUPPORTED,  # Agilent-039494
    "GPL4133": PlatformCompatibility.UNSUPPORTED,  # Agilent-014850 Whole Human Genome
    "GPL1708": PlatformCompatibility.UNSUPPORTED,  # Agilent-012391 Whole Human Genome
    "GPL7202": PlatformCompatibility.UNSUPPORTED,  # Agilent-014868 Whole Mouse Genome
    # Other microarray platforms
    "GPL341": PlatformCompatibility.UNSUPPORTED,  # Affymetrix RG_U34A
    "GPL85": PlatformCompatibility.UNSUPPORTED,  # Affymetrix RG_U34B
    "GPL1355": PlatformCompatibility.UNSUPPORTED,  # Affymetrix Rat Genome 230 2.0
}

# Keyword patterns for unknown platform detection
UNSUPPORTED_KEYWORDS = [
    "affymetrix",
    "agilent",
    "beadarray",
    "beadchip",
    "genechip",
    "microarray",
    "array",
    "snp chip",
    "exon array",
    "gene chip",
    "nimblegen",
]

# Matched against the lowercased title as words, not substrings: "oligo" must
# hit "VUMC MACF human 19K oligo v33" but not "oligodendrocyte" or
# "oligo-dT primed RNA-seq".
UNSUPPORTED_TITLE_PATTERNS = [re.compile(r"\boligo\b(?!-)")]

SUPPORTED_KEYWORDS = [
    "rna-seq",
    "rnaseq",
    "rna seq",
    "illumina hiseq",
    "illumina nextseq",
    "illumina novaseq",
    "10x",
    "chromium",
    "single cell",
    "single-cell",
    "sequencing",
]

# SUPPORTED_KEYWORDS matched as substrings of the lowercased title, so "rna-seq"
# still hits "scRNA-seq", "snRNA-seq" and "mRNA-seq". Only "sequencing" must
# start a word, so it does not hit "Resequencing Array" (an Affymetrix array).
_WORD_START_KEYWORDS = {"sequencing"}
SUPPORTED_TITLE_PATTERNS = [
    re.compile((r"(?<![a-z0-9])" if kw in _WORD_START_KEYWORDS else "") + re.escape(kw))
    for kw in SUPPORTED_KEYWORDS
]

# GEO "!Platform_technology" values of array platforms. They classify platforms
# whose title matches no keyword (e.g. GPL3255, GPL21145 "Infinium MethylationEPIC").
ARRAY_TECHNOLOGIES = {
    "in situ oligonucleotide",
    "spotted oligonucleotide",
    "spotted dna/cdna",
    "oligonucleotide beads",
}

# GEO "!Platform_technology" values of sequencing platforms. Entrez esummary
# writes "high throughput sequencing" without the hyphen.
SEQUENCING_TECHNOLOGIES = {
    "high-throughput sequencing",
    "high throughput sequencing",
}

# Array platforms that measure DNA methylation rather than gene expression,
# by title keyword or by GPL id (Entrez metadata has no platform titles).
METHYLATION_KEYWORDS = ["methylation"]
METHYLATION_PLATFORMS = {"GPL8490", "GPL13534", "GPL21145"}
