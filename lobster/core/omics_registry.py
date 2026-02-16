"""
Unified omics type registry and data type detection.

This module is the **metadata** registry for omics types — it describes what types
exist, how to detect them, their preferred databases, and QC thresholds. It is
complementary to ComponentRegistry, which handles **component instance** discovery
(adapters, providers, download services, queue preparers).

Built-in types (transcriptomics, proteomics, genomics, metabolomics, metagenomics)
are registered at module level. External types are discovered via the
``lobster.omics_types`` entry point group.

Future Plugin Targets (v2):
    - ExportSchemaRegistry (core/schemas/export_schemas.py) — hardcoded dict
    - ProtocolExtractionRegistry (services/metadata/protocol_extraction/registry.py)
    - DownloadOrchestrator thread safety for concurrent cloud use

Usage:
    from lobster.core.omics_registry import OMICS_TYPE_REGISTRY, DataTypeDetector

    # Check what types are registered
    for name, config in OMICS_TYPE_REGISTRY.items():
        print(f"{config.display_name}: {config.preferred_databases}")

    # Detect omics type from metadata
    detector = DataTypeDetector()
    results = detector.detect_from_metadata({"title": "TMT proteomics"})
    # => [("proteomics", 0.85), ("transcriptomics", 0.1), ...]
"""

import logging
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class OmicsDetectionConfig:
    """Configuration for detecting an omics type from metadata or data.

    Each field contributes to a weighted score when matching against
    dataset metadata or loaded AnnData objects.
    """

    # Text-based detection (searched in title, summary, description, etc.)
    keywords: List[str] = field(default_factory=list)

    # Platform pattern matching (GPL IDs, instrument names)
    platform_patterns: List[str] = field(default_factory=list)

    # Feature count range — (min_features, max_features) for data-based detection
    feature_count_range: Tuple[int, int] = (0, 999_999)

    # Characteristic patterns (matched against samples characteristics_ch1)
    characteristic_patterns: List[str] = field(default_factory=list)

    # Column signatures — var/obs column names that indicate this type
    column_signatures: Dict[str, List[str]] = field(default_factory=dict)

    # Library strategy patterns (matched against library_strategy field)
    library_strategy_patterns: List[str] = field(default_factory=list)

    # Detection weight — higher = checked first, used as tiebreaker
    weight: int = 10


@dataclass
class OmicsTypeConfig:
    """Complete configuration for an omics type.

    Maps an omics type to everything the system needs: schema, adapters,
    detection heuristics, preferred databases, and QC thresholds.
    """

    # Identity
    name: str  # "proteomics", "metabolomics", etc.
    display_name: str  # "Proteomics"

    # Schema routing
    schema_class: Optional[str] = None  # dotted path "lobster.core.schemas.proteomics:ProteomicsSchema"

    # Adapter names that handle this type
    adapter_names: List[str] = field(default_factory=list)

    # Preferred databases for search routing (ordered by priority)
    preferred_databases: List[str] = field(default_factory=list)

    # Detection configuration
    detection: OmicsDetectionConfig = field(default_factory=OmicsDetectionConfig)

    # Modality-specific QC defaults
    qc_thresholds: Dict[str, Any] = field(default_factory=dict)

    # Data type aliases returned by detection (e.g., "single_cell_rna_seq", "bulk_rna_seq")
    data_type_aliases: List[str] = field(default_factory=list)


# =============================================================================
# OMICS TYPE REGISTRY — single source of truth for type metadata
# =============================================================================

OMICS_TYPE_REGISTRY: Dict[str, OmicsTypeConfig] = {}


def register_omics_type(config: OmicsTypeConfig) -> None:
    """Register an omics type configuration.

    First registration wins. Duplicate names log a warning and are skipped.
    """
    if config.name in OMICS_TYPE_REGISTRY:
        logger.warning(
            f"Omics type '{config.name}' already registered, skipping duplicate"
        )
        return
    OMICS_TYPE_REGISTRY[config.name] = config
    logger.debug(f"Registered omics type: {config.name}")


def _register_builtin_types() -> None:
    """Register the 5 built-in omics types using detection patterns
    extracted from existing code."""

    # --- Transcriptomics ---
    register_omics_type(OmicsTypeConfig(
        name="transcriptomics",
        display_name="Transcriptomics",
        schema_class="lobster.core.schemas.transcriptomics_schema:TranscriptomicsSchema",
        adapter_names=["transcriptomics_single_cell", "transcriptomics_bulk"],
        preferred_databases=["geo", "sra"],
        data_type_aliases=["single_cell_rna_seq", "bulk_rna_seq", "rna_seq"],
        detection=OmicsDetectionConfig(
            keywords=[
                "single-cell", "single cell", "scRNA-seq", "scrna-seq", "scrnaseq",
                "10x", "10X", "droplet", "Drop-seq", "Smart-seq", "CEL-seq",
                "inDrop", "single nuclei", "snRNA-seq", "scATAC-seq", "Chromium",
                "bulk", "tissue", "total rna", "population",
                "RNA-seq", "rna-seq", "transcriptome", "transcriptomics",
                "gene expression", "microarray",
            ],
            platform_patterns=["GPL24676", "GPL24247", "10X", "Chromium", "Illumina"],
            feature_count_range=(5_000, 60_000),
            library_strategy_patterns=["single", "10x", "RNA-Seq", "rna-seq"],
            column_signatures={
                "obs": ["n_genes", "n_genes_by_counts", "total_counts", "pct_counts_mt"],
                "var": ["n_cells", "n_cells_by_counts", "highly_variable"],
            },
            weight=8,  # Lower than proteomics — GEO default is transcriptomics
        ),
        qc_thresholds={
            "min_genes": 200,
            "min_cells": 3,
            "max_pct_mt": 20.0,
        },
    ))

    # --- Proteomics ---
    register_omics_type(OmicsTypeConfig(
        name="proteomics",
        display_name="Proteomics",
        schema_class="lobster.core.schemas.proteomics_schema:ProteomicsSchema",
        adapter_names=["proteomics_ms", "proteomics_affinity"],
        preferred_databases=["pride", "massive", "geo"],
        data_type_aliases=["mass_spectrometry_proteomics", "ms_proteomics", "affinity_proteomics"],
        detection=OmicsDetectionConfig(
            keywords=[
                "proteomics", "proteome", "mass spectrometry", "mass spec",
                "ms/ms", "lc-ms", "lc/ms", "orbitrap", "q-tof", "qtof",
                "maldi", "triple tof", "tripletof", "q exactive", "qexactive",
                "tmt", "itraq", "silac", "label-free quantification",
                "label free quantification", "lfq", "dia", "dda",
                "data-independent acquisition", "data-dependent acquisition",
                "swath", "olink", "somascan", "soma logic",
                "proximity extension assay", "protein expression",
                "protein abundance", "peptide", "tandem mass",
            ],
            platform_patterns=[
                "Orbitrap", "Q-TOF", "MALDI", "TripleTOF", "QExactive",
                "Lumos", "Exploris", "timsTOF", "Olink", "SomaScan",
            ],
            feature_count_range=(100, 12_000),
            characteristic_patterns=[
                "assay: protein", "technology: ms", "instrument:",
                "mass spectrometer", "fractionation:", "enzyme: trypsin",
            ],
            column_signatures={
                "var": ["protein_id", "gene_name", "peptide_count", "coverage"],
            },
            weight=12,  # Higher priority — proteomics keywords are more specific
        ),
        qc_thresholds={
            "max_missing_pct": 50.0,
            "min_proteins": 100,
            "cv_threshold": 30.0,
        },
    ))

    # --- Genomics ---
    register_omics_type(OmicsTypeConfig(
        name="genomics",
        display_name="Genomics",
        schema_class="lobster.core.schemas.genomics_schema:GenomicsSchema",
        adapter_names=["genomics_vcf", "genomics_plink"],
        preferred_databases=["geo", "sra", "dbgap"],
        data_type_aliases=["vcf", "gwas", "wgs", "wes"],
        detection=OmicsDetectionConfig(
            keywords=[
                "whole genome", "whole exome", "WGS", "WES", "exome sequencing",
                "variant calling", "VCF", "SNP", "SNV", "indel",
                "GWAS", "genome-wide association", "genotyping",
                "PLINK", "copy number variation", "CNV",
                "structural variant", "somatic mutation", "germline",
            ],
            platform_patterns=["Illumina", "NovaSeq", "HiSeq"],
            feature_count_range=(10_000, 10_000_000),
            library_strategy_patterns=["WGS", "WXS", "WES"],
            column_signatures={
                "var": ["chrom", "pos", "ref", "alt", "qual", "rsid"],
            },
            weight=9,
        ),
        qc_thresholds={
            "min_call_rate": 0.95,
            "min_depth": 10,
            "hwe_p_threshold": 1e-6,
        },
    ))

    # --- Metabolomics ---
    register_omics_type(OmicsTypeConfig(
        name="metabolomics",
        display_name="Metabolomics",
        schema_class="lobster.core.schemas.metabolomics:MetabolomicsSchema",
        adapter_names=["metabolomics_lc_ms", "metabolomics_gc_ms", "metabolomics_nmr"],
        preferred_databases=["metabolights", "metabolomics_workbench", "geo"],
        data_type_aliases=["lcms", "gcms", "nmr", "lipidomics"],
        detection=OmicsDetectionConfig(
            keywords=[
                "metabolomics", "metabolome", "metabolite", "metabolic profiling",
                "lipidomics", "lipidome", "lipid profiling",
                "untargeted metabolomics", "targeted metabolomics",
                "LC-MS", "GC-MS", "NMR", "nuclear magnetic resonance",
                "mass spectrometry metabolomics", "HILIC", "RPLC",
                "mzML", "mzXML", "MetaboLights", "MTBLS",
                "metabolomics workbench", "small molecule",
                "flux analysis", "isotope tracing", "stable isotope",
            ],
            platform_patterns=["QTRAP", "Agilent", "Waters", "Bruker"],
            feature_count_range=(50, 5_000),
            characteristic_patterns=[
                "metabolite:", "compound:", "ionization mode:",
                "chromatography:", "derivatization:",
            ],
            column_signatures={
                "var": ["mz", "retention_time", "rt", "m/z", "adduct", "formula"],
            },
            weight=11,  # Higher than transcriptomics to avoid false routing
        ),
        qc_thresholds={
            "max_missing_pct": 30.0,
            "min_metabolites": 30,
            "cv_threshold": 30.0,
            "rsd_qc_threshold": 20.0,
        },
    ))

    # --- Metagenomics ---
    register_omics_type(OmicsTypeConfig(
        name="metagenomics",
        display_name="Metagenomics",
        schema_class="lobster.core.schemas.metagenomics:MetagenomicsSchema",
        adapter_names=["metagenomics_amplicon", "metagenomics_shotgun"],
        preferred_databases=["sra", "geo", "mg-rast"],
        data_type_aliases=["16s", "its", "amplicon", "microbiome", "sra_amplicon"],
        detection=OmicsDetectionConfig(
            keywords=[
                "metagenomics", "metagenome", "metagenomic",
                "16S", "16s rRNA", "ITS", "amplicon", "microbiome",
                "microbiota", "gut microbiome", "oral microbiome",
                "shotgun metagenomics", "taxonomic profiling",
                "OTU", "ASV", "QIIME", "DADA2", "mothur",
                "microbial community", "microbial diversity",
            ],
            platform_patterns=["MiSeq", "Ion Torrent"],
            feature_count_range=(100, 50_000),
            library_strategy_patterns=["AMPLICON", "WGS"],
            column_signatures={
                "var": ["taxonomy", "kingdom", "phylum", "class", "order", "family", "genus", "species"],
                "obs": ["sample_type", "collection_site", "host"],
            },
            weight=10,
        ),
        qc_thresholds={
            "min_reads": 1000,
            "min_taxa": 10,
            "rarefaction_depth": 10000,
        },
    ))


# =============================================================================
# DATA TYPE DETECTOR — unified replacement for scattered detection functions
# =============================================================================

class DataTypeDetector:
    """Unified omics data type detection using the OmicsTypeRegistry.

    Replaces the 4 scattered detection functions:
    - geo_queue_preparer._is_single_cell_dataset()
    - geo_queue_preparer._is_proteomics_dataset()
    - geo_service._determine_data_type_from_metadata()
    - base.detect_data_type()
    """

    # Text fields to search in metadata dicts
    TEXT_FIELDS = ("title", "summary", "overall_design", "type", "description")

    def detect_from_metadata(
        self, metadata: dict
    ) -> List[Tuple[str, float]]:
        """Detect omics type from GEO/repository metadata.

        Scores each registered omics type against metadata text fields,
        platforms, characteristics, and library strategy.

        Args:
            metadata: Dataset metadata dict with keys like title, summary,
                platforms, characteristics_ch1, library_strategy, etc.

        Returns:
            Ranked list of (omics_type_name, confidence) tuples, sorted by
            confidence descending. Confidence is 0.0-1.0.
        """
        scores: Dict[str, float] = {}

        # Build combined text from standard metadata fields
        combined_text = " ".join(
            str(metadata.get(f, "")) for f in self.TEXT_FIELDS
        ).lower()

        # Add platform text (handle both "platforms" and "platform" keys)
        platforms_text = str(metadata.get("platforms", "")).lower()
        platform_text = str(metadata.get("platform", "")).lower()
        combined_text += " " + platforms_text + " " + platform_text

        # Add characteristics text
        chars_text = self._extract_characteristics_text(metadata)
        combined_text += " " + chars_text

        # Add library strategy
        lib_strategy = str(metadata.get("library_strategy", "")).lower()

        for name, config in OMICS_TYPE_REGISTRY.items():
            det = config.detection
            score = 0.0

            # Keyword matching (primary signal)
            keyword_hits = sum(
                1 for kw in det.keywords if kw.lower() in combined_text
            )
            if det.keywords:
                score += (keyword_hits / len(det.keywords)) * 0.6

            # Platform matching
            platform_hits = sum(
                1 for p in det.platform_patterns
                if p.lower() in platforms_text
            )
            if det.platform_patterns:
                score += (platform_hits / len(det.platform_patterns)) * 0.15

            # Characteristics matching
            char_hits = sum(
                1 for cp in det.characteristic_patterns
                if cp.lower() in chars_text
            )
            if det.characteristic_patterns:
                score += (char_hits / len(det.characteristic_patterns)) * 0.15

            # Library strategy matching
            strategy_hits = sum(
                1 for ls in det.library_strategy_patterns
                if ls.lower() in lib_strategy
            )
            if det.library_strategy_patterns:
                score += (strategy_hits / len(det.library_strategy_patterns)) * 0.1

            # Apply weight as tiebreaker (normalized to small range)
            score += det.weight * 0.001

            scores[name] = score

        # Sort by score descending
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked

    def detect_from_data(
        self, adata: Any
    ) -> List[Tuple[str, float]]:
        """Detect omics type from loaded AnnData characteristics.

        Uses feature count ranges and column signatures to score each type.

        Args:
            adata: AnnData object (or any object with n_vars, obs, var attributes).

        Returns:
            Ranked list of (omics_type_name, confidence) tuples.
        """
        scores: Dict[str, float] = {}
        n_vars = getattr(adata, "n_vars", 0)
        obs_cols = set(getattr(adata, "obs", {}).keys()) if hasattr(adata, "obs") else set()
        var_cols = set(getattr(adata, "var", {}).keys()) if hasattr(adata, "var") else set()

        for name, config in OMICS_TYPE_REGISTRY.items():
            det = config.detection
            score = 0.0

            # Feature count range matching (primary signal for data)
            min_f, max_f = det.feature_count_range
            if min_f <= n_vars <= max_f:
                # Score based on how centered the count is in the range
                range_size = max_f - min_f
                if range_size > 0:
                    center = (min_f + max_f) / 2
                    distance = abs(n_vars - center) / (range_size / 2)
                    score += (1.0 - distance * 0.5) * 0.5
                else:
                    score += 0.5
            else:
                score -= 0.2  # Penalty for out-of-range

            # Column signature matching
            sig = det.column_signatures
            if "obs" in sig:
                obs_hits = sum(1 for c in sig["obs"] if c in obs_cols)
                if sig["obs"]:
                    score += (obs_hits / len(sig["obs"])) * 0.25
            if "var" in sig:
                var_hits = sum(1 for c in sig["var"] if c in var_cols)
                if sig["var"]:
                    score += (var_hits / len(sig["var"])) * 0.25

            # Weight tiebreaker
            score += det.weight * 0.001

            scores[name] = max(score, 0.0)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked

    def detect(
        self,
        metadata: Optional[dict] = None,
        adata: Any = None,
    ) -> List[Tuple[str, float]]:
        """Combined detection using all available signals.

        When both metadata and data are provided, scores are merged with
        metadata weighted at 60% and data at 40%.

        Args:
            metadata: Dataset metadata dict (optional).
            adata: Loaded AnnData object (optional).

        Returns:
            Ranked list of (omics_type_name, confidence) tuples.
        """
        if metadata is None and adata is None:
            return []

        if metadata is not None and adata is not None:
            meta_scores = dict(self.detect_from_metadata(metadata))
            data_scores = dict(self.detect_from_data(adata))
            all_types = set(meta_scores) | set(data_scores)
            combined = {
                t: meta_scores.get(t, 0.0) * 0.6 + data_scores.get(t, 0.0) * 0.4
                for t in all_types
            }
            return sorted(combined.items(), key=lambda x: x[1], reverse=True)

        if metadata is not None:
            return self.detect_from_metadata(metadata)

        return self.detect_from_data(adata)

    # -- Backward-compatible helpers --

    def is_single_cell(self, metadata: dict) -> bool:
        """Backward-compatible: replaces _is_single_cell_dataset().

        Returns True if transcriptomics is the top match and single-cell
        keywords are present in the metadata.
        """
        results = self.detect_from_metadata(metadata)
        if not results:
            return False

        # Must be transcriptomics as top match
        top_type, top_score = results[0]
        if top_type != "transcriptomics" or top_score < 0.01:
            return False

        # Additionally check for single-cell specific keywords
        # Include text fields + platform fields for full coverage
        combined = " ".join(
            str(metadata.get(f, "")) for f in self.TEXT_FIELDS
        ).lower()
        combined += " " + str(metadata.get("platform", "")).lower()
        combined += " " + str(metadata.get("platforms", "")).lower()
        combined += " " + str(metadata.get("library_strategy", "")).lower()
        sc_keywords = [
            "single-cell", "single cell", "scRNA-seq", "scrna-seq", "scrnaseq",
            "10x", "droplet", "Drop-seq", "Smart-seq", "CEL-seq",
            "inDrop", "single nuclei", "snRNA-seq", "scATAC-seq", "Chromium",
            "GPL24676", "GPL24247",
        ]
        return any(kw.lower() in combined for kw in sc_keywords)

    def is_proteomics(self, metadata: dict) -> bool:
        """Backward-compatible: replaces _is_proteomics_dataset()."""
        results = self.detect_from_metadata(metadata)
        if not results:
            return False
        top_type, top_score = results[0]
        return top_type == "proteomics" and top_score > 0.01

    def determine_data_type(self, metadata: dict) -> str:
        """Backward-compatible: replaces _determine_data_type_from_metadata().

        Returns one of: "proteomics", "single_cell_rna_seq", "bulk_rna_seq".
        """
        results = self.detect_from_metadata(metadata)
        if not results:
            return "single_cell_rna_seq"  # GEO default

        top_type, _ = results[0]

        if top_type == "proteomics":
            return "proteomics"

        if top_type == "transcriptomics":
            # Distinguish single-cell vs bulk
            if self.is_single_cell(metadata):
                return "single_cell_rna_seq"
            return "bulk_rna_seq"

        # For other types, return the name directly
        return top_type

    @staticmethod
    def _extract_characteristics_text(metadata: dict) -> str:
        """Extract characteristics text from metadata samples."""
        chars_parts = []
        samples = metadata.get("samples", [])
        if isinstance(samples, list):
            for sample in samples[:10]:  # Cap at 10 samples to avoid huge strings
                if isinstance(sample, dict):
                    ch = sample.get("characteristics_ch1", [])
                    if isinstance(ch, list):
                        chars_parts.extend(str(c) for c in ch)
                    elif isinstance(ch, str):
                        chars_parts.append(ch)
        return " ".join(chars_parts).lower()


# =============================================================================
# ENTRY POINT DISCOVERY for external omics types
# =============================================================================

def _discover_external_omics_types() -> None:
    """Discover and register omics types from lobster.omics_types entry points."""
    if sys.version_info >= (3, 10):
        from importlib.metadata import entry_points
        discovered = entry_points(group="lobster.omics_types")
    else:
        from importlib.metadata import entry_points
        eps = entry_points()
        discovered = eps.get("lobster.omics_types", [])

    for entry in discovered:
        try:
            config = entry.load()
            if isinstance(config, OmicsTypeConfig):
                register_omics_type(config)
            else:
                logger.warning(
                    f"Entry point '{entry.name}' did not resolve to OmicsTypeConfig, "
                    f"got {type(config).__name__}"
                )
        except Exception as e:
            logger.warning(f"Failed to load omics type '{entry.name}': {e}")


# =============================================================================
# MODULE INITIALIZATION
# =============================================================================

_register_builtin_types()
_discover_external_omics_types()
