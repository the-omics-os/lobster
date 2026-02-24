"""
Variant annotation service for genomics/DNA data.

This service annotates variants with gene names, functional consequences,
and pathogenicity scores using genebe and Ensembl VEP APIs.

All methods return 3-tuples (AnnData, Dict, AnalysisStep) for provenance
tracking and reproducible notebook export via /pipeline export.
"""

import time
from typing import Any, Dict, Tuple

import anndata
import numpy as np
import pandas as pd

from lobster.core.analysis_ir import AnalysisStep, ParameterSpec
from lobster.utils.logger import get_logger

logger = get_logger(__name__)

# Import genebe for variant annotation (https://pypi.org/project/genebe/)
try:
    import genebe as gnb

    HAS_GENEBE = True
except ImportError:
    HAS_GENEBE = False
    gnb = None

# Import requests for Ensembl VEP fallback
try:
    import requests

    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    logger.error("requests not installed. Install with: pip install requests")


class VariantAnnotationError(Exception):
    """Raised when variant annotation fails."""

    pass


class VariantAnnotationService:
    """
    Centralized variant annotation service using genebe and Ensembl VEP.

    This service annotates genomic variants with:
    - Gene names and IDs
    - Functional consequences (e.g., missense, synonymous)
    - Gene biotypes (e.g., protein_coding, lincRNA)
    - Population allele frequencies (gnomAD)
    - Clinical significance (ClinVar)
    - Pathogenicity scores (CADD, SIFT, PolyPhen)

    Primary method: genebe (batch processing, comprehensive annotations)
    Fallback method: Ensembl VEP REST API (rate limited)

    All methods follow Lobster's (adata, stats, ir) tuple pattern.
    """

    # Supported annotation sources
    ANNOTATION_SOURCES = ["genebe", "ensembl_vep"]

    # Supported genome builds
    GENOME_BUILDS = ["hg38", "hg19", "GRCh38", "GRCh37"]

    # Annotation column definitions
    ANNOTATION_COLUMNS = {
        "gene_symbol": "Gene name (e.g., BRCA1)",
        "gene_id": "Ensembl gene ID (e.g., ENSG00000012048)",
        "consequence": "Variant consequence (e.g., missense_variant)",
        "biotype": "Gene biotype (e.g., protein_coding)",
        "gnomad_af": "gnomAD allele frequency (population frequency)",
        "clinvar_significance": "ClinVar clinical significance (e.g., Pathogenic)",
        "cadd_score": "CADD pathogenicity score",
        "sift_score": "SIFT deleteriousness score",
        "polyphen_score": "PolyPhen2 pathogenicity score",
    }

    def __init__(self):
        """Initialize variant annotation service."""
        logger.debug("Initializing VariantAnnotationService")

        # Cache for repeated queries (chr:pos:ref:alt -> annotations)
        self._annotation_cache = {}

        logger.debug("VariantAnnotationService initialized successfully")

    def annotate_variants(
        self,
        adata: anndata.AnnData,
        annotation_source: str = "genebe",
        genome_build: str = "hg38",
        batch_size: int = 5000,
        retry_attempts: int = 3,
        retry_delay: float = 2.0,
        use_cache: bool = True,
    ) -> Tuple[anndata.AnnData, Dict[str, Any], AnalysisStep]:
        """
        Annotate variants with gene names and functional consequences.

        This method enriches variant data with gene annotations, functional
        consequences, and pathogenicity scores. Annotations are added to
        adata.var as new columns.

        Args:
            adata: AnnData object with variant data in .var
            annotation_source: Annotation source ("genebe" or "ensembl_vep")
            genome_build: Reference genome build ("hg38", "hg19", "GRCh38", "GRCh37")
            batch_size: Number of variants per API call (max 10000 for genebe)
            retry_attempts: Number of retry attempts on API failure
            retry_delay: Delay (seconds) between retries (exponential backoff)
            use_cache: Whether to use cached annotations for repeated queries

        Returns:
            Tuple[anndata.AnnData, Dict[str, Any], AnalysisStep]:
                - AnnData with annotations added to .var
                - Annotation statistics dictionary
                - AnalysisStep IR for notebook export

        Raises:
            VariantAnnotationError: If annotation fails or inputs are invalid
        """
        try:
            logger.info(
                f"Starting variant annotation: {adata.n_vars} variants, "
                f"source={annotation_source}, build={genome_build}"
            )

            # Validate inputs
            self._validate_inputs(adata, annotation_source, genome_build, batch_size)

            # Normalize genome build
            genome_build = self._normalize_genome_build(genome_build)

            # Create working copy
            adata_annotated = adata.copy()

            # Prepare variant DataFrame from adata.var
            var_df = self._prepare_variant_dataframe(adata_annotated)

            logger.info(f"Prepared {len(var_df)} variants for annotation")

            # Route to appropriate annotation method
            if annotation_source == "genebe":
                annotations_df = self._annotate_with_genebe(
                    var_df,
                    genome_build,
                    batch_size,
                    retry_attempts,
                    retry_delay,
                    use_cache,
                )
            elif annotation_source == "ensembl_vep":
                annotations_df = self._annotate_with_ensembl(
                    var_df,
                    genome_build,
                    retry_attempts,
                    retry_delay,
                    use_cache,
                )
            else:
                raise VariantAnnotationError(
                    f"Unsupported annotation source: {annotation_source}. "
                    f"Supported: {self.ANNOTATION_SOURCES}"
                )

            # Merge annotations into adata.var
            self._merge_annotations(adata_annotated, annotations_df)

            # Compile statistics
            stats = self._compile_statistics(
                adata_annotated, annotation_source, genome_build
            )

            logger.info(
                f"Annotation completed: {stats['n_variants_annotated']}/{stats['n_variants']} "
                f"variants annotated ({stats['annotation_rate_pct']:.1f}%)"
            )

            # Create IR for notebook export
            ir = self._create_annotation_ir(
                annotation_source=annotation_source,
                genome_build=genome_build,
                batch_size=batch_size,
            )

            return adata_annotated, stats, ir

        except Exception as e:
            logger.exception(f"Error in variant annotation: {e}")
            raise VariantAnnotationError(f"Variant annotation failed: {str(e)}")

    # ===== New Annotation Methods (Plan 01-01) =====

    def normalize_variants(
        self,
        adata: anndata.AnnData,
        ref_col: str = "REF",
        alt_col: str = "ALT",
        pos_col: str = "POS",
        chrom_col: str = "CHROM",
    ) -> Tuple[anndata.AnnData, Dict[str, Any], AnalysisStep]:
        """
        Normalize variant representations: left-trim padding and split multiallelic.

        Performs simplified normalization without requiring a reference genome:
        1. Left-trim common prefix between REF and ALT (keep at least 1 base)
        2. Right-trim common suffix
        3. Split multiallelic variants (comma-separated ALT) into biallelic rows

        Args:
            adata: AnnData object with variant data in .var
            ref_col: Column name for reference allele (default: "REF")
            alt_col: Column name for alternate allele (default: "ALT")
            pos_col: Column name for position (default: "POS")
            chrom_col: Column name for chromosome (default: "CHROM")

        Returns:
            Tuple[AnnData, Dict, AnalysisStep]: Normalized AnnData, stats, IR

        Raises:
            VariantAnnotationError: If required columns are missing
        """
        try:
            logger.info(f"Normalizing variants: {adata.n_vars} variants")

            # Validate required columns
            for col in [ref_col, alt_col, pos_col, chrom_col]:
                if col not in adata.var.columns:
                    raise VariantAnnotationError(
                        f"Required column '{col}' not found in adata.var. "
                        f"Available: {list(adata.var.columns)}"
                    )

            # Create working copy
            adata_norm = adata.copy()

            n_trimmed = 0
            n_multiallelic = 0
            rows_to_add = []
            rows_to_remove = []

            for var_idx in range(adata_norm.n_vars):
                ref = str(adata_norm.var[ref_col].iloc[var_idx])
                alt = str(adata_norm.var[alt_col].iloc[var_idx])
                pos = int(adata_norm.var[pos_col].iloc[var_idx])

                # Handle multiallelic: split on comma
                if "," in alt:
                    n_multiallelic += 1
                    alt_alleles = alt.split(",")
                    rows_to_remove.append(var_idx)

                    for alt_allele in alt_alleles:
                        trimmed_ref, trimmed_alt, trimmed_pos = self._trim_alleles(
                            ref, alt_allele.strip(), pos
                        )
                        if trimmed_ref != ref or trimmed_alt != alt_allele.strip():
                            n_trimmed += 1

                        row_data = adata_norm.var.iloc[var_idx].copy()
                        row_data[ref_col] = trimmed_ref
                        row_data[alt_col] = trimmed_alt
                        row_data[pos_col] = trimmed_pos
                        rows_to_add.append(
                            {
                                "data": row_data,
                                "original_idx": var_idx,
                                "is_split": True,
                            }
                        )
                else:
                    # Single allele: just trim
                    trimmed_ref, trimmed_alt, trimmed_pos = self._trim_alleles(
                        ref, alt, pos
                    )
                    if trimmed_ref != ref or trimmed_alt != alt:
                        n_trimmed += 1
                        adata_norm.var.iloc[
                            var_idx, adata_norm.var.columns.get_loc(ref_col)
                        ] = trimmed_ref
                        adata_norm.var.iloc[
                            var_idx, adata_norm.var.columns.get_loc(alt_col)
                        ] = trimmed_alt
                        adata_norm.var.iloc[
                            var_idx, adata_norm.var.columns.get_loc(pos_col)
                        ] = trimmed_pos

            # Add normalized columns for tracking
            adata_norm.var["REF_normalized"] = adata_norm.var[ref_col].astype(str)
            adata_norm.var["ALT_normalized"] = adata_norm.var[alt_col].astype(str)
            adata_norm.var["POS_normalized"] = adata_norm.var[pos_col]
            adata_norm.var["is_multiallelic_split"] = False

            # Handle multiallelic splits by expanding rows
            if rows_to_add:
                import scipy.sparse as sp

                new_var_rows = []
                new_X_rows = []
                new_layer_rows = {k: [] for k in adata_norm.layers.keys()}

                # First add all non-removed rows
                keep_mask = np.ones(adata_norm.n_vars, dtype=bool)
                for idx in rows_to_remove:
                    keep_mask[idx] = False

                adata_kept = adata_norm[:, keep_mask].copy()

                # Then add split rows
                for row_info in rows_to_add:
                    var_data = row_info["data"].copy()
                    var_data["REF_normalized"] = str(var_data[ref_col])
                    var_data["ALT_normalized"] = str(var_data[alt_col])
                    var_data["POS_normalized"] = var_data[pos_col]
                    var_data["is_multiallelic_split"] = True
                    new_var_rows.append(var_data)

                    orig_idx = row_info["original_idx"]
                    new_X_rows.append(adata_norm.X[:, orig_idx : orig_idx + 1])
                    for layer_name in adata_norm.layers.keys():
                        new_layer_rows[layer_name].append(
                            adata_norm.layers[layer_name][:, orig_idx : orig_idx + 1]
                        )

                if new_var_rows:
                    import anndata as ad

                    new_var_df = pd.DataFrame(new_var_rows)
                    new_var_df.index = [
                        f"{new_var_df[chrom_col].iloc[i]}:{new_var_df[pos_col].iloc[i]}:{new_var_df[ref_col].iloc[i]}:{new_var_df[alt_col].iloc[i]}"
                        for i in range(len(new_var_df))
                    ]

                    new_X = (
                        np.hstack(new_X_rows)
                        if new_X_rows
                        else np.empty((adata_norm.n_obs, 0))
                    )
                    new_layers = {}
                    for layer_name, layer_parts in new_layer_rows.items():
                        if layer_parts:
                            new_layers[layer_name] = np.hstack(layer_parts)

                    new_adata = ad.AnnData(
                        X=new_X,
                        obs=adata_kept.obs.copy(),
                        var=new_var_df,
                        layers=new_layers,
                    )

                    # Concatenate
                    adata_norm = ad.concat(
                        [adata_kept, new_adata],
                        axis=1,
                        merge="same",
                    )

            n_before = adata.n_vars
            n_after = adata_norm.n_vars

            stats = {
                "analysis_type": "variant_normalization",
                "n_variants_before": n_before,
                "n_variants_after": n_after,
                "n_trimmed": n_trimmed,
                "n_multiallelic_split": n_multiallelic,
            }

            logger.info(
                f"Normalization completed: {n_before} -> {n_after} variants, "
                f"{n_trimmed} trimmed, {n_multiallelic} multiallelic split"
            )

            ir = self._create_normalize_ir()

            return adata_norm, stats, ir

        except Exception as e:
            logger.exception(f"Error in variant normalization: {e}")
            raise VariantAnnotationError(f"Variant normalization failed: {str(e)}")

    def _trim_alleles(self, ref: str, alt: str, pos: int) -> Tuple[str, str, int]:
        """
        Left-trim common prefix and right-trim common suffix between REF and ALT.

        Keeps at least 1 base in each allele.

        Args:
            ref: Reference allele
            alt: Alternate allele
            pos: Genomic position

        Returns:
            Tuple of (trimmed_ref, trimmed_alt, adjusted_pos)
        """
        # Right-trim common suffix (keep at least 1 base)
        while len(ref) > 1 and len(alt) > 1 and ref[-1] == alt[-1]:
            ref = ref[:-1]
            alt = alt[:-1]

        # Left-trim common prefix (keep at least 1 base)
        trim_left = 0
        while (
            len(ref) - trim_left > 1
            and len(alt) - trim_left > 1
            and ref[trim_left] == alt[trim_left]
        ):
            trim_left += 1

        if trim_left > 0:
            ref = ref[trim_left:]
            alt = alt[trim_left:]
            pos = pos + trim_left

        return ref, alt, pos

    def query_population_frequencies(
        self,
        adata: anndata.AnnData,
        population: str = None,
    ) -> Tuple[anndata.AnnData, Dict[str, Any], AnalysisStep]:
        """
        Query population allele frequencies (gnomAD) for variants.

        If gnomad_af is already in adata.var (from prior annotate_variants call),
        compiles stats from existing data. Otherwise runs annotation to get frequencies.

        Args:
            adata: AnnData object with variant data
            population: Optional population name (e.g., "AFR", "EUR"). Currently
                       only global AF is available; population-specific noted in stats.

        Returns:
            Tuple[AnnData, Dict, AnalysisStep]: AnnData with gnomad_af in var, stats, IR

        Raises:
            VariantAnnotationError: If annotation fails
        """
        try:
            logger.info(f"Querying population frequencies for {adata.n_vars} variants")

            adata_freq = adata.copy()

            # Check if gnomad_af already exists
            if "gnomad_af" not in adata_freq.var.columns:
                logger.info(
                    "gnomad_af not found, running annotation to retrieve frequencies"
                )
                # Try genebe first, fall back to ensembl_vep
                try:
                    adata_freq, _, _ = self.annotate_variants(
                        adata_freq, annotation_source="genebe"
                    )
                except Exception:
                    adata_freq, _, _ = self.annotate_variants(
                        adata_freq, annotation_source="ensembl_vep"
                    )

            # Compile frequency statistics
            af_values = pd.to_numeric(
                adata_freq.var.get("gnomad_af", pd.Series(dtype=float)), errors="coerce"
            )
            n_with_freq = int(af_values.notna().sum())
            valid_af = af_values.dropna()

            stats = {
                "analysis_type": "population_frequencies",
                "n_variants": adata_freq.n_vars,
                "n_with_frequency": n_with_freq,
                "mean_af": float(valid_af.mean()) if len(valid_af) > 0 else 0.0,
                "median_af": float(valid_af.median()) if len(valid_af) > 0 else 0.0,
                "n_rare": int((valid_af < 0.01).sum()) if len(valid_af) > 0 else 0,
                "n_common": int((valid_af > 0.05).sum()) if len(valid_af) > 0 else 0,
            }

            if population:
                stats["population_requested"] = population
                stats["population_note"] = (
                    f"Population-specific AF for '{population}' was requested "
                    "but only global AF is currently available via genebe/VEP."
                )

            logger.info(
                f"Population frequencies: {n_with_freq}/{adata_freq.n_vars} variants with AF"
            )

            ir = self._create_population_freq_ir(population=population)

            return adata_freq, stats, ir

        except Exception as e:
            logger.exception(f"Error querying population frequencies: {e}")
            raise VariantAnnotationError(f"Population frequency query failed: {str(e)}")

    def query_clinical_databases(
        self,
        adata: anndata.AnnData,
    ) -> Tuple[anndata.AnnData, Dict[str, Any], AnalysisStep]:
        """
        Query clinical databases (ClinVar) for variant significance.

        If clinvar_significance is already in adata.var (from prior annotate_variants),
        compiles stats from existing data. Otherwise runs annotation.

        Args:
            adata: AnnData object with variant data

        Returns:
            Tuple[AnnData, Dict, AnalysisStep]: AnnData with clinvar_significance, stats, IR

        Raises:
            VariantAnnotationError: If annotation fails
        """
        try:
            logger.info(f"Querying clinical databases for {adata.n_vars} variants")

            adata_clin = adata.copy()

            # Check if clinvar_significance already exists
            if "clinvar_significance" not in adata_clin.var.columns:
                logger.info("clinvar_significance not found, running annotation")
                try:
                    adata_clin, _, _ = self.annotate_variants(
                        adata_clin, annotation_source="genebe"
                    )
                except Exception:
                    adata_clin, _, _ = self.annotate_variants(
                        adata_clin, annotation_source="ensembl_vep"
                    )

            # Compile clinical statistics
            clinvar_col = adata_clin.var.get(
                "clinvar_significance", pd.Series(dtype=str)
            )
            n_with_clinvar = int(clinvar_col.notna().sum())

            # Count significance categories
            significance_counts = {}
            if n_with_clinvar > 0:
                significance_counts = clinvar_col.dropna().value_counts().to_dict()

            # Count major categories
            clinvar_lower = clinvar_col.dropna().str.lower()
            n_pathogenic = int(clinvar_lower.str.contains("pathogenic", na=False).sum())
            n_benign = int(clinvar_lower.str.contains("benign", na=False).sum())
            n_uncertain = int(clinvar_lower.str.contains("uncertain", na=False).sum())

            stats = {
                "analysis_type": "clinical_databases",
                "n_variants": adata_clin.n_vars,
                "n_with_clinvar": n_with_clinvar,
                "significance_counts": significance_counts,
                "n_pathogenic": n_pathogenic,
                "n_benign": n_benign,
                "n_uncertain": n_uncertain,
            }

            logger.info(
                f"Clinical databases: {n_with_clinvar}/{adata_clin.n_vars} variants with ClinVar"
            )

            ir = self._create_clinical_db_ir()

            return adata_clin, stats, ir

        except Exception as e:
            logger.exception(f"Error querying clinical databases: {e}")
            raise VariantAnnotationError(f"Clinical database query failed: {str(e)}")

    def prioritize_variants(
        self,
        adata: anndata.AnnData,
        consequence_col: str = "consequence",
        frequency_col: str = "gnomad_af",
        clinvar_col: str = "clinvar_significance",
        cadd_col: str = "cadd_score",
        sift_col: str = "sift_prediction",
        polyphen_col: str = "polyphen_prediction",
    ) -> Tuple[anndata.AnnData, Dict[str, Any], AnalysisStep]:
        """
        Prioritize variants by composite score based on consequence, rarity, and pathogenicity.

        Computes a priority score (0-1) for each variant:
        - Consequence severity (0-0.4): Based on VEP consequence types
        - Population rarity (0-0.3): Lower allele frequency = higher score
        - Pathogenicity (0-0.3): ClinVar, CADD, SIFT, PolyPhen

        Gracefully handles missing annotation columns (components score 0 if absent).

        Args:
            adata: AnnData object with variant annotations in .var
            consequence_col: Column with VEP consequence types
            frequency_col: Column with gnomAD allele frequency
            clinvar_col: Column with ClinVar significance
            cadd_col: Column with CADD phred scores
            sift_col: Column with SIFT predictions
            polyphen_col: Column with PolyPhen predictions

        Returns:
            Tuple[AnnData, Dict, AnalysisStep]: AnnData with priority_score/rank, stats, IR
        """
        try:
            logger.info(f"Prioritizing {adata.n_vars} variants")

            adata_pri = adata.copy()
            n_variants = adata_pri.n_vars

            # Consequence severity scoring (0-0.4)
            consequence_severity_map = {
                "transcript_ablation": 0.4,
                "splice_acceptor_variant": 0.4,
                "splice_donor_variant": 0.4,
                "stop_gained": 0.4,
                "frameshift_variant": 0.4,
                "stop_lost": 0.35,
                "start_lost": 0.35,
                "missense_variant": 0.3,
                "inframe_insertion": 0.2,
                "inframe_deletion": 0.2,
                "protein_altering_variant": 0.2,
                "splice_region_variant": 0.15,
                "incomplete_terminal_codon_variant": 0.1,
                "synonymous_variant": 0.05,
                "coding_sequence_variant": 0.05,
                "5_prime_UTR_variant": 0.03,
                "3_prime_UTR_variant": 0.03,
                "intron_variant": 0.01,
                "intergenic_variant": 0.01,
                "upstream_gene_variant": 0.01,
                "downstream_gene_variant": 0.01,
            }

            consequence_scores = np.zeros(n_variants)
            if consequence_col in adata_pri.var.columns:
                for i in range(n_variants):
                    cons_val = adata_pri.var[consequence_col].iloc[i]
                    if pd.notna(cons_val):
                        # Handle multiple consequences (comma-separated)
                        consequences = str(cons_val).split(",")
                        max_score = 0.0
                        for c in consequences:
                            c = c.strip()
                            max_score = max(
                                max_score,
                                consequence_severity_map.get(c, 0.01),
                            )
                        consequence_scores[i] = max_score

            # Population rarity scoring (0-0.3)
            rarity_scores = np.full(n_variants, 0.15)  # Default: unknown = moderate
            if frequency_col in adata_pri.var.columns:
                for i in range(n_variants):
                    af_val = adata_pri.var[frequency_col].iloc[i]
                    try:
                        af = float(af_val) if pd.notna(af_val) else None
                    except (ValueError, TypeError):
                        af = None

                    if af is not None:
                        if af == 0:
                            rarity_scores[i] = 0.3
                        elif af < 0.001:
                            rarity_scores[i] = 0.25
                        elif af < 0.01:
                            rarity_scores[i] = 0.2
                        elif af < 0.05:
                            rarity_scores[i] = 0.1
                        else:
                            rarity_scores[i] = 0.0

            # Pathogenicity scoring (0-0.3)
            pathogenicity_scores = np.zeros(n_variants)

            # ClinVar component (0-0.3)
            if clinvar_col in adata_pri.var.columns:
                for i in range(n_variants):
                    cv = adata_pri.var[clinvar_col].iloc[i]
                    if pd.notna(cv):
                        cv_lower = str(cv).lower()
                        if "pathogenic" in cv_lower and "likely" not in cv_lower:
                            pathogenicity_scores[i] += 0.3
                        elif (
                            "likely_pathogenic" in cv_lower
                            or "likely pathogenic" in cv_lower
                        ):
                            pathogenicity_scores[i] += 0.25
                        elif "uncertain" in cv_lower:
                            pathogenicity_scores[i] += 0.1

            # CADD component (adds up to 0.1)
            if cadd_col in adata_pri.var.columns:
                for i in range(n_variants):
                    cadd_val = adata_pri.var[cadd_col].iloc[i]
                    try:
                        cadd = float(cadd_val) if pd.notna(cadd_val) else None
                    except (ValueError, TypeError):
                        cadd = None
                    if cadd is not None and cadd > 20:
                        pathogenicity_scores[i] = min(
                            pathogenicity_scores[i] + 0.1, 0.3
                        )

            # SIFT component (adds up to 0.05)
            if sift_col in adata_pri.var.columns:
                for i in range(n_variants):
                    sift_val = adata_pri.var[sift_col].iloc[i]
                    if pd.notna(sift_val) and "deleterious" in str(sift_val).lower():
                        pathogenicity_scores[i] = min(
                            pathogenicity_scores[i] + 0.05, 0.3
                        )

            # PolyPhen component (adds up to 0.05)
            if polyphen_col in adata_pri.var.columns:
                for i in range(n_variants):
                    pp_val = adata_pri.var[polyphen_col].iloc[i]
                    if pd.notna(pp_val) and "probably_damaging" in str(pp_val).lower():
                        pathogenicity_scores[i] = min(
                            pathogenicity_scores[i] + 0.05, 0.3
                        )

            # Composite score
            priority_scores = consequence_scores + rarity_scores + pathogenicity_scores

            # Store results
            adata_pri.var["priority_score"] = priority_scores
            adata_pri.var["priority_rank"] = (
                pd.Series(priority_scores)
                .rank(ascending=False, method="min")
                .astype(int)
                .values
            )

            # Build stats
            n_high = int((priority_scores > 0.6).sum())
            n_medium = int(((priority_scores >= 0.3) & (priority_scores <= 0.6)).sum())
            n_low = int((priority_scores < 0.3).sum())

            # Top 10 variants
            top_indices = np.argsort(priority_scores)[::-1][:10]
            top_variants = []
            for idx in top_indices:
                variant_info = {
                    "variant_id": adata_pri.var.index[idx],
                    "priority_score": float(priority_scores[idx]),
                    "priority_rank": int(adata_pri.var["priority_rank"].iloc[idx]),
                }
                if consequence_col in adata_pri.var.columns:
                    variant_info["consequence"] = str(
                        adata_pri.var[consequence_col].iloc[idx]
                    )
                top_variants.append(variant_info)

            stats = {
                "analysis_type": "variant_prioritization",
                "n_variants_scored": n_variants,
                "n_high_priority": n_high,
                "n_medium_priority": n_medium,
                "n_low_priority": n_low,
                "top_variants": top_variants,
            }

            logger.info(
                f"Prioritization completed: {n_high} high, {n_medium} medium, {n_low} low priority"
            )

            ir = self._create_prioritize_ir()

            return adata_pri, stats, ir

        except Exception as e:
            logger.exception(f"Error in variant prioritization: {e}")
            raise VariantAnnotationError(f"Variant prioritization failed: {str(e)}")

    # ===== Validation Methods =====

    def _validate_inputs(
        self,
        adata: anndata.AnnData,
        annotation_source: str,
        genome_build: str,
        batch_size: int,
    ) -> None:
        """
        Validate inputs for variant annotation.

        Args:
            adata: AnnData object
            annotation_source: Annotation source
            genome_build: Genome build
            batch_size: Batch size

        Raises:
            VariantAnnotationError: If inputs are invalid
        """
        # Check AnnData has variants
        if adata.n_vars == 0:
            raise VariantAnnotationError("No variants found in AnnData object")

        # Check required columns in adata.var
        required_cols = ["CHROM", "POS", "REF", "ALT"]
        missing = [col for col in required_cols if col not in adata.var.columns]
        if missing:
            raise VariantAnnotationError(
                f"Missing required columns in adata.var: {missing}. "
                f"Expected: {required_cols}"
            )

        # Validate annotation source
        if annotation_source not in self.ANNOTATION_SOURCES:
            raise VariantAnnotationError(
                f"Invalid annotation_source: {annotation_source}. "
                f"Supported: {self.ANNOTATION_SOURCES}"
            )

        # Validate genome build
        if genome_build not in self.GENOME_BUILDS:
            raise VariantAnnotationError(
                f"Invalid genome_build: {genome_build}. Supported: {self.GENOME_BUILDS}"
            )

        # Validate batch size
        if batch_size <= 0:
            raise VariantAnnotationError(
                f"Invalid batch_size: {batch_size}. Must be > 0"
            )

        if batch_size > 10000:
            logger.warning(
                f"batch_size={batch_size} exceeds recommended limit of 10000. "
                "Large batches may cause API failures."
            )

        # Check dependencies
        if annotation_source == "genebe" and not HAS_GENEBE:
            raise VariantAnnotationError(
                "genebe not installed. Install with: pip install genebe"
            )

        if annotation_source == "ensembl_vep" and not HAS_REQUESTS:
            raise VariantAnnotationError(
                "requests library not installed. Install with: pip install requests"
            )

    def _normalize_genome_build(self, genome_build: str) -> str:
        """
        Normalize genome build name to standard format.

        Args:
            genome_build: Genome build (hg38, hg19, GRCh38, GRCh37)

        Returns:
            Normalized build name (hg38 or hg19)
        """
        build_map = {
            "hg38": "hg38",
            "grch38": "hg38",
            "hg19": "hg19",
            "grch37": "hg19",
        }
        return build_map[genome_build.lower()]

    # ===== Data Preparation Methods =====

    def _prepare_variant_dataframe(self, adata: anndata.AnnData) -> pd.DataFrame:
        """
        Prepare variant DataFrame from adata.var.

        Args:
            adata: AnnData object with variant data

        Returns:
            DataFrame with columns: chr, pos, ref, alt, variant_id
        """
        var_df = adata.var[["CHROM", "POS", "REF", "ALT"]].copy()
        var_df.columns = ["chr", "pos", "ref", "alt"]

        # Normalize chromosome names (remove "chr" prefix for genebe)
        var_df["chr"] = var_df["chr"].astype(str).str.replace("chr", "", regex=False)

        # Create unique variant identifier
        var_df["variant_id"] = (
            var_df["chr"].astype(str)
            + ":"
            + var_df["pos"].astype(str)
            + ":"
            + var_df["ref"].astype(str)
            + ":"
            + var_df["alt"].astype(str)
        )

        return var_df

    # ===== Annotation Methods =====

    def _annotate_with_genebe(
        self,
        var_df: pd.DataFrame,
        genome_build: str,
        batch_size: int,
        retry_attempts: int,
        retry_delay: float,
        use_cache: bool,
    ) -> pd.DataFrame:
        """
        Annotate variants using genebe batch API.

        Args:
            var_df: Variant DataFrame with columns: chr, pos, ref, alt, variant_id
            genome_build: Genome build (hg38 or hg19)
            batch_size: Number of variants per API call
            retry_attempts: Number of retry attempts
            retry_delay: Delay between retries (seconds)
            use_cache: Whether to use cached annotations

        Returns:
            DataFrame with annotation columns

        Raises:
            VariantAnnotationError: If annotation fails
        """
        logger.info(f"Annotating with genebe (batch_size={batch_size})")

        if not HAS_GENEBE:
            raise VariantAnnotationError(
                "genebe not installed. Install with: pip install genebe"
            )

        # Initialize annotations list
        all_annotations = []

        # Process in batches
        n_variants = len(var_df)
        n_batches = (n_variants + batch_size - 1) // batch_size

        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_variants)
            batch = var_df.iloc[start_idx:end_idx].copy()

            logger.debug(
                f"Processing batch {batch_idx + 1}/{n_batches} ({len(batch)} variants)"
            )

            # Check cache
            if use_cache:
                cached_variants = []
                uncached_variants = []
                for _, row in batch.iterrows():
                    variant_id = row["variant_id"]
                    if variant_id in self._annotation_cache:
                        cached_variants.append(self._annotation_cache[variant_id])
                    else:
                        uncached_variants.append(row)

                if cached_variants:
                    logger.debug(
                        f"Using cached annotations for {len(cached_variants)} variants"
                    )
                    all_annotations.extend(cached_variants)

                if not uncached_variants:
                    continue  # All variants cached

                batch = pd.DataFrame(uncached_variants)

            # Retry logic with exponential backoff
            for attempt in range(retry_attempts):
                try:
                    # Call genebe API
                    result = gnb.annotate(
                        batch[["chr", "pos", "ref", "alt"]],
                        genome=genome_build,
                        output_format="dataframe",
                    )

                    # Validate result
                    if result is None or result.empty:
                        logger.warning(
                            f"Batch {batch_idx + 1}: No annotations returned"
                        )
                        break

                    # Add variant_id for merging
                    result["variant_id"] = batch["variant_id"].values

                    # Cache annotations
                    if use_cache:
                        for _, row in result.iterrows():
                            self._annotation_cache[row["variant_id"]] = row.to_dict()

                    all_annotations.append(result)
                    break  # Success

                except Exception as e:
                    if attempt < retry_attempts - 1:
                        delay = retry_delay * (2**attempt)  # Exponential backoff
                        logger.warning(
                            f"Batch {batch_idx + 1} failed (attempt {attempt + 1}/{retry_attempts}): {e}. "
                            f"Retrying in {delay}s..."
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"Batch {batch_idx + 1} failed after {retry_attempts} attempts: {e}"
                        )
                        raise VariantAnnotationError(
                            f"genebe annotation failed: {str(e)}"
                        )

        # Combine all batches
        if not all_annotations:
            logger.warning("No annotations retrieved from genebe")
            return pd.DataFrame()

        annotations_df = pd.concat(all_annotations, ignore_index=True)

        logger.info(
            f"genebe annotation completed: {len(annotations_df)} variants annotated"
        )

        return annotations_df

    def _annotate_with_ensembl(
        self,
        var_df: pd.DataFrame,
        genome_build: str,
        retry_attempts: int,
        retry_delay: float,
        use_cache: bool,
    ) -> pd.DataFrame:
        """
        Annotate variants using Ensembl VEP REST API (fallback method).

        Note: Ensembl VEP has rate limits (15 requests/second).
        For large datasets (>100K variants), consider local VEP installation.

        Args:
            var_df: Variant DataFrame with columns: chr, pos, ref, alt, variant_id
            genome_build: Genome build (hg38 or hg19)
            retry_attempts: Number of retry attempts
            retry_delay: Delay between retries (seconds)
            use_cache: Whether to use cached annotations

        Returns:
            DataFrame with annotation columns

        Raises:
            VariantAnnotationError: If annotation fails
        """
        logger.info("Annotating with Ensembl VEP REST API (rate limited)")

        if not HAS_REQUESTS:
            raise VariantAnnotationError(
                "requests library not installed. Install with: pip install requests"
            )

        # Ensembl VEP REST API endpoint - use build-specific server
        # GRCh37/hg19 requires the dedicated grch37 server
        # GRCh38/hg38 uses the main rest.ensembl.org server
        if genome_build == "hg19":
            base_url = "https://grch37.rest.ensembl.org"
            logger.info(f"Using GRCh37 VEP server for genome build {genome_build}")
        else:
            base_url = "https://rest.ensembl.org"
            logger.info(f"Using GRCh38 VEP server for genome build {genome_build}")

        # Initialize annotations list
        all_annotations = []

        # Process variants one by one (rate limited)
        n_variants = len(var_df)
        rate_limit_delay = 1.0 / 15  # 15 requests per second

        for i, (idx, row) in enumerate(var_df.iterrows()):
            variant_id = row["variant_id"]

            # Check cache
            if use_cache and variant_id in self._annotation_cache:
                logger.debug(f"Using cached annotation for {variant_id}")
                all_annotations.append(self._annotation_cache[variant_id])
                continue

            chrom = row["chr"]
            pos = int(row["pos"])
            ref = row["ref"]
            alt = row["alt"]

            # Skip variants with non-standard alleles (CNVs like <CN0>, <DEL>, etc.)
            if alt.startswith("<") or ref.startswith("<"):
                logger.debug(f"Skipping non-standard allele variant: {variant_id}")
                continue

            # Construct VEP URL - format: {chrom}:{start}-{end}:{strand}/{alt}
            # For SNPs: start == end (single position)
            # For indels: end = start + len(ref) - 1 (span the entire reference allele)
            # strand=1 (forward)
            end_pos = pos + len(ref) - 1
            url = f"{base_url}/vep/human/region/{chrom}:{pos}-{end_pos}:1/{alt}"

            # Retry logic
            for attempt in range(retry_attempts):
                try:
                    response = requests.get(
                        url,
                        headers={"Content-Type": "application/json"},
                        timeout=10,
                    )

                    if response.status_code == 200:
                        data = response.json()

                        # Parse VEP response
                        annotation = self._parse_vep_response(data, variant_id)
                        all_annotations.append(annotation)

                        # Cache annotation
                        if use_cache:
                            self._annotation_cache[variant_id] = annotation

                        break  # Success

                    elif response.status_code == 429:
                        # Rate limit exceeded
                        delay = retry_delay * (2**attempt)
                        logger.warning(f"Rate limit exceeded. Retrying in {delay}s...")
                        time.sleep(delay)

                    else:
                        logger.warning(
                            f"VEP API error for {variant_id}: {response.status_code}"
                        )
                        break

                except Exception as e:
                    if attempt < retry_attempts - 1:
                        delay = retry_delay * (2**attempt)
                        logger.warning(
                            f"VEP request failed (attempt {attempt + 1}/{retry_attempts}): {e}. "
                            f"Retrying in {delay}s..."
                        )
                        time.sleep(delay)
                    else:
                        logger.error(f"VEP annotation failed for {variant_id}: {e}")

            # Rate limiting
            time.sleep(rate_limit_delay)

            # Progress logging
            if (i + 1) % 100 == 0:
                logger.info(f"Annotated {i + 1}/{n_variants} variants")

        # Convert to DataFrame
        if not all_annotations:
            logger.warning("No annotations retrieved from Ensembl VEP")
            return pd.DataFrame()

        annotations_df = pd.DataFrame(all_annotations)

        logger.info(
            f"Ensembl VEP annotation completed: {len(annotations_df)} variants annotated"
        )

        return annotations_df

    def _parse_vep_response(
        self, vep_data: Dict[str, Any], variant_id: str
    ) -> Dict[str, Any]:
        """
        Parse Ensembl VEP API response.

        Args:
            vep_data: JSON response from VEP API
            variant_id: Variant identifier (chr:pos:ref:alt)

        Returns:
            Dictionary with annotation fields
        """
        annotation = {
            "variant_id": variant_id,
            "gene_symbol": None,
            "gene_id": None,
            "consequence": None,
            "biotype": None,
            "gnomad_af": None,
            "clinvar_significance": None,
            "cadd_score": None,
            "sift_score": None,
            "polyphen_score": None,
        }

        # Extract from VEP response
        if vep_data and isinstance(vep_data, list) and len(vep_data) > 0:
            variant_data = vep_data[0]

            # Always extract most_severe_consequence (works for all variants including intergenic)
            most_severe = variant_data.get("most_severe_consequence")
            if most_severe:
                annotation["consequence"] = most_severe

            # Extract from transcript consequences (if available - not for intergenic)
            transcript_consequences = variant_data.get("transcript_consequences", [])

            if transcript_consequences:
                tc = transcript_consequences[0]
                annotation["gene_symbol"] = tc.get("gene_symbol")
                annotation["gene_id"] = tc.get("gene_id")
                # Use detailed consequence from transcript if available
                annotation["consequence"] = ",".join(tc.get("consequence_terms", []))
                annotation["biotype"] = tc.get("biotype")

                # SIFT and PolyPhen scores
                if "sift_score" in tc:
                    annotation["sift_score"] = tc["sift_score"]
                if "polyphen_score" in tc:
                    annotation["polyphen_score"] = tc["polyphen_score"]

            # Colocated variants (for gnomAD, ClinVar)
            colocated = variant_data.get("colocated_variants", [])
            for cv in colocated:
                # gnomAD allele frequency - check multiple possible keys
                if "frequencies" in cv:
                    freqs = cv["frequencies"]
                    # Try gnomad first, then general af
                    if "gnomad" in freqs:
                        annotation["gnomad_af"] = freqs["gnomad"].get("af")
                    else:
                        # Get first alt allele frequency
                        for allele, freq_data in freqs.items():
                            if isinstance(freq_data, dict) and "af" in freq_data:
                                annotation["gnomad_af"] = freq_data["af"]
                                break

                # ClinVar significance
                if "clin_sig" in cv:
                    annotation["clinvar_significance"] = ",".join(cv["clin_sig"])

        return annotation

    # ===== Merging Methods =====

    def _merge_annotations(
        self, adata: anndata.AnnData, annotations_df: pd.DataFrame
    ) -> None:
        """
        Merge annotations into adata.var.

        Args:
            adata: AnnData object (modified in place)
            annotations_df: Annotations DataFrame
        """
        if annotations_df.empty:
            logger.warning("No annotations to merge")
            return

        # Create variant_id in adata.var for merging
        adata.var["variant_id"] = (
            adata.var["CHROM"].astype(str).str.replace("chr", "", regex=False)
            + ":"
            + adata.var["POS"].astype(str)
            + ":"
            + adata.var["REF"].astype(str)
            + ":"
            + adata.var["ALT"].astype(str)
        )

        # Merge annotations
        for col in self.ANNOTATION_COLUMNS.keys():
            if col in annotations_df.columns:
                # Create mapping dict
                mapping = annotations_df.set_index("variant_id")[col].to_dict()
                # Apply mapping
                adata.var[col] = adata.var["variant_id"].map(mapping)
            else:
                # Column not present, set to None
                adata.var[col] = None

        # Remove temporary variant_id column
        adata.var.drop(columns=["variant_id"], inplace=True)

        logger.debug(f"Merged {len(annotations_df)} annotations into adata.var")

    # ===== Statistics Methods =====

    def _compile_statistics(
        self,
        adata: anndata.AnnData,
        annotation_source: str,
        genome_build: str,
    ) -> Dict[str, Any]:
        """
        Compile annotation statistics.

        Args:
            adata: Annotated AnnData object
            annotation_source: Annotation source used
            genome_build: Genome build used

        Returns:
            Statistics dictionary
        """
        total_variants = adata.n_vars

        # Count annotated variants (those with consequence - includes intergenic)
        # Note: gene_symbol is only present for genic variants, not intergenic
        # Using consequence as the indicator since ALL VEP responses include it
        variants_annotated = (
            adata.var["consequence"].notna().sum()
            if "consequence" in adata.var.columns
            else 0
        )

        annotation_rate_pct = (
            (variants_annotated / total_variants * 100) if total_variants > 0 else 0
        )

        # Count by consequence type
        consequence_counts = {}
        if "consequence" in adata.var.columns:
            consequence_counts = (
                adata.var["consequence"].value_counts().head(10).to_dict()
            )

        # Count by biotype
        biotype_counts = {}
        if "biotype" in adata.var.columns:
            biotype_counts = adata.var["biotype"].value_counts().head(10).to_dict()

        # Clinical significance counts
        clinvar_counts = {}
        if "clinvar_significance" in adata.var.columns:
            clinvar_counts = (
                adata.var["clinvar_significance"]
                .dropna()
                .value_counts()
                .head(10)
                .to_dict()
            )

        stats = {
            "analysis_type": "variant_annotation",
            "annotation_source": annotation_source,
            "genome_build": genome_build,
            "n_variants": total_variants,
            "n_variants_annotated": int(variants_annotated),
            "n_variants_unannotated": int(total_variants - variants_annotated),
            "annotation_rate_pct": float(annotation_rate_pct),
            "top_consequences": consequence_counts,
            "top_biotypes": biotype_counts,
            "clinvar_significance_counts": clinvar_counts,
            "annotation_columns_added": list(self.ANNOTATION_COLUMNS.keys()),
        }

        return stats

    # ===== IR Creation Methods =====

    def _create_annotation_ir(
        self,
        annotation_source: str,
        genome_build: str,
        batch_size: int,
    ) -> AnalysisStep:
        """
        Create Intermediate Representation for variant annotation.

        Args:
            annotation_source: Annotation source
            genome_build: Genome build
            batch_size: Batch size

        Returns:
            AnalysisStep with complete code generation instructions
        """
        # Create parameter schema with Papermill flags
        parameter_schema = {
            "annotation_source": ParameterSpec(
                param_type="str",
                papermill_injectable=True,
                default_value="genebe",
                required=False,
                validation_rule="annotation_source in ['genebe', 'ensembl_vep']",
                description="Annotation source (genebe or ensembl_vep)",
            ),
            "genome_build": ParameterSpec(
                param_type="str",
                papermill_injectable=True,
                default_value="hg38",
                required=False,
                validation_rule="genome_build in ['hg38', 'hg19', 'GRCh38', 'GRCh37']",
                description="Reference genome build",
            ),
            "batch_size": ParameterSpec(
                param_type="int",
                papermill_injectable=True,
                default_value=5000,
                required=False,
                validation_rule="batch_size > 0 and batch_size <= 10000",
                description="Number of variants per API call",
            ),
        }

        # Jinja2 template with parameter placeholders
        code_template = """# Annotate variants with genebe
import pandas as pd
import genebe as gnb

# Prepare variant DataFrame
var_df = adata.var[['CHROM', 'POS', 'REF', 'ALT']].copy()
var_df.columns = ['chr', 'pos', 'ref', 'alt']
var_df['chr'] = var_df['chr'].astype(str).str.replace('chr', '', regex=False)

# Annotate in batches
batch_size = {{ batch_size }}
n_variants = len(var_df)
n_batches = (n_variants + batch_size - 1) // batch_size

all_annotations = []
for batch_idx in range(n_batches):
    start_idx = batch_idx * batch_size
    end_idx = min(start_idx + batch_size, n_variants)
    batch = var_df.iloc[start_idx:end_idx]

    # Call genebe API
    result = gnb.annotate(
        batch[['chr', 'pos', 'ref', 'alt']],
        genome={{ genome_build | tojson }},
        output_format='dataframe'
    )

    if result is not None and not result.empty:
        all_annotations.append(result)

# Combine annotations
if all_annotations:
    annotations_df = pd.concat(all_annotations, ignore_index=True)

    # Merge into adata.var
    for idx, row in annotations_df.iterrows():
        adata.var.loc[idx, 'gene_symbol'] = row.get('gene_symbol')
        adata.var.loc[idx, 'consequence'] = row.get('consequence')
        adata.var.loc[idx, 'gnomad_af'] = row.get('gnomad_af')
        adata.var.loc[idx, 'clinvar_significance'] = row.get('clinvar_significance')
        adata.var.loc[idx, 'cadd_score'] = row.get('cadd_score')

    print(f"Annotated {len(annotations_df)}/{n_variants} variants")
else:
    print("No annotations retrieved")
"""

        # Create AnalysisStep
        ir = AnalysisStep(
            operation="genomics.annotation.annotate_variants",
            tool_name="VariantAnnotationService.annotate_variants",
            description="Annotate variants with gene names, consequences, and pathogenicity scores",
            library="genebe",
            code_template=code_template,
            imports=["import pandas as pd", "import genebe as gnb"],
            parameters={
                "annotation_source": annotation_source,
                "genome_build": genome_build,
                "batch_size": batch_size,
            },
            parameter_schema=parameter_schema,
            input_entities=["adata"],
            output_entities=["adata"],
            execution_context={
                "annotation_type": "functional_genomics",
                "annotation_source": annotation_source,
                "genome_build": genome_build,
            },
            validates_on_export=True,
            requires_validation=False,
        )

        logger.debug(f"Created IR for variant annotation: {ir.operation}")
        return ir

    def _create_normalize_ir(self) -> AnalysisStep:
        """Create AnalysisStep IR for variant normalization."""
        return AnalysisStep(
            operation="genomics.annotation.normalize_variants",
            tool_name="VariantAnnotationService.normalize_variants",
            description="Normalize variant representations (left-trim, split multiallelic)",
            library="lobster",
            code_template="# Variant normalization\n# Left-trim padding, split multiallelic ALTs",
            imports=["import numpy as np", "import pandas as pd"],
            parameters={},
            parameter_schema={},
            input_entities=["adata"],
            output_entities=["adata"],
        )

    def _create_population_freq_ir(self, population: str = None) -> AnalysisStep:
        """Create AnalysisStep IR for population frequency query."""
        return AnalysisStep(
            operation="genomics.annotation.query_population_frequencies",
            tool_name="VariantAnnotationService.query_population_frequencies",
            description="Query gnomAD population allele frequencies",
            library="genebe",
            code_template="# Query population frequencies (gnomAD via genebe/VEP)",
            imports=["import pandas as pd"],
            parameters={"population": population},
            parameter_schema={},
            input_entities=["adata"],
            output_entities=["adata"],
        )

    def _create_clinical_db_ir(self) -> AnalysisStep:
        """Create AnalysisStep IR for clinical database query."""
        return AnalysisStep(
            operation="genomics.annotation.query_clinical_databases",
            tool_name="VariantAnnotationService.query_clinical_databases",
            description="Query ClinVar clinical significance",
            library="genebe",
            code_template="# Query clinical databases (ClinVar via genebe/VEP)",
            imports=["import pandas as pd"],
            parameters={},
            parameter_schema={},
            input_entities=["adata"],
            output_entities=["adata"],
        )

    def _create_prioritize_ir(self) -> AnalysisStep:
        """Create AnalysisStep IR for variant prioritization."""
        return AnalysisStep(
            operation="genomics.annotation.prioritize_variants",
            tool_name="VariantAnnotationService.prioritize_variants",
            description="Prioritize variants by composite score (consequence + rarity + pathogenicity)",
            library="lobster",
            code_template="# Variant prioritization (composite score 0-1)",
            imports=["import numpy as np", "import pandas as pd"],
            parameters={},
            parameter_schema={},
            input_entities=["adata"],
            output_entities=["adata"],
        )
