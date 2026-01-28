"""
Variant annotation service for genomics/DNA data.

This service annotates variants with gene names, functional consequences,
and pathogenicity scores using pygenebe and Ensembl VEP APIs.

All methods return 3-tuples (AnnData, Dict, AnalysisStep) for provenance
tracking and reproducible notebook export via /pipeline export.
"""

import time
from typing import Any, Dict, Optional, Tuple

import anndata
import numpy as np
import pandas as pd

from lobster.core.analysis_ir import AnalysisStep, ParameterSpec
from lobster.utils.logger import get_logger

logger = get_logger(__name__)

# Import pygenebe at module level for testability
try:
    import pygenebe as gnb

    HAS_PYGENEBE = True
except ImportError:
    HAS_PYGENEBE = False
    logger.warning(
        "pygenebe not installed. Install with: pip install pygenebe. "
        "Falling back to Ensembl VEP for annotation."
    )

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
    Centralized variant annotation service using pygenebe and Ensembl VEP.

    This service annotates genomic variants with:
    - Gene names and IDs
    - Functional consequences (e.g., missense, synonymous)
    - Gene biotypes (e.g., protein_coding, lincRNA)
    - Population allele frequencies (gnomAD)
    - Clinical significance (ClinVar)
    - Pathogenicity scores (CADD, SIFT, PolyPhen)

    Primary method: pygenebe (batch processing, comprehensive annotations)
    Fallback method: Ensembl VEP REST API (rate limited)

    All methods follow Lobster's (adata, stats, ir) tuple pattern.
    """

    # Supported annotation sources
    ANNOTATION_SOURCES = ["pygenebe", "ensembl_vep"]

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
        annotation_source: str = "pygenebe",
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
            annotation_source: Annotation source ("pygenebe" or "ensembl_vep")
            genome_build: Reference genome build ("hg38", "hg19", "GRCh38", "GRCh37")
            batch_size: Number of variants per API call (max 10000 for pygenebe)
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
            if annotation_source == "pygenebe":
                annotations_df = self._annotate_with_pygenebe(
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
                f"Invalid genome_build: {genome_build}. "
                f"Supported: {self.GENOME_BUILDS}"
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
        if annotation_source == "pygenebe" and not HAS_PYGENEBE:
            raise VariantAnnotationError(
                "pygenebe not installed. Install with: pip install pygenebe"
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

        # Normalize chromosome names (remove "chr" prefix for pygenebe)
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

    def _annotate_with_pygenebe(
        self,
        var_df: pd.DataFrame,
        genome_build: str,
        batch_size: int,
        retry_attempts: int,
        retry_delay: float,
        use_cache: bool,
    ) -> pd.DataFrame:
        """
        Annotate variants using pygenebe batch API.

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
        logger.info(f"Annotating with pygenebe (batch_size={batch_size})")

        if not HAS_PYGENEBE:
            raise VariantAnnotationError(
                "pygenebe not installed. Install with: pip install pygenebe"
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
                f"Processing batch {batch_idx + 1}/{n_batches} "
                f"({len(batch)} variants)"
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
                    logger.debug(f"Using cached annotations for {len(cached_variants)} variants")
                    all_annotations.extend(cached_variants)

                if not uncached_variants:
                    continue  # All variants cached

                batch = pd.DataFrame(uncached_variants)

            # Retry logic with exponential backoff
            for attempt in range(retry_attempts):
                try:
                    # Call pygenebe API
                    result = gnb.annotate(
                        batch[["chr", "pos", "ref", "alt"]],
                        genome=genome_build,
                        output_format="dataframe",
                    )

                    # Validate result
                    if result is None or result.empty:
                        logger.warning(f"Batch {batch_idx + 1}: No annotations returned")
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
                            f"pygenebe annotation failed: {str(e)}"
                        )

        # Combine all batches
        if not all_annotations:
            logger.warning("No annotations retrieved from pygenebe")
            return pd.DataFrame()

        annotations_df = pd.concat(all_annotations, ignore_index=True)

        logger.info(f"pygenebe annotation completed: {len(annotations_df)} variants annotated")

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

        # Map genome build to Ensembl assembly
        assembly_map = {"hg38": "GRCh38", "hg19": "GRCh37"}
        assembly = assembly_map[genome_build]

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

        logger.info(f"Ensembl VEP annotation completed: {len(annotations_df)} variants annotated")

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
                default_value="pygenebe",
                required=False,
                validation_rule="annotation_source in ['pygenebe', 'ensembl_vep']",
                description="Annotation source (pygenebe or ensembl_vep)",
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
        code_template = """# Annotate variants with pygenebe
import pandas as pd
import pygenebe as gnb

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

    # Call pygenebe API
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
            library="pygenebe",
            code_template=code_template,
            imports=["import pandas as pd", "import pygenebe as gnb"],
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
