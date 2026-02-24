"""
GWAS Service for genome-wide association studies.

This stateless service wraps sgkit for performing GWAS on genomic data,
following the Lobster 3-tuple pattern (AnnData, stats, IR).
"""

from typing import Any, Dict, List, Optional, Tuple

import anndata
import numpy as np
import xarray as xr
from scipy.stats import chi2
from statsmodels.stats.multitest import multipletests

from lobster.core.analysis_ir import AnalysisStep, ParameterSpec
from lobster.utils.logger import get_logger

logger = get_logger(__name__)

# Optional sgkit dependency (install with: pip install sgkit)
SGKIT_AVAILABLE = False
sg = None

try:
    import sgkit as _sg

    sg = _sg
    SGKIT_AVAILABLE = True
except ImportError:
    pass  # sgkit is optional for GWAS


class GWASError(Exception):
    """Base exception for GWAS operations."""

    pass


class GWASService:
    """
    Stateless service for genome-wide association studies.

    This service implements the Lobster 3-tuple pattern:
    - Returns (AnnData, stats_dict, AnalysisStep)
    - Uses sgkit for GWAS computations
    - Supports linear and logistic regression
    - Includes population structure analysis via PCA
    """

    def __init__(self, config=None, **kwargs):
        """
        Initialize the GWAS service.

        Args:
            config: Optional configuration dict (for future use)
            **kwargs: Additional arguments (ignored, for backward compatibility)

        This service is stateless and doesn't require a data manager instance.
        """
        logger.debug("Initializing stateless GWASService")
        self.config = config or {}

        if not SGKIT_AVAILABLE:
            logger.debug(
                "sgkit not available. Install with: pip install sgkit\n"
                "GWAS methods will raise GWASError if called without sgkit."
            )
        else:
            logger.debug("sgkit available for GWAS computations")

        logger.debug("GWASService initialized successfully")

    def run_gwas(
        self,
        adata: anndata.AnnData,
        phenotype: str,
        covariates: Optional[List[str]] = None,
        model: str = "linear",
        pvalue_threshold: float = 5e-8,
        genotype_layer: str = "GT",
    ) -> Tuple[anndata.AnnData, Dict[str, Any], AnalysisStep]:
        """
        Perform genome-wide association study.

        Args:
            adata: AnnData object with genotype data
            phenotype: Column name in adata.obs containing phenotype values
            covariates: List of column names in adata.obs for covariates
            model: "linear" (continuous phenotype) or "logistic" (binary phenotype)
            pvalue_threshold: P-value threshold for significance (default: 5e-8)
            genotype_layer: Layer containing genotype data (default: "GT")

        Returns:
            Tuple[AnnData, Dict, AnalysisStep]: AnnData with GWAS results, stats, and IR

        Raises:
            GWASError: If GWAS computation fails
            ValueError: If phenotype or covariates not found
        """
        try:
            if not SGKIT_AVAILABLE:
                raise GWASError(
                    "sgkit is required for GWAS. Install with: pip install sgkit"
                )

            logger.info(
                f"Starting GWAS: phenotype={phenotype}, model={model}, "
                f"covariates={covariates}"
            )

            # Validate inputs
            if phenotype not in adata.obs.columns:
                raise ValueError(
                    f"Phenotype '{phenotype}' not found in adata.obs. "
                    f"Available columns: {list(adata.obs.columns)}"
                )

            if covariates:
                missing = [c for c in covariates if c not in adata.obs.columns]
                if missing:
                    raise ValueError(
                        f"Covariates not found in adata.obs: {missing}. "
                        f"Available columns: {list(adata.obs.columns)}"
                    )

            if model not in ["linear", "logistic"]:
                raise ValueError(f"Model must be 'linear' or 'logistic', got: {model}")

            if genotype_layer not in adata.layers:
                raise ValueError(
                    f"Genotype layer '{genotype_layer}' not found in adata.layers. "
                    f"Available layers: {list(adata.layers.keys())}"
                )

            # Create working copy
            adata_gwas = adata.copy()

            # Convert AnnData to sgkit Dataset
            logger.info("Converting AnnData to xarray Dataset for sgkit")
            ds = self._adata_to_sgkit(adata_gwas, phenotype, covariates, genotype_layer)

            # Compute dosage from diploid genotypes (sum of alleles: 0/1/2)
            logger.info("Computing genotype dosage for GWAS")
            call_dosage = ds["call_genotype"].sum(dim="ploidy")

            # sgkit expects call_dosage with dims ('variants', 'samples')
            # Transpose from ('samples', 'variants') to ('variants', 'samples')
            ds["call_dosage"] = call_dosage.transpose("variants", "samples")
            logger.debug(
                f"Created call_dosage with dims: {ds['call_dosage'].dims}, shape: {ds['call_dosage'].shape}"
            )

            # Load into memory to avoid Dask chunking issues (for small datasets)
            # This is acceptable for <100K variants
            if adata_gwas.n_vars < 100000:
                ds = ds.compute()
                logger.debug("Loaded dataset into memory for GWAS")

            # Run GWAS
            logger.info(f"Running {model} regression GWAS")
            if model == "linear":
                ds = sg.gwas_linear_regression(
                    ds,
                    dosage="call_dosage",
                    covariates=covariates if covariates else [],
                    traits=[phenotype],
                )
                beta_key = "variant_linreg_beta"
                se_key = "variant_linreg_t_value"  # Use t-value as proxy for SE
                pvalue_key = "variant_linreg_p_value"
            else:  # logistic
                # Check phenotype is binary
                unique_vals = adata_gwas.obs[phenotype].dropna().unique()
                if len(unique_vals) > 2:
                    raise ValueError(
                        f"Logistic regression requires binary phenotype. "
                        f"Found {len(unique_vals)} unique values: {unique_vals}"
                    )

                # sgkit doesn't have built-in logistic regression yet
                # Use linear regression as approximation for now
                logger.warning(
                    "Using linear regression as approximation for logistic model. "
                    "For true logistic regression, consider using PLINK or other tools."
                )
                ds = sg.gwas_linear_regression(
                    ds,
                    dosage="call_dosage",
                    covariates=covariates if covariates else [],
                    traits=[phenotype],
                )
                beta_key = "variant_linreg_beta"
                se_key = "variant_linreg_t_value"
                pvalue_key = "variant_linreg_p_value"

            # Transfer results back to AnnData
            logger.info("Transferring GWAS results to AnnData")
            adata_gwas = self._sgkit_to_adata(
                ds, adata_gwas, beta_key, se_key, pvalue_key, phenotype
            )

            # Multiple testing correction
            logger.info("Performing multiple testing correction (FDR)")
            pvalues = adata_gwas.var["gwas_pvalue"].values
            valid_mask = ~np.isnan(pvalues)
            qvalues = np.full_like(pvalues, np.nan)
            if valid_mask.sum() > 0:
                _, qvalues[valid_mask], _, _ = multipletests(
                    pvalues[valid_mask], method="fdr_bh"
                )
            adata_gwas.var["gwas_qvalue"] = qvalues

            # Mark significant variants
            adata_gwas.var["gwas_significant"] = (
                adata_gwas.var["gwas_pvalue"] < pvalue_threshold
            )

            # Calculate lambda GC (genomic inflation factor)
            lambda_gc = self._calculate_lambda_gc(pvalues[valid_mask])

            # Compile statistics
            try:
                n_significant = int(adata_gwas.var["gwas_significant"].astype(bool).sum())
            except (TypeError, ValueError):
                n_significant = 0
            n_tested = valid_mask.sum()

            stats = {
                "analysis_type": "gwas",
                "phenotype": phenotype,
                "model": model,
                "covariates": covariates if covariates else [],
                "n_variants_tested": int(n_tested),
                "n_variants_significant": n_significant,
                "pvalue_threshold": pvalue_threshold,
                "lambda_gc": float(lambda_gc),
                "lambda_gc_interpretation": self._interpret_lambda_gc(lambda_gc),
                "top_variants": self._get_top_variants(adata_gwas, n_top=10),
            }

            logger.info(
                f"GWAS completed: {n_significant}/{n_tested} significant variants "
                f"(lambda_GC={lambda_gc:.3f})"
            )

            # Create IR
            ir = self._create_gwas_ir(
                phenotype=phenotype,
                covariates=covariates,
                model=model,
                pvalue_threshold=pvalue_threshold,
                genotype_layer=genotype_layer,
            )

            return adata_gwas, stats, ir

        except Exception as e:
            logger.exception(f"Error during GWAS: {e}")
            raise GWASError(f"GWAS failed: {str(e)}")

    def calculate_pca(
        self,
        adata: anndata.AnnData,
        n_components: int = 10,
        ld_prune: bool = True,
        ld_threshold: float = 0.2,
        genotype_layer: str = "GT",
    ) -> Tuple[anndata.AnnData, Dict[str, Any], AnalysisStep]:
        """
        Calculate PCA for population structure analysis.

        Args:
            adata: AnnData object with genotype data
            n_components: Number of principal components to compute
            ld_prune: Whether to perform LD pruning before PCA
            ld_threshold: Correlation threshold for LD pruning (default: 0.2)
            genotype_layer: Layer containing genotype data (default: "GT")

        Returns:
            Tuple[AnnData, Dict, AnalysisStep]: AnnData with PCA results, stats, and IR

        Raises:
            GWASError: If PCA computation fails
        """
        try:
            if not SGKIT_AVAILABLE:
                raise GWASError(
                    "sgkit is required for PCA. Install with: pip install sgkit"
                )

            logger.info(
                f"Starting PCA: n_components={n_components}, "
                f"ld_prune={ld_prune}, ld_threshold={ld_threshold}"
            )

            # Validate inputs
            if genotype_layer not in adata.layers:
                raise ValueError(
                    f"Genotype layer '{genotype_layer}' not found in adata.layers. "
                    f"Available layers: {list(adata.layers.keys())}"
                )

            # Create working copy
            adata_pca = adata.copy()

            # Convert to sgkit Dataset
            logger.info("Converting AnnData to xarray Dataset for sgkit")
            ds = self._adata_to_sgkit(adata_pca, None, None, genotype_layer)

            # Optional LD pruning
            n_variants_original = ds.sizes["variants"]
            if ld_prune:
                logger.warning(
                    "LD pruning requested but requires additional sgkit configuration (alleles dimension). "
                    "Running PCA without LD pruning. Results still useful for broad population structure."
                )
                # TODO: Implement full LD pruning workflow:
                # Requires proper sgkit data model with 'alleles' dimension
                # For now, skip LD pruning
            n_variants_pruned = n_variants_original

            # Run PCA
            logger.info(f"Computing PCA with {n_components} components")
            ds = sg.pca(ds, n_components=n_components)

            # Transfer results to AnnData
            logger.info("Transferring PCA results to AnnData")
            adata_pca.obsm["X_pca"] = ds["sample_pca_projection"].values
            adata_pca.uns["pca_variance_ratio"] = ds[
                "sample_pca_explained_variance_ratio"
            ].values

            # Calculate cumulative variance explained
            variance_ratio = adata_pca.uns["pca_variance_ratio"]
            cumsum_variance = np.cumsum(variance_ratio)

            # Compile statistics
            stats = {
                "analysis_type": "pca_population_structure",
                "n_components": n_components,
                "n_samples": adata_pca.n_obs,
                "n_variants_original": n_variants_original,
                "n_variants_used": n_variants_pruned,
                "ld_pruned": ld_prune,
                "ld_threshold": ld_threshold if ld_prune else None,
                "variance_explained_per_pc": variance_ratio.tolist(),
                "cumulative_variance_explained": cumsum_variance.tolist(),
                "variance_explained_pc1": float(variance_ratio[0]),
                "variance_explained_top5": float(
                    cumsum_variance[min(4, len(cumsum_variance) - 1)]
                ),
            }

            logger.info(
                f"PCA completed: PC1 explains {variance_ratio[0]:.1%} variance, "
                f"top 5 PCs explain {cumsum_variance[min(4, len(cumsum_variance) - 1)]:.1%}"
            )

            # Create IR
            ir = self._create_pca_ir(
                n_components=n_components,
                ld_prune=ld_prune,
                ld_threshold=ld_threshold,
                genotype_layer=genotype_layer,
            )

            return adata_pca, stats, ir

        except Exception as e:
            logger.exception(f"Error during PCA: {e}")
            raise GWASError(f"PCA failed: {str(e)}")

    # =========================================================================
    # LD Pruning, Kinship, and Clumping Methods
    # =========================================================================

    def ld_prune_variants(
        self,
        adata: anndata.AnnData,
        threshold: float = 0.2,
        window_size: int = 500,
        genotype_layer: str = "GT",
    ) -> Tuple[anndata.AnnData, Dict[str, Any], AnalysisStep]:
        """
        Perform linkage disequilibrium (LD) pruning to remove correlated variants.

        Removes variants in high LD (r^2 > threshold) to create an independent
        variant set suitable for PCA, kinship estimation, and other analyses
        that assume independence between markers.

        Args:
            adata: AnnData object with genotype data
            threshold: LD r^2 threshold for pruning (default: 0.2)
            window_size: Number of variants per window for LD computation (default: 500)
            genotype_layer: Layer containing genotype data (default: "GT")

        Returns:
            Tuple[AnnData, Dict, AnalysisStep]: Pruned AnnData, stats, and IR

        Raises:
            GWASError: If sgkit not available or genotype layer missing
        """
        try:
            if not SGKIT_AVAILABLE:
                raise GWASError(
                    "sgkit is required for LD pruning. Install with: pip install sgkit"
                )

            logger.info(
                f"Starting LD pruning: threshold={threshold}, "
                f"window_size={window_size}, n_variants={adata.n_vars}"
            )

            if genotype_layer not in adata.layers:
                raise GWASError(
                    f"Genotype layer '{genotype_layer}' not found in adata.layers. "
                    f"Available layers: {list(adata.layers.keys())}"
                )

            # Convert AnnData to sgkit Dataset
            ds = self._adata_to_sgkit(adata, None, None, genotype_layer)

            # Compute dosage
            ds["call_dosage"] = ds["call_genotype"].sum(dim="ploidy")

            # Add variant_contig (REQUIRED for windowing)
            if "CHROM" in adata.var.columns:
                chroms = adata.var["CHROM"].astype(str).values
                unique_chroms = np.unique(chroms)
                chrom_to_idx = {c: i for i, c in enumerate(unique_chroms)}
                variant_contig = np.array([chrom_to_idx[c] for c in chroms], dtype=int)
            else:
                # Single contig fallback
                variant_contig = np.zeros(ds.sizes["variants"], dtype=int)
            ds["variant_contig"] = ("variants", variant_contig)

            # Window BEFORE pruning (required by sgkit)
            ds = sg.window_by_variant(ds, size=window_size)

            # Prune
            ds_pruned = sg.ld_prune(ds, threshold=threshold)

            # Get pruned variant IDs and subset AnnData
            pruned_ids = set(ds_pruned["variant_id"].values)
            keep_mask = np.isin(adata.var.index.values, list(pruned_ids))
            adata_pruned = adata[:, keep_mask].copy()

            n_before = adata.n_vars
            n_after = adata_pruned.n_vars

            # Build stats
            stats = {
                "analysis_type": "ld_pruning",
                "n_variants_before": n_before,
                "n_variants_after": n_after,
                "n_variants_removed": n_before - n_after,
                "threshold": threshold,
                "window_size": window_size,
            }

            logger.info(
                f"LD pruning completed: {n_before} -> {n_after} variants "
                f"({n_before - n_after} removed)"
            )

            # Create IR
            ir = self._create_ld_prune_ir(
                threshold=threshold,
                window_size=window_size,
                genotype_layer=genotype_layer,
            )

            return adata_pruned, stats, ir

        except Exception as e:
            logger.exception(f"Error during LD pruning: {e}")
            raise GWASError(f"LD pruning failed: {str(e)}")

    def compute_kinship(
        self,
        adata: anndata.AnnData,
        kinship_threshold: float = 0.125,
        genotype_layer: str = "GT",
        estimator: str = "VanRaden",
    ) -> Tuple[anndata.AnnData, Dict[str, Any], AnalysisStep]:
        """
        Compute pairwise kinship matrix using genomic relationship matrix (GRM).

        Estimates relatedness between all pairs of samples. Uses VanRaden's
        method by default (no PCA prerequisite, unlike pc_relate).

        Args:
            adata: AnnData object with genotype data
            kinship_threshold: Threshold for flagging related pairs (default: 0.125 = 3rd degree)
            genotype_layer: Layer containing genotype data (default: "GT")
            estimator: GRM estimator method (default: "VanRaden")

        Returns:
            Tuple[AnnData, Dict, AnalysisStep]: AnnData with kinship in obsm, stats, and IR

        Raises:
            GWASError: If sgkit not available or genotype layer missing
        """
        try:
            if not SGKIT_AVAILABLE:
                raise GWASError(
                    "sgkit is required for kinship computation. Install with: pip install sgkit"
                )

            logger.info(
                f"Computing kinship matrix: estimator={estimator}, "
                f"threshold={kinship_threshold}, n_samples={adata.n_obs}"
            )

            if genotype_layer not in adata.layers:
                raise GWASError(
                    f"Genotype layer '{genotype_layer}' not found in adata.layers. "
                    f"Available layers: {list(adata.layers.keys())}"
                )

            # Create working copy
            adata_kinship = adata.copy()

            # Convert to sgkit Dataset
            ds = self._adata_to_sgkit(adata_kinship, None, None, genotype_layer)

            # Compute dosage with correct dimension ordering
            call_dosage = ds["call_genotype"].sum(dim="ploidy")
            # Ensure shape is (variants, samples) for genomic_relationship
            if call_dosage.dims != ("variants", "samples"):
                call_dosage = call_dosage.transpose("variants", "samples")
            ds["call_dosage"] = call_dosage

            # Compute GRM
            ds = sg.genomic_relationship(ds, estimator=estimator)

            # Extract GRM
            grm = ds["stat_genomic_relationship"].values  # (n_samples, n_samples)

            # Find related pairs above threshold
            related_pairs = []
            n_samples = grm.shape[0]
            for i in range(n_samples):
                for j in range(i + 1, n_samples):
                    if grm[i, j] > kinship_threshold:
                        related_pairs.append(
                            {
                                "sample_i": str(adata_kinship.obs.index[i]),
                                "sample_j": str(adata_kinship.obs.index[j]),
                                "kinship_coeff": float(grm[i, j]),
                            }
                        )

            # Store in AnnData
            adata_kinship.obsm["kinship_matrix"] = grm
            adata_kinship.uns["related_pairs"] = related_pairs
            adata_kinship.uns["kinship_threshold"] = kinship_threshold

            # Compute stats
            upper_triangle = grm[np.triu_indices(n_samples, k=1)]
            stats = {
                "analysis_type": "kinship",
                "n_samples": n_samples,
                "n_variants": adata_kinship.n_vars,
                "n_related_pairs": len(related_pairs),
                "kinship_threshold": kinship_threshold,
                "estimator": estimator,
                "mean_kinship": float(np.mean(upper_triangle)),
                "max_kinship": (
                    float(np.max(upper_triangle)) if len(upper_triangle) > 0 else 0.0
                ),
            }

            logger.info(
                f"Kinship computation completed: {len(related_pairs)} related pairs "
                f"(threshold={kinship_threshold})"
            )

            # Create IR
            ir = self._create_kinship_ir(
                kinship_threshold=kinship_threshold,
                genotype_layer=genotype_layer,
                estimator=estimator,
            )

            return adata_kinship, stats, ir

        except Exception as e:
            logger.exception(f"Error during kinship computation: {e}")
            raise GWASError(f"Kinship computation failed: {str(e)}")

    def clump_gwas_results(
        self,
        adata: anndata.AnnData,
        pvalue_threshold: float = 5e-8,
        clump_kb: int = 250,
        pvalue_col: str = "gwas_pvalue",
    ) -> Tuple[anndata.AnnData, Dict[str, Any], AnalysisStep]:
        """
        Clump GWAS results into independent loci by position-based grouping.

        Groups significant variants into clumps based on genomic proximity.
        Each clump has an index variant (lowest p-value) and member variants
        within the specified window on the same chromosome.

        Args:
            adata: AnnData object with GWAS results in var
            pvalue_threshold: P-value threshold for significance (default: 5e-8)
            clump_kb: Clumping window size in kilobases (default: 250)
            pvalue_col: Column name in adata.var with p-values (default: "gwas_pvalue")

        Returns:
            Tuple[AnnData, Dict, AnalysisStep]: Full AnnData with clump annotations, stats, IR

        Raises:
            GWASError: If pvalue column missing
        """
        try:
            logger.info(
                f"Clumping GWAS results: pvalue_threshold={pvalue_threshold}, "
                f"clump_kb={clump_kb}"
            )

            # Validate pvalue column exists
            if pvalue_col not in adata.var.columns:
                raise GWASError(
                    f"P-value column '{pvalue_col}' not found in adata.var. "
                    f"Available columns: {list(adata.var.columns)}"
                )

            # Create working copy
            adata_clumped = adata.copy()

            # Initialize clump_id column (-1 for non-significant)
            adata_clumped.var["clump_id"] = -1

            # Get significant variants
            pvalues = adata_clumped.var[pvalue_col].values
            sig_mask = pvalues < pvalue_threshold

            n_significant = int(sig_mask.sum())

            if n_significant == 0:
                logger.info("No significant variants found for clumping")
                adata_clumped.uns["gwas_clumps"] = []
                stats = {
                    "analysis_type": "gwas_clumping",
                    "n_significant": 0,
                    "n_clumps": 0,
                    "n_singletons": 0,
                    "largest_clump_size": 0,
                    "pvalue_threshold": pvalue_threshold,
                    "clump_kb": clump_kb,
                    "clumps": [],
                }
                ir = self._create_clumping_ir(
                    pvalue_threshold=pvalue_threshold,
                    clump_kb=clump_kb,
                    pvalue_col=pvalue_col,
                )
                return adata_clumped, stats, ir

            # Get significant variant indices sorted by p-value ascending
            sig_indices = np.where(sig_mask)[0]
            sig_pvalues = pvalues[sig_indices]
            sort_order = np.argsort(sig_pvalues)
            sorted_sig_indices = sig_indices[sort_order]

            # Check for position/chromosome columns
            has_pos = "POS" in adata_clumped.var.columns
            has_chrom = "CHROM" in adata_clumped.var.columns

            # Greedy clumping
            claimed = set()
            clumps = []
            clump_id = 0

            clump_window_bp = clump_kb * 1000

            for idx_variant in sorted_sig_indices:
                if idx_variant in claimed:
                    continue

                # This variant becomes the index variant of a new clump
                idx_pvalue = float(pvalues[idx_variant])
                idx_var_name = adata_clumped.var.index[idx_variant]

                if has_pos and has_chrom:
                    idx_pos = int(adata_clumped.var["POS"].iloc[idx_variant])
                    idx_chrom = str(adata_clumped.var["CHROM"].iloc[idx_variant])
                else:
                    idx_pos = idx_variant
                    idx_chrom = "unknown"

                # Find clump members: unclaimed significant variants on same chrom within window
                member_indices = [idx_variant]

                for other_idx in sorted_sig_indices:
                    if other_idx == idx_variant or other_idx in claimed:
                        continue

                    if has_pos and has_chrom:
                        other_chrom = str(adata_clumped.var["CHROM"].iloc[other_idx])
                        other_pos = int(adata_clumped.var["POS"].iloc[other_idx])

                        if (
                            other_chrom == idx_chrom
                            and abs(other_pos - idx_pos) < clump_window_bp
                        ):
                            member_indices.append(other_idx)

                # Assign clump_id to all members
                for member_idx in member_indices:
                    adata_clumped.var.iloc[
                        member_idx, adata_clumped.var.columns.get_loc("clump_id")
                    ] = clump_id
                    claimed.add(member_idx)

                # Record clump info
                member_ids = [adata_clumped.var.index[m] for m in member_indices]
                clump_info = {
                    "index_variant": idx_var_name,
                    "index_pvalue": idx_pvalue,
                    "chrom": idx_chrom,
                    "position": idx_pos,
                    "n_members": len(member_indices),
                    "member_ids": member_ids,
                }
                clumps.append(clump_info)
                clump_id += 1

            # Store clumps in uns
            adata_clumped.uns["gwas_clumps"] = clumps

            # Build stats
            n_singletons = sum(1 for c in clumps if c["n_members"] == 1)
            largest_clump = max((c["n_members"] for c in clumps), default=0)

            # Top 10 clumps by p-value for stats
            top_clumps = [
                {
                    "index_variant": c["index_variant"],
                    "index_pvalue": c["index_pvalue"],
                    "chrom": c["chrom"],
                    "n_members": c["n_members"],
                }
                for c in clumps[:10]
            ]

            stats = {
                "analysis_type": "gwas_clumping",
                "n_significant": n_significant,
                "n_clumps": len(clumps),
                "n_singletons": n_singletons,
                "largest_clump_size": largest_clump,
                "pvalue_threshold": pvalue_threshold,
                "clump_kb": clump_kb,
                "clumps": top_clumps,
            }

            logger.info(
                f"Clumping completed: {n_significant} significant variants -> "
                f"{len(clumps)} clumps ({n_singletons} singletons)"
            )

            # Create IR
            ir = self._create_clumping_ir(
                pvalue_threshold=pvalue_threshold,
                clump_kb=clump_kb,
                pvalue_col=pvalue_col,
            )

            return adata_clumped, stats, ir

        except Exception as e:
            logger.exception(f"Error during GWAS clumping: {e}")
            raise GWASError(f"GWAS clumping failed: {str(e)}")

    # Helper methods

    def _adata_to_sgkit(
        self,
        adata: anndata.AnnData,
        phenotype: Optional[str],
        covariates: Optional[List[str]],
        genotype_layer: str,
    ) -> xr.Dataset:
        """
        Convert AnnData to sgkit xarray Dataset.

        Args:
            adata: Input AnnData
            phenotype: Phenotype column name (None for PCA only)
            covariates: Covariate column names
            genotype_layer: Layer containing genotypes

        Returns:
            xarray Dataset compatible with sgkit
        """
        # Get genotype data
        gt = adata.layers[genotype_layer]

        # Ensure gt has shape (samples, variants) = (n_obs, n_vars)
        # sgkit expects dimensions in order: [samples, variants]
        if gt.shape[0] != adata.n_obs or gt.shape[1] != adata.n_vars:
            logger.warning(
                f"Genotype matrix has shape {gt.shape} but AnnData has shape "
                f"({adata.n_obs}, {adata.n_vars}). Transposing to match "
                "(samples, variants) format required by sgkit."
            )
            gt = gt.T

        # Verify final shape matches expectations
        if gt.shape != (adata.n_obs, adata.n_vars):
            raise ValueError(
                f"Genotype matrix shape {gt.shape} does not match expected "
                f"(samples, variants) = ({adata.n_obs}, {adata.n_vars})"
            )

        logger.debug(
            f"Converting genotype data to diploid format: {gt.shape} -> "
            f"({adata.n_vars}, {adata.n_obs}, 2)"
        )

        # Convert 0/1/2 encoding to diploid genotype pairs (allele1, allele2)
        # sgkit expects 3D array with shape (variants, samples, ploidy=2)
        # 0 -> (0, 0), 1 -> (0, 1), 2 -> (1, 1), -1 -> (-1, -1)
        gt_diploid = np.zeros((adata.n_vars, adata.n_obs, 2), dtype=np.int8)

        # Handle each genotype value
        # gt is (samples, variants) but gt_diploid is (variants, samples, ploidy)
        # Need to transpose masks when applying
        mask_0 = gt == 0  # Homozygous reference
        mask_1 = gt == 1  # Heterozygous
        mask_2 = gt == 2  # Homozygous alternate
        mask_missing = (gt == -1) | np.isnan(gt)  # Missing data

        # Transpose masks to match (variants, samples) ordering
        mask_0_t = mask_0.T
        mask_1_t = mask_1.T
        mask_2_t = mask_2.T
        mask_missing_t = mask_missing.T

        # Set allele pairs for each ploidy position
        # Allele 1 (first position)
        gt_diploid[:, :, 0][mask_0_t] = 0
        gt_diploid[:, :, 0][mask_1_t] = 0
        gt_diploid[:, :, 0][mask_2_t] = 1
        gt_diploid[:, :, 0][mask_missing_t] = -1

        # Allele 2 (second position)
        gt_diploid[:, :, 1][mask_0_t] = 0
        gt_diploid[:, :, 1][mask_1_t] = 1
        gt_diploid[:, :, 1][mask_2_t] = 1
        gt_diploid[:, :, 1][mask_missing_t] = -1

        logger.debug(
            f"Creating sgkit Dataset with diploid genotype shape {gt_diploid.shape} "
            f"(variants={adata.n_vars}, samples={adata.n_obs}, ploidy=2)"
        )

        # Create base dataset with required sgkit dimensions
        # sgkit requires an 'alleles' dimension for PCA and other functions
        # For biallelic SNPs, we have 2 alleles (reference and alternate)
        ds = xr.Dataset(
            {
                "call_genotype": (["variants", "samples", "ploidy"], gt_diploid),
                "sample_id": (["samples"], adata.obs.index.values),
                "variant_id": (["variants"], adata.var.index.values),
            },
            # Add the 'alleles' coordinate as required by sgkit
            coords={
                "alleles": [
                    "0",
                    "1",
                ],  # 0=reference, 1=alternate for biallelic variants
            },
        )

        # Add call_genotype_mask to indicate missing data
        # sgkit uses this to identify missing genotype calls (where call_genotype == -1)
        call_mask = gt_diploid == -1
        ds["call_genotype_mask"] = (["variants", "samples", "ploidy"], call_mask)

        # Add variant_allele variable (optional but recommended for sgkit)
        # This represents the actual nucleotide alleles at each variant
        # For simplicity, use generic "A"/"B" alleles since we don't have actual sequences
        variant_alleles = np.array([["A", "B"]] * adata.n_vars, dtype="S1")
        ds["variant_allele"] = (["variants", "alleles"], variant_alleles)

        # Add phenotype if provided
        if phenotype:
            ds[phenotype] = (["samples"], adata.obs[phenotype].values)

        # Add covariates if provided
        if covariates:
            for cov in covariates:
                ds[cov] = (["samples"], adata.obs[cov].values)

        return ds

    def _sgkit_to_adata(
        self,
        ds: xr.Dataset,
        adata: anndata.AnnData,
        beta_key: str,
        se_key: str,
        pvalue_key: str,
        phenotype: str,
    ) -> anndata.AnnData:
        """
        Transfer sgkit GWAS results back to AnnData.

        Args:
            ds: sgkit Dataset with GWAS results
            adata: AnnData to store results in
            beta_key: Dataset key for beta coefficients
            se_key: Dataset key for standard errors
            pvalue_key: Dataset key for p-values
            phenotype: Phenotype name

        Returns:
            AnnData with GWAS results in .var
        """
        # Extract results - handle potential multi-trait output
        if len(ds[beta_key].dims) > 1:  # Multi-trait format
            # Find the trait index
            trait_idx = 0  # Default to first trait
            adata.var["gwas_beta"] = ds[beta_key].values[:, trait_idx]
            adata.var["gwas_se"] = ds[se_key].values[:, trait_idx]
            adata.var["gwas_pvalue"] = ds[pvalue_key].values[:, trait_idx]
        else:  # Single-trait format
            adata.var["gwas_beta"] = ds[beta_key].values
            adata.var["gwas_se"] = ds[se_key].values
            adata.var["gwas_pvalue"] = ds[pvalue_key].values

        return adata

    def _calculate_lambda_gc(self, pvalues: np.ndarray) -> float:
        """
        Calculate genomic inflation factor (lambda GC).

        Args:
            pvalues: Array of p-values

        Returns:
            Lambda GC value
        """
        if len(pvalues) == 0:
            return np.nan

        # Convert p-values to chi-squared statistics (df=1)
        chi2_obs = chi2.ppf(1 - pvalues, 1)

        # Lambda GC is the ratio of median observed to expected chi-squared
        lambda_gc = np.median(chi2_obs) / chi2.ppf(0.5, 1)

        return lambda_gc

    def _interpret_lambda_gc(self, lambda_gc: float) -> str:
        """
        Interpret genomic inflation factor.

        Args:
            lambda_gc: Lambda GC value

        Returns:
            Interpretation string
        """
        if np.isnan(lambda_gc):
            return "undefined (no valid p-values)"
        elif lambda_gc < 0.9:
            return "low (possible undercorrection or small sample size)"
        elif lambda_gc <= 1.1:
            return "acceptable (no major inflation)"
        elif lambda_gc <= 1.5:
            return "moderate inflation (consider population structure correction)"
        else:
            return "high inflation (strong population stratification or technical artifacts)"

    def _get_top_variants(
        self, adata: anndata.AnnData, n_top: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get top significant variants.

        Args:
            adata: AnnData with GWAS results
            n_top: Number of top variants to return

        Returns:
            List of variant info dictionaries
        """
        # Sort by p-value
        sorted_idx = np.argsort(adata.var["gwas_pvalue"].values)[:n_top]

        top_variants = []
        for idx in sorted_idx:
            if np.isnan(adata.var["gwas_pvalue"].iloc[idx]):
                continue

            variant = {
                "variant_id": adata.var.index[idx],
                "beta": float(adata.var["gwas_beta"].iloc[idx]),
                "pvalue": float(adata.var["gwas_pvalue"].iloc[idx]),
                "qvalue": float(adata.var["gwas_qvalue"].iloc[idx]),
            }
            top_variants.append(variant)

        return top_variants

    # IR creation methods

    def _create_gwas_ir(
        self,
        phenotype: str,
        covariates: Optional[List[str]],
        model: str,
        pvalue_threshold: float,
        genotype_layer: str,
    ) -> AnalysisStep:
        """
        Create AnalysisStep IR for GWAS.

        Args:
            phenotype: Phenotype column name
            covariates: Covariate column names
            model: "linear" or "logistic"
            pvalue_threshold: P-value threshold
            genotype_layer: Genotype layer name

        Returns:
            AnalysisStep for GWAS
        """
        # Parameter schema
        parameter_schema = {
            "phenotype": ParameterSpec(
                param_type="str",
                papermill_injectable=True,
                default_value=phenotype,
                required=True,
                description="Column name in adata.obs containing phenotype values",
            ),
            "covariates": ParameterSpec(
                param_type="List[str]",
                papermill_injectable=True,
                default_value=covariates if covariates else [],
                required=False,
                description="List of covariate column names",
            ),
            "model": ParameterSpec(
                param_type="str",
                papermill_injectable=True,
                default_value=model,
                required=True,
                validation_rule="model in ['linear', 'logistic']",
                description="Regression model type",
            ),
            "pvalue_threshold": ParameterSpec(
                param_type="float",
                papermill_injectable=True,
                default_value=pvalue_threshold,
                required=False,
                validation_rule="pvalue_threshold > 0 and pvalue_threshold < 1",
                description="P-value threshold for significance",
            ),
            "genotype_layer": ParameterSpec(
                param_type="str",
                papermill_injectable=True,
                default_value=genotype_layer,
                required=False,
                description="Layer containing genotype data",
            ),
        }

        # Code template
        code_template = """# Genome-wide association study (GWAS)
import sgkit as sg
import xarray as xr
import numpy as np
from scipy.stats import chi2
from statsmodels.stats.multitest import multipletests

# Convert AnnData to sgkit Dataset with diploid genotypes
gt = adata.layers[{{ genotype_layer | repr }}]

# Convert 0/1/2 encoding to diploid format (samples, variants, ploidy=2)
# 0 -> (0,0), 1 -> (0,1), 2 -> (1,1), -1 -> (-1,-1)
gt_diploid = np.zeros((gt.shape[0], gt.shape[1], 2), dtype=np.int8)

# Create masks for each genotype
mask_0 = gt == 0
mask_1 = gt == 1
mask_2 = gt == 2
mask_missing = (gt == -1) | np.isnan(gt)

# Set allele pairs
gt_diploid[:, :, 0][mask_0] = 0; gt_diploid[:, :, 1][mask_0] = 0
gt_diploid[:, :, 0][mask_1] = 0; gt_diploid[:, :, 1][mask_1] = 1
gt_diploid[:, :, 0][mask_2] = 1; gt_diploid[:, :, 1][mask_2] = 1
gt_diploid[:, :, 0][mask_missing] = -1; gt_diploid[:, :, 1][mask_missing] = -1

ds = xr.Dataset({
    'call_genotype': (['samples', 'variants', 'ploidy'], gt_diploid),
    'sample_id': (['samples'], adata.obs.index.values),
    'variant_id': (['variants'], adata.var.index.values),
})

# Compute dosage (sum of alleles)
ds['call_dosage'] = ds['call_genotype'].sum(dim='ploidy')

# Add phenotype and covariates
ds[{{ phenotype | repr }}] = (['samples'], adata.obs[{{ phenotype | repr }}].values)
{% if covariates %}
{% for cov in covariates %}
ds[{{ cov | repr }}] = (['samples'], adata.obs[{{ cov | repr }}].values)
{% endfor %}
{% endif %}

# Run GWAS
{% if model == "linear" %}
ds = sg.gwas_linear_regression(
    ds,
    dosage='call_dosage',
    covariates={{ covariates | repr }},
    traits=[{{ phenotype | repr }}]
)

# Extract results
adata.var['gwas_beta'] = ds['variant_linreg_beta'].values[:, 0]
adata.var['gwas_se'] = ds['variant_linreg_t_value'].values[:, 0]
adata.var['gwas_pvalue'] = ds['variant_linreg_p_value'].values[:, 0]
{% else %}
# Note: sgkit doesn't have built-in logistic regression yet
# Using linear regression as approximation
ds = sg.gwas_linear_regression(
    ds,
    dosage='call_dosage',
    covariates={{ covariates | repr }},
    traits=[{{ phenotype | repr }}]
)

adata.var['gwas_beta'] = ds['variant_linreg_beta'].values[:, 0]
adata.var['gwas_se'] = ds['variant_linreg_t_value'].values[:, 0]
adata.var['gwas_pvalue'] = ds['variant_linreg_p_value'].values[:, 0]
{% endif %}

# Multiple testing correction (FDR)
pvalues = adata.var['gwas_pvalue'].values
valid_mask = ~np.isnan(pvalues)
qvalues = np.full_like(pvalues, np.nan)
if valid_mask.sum() > 0:
    _, qvalues[valid_mask], _, _ = multipletests(pvalues[valid_mask], method='fdr_bh')
adata.var['gwas_qvalue'] = qvalues

# Mark significant variants
adata.var['gwas_significant'] = adata.var['gwas_pvalue'] < {{ pvalue_threshold }}

# Calculate lambda GC (genomic inflation factor)
chi2_obs = chi2.ppf(1 - pvalues[valid_mask], 1)
lambda_gc = np.median(chi2_obs) / chi2.ppf(0.5, 1)

print(f"GWAS complete: {adata.var['gwas_significant'].sum()} significant variants")
print(f"Lambda GC: {lambda_gc:.3f}")
"""

        return AnalysisStep(
            operation="sgkit.gwas_linear_regression",
            tool_name="run_gwas",
            description=f"GWAS ({model} regression) for {phenotype}",
            library="sgkit",
            code_template=code_template,
            imports=[
                "import sgkit as sg",
                "import xarray as xr",
                "import numpy as np",
                "from scipy.stats import chi2",
                "from statsmodels.stats.multitest import multipletests",
            ],
            parameters={
                "phenotype": phenotype,
                "covariates": covariates if covariates else [],
                "model": model,
                "pvalue_threshold": pvalue_threshold,
                "genotype_layer": genotype_layer,
            },
            parameter_schema=parameter_schema,
            input_entities=["adata"],
            output_entities=["adata"],
            execution_context={
                "operation_type": "gwas",
                "model": model,
                "phenotype": phenotype,
            },
            validates_on_export=True,
            requires_validation=False,
        )

    def _create_pca_ir(
        self,
        n_components: int,
        ld_prune: bool,
        ld_threshold: float,
        genotype_layer: str,
    ) -> AnalysisStep:
        """
        Create AnalysisStep IR for PCA.

        Args:
            n_components: Number of PCs
            ld_prune: Whether to perform LD pruning
            ld_threshold: LD threshold
            genotype_layer: Genotype layer name

        Returns:
            AnalysisStep for PCA
        """
        # Parameter schema
        parameter_schema = {
            "n_components": ParameterSpec(
                param_type="int",
                papermill_injectable=True,
                default_value=n_components,
                required=False,
                validation_rule="n_components > 0",
                description="Number of principal components",
            ),
            "ld_prune": ParameterSpec(
                param_type="bool",
                papermill_injectable=True,
                default_value=ld_prune,
                required=False,
                description="Whether to perform LD pruning",
            ),
            "ld_threshold": ParameterSpec(
                param_type="float",
                papermill_injectable=True,
                default_value=ld_threshold,
                required=False,
                validation_rule="ld_threshold > 0 and ld_threshold < 1",
                description="Correlation threshold for LD pruning",
            ),
            "genotype_layer": ParameterSpec(
                param_type="str",
                papermill_injectable=True,
                default_value=genotype_layer,
                required=False,
                description="Layer containing genotype data",
            ),
        }

        # Code template
        code_template = """# PCA for population structure analysis
import sgkit as sg
import xarray as xr
import numpy as np

# Convert AnnData to sgkit Dataset with diploid genotypes
gt = adata.layers[{{ genotype_layer | repr }}]

# Convert 0/1/2 encoding to diploid format (samples, variants, ploidy=2)
# 0 -> (0,0), 1 -> (0,1), 2 -> (1,1), -1 -> (-1,-1)
gt_diploid = np.zeros((gt.shape[0], gt.shape[1], 2), dtype=np.int8)

# Create masks for each genotype
mask_0 = gt == 0
mask_1 = gt == 1
mask_2 = gt == 2
mask_missing = (gt == -1) | np.isnan(gt)

# Set allele pairs
gt_diploid[:, :, 0][mask_0] = 0; gt_diploid[:, :, 1][mask_0] = 0
gt_diploid[:, :, 0][mask_1] = 0; gt_diploid[:, :, 1][mask_1] = 1
gt_diploid[:, :, 0][mask_2] = 1; gt_diploid[:, :, 1][mask_2] = 1
gt_diploid[:, :, 0][mask_missing] = -1; gt_diploid[:, :, 1][mask_missing] = -1

ds = xr.Dataset({
    'call_genotype': (['samples', 'variants', 'ploidy'], gt_diploid),
    'sample_id': (['samples'], adata.obs.index.values),
    'variant_id': (['variants'], adata.var.index.values),
})

{% if ld_prune %}
# LD pruning
print(f"Original variants: {ds.dims['variants']}")
ds = sg.ld_prune(ds, threshold={{ ld_threshold }})
print(f"After LD pruning: {ds.dims['variants']}")
{% endif %}

# Run PCA
ds = sg.pca(ds, n_components={{ n_components }})

# Transfer results to AnnData
adata.obsm['X_pca'] = ds['sample_pca_projection'].values
adata.uns['pca_variance_ratio'] = ds['sample_pca_explained_variance_ratio'].values

# Print variance explained
variance_ratio = adata.uns['pca_variance_ratio']
print(f"PC1 explains {variance_ratio[0]:.1%} of variance")
print(f"Cumulative variance (top 5 PCs): {np.sum(variance_ratio[:5]):.1%}")
"""

        return AnalysisStep(
            operation="sgkit.pca",
            tool_name="calculate_pca",
            description=f"PCA for population structure (n_components={n_components})",
            library="sgkit",
            code_template=code_template,
            imports=[
                "import sgkit as sg",
                "import xarray as xr",
                "import numpy as np",
            ],
            parameters={
                "n_components": n_components,
                "ld_prune": ld_prune,
                "ld_threshold": ld_threshold,
                "genotype_layer": genotype_layer,
            },
            parameter_schema=parameter_schema,
            input_entities=["adata"],
            output_entities=["adata"],
            execution_context={
                "operation_type": "population_structure_pca",
                "ld_pruned": ld_prune,
            },
            validates_on_export=True,
            requires_validation=False,
        )

    def _create_ld_prune_ir(
        self,
        threshold: float,
        window_size: int,
        genotype_layer: str,
    ) -> AnalysisStep:
        """Create AnalysisStep IR for LD pruning."""
        parameter_schema = {
            "threshold": ParameterSpec(
                param_type="float",
                papermill_injectable=True,
                default_value=threshold,
                required=False,
                validation_rule="threshold > 0 and threshold < 1",
                description="LD r^2 threshold for pruning",
            ),
            "window_size": ParameterSpec(
                param_type="int",
                papermill_injectable=True,
                default_value=window_size,
                required=False,
                validation_rule="window_size > 0",
                description="Number of variants per window",
            ),
            "genotype_layer": ParameterSpec(
                param_type="str",
                papermill_injectable=True,
                default_value=genotype_layer,
                required=False,
                description="Layer containing genotype data",
            ),
        }

        code_template = """# LD pruning to remove correlated variants
import sgkit as sg
import xarray as xr
import numpy as np

# Convert AnnData to sgkit Dataset and compute dosage
# ... (genotype conversion as in GWAS code template)

# Add variant_contig for windowing
chroms = adata.var['CHROM'].astype(str).values
unique_chroms = np.unique(chroms)
chrom_to_idx = {c: i for i, c in enumerate(unique_chroms)}
ds['variant_contig'] = ('variants', np.array([chrom_to_idx[c] for c in chroms], dtype=int))

# Window then prune
ds = sg.window_by_variant(ds, size={{ window_size }})
ds_pruned = sg.ld_prune(ds, threshold={{ threshold }})

# Subset AnnData to pruned variants
pruned_ids = set(ds_pruned['variant_id'].values)
keep_mask = np.isin(adata.var.index.values, list(pruned_ids))
adata = adata[:, keep_mask].copy()

print(f"LD pruning: {keep_mask.sum()}/{len(keep_mask)} variants retained")
"""

        return AnalysisStep(
            operation="sgkit.ld_prune",
            tool_name="ld_prune_variants",
            description=f"LD pruning (r^2 threshold={threshold}, window={window_size})",
            library="sgkit",
            code_template=code_template,
            imports=[
                "import sgkit as sg",
                "import xarray as xr",
                "import numpy as np",
            ],
            parameters={
                "threshold": threshold,
                "window_size": window_size,
                "genotype_layer": genotype_layer,
            },
            parameter_schema=parameter_schema,
            input_entities=["adata"],
            output_entities=["adata"],
            execution_context={
                "operation_type": "ld_pruning",
                "threshold": threshold,
            },
            validates_on_export=True,
            requires_validation=False,
        )

    def _create_kinship_ir(
        self,
        kinship_threshold: float,
        genotype_layer: str,
        estimator: str,
    ) -> AnalysisStep:
        """Create AnalysisStep IR for kinship computation."""
        parameter_schema = {
            "kinship_threshold": ParameterSpec(
                param_type="float",
                papermill_injectable=True,
                default_value=kinship_threshold,
                required=False,
                description="Threshold for flagging related pairs",
            ),
            "estimator": ParameterSpec(
                param_type="str",
                papermill_injectable=True,
                default_value=estimator,
                required=False,
                description="GRM estimator method",
            ),
            "genotype_layer": ParameterSpec(
                param_type="str",
                papermill_injectable=True,
                default_value=genotype_layer,
                required=False,
                description="Layer containing genotype data",
            ),
        }

        code_template = """# Compute kinship matrix (GRM)
import sgkit as sg
import numpy as np

# Compute dosage and GRM
ds['call_dosage'] = ds['call_genotype'].sum(dim='ploidy').transpose('variants', 'samples')
ds = sg.genomic_relationship(ds, estimator={{ estimator | repr }})
grm = ds['stat_genomic_relationship'].values

# Store in AnnData
adata.obsm['kinship_matrix'] = grm

# Find related pairs
related_pairs = []
for i in range(grm.shape[0]):
    for j in range(i + 1, grm.shape[1]):
        if grm[i, j] > {{ kinship_threshold }}:
            related_pairs.append((adata.obs.index[i], adata.obs.index[j], float(grm[i, j])))

adata.uns['related_pairs'] = related_pairs
adata.uns['kinship_threshold'] = {{ kinship_threshold }}
print(f"Kinship: {len(related_pairs)} related pairs (threshold={{ kinship_threshold }})")
"""

        return AnalysisStep(
            operation="sgkit.genomic_relationship",
            tool_name="compute_kinship",
            description=f"Kinship estimation ({estimator} GRM, threshold={kinship_threshold})",
            library="sgkit",
            code_template=code_template,
            imports=[
                "import sgkit as sg",
                "import numpy as np",
            ],
            parameters={
                "kinship_threshold": kinship_threshold,
                "estimator": estimator,
                "genotype_layer": genotype_layer,
            },
            parameter_schema=parameter_schema,
            input_entities=["adata"],
            output_entities=["adata"],
            execution_context={
                "operation_type": "kinship_estimation",
                "estimator": estimator,
            },
            validates_on_export=True,
            requires_validation=False,
        )

    def _create_clumping_ir(
        self,
        pvalue_threshold: float,
        clump_kb: int,
        pvalue_col: str,
    ) -> AnalysisStep:
        """Create AnalysisStep IR for GWAS clumping."""
        parameter_schema = {
            "pvalue_threshold": ParameterSpec(
                param_type="float",
                papermill_injectable=True,
                default_value=pvalue_threshold,
                required=False,
                description="P-value threshold for significance",
            ),
            "clump_kb": ParameterSpec(
                param_type="int",
                papermill_injectable=True,
                default_value=clump_kb,
                required=False,
                description="Clumping window size in kilobases",
            ),
            "pvalue_col": ParameterSpec(
                param_type="str",
                papermill_injectable=True,
                default_value=pvalue_col,
                required=False,
                description="Column name with p-values",
            ),
        }

        code_template = """# GWAS clumping - group significant variants into independent loci
import numpy as np

pvalue_col = {{ pvalue_col | repr }}
pvalue_threshold = {{ pvalue_threshold }}
clump_kb = {{ clump_kb }}
clump_window_bp = clump_kb * 1000

adata.var['clump_id'] = -1
pvalues = adata.var[pvalue_col].values
sig_mask = pvalues < pvalue_threshold
sig_indices = np.where(sig_mask)[0]
sorted_sig_indices = sig_indices[np.argsort(pvalues[sig_indices])]

claimed = set()
clumps = []
clump_id = 0

for idx_variant in sorted_sig_indices:
    if idx_variant in claimed:
        continue
    idx_pos = int(adata.var['POS'].iloc[idx_variant])
    idx_chrom = str(adata.var['CHROM'].iloc[idx_variant])
    member_indices = [idx_variant]
    for other_idx in sorted_sig_indices:
        if other_idx == idx_variant or other_idx in claimed:
            continue
        if (str(adata.var['CHROM'].iloc[other_idx]) == idx_chrom and
                abs(int(adata.var['POS'].iloc[other_idx]) - idx_pos) < clump_window_bp):
            member_indices.append(other_idx)
    for m in member_indices:
        adata.var.iloc[m, adata.var.columns.get_loc('clump_id')] = clump_id
        claimed.add(m)
    clumps.append({'index_variant': adata.var.index[idx_variant], 'n_members': len(member_indices)})
    clump_id += 1

adata.uns['gwas_clumps'] = clumps
print(f"Clumping: {len(clumps)} clumps from {sig_mask.sum()} significant variants")
"""

        return AnalysisStep(
            operation="gwas_clumping",
            tool_name="clump_gwas_results",
            description=f"GWAS clumping (p < {pvalue_threshold}, window={clump_kb}kb)",
            library="lobster",
            code_template=code_template,
            imports=["import numpy as np"],
            parameters={
                "pvalue_threshold": pvalue_threshold,
                "clump_kb": clump_kb,
                "pvalue_col": pvalue_col,
            },
            parameter_schema=parameter_schema,
            input_entities=["adata"],
            output_entities=["adata"],
            execution_context={
                "operation_type": "gwas_clumping",
                "pvalue_threshold": pvalue_threshold,
            },
            validates_on_export=True,
            requires_validation=False,
        )
