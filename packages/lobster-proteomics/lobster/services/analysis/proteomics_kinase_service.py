"""
Proteomics kinase-substrate enrichment analysis (KSEA) service.

Computes KSEA z-scores from phosphosite fold changes to infer kinase activity.
Uses a built-in minimal SIGNOR-style kinase-substrate mapping with ~20 well-known
kinases, or accepts custom CSV mappings.

All methods return 3-tuples (AnnData, Dict, AnalysisStep) for provenance tracking.
"""

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import anndata
import numpy as np
from scipy.stats import norm

from lobster.core.analysis_ir import AnalysisStep, ParameterSpec
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


class ProteomicsKinaseError(Exception):
    """Raised when kinase enrichment analysis fails."""

    pass


# Built-in minimal SIGNOR-style kinase-substrate mapping
# Format: kinase -> list of substrate_site (gene_residuePosition)
DEFAULT_KINASE_SUBSTRATE_MAP: Dict[str, List[str]] = {
    "AKT1": ["GSK3B_S9", "FOXO3_S253", "MTOR_S1261", "BAD_S136", "TSC2_S939", "PRAS40_T246"],
    "AKT2": ["GSK3A_S21", "FOXO1_S256", "AS160_T642"],
    "MAPK1": ["ELK1_S383", "RSK1_T573", "MNK1_T255", "STMN1_S25", "ETS1_T38"],
    "MAPK3": ["ELK1_S383", "RSK1_T359", "STMN1_S16", "MYC_S62"],
    "MAPK14": ["MAPKAPK2_T334", "ATF2_T69", "MSK1_S376", "MK2_T334", "HSPB1_S82"],
    "CDK1": ["LMNA_S22", "RB1_S807", "VIM_S55", "NPM1_S4"],
    "CDK2": ["RB1_S807", "RB1_T821", "NPAT_S1072", "CDC6_S54"],
    "SRC": ["STAT3_Y705", "FAK_Y397", "P130CAS_Y410", "CORTACTIN_Y421"],
    "EGFR": ["PLCG1_Y783", "STAT3_Y705", "SHC1_Y317", "GAB1_Y627"],
    "ABL1": ["CRK_Y221", "STAT5_Y694", "WASL_Y256"],
    "MTOR": ["RPS6KB1_T389", "EIF4EBP1_T37", "EIF4EBP1_S65", "ULK1_S757"],
    "AMPK": ["ACC1_S79", "TSC2_S1387", "RAPTOR_S792", "ULK1_S555"],
    "CSNK2A1": ["AKT1_S129", "PTEN_S380", "TOP2A_S1106", "TP53_S392"],
    "GSK3B": ["CTNNB1_S33", "CTNNB1_S37", "GS_S641", "MYC_T58", "SNAI1_S104"],
    "CHEK1": ["CDC25C_S216", "CDC25A_S76", "RAD51_T309"],
    "CHEK2": ["CDC25C_S216", "TP53_S20", "BRCA1_S988"],
    "PLK1": ["CDC25C_S198", "BUB1B_S676", "ECT2_T359"],
    "AURKA": ["TPX2_S121", "TACC3_S558", "PLK1_T210"],
    "AURKB": ["H3F3A_S10", "INCENP_S893", "VIM_S72"],
    "JAK2": ["STAT3_Y705", "STAT5A_Y694", "STAT5B_Y699"],
}


class ProteomicsKinaseService:
    """
    Kinase-Substrate Enrichment Analysis (KSEA) service.

    Computes z-scores from phosphosite fold changes to infer kinase activity
    using the KSEA algorithm. Supports a built-in SIGNOR-style kinase-substrate
    mapping or custom CSV mappings.
    """

    def __init__(self):
        """Initialize the KSEA service."""
        logger.debug("Initializing ProteomicsKinaseService")

    def _load_kinase_substrate_map(
        self,
        kinase_substrate_map: Optional[Dict[str, List[str]]] = None,
        custom_mapping_path: Optional[str] = None,
    ) -> Dict[str, List[str]]:
        """
        Load kinase-substrate mapping from provided map, CSV, or defaults.

        Args:
            kinase_substrate_map: Pre-built mapping dict
            custom_mapping_path: Path to CSV with columns: kinase, substrate_site

        Returns:
            Dict mapping kinase names to lists of substrate_site identifiers
        """
        if kinase_substrate_map is not None:
            return kinase_substrate_map

        if custom_mapping_path is not None:
            path = Path(custom_mapping_path)
            if not path.exists():
                raise ProteomicsKinaseError(
                    f"Custom kinase-substrate mapping not found: {custom_mapping_path}"
                )

            mapping: Dict[str, List[str]] = {}
            with open(path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    kinase = row.get("kinase", "").strip()
                    substrate = row.get("substrate_site", "").strip()
                    if kinase and substrate:
                        if kinase not in mapping:
                            mapping[kinase] = []
                        mapping[kinase].append(substrate)

            if not mapping:
                raise ProteomicsKinaseError(
                    f"No valid kinase-substrate pairs found in {custom_mapping_path}. "
                    "Expected CSV with columns: kinase, substrate_site"
                )

            logger.info(
                f"Loaded custom mapping: {len(mapping)} kinases from {custom_mapping_path}"
            )
            return mapping

        # Use built-in default
        logger.info(
            f"Using built-in SIGNOR-style mapping: {len(DEFAULT_KINASE_SUBSTRATE_MAP)} kinases"
        )
        return DEFAULT_KINASE_SUBSTRATE_MAP

    def _extract_site_fold_changes(
        self, adata: anndata.AnnData
    ) -> Dict[str, float]:
        """
        Extract site-level fold changes from DE results or var annotations.

        Looks for fold changes in:
        1. adata.uns['differential_expression']['significant_results'] (from DE analysis)
        2. adata.var['log2_fold_change'] (from var annotations)

        Args:
            adata: AnnData with DE results or fold change annotations

        Returns:
            Dict mapping site identifiers to log2 fold changes
        """
        site_fcs: Dict[str, float] = {}

        # Try DE results first
        de_data = adata.uns.get("differential_expression", {})
        de_results = de_data.get("significant_results", [])

        if de_results:
            for result in de_results:
                protein = result.get("protein", "")
                log2fc = result.get("log2_fold_change", 0.0)
                if protein:
                    site_fcs[protein] = float(log2fc)
            logger.info(
                f"Extracted {len(site_fcs)} site fold changes from DE results"
            )
            return site_fcs

        # Try differential_ptm results
        ptm_data = adata.uns.get("differential_ptm", {})
        ptm_results = ptm_data.get("significant_sites", [])

        if ptm_results:
            for result in ptm_results:
                site = result.get("site", "")
                log2fc = result.get("adjusted_log2fc", result.get("log2_fold_change", 0.0))
                if site:
                    site_fcs[site] = float(log2fc)
            logger.info(
                f"Extracted {len(site_fcs)} site fold changes from PTM DE results"
            )
            return site_fcs

        # Try var annotations
        if "log2_fold_change" in adata.var.columns:
            for name, fc in zip(adata.var_names, adata.var["log2_fold_change"]):
                if not np.isnan(fc):
                    site_fcs[name] = float(fc)
            logger.info(
                f"Extracted {len(site_fcs)} fold changes from var annotations"
            )
            return site_fcs

        raise ProteomicsKinaseError(
            "No fold change data found. Run differential expression analysis first, "
            "or ensure adata.var contains 'log2_fold_change' column."
        )

    def compute_ksea(
        self,
        adata: anndata.AnnData,
        kinase_substrate_map: Optional[Dict[str, List[str]]] = None,
        custom_mapping_path: Optional[str] = None,
        min_substrates: int = 3,
        fdr_threshold: float = 0.05,
    ) -> Tuple[anndata.AnnData, Dict[str, Any], AnalysisStep]:
        """
        Compute Kinase-Substrate Enrichment Analysis (KSEA) z-scores.

        For each kinase, collects matched substrate fold changes and computes:
        z = (mean_substrate_fc - global_mean) / (global_std / sqrt(n))

        Args:
            adata: AnnData with DE or PTM DE results
            kinase_substrate_map: Pre-built mapping (overrides default)
            custom_mapping_path: Path to CSV with kinase-substrate mapping
            min_substrates: Minimum matched substrates required per kinase
            fdr_threshold: FDR threshold for significance

        Returns:
            Tuple[AnnData, Dict, AnalysisStep]: AnnData with KSEA results, stats, IR

        Raises:
            ProteomicsKinaseError: If KSEA computation fails
        """
        try:
            # Load kinase-substrate mapping
            ks_map = self._load_kinase_substrate_map(
                kinase_substrate_map, custom_mapping_path
            )

            # Extract site-level fold changes
            site_fcs = self._extract_site_fold_changes(adata)

            if not site_fcs:
                raise ProteomicsKinaseError("No site-level fold changes available")

            # Compute global statistics
            all_fcs = np.array(list(site_fcs.values()))
            global_mean = float(np.mean(all_fcs))
            global_std = float(np.std(all_fcs, ddof=1))

            if global_std < 1e-10:
                raise ProteomicsKinaseError(
                    "Global fold change standard deviation is near zero. "
                    "Cannot compute KSEA z-scores."
                )

            logger.info(
                f"Global FC stats: mean={global_mean:.4f}, std={global_std:.4f}, "
                f"n_sites={len(all_fcs)}"
            )

            # Compute KSEA z-scores per kinase
            ksea_results = []
            site_fc_keys_upper = {k.upper(): v for k, v in site_fcs.items()}

            for kinase, substrates in ks_map.items():
                # Match substrates to measured sites (case-insensitive)
                matched_fcs = []
                matched_sites = []
                for substrate in substrates:
                    substrate_upper = substrate.upper()
                    if substrate_upper in site_fc_keys_upper:
                        matched_fcs.append(site_fc_keys_upper[substrate_upper])
                        matched_sites.append(substrate)

                if len(matched_fcs) < min_substrates:
                    continue

                # KSEA z-score calculation
                n = len(matched_fcs)
                mean_substrate_fc = float(np.mean(matched_fcs))
                z_score = (mean_substrate_fc - global_mean) / (global_std / np.sqrt(n))
                p_value = float(2 * (1 - norm.cdf(abs(z_score))))

                ksea_results.append(
                    {
                        "kinase": kinase,
                        "n_substrates": n,
                        "matched_sites": matched_sites,
                        "mean_fc": mean_substrate_fc,
                        "z_score": float(z_score),
                        "p_value": p_value,
                    }
                )

            if not ksea_results:
                logger.warning(
                    f"No kinases had >= {min_substrates} matched substrates. "
                    f"Available sites: {len(site_fcs)}, kinases in map: {len(ks_map)}"
                )
                # Still return valid results with empty KSEA
                adata.uns["ksea_results"] = []
                stats = {
                    "n_kinases_tested": 0,
                    "n_significant": 0,
                    "n_sites_available": len(site_fcs),
                    "n_kinases_in_map": len(ks_map),
                    "min_substrates": min_substrates,
                    "top_kinases": [],
                    "analysis_type": "ksea",
                }
                ir = self._create_ir_ksea(
                    min_substrates, fdr_threshold, custom_mapping_path
                )
                return adata, stats, ir

            # Apply FDR correction
            from statsmodels.stats.multitest import fdrcorrection

            p_values = [r["p_value"] for r in ksea_results]
            _, fdr_values = fdrcorrection(p_values, alpha=fdr_threshold, method="indep")

            for i, result in enumerate(ksea_results):
                result["fdr"] = float(fdr_values[i])

            # Store results in AnnData
            adata.uns["ksea_results"] = ksea_results

            # Calculate stats
            significant = [r for r in ksea_results if r["fdr"] < fdr_threshold]
            top_kinases = sorted(
                ksea_results, key=lambda x: abs(x["z_score"]), reverse=True
            )[:10]

            stats = {
                "n_kinases_tested": len(ksea_results),
                "n_significant": len(significant),
                "n_sites_available": len(site_fcs),
                "n_kinases_in_map": len(ks_map),
                "min_substrates": min_substrates,
                "fdr_threshold": fdr_threshold,
                "global_mean_fc": global_mean,
                "global_std_fc": global_std,
                "top_kinases": [
                    {
                        "kinase": k["kinase"],
                        "z_score": k["z_score"],
                        "n_substrates": k["n_substrates"],
                        "fdr": k["fdr"],
                    }
                    for k in top_kinases
                ],
                "analysis_type": "ksea",
            }

            ir = self._create_ir_ksea(
                min_substrates, fdr_threshold, custom_mapping_path
            )

            logger.info(
                f"KSEA complete: {len(ksea_results)} kinases tested, "
                f"{len(significant)} significant (FDR < {fdr_threshold})"
            )
            return adata, stats, ir

        except ProteomicsKinaseError:
            raise
        except Exception as e:
            logger.exception(f"KSEA computation failed: {e}")
            raise ProteomicsKinaseError(f"KSEA computation failed: {str(e)}")

    def _create_ir_ksea(
        self,
        min_substrates: int,
        fdr_threshold: float,
        custom_mapping_path: Optional[str],
    ) -> AnalysisStep:
        """Create IR for KSEA analysis."""
        return AnalysisStep(
            operation="proteomics.analysis.ksea",
            tool_name="ProteomicsKinaseService.compute_ksea",
            description="Kinase-Substrate Enrichment Analysis (KSEA) z-score computation",
            library="scipy.stats",
            code_template="""import numpy as np
from scipy.stats import norm
from statsmodels.stats.multitest import fdrcorrection

# Extract site-level fold changes from DE results
de_results = adata.uns['differential_expression']['significant_results']
site_fcs = {r['protein']: r['log2_fold_change'] for r in de_results}

# Global statistics
all_fcs = np.array(list(site_fcs.values()))
global_mean = np.mean(all_fcs)
global_std = np.std(all_fcs, ddof=1)

# KSEA z-score per kinase
# z = (mean_substrate_fc - global_mean) / (global_std / sqrt(n))
ksea_results = []
for kinase, substrates in kinase_substrate_map.items():
    matched = [site_fcs[s] for s in substrates if s in site_fcs]
    if len(matched) >= {{ min_substrates }}:
        n = len(matched)
        mean_fc = np.mean(matched)
        z = (mean_fc - global_mean) / (global_std / np.sqrt(n))
        p = 2 * (1 - norm.cdf(abs(z)))
        ksea_results.append({'kinase': kinase, 'z_score': z, 'p_value': p, 'n': n})

# FDR correction
p_vals = [r['p_value'] for r in ksea_results]
_, fdr_vals = fdrcorrection(p_vals)
for r, fdr in zip(ksea_results, fdr_vals):
    r['fdr'] = fdr

adata.uns['ksea_results'] = ksea_results""",
            imports=[
                "import numpy as np",
                "from scipy.stats import norm",
                "from statsmodels.stats.multitest import fdrcorrection",
            ],
            parameters={
                "min_substrates": min_substrates,
                "fdr_threshold": fdr_threshold,
                "custom_mapping_path": custom_mapping_path,
            },
            parameter_schema={
                "min_substrates": ParameterSpec(
                    param_type="int",
                    papermill_injectable=True,
                    default_value=3,
                    required=False,
                    validation_rule="min_substrates >= 1",
                    description="Minimum matched substrates per kinase",
                ),
                "fdr_threshold": ParameterSpec(
                    param_type="float",
                    papermill_injectable=True,
                    default_value=0.05,
                    required=False,
                    validation_rule="0 < fdr_threshold <= 1",
                    description="FDR threshold for significance",
                ),
                "custom_mapping_path": ParameterSpec(
                    param_type="Optional[str]",
                    papermill_injectable=True,
                    default_value=None,
                    required=False,
                    description="Path to custom kinase-substrate CSV mapping",
                ),
            },
            input_entities=["adata"],
            output_entities=["adata_ksea"],
        )
