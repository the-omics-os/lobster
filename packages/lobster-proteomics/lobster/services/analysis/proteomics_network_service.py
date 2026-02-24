"""
Proteomics network analysis service for WGCNA-like co-expression module identification.

This service implements weighted gene co-expression network analysis (WGCNA) concepts
for proteomics data, enabling identification of protein co-expression modules and
correlation with clinical traits.

All methods return 3-tuples (AnnData, Dict, AnalysisStep) for provenance tracking and
reproducible notebook export via /pipeline export.
"""

from typing import Any, Dict, List, Optional, Tuple

import anndata
import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from lobster.core.analysis_ir import AnalysisStep, ParameterSpec
from lobster.utils.logger import get_logger
from lobster.utils.statistics import benjamini_hochberg

logger = get_logger(__name__)

# Standard WGCNA module colors
WGCNA_COLORS = [
    "turquoise",
    "blue",
    "brown",
    "yellow",
    "green",
    "red",
    "black",
    "pink",
    "magenta",
    "purple",
    "greenyellow",
    "tan",
    "salmon",
    "cyan",
    "midnightblue",
    "lightcyan",
    "grey60",
    "lightgreen",
    "lightyellow",
    "royalblue",
    "darkred",
    "darkgreen",
    "darkturquoise",
    "darkgrey",
    "orange",
    "darkorange",
    "white",
    "skyblue",
    "saddlebrown",
    "steelblue",
    "paleturquoise",
    "violet",
    "darkolivegreen",
    "darkmagenta",
]

# Grey is reserved for unassigned proteins
GREY_MODULE = "grey"

# ============================================================================
# Class Constants - Configurable thresholds and defaults
# ============================================================================

# Power law fitting thresholds
MIN_POWERLAW_DATAPOINTS = 10  # Minimum data points for power law R² fit
MAX_HISTOGRAM_BINS = 20  # Maximum bins for connectivity histogram
MIN_HISTOGRAM_BINS = 3  # Minimum bins for histogram analysis
MIN_NONZERO_BINS = 3  # Minimum non-zero frequency bins for fitting

# WGCNA defaults
DEFAULT_SOFT_POWER = 6  # Standard WGCNA fallback power
DEFAULT_R_SQUARED_CUTOFF = 0.85  # Scale-free topology R² threshold
DEFAULT_MEAN_CONNECTIVITY_CUTOFF = 100  # Maximum mean connectivity

# Module eigengene calculation
MIN_PROTEINS_FOR_PCA = 3  # Minimum proteins to use PCA for eigengene
MIN_SAMPLES_FOR_CORRELATION = 5  # Minimum samples for correlation calculation


class ProteomicsNetworkError(Exception):
    """Base exception for proteomics network analysis operations."""

    pass


class WGCNALiteService:
    """
    WGCNA-lite network analysis service for proteomics data.

    This stateless service provides WGCNA-style co-expression network analysis
    using scikit-learn and scipy, without requiring the R WGCNA package.

    Key features:
    - Protein-protein correlation network construction
    - Hierarchical clustering for module identification
    - Module eigengene (1st PC) calculation
    - Module-trait correlation analysis
    - Standard WGCNA color coding for modules

    Note: This is a "lite" implementation that provides ~80% of WGCNA functionality
    using pure Python. For exact WGCNA parity, use the full PyWGCNA or R WGCNA.

    Example usage:
        service = WGCNALiteService()
        adata_modules, stats, ir = service.identify_modules(
            adata,
            n_top_variable=5000,
            distance_threshold=0.3,
            min_module_size=20
        )
    """

    def __init__(self):
        """Initialize the WGCNA-lite network service."""
        logger.debug("Initializing WGCNALiteService")

    def _create_ir_identify_modules(
        self,
        n_top_variable: int,
        correlation_method: str,
        soft_power: Optional[int],
        distance_threshold: float,
        min_module_size: int,
        merge_cut_height: float,
    ) -> AnalysisStep:
        """Create IR for module identification."""
        return AnalysisStep(
            operation="proteomics.network.identify_modules",
            tool_name="identify_modules",
            description="Identify protein co-expression modules using WGCNA-lite algorithm",
            library="lobster.services.analysis.proteomics_network_service",
            code_template="""# WGCNA-lite module identification
from lobster.services.analysis.proteomics_network_service import WGCNALiteService

service = WGCNALiteService()
adata_modules, stats, _ = service.identify_modules(
    adata,
    n_top_variable={{ n_top_variable }},
    correlation_method={{ correlation_method | tojson }},
    soft_power={{ soft_power }},
    distance_threshold={{ distance_threshold }},
    min_module_size={{ min_module_size }},
    merge_cut_height={{ merge_cut_height }}
)
print(f"Identified {stats['n_modules']} modules")
print(f"Module sizes: {stats['module_sizes']}")""",
            imports=[
                "from lobster.services.analysis.proteomics_network_service import WGCNALiteService"
            ],
            parameters={
                "n_top_variable": n_top_variable,
                "correlation_method": correlation_method,
                "soft_power": soft_power,
                "distance_threshold": distance_threshold,
                "min_module_size": min_module_size,
                "merge_cut_height": merge_cut_height,
            },
            parameter_schema={
                "n_top_variable": ParameterSpec(
                    param_type="int",
                    papermill_injectable=True,
                    default_value=5000,
                    required=False,
                    validation_rule="n_top_variable > 0",
                    description="Number of most variable proteins to use",
                ),
                "correlation_method": ParameterSpec(
                    param_type="str",
                    papermill_injectable=True,
                    default_value="pearson",
                    required=False,
                    validation_rule="correlation_method in ['pearson', 'spearman']",
                    description="Correlation method for network construction",
                ),
                "soft_power": ParameterSpec(
                    param_type="Optional[int]",
                    papermill_injectable=True,
                    default_value=None,
                    required=False,
                    description="Soft thresholding power (if None, uses signed correlation)",
                ),
                "distance_threshold": ParameterSpec(
                    param_type="float",
                    papermill_injectable=True,
                    default_value=0.3,
                    required=False,
                    validation_rule="0 < distance_threshold < 1",
                    description="Distance threshold for hierarchical clustering",
                ),
                "min_module_size": ParameterSpec(
                    param_type="int",
                    papermill_injectable=True,
                    default_value=20,
                    required=False,
                    validation_rule="min_module_size >= 5",
                    description="Minimum number of proteins per module",
                ),
                "merge_cut_height": ParameterSpec(
                    param_type="float",
                    papermill_injectable=True,
                    default_value=0.25,
                    required=False,
                    validation_rule="0 < merge_cut_height < 1",
                    description="Height threshold for merging similar modules",
                ),
            },
            input_entities=["adata"],
            output_entities=["adata_modules"],
        )

    def _create_ir_correlate_modules_with_traits(
        self,
        traits: List[str],
        correlation_method: str,
    ) -> AnalysisStep:
        """Create IR for module-trait correlation."""
        return AnalysisStep(
            operation="proteomics.network.correlate_modules_with_traits",
            tool_name="correlate_modules_with_traits",
            description="Correlate module eigengenes with clinical traits",
            library="lobster.services.analysis.proteomics_network_service",
            code_template="""# Module-trait correlation
from lobster.services.analysis.proteomics_network_service import WGCNALiteService

service = WGCNALiteService()
adata_corr, stats, _ = service.correlate_modules_with_traits(
    adata,
    traits={{ traits | tojson }},
    correlation_method={{ correlation_method | tojson }}
)
print(f"Significant module-trait correlations: {stats['n_significant_correlations']}")""",
            imports=[
                "from lobster.services.analysis.proteomics_network_service import WGCNALiteService"
            ],
            parameters={
                "traits": traits,
                "correlation_method": correlation_method,
            },
            parameter_schema={
                "traits": ParameterSpec(
                    param_type="List[str]",
                    papermill_injectable=True,
                    default_value=[],
                    required=True,
                    description="Clinical traits to correlate with modules",
                ),
                "correlation_method": ParameterSpec(
                    param_type="str",
                    papermill_injectable=True,
                    default_value="pearson",
                    required=False,
                    validation_rule="correlation_method in ['pearson', 'spearman']",
                    description="Correlation method",
                ),
            },
            input_entities=["adata"],
            output_entities=["adata_corr"],
        )

    def _create_ir_pick_soft_threshold(
        self,
        powers: List[int],
        r_squared_cutoff: float,
        mean_connectivity_cutoff: float,
        n_top_variable: int,
        correlation_method: str,
    ) -> AnalysisStep:
        """Create IR for soft power selection."""
        return AnalysisStep(
            operation="proteomics.network.pick_soft_threshold",
            tool_name="pick_soft_threshold",
            description="Determine optimal soft thresholding power using scale-free topology criterion",
            library="lobster.services.analysis.proteomics_network_service",
            code_template="""# Soft power selection for WGCNA
from lobster.services.analysis.proteomics_network_service import WGCNALiteService

service = WGCNALiteService()
results, stats, _ = service.pick_soft_threshold(
    adata,
    powers={{ powers | tojson }},
    r_squared_cutoff={{ r_squared_cutoff }},
    mean_connectivity_cutoff={{ mean_connectivity_cutoff }},
    n_top_variable={{ n_top_variable }},
    correlation_method={{ correlation_method | tojson }}
)

# Display power table
print(results['power_table'].to_string())
print(f"\\nSelected power: {results['selected_power']}")

# Use selected power for module identification
adata_modules, _, _ = service.identify_modules(
    adata, soft_power=results['selected_power']
)""",
            imports=[
                "from lobster.services.analysis.proteomics_network_service import WGCNALiteService"
            ],
            parameters={
                "powers": powers,
                "r_squared_cutoff": r_squared_cutoff,
                "mean_connectivity_cutoff": mean_connectivity_cutoff,
                "n_top_variable": n_top_variable,
                "correlation_method": correlation_method,
            },
            parameter_schema={
                "powers": ParameterSpec(
                    param_type="List[int]",
                    papermill_injectable=True,
                    default_value=list(range(1, 21)),
                    required=False,
                    description="Powers to evaluate (default: 1-20)",
                ),
                "r_squared_cutoff": ParameterSpec(
                    param_type="float",
                    papermill_injectable=True,
                    default_value=0.85,
                    required=False,
                    validation_rule="0 < r_squared_cutoff <= 1",
                    description="R² threshold for scale-free topology",
                ),
                "mean_connectivity_cutoff": ParameterSpec(
                    param_type="float",
                    papermill_injectable=True,
                    default_value=100,
                    required=False,
                    validation_rule="mean_connectivity_cutoff > 0",
                    description="Maximum mean connectivity",
                ),
                "n_top_variable": ParameterSpec(
                    param_type="int",
                    papermill_injectable=True,
                    default_value=5000,
                    required=False,
                    validation_rule="n_top_variable > 0",
                    description="Number of most variable proteins to use",
                ),
                "correlation_method": ParameterSpec(
                    param_type="str",
                    papermill_injectable=True,
                    default_value="pearson",
                    required=False,
                    validation_rule="correlation_method in ['pearson', 'spearman']",
                    description="Correlation method for network construction",
                ),
            },
            input_entities=["adata"],
            output_entities=["results"],
        )

    def pick_soft_threshold(
        self,
        adata: anndata.AnnData,
        powers: Optional[List[int]] = None,
        r_squared_cutoff: float = 0.85,
        mean_connectivity_cutoff: float = 100,
        n_top_variable: int = 5000,
        correlation_method: str = "pearson",
    ) -> Tuple[Dict[str, Any], Dict[str, Any], AnalysisStep]:
        """
        Determine optimal soft thresholding power using scale-free topology criterion.

        This method evaluates multiple powers and selects the first one that achieves
        scale-free topology (R² > cutoff for power law fit of connectivity distribution).
        This is a critical step in WGCNA to ensure the network has scale-free properties.

        Args:
            adata: AnnData object with proteomics data (samples × proteins)
            powers: List of powers to evaluate (default: [1, 2, 3, ..., 20])
            r_squared_cutoff: R² threshold for scale-free topology (default: 0.85)
            mean_connectivity_cutoff: Maximum mean connectivity threshold (default: 100)
            n_top_variable: Number of most variable proteins to use
            correlation_method: Correlation method ('pearson' or 'spearman')

        Returns:
            Tuple[Dict[str, Any], Dict[str, Any], AnalysisStep]:
                - Results dict with 'selected_power' and power evaluation table
                - Statistics dict
                - IR for notebook export

        Example:
            service = WGCNALiteService()
            results, stats, ir = service.pick_soft_threshold(adata)
            print(f"Optimal power: {results['selected_power']}")

            # Use selected power for module identification
            adata_modules, _, _ = service.identify_modules(
                adata, soft_power=results['selected_power']
            )
        """
        try:
            from sklearn.impute import SimpleImputer
            from sklearn.linear_model import LinearRegression

            logger.info("Starting soft power threshold selection")

            if powers is None:
                powers = list(range(1, 21))

            # Get expression matrix
            X = adata.X.copy()
            n_samples, n_proteins = X.shape
            logger.info(f"Input data: {n_samples} samples × {n_proteins} proteins")

            # Handle missing values
            imputer = SimpleImputer(strategy="median")
            X_imputed = imputer.fit_transform(X)

            # Select top variable proteins
            n_to_use = min(n_top_variable, n_proteins)
            protein_variance = np.nanvar(X_imputed, axis=0)
            top_var_indices = np.argsort(protein_variance)[-n_to_use:]
            X_top = X_imputed[:, top_var_indices]

            logger.info(f"Using top {n_to_use} variable proteins for power selection")

            # Compute correlation matrix
            logger.info(f"Computing {correlation_method} correlation matrix...")
            if correlation_method == "pearson":
                corr_matrix = np.corrcoef(X_top.T)
            elif correlation_method == "spearman":
                corr_matrix, _ = stats.spearmanr(X_top, axis=0)
            else:
                raise ProteomicsNetworkError(
                    f"Unknown correlation method: {correlation_method}"
                )

            # Handle NaN values in correlation matrix
            corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

            # Evaluate each power
            logger.info(f"Evaluating {len(powers)} power values...")
            power_results = []

            for power in powers:
                # Compute adjacency: (0.5 + 0.5 * cor)^power (signed network)
                adjacency = np.power(0.5 + 0.5 * corr_matrix, power)
                np.fill_diagonal(adjacency, 0)  # No self-connections

                # Calculate connectivity (sum of adjacency for each node)
                k = adjacency.sum(axis=1)

                # Scale-free topology fitting
                # Fit: log10(P(k)) ~ log10(k)
                k_positive = k[k > 0]

                if len(k_positive) < MIN_POWERLAW_DATAPOINTS:
                    # Not enough data points
                    power_results.append(
                        {
                            "power": power,
                            "r_squared": 0.0,
                            "slope": 0.0,
                            "mean_connectivity": float(np.mean(k)),
                            "median_connectivity": float(np.median(k)),
                            "truncated_r_squared": 0.0,
                        }
                    )
                    continue

                # Discretize k into bins for frequency calculation
                n_bins = min(MAX_HISTOGRAM_BINS, len(k_positive) // 5)
                if n_bins < MIN_HISTOGRAM_BINS:
                    n_bins = MIN_HISTOGRAM_BINS

                hist, bin_edges = np.histogram(k_positive, bins=n_bins)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

                # Filter zero-frequency bins
                nonzero = hist > 0
                if nonzero.sum() < MIN_NONZERO_BINS:
                    power_results.append(
                        {
                            "power": power,
                            "r_squared": 0.0,
                            "slope": 0.0,
                            "mean_connectivity": float(np.mean(k)),
                            "median_connectivity": float(np.median(k)),
                            "truncated_r_squared": 0.0,
                        }
                    )
                    continue

                log_bin = np.log10(bin_centers[nonzero])
                log_freq = np.log10(hist[nonzero])

                # Linear regression for R²
                reg = LinearRegression()
                reg.fit(log_bin.reshape(-1, 1), log_freq)
                r_squared = reg.score(log_bin.reshape(-1, 1), log_freq)
                slope = reg.coef_[0]

                # Calculate mean/median connectivity
                mean_k = np.mean(k)
                median_k = np.median(k)

                # Truncated R² (signed): negative slope expected for scale-free
                # Use sign of slope to indicate directionality
                truncated_r_squared = float(r_squared) * np.sign(slope)

                power_results.append(
                    {
                        "power": power,
                        "r_squared": float(r_squared),
                        "slope": float(slope),
                        "mean_connectivity": float(mean_k),
                        "median_connectivity": float(median_k),
                        "truncated_r_squared": truncated_r_squared,
                    }
                )

            # Select optimal power
            # WGCNA criterion: R² > cutoff AND negative slope AND mean_k < connectivity_cutoff
            selected_power = None
            for result in power_results:
                # Scale-free networks have negative slope (few hubs, many low-degree nodes)
                if (
                    result["r_squared"] > r_squared_cutoff
                    and result["slope"] < 0
                    and result["mean_connectivity"] < mean_connectivity_cutoff
                ):
                    selected_power = result["power"]
                    break

            # Fallback: highest R² with negative slope if none meet all criteria
            if selected_power is None:
                negative_slope_results = [r for r in power_results if r["slope"] < 0]
                if negative_slope_results:
                    selected_power = max(
                        negative_slope_results, key=lambda x: x["r_squared"]
                    )["power"]
                    logger.warning(
                        f"No power achieved R² > {r_squared_cutoff} with acceptable connectivity. "
                        f"Using power={selected_power} (highest R² with negative slope)"
                    )
                else:
                    # Last resort: use power 6 (common WGCNA default)
                    selected_power = DEFAULT_SOFT_POWER
                    logger.warning(
                        f"No suitable power found. Using default power={selected_power}"
                    )

            # Get achieved R² for selected power
            achieved_r_squared = next(
                (r["r_squared"] for r in power_results if r["power"] == selected_power),
                0.0,
            )

            # Prepare results
            power_table = pd.DataFrame(power_results)
            results = {
                "selected_power": selected_power,
                "power_table": power_table,
                "r_squared_cutoff": r_squared_cutoff,
                "mean_connectivity_cutoff": mean_connectivity_cutoff,
                "n_proteins_used": n_to_use,
            }

            analysis_stats = {
                "selected_power": selected_power,
                "achieved_r_squared": achieved_r_squared,
                "n_powers_evaluated": len(powers),
                "n_proteins_used": n_to_use,
                "correlation_method": correlation_method,
                "analysis_type": "soft_power_selection",
            }

            logger.info(
                f"Soft power selection complete: power={selected_power}, R²={achieved_r_squared:.3f}"
            )

            # Create IR
            ir = self._create_ir_pick_soft_threshold(
                powers,
                r_squared_cutoff,
                mean_connectivity_cutoff,
                n_top_variable,
                correlation_method,
            )

            return results, analysis_stats, ir

        except Exception as e:
            logger.exception(f"Error in soft power selection: {e}")
            raise ProteomicsNetworkError(f"Soft power selection failed: {str(e)}")

    def identify_modules(
        self,
        adata: anndata.AnnData,
        n_top_variable: int = 5000,
        correlation_method: str = "pearson",
        soft_power: Optional[int] = None,
        distance_threshold: float = 0.3,
        min_module_size: int = 20,
        merge_cut_height: float = 0.25,
    ) -> Tuple[anndata.AnnData, Dict[str, Any], AnalysisStep]:
        """
        Identify protein co-expression modules using WGCNA-lite algorithm.

        Constructs a correlation network from the most variable proteins,
        applies hierarchical clustering, and assigns WGCNA-style color labels
        to each module.

        Args:
            adata: AnnData object with proteomics data (samples × proteins)
            n_top_variable: Number of most variable proteins to use
            correlation_method: Correlation method ('pearson' or 'spearman')
            soft_power: Soft thresholding power (if None, uses signed correlation)
            distance_threshold: Distance threshold for hierarchical clustering
            min_module_size: Minimum number of proteins per module
            merge_cut_height: Height threshold for merging similar modules

        Returns:
            Tuple[anndata.AnnData, Dict[str, Any], AnalysisStep]:
                - AnnData with module assignments in var['module']
                - Analysis statistics with module info
                - IR for notebook export

        Raises:
            ProteomicsNetworkError: If analysis fails
        """
        try:
            logger.info("Starting WGCNA-lite module identification")

            # Create working copy
            adata_modules = adata.copy()
            original_shape = adata_modules.shape
            logger.info(
                f"Input data shape: {original_shape[0]} samples × {original_shape[1]} proteins"
            )

            # Get expression matrix (samples × proteins)
            X = adata_modules.X.copy()
            n_samples, n_proteins = X.shape
            protein_names = adata_modules.var_names.tolist()

            # C2 FIX: Minimum sample size guard for WGCNA
            MIN_WGCNA_SAMPLES = 15
            if n_samples < MIN_WGCNA_SAMPLES:
                raise ProteomicsNetworkError(
                    f"WGCNA requires at least {MIN_WGCNA_SAMPLES} samples for reliable "
                    f"co-expression network estimation (got {n_samples}). "
                    f"With fewer samples, pairwise correlations are dominated by noise "
                    f"and modules are not biologically meaningful. "
                    f"Consider using differential expression analysis instead."
                )

            if n_samples < 30:
                logger.warning(
                    f"WGCNA with {n_samples} samples is underpowered. "
                    f"Recommend n >= 30 for robust module detection. "
                    f"Results should be interpreted cautiously."
                )

            # Handle missing values with imputation
            from sklearn.impute import SimpleImputer

            imputer = SimpleImputer(strategy="median")
            X_imputed = imputer.fit_transform(X)

            # Select top variable proteins
            n_to_use = min(n_top_variable, n_proteins)
            protein_variance = np.nanvar(X_imputed, axis=0)
            top_var_indices = np.argsort(protein_variance)[-n_to_use:]
            X_top = X_imputed[:, top_var_indices]
            proteins_used = [protein_names[i] for i in top_var_indices]

            logger.info(f"Using top {n_to_use} variable proteins for module detection")

            # Compute correlation matrix (proteins × proteins)
            logger.info(f"Computing {correlation_method} correlation matrix...")
            if correlation_method == "pearson":
                corr_matrix = np.corrcoef(X_top.T)
            elif correlation_method == "spearman":
                corr_matrix, _ = stats.spearmanr(X_top, axis=0)
            else:
                raise ProteomicsNetworkError(
                    f"Unknown correlation method: {correlation_method}"
                )

            # Handle NaN values in correlation matrix
            corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

            # C3 FIX: Apply soft thresholding — auto-pick power if not specified
            if soft_power is not None:
                # Signed network: (0.5 + 0.5 * cor)^power
                adjacency = np.power(0.5 + 0.5 * corr_matrix, soft_power)
                logger.info(f"Applied soft power: {soft_power}")
            else:
                # Auto-select soft power using pick_soft_threshold
                threshold_results, _threshold_stats, _threshold_ir = (
                    self.pick_soft_threshold(adata_modules)
                )
                auto_power = threshold_results.get("selected_power", DEFAULT_SOFT_POWER)
                adjacency = np.power(0.5 + 0.5 * corr_matrix, auto_power)
                logger.info(
                    f"Auto-selected soft power: {auto_power} "
                    f"(R²={threshold_results.get('r_squared_at_selected', 0):.3f})"
                )

            # Convert adjacency to distance (note: this is adjacency-based distance, not TOM)
            np.fill_diagonal(adjacency, 1.0)
            adjacency = np.clip(adjacency, 0, 1)
            adjacency_dissimilarity = 1 - adjacency

            # Hierarchical clustering
            logger.info("Performing hierarchical clustering...")
            # Convert to condensed distance matrix
            condensed_dist = squareform(adjacency_dissimilarity, checks=False)
            # Replace any invalid values
            condensed_dist = np.nan_to_num(
                condensed_dist, nan=1.0, posinf=1.0, neginf=0.0
            )
            condensed_dist = np.clip(condensed_dist, 0, 1)

            linkage_matrix = linkage(condensed_dist, method="average")

            # Cut tree at specified threshold
            clusters = fcluster(
                linkage_matrix, t=distance_threshold, criterion="distance"
            )

            # Identify and handle small modules
            unique_clusters = np.unique(clusters)
            module_sizes = {c: np.sum(clusters == c) for c in unique_clusters}

            # Reassign small modules to grey (unassigned)
            grey_cluster = 0  # Use 0 for grey/unassigned
            for cluster, size in module_sizes.items():
                if size < min_module_size:
                    clusters[clusters == cluster] = grey_cluster

            # Renumber clusters
            unique_clusters = np.unique(clusters)
            cluster_mapping = {}
            color_idx = 0
            for old_cluster in sorted(unique_clusters):
                if old_cluster == grey_cluster:
                    cluster_mapping[old_cluster] = GREY_MODULE
                else:
                    if color_idx < len(WGCNA_COLORS):
                        cluster_mapping[old_cluster] = WGCNA_COLORS[color_idx]
                        color_idx += 1
                    else:
                        cluster_mapping[old_cluster] = f"module_{color_idx}"
                        color_idx += 1

            # Assign colors
            module_colors = [cluster_mapping[c] for c in clusters]

            # Merge similar modules based on eigengene correlation
            if merge_cut_height < 1.0:
                module_colors = self._merge_similar_modules(
                    X_top, module_colors, merge_cut_height
                )

            # Calculate module eigengenes (1st PC of each module)
            logger.info("Calculating module eigengenes...")
            module_eigengenes = self._calculate_module_eigengenes(X_top, module_colors)

            # Store results
            # Initialize all proteins as grey
            all_module_colors = [GREY_MODULE] * n_proteins

            # Assign colors to used proteins
            for i, idx in enumerate(top_var_indices):
                all_module_colors[idx] = module_colors[i]

            adata_modules.var["module"] = all_module_colors
            adata_modules.var["in_network"] = [
                i in top_var_indices for i in range(n_proteins)
            ]

            # Store eigengenes in obsm
            if module_eigengenes:
                # Create DataFrame with module eigengenes
                eigengene_df = pd.DataFrame(
                    {f"ME_{m}": vals for m, vals in module_eigengenes.items()},
                    index=adata_modules.obs_names,
                )
                for col in eigengene_df.columns:
                    adata_modules.obs[col] = eigengene_df[col].values

                # Also store as matrix in obsm
                adata_modules.obsm["module_eigengenes"] = np.column_stack(
                    list(module_eigengenes.values())
                )
            else:
                # No modules found - create empty structures
                adata_modules.obsm["module_eigengenes"] = np.empty((n_samples, 0))

            # Store module info in uns
            module_sizes_final = pd.Series(module_colors).value_counts().to_dict()
            adata_modules.uns["wgcna"] = {
                "modules": {
                    color: [
                        proteins_used[i]
                        for i, c in enumerate(module_colors)
                        if c == color
                    ]
                    for color in set(module_colors)
                },
                "module_sizes": module_sizes_final,
                "module_colors": list(set(module_colors) - {GREY_MODULE}),
                "correlation_method": correlation_method,
                "soft_power": soft_power,
                "distance_threshold": distance_threshold,
                "min_module_size": min_module_size,
                "merge_cut_height": merge_cut_height,
                "n_proteins_used": n_to_use,
                "proteins_used": proteins_used,
                "linkage_matrix": linkage_matrix.tolist(),
            }

            # Calculate statistics
            n_modules = len(set(module_colors) - {GREY_MODULE})
            n_grey = module_colors.count(GREY_MODULE)

            analysis_stats = {
                "n_modules": n_modules,
                "n_proteins_in_modules": len(module_colors) - n_grey,
                "n_proteins_unassigned": n_grey,
                "module_sizes": {
                    k: v for k, v in module_sizes_final.items() if k != GREY_MODULE
                },
                "module_colors": list(set(module_colors) - {GREY_MODULE}),
                "n_proteins_analyzed": n_to_use,
                "correlation_method": correlation_method,
                "soft_power": soft_power,
                "distance_threshold": distance_threshold,
                "analysis_type": "wgcna_module_identification",
            }

            logger.info(
                f"Module identification complete: {n_modules} modules, "
                f"{len(module_colors) - n_grey} proteins assigned, {n_grey} unassigned"
            )

            # Create IR
            ir = self._create_ir_identify_modules(
                n_top_variable,
                correlation_method,
                soft_power,
                distance_threshold,
                min_module_size,
                merge_cut_height,
            )

            return adata_modules, analysis_stats, ir

        except Exception as e:
            logger.exception(f"Error in module identification: {e}")
            raise ProteomicsNetworkError(f"Module identification failed: {str(e)}")

    def correlate_modules_with_traits(
        self,
        adata: anndata.AnnData,
        traits: List[str],
        correlation_method: str = "pearson",
    ) -> Tuple[anndata.AnnData, Dict[str, Any], AnalysisStep]:
        """
        Correlate module eigengenes with clinical traits.

        Args:
            adata: AnnData object with module eigengenes (run identify_modules first)
            traits: List of clinical trait columns in obs
            correlation_method: Correlation method ('pearson' or 'spearman')

        Returns:
            Tuple[anndata.AnnData, Dict[str, Any], AnalysisStep]:
                - AnnData with correlation results in uns
                - Analysis statistics
                - IR for notebook export
        """
        try:
            logger.info(f"Correlating modules with traits: {traits}")

            adata_corr = adata.copy()

            # Check for module eigengenes
            if "module_eigengenes" not in adata_corr.obsm:
                raise ProteomicsNetworkError(
                    "Module eigengenes not found. Run identify_modules first."
                )

            # Get eigengene columns
            me_cols = [c for c in adata_corr.obs.columns if c.startswith("ME_")]
            if not me_cols:
                raise ProteomicsNetworkError(
                    "Module eigengene columns (ME_*) not found in obs"
                )

            # Validate traits
            valid_traits = []
            for trait in traits:
                if trait in adata_corr.obs.columns:
                    valid_traits.append(trait)
                else:
                    logger.warning(f"Trait '{trait}' not found in obs, skipping")

            if not valid_traits:
                raise ProteomicsNetworkError("No valid traits found in obs")

            # Calculate correlations
            correlation_results = []

            for me_col in me_cols:
                module_name = me_col.replace("ME_", "")
                eigengene_values = adata_corr.obs[me_col].values

                for trait in valid_traits:
                    trait_values = adata_corr.obs[trait].values

                    # Handle missing values
                    valid_mask = ~(np.isnan(eigengene_values) | np.isnan(trait_values))
                    if valid_mask.sum() < MIN_SAMPLES_FOR_CORRELATION:
                        continue

                    eg_valid = eigengene_values[valid_mask]
                    trait_valid = trait_values[valid_mask]

                    # Calculate correlation
                    if correlation_method == "pearson":
                        corr, p_value = stats.pearsonr(eg_valid, trait_valid)
                    else:
                        corr, p_value = stats.spearmanr(eg_valid, trait_valid)

                    correlation_results.append(
                        {
                            "module": module_name,
                            "trait": trait,
                            "correlation": float(corr),
                            "p_value": float(p_value),
                            "n_samples": int(valid_mask.sum()),
                        }
                    )

            # Apply FDR correction
            if correlation_results:
                p_values = [r["p_value"] for r in correlation_results]
                fdr_values = self._benjamini_hochberg(p_values)
                for i, result in enumerate(correlation_results):
                    result["fdr"] = fdr_values[i]
                    result["significant"] = fdr_values[i] < 0.05

            # Create correlation matrix for visualization
            modules = list(set(r["module"] for r in correlation_results))
            corr_matrix = pd.DataFrame(index=modules, columns=valid_traits, dtype=float)
            p_matrix = pd.DataFrame(index=modules, columns=valid_traits, dtype=float)

            for result in correlation_results:
                corr_matrix.loc[result["module"], result["trait"]] = result[
                    "correlation"
                ]
                p_matrix.loc[result["module"], result["trait"]] = result["p_value"]

            # Store results
            adata_corr.uns["module_trait_correlation"] = {
                "results": correlation_results,
                "correlation_matrix": corr_matrix.to_dict(),
                "p_value_matrix": p_matrix.to_dict(),
                "traits": valid_traits,
                "modules": modules,
                "correlation_method": correlation_method,
            }

            # Statistics
            significant_results = [
                r for r in correlation_results if r.get("significant", False)
            ]

            analysis_stats = {
                "n_modules": len(modules),
                "n_traits": len(valid_traits),
                "n_tests": len(correlation_results),
                "n_significant_correlations": len(significant_results),
                "significant_pairs": [
                    {
                        "module": r["module"],
                        "trait": r["trait"],
                        "correlation": r["correlation"],
                    }
                    for r in significant_results
                ],
                "correlation_method": correlation_method,
                "analysis_type": "module_trait_correlation",
            }

            logger.info(
                f"Module-trait correlation complete: {len(significant_results)} significant"
            )

            # Create IR
            ir = self._create_ir_correlate_modules_with_traits(
                traits, correlation_method
            )

            return adata_corr, analysis_stats, ir

        except Exception as e:
            logger.exception(f"Error in module-trait correlation: {e}")
            raise ProteomicsNetworkError(f"Module-trait correlation failed: {str(e)}")

    def get_module_proteins(
        self,
        adata: anndata.AnnData,
        module_color: str,
    ) -> List[str]:
        """
        Get list of proteins in a specific module.

        Args:
            adata: AnnData object with module assignments
            module_color: Module color (e.g., 'cyan', 'turquoise')

        Returns:
            List of protein names in the module
        """
        if "module" not in adata.var.columns:
            raise ProteomicsNetworkError(
                "Module assignments not found. Run identify_modules first."
            )

        module_mask = adata.var["module"] == module_color
        proteins = adata.var_names[module_mask].tolist()

        if not proteins:
            available_modules = adata.var["module"].unique().tolist()
            logger.warning(
                f"No proteins found in module '{module_color}'. "
                f"Available modules: {available_modules}"
            )

        return proteins

    def get_module_summary(
        self,
        adata: anndata.AnnData,
    ) -> Dict[str, Any]:
        """
        Get summary of all modules.

        Args:
            adata: AnnData object with module assignments

        Returns:
            Dictionary with module summary statistics
        """
        if "module" not in adata.var.columns:
            raise ProteomicsNetworkError(
                "Module assignments not found. Run identify_modules first."
            )

        module_counts = adata.var["module"].value_counts().to_dict()
        modules = [m for m in module_counts.keys() if m != GREY_MODULE]

        summary = {
            "n_modules": len(modules),
            "module_sizes": {m: module_counts[m] for m in modules},
            "total_assigned": sum(module_counts[m] for m in modules),
            "total_unassigned": module_counts.get(GREY_MODULE, 0),
            "modules": modules,
        }

        return summary

    def calculate_module_membership(
        self,
        adata: anndata.AnnData,
    ) -> Tuple[anndata.AnnData, Dict[str, Any], AnalysisStep]:
        """
        Calculate module membership (kME) for all proteins.

        Module membership is the correlation between a protein's expression
        and the module eigengene. High kME indicates the protein is a
        hub protein for that module.

        Args:
            adata: AnnData object with module eigengenes

        Returns:
            Tuple with AnnData containing kME values, statistics, and IR
        """
        try:
            logger.info("Calculating module membership (kME)")

            adata_kme = adata.copy()

            # Check for module eigengenes
            me_cols = [c for c in adata_kme.obs.columns if c.startswith("ME_")]
            if not me_cols:
                raise ProteomicsNetworkError(
                    "Module eigengene columns not found. Run identify_modules first."
                )

            X = adata_kme.X.copy()
            n_proteins = X.shape[1]

            # Calculate kME for each protein-module pair
            kme_results = {}

            for me_col in me_cols:
                module_name = me_col.replace("ME_", "")
                eigengene = adata_kme.obs[me_col].values

                kme_values = []
                for i in range(n_proteins):
                    protein_expr = X[:, i]

                    # Handle missing values
                    valid_mask = ~(np.isnan(protein_expr) | np.isnan(eigengene))
                    if valid_mask.sum() < MIN_SAMPLES_FOR_CORRELATION:
                        kme_values.append(np.nan)
                        continue

                    corr, _ = stats.pearsonr(
                        protein_expr[valid_mask], eigengene[valid_mask]
                    )
                    kme_values.append(corr)

                adata_kme.var[f"kME_{module_name}"] = kme_values
                kme_results[module_name] = kme_values

            # Identify hub proteins (top 10% kME in their assigned module)
            if "module" in adata_kme.var.columns:
                is_hub = []
                for i, protein in enumerate(adata_kme.var_names):
                    module = adata_kme.var.loc[protein, "module"]
                    if module == GREY_MODULE:
                        is_hub.append(False)
                    else:
                        kme_col = f"kME_{module}"
                        if kme_col in adata_kme.var.columns:
                            module_mask = adata_kme.var["module"] == module
                            module_kme = adata_kme.var.loc[module_mask, kme_col]
                            threshold = module_kme.quantile(0.9)
                            is_hub.append(
                                adata_kme.var.loc[protein, kme_col] >= threshold
                            )
                        else:
                            is_hub.append(False)

                adata_kme.var["is_hub"] = is_hub

            # Statistics
            n_hubs = (
                sum(adata_kme.var.get("is_hub", [])) if "is_hub" in adata_kme.var else 0
            )

            analysis_stats = {
                "n_modules_analyzed": len(me_cols),
                "n_proteins": n_proteins,
                "n_hub_proteins": n_hubs,
                "analysis_type": "module_membership",
            }

            # Simple IR
            ir = AnalysisStep(
                operation="proteomics.network.calculate_module_membership",
                tool_name="calculate_module_membership",
                description="Calculate module membership (kME) for all proteins",
                library="lobster.services.analysis.proteomics_network_service",
                code_template="# See identify_modules for full analysis",
                imports=[],
                parameters={},
                parameter_schema={},
                input_entities=["adata"],
                output_entities=["adata_kme"],
            )

            return adata_kme, analysis_stats, ir

        except Exception as e:
            logger.exception(f"Error calculating module membership: {e}")
            raise ProteomicsNetworkError(
                f"Module membership calculation failed: {str(e)}"
            )

    def _calculate_module_eigengenes(
        self,
        X: np.ndarray,
        module_colors: List[str],
    ) -> Dict[str, np.ndarray]:
        """
        Calculate module eigengenes (1st PC of each module).

        Args:
            X: Expression matrix (samples × proteins)
            module_colors: Module color assignments for each protein

        Returns:
            Dictionary mapping module colors to eigengene vectors
        """
        eigengenes = {}
        unique_modules = set(module_colors) - {GREY_MODULE}

        for module in unique_modules:
            module_mask = [c == module for c in module_colors]
            X_module = X[:, module_mask]

            if X_module.shape[1] < MIN_PROTEINS_FOR_PCA:
                # Not enough proteins, use mean
                eigengenes[module] = np.mean(X_module, axis=1)
            else:
                # Standardize and compute 1st PC
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_module)

                pca = PCA(n_components=1)
                eigengene = pca.fit_transform(X_scaled).flatten()

                # Ensure positive correlation with mean expression
                mean_expr = np.mean(X_module, axis=1)
                if np.corrcoef(eigengene, mean_expr)[0, 1] < 0:
                    eigengene = -eigengene

                eigengenes[module] = eigengene

        return eigengenes

    def _merge_similar_modules(
        self,
        X: np.ndarray,
        module_colors: List[str],
        merge_cut_height: float,
    ) -> List[str]:
        """
        Merge similar modules based on eigengene correlation.

        Args:
            X: Expression matrix
            module_colors: Current module assignments
            merge_cut_height: Correlation threshold for merging

        Returns:
            Updated module color assignments
        """
        # Calculate eigengenes
        eigengenes = self._calculate_module_eigengenes(X, module_colors)

        if len(eigengenes) < 2:
            return module_colors

        # Calculate eigengene correlation matrix
        modules = list(eigengenes.keys())
        n_modules = len(modules)
        eigengene_matrix = np.column_stack([eigengenes[m] for m in modules])

        corr_matrix = np.corrcoef(eigengene_matrix.T)

        # Convert to distance and cluster
        distance_matrix = 1 - corr_matrix
        np.fill_diagonal(distance_matrix, 0)
        condensed_dist = squareform(distance_matrix, checks=False)
        condensed_dist = np.clip(condensed_dist, 0, 2)

        if len(condensed_dist) > 0:
            linkage_matrix = linkage(condensed_dist, method="average")
            merge_clusters = fcluster(
                linkage_matrix, t=merge_cut_height, criterion="distance"
            )

            # Create merge mapping
            merge_mapping = {}
            for i, cluster in enumerate(merge_clusters):
                if cluster not in merge_mapping:
                    merge_mapping[cluster] = modules[i]

            # Update module colors
            module_to_merged = {
                modules[i]: merge_mapping[merge_clusters[i]] for i in range(n_modules)
            }

            new_colors = [
                module_to_merged.get(c, c) if c != GREY_MODULE else GREY_MODULE
                for c in module_colors
            ]

            n_merged = n_modules - len(set(new_colors) - {GREY_MODULE})
            if n_merged > 0:
                logger.info(f"Merged {n_merged} similar modules")

            return new_colors

        return module_colors

    def _benjamini_hochberg(self, p_values: List[float]) -> List[float]:
        """Apply Benjamini-Hochberg FDR correction.

        Delegates to shared implementation in lobster.utils.statistics.
        """
        return benjamini_hochberg(p_values)
