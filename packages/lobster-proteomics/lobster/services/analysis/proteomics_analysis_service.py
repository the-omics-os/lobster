"""
Proteomics analysis service for statistical analysis and pathway enrichment.

This service implements professional-grade analysis methods specifically designed for
proteomics data including statistical testing with missing value handling, pathway enrichment
analysis, GO term analysis, and protein network analysis.

All methods return 3-tuples (AnnData, Dict, AnalysisStep) for provenance tracking and
reproducible notebook export via /pipeline export.
"""

from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

import anndata
import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from lobster.core.analysis_ir import AnalysisStep, ParameterSpec
from lobster.services.analysis.pathway_enrichment_service import (
    PathwayEnrichmentError,
    PathwayEnrichmentService,
)
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


class ProteomicsAnalysisError(Exception):
    """Base exception for proteomics analysis operations."""

    pass


class ProteomicsAnalysisService:
    """
    Advanced analysis service for proteomics data.

    This stateless service provides comprehensive analysis methods including
    statistical testing, dimensionality reduction, clustering, pathway enrichment,
    and protein network analysis following best practices from proteomics pipelines.
    """

    def __init__(self):
        """
        Initialize the proteomics analysis service.

        This service is stateless and doesn't require a data manager instance.
        """
        logger.debug("Initializing stateless ProteomicsAnalysisService")

        # Define common pathway databases (simplified for demonstration)
        self.pathway_databases = {
            "go_biological_process": "Gene Ontology Biological Process",
            "go_molecular_function": "Gene Ontology Molecular Function",
            "go_cellular_component": "Gene Ontology Cellular Component",
            "kegg_pathway": "KEGG Pathway",
            "reactome": "Reactome Pathway",
            "string_db": "STRING Database",
        }

        logger.debug("ProteomicsAnalysisService initialized successfully")

    def _create_ir_statistical_testing(
        self,
        group_column: str,
        test_method: str,
        comparison_type: str,
        min_observations: int,
        handle_missing: str,
    ) -> AnalysisStep:
        """Create IR for statistical testing."""
        return AnalysisStep(
            operation="proteomics.analysis.perform_statistical_testing",
            tool_name="perform_statistical_testing",
            description="Perform statistical testing between groups with missing value handling",
            library="lobster.services.analysis.proteomics_analysis_service",
            code_template="""# Statistical testing
from lobster.services.analysis.proteomics_analysis_service import ProteomicsAnalysisService

service = ProteomicsAnalysisService()
adata_stats, stats, _ = service.perform_statistical_testing(
    adata,
    group_column={{ group_column | tojson }},
    test_method={{ test_method | tojson }},
    comparison_type={{ comparison_type | tojson }},
    min_observations={{ min_observations }},
    handle_missing={{ handle_missing | tojson }}
)
print(f"Tests: {stats['n_tests_performed']}, Significant: {stats['n_significant_results']}")""",
            imports=[
                "from lobster.services.analysis.proteomics_analysis_service import ProteomicsAnalysisService"
            ],
            parameters={
                "group_column": group_column,
                "test_method": test_method,
                "comparison_type": comparison_type,
                "min_observations": min_observations,
                "handle_missing": handle_missing,
            },
            parameter_schema={
                "group_column": ParameterSpec(
                    param_type="str",
                    papermill_injectable=True,
                    default_value="group",
                    required=True,
                    description="Column in obs containing group labels",
                ),
                "test_method": ParameterSpec(
                    param_type="str",
                    papermill_injectable=True,
                    default_value="t_test",
                    required=False,
                    validation_rule="test_method in ['t_test', 'mann_whitney', 'anova', 'kruskal']",
                    description="Statistical test method",
                ),
                "comparison_type": ParameterSpec(
                    param_type="str",
                    papermill_injectable=True,
                    default_value="all_pairs",
                    required=False,
                    validation_rule="comparison_type in ['all_pairs', 'vs_rest', 'multi_group']",
                    description="Type of comparison between groups",
                ),
                "min_observations": ParameterSpec(
                    param_type="int",
                    papermill_injectable=True,
                    default_value=3,
                    required=False,
                    validation_rule="min_observations >= 2",
                    description="Minimum observations per group for testing",
                ),
                "handle_missing": ParameterSpec(
                    param_type="str",
                    papermill_injectable=True,
                    default_value="skip",
                    required=False,
                    validation_rule="handle_missing in ['skip', 'impute_mean', 'impute_median']",
                    description="How to handle missing values",
                ),
            },
            input_entities=["adata"],
            output_entities=["adata_stats"],
        )

    def _create_ir_dimensionality_reduction(
        self,
        method: str,
        n_components: int,
        perplexity: float,
        random_state: int,
    ) -> AnalysisStep:
        """Create IR for dimensionality reduction."""
        return AnalysisStep(
            operation="proteomics.analysis.perform_dimensionality_reduction",
            tool_name="perform_dimensionality_reduction",
            description="Perform dimensionality reduction on proteomics data (PCA, t-SNE, UMAP-like)",
            library="lobster.services.analysis.proteomics_analysis_service",
            code_template="""# Dimensionality reduction
from lobster.services.analysis.proteomics_analysis_service import ProteomicsAnalysisService

service = ProteomicsAnalysisService()
adata_reduced, stats, _ = service.perform_dimensionality_reduction(
    adata,
    method={{ method | tojson }},
    n_components={{ n_components }},
    perplexity={{ perplexity }},
    random_state={{ random_state }}
)
print(f"Reduced to {stats['output_dimensions']} dimensions using {method}")""",
            imports=[
                "from lobster.services.analysis.proteomics_analysis_service import ProteomicsAnalysisService"
            ],
            parameters={
                "method": method,
                "n_components": n_components,
                "perplexity": perplexity,
                "random_state": random_state,
            },
            parameter_schema={
                "method": ParameterSpec(
                    param_type="str",
                    papermill_injectable=True,
                    default_value="pca",
                    required=False,
                    validation_rule="method in ['pca', 'tsne', 'umap_like']",
                    description="Dimensionality reduction method",
                ),
                "n_components": ParameterSpec(
                    param_type="int",
                    papermill_injectable=True,
                    default_value=50,
                    required=False,
                    validation_rule="n_components > 0",
                    description="Number of components for PCA or output dimensions",
                ),
                "perplexity": ParameterSpec(
                    param_type="float",
                    papermill_injectable=True,
                    default_value=30.0,
                    required=False,
                    validation_rule="perplexity > 0",
                    description="Perplexity parameter for t-SNE",
                ),
                "random_state": ParameterSpec(
                    param_type="int",
                    papermill_injectable=True,
                    default_value=42,
                    required=False,
                    description="Random state for reproducibility",
                ),
            },
            input_entities=["adata"],
            output_entities=["adata_reduced"],
        )

    def _create_ir_clustering_analysis(
        self,
        clustering_method: str,
        n_clusters: int,
        use_pca: bool,
        n_pca_components: int,
    ) -> AnalysisStep:
        """Create IR for clustering analysis."""
        return AnalysisStep(
            operation="proteomics.analysis.perform_clustering_analysis",
            tool_name="perform_clustering_analysis",
            description="Perform clustering analysis on proteomics data (K-means, hierarchical, Gaussian mixture)",
            library="lobster.services.analysis.proteomics_analysis_service",
            code_template="""# Clustering analysis
from lobster.services.analysis.proteomics_analysis_service import ProteomicsAnalysisService

service = ProteomicsAnalysisService()
adata_clustered, stats, _ = service.perform_clustering_analysis(
    adata,
    clustering_method={{ clustering_method | tojson }},
    n_clusters={{ n_clusters }},
    use_pca={{ use_pca | tojson }},
    n_pca_components={{ n_pca_components }}
)
print(f"Identified {stats['n_clusters']} clusters using {clustering_method}")""",
            imports=[
                "from lobster.services.analysis.proteomics_analysis_service import ProteomicsAnalysisService"
            ],
            parameters={
                "clustering_method": clustering_method,
                "n_clusters": n_clusters,
                "use_pca": use_pca,
                "n_pca_components": n_pca_components,
            },
            parameter_schema={
                "clustering_method": ParameterSpec(
                    param_type="str",
                    papermill_injectable=True,
                    default_value="kmeans",
                    required=False,
                    validation_rule="clustering_method in ['kmeans', 'hierarchical', 'gaussian_mixture']",
                    description="Clustering method",
                ),
                "n_clusters": ParameterSpec(
                    param_type="int",
                    papermill_injectable=True,
                    default_value=4,
                    required=False,
                    validation_rule="n_clusters > 0",
                    description="Number of clusters",
                ),
                "use_pca": ParameterSpec(
                    param_type="bool",
                    papermill_injectable=True,
                    default_value=True,
                    required=False,
                    description="Whether to use PCA-reduced data for clustering",
                ),
                "n_pca_components": ParameterSpec(
                    param_type="int",
                    papermill_injectable=True,
                    default_value=50,
                    required=False,
                    validation_rule="n_pca_components > 0",
                    description="Number of PCA components if use_pca is True",
                ),
            },
            input_entities=["adata"],
            output_entities=["adata_clustered"],
        )

    def perform_statistical_testing(
        self,
        adata: anndata.AnnData,
        group_column: str,
        test_method: str = "t_test",
        comparison_type: str = "all_pairs",
        min_observations: int = 3,
        handle_missing: str = "skip",
    ) -> Tuple[anndata.AnnData, Dict[str, Any], AnalysisStep]:
        """
        Perform statistical testing between groups with missing value handling.

        Args:
            adata: AnnData object with proteomics data
            group_column: Column in obs containing group labels
            test_method: Statistical test ('t_test', 'mann_whitney', 'anova', 'kruskal')
            comparison_type: Type of comparison ('all_pairs', 'vs_rest', 'multi_group')
            min_observations: Minimum observations per group for testing
            handle_missing: How to handle missing values ('skip', 'impute_mean', 'impute_median')

        Returns:
            Tuple[anndata.AnnData, Dict[str, Any], AnalysisStep]: AnnData with test results,
                analysis stats, and IR for notebook export

        Raises:
            ProteomicsAnalysisError: If testing fails
        """
        try:
            logger.info(f"Starting statistical testing with method: {test_method}")

            # Create working copy
            adata_stats = adata.copy()
            original_shape = adata_stats.shape
            logger.info(
                f"Input data shape: {original_shape[0]} samples × {original_shape[1]} proteins"
            )

            if group_column not in adata_stats.obs.columns:
                raise ProteomicsAnalysisError(
                    f"Group column '{group_column}' not found in obs"
                )

            X = adata_stats.X.copy()
            groups = adata_stats.obs[group_column]
            unique_groups = groups.unique()

            logger.info(f"Found {len(unique_groups)} groups: {list(unique_groups)}")

            # Handle missing values if requested
            if handle_missing != "skip":
                X = self._handle_missing_values_for_testing(X, handle_missing)

            # Perform statistical tests based on comparison type
            if comparison_type == "multi_group" and len(unique_groups) > 2:
                test_results = self._perform_multigroup_testing(
                    X, groups, unique_groups, test_method, min_observations
                )
            else:
                test_results = self._perform_pairwise_testing(
                    X,
                    groups,
                    unique_groups,
                    test_method,
                    comparison_type,
                    min_observations,
                )

            # Store results in AnnData
            if test_results:
                # Convert results to DataFrame for easier handling
                pd.DataFrame(test_results)

                # Store in uns
                adata_stats.uns["statistical_tests"] = {
                    "results": test_results,
                    "parameters": {
                        "group_column": group_column,
                        "test_method": test_method,
                        "comparison_type": comparison_type,
                        "min_observations": min_observations,
                        "handle_missing": handle_missing,
                    },
                }

                # Add significant proteins flag to var
                if len(test_results) > 0:
                    significant_proteins = set()
                    for result in test_results:
                        if result.get("p_adjusted", 1.0) < 0.05:
                            significant_proteins.add(result["protein"])

                    adata_stats.var["is_significant"] = [
                        protein in significant_proteins
                        for protein in adata_stats.var_names
                    ]

            # Calculate analysis statistics
            n_tests_performed = len(test_results) if test_results else 0
            n_significant = (
                sum(1 for r in test_results if r.get("p_adjusted", 1.0) < 0.05)
                if test_results
                else 0
            )

            analysis_stats = {
                "test_method": test_method,
                "comparison_type": comparison_type,
                "n_groups": len(unique_groups),
                "group_sizes": groups.value_counts().to_dict(),
                "n_tests_performed": n_tests_performed,
                "n_significant_results": n_significant,
                "significance_rate": (
                    (n_significant / n_tests_performed)
                    if n_tests_performed > 0
                    else 0.0
                ),
                "handle_missing": handle_missing,
                "samples_processed": adata_stats.n_obs,
                "proteins_processed": adata_stats.n_vars,
                "analysis_type": "statistical_testing",
            }

            logger.info(
                f"Statistical testing completed: {n_tests_performed} tests, {n_significant} significant"
            )

            # Create IR for provenance tracking
            ir = self._create_ir_statistical_testing(
                group_column,
                test_method,
                comparison_type,
                min_observations,
                handle_missing,
            )
            return adata_stats, analysis_stats, ir

        except Exception as e:
            logger.exception(f"Error in statistical testing: {e}")
            raise ProteomicsAnalysisError(f"Statistical testing failed: {str(e)}")

    def perform_dimensionality_reduction(
        self,
        adata: anndata.AnnData,
        method: str = "pca",
        n_components: int = 50,
        perplexity: float = 30.0,
        random_state: int = 42,
    ) -> Tuple[anndata.AnnData, Dict[str, Any], AnalysisStep]:
        """
        Perform dimensionality reduction on proteomics data.

        Args:
            adata: AnnData object with proteomics data
            method: Reduction method ('pca', 'tsne', 'umap_like')
            n_components: Number of components for PCA or output dimensions for t-SNE
            perplexity: Perplexity parameter for t-SNE
            random_state: Random state for reproducibility

        Returns:
            Tuple[anndata.AnnData, Dict[str, Any], AnalysisStep]: AnnData with reduced dimensions,
                analysis stats, and IR for notebook export

        Raises:
            ProteomicsAnalysisError: If reduction fails
        """
        try:
            logger.info(f"Starting dimensionality reduction with method: {method}")

            # Create working copy
            adata_reduced = adata.copy()
            original_shape = adata_reduced.shape
            logger.info(
                f"Input data shape: {original_shape[0]} samples × {original_shape[1]} proteins"
            )

            X = adata_reduced.X.copy()

            # Handle missing values for dimensionality reduction
            from sklearn.impute import SimpleImputer

            imputer = SimpleImputer(strategy="mean")
            X_imputed = imputer.fit_transform(X)

            # Standardize data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_imputed)

            # Perform dimensionality reduction
            if method == "pca":
                reduction_results = self._perform_pca(X_scaled, n_components)
            elif method == "tsne":
                reduction_results = self._perform_tsne(
                    X_scaled, n_components, perplexity, random_state
                )
            elif method == "umap_like":
                reduction_results = self._perform_umap_like(
                    X_scaled, n_components, random_state
                )
            else:
                raise ProteomicsAnalysisError(
                    f"Unknown dimensionality reduction method: {method}"
                )

            # Store results in AnnData
            for key, value in reduction_results["embeddings"].items():
                adata_reduced.obsm[key] = value

            if "loadings" in reduction_results:
                for key, value in reduction_results["loadings"].items():
                    adata_reduced.varm[key] = value

            if "explained_variance" in reduction_results:
                adata_reduced.uns[f"{method}_explained_variance"] = reduction_results[
                    "explained_variance"
                ]

            # Calculate reduction statistics
            reduction_stats = {
                "method": method,
                "n_components": n_components,
                "input_dimensions": X_scaled.shape[1],
                "output_dimensions": reduction_results.get("output_dims", n_components),
                "variance_explained": reduction_results.get(
                    "total_variance_explained", None
                ),
                "perplexity": perplexity if method == "tsne" else None,
                "random_state": random_state,
                "samples_processed": adata_reduced.n_obs,
                "proteins_processed": adata_reduced.n_vars,
                "analysis_type": "dimensionality_reduction",
            }

            logger.info(
                f"Dimensionality reduction completed: {method} -> {reduction_results.get('output_dims', n_components)} dimensions"
            )

            # Create IR for provenance tracking
            ir = self._create_ir_dimensionality_reduction(
                method, n_components, perplexity, random_state
            )
            return adata_reduced, reduction_stats, ir

        except Exception as e:
            logger.exception(f"Error in dimensionality reduction: {e}")
            raise ProteomicsAnalysisError(f"Dimensionality reduction failed: {str(e)}")

    def perform_clustering_analysis(
        self,
        adata: anndata.AnnData,
        clustering_method: str = "kmeans",
        n_clusters: int = 4,
        use_pca: bool = True,
        n_pca_components: int = 50,
    ) -> Tuple[anndata.AnnData, Dict[str, Any], AnalysisStep]:
        """
        Perform clustering analysis on proteomics data.

        Args:
            adata: AnnData object with proteomics data
            clustering_method: Clustering method ('kmeans', 'hierarchical', 'gaussian_mixture')
            n_clusters: Number of clusters (for kmeans and gaussian mixture)
            use_pca: Whether to use PCA-reduced data for clustering
            n_pca_components: Number of PCA components if use_pca is True

        Returns:
            Tuple[anndata.AnnData, Dict[str, Any], AnalysisStep]: AnnData with cluster assignments,
                analysis stats, and IR for notebook export

        Raises:
            ProteomicsAnalysisError: If clustering fails
        """
        try:
            logger.info(
                f"Starting clustering analysis with method: {clustering_method}"
            )

            # Create working copy
            adata_clustered = adata.copy()
            original_shape = adata_clustered.shape
            logger.info(
                f"Input data shape: {original_shape[0]} samples × {original_shape[1]} proteins"
            )

            X = adata_clustered.X.copy()

            # Handle missing values
            from sklearn.impute import SimpleImputer

            imputer = SimpleImputer(strategy="mean")
            X_imputed = imputer.fit_transform(X)

            # Standardize data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_imputed)

            # Optionally reduce dimensions with PCA
            if use_pca:
                pca = PCA(n_components=min(n_pca_components, min(X_scaled.shape) - 1))
                X_for_clustering = pca.fit_transform(X_scaled)
                adata_clustered.obsm["X_pca_clustering"] = X_for_clustering
            else:
                X_for_clustering = X_scaled

            # Perform clustering
            if clustering_method == "kmeans":
                clustering_results = self._perform_kmeans_clustering(
                    X_for_clustering, n_clusters
                )
            elif clustering_method == "hierarchical":
                clustering_results = self._perform_hierarchical_clustering(
                    X_for_clustering, n_clusters
                )
            elif clustering_method == "gaussian_mixture":
                clustering_results = self._perform_gaussian_mixture_clustering(
                    X_for_clustering, n_clusters
                )
            else:
                raise ProteomicsAnalysisError(
                    f"Unknown clustering method: {clustering_method}"
                )

            # Store cluster assignments
            adata_clustered.obs["cluster"] = clustering_results["labels"].astype(str)
            adata_clustered.obs["cluster_numeric"] = clustering_results["labels"]

            # Store clustering metadata
            adata_clustered.uns["clustering"] = {
                "method": clustering_method,
                "n_clusters": clustering_results["n_clusters"],
                "use_pca": use_pca,
                "n_pca_components": n_pca_components if use_pca else None,
                **clustering_results.get("metadata", {}),
            }

            # Calculate clustering statistics
            cluster_sizes = (
                pd.Series(clustering_results["labels"]).value_counts().to_dict()
            )

            clustering_stats = {
                "clustering_method": clustering_method,
                "n_clusters": clustering_results["n_clusters"],
                "cluster_sizes": cluster_sizes,
                "use_pca": use_pca,
                "n_pca_components": n_pca_components if use_pca else None,
                "clustering_quality": clustering_results.get("quality_metric", None),
                "samples_processed": adata_clustered.n_obs,
                "proteins_processed": adata_clustered.n_vars,
                "analysis_type": "clustering_analysis",
            }

            logger.info(
                f"Clustering analysis completed: {clustering_results['n_clusters']} clusters identified"
            )

            # Create IR for provenance tracking
            ir = self._create_ir_clustering_analysis(
                clustering_method, n_clusters, use_pca, n_pca_components
            )
            return adata_clustered, clustering_stats, ir

        except Exception as e:
            logger.exception(f"Error in clustering analysis: {e}")
            raise ProteomicsAnalysisError(f"Clustering analysis failed: {str(e)}")

    def perform_pathway_enrichment(
        self,
        adata: anndata.AnnData,
        protein_list: Optional[List[str]] = None,
        database: str = "go_biological_process",
        background_proteins: Optional[List[str]] = None,
        p_value_threshold: float = 0.05,
    ) -> Tuple[anndata.AnnData, Dict[str, Any], AnalysisStep]:
        """
        Perform pathway enrichment analysis on a set of proteins using gseapy.

        Uses real GO, KEGG, and Reactome databases via the Enrichr API.
        Powered by PathwayEnrichmentService.

        Args:
            adata: AnnData object with proteomics data
            protein_list: List of proteins for enrichment (uses significant if None)
            database: Pathway database to use (go_biological_process, kegg_pathway, reactome, etc.)
            background_proteins: Background protein set (uses all proteins if None)
            p_value_threshold: P-value threshold for significance

        Returns:
            Tuple[anndata.AnnData, Dict[str, Any], AnalysisStep]: AnnData with enrichment results,
                analysis stats, and IR for notebook export

        Raises:
            ProteomicsAnalysisError: If enrichment fails
        """
        try:
            logger.info(
                f"Starting pathway enrichment analysis with database: {database}"
            )

            # Create working copy
            adata_enriched = adata.copy()
            original_shape = adata_enriched.shape
            logger.info(
                f"Input data shape: {original_shape[0]} samples × {original_shape[1]} proteins"
            )

            # Determine protein list for enrichment
            if protein_list is None:
                # Use significant proteins if available
                if "is_significant" in adata_enriched.var.columns:
                    protein_list = adata_enriched.var_names[
                        adata_enriched.var["is_significant"]
                    ].tolist()
                else:
                    # Use top variable proteins as fallback
                    if "intensity_cv" in adata_enriched.var.columns:
                        top_variable = adata_enriched.var.nlargest(100, "intensity_cv")
                        protein_list = top_variable.index.tolist()
                    else:
                        protein_list = adata_enriched.var_names[
                            :100
                        ].tolist()  # Fallback

            # Determine background proteins
            if background_proteins is None:
                background_proteins = adata_enriched.var_names.tolist()

            logger.info(
                f"Enrichment analysis: {len(protein_list)} query proteins, {len(background_proteins)} background"
            )

            # Delegate to PathwayEnrichmentService (real gseapy implementation)
            pathway_service = PathwayEnrichmentService()

            # Map legacy database name to Enrichr database (or use as-is)
            databases = [database]  # Can be expanded to multiple databases

            # Perform real enrichment analysis
            adata_enriched, enrichment_stats, ir = (
                pathway_service.over_representation_analysis(
                    adata=adata_enriched,
                    gene_list=protein_list,
                    databases=databases,
                    organism="human",  # Proteomics typically uses human protein IDs
                    background_genes=background_proteins,
                    p_value_threshold=p_value_threshold,
                    store_in_uns=True,  # Store results in adata.uns['pathway_enrichment']
                )
            )

            logger.info(
                f"Pathway enrichment completed: {enrichment_stats['n_significant_pathways']} significant pathways"
            )

            return adata_enriched, enrichment_stats, ir

        except PathwayEnrichmentError as e:
            logger.exception(f"Pathway enrichment error: {e}")
            raise ProteomicsAnalysisError(f"Pathway enrichment failed: {str(e)}")
        except Exception as e:
            logger.exception(f"Unexpected error in pathway enrichment: {e}")
            raise ProteomicsAnalysisError(f"Pathway enrichment failed: {str(e)}")

    # Helper methods for statistical testing
    def _handle_missing_values_for_testing(
        self, X: np.ndarray, method: str
    ) -> np.ndarray:
        """Handle missing values for statistical testing."""
        if method == "impute_mean":
            from sklearn.impute import SimpleImputer

            imputer = SimpleImputer(strategy="mean")
            return imputer.fit_transform(X)
        elif method == "impute_median":
            from sklearn.impute import SimpleImputer

            imputer = SimpleImputer(strategy="median")
            return imputer.fit_transform(X)
        else:
            return X

    def _perform_pairwise_testing(
        self,
        X: np.ndarray,
        groups: pd.Series,
        unique_groups: np.ndarray,
        test_method: str,
        comparison_type: str,
        min_observations: int,
    ) -> List[Dict[str, Any]]:
        """Perform pairwise statistical testing."""
        results = []

        if comparison_type == "all_pairs":
            # All pairwise comparisons
            for group1, group2 in combinations(unique_groups, 2):
                group1_data = X[groups == group1, :]
                group2_data = X[groups == group2, :]

                if (
                    group1_data.shape[0] < min_observations
                    or group2_data.shape[0] < min_observations
                ):
                    continue

                for i in range(X.shape[1]):
                    protein_name = f"protein_{i}"  # Would use actual protein names in real implementation

                    g1_values = group1_data[:, i]
                    g2_values = group2_data[:, i]

                    # Remove missing values
                    if hasattr(g1_values, "isnan"):
                        g1_clean = g1_values[~np.isnan(g1_values)]
                        g2_clean = g2_values[~np.isnan(g2_values)]
                    else:
                        g1_clean = g1_values
                        g2_clean = g2_values

                    if (
                        len(g1_clean) < min_observations
                        or len(g2_clean) < min_observations
                    ):
                        continue

                    # Perform statistical test
                    if test_method == "t_test":
                        statistic, p_value = stats.ttest_ind(g1_clean, g2_clean)
                    elif test_method == "mann_whitney":
                        statistic, p_value = stats.mannwhitneyu(
                            g1_clean, g2_clean, alternative="two-sided"
                        )
                    else:
                        continue

                    # Calculate effect size
                    effect_size = (np.mean(g1_clean) - np.mean(g2_clean)) / np.sqrt(
                        (
                            (len(g1_clean) - 1) * np.var(g1_clean)
                            + (len(g2_clean) - 1) * np.var(g2_clean)
                        )
                        / (len(g1_clean) + len(g2_clean) - 2)
                    )

                    results.append(
                        {
                            "protein": protein_name,
                            "protein_index": i,
                            "group1": group1,
                            "group2": group2,
                            "comparison": f"{group1}_vs_{group2}",
                            "statistic": float(statistic),
                            "p_value": float(p_value),
                            "effect_size": float(effect_size),
                            "n_group1": len(g1_clean),
                            "n_group2": len(g2_clean),
                            "mean_group1": float(np.mean(g1_clean)),
                            "mean_group2": float(np.mean(g2_clean)),
                        }
                    )

        # Apply multiple testing correction
        if results:
            from statsmodels.stats.multitest import fdrcorrection

            p_values = [r["p_value"] for r in results]
            _, p_adjusted = fdrcorrection(p_values)

            for i, result in enumerate(results):
                result["p_adjusted"] = float(p_adjusted[i])

        return results

    def _perform_multigroup_testing(
        self,
        X: np.ndarray,
        groups: pd.Series,
        unique_groups: np.ndarray,
        test_method: str,
        min_observations: int,
    ) -> List[Dict[str, Any]]:
        """Perform multi-group statistical testing (ANOVA/Kruskal)."""
        results = []

        for i in range(X.shape[1]):
            protein_name = f"protein_{i}"

            # Collect data for all groups
            group_data = []
            group_sizes = []

            for group in unique_groups:
                group_values = X[groups == group, i]

                # Remove missing values
                if hasattr(group_values, "isnan"):
                    clean_values = group_values[~np.isnan(group_values)]
                else:
                    clean_values = group_values

                if len(clean_values) >= min_observations:
                    group_data.append(clean_values)
                    group_sizes.append(len(clean_values))

            if len(group_data) < 2:
                continue

            # Perform statistical test
            if test_method == "anova":
                statistic, p_value = stats.f_oneway(*group_data)
            elif test_method == "kruskal":
                statistic, p_value = stats.kruskal(*group_data)
            else:
                continue

            results.append(
                {
                    "protein": protein_name,
                    "protein_index": i,
                    "test_type": "multi_group",
                    "statistic": float(statistic),
                    "p_value": float(p_value),
                    "n_groups": len(group_data),
                    "group_sizes": group_sizes,
                    "group_means": [float(np.mean(gd)) for gd in group_data],
                }
            )

        # Apply multiple testing correction
        if results:
            from statsmodels.stats.multitest import fdrcorrection

            p_values = [r["p_value"] for r in results]
            _, p_adjusted = fdrcorrection(p_values)

            for i, result in enumerate(results):
                result["p_adjusted"] = float(p_adjusted[i])

        return results

    # Helper methods for dimensionality reduction
    def _perform_pca(self, X: np.ndarray, n_components: int) -> Dict[str, Any]:
        """Perform PCA analysis."""
        pca = PCA(n_components=min(n_components, min(X.shape) - 1))
        X_pca = pca.fit_transform(X)

        return {
            "embeddings": {"X_pca": X_pca},
            "loadings": {"pca_loadings": pca.components_.T},
            "explained_variance": pca.explained_variance_ratio_,
            "total_variance_explained": pca.explained_variance_ratio_.sum(),
            "output_dims": X_pca.shape[1],
        }

    def _perform_tsne(
        self, X: np.ndarray, n_components: int, perplexity: float, random_state: int
    ) -> Dict[str, Any]:
        """Perform t-SNE analysis."""
        # Use PCA preprocessing for t-SNE if data is high-dimensional
        if X.shape[1] > 50:
            n_pca_components = min(50, min(X.shape) - 1)
            pca = PCA(n_components=n_pca_components)
            X_pca = pca.fit_transform(X)
        else:
            X_pca = X

        tsne = TSNE(
            n_components=n_components, perplexity=perplexity, random_state=random_state
        )
        X_tsne = tsne.fit_transform(X_pca)

        return {"embeddings": {"X_tsne": X_tsne}, "output_dims": X_tsne.shape[1]}

    def _perform_umap_like(
        self, X: np.ndarray, n_components: int, random_state: int
    ) -> Dict[str, Any]:
        """Perform UMAP-like analysis (simplified version using PCA + scaling).

        WARNING: This is an APPROXIMATE dimensionality reduction, NOT true UMAP.
        Uses PCA with variance-weighted scaling instead of UMAP's topological
        manifold learning algorithm. Results will differ from true UMAP in:
        - Local neighborhood preservation
        - Non-linear structure detection
        - Cluster separation

        For true UMAP, install umap-learn:
            pip install umap-learn
            import umap
            reducer = umap.UMAP(n_components=2)
            embedding = reducer.fit_transform(X)

        This approximation is suitable for quick exploration but NOT recommended
        for publication figures requiring proper UMAP embeddings.
        """
        # Simplified UMAP-like approach using PCA
        pca = PCA(n_components=min(n_components * 2, min(X.shape) - 1))
        X_pca = pca.fit_transform(X)

        # Apply additional non-linear scaling (simplified)
        X_umap_like = X_pca[:, :n_components] * np.sqrt(
            pca.explained_variance_ratio_[:n_components]
        )

        return {
            "embeddings": {"X_umap_like": X_umap_like},
            "output_dims": X_umap_like.shape[1],
        }

    # Helper methods for clustering
    def _perform_kmeans_clustering(
        self, X: np.ndarray, n_clusters: int
    ) -> Dict[str, Any]:
        """Perform K-means clustering."""
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)

        # Calculate silhouette score as quality metric
        try:
            from sklearn.metrics import silhouette_score

            quality_score = silhouette_score(X, labels)
        except Exception:
            quality_score = None

        return {
            "labels": labels,
            "n_clusters": n_clusters,
            "quality_metric": quality_score,
            "metadata": {
                "cluster_centers": kmeans.cluster_centers_,
                "inertia": kmeans.inertia_,
            },
        }

    def _perform_hierarchical_clustering(
        self, X: np.ndarray, n_clusters: int
    ) -> Dict[str, Any]:
        """Perform hierarchical clustering."""
        # Calculate distance matrix
        distances = pdist(X, metric="euclidean")
        linkage_matrix = linkage(distances, method="ward")

        # Get cluster labels
        labels = (
            fcluster(linkage_matrix, n_clusters, criterion="maxclust") - 1
        )  # Convert to 0-based

        return {
            "labels": labels,
            "n_clusters": n_clusters,
            "metadata": {"linkage_matrix": linkage_matrix, "distances": distances},
        }

    def _perform_gaussian_mixture_clustering(
        self, X: np.ndarray, n_clusters: int
    ) -> Dict[str, Any]:
        """Perform Gaussian Mixture Model clustering."""
        from sklearn.mixture import GaussianMixture

        gmm = GaussianMixture(n_components=n_clusters, random_state=42)
        labels = gmm.fit_predict(X)

        # Calculate AIC/BIC as quality metrics
        quality_score = -gmm.aic(X)  # Negative AIC (higher is better)

        return {
            "labels": labels,
            "n_clusters": n_clusters,
            "quality_metric": quality_score,
            "metadata": {
                "aic": gmm.aic(X),
                "bic": gmm.bic(X),
                "log_likelihood": gmm.score(X),
            },
        }
