"""
Pathway enrichment analysis service using gseapy.

Provides GO, KEGG, and Reactome enrichment via Enrichr API.
Follows Lobster's (adata, stats, ir) pattern.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from anndata import AnnData

from lobster.core.analysis_ir import AnalysisStep, ParameterSpec
from lobster.tools.rate_limiter import RateLimiter
from lobster.utils.logger import get_logger

logger = get_logger(__name__)

# Import gseapy at module level for testability
try:
    import gseapy as gp
except ImportError:
    gp = None  # Will be caught in methods with helpful error message


class PathwayEnrichmentError(Exception):
    """Raised when pathway enrichment fails."""

    pass


class PathwayEnrichmentService:
    """
    Centralized pathway enrichment using gseapy.

    This service provides Over-Representation Analysis (ORA) and Gene Set
    Enrichment Analysis (GSEA) using the gseapy library and Enrichr API.

    Supports 100+ databases including:
    - Gene Ontology (GO)
    - KEGG pathways
    - Reactome pathways
    - MSigDB collections
    - WikiPathways

    All methods follow Lobster's (adata, stats, ir) tuple pattern.
    """

    # Database presets for convenience (maps legacy names to Enrichr 2023 names)
    DATABASE_PRESETS = {
        "go_biological_process": "GO_Biological_Process_2023",
        "go_molecular_function": "GO_Molecular_Function_2023",
        "go_cellular_component": "GO_Cellular_Component_2023",
        "kegg_pathway": "KEGG_2021_Human",
        "kegg": "KEGG_2021_Human",
        "reactome": "Reactome_2022",
        "msigdb_hallmark": "MSigDB_Hallmark_2020",
        "wikipathways": "WikiPathway_2023_Human",
        "string_db": "PPI_Hub_Proteins",  # Closest equivalent for protein interactions
    }

    def __init__(self):
        """Initialize service with rate limiter for Enrichr API."""
        # Rate limit: 3 requests per second (Enrichr recommendation)
        self._rate_limiter = RateLimiter("enrichr", max_requests_per_second=3)

    def over_representation_analysis(
        self,
        adata: AnnData,
        gene_list: List[str],
        databases: Optional[List[str]] = None,
        organism: str = "human",
        background_genes: Optional[List[str]] = None,
        p_value_threshold: float = 0.05,
        store_in_uns: bool = True,
    ) -> Tuple[AnnData, Dict[str, Any], AnalysisStep]:
        """
        Perform Over-Representation Analysis (ORA) using Enrichr.

        ORA tests whether a gene set is enriched in specific pathways
        compared to a background set using Fisher's exact test.

        Args:
            adata: AnnData object to store results
            gene_list: Genes to test for enrichment (gene symbols)
            databases: List of Enrichr databases (uses GO+KEGG+Reactome if None)
            organism: Organism for pathway mapping ("human", "mouse", "rat")
            background_genes: Background gene set (uses all var_names if None)
            p_value_threshold: Significance threshold for adjusted p-value
            store_in_uns: Whether to store results in adata.uns

        Returns:
            Tuple of (adata_with_results, stats_dict, analysis_ir)

        Raises:
            PathwayEnrichmentError: If enrichment fails or inputs are invalid
        """
        try:
            # Validate inputs
            if not gene_list or len(gene_list) == 0:
                raise PathwayEnrichmentError("Empty gene list provided for enrichment")

            # Normalize gene symbols to uppercase (Enrichr convention)
            gene_list = [g.upper() for g in gene_list]

            # Default to GO + KEGG + Reactome if no databases specified
            if databases is None:
                databases = [
                    "GO_Biological_Process_2023",
                    "KEGG_2021_Human",
                    "Reactome_2022",
                ]
            else:
                # Map legacy database names to Enrichr names
                databases = [self.DATABASE_PRESETS.get(db, db) for db in databases]

            # Adjust for organism
            databases = self._adjust_databases_for_organism(databases, organism)

            logger.info(
                f"Running ORA on {len(gene_list)} genes using databases: {databases}"
            )

            # Check gseapy is available
            if gp is None:
                raise PathwayEnrichmentError(
                    "gseapy not installed. Install with: pip install gseapy\n"
                    "Note: gseapy >=1.1.0 requires Rust compiler. "
                    "Install Rust: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
                )

            # Run enrichment with rate limiting
            all_results = []
            for database in databases:
                try:
                    logger.info(f"Querying {database}...")
                    with self._rate_limiter:
                        enr_result = gp.enrichr(
                            gene_list=gene_list,
                            gene_sets=[database],
                            organism=organism.capitalize(),
                            background=background_genes,
                            cutoff=p_value_threshold,
                        )

                    if enr_result.results is not None and not enr_result.results.empty:
                        results_df = enr_result.results.copy()
                        results_df["database"] = database
                        all_results.append(results_df)
                        logger.info(f"Found {len(results_df)} enriched terms in {database}")
                except Exception as e:
                    logger.warning(f"Failed to query {database}: {e}")
                    continue

            # Combine results
            if not all_results:
                logger.warning("No significant enrichment found")
                enrichment_df = pd.DataFrame(
                    columns=[
                        "Term",
                        "Overlap",
                        "P-value",
                        "Adjusted P-value",
                        "Genes",
                        "database",
                    ]
                )
            else:
                enrichment_df = pd.concat(all_results, ignore_index=True)
                enrichment_df = enrichment_df.sort_values("Adjusted P-value")

            # Calculate statistics
            significant_terms = enrichment_df[
                enrichment_df["Adjusted P-value"] < p_value_threshold
            ]

            stats = {
                "n_genes_input": len(gene_list),
                "n_databases_queried": len(databases),
                "n_terms_total": len(enrichment_df),
                "n_significant_pathways": len(significant_terms),
                "enrichment_rate": (
                    len(significant_terms) / len(enrichment_df)
                    if len(enrichment_df) > 0
                    else 0.0
                ),
                "top_pathways": (
                    significant_terms["Term"].head(5).tolist()
                    if len(significant_terms) > 0
                    else []
                ),
                "organism": organism,
                "p_value_threshold": p_value_threshold,
                "method": "gseapy.enrichr (ORA)",
            }

            # Store in AnnData if requested
            if store_in_uns:
                adata.uns["pathway_enrichment"] = {
                    "method": "gseapy.enrichr",
                    "analysis_type": "over_representation_analysis",
                    "databases": databases,
                    "timestamp": str(datetime.now()),
                    "results": enrichment_df.to_dict("records"),
                    "parameters": {
                        "organism": organism,
                        "p_value_threshold": p_value_threshold,
                        "n_genes_input": len(gene_list),
                    },
                }

            # Create IR for provenance
            ir = self._create_ora_ir(
                gene_list=gene_list,
                databases=databases,
                organism=organism,
                p_value_threshold=p_value_threshold,
                background_genes=background_genes,
            )

            logger.info(
                f"ORA completed: {len(significant_terms)} significant pathways found"
            )

            return adata, stats, ir

        except Exception as e:
            if isinstance(e, PathwayEnrichmentError):
                raise
            else:
                logger.exception(f"Error in pathway enrichment: {e}")
                raise PathwayEnrichmentError(f"Pathway enrichment failed: {str(e)}")

    def gene_set_enrichment_analysis(
        self,
        adata: AnnData,
        ranked_genes: pd.DataFrame,
        databases: Optional[List[str]] = None,
        organism: str = "human",
        min_size: int = 15,
        max_size: int = 500,
        permutation_num: int = 1000,
    ) -> Tuple[AnnData, Dict[str, Any], AnalysisStep]:
        """
        Perform Gene Set Enrichment Analysis (GSEA) on ranked gene list.

        GSEA identifies pathways enriched at the top or bottom of a ranked
        gene list (e.g., sorted by log2 fold change from differential expression).

        Args:
            adata: AnnData object to store results
            ranked_genes: DataFrame with columns ['gene', 'score'] (e.g., log2FC)
            databases: List of Enrichr databases
            organism: Organism for pathway mapping
            min_size: Minimum gene set size
            max_size: Maximum gene set size
            permutation_num: Number of permutations for p-value calculation

        Returns:
            Tuple of (adata_with_results, stats_dict, analysis_ir)

        Raises:
            PathwayEnrichmentError: If GSEA fails
        """
        try:
            # Validate inputs
            if ranked_genes is None or len(ranked_genes) == 0:
                raise PathwayEnrichmentError("Empty ranked gene list")

            if "gene" not in ranked_genes.columns or "score" not in ranked_genes.columns:
                raise PathwayEnrichmentError(
                    "ranked_genes must have 'gene' and 'score' columns"
                )

            # Check gseapy is available
            if gp is None:
                raise PathwayEnrichmentError(
                    "gseapy not installed. Install with: pip install gseapy"
                )

            # Default databases
            if databases is None:
                databases = ["GO_Biological_Process_2023", "KEGG_2021_Human"]
            else:
                databases = [self.DATABASE_PRESETS.get(db, db) for db in databases]

            databases = self._adjust_databases_for_organism(databases, organism)

            logger.info(
                f"Running GSEA on {len(ranked_genes)} ranked genes using {databases}"
            )

            # Run GSEA with rate limiting
            all_results = []
            for database in databases:
                try:
                    with self._rate_limiter:
                        gsea_result = gp.prerank(
                            rnk=ranked_genes,
                            gene_sets=database,
                            min_size=min_size,
                            max_size=max_size,
                            permutation_num=permutation_num,
                            seed=42,
                        )

                    if gsea_result.res2d is not None and not gsea_result.res2d.empty:
                        results_df = gsea_result.res2d.copy()
                        results_df["database"] = database
                        all_results.append(results_df)
                        logger.info(f"GSEA found {len(results_df)} gene sets in {database}")
                except Exception as e:
                    logger.warning(f"GSEA failed for {database}: {e}")
                    continue

            # Combine results
            if not all_results:
                raise PathwayEnrichmentError("No GSEA results found")

            gsea_df = pd.concat(all_results, ignore_index=True)

            # Sort by NOM p-val if available, otherwise by FDR q-val
            if "NOM p-val" in gsea_df.columns:
                gsea_df = gsea_df.sort_values("NOM p-val")
            elif "FDR q-val" in gsea_df.columns:
                gsea_df = gsea_df.sort_values("FDR q-val")

            # Calculate statistics
            significant = gsea_df[gsea_df["FDR q-val"] < 0.25]  # Standard GSEA cutoff

            stats = {
                "n_genes_ranked": len(ranked_genes),
                "n_gene_sets_tested": len(gsea_df),
                "n_significant_gene_sets": len(significant),
                "method": "gseapy.prerank (GSEA)",
            }

            # Store in AnnData
            adata.uns["gsea_results"] = {
                "method": "gseapy.prerank",
                "timestamp": str(datetime.now()),
                "results": gsea_df.to_dict("records"),
            }

            # Create IR
            ir = self._create_gsea_ir(
                ranked_genes=ranked_genes,
                databases=databases,
                organism=organism,
                min_size=min_size,
                max_size=max_size,
            )

            logger.info(f"GSEA completed: {len(significant)} significant gene sets")

            return adata, stats, ir

        except Exception as e:
            if isinstance(e, PathwayEnrichmentError):
                raise
            logger.exception(f"GSEA failed: {e}")
            raise PathwayEnrichmentError(f"GSEA failed: {str(e)}")

    @staticmethod
    def get_available_databases() -> Dict[str, str]:
        """
        Return available database presets with descriptions.

        Returns:
            Dict mapping database keys to descriptions
        """
        return {
            "go_biological_process": "Gene Ontology Biological Process (2023)",
            "go_molecular_function": "Gene Ontology Molecular Function (2023)",
            "go_cellular_component": "Gene Ontology Cellular Component (2023)",
            "kegg_pathway": "KEGG Human Pathways (2021)",
            "reactome": "Reactome Pathways (2022)",
            "msigdb_hallmark": "MSigDB Hallmark Gene Sets (2020)",
            "wikipathways": "WikiPathways Human (2023)",
        }

    def _adjust_databases_for_organism(
        self, databases: List[str], organism: str
    ) -> List[str]:
        """Adjust database names for non-human organisms."""
        if organism.lower() == "human":
            return databases

        adjusted = []
        for db in databases:
            if "Human" in db:
                if organism.lower() == "mouse":
                    adjusted.append(db.replace("Human", "Mouse"))
                elif organism.lower() == "rat":
                    adjusted.append(db.replace("Human", "Rat"))
                else:
                    logger.warning(
                        f"No {organism} version for {db}, using human version"
                    )
                    adjusted.append(db)
            else:
                adjusted.append(db)

        return adjusted

    def _create_ora_ir(
        self,
        gene_list: List[str],
        databases: List[str],
        organism: str,
        p_value_threshold: float,
        background_genes: Optional[List[str]],
    ) -> AnalysisStep:
        """Create AnalysisStep IR for ORA."""
        return AnalysisStep(
            operation="pathway.enrichment.ora",
            tool_name="PathwayEnrichmentService.over_representation_analysis",
            description="Perform Over-Representation Analysis (ORA) using gseapy and Enrichr API",
            library="gseapy",
            code_template="""import gseapy as gp

# Run Over-Representation Analysis
enr = gp.enrichr(
    gene_list={{ gene_list | tojson }},
    gene_sets={{ databases | tojson }},
    organism={{ organism | tojson }},
    background={{ background_genes | tojson }},
    cutoff={{ p_value_threshold }}
)

# Extract results
results_df = enr.results
significant = results_df[results_df['Adjusted P-value'] < {{ p_value_threshold }}]
print(f"Found {len(significant)} significant pathways out of {len(results_df)} tested")

# Store in AnnData
adata.uns['pathway_enrichment'] = {
    'method': 'gseapy.enrichr',
    'results': results_df.to_dict('records')
}""",
            imports=["import gseapy as gp"],
            parameters={
                "gene_list": gene_list,
                "databases": databases,
                "organism": organism,
                "p_value_threshold": p_value_threshold,
                "background_genes": background_genes,
            },
            parameter_schema={
                "gene_list": ParameterSpec(
                    param_type="list[str]",
                    papermill_injectable=True,
                    default_value=None,
                    required=True,
                    description="List of gene symbols for enrichment analysis",
                ),
                "databases": ParameterSpec(
                    param_type="list[str]",
                    papermill_injectable=True,
                    default_value=None,
                    required=True,
                    description="List of Enrichr databases to query",
                ),
                "organism": ParameterSpec(
                    param_type="str",
                    papermill_injectable=True,
                    default_value="human",
                    required=False,
                    description="Organism name (human, mouse, rat)",
                ),
                "p_value_threshold": ParameterSpec(
                    param_type="float",
                    papermill_injectable=True,
                    default_value=0.05,
                    required=False,
                    validation_rule="0 < p_value_threshold <= 1",
                    description="Adjusted p-value significance threshold",
                ),
            },
            input_entities=["adata", "gene_list"],
            output_entities=["adata_with_enrichment"],
        )

    def _create_gsea_ir(
        self,
        ranked_genes: pd.DataFrame,
        databases: List[str],
        organism: str,
        min_size: int,
        max_size: int,
    ) -> AnalysisStep:
        """Create AnalysisStep IR for GSEA."""
        return AnalysisStep(
            operation="pathway.enrichment.gsea",
            tool_name="PathwayEnrichmentService.gene_set_enrichment_analysis",
            description="Perform Gene Set Enrichment Analysis (GSEA) using gseapy",
            library="gseapy",
            code_template="""import gseapy as gp
import pandas as pd

# Prepare ranked gene list (gene, score)
ranked_df = pd.DataFrame({
    'gene': {{ ranked_genes_genes | tojson }},
    'score': {{ ranked_genes_scores | tojson }}
})

# Run GSEA
gsea_result = gp.prerank(
    rnk=ranked_df,
    gene_sets={{ databases | tojson }},
    min_size={{ min_size }},
    max_size={{ max_size }},
    permutation_num=1000,
    seed=42
)

# Extract results
results_df = gsea_result.res2d
significant = results_df[results_df['FDR q-val'] < 0.25]
print(f"Found {len(significant)} significant gene sets (FDR < 0.25)")

# Store in AnnData
adata.uns['gsea_results'] = {
    'method': 'gseapy.prerank',
    'results': results_df.to_dict('records')
}""",
            imports=["import gseapy as gp", "import pandas as pd"],
            parameters={
                "ranked_genes_genes": ranked_genes["gene"].tolist(),
                "ranked_genes_scores": ranked_genes["score"].tolist(),
                "databases": databases,
                "organism": organism,
                "min_size": min_size,
                "max_size": max_size,
            },
            parameter_schema={
                "databases": ParameterSpec(
                    param_type="list[str]",
                    papermill_injectable=True,
                    default_value=None,
                    required=True,
                    description="Gene set databases to test",
                ),
                "min_size": ParameterSpec(
                    param_type="int",
                    papermill_injectable=True,
                    default_value=15,
                    required=False,
                    description="Minimum gene set size",
                ),
                "max_size": ParameterSpec(
                    param_type="int",
                    papermill_injectable=True,
                    default_value=500,
                    required=False,
                    description="Maximum gene set size",
                ),
            },
            input_entities=["adata", "ranked_genes"],
            output_entities=["adata_with_gsea"],
        )
