"""
Proteomics pathway enrichment service wrapping core PathwayEnrichmentService.

Provides a proteomics-specific interface for pathway enrichment analysis
using differentially expressed protein gene symbols extracted from DE results.
Maps database shorthand names to Enrichr database identifiers.

All methods return 3-tuples (AnnData, Dict, AnalysisStep) for provenance tracking.
"""

from typing import Any, Dict, List, Optional, Tuple

import anndata

from lobster.core.analysis_ir import AnalysisStep, ParameterSpec
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


class ProteomicsPathwayError(Exception):
    """Raised when proteomics pathway enrichment fails."""

    pass


# Database shorthand mapping to Enrichr database names
DATABASE_SHORTHAND = {
    "go": ["GO_Biological_Process_2023"],
    "reactome": ["Reactome_2022"],
    "kegg": ["KEGG_2021_Human"],
    "go_reactome": ["GO_Biological_Process_2023", "Reactome_2022"],
    "go_reactome_kegg": [
        "GO_Biological_Process_2023",
        "Reactome_2022",
        "KEGG_2021_Human",
    ],
}


class ProteomicsPathwayService:
    """
    Proteomics-specific pathway enrichment service.

    Wraps the core PathwayEnrichmentService to provide a convenient interface
    for running pathway enrichment on proteomics differential expression results.
    Extracts significant protein gene symbols from DE results and maps database
    shorthand names to Enrichr identifiers.
    """

    def __init__(self):
        """Initialize the proteomics pathway service."""
        logger.debug("Initializing ProteomicsPathwayService")

    def _extract_gene_symbols(
        self, adata: anndata.AnnData, max_genes: int = 500
    ) -> List[str]:
        """
        Extract gene symbols from DE significant results.

        Cleans protein names by stripping UniProt IDs and handling protein group format.

        Args:
            adata: AnnData with differential_expression results in uns
            max_genes: Maximum number of genes to extract

        Returns:
            List of cleaned gene symbol strings
        """
        de_data = adata.uns.get("differential_expression", {})
        significant_results = de_data.get("significant_results", [])

        if not significant_results:
            raise ProteomicsPathwayError(
                "No significant DE results found in adata.uns['differential_expression']['significant_results']. "
                "Run find_differential_proteins first."
            )

        gene_symbols = set()
        for result in significant_results:
            protein_name = result.get("protein", "")
            if not protein_name:
                continue

            # Clean protein name: strip UniProt IDs, handle protein group format
            # e.g., "EGFR;P00533" -> "EGFR", "EGFR_HUMAN" -> "EGFR"
            cleaned = protein_name.split(";")[0].strip()
            # Remove _HUMAN, _MOUSE suffixes (UniProt mnemonic)
            if "_" in cleaned and cleaned.split("_")[-1] in (
                "HUMAN",
                "MOUSE",
                "RAT",
                "BOVIN",
            ):
                cleaned = cleaned.rsplit("_", 1)[0]

            if cleaned:
                gene_symbols.add(cleaned)

        gene_list = sorted(gene_symbols)[:max_genes]
        logger.info(
            f"Extracted {len(gene_list)} gene symbols from {len(significant_results)} significant proteins"
        )
        return gene_list

    def _resolve_databases(self, databases: str) -> List[str]:
        """
        Resolve database shorthand to Enrichr database names.

        Args:
            databases: Shorthand string (e.g., "go_reactome", "kegg")

        Returns:
            List of Enrichr database name strings
        """
        if databases in DATABASE_SHORTHAND:
            return DATABASE_SHORTHAND[databases]

        # If not a shorthand, treat as a direct Enrichr database name
        return [databases]

    def run_enrichment(
        self,
        adata: anndata.AnnData,
        databases: str = "go_reactome",
        fdr_threshold: float = 0.05,
        max_genes: int = 500,
    ) -> Tuple[anndata.AnnData, Dict[str, Any], AnalysisStep]:
        """
        Run pathway enrichment on proteomics DE results.

        Extracts significant protein gene symbols from DE results and runs
        Over-Representation Analysis via the core PathwayEnrichmentService.

        Args:
            adata: AnnData with DE results in uns['differential_expression']
            databases: Database shorthand: "go", "reactome", "kegg",
                      "go_reactome", or "go_reactome_kegg"
            fdr_threshold: FDR threshold for enrichment significance
            max_genes: Maximum number of genes to include

        Returns:
            Tuple[AnnData, Dict, AnalysisStep]: Enriched AnnData, stats, IR

        Raises:
            ProteomicsPathwayError: If enrichment fails
        """
        try:
            # Extract gene symbols from DE results
            gene_list = self._extract_gene_symbols(adata, max_genes)

            if not gene_list:
                raise ProteomicsPathwayError(
                    "No valid gene symbols extracted from DE results"
                )

            # Resolve database shorthand
            enrichr_dbs = self._resolve_databases(databases)

            logger.info(
                f"Running pathway enrichment: {len(gene_list)} genes, "
                f"databases={enrichr_dbs}, FDR={fdr_threshold}"
            )

            # Import and call core PathwayEnrichmentService
            from lobster.services.analysis.pathway_enrichment_service import (
                PathwayEnrichmentService,
            )

            core_service = PathwayEnrichmentService()
            adata_enriched, ora_stats, ora_ir = (
                core_service.over_representation_analysis(
                    adata,
                    gene_list=gene_list,
                    databases=enrichr_dbs,
                    p_value_threshold=fdr_threshold,
                )
            )

            # Build proteomics-specific stats
            enrichment_data = adata_enriched.uns.get("pathway_enrichment", {})
            results = enrichment_data.get("results", [])

            # Extract top terms
            significant_terms = [
                r for r in results if r.get("Adjusted P-value", 1.0) < fdr_threshold
            ]
            top_terms = []
            for term in sorted(
                significant_terms, key=lambda x: x.get("Adjusted P-value", 1.0)
            )[:10]:
                top_terms.append(
                    {
                        "term": term.get("Term", "Unknown"),
                        "p_value": term.get("Adjusted P-value", 1.0),
                        "overlap": term.get("Overlap", ""),
                        "database": term.get("database", ""),
                    }
                )

            stats = {
                "n_genes_input": len(gene_list),
                "n_significant_terms": len(significant_terms),
                "n_total_terms": len(results),
                "top_terms": top_terms,
                "databases_queried": enrichr_dbs,
                "fdr_threshold": fdr_threshold,
                "analysis_type": "proteomics_pathway_enrichment",
            }

            # Create proteomics-specific IR
            ir = self._create_ir_pathway_enrichment(
                gene_list=gene_list,
                databases=databases,
                enrichr_dbs=enrichr_dbs,
                fdr_threshold=fdr_threshold,
                max_genes=max_genes,
            )

            logger.info(
                f"Pathway enrichment complete: {len(significant_terms)} significant terms"
            )
            return adata_enriched, stats, ir

        except ProteomicsPathwayError:
            raise
        except Exception as e:
            logger.exception(f"Pathway enrichment failed: {e}")
            raise ProteomicsPathwayError(f"Pathway enrichment failed: {str(e)}")

    def _create_ir_pathway_enrichment(
        self,
        gene_list: List[str],
        databases: str,
        enrichr_dbs: List[str],
        fdr_threshold: float,
        max_genes: int,
    ) -> AnalysisStep:
        """Create IR for proteomics pathway enrichment."""
        return AnalysisStep(
            operation="proteomics.analysis.pathway_enrichment",
            tool_name="ProteomicsPathwayService.run_enrichment",
            description="Pathway enrichment analysis on proteomics DE results using Enrichr API",
            library="gseapy",
            code_template="""import gseapy as gp

# Extract significant protein gene symbols from DE results
de_results = adata.uns['differential_expression']['significant_results']
gene_list = list(set(r['protein'].split(';')[0].strip() for r in de_results))[:{{ max_genes }}]

# Run Over-Representation Analysis via Enrichr
enr = gp.enrichr(
    gene_list=gene_list,
    gene_sets={{ enrichr_dbs | tojson }},
    organism='Human',
    cutoff={{ fdr_threshold }}
)

# Extract significant results
results_df = enr.results
significant = results_df[results_df['Adjusted P-value'] < {{ fdr_threshold }}]
print(f"Found {len(significant)} significant pathways from {len(gene_list)} proteins")

# Store in AnnData
adata.uns['pathway_enrichment'] = {
    'method': 'gseapy.enrichr',
    'results': results_df.to_dict('records')
}""",
            imports=["import gseapy as gp"],
            parameters={
                "databases": databases,
                "enrichr_dbs": enrichr_dbs,
                "fdr_threshold": fdr_threshold,
                "max_genes": max_genes,
                "gene_list_size": len(gene_list),
            },
            parameter_schema={
                "databases": ParameterSpec(
                    param_type="str",
                    papermill_injectable=True,
                    default_value="go_reactome",
                    required=False,
                    description="Database shorthand: go, reactome, kegg, go_reactome, go_reactome_kegg",
                ),
                "fdr_threshold": ParameterSpec(
                    param_type="float",
                    papermill_injectable=True,
                    default_value=0.05,
                    required=False,
                    validation_rule="0 < fdr_threshold <= 1",
                    description="FDR threshold for enrichment significance",
                ),
                "max_genes": ParameterSpec(
                    param_type="int",
                    papermill_injectable=True,
                    default_value=500,
                    required=False,
                    validation_rule="max_genes > 0",
                    description="Maximum number of genes for enrichment",
                ),
            },
            input_entities=["adata"],
            output_entities=["adata_enriched"],
        )
