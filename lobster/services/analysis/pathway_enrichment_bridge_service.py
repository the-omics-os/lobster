"""
Pathway enrichment bridge service using INDRA Discovery API.

Bridges ML feature selection to biological pathway enrichment via INDRA's
hosted REST API - NO credentials, NO Neo4j, just HTTP POST requests.

Key Features:
- Auto-detects feature selection columns (*_selected)
- INDRA Discovery API for pathway enrichment (hosted, public)
- Dual storage: adata.uns + workspace CSV
- Gene overlap tracking (which selected genes are in each pathway)
- Follows Lobster's (adata, stats, ir) pattern
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
from anndata import AnnData

from lobster.core.analysis_ir import AnalysisStep, ParameterSpec
from lobster.core.workspace import resolve_workspace
from lobster.utils.logger import get_logger

logger = get_logger(__name__)

# INDRA API endpoints (PUBLIC - no auth required)
INDRA_DISCOVERY_API_URL = "https://discovery.indra.bio/api"
GILDA_GROUNDING_URL = "http://grounding.indra.bio/ground"


class PathwayEnrichmentError(Exception):
    """Raised when pathway enrichment fails."""

    pass


class PathwayEnrichmentBridgeService:
    """
    Bridge ML feature selection to pathway enrichment via INDRA Discovery API.

    This service connects feature selection results (stability_selected,
    lasso_selected, variance_selected) to biological pathway enrichment
    using INDRA's hosted REST API.

    **CRITICAL:** INDRA Discovery API accepts HGNC symbols directly - no
    grounding step required for enrichment. GILDA is optional for validation.

    Key Features:
    - Auto-detects which selection method was used (*_selected columns)
    - Uses INDRA Discovery API (hosted, public, no credentials)
    - Dual storage: uns summary + workspace CSV
    - Gene overlap column shows which selected genes appear in pathway
    - Returns 3-tuple (adata, stats, ir) following Lobster pattern

    API Endpoints:
    - POST https://discovery.indra.bio/api/discrete_analysis (enrichment)
    - POST http://grounding.indra.bio/ground (optional validation)
    """

    def __init__(self, timeout: int = 60):
        """
        Initialize service with configurable timeout.

        Args:
            timeout: HTTP timeout in seconds for INDRA API calls
        """
        self.timeout = timeout

    def enrich_selected_features(
        self,
        adata: AnnData,
        modality_name: str,
        selection_method: Optional[str] = None,
        sources: Optional[List[str]] = None,
        fdr_threshold: float = 0.05,
        organism: str = "human",
    ) -> Tuple[AnnData, Dict[str, Any], AnalysisStep]:
        """
        Perform pathway enrichment on selected features using INDRA Discovery API.

        Auto-detects which feature selection method was used by looking for
        *_selected columns in adata.var. Queries INDRA Discovery API for
        pathway enrichment (GO, Reactome, etc.) and stores results in both
        adata.uns and workspace CSV.

        Args:
            adata: AnnData with feature selection results in .var
            modality_name: Name of the modality (for provenance)
            selection_method: Which selection column to use (auto-detect if None)
                Options: "stability", "lasso", "variance"
            sources: INDRA sources to query (default: ["go", "reactome"])
            fdr_threshold: FDR significance threshold (default: 0.05)
            organism: Species (default: "human"; also supports "mouse")

        Returns:
            Tuple of (updated_adata, stats_dict, analysis_step_ir)

        Raises:
            PathwayEnrichmentError: If enrichment fails or inputs are invalid
            ValueError: If no selection columns found or selection method invalid

        Example:
            >>> service = PathwayEnrichmentBridgeService()
            >>> adata, stats, ir = service.enrich_selected_features(
            ...     adata,
            ...     modality_name="geo_gse12345_filtered",
            ...     sources=["go", "reactome"],
            ...     fdr_threshold=0.05
            ... )
            >>> print(f"Found {stats['n_pathways_significant']} significant pathways")
        """
        try:
            # 1. Detect selection method
            selection_column = self._detect_selection_method(adata, selection_method)
            logger.info(f"Using selection column: {selection_column}")

            # 2. Extract selected genes
            selected_mask = adata.var[selection_column].astype(bool)
            selected_genes = adata.var_names[selected_mask].tolist()

            if not selected_genes:
                raise PathwayEnrichmentError(
                    f"No genes selected in column '{selection_column}'. "
                    "Run feature selection first."
                )

            logger.info(f"Found {len(selected_genes)} selected features for enrichment")

            # 3. Call INDRA Discovery API for enrichment
            # Note: Discovery API accepts HGNC symbols directly - no grounding needed
            sources = sources or ["go", "reactome"]
            enrichment_results = self._call_discrete_analysis(
                selected_genes, sources, fdr_threshold
            )

            # 4. Create enrichment DataFrame with gene overlap
            enrichment_df = self._create_enrichment_df(
                enrichment_results, selected_genes
            )

            # 5. Calculate statistics
            significant_pathways = enrichment_df[enrichment_df["fdr"] < fdr_threshold]

            stats = {
                "modality_name": modality_name,
                "selection_method": selection_column,
                "n_genes_selected": len(selected_genes),
                "n_pathways_tested": len(enrichment_df),
                "n_pathways_significant": len(significant_pathways),
                "sources": sources,
                "fdr_threshold": fdr_threshold,
                "organism": organism,
                "api_source": "indra_discovery",
                "timestamp": datetime.now().isoformat(),
            }

            logger.info(
                f"Enrichment complete: {stats['n_pathways_significant']} "
                f"significant pathways (FDR < {fdr_threshold})"
            )

            # 6. Store results (dual storage: uns + CSV)
            adata = self._store_results(
                adata, enrichment_df, modality_name, selection_column, stats
            )

            # 7. Create IR for provenance
            ir = self._create_ir(
                modality_name,
                selection_column,
                selected_genes,
                sources,
                fdr_threshold,
            )

            return adata, stats, ir

        except Exception as e:
            if isinstance(e, (PathwayEnrichmentError, ValueError)):
                raise
            logger.exception(f"Pathway enrichment failed: {e}")
            raise PathwayEnrichmentError(f"Pathway enrichment failed: {str(e)}") from e

    def _detect_selection_method(
        self, adata: AnnData, selection_method: Optional[str]
    ) -> str:
        """
        Detect which selection column to use from *_selected columns.

        Args:
            adata: AnnData with feature selection results
            selection_method: Explicit method name (e.g., "stability", "lasso", "variance")

        Returns:
            Column name (e.g., "stability_selected")

        Raises:
            ValueError: If no selection columns found or invalid method specified
        """
        # Find all *_selected columns
        selected_columns = [c for c in adata.var.columns if c.endswith("_selected")]

        if not selected_columns:
            raise ValueError(
                "No feature selection found. Run stability_selection, "
                "lasso_selection, or variance_filter first."
            )

        # If explicit method provided, validate it
        if selection_method is not None:
            col = f"{selection_method}_selected"
            if col not in adata.var.columns:
                raise ValueError(
                    f"Selection method '{selection_method}' not found. "
                    f"Available methods: {[c.replace('_selected', '') for c in selected_columns]}"
                )
            return col

        # Auto-detect if only one method
        if len(selected_columns) == 1:
            logger.info(f"Auto-detected selection method: {selected_columns[0]}")
            return selected_columns[0]

        # Multiple methods - require explicit selection
        raise ValueError(
            f"Multiple selection methods found: {selected_columns}. "
            "Specify which to use via selection_method parameter "
            "(e.g., selection_method='stability')."
        )

    def _call_discrete_analysis(
        self,
        gene_list: List[str],
        sources: List[str],
        alpha: float,
    ) -> Dict[str, List[Dict]]:
        """
        Call INDRA Discovery API discrete_analysis endpoint.

        Args:
            gene_list: List of gene symbols (HGNC format)
            sources: INDRA sources (e.g., ["go", "reactome"])
            alpha: FDR significance threshold

        Returns:
            Dict mapping source name to list of enriched pathways

        Raises:
            PathwayEnrichmentError: If API call fails or times out
        """
        try:
            logger.info(f"Calling INDRA Discovery API with {len(gene_list)} genes...")

            # Check if INDRA path analysis sources requested
            indra_path_analysis = (
                "indra-upstream" in sources or "indra-downstream" in sources
            )

            response = requests.post(
                f"{INDRA_DISCOVERY_API_URL}/discrete_analysis",
                json={
                    "gene_list": gene_list,
                    "method": "fdr_bh",  # Benjamini-Hochberg FDR correction
                    "alpha": alpha,
                    "keep_insignificant": False,
                    "minimum_evidence_count": 1,
                    "minimum_belief": 0.0,
                    "indra_path_analysis": indra_path_analysis,
                },
                timeout=self.timeout,
            )
            response.raise_for_status()

            results = response.json()
            logger.info(f"INDRA API returned results for {len(results)} sources")

            return results

        except requests.Timeout:
            logger.error("INDRA Discovery API timeout")
            raise PathwayEnrichmentError(
                "INDRA API timeout - try again later or reduce gene list size"
            )
        except requests.RequestException as e:
            logger.error(f"INDRA Discovery API call failed: {e}")
            raise PathwayEnrichmentError(
                f"INDRA API unavailable: {e}. Check network connection."
            )

    def _create_enrichment_df(
        self,
        results: Dict[str, List[Dict]],
        selected_genes: List[str],
    ) -> pd.DataFrame:
        """
        Create DataFrame from INDRA enrichment results.

        Preserves gene_overlap column from INDRA response - this shows
        which selected genes appear in each pathway.

        Args:
            results: INDRA API response dict
            selected_genes: List of genes that were queried

        Returns:
            DataFrame with columns:
                - pathway_id (e.g., "GO:0006915")
                - pathway_name (e.g., "apoptotic process")
                - p_value
                - fdr (q_value from INDRA)
                - gene_overlap (comma-separated gene symbols)
                - overlap_count (number of genes in overlap)
                - pathway_size (total genes in pathway)
                - source (e.g., "go", "reactome")
        """
        rows = []
        for source, pathways in results.items():
            if not isinstance(pathways, list):
                logger.warning(f"Skipping non-list source: {source}")
                continue

            for pathway in pathways:
                # gene_overlap comes directly from INDRA
                gene_overlap = pathway.get("gene_overlap", [])
                if isinstance(gene_overlap, list):
                    gene_overlap_str = ",".join(gene_overlap)
                    overlap_count = len(gene_overlap)
                else:
                    gene_overlap_str = str(gene_overlap)
                    overlap_count = 0

                rows.append(
                    {
                        "pathway_id": pathway.get("curie", ""),
                        "pathway_name": pathway.get("name", ""),
                        "p_value": pathway.get("p_value", 1.0),
                        "fdr": pathway.get("q_value", 1.0),
                        "gene_overlap": gene_overlap_str,
                        "overlap_count": overlap_count,
                        "pathway_size": pathway.get("pathway_size", 0),
                        "source": source,
                    }
                )

        if not rows:
            logger.warning("No pathways returned from INDRA API")
            return pd.DataFrame(
                columns=[
                    "pathway_id",
                    "pathway_name",
                    "p_value",
                    "fdr",
                    "gene_overlap",
                    "overlap_count",
                    "pathway_size",
                    "source",
                ]
            )

        df = pd.DataFrame(rows)
        # Sort by FDR (most significant first)
        df = df.sort_values("fdr")
        return df

    def _store_results(
        self,
        adata: AnnData,
        enrichment_df: pd.DataFrame,
        modality_name: str,
        selection_column: str,
        stats: Dict,
    ) -> AnnData:
        """
        Store results in dual locations: adata.uns + workspace CSV.

        Args:
            adata: AnnData to modify
            enrichment_df: Enrichment results DataFrame
            modality_name: Name of the modality
            selection_column: Which selection column was used
            stats: Statistics dict

        Returns:
            Updated AnnData with results in uns["pathway_enrichment"]
        """
        # 1. Summary in uns (portable with data)
        top_pathways = enrichment_df.nsmallest(10, "fdr")
        adata.uns["pathway_enrichment"] = {
            "modality_name": modality_name,
            "selection_method": selection_column,
            "n_genes_selected": stats["n_genes_selected"],
            "n_pathways_significant": stats["n_pathways_significant"],
            "fdr_threshold": stats["fdr_threshold"],
            "sources": stats["sources"],
            "organism": stats["organism"],
            "api_source": "indra_discovery",
            "timestamp": stats["timestamp"],
            "top_pathways": top_pathways[
                ["pathway_name", "fdr", "gene_overlap", "source"]
            ].to_dict("records"),
        }

        # 2. Full CSV in workspace (queryable, detailed)
        workspace_path = resolve_workspace()
        csv_filename = f"{modality_name}_pathway_enrichment.csv"
        csv_path = workspace_path / csv_filename

        # Add provenance columns
        enrichment_df["source_modality"] = modality_name
        enrichment_df["selection_method"] = selection_column
        enrichment_df["timestamp"] = stats["timestamp"]

        # Write CSV
        enrichment_df.to_csv(csv_path, index=False)
        logger.info(f"Wrote enrichment results to {csv_path}")

        # Store CSV path in uns for reference
        adata.uns["pathway_enrichment"]["csv_file"] = str(csv_path)

        return adata

    def _create_ir(
        self,
        modality_name: str,
        selection_column: str,
        gene_list: List[str],
        sources: List[str],
        fdr_threshold: float,
    ) -> AnalysisStep:
        """
        Create AnalysisStep IR for provenance tracking and notebook export.

        Returns a complete, executable code template (50+ lines) that users
        can run to reproduce the pathway enrichment analysis.

        Args:
            modality_name: Name of the modality
            selection_column: Which selection column was used
            gene_list: List of selected genes
            sources: INDRA sources queried
            fdr_threshold: FDR threshold used

        Returns:
            AnalysisStep with executable Jinja2 template
        """
        code_template = '''"""Pathway Enrichment via INDRA Discovery API

This notebook performs pathway enrichment on feature selection results
using INDRA's hosted Discovery API.

**Key Points:**
- INDRA Discovery API: https://discovery.indra.bio/api/discrete_analysis
- No credentials required (public API)
- Accepts HGNC gene symbols directly
- Returns GO, Reactome, WikiPathways enrichment
"""

import requests
import pandas as pd

# =============================================================================
# Parameters
# =============================================================================
gene_list = {{ gene_list | tojson }}
sources = {{ sources | tojson }}
fdr_threshold = {{ fdr_threshold }}
modality_name = {{ modality_name | tojson }}
selection_method = {{ selection_column | tojson }}

print(f"Enriching {len(gene_list)} selected genes")
print(f"Sources: {sources}")
print(f"FDR threshold: {fdr_threshold}")

# =============================================================================
# Call INDRA Discovery API
# =============================================================================
print("\\nCalling INDRA Discovery API...")

response = requests.post(
    "https://discovery.indra.bio/api/discrete_analysis",
    json={
        "gene_list": gene_list,
        "method": "fdr_bh",           # Benjamini-Hochberg FDR correction
        "alpha": fdr_threshold,
        "keep_insignificant": False,  # Only return significant results
        "minimum_evidence_count": 1,
        "minimum_belief": 0.0,
    },
    timeout=60,
)
response.raise_for_status()
results = response.json()

print(f"API returned results for {len(results)} sources")

# =============================================================================
# Parse Results into DataFrame
# =============================================================================
rows = []
for source in sources:
    if source in results:
        for pathway in results[source]:
            # gene_overlap shows which selected genes are in this pathway
            gene_overlap = pathway.get("gene_overlap", [])
            rows.append({
                "pathway_id": pathway.get("curie", ""),
                "pathway_name": pathway.get("name", ""),
                "p_value": pathway.get("p_value", 1.0),
                "fdr": pathway.get("q_value", 1.0),
                "gene_overlap": ",".join(gene_overlap) if isinstance(gene_overlap, list) else str(gene_overlap),
                "overlap_count": len(gene_overlap) if isinstance(gene_overlap, list) else 0,
                "pathway_size": pathway.get("pathway_size", 0),
                "source": source,
            })

enrichment_df = pd.DataFrame(rows)
enrichment_df = enrichment_df.sort_values("fdr")

print(f"\\nTotal pathways: {len(enrichment_df)}")
significant = enrichment_df[enrichment_df["fdr"] < fdr_threshold]
print(f"Significant (FDR < {fdr_threshold}): {len(significant)}")

# =============================================================================
# Display Top Pathways
# =============================================================================
print("\\n=== Top 10 Enriched Pathways ===")
top_10 = enrichment_df.head(10)[
    ["pathway_name", "fdr", "gene_overlap", "source"]
]
display(top_10)

# =============================================================================
# Store Results
# =============================================================================
adata.uns["pathway_enrichment"] = {
    "modality_name": modality_name,
    "selection_method": selection_method,
    "n_genes_selected": len(gene_list),
    "n_pathways_significant": len(significant),
    "fdr_threshold": fdr_threshold,
    "sources": sources,
    "api_source": "indra_discovery",
    "top_pathways": top_10.to_dict("records"),
}

print(f"\\nResults stored in adata.uns['pathway_enrichment']")
'''

        return AnalysisStep(
            operation="pathway_enrichment.bridge",
            tool_name="PathwayEnrichmentBridgeService.enrich_selected_features",
            description=f"Pathway enrichment on {len(gene_list)} selected genes via INDRA Discovery API",
            library="indra_discovery_api",
            code_template=code_template,
            imports=["import requests", "import pandas as pd"],
            parameters={
                "modality_name": modality_name,
                "selection_column": selection_column,
                "gene_list": gene_list,
                "sources": sources,
                "fdr_threshold": fdr_threshold,
            },
            parameter_schema=[
                ParameterSpec(
                    param_type="List[str]",
                    papermill_injectable=True,
                    default_value=None,
                    required=True,
                    description="Selected gene symbols for enrichment",
                ),
                ParameterSpec(
                    param_type="List[str]",
                    papermill_injectable=True,
                    default_value=["go", "reactome"],
                    required=False,
                    description="INDRA sources to query (go, reactome, wikipathways, etc.)",
                ),
                ParameterSpec(
                    param_type="float",
                    papermill_injectable=True,
                    default_value=0.05,
                    required=False,
                    validation_rule="0 < fdr_threshold <= 1",
                    description="FDR significance threshold",
                ),
            ],
            input_entities=["adata", "selected_genes"],
            output_entities=["adata_with_enrichment"],
        )
