"""
Proteomics STRING PPI network analysis service.

Queries the STRING REST API for protein-protein interaction networks
from differentially expressed proteins. Computes basic network topology
metrics using networkx if available.

All methods return 3-tuples (AnnData, Dict, AnalysisStep) for provenance tracking.
"""

from typing import Any, Dict, List, Optional, Tuple

import anndata
import numpy as np

from lobster.core.analysis_ir import AnalysisStep, ParameterSpec
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


class ProteomicsStringError(Exception):
    """Raised when STRING network analysis fails."""

    pass


# STRING API base URL
STRING_API_URL = "https://string-db.org/api/json/network"


class ProteomicsStringService:
    """
    STRING PPI network analysis service.

    Queries the STRING REST API for protein-protein interaction networks
    and computes basic topology metrics (degree distribution, hubs, density).
    Uses networkx for graph analysis when available, falls back to raw
    edge list with basic counts otherwise.
    """

    def __init__(self):
        """Initialize the STRING network service."""
        logger.debug("Initializing ProteomicsStringService")

    def _extract_proteins(
        self, adata: anndata.AnnData, max_proteins: int = 200
    ) -> List[str]:
        """
        Extract protein names from DE significant results.

        Args:
            adata: AnnData with DE results
            max_proteins: Maximum number of proteins to query

        Returns:
            List of protein name strings
        """
        de_data = adata.uns.get("differential_expression", {})
        significant_results = de_data.get("significant_results", [])

        if not significant_results:
            raise ProteomicsStringError(
                "No significant DE results found. Run find_differential_proteins first."
            )

        # Extract and clean protein names
        proteins = set()
        for result in significant_results:
            protein = result.get("protein", "")
            if not protein:
                continue
            # Clean: take first part before semicolon, strip whitespace
            cleaned = protein.split(";")[0].strip()
            if cleaned:
                proteins.add(cleaned)

        protein_list = sorted(proteins)[:max_proteins]
        logger.info(
            f"Extracted {len(protein_list)} proteins for STRING query "
            f"(from {len(significant_results)} significant results)"
        )
        return protein_list

    def _query_string_api(
        self,
        proteins: List[str],
        species: int,
        score_threshold: int,
        network_type: str,
    ) -> List[Dict[str, Any]]:
        """
        Query STRING REST API for PPI network.

        Args:
            proteins: List of protein identifiers
            species: NCBI taxonomy ID (9606 for human)
            score_threshold: Minimum combined score (0-1000)
            network_type: "functional" or "physical"

        Returns:
            List of edge dicts from STRING API

        Raises:
            ProteomicsStringError: On API errors
        """
        try:
            import requests
        except ImportError:
            raise ProteomicsStringError(
                "requests library required for STRING API. Install with: pip install requests"
            )

        params = {
            "identifiers": "\r".join(proteins),
            "species": species,
            "required_score": score_threshold,
            "network_type": network_type,
            "caller_identity": "lobster-ai",
        }

        try:
            logger.info(
                f"Querying STRING API: {len(proteins)} proteins, "
                f"species={species}, score>={score_threshold}"
            )
            response = requests.post(
                STRING_API_URL, data=params, timeout=30
            )

            if response.status_code == 429:
                raise ProteomicsStringError(
                    "STRING API rate limit exceeded. Please wait a few minutes and try again. "
                    "STRING allows ~1 request per second for batch queries."
                )

            if response.status_code == 400:
                raise ProteomicsStringError(
                    f"STRING API bad request: {response.text[:200]}. "
                    "Check protein identifiers are valid gene symbols."
                )

            response.raise_for_status()

            edges = response.json()
            logger.info(f"STRING API returned {len(edges)} interactions")
            return edges

        except ProteomicsStringError:
            raise
        except requests.exceptions.Timeout:
            raise ProteomicsStringError(
                "STRING API request timed out after 30 seconds. "
                "Try reducing the number of proteins or check your network connection."
            )
        except requests.exceptions.ConnectionError:
            raise ProteomicsStringError(
                "Could not connect to STRING API (https://string-db.org). "
                "Check your internet connection. STRING API may also be temporarily unavailable."
            )
        except Exception as e:
            raise ProteomicsStringError(f"STRING API query failed: {str(e)}")

    def _analyze_network(
        self, edges: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze network topology using networkx if available.

        Args:
            edges: List of edge dicts from STRING API

        Returns:
            Dict with network metrics
        """
        if not edges:
            return {
                "n_nodes": 0,
                "n_edges": 0,
                "density": 0.0,
                "hub_proteins": [],
                "edges": [],
            }

        # Parse edges into simplified format
        parsed_edges = []
        for edge in edges:
            parsed_edges.append(
                {
                    "protein1": edge.get("preferredName_A", edge.get("stringId_A", "")),
                    "protein2": edge.get("preferredName_B", edge.get("stringId_B", "")),
                    "score": edge.get("score", 0),
                    "nscore": edge.get("nscore", 0),
                    "escore": edge.get("escore", 0),
                    "tscore": edge.get("tscore", 0),
                }
            )

        # Try networkx for topology analysis
        try:
            import networkx as nx

            G = nx.Graph()
            for e in parsed_edges:
                G.add_edge(e["protein1"], e["protein2"], weight=e["score"])

            n_nodes = G.number_of_nodes()
            n_edges = G.number_of_edges()
            density = nx.density(G) if n_nodes > 1 else 0.0

            # Hub proteins by degree centrality
            degree_dict = dict(G.degree())
            sorted_by_degree = sorted(
                degree_dict.items(), key=lambda x: x[1], reverse=True
            )
            hub_proteins = [
                {"protein": name, "degree": degree}
                for name, degree in sorted_by_degree[:10]
            ]

            logger.info(
                f"Network analysis: {n_nodes} nodes, {n_edges} edges, "
                f"density={density:.4f}"
            )

            return {
                "n_nodes": n_nodes,
                "n_edges": n_edges,
                "density": float(density),
                "hub_proteins": hub_proteins,
                "edges": parsed_edges,
            }

        except ImportError:
            # Fallback: basic counts without networkx
            logger.info(
                "networkx not available, returning basic edge counts"
            )
            nodes = set()
            for e in parsed_edges:
                nodes.add(e["protein1"])
                nodes.add(e["protein2"])

            n_nodes = len(nodes)
            n_edges = len(parsed_edges)
            # Approximate density for undirected graph
            density = (
                (2 * n_edges) / (n_nodes * (n_nodes - 1))
                if n_nodes > 1
                else 0.0
            )

            return {
                "n_nodes": n_nodes,
                "n_edges": n_edges,
                "density": float(density),
                "hub_proteins": [],
                "edges": parsed_edges,
            }

    def query_network(
        self,
        adata: anndata.AnnData,
        proteins: Optional[List[str]] = None,
        species: int = 9606,
        score_threshold: int = 400,
        network_type: str = "functional",
    ) -> Tuple[anndata.AnnData, Dict[str, Any], AnalysisStep]:
        """
        Query STRING for PPI network and analyze topology.

        Args:
            adata: AnnData with DE results (used if proteins is None)
            proteins: Explicit protein list (if None, extracts from DE results)
            species: NCBI taxonomy ID (default: 9606 for human)
            score_threshold: Minimum combined score 0-1000 (default: 400 = medium confidence)
            network_type: "functional" (all evidence) or "physical" (binding only)

        Returns:
            Tuple[AnnData, Dict, AnalysisStep]: AnnData with network results, stats, IR

        Raises:
            ProteomicsStringError: If network query or analysis fails
        """
        try:
            # Get protein list
            if proteins is None:
                proteins = self._extract_proteins(adata, max_proteins=200)

            if not proteins:
                raise ProteomicsStringError("No proteins provided for STRING query")

            if len(proteins) > 200:
                logger.warning(
                    f"Truncating protein list from {len(proteins)} to 200 "
                    "(STRING API limit for batch queries)"
                )
                proteins = proteins[:200]

            # Query STRING API
            raw_edges = self._query_string_api(
                proteins, species, score_threshold, network_type
            )

            # Analyze network topology
            network_data = self._analyze_network(raw_edges)

            # Store results in AnnData
            adata.uns["string_network"] = {
                "edges": network_data["edges"],
                "hub_proteins": network_data["hub_proteins"],
                "n_nodes": network_data["n_nodes"],
                "n_edges": network_data["n_edges"],
                "density": network_data["density"],
                "parameters": {
                    "species": species,
                    "score_threshold": score_threshold,
                    "network_type": network_type,
                    "n_proteins_queried": len(proteins),
                },
            }

            # Build stats
            stats = {
                "n_proteins_queried": len(proteins),
                "n_interactions_found": network_data["n_edges"],
                "n_hub_proteins": len(network_data["hub_proteins"]),
                "network_density": network_data["density"],
                "n_nodes_in_network": network_data["n_nodes"],
                "species": species,
                "score_threshold": score_threshold,
                "network_type": network_type,
                "analysis_type": "string_network",
            }

            ir = self._create_ir_string_network(
                species, score_threshold, network_type, len(proteins)
            )

            logger.info(
                f"STRING network analysis complete: "
                f"{network_data['n_nodes']} nodes, {network_data['n_edges']} edges"
            )
            return adata, stats, ir

        except ProteomicsStringError:
            raise
        except Exception as e:
            logger.exception(f"STRING network analysis failed: {e}")
            raise ProteomicsStringError(f"STRING network analysis failed: {str(e)}")

    def _create_ir_string_network(
        self,
        species: int,
        score_threshold: int,
        network_type: str,
        n_proteins: int,
    ) -> AnalysisStep:
        """Create IR for STRING network analysis."""
        return AnalysisStep(
            operation="proteomics.analysis.string_network",
            tool_name="ProteomicsStringService.query_network",
            description="Protein-protein interaction network query from STRING database",
            library="requests",
            code_template="""import requests
import json

# Extract significant proteins from DE results
de_results = adata.uns['differential_expression']['significant_results']
proteins = list(set(r['protein'].split(';')[0].strip() for r in de_results))[:200]

# Query STRING REST API
params = {
    'identifiers': '\\r'.join(proteins),
    'species': {{ species }},
    'required_score': {{ score_threshold }},
    'network_type': {{ network_type | tojson }},
    'caller_identity': 'lobster-ai',
}
response = requests.post('https://string-db.org/api/json/network', data=params, timeout=30)
edges = response.json()

# Build network with networkx (optional)
try:
    import networkx as nx
    G = nx.Graph()
    for e in edges:
        G.add_edge(e['preferredName_A'], e['preferredName_B'], weight=e['score'])
    print(f"Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"Density: {nx.density(G):.4f}")
    # Top hub proteins
    degrees = sorted(dict(G.degree()).items(), key=lambda x: x[1], reverse=True)
    for name, deg in degrees[:10]:
        print(f"  {name}: degree={deg}")
except ImportError:
    print(f"Found {len(edges)} interactions (install networkx for topology analysis)")

# Store in AnnData
adata.uns['string_network'] = {
    'edges': [{'protein1': e['preferredName_A'], 'protein2': e['preferredName_B'],
               'score': e['score']} for e in edges],
    'n_edges': len(edges)
}""",
            imports=["import requests"],
            parameters={
                "species": species,
                "score_threshold": score_threshold,
                "network_type": network_type,
                "n_proteins": n_proteins,
            },
            parameter_schema={
                "species": ParameterSpec(
                    param_type="int",
                    papermill_injectable=True,
                    default_value=9606,
                    required=False,
                    description="NCBI taxonomy ID (9606=human, 10090=mouse)",
                ),
                "score_threshold": ParameterSpec(
                    param_type="int",
                    papermill_injectable=True,
                    default_value=400,
                    required=False,
                    validation_rule="0 <= score_threshold <= 1000",
                    description="Minimum combined score (400=medium, 700=high, 900=highest)",
                ),
                "network_type": ParameterSpec(
                    param_type="str",
                    papermill_injectable=True,
                    default_value="functional",
                    required=False,
                    validation_rule="network_type in ['functional', 'physical']",
                    description="Network evidence type: functional (all) or physical (binding)",
                ),
            },
            input_entities=["adata"],
            output_entities=["adata_network"],
        )
