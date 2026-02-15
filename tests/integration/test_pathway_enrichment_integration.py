"""
Integration tests for PathwayEnrichmentService with real gseapy calls.

These tests make ACTUAL API calls to Enrichr (rate-limited, slow).
Mark with @pytest.mark.real_api to run separately.

Run with: pytest tests/integration/ -m real_api -v
"""

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from lobster.services.analysis.pathway_enrichment_service import (
    PathwayEnrichmentError,
    PathwayEnrichmentService,
)


@pytest.mark.real_api
class TestRealEnrichrAPI:
    """Test with real Enrichr API calls (requires internet)."""

    def test_ora_with_real_api(self):
        """Test ORA with real API call using well-known genes."""
        service = PathwayEnrichmentService()

        # Well-known cancer genes (should enrich for cancer pathways)
        cancer_genes = ["TP53", "BRCA1", "BRCA2", "EGFR", "MYC", "KRAS"]

        adata = ad.AnnData(X=np.random.rand(10, 100))

        result_adata, stats, ir = service.over_representation_analysis(
            adata=adata,
            gene_list=cancer_genes,
            databases=["KEGG_2021_Human"],  # Single database for faster test
            p_value_threshold=0.05,
        )

        # Verify results structure
        assert "pathway_enrichment" in result_adata.uns
        assert stats["n_genes_input"] == 6
        assert isinstance(stats["n_significant_pathways"], int)

        # Verify IR for notebook export
        assert ir.operation == "pathway.enrichment.ora"
        assert "gseapy" in ir.code_template
        assert "gp.enrichr" in ir.code_template

        # Verify enrichment metadata
        enrichment_data = result_adata.uns["pathway_enrichment"]
        assert enrichment_data["method"] == "gseapy.enrichr"
        assert "timestamp" in enrichment_data

        print(
            f"✅ Real API test passed: {stats['n_significant_pathways']} pathways found"
        )
        print(f"   Genes tested: {', '.join(cancer_genes)}")
        if stats["n_significant_pathways"] > 0:
            results_df = pd.DataFrame(enrichment_data["results"])
            print(f"   Top pathway: {results_df.iloc[0]['Term']}")

    def test_proteomics_service_integration(self):
        """Test proteomics service calls PathwayEnrichmentService."""
        from lobster.services.analysis.proteomics_analysis_service import (
            ProteomicsAnalysisService,
        )

        service = ProteomicsAnalysisService()

        # Create mock proteomics data
        significant_proteins = [
            "TP53",
            "EGFR",
            "MYC",
            "KRAS",
            "BCL2",
            "PIK3CA",
            "PTEN",
            "RB1",
        ]
        all_proteins = significant_proteins + [f"PROT{i}" for i in range(42)]

        adata = ad.AnnData(
            X=np.random.rand(10, 50),
            var=pd.DataFrame(
                {"is_significant": [True] * 8 + [False] * 42},
                index=all_proteins,
            ),
        )

        # Should delegate to PathwayEnrichmentService
        result_adata, stats, ir = service.perform_pathway_enrichment(
            adata=adata,
            database="kegg",  # Uses DATABASE_PRESETS mapping
            p_value_threshold=0.05,
        )

        # Verify PathwayEnrichmentService was used
        assert "pathway_enrichment" in result_adata.uns
        assert result_adata.uns["pathway_enrichment"]["method"] == "gseapy.enrichr"
        assert stats["n_genes_input"] == 8

        # Verify IR provenance
        assert ir.operation == "pathway.enrichment.ora"
        assert ir.library == "gseapy"

        print(f"✅ Proteomics integration test passed")
        print(f"   Significant proteins: {stats['n_genes_input']}")
        print(f"   Enriched pathways: {stats['n_significant_pathways']}")

    def test_singlecell_service_integration(self):
        """Test single-cell service calls PathwayEnrichmentService."""
        from lobster.services.analysis.enhanced_singlecell_service import (
            EnhancedSingleCellService,
        )

        service = EnhancedSingleCellService()

        # Create mock single-cell data with marker genes
        adata = ad.AnnData(X=np.random.rand(100, 50))

        # Well-known T-cell markers
        marker_genes = ["CD3D", "CD4", "CD8A", "IL2", "IFNG", "TNF", "GZMB", "PRF1"]

        result_adata, stats, ir = service.run_pathway_enrichment(
            adata=adata,
            marker_genes=marker_genes,
            databases=["GO_Biological_Process_2023"],
        )

        # Verify PathwayEnrichmentService was used
        assert "pathway_enrichment" in result_adata.uns
        assert result_adata.uns["pathway_enrichment"]["method"] == "gseapy.enrichr"

        # Verify stats
        assert stats["n_genes_input"] == 8
        assert "n_significant_pathways" in stats

        # Verify IR
        assert ir.operation == "pathway.enrichment.ora"
        assert "import gseapy as gp" in ir.code_template

        print(f"✅ Single-cell integration test passed")
        print(f"   Marker genes: {', '.join(marker_genes)}")
        print(f"   Enriched pathways: {stats['n_significant_pathways']}")


@pytest.mark.real_api
class TestDatabaseVariety:
    """Test different database types."""

    def test_go_biological_process(self):
        """Test GO Biological Process enrichment."""
        service = PathwayEnrichmentService()
        genes = ["TP53", "MDM2", "CDKN1A", "BAX", "BCL2"]  # Apoptosis genes

        adata = ad.AnnData(X=np.random.rand(10, 50))
        result_adata, stats, _ = service.over_representation_analysis(
            adata=adata,
            gene_list=genes,
            databases=["GO_Biological_Process_2023"],
        )

        assert "pathway_enrichment" in result_adata.uns
        print(f"✅ GO BP test: {stats['n_significant_pathways']} pathways found")

    def test_kegg_pathways(self):
        """Test KEGG pathway enrichment."""
        service = PathwayEnrichmentService()
        genes = ["EGFR", "KRAS", "PIK3CA", "AKT1", "MTOR"]  # PI3K-AKT pathway

        adata = ad.AnnData(X=np.random.rand(10, 50))
        result_adata, stats, _ = service.over_representation_analysis(
            adata=adata,
            gene_list=genes,
            databases=["KEGG_2021_Human"],
        )

        assert "pathway_enrichment" in result_adata.uns
        print(f"✅ KEGG test: {stats['n_significant_pathways']} pathways found")

    def test_reactome_pathways(self):
        """Test Reactome pathway enrichment."""
        service = PathwayEnrichmentService()
        genes = ["TP53", "ATM", "CHK2", "BRCA1", "BRCA2"]  # DNA damage response

        adata = ad.AnnData(X=np.random.rand(10, 50))
        result_adata, stats, _ = service.over_representation_analysis(
            adata=adata,
            gene_list=genes,
            databases=["Reactome_2022"],
        )

        assert "pathway_enrichment" in result_adata.uns
        print(f"✅ Reactome test: {stats['n_significant_pathways']} pathways found")


@pytest.mark.real_api
class TestEdgeCases:
    """Test edge cases with real API."""

    def test_no_enrichment_found(self):
        """Test behavior when no significant enrichment."""
        service = PathwayEnrichmentService()

        # Random genes unlikely to enrich
        genes = ["ZZZ1", "ZZZ2", "ZZZ3"]

        adata = ad.AnnData(X=np.random.rand(10, 50))
        result_adata, stats, _ = service.over_representation_analysis(
            adata=adata,
            gene_list=genes,
            databases=["GO_Biological_Process_2023"],
        )

        # Should handle gracefully (no exception)
        assert stats["n_significant_pathways"] == 0
        assert "pathway_enrichment" in result_adata.uns
        print("✅ No enrichment case handled gracefully")

    def test_single_gene_enrichment(self):
        """Test with single gene (edge case)."""
        service = PathwayEnrichmentService()

        adata = ad.AnnData(X=np.random.rand(10, 50))
        result_adata, stats, _ = service.over_representation_analysis(
            adata=adata,
            gene_list=["TP53"],
            databases=["KEGG_2021_Human"],
        )

        # Should work (Enrichr accepts single genes)
        assert stats["n_genes_input"] == 1
        print("✅ Single gene enrichment works")

    def test_large_gene_list(self):
        """Test with large gene list (100+ genes)."""
        service = PathwayEnrichmentService()

        # Generate 100 diverse genes
        genes = [f"GENE{i}" for i in range(1, 101)]

        adata = ad.AnnData(X=np.random.rand(10, 500))
        result_adata, stats, _ = service.over_representation_analysis(
            adata=adata,
            gene_list=genes,
            databases=["GO_Biological_Process_2023"],
        )

        assert stats["n_genes_input"] == 100
        print(f"✅ Large gene list handled: {stats['n_genes_input']} genes")


@pytest.mark.real_api
@pytest.mark.slow
class TestGSEA:
    """Test Gene Set Enrichment Analysis (slow)."""

    def test_gsea_with_real_data(self):
        """Test GSEA with ranked gene list."""
        service = PathwayEnrichmentService()

        # Create realistic ranked genes (e.g., from differential expression)
        genes = [f"GENE{i}" for i in range(1, 51)]
        scores = np.linspace(5, -5, 50)  # Fold changes from high to low

        ranked_genes = pd.DataFrame({"gene": genes, "score": scores})

        adata = ad.AnnData(X=np.random.rand(10, 50))
        result_adata, stats, ir = service.gene_set_enrichment_analysis(
            adata=adata,
            ranked_genes=ranked_genes,
            databases=["GO_Biological_Process_2023"],
        )

        # Verify GSEA results
        assert "gsea_results" in result_adata.uns
        assert stats["n_genes_ranked"] == 50
        assert "n_gene_sets_tested" in stats

        # Verify IR
        assert ir.operation == "pathway.enrichment.gsea"
        assert "gp.prerank" in ir.code_template

        print(f"✅ GSEA test passed: {stats['n_gene_sets_tested']} gene sets tested")


if __name__ == "__main__":
    print(
        "Run with: pytest tests/integration/test_pathway_enrichment_integration.py -m real_api -v -s"
    )
