"""Unit tests for PathwayEnrichmentService."""

from unittest.mock import MagicMock, Mock, patch

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from lobster.services.analysis.pathway_enrichment_service import (
    PathwayEnrichmentError,
    PathwayEnrichmentService,
)


class TestOverRepresentationAnalysis:
    """Test ORA functionality."""

    @pytest.fixture
    def service(self):
        return PathwayEnrichmentService()

    @pytest.fixture
    def mock_adata(self):
        return ad.AnnData(
            X=np.random.rand(100, 50),
            var=pd.DataFrame(index=[f"GENE{i}" for i in range(50)]),
        )

    @patch("lobster.services.analysis.pathway_enrichment_service.gp")
    def test_ora_returns_3_tuple(self, mock_gp, service, mock_adata):
        """Test that ORA returns (adata, stats, ir) tuple."""
        # Mock gseapy.enrichr response
        mock_enrichr_result = Mock()
        mock_enrichr_result.results = pd.DataFrame(
            {
                "Term": ["GO:0008150"],
                "Overlap": ["2/50"],
                "P-value": [0.01],
                "Adjusted P-value": [0.01],
                "Genes": ["TP53;BRCA1"],
                "Combined Score": [10.5],
            }
        )
        mock_gp.enrichr.return_value = mock_enrichr_result

        result = service.over_representation_analysis(
            adata=mock_adata,
            gene_list=["TP53", "BRCA1", "EGFR"],
            databases=["GO_Biological_Process_2023"],
        )

        assert len(result) == 3
        assert isinstance(result[0], ad.AnnData)
        assert isinstance(result[1], dict)
        assert "n_significant_pathways" in result[1]

    @patch("lobster.services.analysis.pathway_enrichment_service.gp")
    def test_gene_symbols_uppercased(self, mock_gp, service, mock_adata):
        """Test that gene symbols are normalized to uppercase."""
        mock_enrichr_result = Mock()
        mock_enrichr_result.results = pd.DataFrame()
        mock_gp.enrichr.return_value = mock_enrichr_result

        service.over_representation_analysis(
            adata=mock_adata,
            gene_list=["tp53", "brca1"],  # lowercase
            databases=["GO_Biological_Process_2023"],
        )

        # Verify gseapy was called with uppercase
        call_args = mock_gp.enrichr.call_args
        assert call_args[1]["gene_list"] == ["TP53", "BRCA1"]

    @patch("lobster.services.analysis.pathway_enrichment_service.gp")
    def test_results_stored_in_uns(self, mock_gp, service, mock_adata):
        """Test that results are stored in adata.uns."""
        mock_enrichr_result = Mock()
        mock_enrichr_result.results = pd.DataFrame(
            {
                "Term": ["Pathway1"],
                "Overlap": ["1/10"],
                "P-value": [0.01],
                "Adjusted P-value": [0.01],
                "Genes": ["TP53"],
                "Combined Score": [5.0],
            }
        )
        mock_gp.enrichr.return_value = mock_enrichr_result

        result_adata, _, _ = service.over_representation_analysis(
            adata=mock_adata,
            gene_list=["TP53"],
            databases=["KEGG_2021_Human"],
        )

        assert "pathway_enrichment" in result_adata.uns
        assert "results" in result_adata.uns["pathway_enrichment"]
        assert result_adata.uns["pathway_enrichment"]["method"] == "gseapy.enrichr"

    @patch("lobster.services.analysis.pathway_enrichment_service.gp")
    def test_default_databases(self, mock_gp, service, mock_adata):
        """Test that default databases are used when none specified."""
        mock_enrichr_result = Mock()
        mock_enrichr_result.results = pd.DataFrame()
        mock_gp.enrichr.return_value = mock_enrichr_result

        service.over_representation_analysis(
            adata=mock_adata,
            gene_list=["TP53", "BRCA1"],
            databases=None,  # Should use defaults
        )

        # Should query 3 databases (GO + KEGG + Reactome)
        assert mock_gp.enrichr.call_count == 3

    @patch("lobster.services.analysis.pathway_enrichment_service.gp")
    def test_rate_limiting_called(self, mock_gp, service, mock_adata):
        """Test that rate limiter is used."""
        mock_enrichr_result = Mock()
        mock_enrichr_result.results = pd.DataFrame()
        mock_gp.enrichr.return_value = mock_enrichr_result

        # Mock the rate limiter's wait method which is called by __enter__
        with patch.object(
            service._rate_limiter, "wait", return_value=True
        ) as mock_wait:
            service.over_representation_analysis(
                adata=mock_adata,
                gene_list=["TP53"],
                databases=["GO_Biological_Process_2023"],
            )

            # Rate limiter should be used once per database
            assert mock_wait.call_count == 1

    @patch("lobster.services.analysis.pathway_enrichment_service.gp")
    def test_empty_results_handled_gracefully(self, mock_gp, service, mock_adata):
        """Test that empty results don't crash."""
        mock_enrichr_result = Mock()
        mock_enrichr_result.results = None
        mock_gp.enrichr.return_value = mock_enrichr_result

        result_adata, stats, _ = service.over_representation_analysis(
            adata=mock_adata,
            gene_list=["TP53"],
            databases=["GO_Biological_Process_2023"],
        )

        assert stats["n_significant_pathways"] == 0
        assert stats["n_terms_total"] == 0


class TestDatabasePresets:
    """Test database preset mappings."""

    def test_presets_map_to_enrichr_names(self):
        presets = PathwayEnrichmentService.DATABASE_PRESETS

        assert presets["go_biological_process"] == "GO_Biological_Process_2023"
        assert presets["kegg_pathway"] == "KEGG_2021_Human"
        assert presets["kegg"] == "KEGG_2021_Human"
        assert presets["reactome"] == "Reactome_2022"

    def test_get_available_databases(self):
        """Test that get_available_databases returns descriptions."""
        databases = PathwayEnrichmentService.get_available_databases()

        assert isinstance(databases, dict)
        assert "go_biological_process" in databases
        assert "KEGG" in databases["kegg_pathway"]


class TestErrorHandling:
    """Test error handling scenarios."""

    def test_empty_gene_list_raises_error(self):
        service = PathwayEnrichmentService()
        adata = ad.AnnData(X=np.random.rand(10, 5))

        with pytest.raises(PathwayEnrichmentError):
            service.over_representation_analysis(
                adata=adata, gene_list=[], databases=["GO_Biological_Process_2023"]
            )

    @patch("lobster.services.analysis.pathway_enrichment_service.gp")
    def test_network_error_handled_gracefully(self, mock_gp):
        """Test that network errors are logged but service degrades gracefully."""
        mock_gp.enrichr.side_effect = ConnectionError("No internet")

        service = PathwayEnrichmentService()
        adata = ad.AnnData(X=np.random.rand(10, 5))

        # Service should degrade gracefully (not raise) when individual databases fail
        result_adata, stats, ir = service.over_representation_analysis(
            adata=adata,
            gene_list=["TP53"],
            databases=["GO_Biological_Process_2023"],
        )

        # Should return empty results but not crash
        assert result_adata is not None
        assert stats["n_significant_pathways"] == 0
        assert stats["n_terms_total"] == 0

    @patch("lobster.services.analysis.pathway_enrichment_service.gp", None)
    def test_missing_gseapy_dependency(self):
        """Test that missing gseapy raises helpful error."""
        service = PathwayEnrichmentService()
        adata = ad.AnnData(X=np.random.rand(10, 5))

        with pytest.raises(PathwayEnrichmentError) as exc_info:
            service.over_representation_analysis(
                adata=adata,
                gene_list=["TP53"],
                databases=["GO_Biological_Process_2023"],
            )

        assert "gseapy not installed" in str(exc_info.value)
        assert "pip install gseapy" in str(exc_info.value)


class TestOrganismMapping:
    """Test organism-specific database mapping."""

    @pytest.fixture
    def service(self):
        return PathwayEnrichmentService()

    def test_adjust_databases_for_human(self, service):
        """Test that human databases are not modified."""
        databases = ["KEGG_2021_Human", "GO_Biological_Process_2023"]
        adjusted = service._adjust_databases_for_organism(databases, "human")

        assert adjusted == databases

    def test_adjust_databases_for_mouse(self, service):
        """Test that mouse databases are mapped correctly."""
        databases = ["KEGG_2021_Human"]
        adjusted = service._adjust_databases_for_organism(databases, "mouse")

        assert "KEGG_2021_Mouse" in adjusted
        assert "KEGG_2021_Human" not in adjusted

    def test_adjust_databases_for_rat(self, service):
        """Test that rat databases are mapped correctly."""
        databases = ["KEGG_2021_Human"]
        adjusted = service._adjust_databases_for_organism(databases, "rat")

        assert "KEGG_2021_Rat" in adjusted


class TestGSEA:
    """Test Gene Set Enrichment Analysis functionality."""

    @pytest.fixture
    def service(self):
        return PathwayEnrichmentService()

    @pytest.fixture
    def mock_adata(self):
        return ad.AnnData(X=np.random.rand(100, 50))

    @pytest.fixture
    def ranked_genes(self):
        """Mock ranked gene list (gene, score)."""
        return pd.DataFrame(
            {
                "gene": [f"GENE{i}" for i in range(100)],
                "score": np.random.randn(100),
            }
        )

    @patch("lobster.services.analysis.pathway_enrichment_service.gp")
    def test_gsea_returns_3_tuple(self, mock_gp, service, mock_adata, ranked_genes):
        """Test that GSEA returns (adata, stats, ir) tuple."""
        # Mock gseapy.prerank response
        mock_gsea_result = Mock()
        mock_gsea_result.res2d = pd.DataFrame(
            {
                "Term": ["Pathway1"],
                "ES": [0.5],
                "NES": [1.8],
                "NOM p-val": [0.01],
                "FDR q-val": [0.05],
                "Lead_genes": ["TP53;BRCA1"],
            }
        )
        mock_gp.prerank.return_value = mock_gsea_result

        result = service.gene_set_enrichment_analysis(
            adata=mock_adata,
            ranked_genes=ranked_genes,
            databases=["GO_Biological_Process_2023"],
        )

        assert len(result) == 3
        assert isinstance(result[0], ad.AnnData)
        assert isinstance(result[1], dict)
        assert "n_significant_gene_sets" in result[1]

    def test_gsea_validates_ranked_genes_format(self, service, mock_adata):
        """Test that GSEA validates ranked_genes DataFrame format."""
        # Missing 'score' column
        bad_ranked = pd.DataFrame({"gene": ["TP53", "BRCA1"]})

        with pytest.raises(PathwayEnrichmentError) as exc_info:
            service.gene_set_enrichment_analysis(
                adata=mock_adata, ranked_genes=bad_ranked
            )

        assert "'gene' and 'score' columns" in str(exc_info.value)

    def test_gsea_empty_ranked_genes_raises_error(self, service, mock_adata):
        """Test that empty ranked genes raises error."""
        empty_ranked = pd.DataFrame(columns=["gene", "score"])

        with pytest.raises(PathwayEnrichmentError):
            service.gene_set_enrichment_analysis(
                adata=mock_adata, ranked_genes=empty_ranked
            )


class TestAnalysisStepIR:
    """Test that AnalysisStep IR is generated correctly."""

    @pytest.fixture
    def service(self):
        return PathwayEnrichmentService()

    @pytest.fixture
    def mock_adata(self):
        return ad.AnnData(X=np.random.rand(10, 5))

    @patch("lobster.services.analysis.pathway_enrichment_service.gp")
    def test_ora_ir_has_required_fields(self, mock_gp, service, mock_adata):
        """Test that ORA IR has all required fields for notebook export."""
        mock_enrichr_result = Mock()
        mock_enrichr_result.results = pd.DataFrame(
            {
                "Term": ["Pathway1"],
                "Adjusted P-value": [0.01],
            }
        )
        mock_gp.enrichr.return_value = mock_enrichr_result

        _, _, ir = service.over_representation_analysis(
            adata=mock_adata,
            gene_list=["TP53"],
            databases=["GO_Biological_Process_2023"],
        )

        assert ir.operation == "pathway.enrichment.ora"
        assert ir.tool_name == "PathwayEnrichmentService.over_representation_analysis"
        assert ir.library == "gseapy"
        assert "import gseapy as gp" in ir.code_template
        assert len(ir.parameters) > 0
        assert "gene_list" in ir.parameters

    @patch("lobster.services.analysis.pathway_enrichment_service.gp")
    def test_gsea_ir_has_required_fields(self, mock_gp, service, mock_adata):
        """Test that GSEA IR has all required fields."""
        mock_gsea_result = Mock()
        mock_gsea_result.res2d = pd.DataFrame(
            {"Term": ["Pathway1"], "FDR q-val": [0.05]}
        )
        mock_gp.prerank.return_value = mock_gsea_result

        ranked_genes = pd.DataFrame({"gene": ["TP53"], "score": [2.0]})

        _, _, ir = service.gene_set_enrichment_analysis(
            adata=mock_adata, ranked_genes=ranked_genes
        )

        assert ir.operation == "pathway.enrichment.gsea"
        assert ir.library == "gseapy"
        assert "gp.prerank" in ir.code_template


class TestIntegration:
    """Integration tests (mocked but realistic scenarios)."""

    @pytest.fixture
    def service(self):
        return PathwayEnrichmentService()

    @pytest.fixture
    def realistic_adata(self):
        """Create realistic proteomics AnnData."""
        n_samples = 10
        n_proteins = 100

        return ad.AnnData(
            X=np.random.rand(n_samples, n_proteins),
            var=pd.DataFrame(
                {
                    "protein_id": [f"P{i:05d}" for i in range(n_proteins)],
                    "is_significant": np.random.rand(n_proteins) < 0.1,
                },
                index=[f"PROT{i}" for i in range(n_proteins)],
            ),
        )

    @patch("lobster.services.analysis.pathway_enrichment_service.gp")
    def test_realistic_proteomics_workflow(self, mock_gp, service, realistic_adata):
        """Test realistic proteomics pathway enrichment workflow."""
        # Mock realistic response with multiple pathways
        mock_enrichr_result = Mock()
        mock_enrichr_result.results = pd.DataFrame(
            {
                "Term": [
                    "GO:0006412 translation",
                    "GO:0006457 protein folding",
                    "GO:0006508 proteolysis",
                ],
                "Overlap": ["3/50", "2/30", "4/80"],
                "P-value": [0.001, 0.005, 0.01],
                "Adjusted P-value": [0.005, 0.015, 0.03],
                "Genes": [
                    "PROT1;PROT2;PROT3",
                    "PROT4;PROT5",
                    "PROT6;PROT7;PROT8;PROT9",
                ],
                "Combined Score": [15.2, 12.1, 10.5],
            }
        )
        mock_gp.enrichr.return_value = mock_enrichr_result

        # Get significant proteins
        significant_proteins = realistic_adata.var_names[
            realistic_adata.var["is_significant"]
        ].tolist()

        # Run enrichment
        result_adata, stats, ir = service.over_representation_analysis(
            adata=realistic_adata,
            gene_list=significant_proteins,
            databases=["GO_Biological_Process_2023"],
            p_value_threshold=0.05,
        )

        # Verify results
        assert "pathway_enrichment" in result_adata.uns
        assert stats["n_significant_pathways"] == 3
        assert stats["n_genes_input"] == len(significant_proteins)
        assert len(stats["top_pathways"]) > 0

        # Verify IR can be used for notebook export
        assert "{{ gene_list | tojson }}" in ir.code_template
