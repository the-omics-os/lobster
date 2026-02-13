"""
Unit tests for PathwayEnrichmentBridgeService.

Tests INDRA Discovery API integration, selection detection,
dual storage, and error handling.

NOTE: Tests mock requests.post since we use hosted REST API, NOT Neo4j.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import json

import pandas as pd
import numpy as np
from anndata import AnnData

from lobster.services.analysis.pathway_enrichment_bridge_service import (
    PathwayEnrichmentBridgeService,
    PathwayEnrichmentError,
)


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def pathway_service():
    """Fixture for PathwayEnrichmentBridgeService."""
    return PathwayEnrichmentBridgeService(timeout=10)


@pytest.fixture
def adata_no_selection():
    """AnnData without any *_selected columns."""
    adata = AnnData(X=np.random.rand(100, 50))
    adata.var_names = [f"GENE{i}" for i in range(50)]
    return adata


@pytest.fixture
def adata_single_selection():
    """AnnData with stability_selected column and real gene names."""
    adata = AnnData(X=np.random.rand(100, 50))
    gene_names = ["TP53", "BRCA1", "EGFR", "KRAS", "MYC"] + [
        f"GENE{i}" for i in range(45)
    ]
    adata.var_names = gene_names
    adata.var["stability_selected"] = False
    # Select first 10 genes (includes the known ones)
    adata.var.loc[adata.var_names[:10], "stability_selected"] = True
    return adata


@pytest.fixture
def adata_multiple_selections():
    """AnnData with both stability_selected and lasso_selected."""
    adata = AnnData(X=np.random.rand(100, 50))
    adata.var_names = [f"GENE{i}" for i in range(50)]
    adata.var["stability_selected"] = False
    adata.var["lasso_selected"] = False
    adata.var.loc[adata.var_names[:10], "stability_selected"] = True
    adata.var.loc[adata.var_names[5:15], "lasso_selected"] = True
    return adata


@pytest.fixture
def mock_indra_response():
    """Mock INDRA Discovery API response."""
    return {
        "go": [
            {
                "curie": "GO:0006915",
                "name": "apoptotic process",
                "p_value": 0.0001,
                "q_value": 0.001,
                "gene_overlap": ["TP53", "BRCA1"],
                "pathway_size": 1500,
            },
            {
                "curie": "GO:0008283",
                "name": "cell population proliferation",
                "p_value": 0.001,
                "q_value": 0.005,
                "gene_overlap": ["EGFR", "KRAS", "MYC"],
                "pathway_size": 2000,
            },
        ],
        "reactome": [
            {
                "curie": "R-HSA-109582",
                "name": "Hemostasis",
                "p_value": 0.01,
                "q_value": 0.02,
                "gene_overlap": ["TP53"],
                "pathway_size": 500,
            },
        ],
    }


# ============================================================
# Selection Detection Tests
# ============================================================


class TestSelectionDetection:
    """Tests for _detect_selection_method."""

    def test_no_selection_columns_raises_error(
        self, pathway_service, adata_no_selection
    ):
        """Raise error when no *_selected columns exist."""
        with pytest.raises(ValueError, match="No feature selection found"):
            pathway_service._detect_selection_method(adata_no_selection, None)

    def test_single_column_auto_detected(self, pathway_service, adata_single_selection):
        """Auto-detect when single *_selected column exists."""
        result = pathway_service._detect_selection_method(adata_single_selection, None)
        assert result == "stability_selected"

    def test_multiple_columns_no_explicit_raises_error(
        self, pathway_service, adata_multiple_selections
    ):
        """Raise error when multiple *_selected columns and no selection_method."""
        with pytest.raises(ValueError, match="Multiple selection methods found"):
            pathway_service._detect_selection_method(adata_multiple_selections, None)

    def test_multiple_columns_with_explicit(
        self, pathway_service, adata_multiple_selections
    ):
        """Use explicit selection_method when multiple columns exist."""
        result = pathway_service._detect_selection_method(
            adata_multiple_selections, "stability"
        )
        assert result == "stability_selected"

    def test_invalid_explicit_raises_error(
        self, pathway_service, adata_single_selection
    ):
        """Raise error when explicit selection_method doesn't exist."""
        with pytest.raises(ValueError, match="not found"):
            pathway_service._detect_selection_method(adata_single_selection, "lasso")


# ============================================================
# INDRA Discovery API Tests
# ============================================================


class TestIndraDiscoveryAPI:
    """Tests for _call_discrete_analysis (mocking requests.post)."""

    def test_successful_api_call(self, pathway_service, mock_indra_response):
        """Test successful INDRA Discovery API call."""
        with patch(
            "lobster.services.analysis.pathway_enrichment_bridge_service.requests.post"
        ) as mock_post:
            mock_response = Mock()
            mock_response.json.return_value = mock_indra_response
            mock_response.raise_for_status = Mock()
            mock_post.return_value = mock_response

            result = pathway_service._call_discrete_analysis(
                gene_list=["TP53", "BRCA1", "EGFR"],
                sources=["go", "reactome"],
                alpha=0.05,
            )

            # Verify API was called correctly
            mock_post.assert_called_once()
            call_kwargs = mock_post.call_args
            assert "discovery.indra.bio" in call_kwargs[0][0]

            # Verify JSON payload
            json_payload = call_kwargs[1]["json"]
            assert json_payload["gene_list"] == ["TP53", "BRCA1", "EGFR"]
            assert json_payload["method"] == "fdr_bh"
            assert json_payload["alpha"] == 0.05

            # Verify response
            assert "go" in result
            assert "reactome" in result
            assert len(result["go"]) == 2

    def test_api_timeout_raises_error(self, pathway_service):
        """Raise PathwayEnrichmentError on timeout."""
        with patch(
            "lobster.services.analysis.pathway_enrichment_bridge_service.requests.post"
        ) as mock_post:
            import requests

            mock_post.side_effect = requests.Timeout("Connection timeout")

            with pytest.raises(PathwayEnrichmentError, match="timeout"):
                pathway_service._call_discrete_analysis(
                    gene_list=["TP53"],
                    sources=["go"],
                    alpha=0.05,
                )

    def test_api_error_raises_error(self, pathway_service):
        """Raise PathwayEnrichmentError on API error."""
        with patch(
            "lobster.services.analysis.pathway_enrichment_bridge_service.requests.post"
        ) as mock_post:
            import requests

            mock_post.side_effect = requests.RequestException("API Error")

            with pytest.raises(PathwayEnrichmentError, match="unavailable"):
                pathway_service._call_discrete_analysis(
                    gene_list=["TP53"],
                    sources=["go"],
                    alpha=0.05,
                )


# ============================================================
# DataFrame Creation Tests
# ============================================================


class TestEnrichmentDataFrame:
    """Tests for _create_enrichment_df."""

    def test_creates_dataframe_with_correct_columns(
        self, pathway_service, mock_indra_response
    ):
        """Create DataFrame with expected columns."""
        selected_genes = ["TP53", "BRCA1", "EGFR", "KRAS", "MYC"]
        df = pathway_service._create_enrichment_df(mock_indra_response, selected_genes)

        expected_columns = [
            "pathway_id",
            "pathway_name",
            "p_value",
            "fdr",
            "gene_overlap",
            "overlap_count",
            "pathway_size",
            "source",
        ]
        for col in expected_columns:
            assert col in df.columns

    def test_gene_overlap_as_comma_separated(
        self, pathway_service, mock_indra_response
    ):
        """Gene overlap stored as comma-separated string."""
        selected_genes = ["TP53", "BRCA1", "EGFR"]
        df = pathway_service._create_enrichment_df(mock_indra_response, selected_genes)

        # First GO pathway has TP53, BRCA1
        first_overlap = df[df["pathway_name"] == "apoptotic process"][
            "gene_overlap"
        ].iloc[0]
        assert "TP53" in first_overlap
        assert "BRCA1" in first_overlap
        assert "," in first_overlap

    def test_sorted_by_fdr(self, pathway_service, mock_indra_response):
        """DataFrame sorted by FDR ascending."""
        selected_genes = ["TP53", "BRCA1"]
        df = pathway_service._create_enrichment_df(mock_indra_response, selected_genes)

        if len(df) > 1:
            assert df["fdr"].iloc[0] <= df["fdr"].iloc[1]

    def test_empty_response_returns_empty_df(self, pathway_service):
        """Empty API response returns empty DataFrame."""
        selected_genes = ["TP53"]
        df = pathway_service._create_enrichment_df({}, selected_genes)
        assert len(df) == 0


# ============================================================
# Dual Storage Tests
# ============================================================


class TestDualStorage:
    """Tests for _store_results (uns + CSV)."""

    def test_stores_in_uns(
        self, pathway_service, adata_single_selection, mock_indra_response
    ):
        """Results stored in adata.uns['pathway_enrichment']."""
        from datetime import datetime

        selected_genes = ["TP53", "BRCA1", "EGFR"]
        enrichment_df = pathway_service._create_enrichment_df(
            mock_indra_response, selected_genes
        )
        stats = {
            "n_genes_selected": 10,
            "n_pathways_significant": 2,
            "fdr_threshold": 0.05,
            "sources": ["go", "reactome"],
            "organism": "human",
            "timestamp": datetime.now().isoformat(),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch(
                "lobster.services.analysis.pathway_enrichment_bridge_service.resolve_workspace"
            ) as mock_ws:
                mock_ws.return_value = Path(tmpdir)

                result = pathway_service._store_results(
                    adata_single_selection,
                    enrichment_df,
                    "test_modality",
                    "stability_selected",
                    stats,
                )

                assert "pathway_enrichment" in result.uns
                assert (
                    result.uns["pathway_enrichment"]["modality_name"] == "test_modality"
                )
                assert (
                    result.uns["pathway_enrichment"]["selection_method"]
                    == "stability_selected"
                )
                assert (
                    result.uns["pathway_enrichment"]["api_source"] == "indra_discovery"
                )
                assert "top_pathways" in result.uns["pathway_enrichment"]

    def test_writes_csv_to_workspace(
        self, pathway_service, adata_single_selection, mock_indra_response
    ):
        """Full results written to workspace CSV."""
        from datetime import datetime

        selected_genes = ["TP53", "BRCA1", "EGFR"]
        enrichment_df = pathway_service._create_enrichment_df(
            mock_indra_response, selected_genes
        )
        stats = {
            "n_genes_selected": 10,
            "n_pathways_significant": 2,
            "fdr_threshold": 0.05,
            "sources": ["go", "reactome"],
            "organism": "human",
            "timestamp": datetime.now().isoformat(),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch(
                "lobster.services.analysis.pathway_enrichment_bridge_service.resolve_workspace"
            ) as mock_ws:
                mock_ws.return_value = Path(tmpdir)

                result = pathway_service._store_results(
                    adata_single_selection,
                    enrichment_df,
                    "test_modality",
                    "stability_selected",
                    stats,
                )

                # Check CSV exists
                csv_path = Path(result.uns["pathway_enrichment"]["csv_file"])
                assert csv_path.exists()

                # Verify CSV content
                csv_df = pd.read_csv(csv_path)
                assert "source_modality" in csv_df.columns
                assert "selection_method" in csv_df.columns
                assert "timestamp" in csv_df.columns


# ============================================================
# Full Enrichment Flow Tests
# ============================================================


class TestFullEnrichmentFlow:
    """Tests for enrich_selected_features end-to-end."""

    def test_full_flow_success(
        self, pathway_service, adata_single_selection, mock_indra_response
    ):
        """Full enrichment flow with mocked API."""
        with patch(
            "lobster.services.analysis.pathway_enrichment_bridge_service.requests.post"
        ) as mock_post:
            mock_response = Mock()
            mock_response.json.return_value = mock_indra_response
            mock_response.raise_for_status = Mock()
            mock_post.return_value = mock_response

            with tempfile.TemporaryDirectory() as tmpdir:
                with patch(
                    "lobster.services.analysis.pathway_enrichment_bridge_service.resolve_workspace"
                ) as mock_ws:
                    mock_ws.return_value = Path(tmpdir)

                    result_adata, stats, ir = pathway_service.enrich_selected_features(
                        adata=adata_single_selection,
                        modality_name="test_modality",
                        sources=["go", "reactome"],
                    )

                    # Verify results
                    assert "pathway_enrichment" in result_adata.uns
                    assert stats["n_genes_selected"] == 10
                    assert stats["api_source"] == "indra_discovery"

                    # top_pathways is in adata.uns, not stats
                    assert "top_pathways" in result_adata.uns["pathway_enrichment"]

                    # Verify IR
                    assert ir.operation == "pathway_enrichment.bridge"
                    assert "discovery.indra.bio" in ir.code_template

    def test_empty_selection_raises_error(
        self, pathway_service, adata_single_selection
    ):
        """Raise error when no genes selected."""
        adata_single_selection.var["stability_selected"] = False  # All False

        with pytest.raises(PathwayEnrichmentError, match="No genes selected"):
            pathway_service.enrich_selected_features(
                adata=adata_single_selection,
                modality_name="test_modality",
            )

    def test_organism_parameter_accepted(
        self, pathway_service, adata_single_selection, mock_indra_response
    ):
        """Service accepts organism parameter without error."""
        with patch(
            "lobster.services.analysis.pathway_enrichment_bridge_service.requests.post"
        ) as mock_post:
            mock_response = Mock()
            mock_response.json.return_value = mock_indra_response
            mock_response.raise_for_status = Mock()
            mock_post.return_value = mock_response

            with tempfile.TemporaryDirectory() as tmpdir:
                with patch(
                    "lobster.services.analysis.pathway_enrichment_bridge_service.resolve_workspace"
                ) as mock_ws:
                    mock_ws.return_value = Path(tmpdir)

                    result_adata, stats, ir = pathway_service.enrich_selected_features(
                        adata=adata_single_selection,
                        modality_name="test_modality",
                        organism="zebrafish",
                    )

                    # Check organism is in stats
                    assert stats["organism"] == "zebrafish"
                    assert "pathway_enrichment" in result_adata.uns


# ============================================================
# IR Template Tests
# ============================================================


class TestIRTemplate:
    """Tests for _create_ir."""

    def test_ir_contains_api_url(self, pathway_service):
        """IR template contains INDRA Discovery API URL."""
        ir = pathway_service._create_ir(
            modality_name="test_mod",
            selection_column="stability_selected",
            gene_list=["TP53", "BRCA1"],
            sources=["go", "reactome"],
            fdr_threshold=0.05,
        )

        assert "discovery.indra.bio" in ir.code_template
        assert "discrete_analysis" in ir.code_template

    def test_ir_has_correct_operation(self, pathway_service):
        """IR operation is pathway_enrichment.bridge."""
        ir = pathway_service._create_ir(
            modality_name="test_mod",
            selection_column="stability_selected",
            gene_list=["TP53"],
            sources=["go"],
            fdr_threshold=0.05,
        )

        assert ir.operation == "pathway_enrichment.bridge"

    def test_ir_parameters_complete(self, pathway_service):
        """IR parameters include all required fields."""
        ir = pathway_service._create_ir(
            modality_name="test_mod",
            selection_column="stability_selected",
            gene_list=["TP53", "BRCA1"],
            sources=["go", "reactome"],
            fdr_threshold=0.05,
        )

        assert "gene_list" in ir.parameters
        assert "sources" in ir.parameters
        assert "fdr_threshold" in ir.parameters
        assert ir.parameters["gene_list"] == ["TP53", "BRCA1"]
