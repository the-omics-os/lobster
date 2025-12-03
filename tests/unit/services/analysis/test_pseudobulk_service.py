"""
Unit tests for pseudobulk aggregation service.

Tests the PseudobulkService class functionality including single-cell
to pseudobulk aggregation, validation, provenance tracking, and export.
"""

from unittest.mock import MagicMock, Mock, patch

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from lobster.core import (
    AggregationError,
    InsufficientCellsError,
    ProvenanceError,
    PseudobulkError,
)
from lobster.core.adapters.pseudobulk_adapter import PseudobulkAdapter
from lobster.core.provenance import ProvenanceTracker
from lobster.services.analysis.pseudobulk_service import PseudobulkService


@pytest.fixture
def mock_single_cell_data():
    """Create mock single-cell data for pseudobulk aggregation."""
    np.random.seed(42)  # For reproducible tests

    # Create 1000 cells with 3 samples and 4 cell types
    n_cells = 1000
    n_genes = 200

    # Generate realistic cell assignments
    samples = np.random.choice(
        ["Sample_A", "Sample_B", "Sample_C"], n_cells, p=[0.4, 0.3, 0.3]
    )
    cell_types = np.random.choice(
        ["T_cell", "B_cell", "Monocyte", "NK_cell"], n_cells, p=[0.4, 0.3, 0.2, 0.1]
    )
    conditions = [
        "Control" if s in ["Sample_A", "Sample_B"] else "Treatment" for s in samples
    ]

    # Generate expression matrix
    expression_matrix = np.random.negative_binomial(5, 0.3, (n_cells, n_genes))
    gene_names = [f"GENE_{i:03d}" for i in range(n_genes)]

    adata = ad.AnnData(
        X=expression_matrix,
        obs=pd.DataFrame(
            {
                "sample_id": samples,
                "cell_type": cell_types,
                "condition": conditions,
                "batch": np.random.choice(["Batch1", "Batch2"], n_cells),
            }
        ),
        var=pd.DataFrame(
            {
                "gene_symbol": [f"Gene_{i}" for i in range(n_genes)],
                "feature_type": ["Gene Expression"] * n_genes,
            },
            index=gene_names,
        ),
    )

    return adata


@pytest.fixture
def mock_provenance_tracker():
    """Create mock provenance tracker."""
    return Mock(spec=ProvenanceTracker)


@pytest.fixture
def pseudobulk_service():
    """Create PseudobulkService for testing."""
    return PseudobulkService()


@pytest.fixture
def pseudobulk_service_with_provenance(mock_provenance_tracker):
    """Create PseudobulkService with mock provenance tracker."""
    return PseudobulkService(provenance_tracker=mock_provenance_tracker)


@pytest.mark.unit
class TestPseudobulkServiceInit:
    """Test PseudobulkService initialization."""

    def test_service_initialization_default(self):
        """Test service initialization with default settings."""
        service = PseudobulkService()

        assert service.provenance_tracker is not None
        assert isinstance(service.adapter, PseudobulkAdapter)
        assert len(service.aggregation_methods) == 3
        assert "sum" in service.aggregation_methods
        assert "mean" in service.aggregation_methods
        assert "median" in service.aggregation_methods

    def test_service_initialization_with_provenance(self, mock_provenance_tracker):
        """Test service initialization with custom provenance tracker."""
        service = PseudobulkService(provenance_tracker=mock_provenance_tracker)

        assert service.provenance_tracker is mock_provenance_tracker


@pytest.mark.unit
class TestPseudobulkServiceValidation:
    """Test PseudobulkService input validation."""

    def test_validate_aggregation_inputs_valid(
        self, pseudobulk_service, mock_single_cell_data
    ):
        """Test input validation with valid data."""
        # Should not raise any exceptions
        pseudobulk_service._validate_aggregation_inputs(
            mock_single_cell_data, "sample_id", "cell_type", "sum"
        )

    def test_validate_aggregation_inputs_missing_sample_col(
        self, pseudobulk_service, mock_single_cell_data
    ):
        """Test validation with missing sample column."""
        with pytest.raises(
            AggregationError, match="Sample column 'nonexistent' not found"
        ):
            pseudobulk_service._validate_aggregation_inputs(
                mock_single_cell_data, "nonexistent", "cell_type", "sum"
            )

    def test_validate_aggregation_inputs_missing_celltype_col(
        self, pseudobulk_service, mock_single_cell_data
    ):
        """Test validation with missing cell type column."""
        with pytest.raises(
            AggregationError, match="Cell type column 'nonexistent' not found"
        ):
            pseudobulk_service._validate_aggregation_inputs(
                mock_single_cell_data, "sample_id", "nonexistent", "sum"
            )

    def test_validate_aggregation_inputs_invalid_method(
        self, pseudobulk_service, mock_single_cell_data
    ):
        """Test validation with invalid aggregation method."""
        with pytest.raises(
            AggregationError, match="Unsupported aggregation method 'invalid'"
        ):
            pseudobulk_service._validate_aggregation_inputs(
                mock_single_cell_data, "sample_id", "cell_type", "invalid"
            )

    def test_validate_aggregation_inputs_missing_values(self, pseudobulk_service):
        """Test validation with missing values in grouping columns."""
        # Create data with missing values
        adata = ad.AnnData(
            X=np.random.randint(0, 100, (100, 50)),
            obs=pd.DataFrame(
                {
                    "sample_id": ["S1"] * 50 + [None] * 50,  # Missing values
                    "cell_type": ["T"] * 100,
                }
            ),
        )

        with pytest.raises(
            AggregationError, match="cells have missing sample identifiers"
        ):
            pseudobulk_service._validate_aggregation_inputs(
                adata, "sample_id", "cell_type", "sum"
            )


@pytest.mark.unit
class TestPseudobulkServiceAggregation:
    """Test PseudobulkService aggregation functionality."""

    def test_aggregate_to_pseudobulk_basic(
        self, pseudobulk_service, mock_single_cell_data
    ):
        """Test basic pseudobulk aggregation."""
        # Mock the adapter's from_source method to skip validation
        with patch.object(
            pseudobulk_service.adapter, "from_source"
        ) as mock_from_source:
            # Create a simple pseudobulk result
            pseudobulk_result = ad.AnnData(
                X=np.random.randint(
                    0, 1000, (12, 200)
                ),  # 12 pseudobulk samples, 200 genes
                obs=pd.DataFrame(
                    {
                        "sample_id": ["Sample_A", "Sample_B", "Sample_C"] * 4,
                        "cell_type": ["T_cell"] * 3
                        + ["B_cell"] * 3
                        + ["Monocyte"] * 3
                        + ["NK_cell"] * 3,
                    }
                ),
                var=mock_single_cell_data.var.copy(),
            )
            mock_from_source.return_value = pseudobulk_result

            result_adata, result_stats, result_ir = (
                pseudobulk_service.aggregate_to_pseudobulk(
                    mock_single_cell_data,
                    sample_col="sample_id",
                    celltype_col="cell_type",
                    min_cells=20,
                )
            )

            # Verify 3-tuple return
            assert isinstance(result_adata, ad.AnnData)
            assert isinstance(result_stats, dict)
            assert hasattr(result_ir, "operation")

            # Should have fewer observations than original (aggregated)
            assert result_adata.n_obs < mock_single_cell_data.n_obs
            # Should have same number of genes
            assert result_adata.n_vars == mock_single_cell_data.n_vars

    def test_aggregate_methods(self, pseudobulk_service):
        """Test different aggregation methods."""
        # Create small test data
        test_matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        # Test sum aggregation
        sum_result = pseudobulk_service._aggregate_sum(test_matrix)
        expected_sum = np.array([12, 15, 18])  # Sum of columns
        np.testing.assert_array_equal(sum_result, expected_sum)

        # Test mean aggregation
        mean_result = pseudobulk_service._aggregate_mean(test_matrix)
        expected_mean = np.array([4, 5, 6])  # Mean of columns
        np.testing.assert_array_equal(mean_result, expected_mean)

        # Test median aggregation
        median_result = pseudobulk_service._aggregate_median(test_matrix)
        expected_median = np.array([4, 5, 6])  # Median of columns
        np.testing.assert_array_equal(median_result, expected_median)

    def test_create_grouping_dataframe(self, pseudobulk_service, mock_single_cell_data):
        """Test grouping DataFrame creation."""
        grouping_df = pseudobulk_service._create_grouping_dataframe(
            mock_single_cell_data, "sample_id", "cell_type"
        )

        assert isinstance(grouping_df, pd.DataFrame)
        assert len(grouping_df) == mock_single_cell_data.n_obs
        assert "cell_idx" in grouping_df.columns
        assert "sample_id" in grouping_df.columns
        assert "cell_type" in grouping_df.columns
        assert "group_id" in grouping_df.columns

        # Check group_id format
        assert all("_" in group_id for group_id in grouping_df["group_id"])

    def test_filter_groups_by_cell_count(
        self, pseudobulk_service, mock_single_cell_data
    ):
        """Test filtering groups by minimum cell count."""
        grouping_df = pseudobulk_service._create_grouping_dataframe(
            mock_single_cell_data, "sample_id", "cell_type"
        )

        valid_groups, stats = pseudobulk_service._filter_groups_by_cell_count(
            grouping_df, min_cells=50
        )

        assert isinstance(valid_groups, pd.DataFrame)
        assert isinstance(stats, dict)

        # Check statistics
        assert "total_groups" in stats
        assert "valid_groups" in stats
        assert "excluded_groups" in stats
        assert stats["min_cells_threshold"] == 50

        # Valid groups should have sufficient cells
        group_counts = valid_groups.groupby("group_id").size()
        assert all(count >= 50 for count in group_counts)

    def test_insufficient_cells_error(self, pseudobulk_service, mock_single_cell_data):
        """Test error when no groups have sufficient cells."""
        with pytest.raises(
            InsufficientCellsError, match="No sample-celltype combinations have"
        ):
            pseudobulk_service.aggregate_to_pseudobulk(
                mock_single_cell_data,
                sample_col="sample_id",
                celltype_col="cell_type",
                min_cells=10000,  # Too high threshold
            )

    def test_gene_filtering(self, pseudobulk_service, mock_single_cell_data):
        """Test gene subset filtering."""
        # Test with specific gene subset
        gene_subset = mock_single_cell_data.var_names[:100].tolist()

        adata_filtered, X_filtered = pseudobulk_service._filter_genes(
            mock_single_cell_data, mock_single_cell_data.X, gene_subset
        )

        assert adata_filtered.n_vars == 100
        assert X_filtered.shape[1] == 100

    def test_gene_filtering_no_overlap(self, pseudobulk_service, mock_single_cell_data):
        """Test gene filtering with no overlapping genes."""
        non_existent_genes = ["FAKE_GENE_001", "FAKE_GENE_002"]

        with pytest.raises(AggregationError, match="No requested genes found"):
            pseudobulk_service._filter_genes(
                mock_single_cell_data, mock_single_cell_data.X, non_existent_genes
            )


@pytest.mark.unit
class TestPseudobulkServiceProvenance:
    """Test PseudobulkService provenance tracking."""

    def test_start_aggregation_activity(
        self, pseudobulk_service_with_provenance, mock_single_cell_data
    ):
        """Test provenance activity creation."""
        mock_provenance = pseudobulk_service_with_provenance.provenance_tracker
        mock_provenance.create_activity.return_value = "activity_123"

        activity_id = pseudobulk_service_with_provenance._start_aggregation_activity(
            mock_single_cell_data, "sample_id", "cell_type", "sum", 10
        )

        assert activity_id == "activity_123"
        mock_provenance.create_activity.assert_called_once()

        # Check activity parameters
        call_args = mock_provenance.create_activity.call_args
        assert call_args[1]["activity_type"] == "pseudobulk_aggregation"
        assert call_args[1]["agent"] == "PseudobulkService"
        assert "sample_col" in call_args[1]["parameters"]

    def test_complete_aggregation_activity(self, pseudobulk_service_with_provenance):
        """Test provenance activity completion."""
        # Setup mock activities list
        mock_provenance = pseudobulk_service_with_provenance.provenance_tracker
        mock_provenance.activities = [{"id": "activity_123"}]

        # Create mock result data
        result_adata = ad.AnnData(X=np.random.randn(6, 100))
        result_adata.uns["aggregation_stats"] = {"total_cells_aggregated": 500}

        pseudobulk_service_with_provenance._complete_aggregation_activity(
            "activity_123", "input_entity", "output_entity", result_adata
        )

        # Check activity was updated
        activity = mock_provenance.activities[0]
        assert "inputs" in activity
        assert "outputs" in activity
        assert "result_summary" in activity

    def test_provenance_error_handling(
        self, pseudobulk_service_with_provenance, mock_single_cell_data
    ):
        """Test provenance error handling."""
        mock_provenance = pseudobulk_service_with_provenance.provenance_tracker
        mock_provenance.create_activity.side_effect = Exception("Provenance error")

        with pytest.raises(
            ProvenanceError, match="Failed to create aggregation activity"
        ):
            pseudobulk_service_with_provenance._start_aggregation_activity(
                mock_single_cell_data, "sample_id", "cell_type", "sum", 10
            )


@pytest.mark.unit
class TestPseudobulkServiceAggregationMethods:
    """Test different aggregation methods in detail."""

    def test_perform_aggregation_sum(self, pseudobulk_service, mock_single_cell_data):
        """Test aggregation with sum method."""
        # Create grouping
        grouping_df = pseudobulk_service._create_grouping_dataframe(
            mock_single_cell_data, "sample_id", "cell_type"
        )

        # Filter to reasonable groups
        valid_groups, _ = pseudobulk_service._filter_groups_by_cell_count(
            grouping_df, min_cells=20
        )

        aggregated_matrix, group_metadata = pseudobulk_service._perform_aggregation(
            mock_single_cell_data.X, mock_single_cell_data.obs, valid_groups, "sum"
        )

        assert isinstance(aggregated_matrix, np.ndarray)
        assert isinstance(group_metadata, pd.DataFrame)
        assert aggregated_matrix.shape[0] == len(group_metadata)
        assert aggregated_matrix.shape[1] == mock_single_cell_data.n_vars

        # Check metadata structure
        assert "sample_id" in group_metadata.columns
        assert "cell_type" in group_metadata.columns
        assert "n_cells_aggregated" in group_metadata.columns
        assert "pseudobulk_sample_id" in group_metadata.columns

    def test_perform_aggregation_sparse_matrix(
        self, pseudobulk_service, mock_single_cell_data
    ):
        """Test aggregation with sparse matrix input."""
        # Convert to sparse matrix
        sparse_X = sparse.csr_matrix(mock_single_cell_data.X)
        mock_single_cell_data.X = sparse_X

        grouping_df = pseudobulk_service._create_grouping_dataframe(
            mock_single_cell_data, "sample_id", "cell_type"
        )
        valid_groups, _ = pseudobulk_service._filter_groups_by_cell_count(
            grouping_df, min_cells=20
        )

        # Should handle sparse matrices correctly
        aggregated_matrix, group_metadata = pseudobulk_service._perform_aggregation(
            sparse_X, mock_single_cell_data.obs, valid_groups, "sum"
        )

        assert isinstance(aggregated_matrix, np.ndarray)
        assert aggregated_matrix.shape[0] == len(group_metadata)

    def test_aggregation_with_different_methods(self, pseudobulk_service):
        """Test that different aggregation methods produce different results."""
        # Create simple test data
        test_X = np.array([[10, 20], [30, 40], [50, 60]])  # 3 cells, 2 genes
        test_obs = pd.DataFrame(
            {"sample_id": ["S1", "S1", "S1"], "cell_type": ["T", "T", "T"]}
        )

        grouping_df = pd.DataFrame(
            {
                "cell_idx": [0, 1, 2],
                "sample_id": ["S1", "S1", "S1"],
                "cell_type": ["T", "T", "T"],
                "group_id": ["S1_T", "S1_T", "S1_T"],
            }
        )

        # Test different methods
        sum_result, _ = pseudobulk_service._perform_aggregation(
            test_X, test_obs, grouping_df, "sum"
        )
        mean_result, _ = pseudobulk_service._perform_aggregation(
            test_X, test_obs, grouping_df, "mean"
        )
        median_result, _ = pseudobulk_service._perform_aggregation(
            test_X, test_obs, grouping_df, "median"
        )

        # Results should be different
        np.testing.assert_array_equal(
            sum_result[0], [90, 120]
        )  # Sum: [10+30+50, 20+40+60]
        np.testing.assert_array_equal(mean_result[0], [30, 40])  # Mean: [90/3, 120/3]
        np.testing.assert_array_equal(
            median_result[0], [30, 40]
        )  # Median: middle values


@pytest.mark.unit
class TestPseudobulkServiceFiltering:
    """Test PseudobulkService filtering functionality."""

    def test_filter_zero_genes(self, pseudobulk_service):
        """Test filtering of genes with all zeros."""
        # Create data with some zero genes
        expression_matrix = np.random.randint(0, 100, (6, 100))
        expression_matrix[:, [10, 20, 30]] = 0  # Make some genes all zeros

        adata = ad.AnnData(X=expression_matrix)

        filtered_adata = pseudobulk_service._filter_zero_genes(adata)

        assert filtered_adata.n_vars == 97  # Should remove 3 zero genes

        # Check that remaining genes have non-zero expression
        gene_sums = np.array(filtered_adata.X.sum(axis=0)).flatten()
        assert all(gene_sum > 0 for gene_sum in gene_sums)

    def test_filter_low_gene_samples(self, pseudobulk_service):
        """Test filtering of samples with too few genes."""
        # Create data where some samples have very few genes expressed
        expression_matrix = np.random.randint(0, 100, (6, 100))
        expression_matrix[0, 10:] = 0  # Sample 0 has only 10 genes
        expression_matrix[1, 50:] = 0  # Sample 1 has only 50 genes

        adata = ad.AnnData(X=expression_matrix)

        filtered_adata = pseudobulk_service._filter_low_gene_samples(
            adata, min_genes=30
        )

        # Should filter out the sample with only 10 genes
        assert filtered_adata.n_obs == 5  # Remove 1 sample

    def test_get_expression_matrix_default(
        self, pseudobulk_service, mock_single_cell_data
    ):
        """Test getting expression matrix from default layer."""
        X = pseudobulk_service._get_expression_matrix(mock_single_cell_data, layer=None)

        np.testing.assert_array_equal(X, mock_single_cell_data.X)

    def test_get_expression_matrix_layer(
        self, pseudobulk_service, mock_single_cell_data
    ):
        """Test getting expression matrix from specific layer."""
        # Add a layer
        mock_single_cell_data.layers["raw"] = mock_single_cell_data.X.copy()

        X = pseudobulk_service._get_expression_matrix(
            mock_single_cell_data, layer="raw"
        )

        np.testing.assert_array_equal(X, mock_single_cell_data.layers["raw"])

    def test_get_expression_matrix_missing_layer(
        self, pseudobulk_service, mock_single_cell_data
    ):
        """Test error when requesting missing layer."""
        with pytest.raises(AggregationError, match="Layer 'nonexistent' not found"):
            pseudobulk_service._get_expression_matrix(
                mock_single_cell_data, layer="nonexistent"
            )


@pytest.mark.unit
class TestPseudobulkServiceMetadata:
    """Test PseudobulkService metadata handling."""

    def test_add_aggregation_metadata(self, pseudobulk_service, mock_single_cell_data):
        """Test addition of aggregation metadata."""
        # Create minimal pseudobulk data
        pseudobulk_adata = ad.AnnData(X=np.random.randn(6, 100))
        pseudobulk_adata.obs["n_cells_aggregated"] = [50, 60, 70, 80, 90, 100]
        pseudobulk_adata.obs["sample_id"] = ["S1", "S1", "S2", "S2", "S3", "S3"]
        pseudobulk_adata.obs["cell_type"] = ["T", "B"] * 3

        result = pseudobulk_service._add_aggregation_metadata(
            pseudobulk_adata,
            mock_single_cell_data,
            sample_col="sample_id",
            celltype_col="cell_type",
            layer=None,
            min_cells=10,
            aggregation_method="sum",
            min_genes=200,
            filter_zeros=True,
            filtered_stats={"excluded_groups": 2},
        )

        assert "pseudobulk_params" in result.uns
        assert "aggregation_stats" in result.uns

        # Check parameter storage
        params = result.uns["pseudobulk_params"]
        assert params["sample_col"] == "sample_id"
        assert params["celltype_col"] == "cell_type"
        assert params["aggregation_method"] == "sum"
        assert params["min_cells"] == 10

        # Check statistics
        stats = result.uns["aggregation_stats"]
        assert "total_cells_aggregated" in stats
        assert "n_samples" in stats
        assert "n_cell_types" in stats

    def test_create_pseudobulk_anndata(self, pseudobulk_service):
        """Test creation of pseudobulk AnnData object."""
        # Create test data
        pseudobulk_matrix = np.random.randn(4, 50)
        group_metadata = pd.DataFrame(
            {
                "sample_id": ["S1", "S1", "S2", "S2"],
                "cell_type": ["T", "B", "T", "B"],
                "n_cells_aggregated": [100, 80, 90, 70],
            },
            index=["S1_T", "S1_B", "S2_T", "S2_B"],
        )

        var_metadata = pd.DataFrame(index=[f"Gene_{i}" for i in range(50)])

        result = pseudobulk_service._create_pseudobulk_anndata(
            pseudobulk_matrix,
            group_metadata,
            var_metadata,
            "sample_id",
            "cell_type",
            "sum",
            {},
        )

        assert isinstance(result, ad.AnnData)
        assert result.n_obs == 4
        assert result.n_vars == 50
        assert "sample_id" in result.obs.columns
        assert "cell_type" in result.obs.columns
        assert "aggregation_method" in result.obs.columns
        assert all(result.obs["aggregation_method"] == "sum")


@pytest.mark.unit
class TestPseudobulkServiceExport:
    """Test PseudobulkService export functionality."""

    @patch("pathlib.Path.mkdir")
    def test_export_for_deseq2(self, mock_mkdir, pseudobulk_service):
        """Test exporting pseudobulk data for DESeq2."""
        # Create pseudobulk data
        adata = ad.AnnData(
            X=np.random.randint(0, 1000, (6, 100)),
            obs=pd.DataFrame(
                {
                    "sample_id": ["S1", "S1", "S1", "S2", "S2", "S2"],
                    "cell_type": ["T", "B", "NK", "T", "B", "NK"],
                    "condition": ["Control"] * 3 + ["Treatment"] * 3,
                },
                index=[f"S{i//3+1}_{['T', 'B', 'NK'][i%3]}" for i in range(6)],
            ),
            var=pd.DataFrame(index=[f"Gene_{i}" for i in range(100)]),
        )

        with patch("pandas.DataFrame.to_csv") as mock_to_csv:
            export_paths = pseudobulk_service.export_for_deseq2(
                adata, output_dir="test_output"
            )

            assert "counts" in export_paths
            assert "sample_metadata" in export_paths
            assert "gene_metadata" in export_paths

            # Should call to_csv for each file
            assert mock_to_csv.call_count == 3
            mock_mkdir.assert_called()

    def test_export_with_layer(self, pseudobulk_service):
        """Test export using specific count layer."""
        adata = ad.AnnData(X=np.random.randn(4, 50))
        adata.layers["raw_counts"] = np.random.randint(0, 1000, (4, 50))
        adata.obs = pd.DataFrame({"sample": ["S1", "S2", "S3", "S4"]})
        adata.var = pd.DataFrame(index=[f"Gene_{i}" for i in range(50)])

        with patch("pandas.DataFrame.to_csv"):
            with patch("pathlib.Path.mkdir"):
                export_paths = pseudobulk_service.export_for_deseq2(
                    adata, output_dir="test_output", count_layer="raw_counts"
                )

                assert "counts" in export_paths
                assert "sample_metadata" in export_paths
                assert "gene_metadata" in export_paths


@pytest.mark.unit
class TestPseudobulkServiceSummary:
    """Test PseudobulkService summary functionality."""

    def test_get_aggregation_summary_valid(self, pseudobulk_service):
        """Test aggregation summary with valid data."""
        # Create pseudobulk data with required metadata
        adata = ad.AnnData(X=np.random.randn(6, 100))
        adata.uns["aggregation_stats"] = {
            "n_samples": 2,
            "n_cell_types": 3,
            "total_cells_aggregated": 500,
        }

        with patch.object(
            pseudobulk_service.adapter, "get_quality_metrics"
        ) as mock_metrics:
            mock_metrics.return_value = {"total_counts": 1000000}

            summary = pseudobulk_service.get_aggregation_summary(adata)

            assert isinstance(summary, dict)
            assert "n_samples" in summary
            assert "quality_metrics" in summary
            assert summary["n_samples"] == 2

    def test_get_aggregation_summary_missing_stats(self, pseudobulk_service):
        """Test aggregation summary with missing statistics."""
        adata = ad.AnnData(X=np.random.randn(6, 100))
        # Missing aggregation_stats

        summary = pseudobulk_service.get_aggregation_summary(adata)

        assert "error" in summary
        assert "No aggregation statistics found" in summary["error"]


@pytest.mark.unit
class TestPseudobulkServiceErrorHandling:
    """Test PseudobulkService error handling."""

    def test_aggregation_error_wrapping(
        self, pseudobulk_service, mock_single_cell_data
    ):
        """Test that unexpected errors are wrapped in PseudobulkError."""
        with patch.object(
            pseudobulk_service,
            "_validate_aggregation_inputs",
            side_effect=ValueError("Unexpected error"),
        ):
            with pytest.raises(PseudobulkError, match="Aggregation failed"):
                pseudobulk_service.aggregate_to_pseudobulk(
                    mock_single_cell_data, "sample_id", "cell_type"
                )

    def test_specific_error_propagation(
        self, pseudobulk_service, mock_single_cell_data
    ):
        """Test that specific errors are not wrapped."""
        # AggregationError should propagate without wrapping
        with pytest.raises(AggregationError):
            pseudobulk_service.aggregate_to_pseudobulk(
                mock_single_cell_data, "nonexistent_col", "cell_type"
            )


@pytest.mark.unit
class TestPseudobulkAggregation:
    """Comprehensive tests for pseudobulk aggregation logic."""

    def test_aggregate_basic_sum(self, pseudobulk_service):
        """Test basic sum aggregation."""
        np.random.seed(42)
        n_cells = 100
        n_genes = 50

        X = np.random.negative_binomial(n=5, p=0.5, size=(n_cells, n_genes))
        obs = pd.DataFrame(
            {
                "sample": ["sample1"] * 30 + ["sample2"] * 30 + ["sample3"] * 40,
                "celltype": (["T_cell"] * 15 + ["B_cell"] * 15) * 2
                + ["T_cell"] * 20
                + ["B_cell"] * 20,
            }
        )

        adata = ad.AnnData(
            X=X, obs=obs, var=pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)])
        )

        with patch.object(
            pseudobulk_service.adapter, "from_source"
        ) as mock_from_source:
            # Mock returns the input as-is
            mock_from_source.side_effect = lambda x, **kwargs: x

            adata_pb, stats, ir = pseudobulk_service.aggregate_to_pseudobulk(
                adata,
                sample_col="sample",
                celltype_col="celltype",
                aggregation_method="sum",
                min_genes=0,  # Disable filtering for test
            )

        # Verify basic structure
        assert isinstance(adata_pb, ad.AnnData)
        assert adata_pb.n_obs > 0
        assert adata_pb.n_vars == n_genes

    def test_aggregate_basic_mean(self, pseudobulk_service):
        """Test mean aggregation."""
        np.random.seed(42)
        X = np.array([[10, 20], [30, 40], [50, 60]])
        obs = pd.DataFrame({"sample": ["s1", "s1", "s1"], "celltype": ["T", "T", "T"]})

        adata = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=["gene1", "gene2"]))

        with patch.object(
            pseudobulk_service.adapter, "from_source"
        ) as mock_from_source:
            mock_from_source.side_effect = lambda x, **kwargs: x

            adata_pb, stats, ir = pseudobulk_service.aggregate_to_pseudobulk(
                adata,
                sample_col="sample",
                celltype_col="celltype",
                aggregation_method="mean",
                min_cells=1,
                min_genes=0,  # Disable filtering for test
            )

        # Mean should be [30, 40] for the single group
        expected_mean = np.array([30, 40])
        np.testing.assert_array_almost_equal(adata_pb.X[0], expected_mean)

    def test_aggregate_basic_median(self, pseudobulk_service):
        """Test median aggregation."""
        np.random.seed(42)
        X = np.array([[10, 20], [30, 40], [50, 60]])
        obs = pd.DataFrame({"sample": ["s1", "s1", "s1"], "celltype": ["T", "T", "T"]})

        adata = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=["gene1", "gene2"]))

        with patch.object(
            pseudobulk_service.adapter, "from_source"
        ) as mock_from_source:
            mock_from_source.side_effect = lambda x, **kwargs: x

            adata_pb, stats, ir = pseudobulk_service.aggregate_to_pseudobulk(
                adata,
                sample_col="sample",
                celltype_col="celltype",
                aggregation_method="median",
                min_cells=1,
                min_genes=0,  # Disable filtering for test
            )

        # Median should be [30, 40] for the single group
        expected_median = np.array([30, 40])
        np.testing.assert_array_almost_equal(adata_pb.X[0], expected_median)

    def test_aggregate_returns_three_tuple(self, pseudobulk_service):
        """Verify 3-tuple return (adata, stats, ir)."""
        np.random.seed(42)
        X = np.random.negative_binomial(5, 0.5, (50, 30))
        obs = pd.DataFrame(
            {
                "sample": ["s1"] * 25 + ["s2"] * 25,
                "celltype": ["T"] * 12 + ["B"] * 13 + ["T"] * 13 + ["B"] * 12,
            }
        )

        adata = ad.AnnData(
            X=X, obs=obs, var=pd.DataFrame(index=[f"gene_{i}" for i in range(30)])
        )

        with patch.object(
            pseudobulk_service.adapter, "from_source"
        ) as mock_from_source:
            mock_from_source.side_effect = lambda x, **kwargs: x

            result = pseudobulk_service.aggregate_to_pseudobulk(
                adata, sample_col="sample", celltype_col="celltype"
            )

        # Verify 3-tuple
        assert isinstance(result, tuple)
        assert len(result) == 3

        adata_pb, stats, ir = result

        # Verify types
        assert isinstance(adata_pb, ad.AnnData)
        assert isinstance(stats, dict)
        assert hasattr(ir, "operation")  # AnalysisStep

    def test_aggregate_min_cells_filter(self, pseudobulk_service):
        """Test min_cells threshold filtering."""
        np.random.seed(42)
        # Create data with variable cell counts per group
        X = np.random.negative_binomial(5, 0.5, (100, 50))
        obs = pd.DataFrame(
            {
                "sample": ["s1"] * 5 + ["s2"] * 25 + ["s3"] * 70,  # s1 has only 5 cells
                "celltype": ["T"] * 100,
            }
        )

        adata = ad.AnnData(
            X=X, obs=obs, var=pd.DataFrame(index=[f"gene_{i}" for i in range(50)])
        )

        with patch.object(
            pseudobulk_service.adapter, "from_source"
        ) as mock_from_source:
            mock_from_source.side_effect = lambda x, **kwargs: x

            adata_pb, stats, ir = pseudobulk_service.aggregate_to_pseudobulk(
                adata,
                sample_col="sample",
                celltype_col="celltype",
                min_cells=20,
                min_genes=0,  # Disable filtering for test
            )

        # s1_T should be filtered out (only 5 cells < 20 threshold)
        assert adata_pb.n_obs == 2  # Only s2_T and s3_T remain
        assert all(adata_pb.obs["n_cells_aggregated"] >= 20)

    def test_aggregate_creates_correct_obs(self, pseudobulk_service):
        """Verify obs columns (sample_id, cell_type, n_cells_aggregated)."""
        np.random.seed(42)
        X = np.random.negative_binomial(5, 0.5, (60, 30))
        obs = pd.DataFrame(
            {
                "sample": ["s1"] * 30 + ["s2"] * 30,
                "celltype": ["T"] * 15 + ["B"] * 15 + ["T"] * 15 + ["B"] * 15,
            }
        )

        adata = ad.AnnData(
            X=X, obs=obs, var=pd.DataFrame(index=[f"gene_{i}" for i in range(30)])
        )

        with patch.object(
            pseudobulk_service.adapter, "from_source"
        ) as mock_from_source:
            mock_from_source.side_effect = lambda x, **kwargs: x

            adata_pb, stats, ir = pseudobulk_service.aggregate_to_pseudobulk(
                adata, sample_col="sample", celltype_col="celltype", min_cells=10
            )

        # Verify required columns
        assert "sample_id" in adata_pb.obs.columns
        assert "cell_type" in adata_pb.obs.columns
        assert "n_cells_aggregated" in adata_pb.obs.columns

        # Verify values
        assert adata_pb.obs["sample_id"].nunique() <= 2
        assert adata_pb.obs["cell_type"].nunique() <= 2
        assert all(adata_pb.obs["n_cells_aggregated"] >= 10)

    def test_aggregate_preserves_var(self, pseudobulk_service):
        """Test that var/var_names are preserved."""
        np.random.seed(42)
        gene_names = [f"GENE_{i}" for i in range(50)]
        X = np.random.negative_binomial(5, 0.5, (60, 50))
        obs = pd.DataFrame(
            {"sample": ["s1"] * 30 + ["s2"] * 30, "celltype": ["T"] * 30 + ["B"] * 30}
        )

        var = pd.DataFrame(
            {"gene_symbol": gene_names, "feature_type": ["Gene"] * 50}, index=gene_names
        )

        adata = ad.AnnData(X=X, obs=obs, var=var)

        with patch.object(
            pseudobulk_service.adapter, "from_source"
        ) as mock_from_source:
            mock_from_source.side_effect = lambda x, **kwargs: x

            adata_pb, stats, ir = pseudobulk_service.aggregate_to_pseudobulk(
                adata, sample_col="sample", celltype_col="celltype"
            )

        # Var should be preserved (or filtered but not altered)
        assert "gene_symbol" in adata_pb.var.columns
        assert "feature_type" in adata_pb.var.columns

    def test_aggregate_handles_sparse_matrix(self, pseudobulk_service):
        """Test with scipy sparse matrices."""
        np.random.seed(42)
        X_dense = np.random.negative_binomial(5, 0.5, (100, 50))
        X_sparse = sparse.csr_matrix(X_dense)

        obs = pd.DataFrame(
            {
                "sample": ["s1"] * 50 + ["s2"] * 50,
                "celltype": ["T"] * 25 + ["B"] * 25 + ["T"] * 25 + ["B"] * 25,
            }
        )

        adata = ad.AnnData(
            X=X_sparse,
            obs=obs,
            var=pd.DataFrame(index=[f"gene_{i}" for i in range(50)]),
        )

        with patch.object(
            pseudobulk_service.adapter, "from_source"
        ) as mock_from_source:
            mock_from_source.side_effect = lambda x, **kwargs: x

            adata_pb, stats, ir = pseudobulk_service.aggregate_to_pseudobulk(
                adata,
                sample_col="sample",
                celltype_col="celltype",
                min_genes=0,  # Disable filtering for test
            )

        # Should work with sparse input
        assert adata_pb.n_obs > 0
        assert adata_pb.n_vars == 50

    def test_aggregate_with_normalization(self, pseudobulk_service):
        """Test normalization before aggregation (using mean method)."""
        np.random.seed(42)
        X = np.random.negative_binomial(5, 0.5, (60, 30))
        obs = pd.DataFrame(
            {"sample": ["s1"] * 30 + ["s2"] * 30, "celltype": ["T"] * 30 + ["B"] * 30}
        )

        adata = ad.AnnData(
            X=X, obs=obs, var=pd.DataFrame(index=[f"gene_{i}" for i in range(30)])
        )

        with patch.object(
            pseudobulk_service.adapter, "from_source"
        ) as mock_from_source:
            mock_from_source.side_effect = lambda x, **kwargs: x

            adata_pb, stats, ir = pseudobulk_service.aggregate_to_pseudobulk(
                adata,
                sample_col="sample",
                celltype_col="celltype",
                aggregation_method="mean",  # Normalized aggregation
            )

        assert stats["aggregation_method"] == "mean"

    def test_aggregate_handles_missing_columns(self, pseudobulk_service):
        """Error handling for missing obs columns."""
        X = np.random.negative_binomial(5, 0.5, (50, 30))
        obs = pd.DataFrame(
            {"sample": ["s1"] * 25 + ["s2"] * 25, "celltype": ["T"] * 25 + ["B"] * 25}
        )

        adata = ad.AnnData(
            X=X, obs=obs, var=pd.DataFrame(index=[f"gene_{i}" for i in range(30)])
        )

        with pytest.raises(AggregationError, match="not found"):
            pseudobulk_service.aggregate_to_pseudobulk(
                adata, sample_col="nonexistent", celltype_col="celltype"
            )


@pytest.mark.unit
class TestPseudobulkIR:
    """Test AnalysisStep IR generation and structure."""

    def test_create_pseudobulk_ir_structure(self, pseudobulk_service):
        """Validate IR completeness."""
        ir = pseudobulk_service._create_pseudobulk_ir(
            sample_col="sample",
            celltype_col="celltype",
            aggregation_method="sum",
            min_cells=10,
        )

        # Required fields
        assert hasattr(ir, "operation")
        assert hasattr(ir, "tool_name")
        assert hasattr(ir, "description")
        assert hasattr(ir, "library")
        assert hasattr(ir, "code_template")
        assert hasattr(ir, "imports")
        assert hasattr(ir, "parameters")
        assert hasattr(ir, "parameter_schema")
        assert hasattr(ir, "input_entities")
        assert hasattr(ir, "output_entities")

        # Verify values
        assert ir.operation == "pseudobulk_aggregation"
        assert ir.tool_name == "PseudobulkService.aggregate_to_pseudobulk"
        assert ir.library == "scanpy + numpy"

    def test_pseudobulk_ir_parameter_schema(self, pseudobulk_service):
        """Validate parameter schema."""
        ir = pseudobulk_service._create_pseudobulk_ir(
            sample_col="sample",
            celltype_col="celltype",
            aggregation_method="mean",
            min_cells=20,
        )

        schema = ir.parameter_schema

        # Check required parameters
        assert "sample_col" in schema
        assert "celltype_col" in schema
        assert "aggregation_method" in schema
        assert "min_cells" in schema

        # Check schema structure
        assert schema["sample_col"]["type"] == "string"
        assert schema["celltype_col"]["type"] == "string"
        assert schema["aggregation_method"]["type"] == "string"
        assert schema["aggregation_method"]["enum"] == ["sum", "mean", "median"]
        assert schema["min_cells"]["type"] == "integer"

    def test_pseudobulk_ir_code_template_renders(self, pseudobulk_service):
        """Test Jinja2 code template rendering."""
        from jinja2 import Template

        ir = pseudobulk_service._create_pseudobulk_ir(
            sample_col="sample_id",
            celltype_col="cell_type",
            aggregation_method="sum",
            min_cells=10,
        )

        template = Template(ir.code_template)
        rendered = template.render(**ir.parameters)

        # Verify rendering works
        assert isinstance(rendered, str)
        assert "sample_id" in rendered
        assert "cell_type" in rendered
        assert "10" in rendered  # min_cells value

    def test_pseudobulk_ir_imports(self, pseudobulk_service):
        """Validate imports list."""
        ir = pseudobulk_service._create_pseudobulk_ir(
            sample_col="sample",
            celltype_col="celltype",
            aggregation_method="sum",
            min_cells=10,
        )

        assert isinstance(ir.imports, list)
        assert len(ir.imports) > 0

        # Check for standard imports
        import_str = " ".join(ir.imports)
        assert "scanpy" in import_str
        assert "pandas" in import_str
        assert "numpy" in import_str

    def test_pseudobulk_ir_entities(self, pseudobulk_service):
        """Verify input/output entities."""
        ir = pseudobulk_service._create_pseudobulk_ir(
            sample_col="sample",
            celltype_col="celltype",
            aggregation_method="sum",
            min_cells=10,
        )

        assert isinstance(ir.input_entities, list)
        assert isinstance(ir.output_entities, list)
        assert len(ir.input_entities) > 0
        assert len(ir.output_entities) > 0
        assert "singlecell_adata" in ir.input_entities
        assert "pseudobulk_adata" in ir.output_entities


@pytest.mark.unit
class TestPseudobulkStatistics:
    """Test statistics tracking and reporting."""

    def test_stats_include_all_metrics(self, pseudobulk_service):
        """Verify stats dict completeness."""
        np.random.seed(42)
        X = np.random.negative_binomial(5, 0.5, (100, 50))
        obs = pd.DataFrame(
            {
                "sample": ["s1"] * 30 + ["s2"] * 30 + ["s3"] * 40,
                "celltype": ["T"] * 50 + ["B"] * 50,
            }
        )

        adata = ad.AnnData(
            X=X, obs=obs, var=pd.DataFrame(index=[f"gene_{i}" for i in range(50)])
        )

        with patch.object(
            pseudobulk_service.adapter, "from_source"
        ) as mock_from_source:
            mock_from_source.side_effect = lambda x, **kwargs: x

            adata_pb, stats, ir = pseudobulk_service.aggregate_to_pseudobulk(
                adata, sample_col="sample", celltype_col="celltype"
            )

        required_keys = [
            "n_pseudobulk_samples",
            "n_genes",
            "n_unique_samples",
            "n_cell_types",
            "total_cells_processed",
            "total_cells_aggregated",
            "aggregation_method",
        ]

        for key in required_keys:
            assert key in stats, f"Missing key: {key}"

    def test_stats_cell_counts(self, pseudobulk_service):
        """Check n_cells_per_pseudobulk tracking."""
        np.random.seed(42)
        X = np.random.negative_binomial(5, 0.5, (90, 30))
        obs = pd.DataFrame(
            {
                "sample": ["s1"] * 30 + ["s2"] * 30 + ["s3"] * 30,
                "celltype": ["T"] * 30 + ["B"] * 30 + ["T"] * 30,
            }
        )

        adata = ad.AnnData(
            X=X, obs=obs, var=pd.DataFrame(index=[f"gene_{i}" for i in range(30)])
        )

        with patch.object(
            pseudobulk_service.adapter, "from_source"
        ) as mock_from_source:
            mock_from_source.side_effect = lambda x, **kwargs: x

            adata_pb, stats, ir = pseudobulk_service.aggregate_to_pseudobulk(
                adata, sample_col="sample", celltype_col="celltype", min_cells=10
            )

        # Each pseudobulk should have cell count tracked
        assert "n_cells_aggregated" in adata_pb.obs.columns
        assert all(adata_pb.obs["n_cells_aggregated"] >= 10)

    def test_stats_sample_counts(self, pseudobulk_service):
        """Verify n_samples and n_celltypes."""
        np.random.seed(42)
        X = np.random.negative_binomial(5, 0.5, (120, 40))
        obs = pd.DataFrame(
            {
                "sample": ["s1"] * 40 + ["s2"] * 40 + ["s3"] * 40,
                "celltype": (["T"] * 20 + ["B"] * 20) * 3,
            }
        )

        adata = ad.AnnData(
            X=X, obs=obs, var=pd.DataFrame(index=[f"gene_{i}" for i in range(40)])
        )

        with patch.object(
            pseudobulk_service.adapter, "from_source"
        ) as mock_from_source:
            mock_from_source.side_effect = lambda x, **kwargs: x

            adata_pb, stats, ir = pseudobulk_service.aggregate_to_pseudobulk(
                adata,
                sample_col="sample",
                celltype_col="celltype",
                min_genes=0,  # Disable filtering for test
            )

        assert stats["n_unique_samples"] == 3
        assert stats["n_cell_types"] == 2

    def test_stats_min_max_cells(self, pseudobulk_service):
        """Check min/max cells per pseudobulk (via uns)."""
        np.random.seed(42)
        # Create uneven groups
        X = np.random.negative_binomial(5, 0.5, (100, 30))
        obs = pd.DataFrame(
            {
                "sample": ["s1"] * 20 + ["s2"] * 80,  # Uneven
                "celltype": ["T"] * 20 + ["T"] * 80,
            }
        )

        adata = ad.AnnData(
            X=X, obs=obs, var=pd.DataFrame(index=[f"gene_{i}" for i in range(30)])
        )

        with patch.object(
            pseudobulk_service.adapter, "from_source"
        ) as mock_from_source:
            mock_from_source.side_effect = lambda x, **kwargs: x

            adata_pb, stats, ir = pseudobulk_service.aggregate_to_pseudobulk(
                adata, sample_col="sample", celltype_col="celltype", min_cells=10
            )

        # Check that uns contains aggregation stats
        assert "aggregation_stats" in adata_pb.uns
        agg_stats = adata_pb.uns["aggregation_stats"]
        assert "mean_cells_per_pseudobulk" in agg_stats

    def test_stats_filtered_counts(self, pseudobulk_service):
        """Verify filtered pseudobulk counts."""
        np.random.seed(42)
        X = np.random.negative_binomial(5, 0.5, (100, 30))
        obs = pd.DataFrame(
            {
                "sample": ["s1"] * 5 + ["s2"] * 25 + ["s3"] * 70,  # s1 has too few
                "celltype": ["T"] * 100,
            }
        )

        adata = ad.AnnData(
            X=X, obs=obs, var=pd.DataFrame(index=[f"gene_{i}" for i in range(30)])
        )

        with patch.object(
            pseudobulk_service.adapter, "from_source"
        ) as mock_from_source:
            mock_from_source.side_effect = lambda x, **kwargs: x

            adata_pb, stats, ir = pseudobulk_service.aggregate_to_pseudobulk(
                adata, sample_col="sample", celltype_col="celltype", min_cells=20
            )

        # Check filtering happened
        assert "aggregation_stats" in adata_pb.uns
        agg_stats = adata_pb.uns["aggregation_stats"]
        assert "excluded_groups" in agg_stats


@pytest.mark.unit
class TestPseudobulkEdgeCases:
    """Test edge cases and error conditions."""

    def test_single_sample(self, pseudobulk_service):
        """Handle single sample edge case."""
        np.random.seed(42)
        X = np.random.negative_binomial(5, 0.5, (50, 30))
        obs = pd.DataFrame(
            {
                "sample": ["s1"] * 50,  # Only one sample
                "celltype": ["T"] * 25 + ["B"] * 25,
            }
        )

        adata = ad.AnnData(
            X=X, obs=obs, var=pd.DataFrame(index=[f"gene_{i}" for i in range(30)])
        )

        with patch.object(
            pseudobulk_service.adapter, "from_source"
        ) as mock_from_source:
            mock_from_source.side_effect = lambda x, **kwargs: x

            adata_pb, stats, ir = pseudobulk_service.aggregate_to_pseudobulk(
                adata,
                sample_col="sample",
                celltype_col="celltype",
                min_genes=0,  # Disable filtering for test
            )

        assert adata_pb.n_obs == 2  # s1_T and s1_B
        assert stats["n_unique_samples"] == 1

    def test_single_celltype(self, pseudobulk_service):
        """Handle single cell type."""
        np.random.seed(42)
        X = np.random.negative_binomial(5, 0.5, (60, 30))
        obs = pd.DataFrame(
            {
                "sample": ["s1"] * 30 + ["s2"] * 30,
                "celltype": ["T"] * 60,  # Only one cell type
            }
        )

        adata = ad.AnnData(
            X=X, obs=obs, var=pd.DataFrame(index=[f"gene_{i}" for i in range(30)])
        )

        with patch.object(
            pseudobulk_service.adapter, "from_source"
        ) as mock_from_source:
            mock_from_source.side_effect = lambda x, **kwargs: x

            adata_pb, stats, ir = pseudobulk_service.aggregate_to_pseudobulk(
                adata,
                sample_col="sample",
                celltype_col="celltype",
                min_genes=0,  # Disable filtering for test
            )

        assert adata_pb.n_obs == 2  # s1_T and s2_T
        assert stats["n_cell_types"] == 1

    def test_no_pseudobulks_above_threshold(self, pseudobulk_service):
        """All groups below min_cells."""
        np.random.seed(42)
        X = np.random.negative_binomial(5, 0.5, (30, 20))
        obs = pd.DataFrame(
            {"sample": ["s1"] * 5 + ["s2"] * 10 + ["s3"] * 15, "celltype": ["T"] * 30}
        )

        adata = ad.AnnData(
            X=X, obs=obs, var=pd.DataFrame(index=[f"gene_{i}" for i in range(20)])
        )

        with pytest.raises(InsufficientCellsError):
            pseudobulk_service.aggregate_to_pseudobulk(
                adata,
                sample_col="sample",
                celltype_col="celltype",
                min_cells=50,  # Too high
            )

    def test_uneven_cell_counts(self, pseudobulk_service):
        """Highly variable cell counts per group."""
        np.random.seed(42)
        X = np.random.negative_binomial(5, 0.5, (150, 30))
        obs = pd.DataFrame(
            {
                "sample": ["s1"] * 10 + ["s2"] * 140,  # Very uneven
                "celltype": ["T"] * 10 + ["T"] * 70 + ["B"] * 70,
            }
        )

        adata = ad.AnnData(
            X=X, obs=obs, var=pd.DataFrame(index=[f"gene_{i}" for i in range(30)])
        )

        with patch.object(
            pseudobulk_service.adapter, "from_source"
        ) as mock_from_source:
            mock_from_source.side_effect = lambda x, **kwargs: x

            adata_pb, stats, ir = pseudobulk_service.aggregate_to_pseudobulk(
                adata,
                sample_col="sample",
                celltype_col="celltype",
                min_cells=10,
                min_genes=0,  # Disable filtering for test
            )

        # Should handle uneven distribution
        assert adata_pb.n_obs >= 1
        cell_counts = adata_pb.obs["n_cells_aggregated"].values
        assert max(cell_counts) > min(cell_counts)

    def test_empty_anndata(self, pseudobulk_service):
        """Error handling for empty AnnData."""
        adata = ad.AnnData(
            X=np.array([]).reshape(0, 10),
            obs=pd.DataFrame({"sample": [], "celltype": []}),
            var=pd.DataFrame(index=[f"gene_{i}" for i in range(10)]),
        )

        with pytest.raises((InsufficientCellsError, AggregationError)):
            pseudobulk_service.aggregate_to_pseudobulk(
                adata, sample_col="sample", celltype_col="celltype"
            )

    def test_duplicate_sample_celltype_pairs(self, pseudobulk_service):
        """Handles duplicates gracefully (shouldn't happen but test anyway)."""
        np.random.seed(42)
        X = np.random.negative_binomial(5, 0.5, (60, 30))
        obs = pd.DataFrame(
            {"sample": ["s1"] * 60, "celltype": ["T"] * 60}  # All same group
        )

        adata = ad.AnnData(
            X=X, obs=obs, var=pd.DataFrame(index=[f"gene_{i}" for i in range(30)])
        )

        with patch.object(
            pseudobulk_service.adapter, "from_source"
        ) as mock_from_source:
            mock_from_source.side_effect = lambda x, **kwargs: x

            adata_pb, stats, ir = pseudobulk_service.aggregate_to_pseudobulk(
                adata,
                sample_col="sample",
                celltype_col="celltype",
                min_genes=0,  # Disable filtering for test
            )

        # Should create single pseudobulk sample
        assert adata_pb.n_obs == 1
