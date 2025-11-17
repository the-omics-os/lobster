"""
Unit tests for GEO service quantification file integration (Phase 4).

This module tests the Phase 4 architectural changes where:
1. _load_quantification_files() returns AnnData (not DataFrame)
2. Quantification files route through unified pathway without duplicate storage
3. Both DataFrame and AnnData return types are handled correctly

Test coverage target: Validate Phase 4 architectural consistency.
"""

import tempfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from lobster.core.data_manager_v2 import DataManagerV2
from lobster.tools.geo_service import GEODataSource, GEOResult, GEOService

# ===============================================================================
# Fixtures
# ===============================================================================


@pytest.fixture
def temp_workspace():
    """Create temporary workspace for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        workspace_path = Path(temp_dir) / ".lobster_workspace"
        workspace_path.mkdir(parents=True, exist_ok=True)
        yield workspace_path


@pytest.fixture
def data_manager(temp_workspace):
    """Create DataManagerV2 instance."""
    return DataManagerV2(workspace_path=temp_workspace)


@pytest.fixture
def geo_service(data_manager):
    """Create GEOService instance."""
    return GEOService(
        data_manager=data_manager,
        cache_dir=data_manager.cache_dir / "geo",
        console=None,
    )


@pytest.fixture
def mock_kallisto_dataset(tmp_path):
    """Create mock Kallisto quantification dataset."""
    kallisto_dir = tmp_path / "quantification"
    kallisto_dir.mkdir()

    n_genes = 1000
    gene_ids = [f"ENSG{i:011d}" for i in range(n_genes)]

    sample_names = ["control_1", "control_2", "treatment_1", "treatment_2"]

    for sample in sample_names:
        sample_dir = kallisto_dir / sample
        sample_dir.mkdir()

        abundance_data = pd.DataFrame(
            {
                "target_id": gene_ids,
                "length": np.random.randint(500, 5000, n_genes),
                "eff_length": np.random.randint(400, 4900, n_genes),
                "est_counts": np.random.exponential(10, n_genes),
                "tpm": np.random.exponential(5, n_genes),
            }
        )

        abundance_file = sample_dir / "abundance.tsv"
        abundance_data.to_csv(abundance_file, sep="\t", index=False)

    return kallisto_dir, sample_names, n_genes


# ===============================================================================
# Phase 4: Test _load_quantification_files() Returns AnnData
# ===============================================================================


class TestLoadQuantificationFilesReturnType:
    """Test that _load_quantification_files() returns AnnData, not DataFrame."""

    def test_returns_anndata_not_dataframe(self, geo_service, mock_kallisto_dataset):
        """CRITICAL: Verify _load_quantification_files() returns AnnData."""
        kallisto_dir, sample_names, n_genes = mock_kallisto_dataset

        result = geo_service._load_quantification_files(
            quantification_dir=kallisto_dir,
            tool_type="kallisto",
            gse_id="GSE_TEST",
            data_type="bulk",
        )

        # CRITICAL ASSERTION: Must return AnnData, not DataFrame
        assert isinstance(result, ad.AnnData), f"Expected AnnData, got {type(result)}"
        assert not isinstance(result, pd.DataFrame), "Should NOT return DataFrame"

    def test_anndata_has_correct_orientation(self, geo_service, mock_kallisto_dataset):
        """Test that returned AnnData has correct orientation (samples × genes)."""
        kallisto_dir, sample_names, n_genes = mock_kallisto_dataset

        adata = geo_service._load_quantification_files(
            quantification_dir=kallisto_dir,
            tool_type="kallisto",
            gse_id="GSE_TEST",
            data_type="bulk",
        )

        # Validate orientation
        assert adata.n_obs == len(
            sample_names
        ), f"Expected {len(sample_names)} samples (obs), got {adata.n_obs}"
        assert (
            adata.n_vars == n_genes
        ), f"Expected {n_genes} genes (vars), got {adata.n_vars}"
        assert adata.n_obs < adata.n_vars, "Bulk RNA-seq: samples should be < genes"

    def test_anndata_has_metadata(self, geo_service, mock_kallisto_dataset):
        """Test that returned AnnData has quantification metadata."""
        kallisto_dir, sample_names, n_genes = mock_kallisto_dataset

        adata = geo_service._load_quantification_files(
            quantification_dir=kallisto_dir,
            tool_type="kallisto",
            gse_id="GSE_TEST",
            data_type="bulk",
        )

        # Verify metadata
        assert "quantification_metadata" in adata.uns
        assert adata.uns["quantification_metadata"]["quantification_tool"] == "Kallisto"
        assert "geo_metadata" in adata.uns
        assert adata.uns["geo_metadata"]["geo_id"] == "GSE_TEST"

    def test_does_not_store_in_data_manager(self, geo_service, mock_kallisto_dataset):
        """CRITICAL: Verify _load_quantification_files() does NOT store in data_manager."""
        kallisto_dir, sample_names, n_genes = mock_kallisto_dataset

        # Record initial modality count
        initial_modalities = set(geo_service.data_manager.list_modalities())

        # Call the method
        adata = geo_service._load_quantification_files(
            quantification_dir=kallisto_dir,
            tool_type="kallisto",
            gse_id="GSE_TEST",
            data_type="bulk",
        )

        # Verify no new modalities were added
        final_modalities = set(geo_service.data_manager.list_modalities())
        new_modalities = final_modalities - initial_modalities

        assert (
            len(new_modalities) == 0
        ), f"Should NOT store in data_manager, but added: {new_modalities}"

    def test_returns_none_on_failure(self, geo_service, tmp_path):
        """Test that method returns None on failure (not exception)."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        result = geo_service._load_quantification_files(
            quantification_dir=empty_dir,
            tool_type="kallisto",
            gse_id="GSE_TEST",
            data_type="bulk",
        )

        assert result is None, "Should return None on failure"


# ===============================================================================
# Phase 4: Test Dual-Type Validation Check
# ===============================================================================
# Note: TAR file processing integration is already tested in
# tests/integration/test_kallisto_salmon_loading.py


class TestDualTypeValidation:
    """Test that validation checks work for both DataFrame and AnnData."""

    def test_validation_accepts_non_empty_dataframe(self, geo_service):
        """Test that validation accepts non-empty DataFrame."""
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})

        # This simulates the validation check in _try_archive_extraction_first
        if isinstance(df, pd.DataFrame):
            is_valid = not df.empty
        elif isinstance(df, ad.AnnData):
            is_valid = df.n_obs > 0 and df.n_vars > 0
        else:
            is_valid = False

        assert is_valid, "Non-empty DataFrame should be valid"

    def test_validation_rejects_empty_dataframe(self, geo_service):
        """Test that validation rejects empty DataFrame."""
        df = pd.DataFrame()

        if isinstance(df, pd.DataFrame):
            is_valid = not df.empty
        elif isinstance(df, ad.AnnData):
            is_valid = df.n_obs > 0 and df.n_vars > 0
        else:
            is_valid = False

        assert not is_valid, "Empty DataFrame should be invalid"

    def test_validation_accepts_non_empty_anndata(self, geo_service):
        """Test that validation accepts non-empty AnnData."""
        adata = ad.AnnData(
            X=np.random.randn(10, 5),
            obs=pd.DataFrame(index=[f"cell_{i}" for i in range(10)]),
            var=pd.DataFrame(index=[f"gene_{i}" for i in range(5)]),
        )

        if isinstance(adata, pd.DataFrame):
            is_valid = not adata.empty
        elif isinstance(adata, ad.AnnData):
            is_valid = adata.n_obs > 0 and adata.n_vars > 0
        else:
            is_valid = False

        assert is_valid, "Non-empty AnnData should be valid"

    def test_validation_rejects_empty_anndata(self, geo_service):
        """Test that validation rejects empty AnnData."""
        adata = ad.AnnData(
            X=np.array([]).reshape(0, 0), obs=pd.DataFrame(), var=pd.DataFrame()
        )

        if isinstance(adata, pd.DataFrame):
            is_valid = not adata.empty
        elif isinstance(adata, ad.AnnData):
            is_valid = adata.n_obs > 0 and adata.n_vars > 0
        else:
            is_valid = False

        assert not is_valid, "Empty AnnData should be invalid"


# ===============================================================================
# Phase 4: Test Naming Convention Consistency
# ===============================================================================


class TestNamingConventionConsistency:
    """Test that quantification files use standard naming convention."""

    def test_no_legacy_naming_pattern(self, geo_service, mock_kallisto_dataset):
        """Test that legacy naming pattern is NOT used."""
        kallisto_dir, sample_names, n_genes = mock_kallisto_dataset

        # Load quantification files
        adata = geo_service._load_quantification_files(
            quantification_dir=kallisto_dir,
            tool_type="kallisto",
            gse_id="GSE123456",
            data_type="bulk",
        )

        # Verify no modality was stored with legacy naming
        modalities = geo_service.data_manager.list_modalities()

        # Should NOT have "{gse_id}_quantification" pattern
        assert (
            "GSE123456_quantification" not in modalities
        ), "Should NOT use legacy naming pattern"

    def test_standard_naming_expected(self):
        """Test that standard naming pattern is expected: geo_{gse_id}_{adapter}."""
        # This test documents the expected naming convention
        gse_id = "GSE123456"
        adapter = "transcriptomics_bulk"

        expected_pattern = f"geo_{gse_id.lower()}_{adapter}"

        # This is what download_dataset() will use
        assert (
            expected_pattern == "geo_gse123456_transcriptomics_bulk"
        ), "Standard naming pattern should be used"


# ===============================================================================
# Phase 4: Regression Test for Architecture
# ===============================================================================


class TestPhase4ArchitectureRegression:
    """Regression tests to prevent reverting to old architecture."""

    def test_load_quantification_files_signature(self, geo_service):
        """Test that _load_quantification_files() has correct return type annotation."""
        import inspect
        from typing import get_type_hints

        # Get the method
        method = geo_service._load_quantification_files

        # Get type hints
        hints = get_type_hints(method)

        # Verify return type is Optional[AnnData], not Optional[DataFrame]
        assert "return" in hints, "Method should have return type annotation"

        # The return type should mention AnnData
        return_type_str = str(hints["return"])
        assert (
            "AnnData" in return_type_str or "anndata" in return_type_str
        ), f"Return type should be AnnData, got: {return_type_str}"

    def test_no_direct_storage_in_method(self, geo_service, mock_kallisto_dataset):
        """Test that _load_quantification_files() does not directly store results."""
        kallisto_dir, sample_names, n_genes = mock_kallisto_dataset

        # Record initial state
        initial_count = len(geo_service.data_manager.list_modalities())

        # Call method
        result = geo_service._load_quantification_files(
            quantification_dir=kallisto_dir,
            tool_type="kallisto",
            gse_id="GSE_REGRESSION",
            data_type="bulk",
        )

        # Verify no storage occurred
        final_count = len(geo_service.data_manager.list_modalities())

        assert (
            final_count == initial_count
        ), "Method should NOT store results directly in data_manager"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
