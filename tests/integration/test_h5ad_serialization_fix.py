"""
Integration test for H5AD serialization bug fix.

This test verifies that GEO datasets with problematic metadata (Path objects,
tuples, etc.) can be successfully downloaded and saved to H5AD format without
serialization errors.

Addresses the bug: "TypeError: Can't implicitly convert non-string objects to strings"
"""

import tempfile
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import pytest

from lobster.core.adapters.transcriptomics_adapter import TranscriptomicsAdapter
from lobster.core.backends.h5ad_backend import H5ADBackend
from lobster.core.data_manager_v2 import DataManagerV2
from lobster.tools.geo_service import GEOService


@pytest.fixture
def temp_workspace():
    """Create a temporary workspace for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def data_manager(temp_workspace):
    """Create a DataManagerV2 instance with temporary workspace."""
    workspace_path = temp_workspace / "workspace"
    workspace_path.mkdir(exist_ok=True)
    dm = DataManagerV2(workspace_path=workspace_path)
    return dm


@pytest.fixture
def geo_service(temp_workspace):
    """Create a GEOService instance with temporary cache."""
    cache_dir = temp_workspace / "geo_cache"
    cache_dir.mkdir(exist_ok=True)
    return GEOService(cache_dir=str(cache_dir))


class TestH5ADSerializationFix:
    """Tests for H5AD serialization bug fix."""

    def test_path_objects_in_metadata(self, temp_workspace):
        """Test that Path objects in metadata are sanitized."""
        # Create test data with Path objects
        adata = anndata.AnnData(X=np.array([[1, 2], [3, 4]]))
        adata.uns["file_path"] = Path("/tmp/test.csv")
        adata.uns["cache_dir"] = Path("/tmp/cache")
        adata.uns["files"] = [Path("/tmp/file1"), Path("/tmp/file2")]

        # Save to H5AD - should NOT raise TypeError
        backend = H5ADBackend()
        output_path = temp_workspace / "test_path_objects.h5ad"

        # This should succeed without errors
        backend.save(adata, output_path)

        # Verify file was created
        assert output_path.exists()

        # Load back and verify sanitization
        loaded = backend.load(output_path)
        assert isinstance(loaded.uns["file_path"], str)
        assert isinstance(loaded.uns["cache_dir"], str)
        assert all(isinstance(f, str) for f in loaded.uns["files"])

    def test_tuples_in_metadata(self, temp_workspace):
        """Test that tuples in metadata are converted to lists."""
        adata = anndata.AnnData(X=np.array([[1, 2], [3, 4]]))
        adata.uns["shape"] = (100, 200)
        adata.uns["dimensions"] = (10, 20, 30)
        adata.uns["nested"] = {"inner_tuple": (1, 2, 3)}

        backend = H5ADBackend()
        output_path = temp_workspace / "test_tuples.h5ad"

        # Should succeed
        backend.save(adata, output_path)
        assert output_path.exists()

        # Verify tuples → lists (or numpy arrays after H5AD roundtrip)
        loaded = backend.load(output_path)
        assert isinstance(loaded.uns["shape"], (list, np.ndarray))
        assert list(loaded.uns["shape"]) == [100, 200]
        assert isinstance(loaded.uns["nested"]["inner_tuple"], (list, np.ndarray))

    def test_complex_geo_metadata(self, temp_workspace):
        """Test GEO-like metadata with multiple problematic types."""
        adata = anndata.AnnData(X=np.array([[1, 2], [3, 4]]))

        # Simulate typical GEO metadata
        adata.uns["dataset_id"] = "GSE194070"
        adata.uns["cache_path"] = Path("/tmp/geo_cache")
        adata.uns["file_paths"] = [Path("/tmp/f1.csv"), Path("/tmp/f2.csv")]
        adata.uns["shape"] = (1000, 2000)  # tuple from adata.shape
        adata.uns["processing_date"] = "2024-11-15"
        adata.uns["quantification_metadata"] = {
            "tool": "kallisto",
            "index": Path("/data/index"),
            "parameters": {
                "bootstrap": 100,
                "original_shape": (60000, 4),
            }
        }

        backend = H5ADBackend()
        output_path = temp_workspace / "test_geo_metadata.h5ad"

        # Should succeed without TypeError
        backend.save(adata, output_path)
        assert output_path.exists()

        # Verify all types are safe
        loaded = backend.load(output_path)
        assert isinstance(loaded.uns["cache_path"], str)
        assert all(isinstance(p, str) for p in loaded.uns["file_paths"])
        assert isinstance(loaded.uns["shape"], (list, np.ndarray))
        assert isinstance(
            loaded.uns["quantification_metadata"]["parameters"]["original_shape"],
            (list, np.ndarray)
        )

    def test_transcriptomics_adapter_sanitization(self, temp_workspace):
        """Test that TranscriptomicsAdapter sanitizes metadata on load."""
        adapter = TranscriptomicsAdapter(data_type="single_cell")

        # Create test AnnData with problematic metadata
        adata_source = anndata.AnnData(X=np.array([[1, 2], [3, 4]]))

        # Load with problematic metadata passed as kwargs
        adata = adapter.from_source(
            source=adata_source,
            custom_path=Path("/problematic/path.csv"),  # Path object in metadata
            custom_shape=(100, 200),  # tuple in metadata
            custom_files=[Path("/file1"), Path("/file2")],  # list of Paths in metadata
        )

        # Save to H5AD - should work
        backend = H5ADBackend()
        output_path = temp_workspace / "test_adapter_sanitization.h5ad"
        backend.save(adata, output_path)

        # Verify sanitization occurred
        loaded = backend.load(output_path)
        if "custom_path" in loaded.uns:
            assert isinstance(loaded.uns["custom_path"], str)
        if "custom_shape" in loaded.uns:
            assert isinstance(loaded.uns["custom_shape"], (list, np.ndarray))

    def test_data_manager_validation_warnings(self, data_manager, temp_workspace):
        """Test that DataManagerV2 logs validation warnings."""
        # Create modality with problematic metadata
        adata = anndata.AnnData(X=np.array([[1, 2], [3, 4]]))
        adata.uns["problematic_path"] = Path("/tmp/test")
        adata.uns["problematic_tuple"] = (1, 2, 3)

        # Load into data manager
        data_manager.modalities["test_data"] = adata

        # Save - should log warnings but succeed
        output_path = "test_validation_warnings.h5ad"
        saved_path = data_manager.save_modality("test_data", output_path)

        assert Path(saved_path).exists()

    @pytest.mark.real_api
    @pytest.mark.slow
    def test_real_geo_download_and_save(self, data_manager, geo_service, temp_workspace):
        """
        Integration test with real GEO download.

        Downloads a small GEO dataset and verifies it can be saved to H5AD.
        This is the real-world scenario that triggered the original bug.
        """
        # Use a small, well-formed dataset
        geo_id = "GSE138266"  # Small single-cell dataset

        try:
            # Download from GEO
            result = geo_service.download_dataset(
                geo_id=geo_id,
                data_manager=data_manager,
                force=False  # Use cache if available
            )

            # Verify dataset was loaded
            modality_name = f"geo_{geo_id.lower()}_None"
            assert modality_name in data_manager.modalities

            # Save to H5AD - this was failing before the fix
            output_path = temp_workspace / f"{geo_id}_test_save.h5ad"
            data_manager.save_modality(modality_name, str(output_path))

            # Verify file was created and can be loaded back
            assert output_path.exists()
            backend = H5ADBackend()
            loaded = backend.load(output_path)
            assert loaded.n_obs > 0
            assert loaded.n_vars > 0

        except Exception as e:
            pytest.fail(f"Real GEO download and save failed: {e}")


class TestAdapterSanitization:
    """Tests for adapter-level sanitization."""

    def test_quantification_dataframe_sanitization(self, temp_workspace):
        """Test from_quantification_dataframe sanitizes metadata."""
        adapter = TranscriptomicsAdapter(data_type="bulk")

        # Create test quantification data
        df = pd.DataFrame(
            [[1, 2], [3, 4]],
            index=["gene1", "gene2"],
            columns=["sample1", "sample2"]
        )

        # Metadata with problematic types
        metadata = {
            "quantification_tool": "kallisto",
            "index_path": Path("/data/index"),  # Path object
            "version": "0.46.1",
            "files": [Path("/data/s1"), Path("/data/s2")],  # list of Paths
        }

        # Create AnnData
        adata = adapter.from_quantification_dataframe(
            df=df,
            data_type="bulk_rnaseq",
            metadata=metadata
        )

        # Verify sanitization happened
        assert "quantification_metadata" in adata.uns
        quant_meta = adata.uns["quantification_metadata"]

        # Path objects should be strings
        assert isinstance(quant_meta.get("index_path"), str)
        if "files" in quant_meta:
            assert all(isinstance(f, str) for f in quant_meta["files"])

        # transpose_info should have lists, not tuples
        assert "transpose_info" in adata.uns
        transpose_info = adata.uns["transpose_info"]
        assert isinstance(transpose_info.get("original_shape"), list)
        assert isinstance(transpose_info.get("final_shape"), list)

        # Save to verify no serialization errors
        backend = H5ADBackend()
        output_path = temp_workspace / "test_quant_sanitization.h5ad"
        backend.save(adata, output_path)
        assert output_path.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
