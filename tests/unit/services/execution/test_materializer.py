"""
Tests for the execution locality materializer in CustomCodeExecutionService.

Covers: _detect_required_modalities, _materialize_modality, alias map building,
and end-to-end execution with the locality provider.
"""

from pathlib import Path
from unittest.mock import Mock, patch

import anndata
import numpy as np
import pytest

from lobster.core.data_manager_v2 import DataManagerV2
from lobster.services.execution.custom_code_execution_service import (
    CodeValidationError,
    CustomCodeExecutionService,
)


@pytest.fixture
def mock_dm(tmp_path):
    dm = Mock(spec=DataManagerV2)
    dm.workspace_path = tmp_path
    dm.data_dir = tmp_path / "data"
    dm.list_modalities.return_value = ["rna", "protein"]
    locality = Mock()
    locality.list_available.return_value = ["rna", "protein", "spatial"]
    locality.is_local.return_value = False
    dm.locality_provider = locality
    # Provide real AnnData for get_modality
    dm.get_modality.side_effect = lambda name: {
        "rna": anndata.AnnData(X=np.array([[1, 2], [3, 4]])),
        "protein": anndata.AnnData(X=np.array([[5, 6]])),
    }.get(name, Mock())
    return dm


@pytest.fixture
def service(mock_dm):
    return CustomCodeExecutionService(mock_dm)


class TestDetectRequiredModalities:
    def test_explicit_modality_name(self, service):
        required = service._detect_required_modalities("result = 1", "rna")
        assert "rna" in required

    def test_read_h5ad_pattern(self, service):
        code = 'sc.read_h5ad("/tmp/workspace/modalities/rna.h5ad")'
        required = service._detect_required_modalities(code, None)
        assert "rna" in required

    def test_anndata_read_h5ad_pattern(self, service):
        code = 'adata = anndata.read_h5ad("data/protein.h5ad")'
        required = service._detect_required_modalities(code, None)
        assert "protein" in required

    def test_path_pattern(self, service):
        code = 'p = Path("data/rna.h5ad")'
        required = service._detect_required_modalities(code, None)
        assert "rna" in required

    def test_h5py_file_pattern(self, service):
        code = 'f = h5py.File("data/rna.h5ad", "r")'
        required = service._detect_required_modalities(code, None)
        assert "rna" in required

    def test_sc_read_pattern(self, service):
        code = 'adata = sc.read("protein.h5ad")'
        required = service._detect_required_modalities(code, None)
        assert "protein" in required

    def test_unresolvable_path_raises(self, service):
        code = 'sc.read_h5ad("unknown_dataset.h5ad")'
        with pytest.raises(CodeValidationError, match="don't match any known modality"):
            service._detect_required_modalities(code, None)

    def test_no_h5ad_refs_returns_empty(self, service):
        code = "result = 2 + 2"
        required = service._detect_required_modalities(code, None)
        assert len(required) == 0

    def test_multiple_modalities_detected(self, service):
        code = '''
sc.read_h5ad("rna.h5ad")
anndata.read_h5ad("protein.h5ad")
'''
        required = service._detect_required_modalities(code, None)
        assert "rna" in required
        assert "protein" in required

    def test_explicit_plus_code_ref_deduplicates(self, service):
        code = 'sc.read_h5ad("rna.h5ad")'
        required = service._detect_required_modalities(code, "rna")
        assert required == {"rna"}


class TestMaterializeModality:
    def test_cache_hit(self, service, tmp_path):
        mat_dir = tmp_path / "materialized"
        mat_dir.mkdir()
        target = mat_dir / "rna.h5ad"
        target.write_text("cached")
        result = service._materialize_modality("rna", mat_dir)
        assert result == target

    def test_in_memory_serialize(self, service, tmp_path):
        mat_dir = tmp_path / "materialized"
        mat_dir.mkdir()
        result = service._materialize_modality("rna", mat_dir)
        assert result.exists()
        assert result.stat().st_size > 0
        # Verify it's a valid h5ad
        loaded = anndata.read_h5ad(result)
        assert loaded.shape == (2, 2)

    def test_delegates_to_provider_when_not_in_memory(self, service, mock_dm, tmp_path):
        mock_dm.list_modalities.return_value = []  # Nothing in memory
        mat_dir = tmp_path / "materialized"
        mat_dir.mkdir()

        def fake_ensure_local(name, target_dir):
            target = target_dir / f"{name}.h5ad"
            adata = anndata.AnnData(X=np.array([[9, 9]]))
            adata.write_h5ad(target)
            return target

        mock_dm.locality_provider.ensure_local.side_effect = fake_ensure_local

        result = service._materialize_modality("spatial", mat_dir)
        assert result.exists()
        loaded = anndata.read_h5ad(result)
        assert loaded.shape == (1, 2)

    def test_provider_file_not_found_propagates(self, service, mock_dm, tmp_path):
        mock_dm.list_modalities.return_value = []
        mock_dm.locality_provider.ensure_local.side_effect = FileNotFoundError("not in S3")
        mat_dir = tmp_path / "materialized"
        mat_dir.mkdir()
        with pytest.raises(FileNotFoundError, match="not in S3"):
            service._materialize_modality("missing", mat_dir)

    def test_provider_returning_existing_file_copies_not_moves(self, service, mock_dm, tmp_path):
        """Regression: provider returning a data_dir path must COPY, not MOVE."""
        mock_dm.list_modalities.return_value = []
        source = tmp_path / "source_data" / "rna.h5ad"
        source.parent.mkdir()
        adata = anndata.AnnData(X=np.array([[7, 8]]))
        adata.write_h5ad(source)

        mock_dm.locality_provider.ensure_local.return_value = source

        mat_dir = tmp_path / "materialized"
        mat_dir.mkdir()
        result = service._materialize_modality("rna", mat_dir)
        assert result.exists()
        assert source.exists(), "Original file must NOT be moved — copy only"

    def test_sanitizes_modality_name(self, service, mock_dm, tmp_path):
        mock_dm.list_modalities.return_value = ["../../etc/passwd"]
        mock_dm.get_modality.side_effect = lambda n: anndata.AnnData(X=np.array([[1]]))
        mat_dir = tmp_path / "materialized"
        mat_dir.mkdir()
        result = service._materialize_modality("../../etc/passwd", mat_dir)
        assert "etc" not in str(result.parent)
        assert result.name == "passwd.h5ad"


class TestAliasResolverNoSubstringMatch:
    """Verify that the alias resolver uses exact matching, not substring."""

    def test_rna_does_not_match_scrna(self, service, mock_dm, tmp_path):
        """Ensure 'rna.h5ad' alias doesn't redirect 'scrna.h5ad' paths."""
        mock_dm.locality_provider.list_available.return_value = ["rna", "scrna"]
        mock_dm.list_modalities.return_value = ["rna", "scrna"]
        mock_dm.get_modality.side_effect = lambda n: anndata.AnnData(
            X=np.array([[1, 2]]) if n == "rna" else np.array([[3, 4, 5]])
        )

        # Both modalities should materialize independently
        mat_dir = tmp_path / "materialized"
        mat_dir.mkdir()
        rna_path = service._materialize_modality("rna", mat_dir)
        scrna_path = service._materialize_modality("scrna", mat_dir)
        assert rna_path != scrna_path
        rna_adata = anndata.read_h5ad(rna_path)
        scrna_adata = anndata.read_h5ad(scrna_path)
        assert rna_adata.shape != scrna_adata.shape


class TestEndToEndExecution:
    def test_execute_with_materialized_modality(self, service, mock_dm):
        """End-to-end: custom code execution with locality provider materialization."""
        code = "result = int(adata.n_obs)"
        result, stats, ir = service.execute(
            code=code,
            modality_name="rna",
            persist=False,
        )
        assert result == 2
        assert stats["success"] is True

    def test_execute_no_modality_no_h5ad_refs(self, service):
        """Code without any modality references works normally."""
        code = "result = 42"
        result, stats, ir = service.execute(code=code, persist=False)
        assert result == 42
        assert stats["success"] is True
