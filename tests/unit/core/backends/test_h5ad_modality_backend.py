"""Tests for H5ADModalityBackend — WorkspaceModalityBackend over H5AD files."""

import tempfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from lobster.core.backends.h5ad_modality_backend import (
    H5ADModalityBackend,
    _peek_h5ad_metadata,
)
from lobster.core.backends.modality_backend import (
    ModalityMetadata,
    ModalityRecord,
    ModalitySpec,
    WorkspaceModalityBackend,
)


@pytest.fixture
def data_dir():
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


@pytest.fixture
def sample_adata():
    X = csr_matrix(np.random.rand(100, 50).astype(np.float32))
    obs = pd.DataFrame(
        {"batch": ["A"] * 50 + ["B"] * 50, "n_counts": np.random.rand(100)},
        index=[f"cell_{i}" for i in range(100)],
    )
    var = pd.DataFrame(
        {"gene_symbol": [f"gene_{i}" for i in range(50)]},
        index=[f"ENSG{i:05d}" for i in range(50)],
    )
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.obsm["X_pca"] = np.random.rand(100, 10).astype(np.float32)
    adata.layers["raw"] = X.copy()
    adata.uns["method"] = "test"
    return adata


@pytest.fixture
def backend_with_data(data_dir, sample_adata):
    sample_adata.write_h5ad(data_dir / "dataset_a.h5ad")

    adata_b = sample_adata[0:30].copy()
    adata_b.write_h5ad(data_dir / "dataset_b.h5ad")

    return H5ADModalityBackend(data_dir=data_dir)


class TestH5ADModalityBackendContract:
    """Verify H5ADModalityBackend satisfies WorkspaceModalityBackend ABC."""

    def test_is_subclass(self):
        assert issubclass(H5ADModalityBackend, WorkspaceModalityBackend)

    def test_instantiates(self, data_dir):
        backend = H5ADModalityBackend(data_dir=data_dir)
        assert backend is not None


class TestListModalities:

    def test_empty_dir(self, data_dir):
        backend = H5ADModalityBackend(data_dir=data_dir)
        assert backend.list_modalities() == []

    def test_lists_h5ad_files(self, backend_with_data):
        records = backend_with_data.list_modalities()
        names = {r.name for r in records}
        assert names == {"dataset_a", "dataset_b"}

    def test_returns_modality_records(self, backend_with_data):
        records = backend_with_data.list_modalities()
        for r in records:
            assert isinstance(r, ModalityRecord)
            assert r.n_obs > 0
            assert r.n_vars > 0
            assert r.storage_size_bytes > 0

    def test_nonexistent_dir(self):
        backend = H5ADModalityBackend(data_dir=Path("/nonexistent"))
        assert backend.list_modalities() == []


class TestGetMetadata:

    def test_returns_metadata(self, backend_with_data):
        meta = backend_with_data.get_metadata("dataset_a")
        assert isinstance(meta, ModalityMetadata)
        assert meta.name == "dataset_a"
        assert meta.n_obs == 100
        assert meta.n_vars == 50
        assert "batch" in meta.obs_columns
        assert "gene_symbol" in meta.var_columns
        assert "X_pca" in meta.obsm_keys
        assert "raw" in meta.layers
        assert "method" in meta.uns_keys

    def test_missing_modality_raises(self, backend_with_data):
        with pytest.raises(KeyError, match="not found"):
            backend_with_data.get_metadata("nonexistent")


class TestQueryObs:

    def test_returns_dataframe(self, backend_with_data):
        df = backend_with_data.query_obs("dataset_a")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 100

    def test_column_selection(self, backend_with_data):
        df = backend_with_data.query_obs("dataset_a", columns=["batch"])
        assert list(df.columns) == ["batch"]

    def test_filter(self, backend_with_data):
        df = backend_with_data.query_obs("dataset_a", filters={"batch": "A"})
        assert len(df) == 50

    def test_pagination(self, backend_with_data):
        df = backend_with_data.query_obs("dataset_a", limit=10, offset=5)
        assert len(df) == 10

    def test_missing_raises(self, backend_with_data):
        with pytest.raises(KeyError):
            backend_with_data.query_obs("nonexistent")


class TestMaterializeAnndata:

    def test_returns_anndata(self, backend_with_data):
        adata = backend_with_data.materialize_anndata("dataset_a")
        assert isinstance(adata, ad.AnnData)
        assert adata.n_obs == 100
        assert adata.n_vars == 50

    def test_not_backed(self, backend_with_data):
        adata = backend_with_data.materialize_anndata("dataset_a")
        assert not getattr(adata, "isbacked", False)

    def test_obs_filter(self, backend_with_data):
        adata = backend_with_data.materialize_anndata(
            "dataset_a", obs_filter={"batch": "B"}
        )
        assert adata.n_obs == 50

    def test_feature_subset(self, backend_with_data):
        adata = backend_with_data.materialize_anndata(
            "dataset_a", features=["ENSG00000", "ENSG00001"]
        )
        assert adata.n_vars == 2

    def test_missing_raises(self, backend_with_data):
        with pytest.raises(KeyError):
            backend_with_data.materialize_anndata("nonexistent")


class TestIngestAnndata:

    def test_ingest_creates_file(self, data_dir, sample_adata):
        backend = H5ADModalityBackend(data_dir=data_dir)
        backend.ingest_anndata("new_mod", sample_adata)
        assert (data_dir / "new_mod.h5ad").exists()

    def test_ingest_roundtrip(self, data_dir, sample_adata):
        backend = H5ADModalityBackend(data_dir=data_dir)
        backend.ingest_anndata("roundtrip", sample_adata)
        loaded = backend.materialize_anndata("roundtrip")
        assert loaded.n_obs == sample_adata.n_obs
        assert loaded.n_vars == sample_adata.n_vars


class TestExportH5AD:

    def test_exports_to_target(self, backend_with_data, data_dir):
        target = data_dir / "export" / "exported.h5ad"
        result = backend_with_data.export_h5ad("dataset_a", target)
        assert result.exists()
        assert result == target

    def test_export_missing_raises(self, backend_with_data, data_dir):
        with pytest.raises(KeyError):
            backend_with_data.export_h5ad("nonexistent", data_dir / "out.h5ad")


class TestRemoveAndExists:

    def test_exists_true(self, backend_with_data):
        assert backend_with_data.exists("dataset_a") is True

    def test_exists_false(self, backend_with_data):
        assert backend_with_data.exists("nonexistent") is False

    def test_remove_deletes_file(self, backend_with_data, data_dir):
        backend_with_data.remove("dataset_a")
        assert not (data_dir / "dataset_a.h5ad").exists()
        assert not backend_with_data.exists("dataset_a")

    def test_remove_missing_raises(self, backend_with_data):
        with pytest.raises(KeyError):
            backend_with_data.remove("nonexistent")


class TestPeekMetadata:

    def test_peek_reads_shape(self, data_dir, sample_adata):
        path = data_dir / "test.h5ad"
        sample_adata.write_h5ad(path)
        meta = _peek_h5ad_metadata(path)
        assert meta["n_obs"] == 100
        assert meta["n_vars"] == 50

    def test_peek_reads_columns(self, data_dir, sample_adata):
        path = data_dir / "test.h5ad"
        sample_adata.write_h5ad(path)
        meta = _peek_h5ad_metadata(path)
        assert "batch" in meta["obs_columns"]
        assert "gene_symbol" in meta["var_columns"]

    def test_peek_reads_keys(self, data_dir, sample_adata):
        path = data_dir / "test.h5ad"
        sample_adata.write_h5ad(path)
        meta = _peek_h5ad_metadata(path)
        assert "X_pca" in meta["obsm_keys"]
        assert "raw" in meta["layers"]
        assert "method" in meta["uns_keys"]

    def test_peek_handles_missing_file(self, data_dir):
        meta = _peek_h5ad_metadata(data_dir / "missing.h5ad")
        assert meta["n_obs"] == 0
        assert meta["n_vars"] == 0


class TestAtomicWrites:
    """Fix #6: ingest_anndata uses temp file + os.replace for atomicity."""

    def test_no_partial_file_on_error(self, data_dir):
        """If write fails mid-way, target file must not exist (or be old version)."""
        from unittest.mock import patch

        backend = H5ADModalityBackend(data_dir=data_dir)
        adata = ad.AnnData(
            X=np.random.rand(10, 5).astype(np.float32),
            obs=pd.DataFrame(index=[f"c{i}" for i in range(10)]),
            var=pd.DataFrame(index=[f"g{i}" for i in range(5)]),
        )

        with patch(
            "lobster.core.backends.h5ad_backend.H5ADBackend"
        ) as MockH5AD:
            MockH5AD.return_value.save.side_effect = IOError("disk full")
            with pytest.raises(IOError, match="disk full"):
                backend.ingest_anndata("fail_mod", adata)

        assert not (data_dir / "fail_mod.h5ad").exists()
        tmp_files = list(data_dir.glob("*.h5ad.tmp"))
        assert len(tmp_files) == 0

    def test_atomic_overwrite_preserves_old(self, data_dir, sample_adata):
        """Overwriting existing file is atomic — old file untouched on failure."""
        from unittest.mock import patch

        backend = H5ADModalityBackend(data_dir=data_dir)
        backend.ingest_anndata("overwrite_me", sample_adata)
        original_size = (data_dir / "overwrite_me.h5ad").stat().st_size

        with patch(
            "lobster.core.backends.h5ad_backend.H5ADBackend"
        ) as MockH5AD:
            MockH5AD.return_value.save.side_effect = IOError("disk full")
            with pytest.raises(IOError):
                backend.ingest_anndata("overwrite_me", sample_adata)

        assert (data_dir / "overwrite_me.h5ad").stat().st_size == original_size


class TestPathTraversalSecurity:
    """P0: _h5ad_path must reject directory traversal attempts."""

    def test_rejects_parent_traversal(self, data_dir):
        backend = H5ADModalityBackend(data_dir=data_dir)
        with pytest.raises(ValueError, match="Invalid"):
            backend._h5ad_path("../etc/passwd")

    def test_rejects_dot(self, data_dir):
        backend = H5ADModalityBackend(data_dir=data_dir)
        with pytest.raises(ValueError, match="Invalid"):
            backend._h5ad_path(".")

    def test_rejects_dotdot(self, data_dir):
        backend = H5ADModalityBackend(data_dir=data_dir)
        with pytest.raises(ValueError, match="Invalid"):
            backend._h5ad_path("..")

    def test_rejects_slash_in_name(self, data_dir):
        backend = H5ADModalityBackend(data_dir=data_dir)
        with pytest.raises(ValueError, match="Invalid"):
            backend._h5ad_path("sub/dir")

    def test_rejects_empty_name(self, data_dir):
        backend = H5ADModalityBackend(data_dir=data_dir)
        with pytest.raises(ValueError, match="Invalid"):
            backend._h5ad_path("")

    def test_accepts_valid_name(self, data_dir):
        backend = H5ADModalityBackend(data_dir=data_dir)
        path = backend._h5ad_path("valid_dataset")
        assert path == (data_dir / "valid_dataset.h5ad").resolve()
