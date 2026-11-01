"""X-alias round trips must not manufacture layers or refuse a second save."""

from types import SimpleNamespace

import anndata as ad
import h5py
import numpy as np
import pytest
from scipy import sparse

from lobster.core.backends.h5ad_backend import H5ADBackend

pytestmark = pytest.mark.no_auto_config


def _data(sparse_x=False):
    x = np.arange(6, dtype=np.float32).reshape(3, 2)
    data = ad.AnnData(sparse.csr_matrix(x) if sparse_x else x)
    data.obs_names = ["cell_a", "cell_b", "cell_c"]
    data.var_names = ["gene_a", "gene_b"]
    data.layers["counts"] = data.X.copy()
    data.obsm["X/pca"] = np.arange(6, dtype=np.float32).reshape(3, 2)
    data.varm["PC/loadings"] = np.arange(4, dtype=np.float32).reshape(2, 2)
    data.uns["result"] = np.array([7, 8])
    data.obs["cluster"] = ["a", "a", "b"]
    return data


def _dense(value):
    return value.toarray() if sparse.issparse(value) else value


def _assert_payload(data, original):
    np.testing.assert_array_equal(_dense(data.X), _dense(original.X))
    np.testing.assert_array_equal(_dense(data.layers["counts"]), _dense(original.X))
    np.testing.assert_array_equal(data.obsm["X__pca"], original.obsm["X/pca"])
    np.testing.assert_array_equal(
        data.varm["PC__loadings"], original.varm["PC/loadings"]
    )
    np.testing.assert_array_equal(data.uns["result"], original.uns["result"])
    assert data.obs_names.tolist() == original.obs_names.tolist()
    assert data.var_names.tolist() == original.var_names.tolist()
    assert data.obs["cluster"].tolist() == original.obs["cluster"].tolist()


@pytest.mark.parametrize("sparse_x", [False, True])
def test_gen1_reload_gen2_preserves_all_payloads(tmp_path, sparse_x):
    """The same second-save contract also runs against actual 1.2.1 source."""
    original = _data(sparse_x)
    before_keys = list(original.layers)
    backend = H5ADBackend(base_path=tmp_path)
    backend.save(original, "gen1.h5ad")
    first = backend.load("gen1.h5ad")
    backend.save(first, "gen2.h5ad")
    second = backend.load("gen2.h5ad")
    _assert_payload(first, original)
    _assert_payload(second, original)
    assert list(original.layers) == before_keys
    assert list(original.obsm) == ["X/pca"]
    assert list(original.varm) == ["PC/loadings"]


def test_x_alias_is_not_manufactured_as_a_disk_layer(tmp_path):
    original = _data()
    # Older supported anndata has no virtual None alias; the same no-bloat
    # output contract still applies. Actual admitted dependency is in receipts.
    assert None not in original.obsm and None not in original.varm
    target = tmp_path / "fresh.h5ad"
    H5ADBackend().save(original, target)
    with h5py.File(target, "r") as disk:
        assert set(disk["layers"]) == {"counts"}
        assert set(disk["obsm"]) == {"X__pca"}
        assert set(disk["varm"]) == {"PC__loadings"}
    _assert_payload(ad.read_h5ad(target), original)


@pytest.mark.parametrize("equal_x", [False, True])
def test_legacy_string_none_layer_is_preserved_on_resave(tmp_path, equal_x):
    original = _data()
    legacy = _dense(original.X).copy()
    if not equal_x:
        legacy += 100  # Must not overwrite actual user data with the X alias.
    original.layers["None"] = legacy.copy()
    target = tmp_path / "legacy.h5ad"
    # Native writer materializes a legacy disk layer, not a fake backend seam.
    # Give native I/O already-safe keys; slash conversion is a backend concern.
    native = original.copy()
    native.obsm = {"X__pca": original.obsm["X/pca"].copy()}
    native.varm = {"PC__loadings": original.varm["PC/loadings"].copy()}
    native.write_h5ad(target)
    backend = H5ADBackend(base_path=tmp_path)
    loaded = backend.load(target)
    backend.save(loaded, "resaved.h5ad")
    restored = backend.load("resaved.h5ad")
    np.testing.assert_array_equal(restored.layers["None"], legacy)
    assert "None" in loaded.layers and "None" in original.layers
    _assert_payload(restored, original)
    with h5py.File(tmp_path / "resaved.h5ad", "r") as disk:
        assert set(disk["layers"]) == {"None", "counts"}


@pytest.mark.parametrize("slot", ["obsm", "varm"])
def test_nonlayer_none_key_collision_is_not_dropped(tmp_path, slot):
    data = _data()
    values = getattr(data, slot)
    shape = (data.n_obs if slot == "obsm" else data.n_vars, 1)
    values[None] = np.ones(shape, dtype=np.float32)
    values["None"] = np.full(shape, 2, dtype=np.float32)
    before_none = values[None].copy()
    before_string = values["None"].copy()
    target = tmp_path / "refused.h5ad"
    with pytest.raises(ValueError, match="H5AD key collision after key sanitization"):
        H5ADBackend().save(data, target)
    assert not target.exists()
    np.testing.assert_array_equal(values[None], before_none)
    np.testing.assert_array_equal(values["None"], before_string)


def _nonalias_mapping_data():
    # Simulate pre-0.13's ordinary layer mapping without installing/downgrading
    # anndata. All sanitation logic is real; only mapping semantics differ.
    base = _data()
    return SimpleNamespace(
        X=base.X,
        layers={"counts": base.X.copy()},
        obsm=dict(base.obsm),
        varm=dict(base.varm),
        uns=dict(base.uns),
        obs=base.obs.copy(),
        var=base.var.copy(),
    )


@pytest.mark.parametrize("equal_x", [False, True])
def test_nonalias_none_layer_payload_is_preserved_without_mapping_mutation(equal_x):
    data = _nonalias_mapping_data()
    payload = data.X.copy()  # Even equal values are NOT an identity alias.
    if not equal_x:
        payload += 100
    data.layers[None] = payload
    original_mapping = data.layers
    original_keys = list(original_mapping)
    original_x = data.X
    snapshot = payload.copy()
    assert original_mapping[None] is not data.X

    result = H5ADBackend.sanitize_anndata(data)

    assert result is data
    assert set(result.layers) == {"counts", "None"}
    np.testing.assert_array_equal(result.layers["None"], snapshot)
    np.testing.assert_array_equal(result.layers["counts"], original_x)
    assert data.X is original_x
    assert list(original_mapping) == original_keys
    assert original_mapping[None] is payload
    np.testing.assert_array_equal(original_mapping[None], snapshot)


@pytest.mark.parametrize("reverse", [False, True])
def test_nonalias_none_string_collision_refuses_without_mapping_mutation(reverse):
    data = _nonalias_mapping_data()
    payloads = {None: data.X.copy() + 100, "None": data.X.copy() + 200}
    order = ["None", None] if reverse else [None, "None"]
    data.layers = {key: payloads[key] for key in order}
    original_mapping = data.layers
    original_keys = list(original_mapping)
    original_x = data.X
    snapshots = {key: value.copy() for key, value in original_mapping.items()}
    assert original_mapping[None] is not data.X

    with pytest.raises(ValueError, match="H5AD key collision after key sanitization"):
        H5ADBackend.sanitize_anndata(data)

    assert data.layers is original_mapping
    assert list(data.layers) == original_keys
    assert data.X is original_x
    for key, expected in snapshots.items():
        assert data.layers[key] is payloads[key]
        np.testing.assert_array_equal(data.layers[key], expected)
