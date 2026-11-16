"""Stored GEO samples are concatenated into one dataset and the step is logged with its IR."""

from types import SimpleNamespace

import anndata as ad
import numpy as np
import pytest

from lobster.core.runtime.data_manager import DataManagerV2
from lobster.services.data_access.geo.concatenation import SampleConcatenator

pytestmark = pytest.mark.unit


def _stored(tmp_path):
    manager = DataManagerV2(workspace_path=tmp_path / "ws", auto_scan=False)
    names = []
    for i in (1, 2):
        adata = ad.AnnData(np.ones((5, 4), dtype="float32"))
        adata.obs_names = [f"s{i}c{j}" for j in range(5)]
        adata.var_names = [f"g{j}" for j in range(4)]
        manager.modalities[f"geo_sample_gsm{i}"] = adata
        names.append(f"geo_sample_gsm{i}")
    return manager, names


def test_two_stored_samples_give_one_merged_dataset(tmp_path):
    manager, names = _stored(tmp_path)
    merged = SampleConcatenator(
        SimpleNamespace(data_manager=manager)
    )._concatenate_stored_samples("GSE1", names, False)
    assert merged is not None and merged.shape == (10, 4)


def test_the_concatenation_is_logged_with_its_ir(tmp_path):
    manager, names = _stored(tmp_path)
    SampleConcatenator(
        SimpleNamespace(data_manager=manager)
    )._concatenate_stored_samples("GSE1", names, False)
    (step,) = [
        a
        for a in manager.provenance.activities
        if a["type"] == "concatenate_geo_samples"
    ]
    assert step.get("ir") is not None
