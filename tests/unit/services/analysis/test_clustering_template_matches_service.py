"""The exported clustering code leaves the observation columns in the service's order."""

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scanpy as sc
from scipy import sparse

from lobster.services.analysis.clustering_service import ClusteringService
from lobster.testing.provenance_replay import replay_ir


@pytest.fixture
def normalized():
    rng = np.random.default_rng(0)
    x = rng.poisson(1.0, size=(240, 600)).astype(np.float32)
    x[:120, :60] += 4  # two groups, so clustering has something to find
    adata = ad.AnnData(
        sparse.csr_matrix(x),
        obs=pd.DataFrame(index=[f"c{i}" for i in range(240)]),
        var=pd.DataFrame(index=[f"G{i}" for i in range(600)]),
    )
    adata.raw = adata.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    return adata


def replayed(ir, adata):
    scope = {"adata": adata}
    for statement in ir.imports:
        exec(compile(statement, "<ir-import>", "exec"), scope)
    for helper in ir.helper_code or []:
        exec(compile(helper, "<ir-helper>", "exec"), scope)
    return replay_ir(ir, **scope)["adata"]


@pytest.mark.parametrize(
    "kwargs",
    [{"resolution": 1.0}, {"resolutions": [0.5, 1.0]}, {}],
    ids=["single", "multi", "default"],
)
def test_template_obs_columns_match_the_service_in_order_and_values(normalized, kwargs):
    out, _, ir = ClusteringService().cluster_and_visualize(
        normalized.copy(), **kwargs, demo_mode=True, n_features=200
    )
    notebook = replayed(ir, normalized.copy())
    assert list(notebook.obs.columns) == list(out.obs.columns)
    for column in out.obs.columns:
        assert notebook.obs[column].astype(str).tolist() == (
            out.obs[column].astype(str).tolist()
        ), column
