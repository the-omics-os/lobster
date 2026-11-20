"""The exported normalization cell runs and gives the stored result.

With ``log_transform=True`` the cell read ``log_transform=true``: a NameError,
so the exported notebook failed at its first proteomics step.
"""

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from lobster.services.quality.proteomics_preprocessing_service import (
    ProteomicsPreprocessingService,
)
from lobster.testing.provenance_replay import replay_ir


def _intensities():
    rng = np.random.default_rng(0)
    return ad.AnnData(
        X=rng.lognormal(10, 1, size=(6, 20)),
        obs=pd.DataFrame(index=[f"s{i}" for i in range(6)]),
        var=pd.DataFrame(index=[f"P{i:03d}" for i in range(20)]),
    )


@pytest.mark.parametrize("log_transform", [True, False])
def test_the_exported_normalization_reproduces_the_result(log_transform):
    adata = _intensities()
    result, _, ir = ProteomicsPreprocessingService().normalize_intensities(
        adata.copy(), method="median", log_transform=log_transform
    )

    # the notebook's parameters cell defines the names the cell reads
    scope = replay_ir(
        ir, adata=adata.copy(), method="median", log_transform=log_transform
    )

    np.testing.assert_allclose(scope["adata_norm"].X, result.X)
