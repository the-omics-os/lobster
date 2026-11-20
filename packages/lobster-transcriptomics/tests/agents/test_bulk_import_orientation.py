"""import_bulk_counts must put samples in obs for a CSV in either layout."""

import numpy as np
import pandas as pd
import pytest

from lobster.agents.transcriptomics.transcriptomics_expert import (
    transcriptomics_expert,
)
from lobster.core.runtime.data_manager import DataManagerV2
from lobster.testing.provenance_replay import factory_tools

MODULE = "lobster.agents.transcriptomics.transcriptomics_expert"


@pytest.mark.parametrize("genes_as_rows", [True, False])
def test_csv_import_puts_samples_in_obs(tmp_path, genes_as_rows):
    samples = [f"S{i}" for i in range(1, 7)]
    genes = [f"GENE{i:03d}" for i in range(50)]
    counts = pd.DataFrame(
        np.arange(len(genes) * len(samples)).reshape(len(genes), len(samples)),
        index=genes,
        columns=samples,
    )
    path = tmp_path / "counts.csv"
    (counts if genes_as_rows else counts.T).to_csv(path)
    dm = DataManagerV2(workspace_path=tmp_path / "ws")
    tools = factory_tools(transcriptomics_expert, MODULE, dm)

    tools["import_bulk_counts"].invoke(
        {"modality_name": "bulk", "file_path": str(path)}
    )

    adata = dm.get_modality("bulk")
    assert list(adata.obs_names) == samples
    assert list(adata.var_names) == genes
    np.testing.assert_array_equal(adata.X, counts.T.to_numpy())
