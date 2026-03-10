import anndata as ad
import numpy as np
import pandas as pd

from lobster.core.data_manager_v2 import DataManagerV2


def test_scan_hides_autosave_h5ad_files_from_available_datasets(tmp_path):
    dm = DataManagerV2(workspace_path=tmp_path, auto_scan=False)
    data_dir = dm.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    adata = ad.AnnData(
        X=np.random.rand(3, 2).astype(np.float32),
        obs=pd.DataFrame(index=["s1", "s2", "s3"]),
        var=pd.DataFrame(index=["g1", "g2"]),
    )
    adata.write_h5ad(data_dir / "analysis_autosave.h5ad")

    dm._scan_workspace()

    assert "analysis_autosave" not in dm.available_datasets
