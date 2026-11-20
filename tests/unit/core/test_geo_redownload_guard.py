"""A resumed session asked to download a GEO series it already loaded.

The earlier import's file (data/geo_<id>_<adapter>_raw.h5ad) is loaded again at
start, under its file name. Before, the download ran, registered a second
modality, logged its load, and only then had its save refused. Now the loaded
file is found first, by its recorded path: the reply names the modality, nothing
is downloaded, and the session's modalities and record stay as they were.
"""

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from lobster.core.runtime.data_manager import DataManagerV2
from lobster.services.data_access.geo.download_execution import DownloadExecutor
from lobster.services.data_access.geo.import_requirements import (
    ImportRequirementNotMet,
)
from lobster.services.data_access.geo_service import GEOService

FILE = "geo_gse1_transcriptomics_bulk_raw.h5ad"


def _resumed(tmp_path):
    data = tmp_path / "ws" / "data"
    data.mkdir(parents=True)
    rng = np.random.default_rng(0)
    ad.AnnData(
        X=rng.poisson(5, (6, 20)).astype(np.float32),
        obs=pd.DataFrame(index=[f"GSM{i}" for i in range(6)]),
        var=pd.DataFrame(index=[f"G{i}" for i in range(20)]),
    ).write_h5ad(data / FILE)
    return DataManagerV2(workspace_path=tmp_path / "ws")


@pytest.fixture
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("a download was started")

    monkeypatch.setattr(DownloadExecutor, "download_with_strategy", refuse)
    monkeypatch.setattr(GEOService, "fetch_metadata_only", refuse)


def _state(dm):
    return sorted(dm.list_modalities()), len(dm.provenance.activities)


@pytest.mark.parametrize("renamed", [False, True], ids=["as_file", "renamed"])
def test_a_resumed_re_download_names_the_loaded_modality(tmp_path, no_network, renamed):
    dm = _resumed(tmp_path)
    loaded_as = "geo_gse1_transcriptomics_bulk_raw"
    if renamed:  # the same file, loaded again under another name
        loaded_as = "my_series"
        dm.load_modality(
            loaded_as, str(dm.data_dir / FILE), adapter="transcriptomics_bulk"
        )
    before = _state(dm)
    service = GEOService(data_manager=dm, cache_dir=str(tmp_path / "cache"))

    reply = service.download_dataset("GSE1")

    assert reply == (
        f"Dataset GSE1 is already loaded as modality '{loaded_as}' (from data/{FILE}); "
        "nothing was downloaded. Use that modality, or start a new workspace to "
        "import the series again."
    ), reply
    assert _state(dm) == before


def test_a_resumed_required_import_is_refused_before_a_download(tmp_path, no_network):
    dm = _resumed(tmp_path)
    dm.load_modality(
        "my_series", str(dm.data_dir / FILE), adapter="transcriptomics_bulk"
    )
    del dm.modalities["geo_gse1_transcriptomics_bulk_raw"]  # only the renamed one
    before = _state(dm)
    service = GEOService(data_manager=dm, cache_dir=str(tmp_path / "cache"))

    with pytest.raises(ImportRequirementNotMet) as refused:
        service.download_dataset("GSE1", data_requirement="gene_counts")

    assert "already loaded as modality 'my_series'" in str(refused.value)
    assert "nothing was downloaded" in str(refused.value)
    assert _state(dm) == before
