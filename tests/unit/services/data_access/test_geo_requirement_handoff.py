"""Requirement handoff uses the real wrapper and preserves structured refusals."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import anndata as ad
import numpy as np
import pytest

from lobster.services.data_access.geo.download_execution import DownloadExecutor
from lobster.services.data_access.geo.import_requirements import (
    CandidateEvidence,
    ImportRequirement,
    ImportRequirementNotMet,
)
from lobster.services.data_access.geo_download_service import GEODownloadService
from lobster.testing.provenance_replay import RecordingDataManager

DATASET_ID = "GSE_SYNTHETIC"
MODALITY = "geo_gse_synthetic_bulk"


@pytest.fixture
def wrapper(tmp_path):
    dm = RecordingDataManager(workspace_path=tmp_path)
    dm.modalities[MODALITY] = ad.AnnData(X=np.array([[1, 2], [3, 4]]))
    geo = MagicMock()
    geo.download_dataset.return_value = "Synthetic import completed"
    with patch(
        "lobster.services.data_access.geo_download_service.GEOService", return_value=geo
    ):
        service = GEODownloadService(dm)
    entry = SimpleNamespace(
        entry_id="queue_synthetic",
        dataset_id=DATASET_ID,
        database="geo",
        recommended_strategy=None,
    )
    return service, geo, dm, entry


@pytest.mark.parametrize(
    "params", [None, {}, {"data_requirement": ""}, {"data_requirement": None}]
)
def test_empty_wrapper_handoff_keeps_legacy_kwargs_stats_and_ir(wrapper, params):
    service, geo, _, entry = wrapper
    override = {"strategy_params": params} if params is not None else None
    adata, stats, ir = service.download_dataset(entry, override)
    geo.download_dataset.assert_called_once_with(
        geo_id=DATASET_ID,
        manual_strategy_override=None,
        use_intersecting_genes_only=None,
    )
    assert adata.shape == (2, 2)
    assert stats == {
        "dataset_id": DATASET_ID,
        "database": "geo",
        "modality_name": MODALITY,
        "shape": {"n_obs": 2, "n_vars": 2},
        "strategy_used": "auto",
        "concatenation_strategy": "auto",
        "download_message": "Synthetic import completed",
    }
    assert ir.execution_context == {
        "queue_entry_id": "queue_synthetic",
        "strategy": "auto",
        "use_intersecting_genes_only": None,
        "database": "geo",
        "original_modality_name": MODALITY,
    }
    # nothing was written here, so the load cell names no file
    assert ir.parameters == {"dataset_id": DATASET_ID}


def test_active_wrapper_forwards_and_records_requirement(wrapper):
    service, geo, _, entry = wrapper
    _, stats, ir = service.download_dataset(
        entry,
        {
            "strategy_params": {
                "data_requirement": "gene_counts",
                "use_intersecting_genes_only": True,
            }
        },
    )
    geo.download_dataset.assert_called_once_with(
        geo_id=DATASET_ID,
        manual_strategy_override=None,
        use_intersecting_genes_only=True,
        data_requirement="gene_counts",
    )
    assert stats["data_requirement"] == "gene_counts"
    assert ir.execution_context["data_requirement"] == "gene_counts"


@pytest.mark.parametrize("value", ["synthetic_unknown", False, 0])
def test_wrapper_rejects_unknown_requirement_before_geo_call(wrapper, value):
    service, geo, _, entry = wrapper
    params = {"data_requirement": value}
    valid, message = service.validate_strategy_params(params)
    assert not valid and "Unknown data_requirement" in message
    with pytest.raises(ValueError, match="Unknown data_requirement"):
        service.download_dataset(entry, {"strategy_params": params})
    geo.download_dataset.assert_not_called()


def test_wrapper_propagates_same_structured_exception(wrapper):
    service, geo, dm, entry = wrapper
    error = ImportRequirementNotMet(
        DATASET_ID,
        ImportRequirement.parse("gene_counts"),
        [
            CandidateEvidence(
                "synthetic_gene_counts.tsv",
                level="gene",
                verdict="rejected",
                reason="fractional values",
            )
        ],
        problem="no_candidate",
    )
    geo.download_dataset.side_effect = error
    dm.modalities.clear()
    with pytest.raises(ImportRequirementNotMet) as caught:
        service.download_dataset(
            entry, {"strategy_params": {"data_requirement": "gene_counts"}}
        )
    assert caught.value is error
    assert dm.modalities == {}
    assert "synthetic_gene_counts.tsv" in caught.value.to_message()


@pytest.mark.parametrize("metadata_failure", [False, True])
def test_executor_preserves_structured_error_and_does_not_try_fallback(
    metadata_failure,
):
    error = ImportRequirementNotMet(
        DATASET_ID, ImportRequirement.parse("gene_counts"), [], problem="identity"
    )
    service = MagicMock()
    service.data_manager.metadata_store = {DATASET_ID: {}}
    service.data_manager._get_geo_metadata.return_value = {
        "metadata": {},
        "strategy_config": {},
    }
    first = MagicMock(side_effect=error)
    first.__name__ = "synthetic_first"
    fallback = MagicMock()
    fallback.__name__ = "synthetic_fallback"
    executor = DownloadExecutor(service)
    executor._get_processing_pipeline = MagicMock(return_value=[first, fallback])
    if metadata_failure:
        service.data_manager._get_geo_metadata.side_effect = error
    with pytest.raises(ImportRequirementNotMet) as caught:
        executor.download_with_strategy(DATASET_ID)
    assert caught.value is error
    fallback.assert_not_called()
