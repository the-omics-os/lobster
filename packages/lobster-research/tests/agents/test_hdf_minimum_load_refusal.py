"""The real service/tool must not flatten the excluded pandas-HDF refusal."""

import hashlib
import pickle
from importlib.util import find_spec
from unittest.mock import Mock

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pytest

from lobster.agents.data_expert.data_expert import data_expert
from lobster.core.exceptions import HdfReadRefusal
from lobster.core.runtime.data_manager import DataManagerV2
from lobster.services.data_management.modality_management_service import (
    ModalityManagementService,
)
from lobster.testing.provenance_replay import factory_tools

MODULE = "lobster.agents.data_expert.data_expert"


@pytest.mark.parametrize("boundary", ["service", "tool"])
def test_real_unsupported_hdf_file_terminal_at_actual_caller(
    monkeypatch, tmp_path, caplog, record_property, boundary
):
    # The fixture must also be writable in the isolated package CI environment
    # without optional PyTables. This benign plain HDF is not an AnnData file;
    # the actual native reader must still be attempted before the fixed refusal.
    path = tmp_path / "benign_unsupported.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("fixture_values", data=np.array([1, 2], dtype=np.int32))
    record_property("tables_available", find_spec("tables") is not None)
    record_property("fixture_kind", "plain_hdf_not_anndata")
    record_property("fixture_sha256", hashlib.sha256(path.read_bytes()).hexdigest())
    record_property("fixture_bytes", path.stat().st_size)
    dm = DataManagerV2(workspace_path=tmp_path, auto_scan=False)
    old = ad.AnnData(np.ones((2, 2), dtype=np.float32))
    dm.modalities["unrelated"] = old
    dm._modality_sources["unrelated"] = tmp_path / "original.h5ad"
    dm._modality_dirty.add("unrelated")
    before = (
        dict(dm._modality_sources),
        set(dm._modality_dirty),
        dict(dm._modality_spilled),
    )
    service = ModalityManagementService(dm)
    tools = factory_tools(data_expert, MODULE, dm) if boundary == "tool" else None
    dm.log_tool_usage = Mock()
    dm.on_modality_loaded = Mock()
    dm.get_quality_metrics = Mock()
    candidate_calls = []

    def forbidden(*args, **kwargs):
        candidate_calls.append(True)
        raise AssertionError("prohibited_reader")

    monkeypatch.setattr(pd, "read_hdf", forbidden)
    monkeypatch.setattr(pd, "HDFStore", forbidden)
    monkeypatch.setattr(pickle, "loads", forbidden)
    native_read = Mock(wraps=ad.read_h5ad)
    monkeypatch.setattr(ad, "read_h5ad", native_read)
    caplog.clear()
    with pytest.raises(HdfReadRefusal) as caught:
        if boundary == "tool":
            tools["load_modality"].invoke(
                {
                    "modality_name": "denied",
                    "file_path": str(path),
                    "adapter": "transcriptomics_bulk",
                    "dataset_type": "custom",
                }
            )
        else:
            service.load_modality("denied", str(path), "transcriptomics_bulk")
    native_read.assert_called_once()
    assert str(native_read.call_args.args[0]) == str(path)
    assert candidate_calls == []
    assert caught.value.reason == "pandas_hdf_not_admitted"
    assert caught.value.__context__ is None
    assert caught.value.__cause__ is None
    assert "denied" not in dm.modalities
    assert dm.modalities["unrelated"] is old
    assert (
        dict(dm._modality_sources),
        set(dm._modality_dirty),
        dict(dm._modality_spilled),
    ) == before
    dm.log_tool_usage.assert_not_called()
    dm.on_modality_loaded.assert_not_called()
    dm.get_quality_metrics.assert_not_called()
    for record in caplog.records:
        if record.levelno >= 30:
            assert record.exc_info is None
            assert str(path) not in record.getMessage()
            assert "Error loading modality" not in record.getMessage()


@pytest.mark.parametrize("boundary", ["service", "tool"])
def test_exact_closed_identity_at_caller(monkeypatch, tmp_path, caplog, boundary):
    path = tmp_path / "counts.h5"
    path.touch()
    dm = DataManagerV2(workspace_path=tmp_path, auto_scan=False)
    error = HdfReadRefusal()
    read = Mock(side_effect=error)
    tools = factory_tools(data_expert, MODULE, dm) if boundary == "tool" else None
    monkeypatch.setattr(dm, "load_modality", read)
    caplog.clear()
    with pytest.raises(HdfReadRefusal) as caught:
        if boundary == "tool":
            tools["load_modality"].invoke(
                {
                    "modality_name": "denied",
                    "file_path": str(path),
                    "adapter": "transcriptomics_bulk",
                }
            )
        else:
            ModalityManagementService(dm).load_modality(
                "denied", str(path), "transcriptomics_bulk"
            )
    assert caught.value is error
    read.assert_called_once()
    assert not [record for record in caplog.records if record.levelno >= 40]
