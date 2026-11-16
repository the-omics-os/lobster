"""Only requested GEO count imports reuse the service's one canonical record."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import anndata
import numpy as np
import pytest

from lobster.core.schemas.download_queue import DownloadStatus
from lobster.services.data_access.geo.import_requirements import ImportRequirementNotMet
from lobster.tools.download_orchestrator import DownloadOrchestrator

CANONICAL = "geo_gse900000000001_transcriptomics_bulk"
GSE = "GSE900000000001"
REQUIREMENT = {"strategy_params": {"data_requirement": "gene_counts"}}


class SyntheticService:
    def __init__(self, data, stats, ir):
        self.data, self.stats, self.ir = data, stats, ir
        self.calls = []

    def supported_databases(self):
        return ["geo"]

    def validate_strategy(self, strategy):
        return None

    def download_dataset(self, entry, strategy_override=None):
        self.calls.append(strategy_override)
        return self.data, self.stats, self.ir


@pytest.fixture
def setup():
    manager = MagicMock()
    data = anndata.AnnData(np.array([[1, 2, 3], [2, 4, 5]], dtype=np.float32))
    data.uns["geo_import"] = {
        "data_requirement": "gene_counts",
        "accepted_file": "synthetic_gene.tsv",
    }
    manager.modalities = {CANONICAL: data}
    manager.get_modality.side_effect = manager.modalities.__getitem__
    manager.download_queue.get_entry.return_value = SimpleNamespace(
        database="geo",
        dataset_id=GSE,
        status=DownloadStatus.PENDING,
        modality_name=None,
    )
    ir = object()
    service = SyntheticService(
        data, {"modality_name": CANONICAL, "data_requirement": "gene_counts"}, ir
    )
    with patch.object(DownloadOrchestrator, "_register_default_services"):
        orchestrator = DownloadOrchestrator(manager)
    orchestrator.register_service(service)
    return manager, orchestrator, service, data, ir


@pytest.mark.parametrize("custom_alias", [None, "synthetic_requested_alias"])
@pytest.mark.parametrize("return_ir", [False, True])
def test_active_keeps_one_record_and_completes_with_canonical_name(
    setup, custom_alias, return_ir
):
    manager, orchestrator, service, data, ir = setup
    manager.download_queue.get_entry.return_value.modality_name = custom_alias
    result = orchestrator.execute_download(
        "synthetic_entry", strategy_override=REQUIREMENT, return_ir=return_ir
    )
    assert result[0] == CANONICAL
    assert manager.modalities == {CANONICAL: data}
    assert service.calls == [REQUIREMENT]
    manager.download_queue.update_status.assert_called_with(
        "synthetic_entry", DownloadStatus.COMPLETED, modality_name=CANONICAL
    )
    if return_ir:
        assert result[2] is ir
        manager.log_tool_usage.assert_not_called()
    else:
        manager.log_tool_usage.assert_called_once()
        assert manager.log_tool_usage.call_args.kwargs["ir"] is ir


@pytest.mark.parametrize(
    "override", [None, {}, {"strategy_params": {"data_requirement": ""}}]
)
@pytest.mark.parametrize("custom_alias", [None, "synthetic_requested_alias"])
def test_inactive_literal_legacy_alias_registration_and_provenance_preserved(
    setup, override, custom_alias, caplog
):
    manager, orchestrator, service, data, ir = setup
    manager.download_queue.get_entry.return_value.modality_name = custom_alias
    expected = custom_alias or "geo_" + GSE
    result = orchestrator.execute_download(
        "synthetic_entry", strategy_override=override
    )
    assert result == (expected, service.stats)
    assert list(manager.modalities) == [CANONICAL, expected]
    assert manager.modalities[expected] is data
    assert service.calls == [override]
    assert (
        f"Stored downloaded data as modality '{expected}' (2 obs × 3 vars)"
        in caplog.text
    )
    manager.log_tool_usage.assert_called_once_with(
        "download_orchestrator.execute_download",
        {
            "entry_id": "synthetic_entry",
            "database": "geo",
            "dataset_id": GSE,
            "strategy_override": override or {},
            "service": "SyntheticService",
        },
        service.stats,
        ir=ir,
    )
    manager.download_queue.update_status.assert_called_with(
        "synthetic_entry", DownloadStatus.COMPLETED, modality_name=expected
    )


@pytest.mark.parametrize(
    "bad",
    [
        "missing_stat_marker",
        "wrong_stat_marker",
        "missing_uns",
        "malformed_uns",
        "wrong_uns_marker",
        "missing_name",
        "missing_stored",
        "wrong_identity",
    ],
)
def test_active_bad_consistency_refuses_instead_of_falling_through_to_alias(setup, bad):
    manager, orchestrator, service, data, ir = setup
    if bad == "missing_stat_marker":
        service.stats.pop("data_requirement")
    elif bad == "wrong_stat_marker":
        service.stats["data_requirement"] = ""
    elif bad == "missing_uns":
        data.uns.pop("geo_import")
    elif bad == "malformed_uns":
        data.uns["geo_import"] = "synthetic_malformed"
    elif bad == "wrong_uns_marker":
        data.uns["geo_import"]["data_requirement"] = ""
    elif bad == "missing_name":
        service.stats.pop("modality_name")
    elif bad == "missing_stored":
        manager.modalities.clear()
    else:
        manager.modalities[CANONICAL] = data.copy()
    before = dict(manager.modalities)
    with pytest.raises(ImportRequirementNotMet) as caught:
        orchestrator.execute_download("synthetic_entry", strategy_override=REQUIREMENT)
    assert caught.value.problem == "identity"
    assert list(manager.modalities) == list(before)
    assert all(manager.modalities[key] is before[key] for key in before)
    assert "geo_" + GSE not in manager.modalities
    assert (
        manager.download_queue.update_status.call_args.args[1] == DownloadStatus.FAILED
    )
    manager.log_tool_usage.assert_not_called()


def test_real_active_queue_wrapper_keeps_one_engine_record_and_canonical_provenance(
    tmp_path,
):
    import pandas as pd

    from lobster.core.runtime.data_manager import DataManagerV2
    from lobster.core.schemas.download_queue import DownloadQueueEntry
    from lobster.services.data_access.geo.parser import ParseResult
    from lobster.services.data_access.geo_download_service import GEODownloadService

    manager = DataManagerV2(workspace_path=tmp_path)
    a, b = "GSM900000000001", "GSM900000000002"
    metadata = {
        "samples": {a: {"title": "Synthetic A"}, b: {"title": "Synthetic B"}},
        "supplementary_file": ["https://example.test/synthetic_gene.tsv"],
    }
    manager.metadata_store[GSE] = {"metadata": metadata, "strategy_config": {}}
    entry = DownloadQueueEntry(
        entry_id="synthetic_entry",
        dataset_id=GSE,
        database="geo",
        modality_name="synthetic_requested_alias",
    )
    manager.download_queue.add_entry(entry)
    wrapper = GEODownloadService(manager)
    wrapper.geo_service._check_platform_compatibility = MagicMock(
        return_value=(True, "synthetic")
    )
    matrix = pd.DataFrame(
        [[1, 2], [3, 5], [2, 4]],
        index=["synthetic_a", "synthetic_b", "synthetic_c"],
        columns=[a, b],
    )
    wrapper.geo_service._download_and_parse_result = MagicMock(
        side_effect=lambda *_: ParseResult(matrix.copy(), original_tokens_valid=True)
    )
    manager.log_tool_usage = MagicMock(wraps=manager.log_tool_usage)
    manager.on_modality_loaded = MagicMock()
    with patch.object(DownloadOrchestrator, "_register_default_services"):
        orchestrator = DownloadOrchestrator(manager)
    orchestrator.register_service(wrapper)
    name, stats, ir = orchestrator.execute_download(
        entry.entry_id, strategy_override=REQUIREMENT, return_ir=True
    )
    assert name == CANONICAL and stats["modality_name"] == CANONICAL
    assert manager.list_modalities() == [CANONICAL]
    assert manager.download_queue.get_entry(entry.entry_id).modality_name == CANONICAL
    assert (
        manager.download_queue.get_entry(entry.entry_id).status
        == DownloadStatus.COMPLETED
    )
    # The import is saved and read back right after the load, so the listener got
    # the loaded object and the session holds the file's: same name, same shape.
    manager.on_modality_loaded.assert_called_once()
    loaded_name, loaded = manager.on_modality_loaded.call_args.args
    assert loaded_name == CANONICAL
    assert loaded.shape == manager.get_modality(CANONICAL).shape
    tool_names = [
        call.kwargs.get("tool_name", call.args[0] if call.args else None)
        for call in manager.log_tool_usage.call_args_list
    ]
    assert tool_names.count("load_dataset") == 1
    assert tool_names.count("download_geo_dataset_strategic") == 1
    assert "download_orchestrator.execute_download" not in tool_names
    assert ir.execution_context["data_requirement"] == "gene_counts"
    assert ir.output_entities == [CANONICAL]


def test_prompt_scopes_count_numeric_requirement_and_refusal_to_active_requests():
    from lobster.agents.data_expert.prompts import create_data_expert_prompt

    prompt = create_data_expert_prompt()
    assert 'data_requirement="gene_counts"' in prompt
    assert "whole-number counts (numeric check)" in prompt
    assert "do not retry the same entry unchanged" in prompt
    assert "legacy" in prompt
