"""RC5(c): RAM registration, notebook intent and genuine output paths are distinct."""

import json
from types import SimpleNamespace

import pytest

from lobster.tools.custom_code_tool import (
    _format_response,
    metadata_store_post_processor,
)


@pytest.mark.parametrize("persist", [False, True])
@pytest.mark.parametrize("backend_attached", [False, True])
def test_response_propagates_conditional_modality_storage_not_ir_intent(
    persist, backend_attached
):
    saved = None if backend_attached else False
    storage = (
        "in memory (disk persistence not confirmed)"
        if backend_attached
        else "in memory (not saved to disk)"
    )
    stats = {
        "new_modality_name": "source_custom",
        "write_back_error": None,
        "notebook_export_requested": persist,
        "modality_saved": saved,
        "modality_storage": storage,
        # A separately returned stdout file must not be discarded/renamed.
        "stdout_full_path": "/synthetic/output/stdout.txt",
    }
    figure_report = {"figures": "Saved 1 submitted figure."}
    result = json.loads(
        _format_response(42, stats, persist, figure_report=figure_report)
    )
    assert result["persisted"] is persist  # compatibility, not disk receipt
    assert result["notebook_export_requested"] is persist
    assert result["modality_saved"] is saved
    assert result["modality_storage"] == storage
    assert result["new_modality_name"] == "source_custom"
    assert result["write_back_error"] is None
    assert result["stdout_full_path"] == stats["stdout_full_path"]
    assert result["figures"] == figure_report["figures"]
    # Formatting/forwarding only; no assertion of real stdout/figure disk I/O.


def test_response_without_new_modality_does_not_invent_storage_claim():
    response = json.loads(_format_response(42, {}, persist=True))
    assert response["persisted"] is True
    assert response["notebook_export_requested"] is True
    assert response["new_modality_name"] is None
    assert response["modality_saved"] is None
    assert response["modality_storage"] is None


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("samples", [[], [{"id": 1}, {"id": 2}]])
def test_metadata_ram_messages_explicitly_disclaim_disk_save(existing, samples):
    key = "chosen_samples"
    dm = SimpleNamespace(metadata_store={key: {"samples": []}} if existing else {})
    payload = {"samples": samples, "output_key": key}
    message = metadata_store_post_processor(
        payload, {}, dm, key if existing else None, None
    )
    assert dm.metadata_store[key]["samples"] == samples
    assert "in memory (not saved to disk)" in message
    assert "Persisted" not in message
    if existing:
        assert f"Updated {len(samples)} samples" in message
    else:
        assert "Created metadata_store" in message
        assert "write_to_workspace" in message  # opt-in export guidance retained
