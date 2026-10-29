"""Real autosave/CLI outcomes for ordinary failures, not only managed refusals.

Fakes are confined to deliberate failed sink operations; the data manager,
store/dirty bookkeeping, native successful sibling saves and CLI are real.
No production CLI change or broad backend-safety claim is required here.
"""

import builtins
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import anndata as ad
import numpy as np
import pytest

from lobster.core.runtime import data_manager as dm_module
from lobster.core.runtime.data_manager import DataManagerV2

FAILURE_DETAIL = "benign injected failure detail must not be echoed to the user"


def _adata():
    adata = ad.AnnData(X=np.array([[1.25, 2.5], [3.25, 4.5]]))
    adata.obs_names = ["cell_0", "cell_1"]
    adata.var_names = ["gene_0", "gene_1"]
    return adata


@pytest.fixture
def dm(tmp_path, monkeypatch):
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    monkeypatch.delenv("LOBSTER_MODALITY_MEMORY_BUDGET_GB", raising=False)
    return DataManagerV2(workspace_path=tmp_path / "workspace")


def _not_saved(items, label):
    outcomes = [
        item for item in items if item.startswith("Not saved:") and label in item
    ]
    assert len(outcomes) == 1, "every failed save needs one user-visible outcome"
    assert FAILURE_DETAIL not in outcomes[0]
    assert not any(item.startswith("Modality '") and label in item for item in items)
    assert not any("Skipped" in item and label in item for item in items)
    return outcomes[0]


class _Output:
    def __init__(self):
        self.bodies = []

    def render_blocks(self, blocks):
        self.bodies.extend(block.data.get("body") for block in blocks)


def _cli_save(dm):
    from lobster.cli_internal.commands.heavy.slash_commands import _command_save

    output = _Output()
    returned = _command_save(SimpleNamespace(data_manager=dm), output, force=True)
    return returned, output.bodies


def test_failure_reporting_modules_come_from_this_checkout():
    from lobster.cli_internal.commands.heavy import slash_commands

    root = Path(__file__).resolve().parents[3]
    for module in (dm_module, slash_commands):
        assert Path(module.__file__).resolve().is_relative_to(root)


def test_cli_first_reports_resident_write_failure_and_successful_sibling(
    dm, monkeypatch
):
    dm.store_modality(name="failed", adata=_adata())
    dm.store_modality(name="ordinary", adata=_adata())
    actual_save = dm.save_modality
    calls = []

    def save(name, path, **kwargs):
        calls.append(name)
        if name == "failed":
            raise OSError(FAILURE_DETAIL)
        return actual_save(name, path, **kwargs)

    monkeypatch.setattr(dm, "save_modality", save)
    returned, bodies = _cli_save(dm)
    outcome = _not_saved(bodies, "'failed'")
    assert outcome in returned
    assert "Saved 2 items" in returned  # ordinary native H5AD + actual processing log
    assert "already up-to-date" not in returned
    assert calls == ["failed", "ordinary"]
    assert "failed" in dm._modality_dirty
    assert "ordinary" not in dm._modality_dirty
    assert not (dm.data_dir / "failed_autosave.h5ad").exists()
    restored = ad.read_h5ad(dm.data_dir / "ordinary_autosave.h5ad")
    np.testing.assert_array_equal(restored.X, _adata().X)


def test_autosave_records_resident_failure_and_error_progress(dm, monkeypatch):
    dm.store_modality(name="failed", adata=_adata())
    snapshot = dm.modalities["failed"].copy()
    monkeypatch.setattr(dm, "save_modality", Mock(side_effect=OSError(FAILURE_DETAIL)))
    progress = []
    info = Mock(wraps=dm_module.logger.info)
    monkeypatch.setattr(dm_module.logger, "info", info)
    items = dm.auto_save_state(
        force=True, progress_callback=lambda *args: progress.append(args)
    )
    _not_saved(items, "'failed'")
    successes = [
        call.args[0]
        for call in info.call_args_list
        if call.args and call.args[0].startswith("Auto-saved:")
    ]
    assert all(
        "Not saved:" not in message and "'failed'" not in message
        for message in successes
    )
    assert progress == [(1, 1, "failed", "saving"), (1, 1, "failed", "error")]
    assert "failed" in dm._modality_dirty
    np.testing.assert_array_equal(dm.modalities["failed"].X, snapshot.X)


@pytest.mark.parametrize("failure", ["missing", "copy_failed"])
def test_spilled_modality_failure_is_visible_without_source_state_loss(
    dm, monkeypatch, failure
):
    dm.store_modality(name="failed", adata=_adata())
    assert dm._spill_modality("failed") is not None
    source = Path(dm._modality_spilled["failed"])
    spill_map = dict(dm._modality_spilled)
    dirty_before = set(dm._modality_dirty)  # native spill is already persisted/clean
    source_bytes = source.read_bytes()
    if failure == "missing":
        source.unlink()
    else:
        monkeypatch.setattr(
            dm_module.shutil, "copy2", Mock(side_effect=OSError(FAILURE_DETAIL))
        )
    items = dm.auto_save_state(force=True)
    _not_saved(items, "'failed'")
    assert dm._modality_spilled == spill_map
    assert dm._modality_dirty == dirty_before
    assert "failed" not in dm.modalities
    assert not (dm.data_dir / "failed_autosave.h5ad").exists()
    if failure == "copy_failed":
        assert source.read_bytes() == source_bytes


def _attach_failing_sink(dm):
    # Direct attachment avoids an unrelated migration attempt in set_modality_backend.
    sink = SimpleNamespace(
        ingest_anndata=Mock(side_effect=OSError(FAILURE_DETAIL)),
        export_h5ad=Mock(),
    )
    dm._modality_backend = sink
    return sink


def test_cli_attached_backend_ingest_failure_is_not_up_to_date(dm):
    dm.store_modality(name="failed", adata=_adata())
    sink = _attach_failing_sink(dm)
    returned, bodies = _cli_save(dm)
    outcome = _not_saved(bodies, "'failed'")
    assert outcome in returned
    assert "already up-to-date" not in returned
    assert "failed" in dm._modality_dirty
    sink.ingest_anndata.assert_called_once()
    sink.export_h5ad.assert_not_called()


def test_direct_attached_ingest_failure_refuses_before_stale_export(dm):
    dm.store_modality(name="failed", adata=_adata())
    sink = _attach_failing_sink(dm)
    target = dm.data_dir / "direct.h5ad"
    error = OSError(FAILURE_DETAIL)
    sink.ingest_anndata.side_effect = error
    with pytest.raises(OSError) as caught:
        dm.save_modality("failed", target)
    assert caught.value is error
    sink.export_h5ad.assert_not_called()
    assert "failed" in dm._modality_dirty
    assert not target.exists()


def test_processing_log_failure_is_not_certified_as_saved(dm, monkeypatch):
    dm.store_modality(name="ordinary", adata=_adata())
    target = dm.exports_dir / "processing_log.json"

    def fail_log_only(path, *args, **kwargs):
        if Path(path) == target:
            raise OSError(FAILURE_DETAIL)
        return builtins.open(path, *args, **kwargs)

    monkeypatch.setattr(dm_module, "open", fail_log_only, raising=False)
    items = dm.auto_save_state(force=True)
    outcomes = [
        item
        for item in items
        if item.startswith("Not saved:") and "processing log" in item.lower()
    ]
    assert len(outcomes) == 1
    assert FAILURE_DETAIL not in outcomes[0]
    assert "Processing log" not in items
    assert any(item.startswith("Modality 'ordinary':") for item in items)
    assert "ordinary" not in dm._modality_dirty
    assert not target.exists()


def test_native_ordinary_save_and_log_are_still_reported_successfully(dm):
    dm.store_modality(name="ordinary", adata=_adata())
    items = dm.auto_save_state(force=True)
    assert items == ["Modality 'ordinary': ordinary_autosave.h5ad", "Processing log"]
    assert (dm.data_dir / "ordinary_autosave.h5ad").is_file()
    assert (dm.exports_dir / "processing_log.json").is_file()
    assert "ordinary" not in dm._modality_dirty


def test_spilled_only_empty_history_reports_only_a_real_processing_log(dm):
    dm.store_modality(name="ordinary", adata=_adata())
    assert dm._spill_modality("ordinary") is not None
    assert dm.modalities == {} and "ordinary" in dm._modality_spilled
    dm.processing_log.clear()
    dm.provenance.activities.clear()
    items = dm.auto_save_state(force=True)
    assert "Processing log" in items
    assert (dm.exports_dir / "processing_log.json").is_file()


def test_native_saved_write_done_callback_raise_is_not_a_write_failure(dm):
    dm.store_modality(name="ordinary", adata=_adata())
    callback_error = RuntimeError("benign done notification failure")
    progress = []

    def callback(*args):
        progress.append(args)
        if args[-1] == "done":
            raise callback_error

    raised = None
    outcomes = None
    try:
        outcomes = dm.auto_save_state(force=True, progress_callback=callback)
    except RuntimeError as error:
        raised = error
    # First qualify that the real write completed, independent of the assertion
    # about propagation: the failure is notification-only, not a mocked sink.
    target = dm.data_dir / "ordinary_autosave.h5ad"
    assert target.is_file()
    np.testing.assert_array_equal(ad.read_h5ad(target).X, _adata().X)
    assert "ordinary" not in dm._modality_dirty
    assert raised is callback_error
    assert outcomes is None
    assert progress == [(1, 1, "ordinary", "saving"), (1, 1, "ordinary", "done")]
