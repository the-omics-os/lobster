"""Saving nullable-string arrays must work in a fresh process.

Serialization depends on anndata's nullable-string setting, which must be configured by
the writer itself rather than by unrelated earlier tool execution. These tests exercise
save/load without first invoking custom code.

**Do not import or invoke `execute_custom_code` in these tests**; doing so could configure
the setting before the writer is exercised and mask a missing initialization.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _adata_with_string_metadata():
    """AnnData whose obs/var carry string columns — the case anndata refuses to write."""
    import anndata

    n_obs, n_vars = 6, 4
    obs = pd.DataFrame(
        {
            "cell_type": ["T cell", "B cell", "NK", "T cell", "B cell", "NK"],
            "batch": ["a", "a", "b", "b", "c", "c"],
        },
        index=[f"cell{i}" for i in range(n_obs)],
    )
    var = pd.DataFrame(
        {"gene_symbol": ["CD3D", "MS4A1", "NKG7", "GAPDH"]},
        index=[f"g{i}" for i in range(n_vars)],
    )
    return anndata.AnnData(
        X=np.ones((n_obs, n_vars), dtype="float32"), obs=obs, var=var
    )


def test_save_succeeds_without_a_prior_sandbox_call(tmp_path):
    """The reproducer: store, save, no `execute_custom_code` anywhere in between."""
    from lobster.core.backends.h5ad_backend import H5ADBackend

    target = tmp_path / "out.h5ad"
    H5ADBackend().save(_adata_with_string_metadata(), target)
    assert target.exists() and target.stat().st_size > 0


def test_save_does_not_leak_global_settings(tmp_path):
    """The writer must not mutate state observable by unrelated code.

    Enabling the setting permanently would fix the write and silently change how *every*
    later anndata call in the process behaves.
    """
    import anndata

    from lobster.core.backends.h5ad_backend import H5ADBackend

    before = (
        anndata.settings.allow_write_nullable_strings,
        pd.options.future.infer_string,
    )
    H5ADBackend().save(_adata_with_string_metadata(), tmp_path / "out.h5ad")
    after = (
        anndata.settings.allow_write_nullable_strings,
        pd.options.future.infer_string,
    )
    assert before == after, "save() leaked global settings"


def test_settings_are_restored_even_when_save_fails(tmp_path):
    """Failure must not leave the process in a mutated state either."""
    import anndata

    from lobster.core.backends.h5ad_backend import H5ADBackend

    before = (
        anndata.settings.allow_write_nullable_strings,
        pd.options.future.infer_string,
    )
    # A directory that cannot be created (a file occupies the parent path).
    blocker = tmp_path / "blocker"
    blocker.write_text("x")
    with pytest.raises(Exception):
        H5ADBackend().save(_adata_with_string_metadata(), blocker / "sub" / "out.h5ad")
    after = (
        anndata.settings.allow_write_nullable_strings,
        pd.options.future.infer_string,
    )
    assert before == after, "settings not restored on the failure path"


def test_the_setting_is_applied_in_the_main_process(tmp_path):
    """Guards the root cause, not just the symptom.

    Before the fix the assignment existed ONLY in the generated subprocess preamble. A future
    refactor that removes it from the writer would reintroduce an order-dependent failure that
    passes whenever custom code runs first.
    """
    from pathlib import Path

    source = Path("lobster/core/backends/h5ad_backend.py").read_text()
    assert "allow_write_nullable_strings" in source, (
        "the writer no longer opts in to nullable-string writes; saving will fail on a "
        "fresh process until an unrelated sandbox call happens to run first"
    )


def test_failed_save_is_visible_to_the_caller(tmp_path):
    """A write failure must raise, not return quietly.

    Related to the same incident: the *tool* swallowed this into a formatted string and logged
    no provenance, so the audit trail showed no attempt at all.
    """
    from lobster.core.backends.h5ad_backend import H5ADBackend

    blocker = tmp_path / "blocker"
    blocker.write_text("x")
    with pytest.raises(Exception):
        H5ADBackend().save(_adata_with_string_metadata(), blocker / "sub" / "out.h5ad")
