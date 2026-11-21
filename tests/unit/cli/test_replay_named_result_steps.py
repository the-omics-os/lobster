"""Proteomics and metabolomics sessions replay to MATCH.

Their steps call a service that returns its result in a variable of its own
(``adata_qc, stats, _ = service.assess_...(adata)``) and the tool stores it as a
new modality. The exported cells left that variable unused, so every later step
and every save ran on the loaded input; the lineage did not know the stored
names, so a step on an earlier modality (a side branch) was not restarted from
it and the session's saves of those modalities had no cell. The proteomics QC
tool also exported only the first of its four checks.

Each row is a native session recorded on this build, exported and replayed.
Skipped without the package or a Jupyter kernel.
"""

import json

import anndata as ad  # noqa: E402
import numpy as np
import pandas as pd
import pytest

from lobster.core.runtime.data_manager import DataManagerV2  # noqa: E402
from lobster.testing.provenance_replay import factory_tools  # noqa: E402
from tests.unit.cli.test_replay_commands import app, needs_kernel, runner

pytestmark = [pytest.mark.unit, needs_kernel]


def _replay(dm, source):
    notebook = dm.export_notebook("nb", "S", filter_strategy="successful")
    result = runner.invoke(
        app,
        ["--verify", str(notebook), "--input", str(source),
         "-w", str(dm.workspace_path), "--json"],
    )  # fmt: skip
    return json.loads(result.stdout)


def _diffs(out):
    return [d for slot in out["report"]["slots"] for d in slot.get("diffs", [])]


def _ok(reply):
    text = " ".join(str(reply).split())
    assert not text.lower().startswith(("error", "failed")), text


def _input(tmp_path, monkeypatch, n_vars, mean, prefix):
    monkeypatch.chdir(tmp_path)
    rng = np.random.default_rng(5)
    X = rng.lognormal(mean, 1, size=(24, n_vars))
    X[rng.random(X.shape) < 0.12] = np.nan
    obs = pd.DataFrame(
        {"condition": ["a"] * 12 + ["b"] * 12, "batch": ["x", "y"] * 12,
         "sample_type": ["sample"] * 24},
        index=[f"S{i}" for i in range(24)],
    )  # fmt: skip
    source = tmp_path / "in" / "input.h5ad"
    source.parent.mkdir()
    ad.AnnData(
        X=X.astype(np.float32),
        obs=obs,
        var=pd.DataFrame(index=[f"{prefix}{i}" for i in range(n_vars)]),
    ).write_h5ad(source)
    dm = DataManagerV2(workspace_path=tmp_path / "ws", auto_scan=False)
    return dm, source


@pytest.mark.parametrize("log_transform", [True, False], ids=["log", "linear"])
def test_a_proteomics_session_replays_to_match(tmp_path, monkeypatch, log_transform):
    """QC, a filter on the input (a side branch), then normalize and impute on
    the QC result."""
    pe = pytest.importorskip("lobster.agents.proteomics.proteomics_expert")
    dm, source = _input(tmp_path, monkeypatch, 150, 10, "P")
    dm.load_modality("pr", str(source), adapter="transcriptomics_bulk")
    tools = factory_tools(pe.proteomics_expert, pe.__name__, dm)
    _ok(tools["assess_proteomics_quality"].invoke(
        {"modality_name": "pr", "platform_type": "mass_spec"}
    ))  # fmt: skip
    _ok(tools["filter_proteomics_data"].invoke(
        {"modality_name": "pr", "platform_type": "mass_spec",
         "max_missing_per_sample": 0.6, "max_missing_per_protein": 0.6,
         "save_result": True}
    ))  # fmt: skip
    _ok(tools["normalize_proteomics_data"].invoke(
        {"modality_name": "pr_quality_assessed", "platform_type": "mass_spec",
         "normalization_method": "median", "log_transform": log_transform,
         "save_result": True}
    ))  # fmt: skip
    _ok(tools["impute_missing_values"].invoke(
        {"modality_name": "pr_quality_assessed_normalized", "method": "min_prob",
         "platform_type": "mass_spec", "save_result": True}
    ))  # fmt: skip
    out = _replay(dm, source)
    assert out["verdict"] == "MATCH", (out["notes"], _diffs(out))


def test_a_metabolomics_session_replays_to_match(tmp_path, monkeypatch):
    """QC, filter, impute and normalize, each on the step before, with defaults;
    every modality is saved."""
    me = pytest.importorskip("lobster.agents.metabolomics.metabolomics_expert")
    dm, source = _input(tmp_path, monkeypatch, 120, 8, "M")
    dm.load_modality("mb", str(source), adapter="transcriptomics_bulk")
    tools = factory_tools(me.metabolomics_expert, me.__name__, dm)
    name = "mb"
    for tool, suffix in [
        ("assess_metabolomics_quality", "_quality_assessed"),
        ("filter_metabolomics_features", "_filtered"),
        ("handle_missing_values", "_imputed"),
        ("normalize_metabolomics", "_normalized"),
    ]:
        _ok(tools[tool].invoke({"modality_name": name}))
        name += suffix
        assert name in dm.list_modalities(), dm.list_modalities()
        if not (dm.data_dir / f"{name}.h5ad").exists():
            dm.save_modality(name, f"{name}.h5ad")
    out = _replay(dm, source)
    assert out["verdict"] == "MATCH", (out["notes"], _diffs(out))
