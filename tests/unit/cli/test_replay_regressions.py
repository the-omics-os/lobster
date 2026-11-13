"""Every false MATCH an adversarial pass found, and the usability flows, in one module.

Each test's name or docstring says what it checks, or names its flow (n1-n5).
Run it at every merge sha: a MATCH that should not be one fails here.
Real sessions (DataManagerV2 and the exporter) wherever the original repro was one.
"""

import json
import shutil
from types import SimpleNamespace
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from tests.unit.cli import test_replay_commands as _base

# the helpers, constants and fixtures of the replay tests (not the tests themselves)
globals().update(
    {
        k: v
        for k, v in vars(_base).items()
        if not k.startswith("test_") and not k.startswith("__")
    }
)


# --- the common flows: a raw file is loaded, a step writes a NEW modality ----------


def _flow_session(
    tmp_path,
    monkeypatch,
    flow,
    annotate_loaded=False,
    unsaved=False,
    lost=False,
    nullable=False,
):
    """Usability flows, on a real session and a real export.

    n1 raw outside the workspace; n2 the same under a GEO download's name; n3 the
    raw modality is also saved (as ``<name>.h5ad``, what autosave does); n4 the raw
    file copied into the workspace root as ``<name>.h5ad``, then loaded; n5 copied into
    ``data/``; ``--input`` is the original raw file. A step writes a new modality, which is saved.
    """
    from lobster.core.provenance.analysis_ir import AnalysisStep
    from lobster.core.runtime.data_manager import DataManagerV2

    _clean_env(monkeypatch, tmp_path)
    workspace = tmp_path / "workspace"
    name, rawname = "pbmc", "pbmc_raw.h5ad"
    if flow == "n2":
        name, rawname = "geo_gse12345", "GSE12345_matrix.h5ad"
    source = _adata()
    source.X = np.abs(source.X)
    raw = tmp_path / "raw" / rawname
    raw.parent.mkdir(parents=True)
    _save(source, raw)
    manager = DataManagerV2(workspace_path=workspace, auto_scan=False)
    if flow in ("n4", "n5"):
        home = workspace if flow == "n4" else workspace / "data"
        home.mkdir(parents=True, exist_ok=True)
        shutil.copy(raw, home / f"{name}.h5ad")  # under the modality's own name
        manager.load_modality(
            name, home / f"{name}.h5ad", adapter="transcriptomics_single_cell"
        )
    else:
        manager.load_modality(name, raw, adapter="transcriptomics_single_cell")
    loaded = manager.get_modality(name)
    new = loaded.copy()  # a tool writes a NEW modality; the loaded one stays as loaded
    if unsaved:  # a modality a tool made and nobody saved
        manager.modalities[f"{name}_qc"] = loaded.copy()
    if annotate_loaded:  # ... unless the user annotates it in place
        loaded.obs["cell_type"] = ["T"] * loaded.n_obs
    if flow == "n3":
        manager.save_modality(name, workspace / "data" / f"{name}.h5ad")
    step = AnalysisStep(
        operation="x", tool_name="normalize_log1p", description="d", library="scanpy",
        code_template=NULLABLE_STEP if nullable else STEP,
        imports=["import numpy as np", "import pandas as pd"], parameters={},
        parameter_schema={}, input_entities=["adata"], output_entities=["adata"],
        exportable=True,
    )  # fmt: skip
    new.X = np.log1p(new.X)
    if nullable:  # a nullable integer obs column with one missing value
        new.obs["n"] = pd.array(_nullable_values(new.n_obs), dtype="Int64")
    manager.log_tool_usage("normalize_log1p", {}, "d", ir=step)
    manager.modalities[f"{name}_normalized"] = new
    manager.save_modality(
        f"{name}_normalized", workspace / "data" / f"{name}_normalized.h5ad"
    )
    if lost:  # saved by the session, its file gone before the export
        (workspace / "data" / f"{name}_normalized.h5ad").unlink()
    notebook = Path(manager.export_notebook("nb", "S", filter_strategy="replay"))
    return workspace, notebook, raw


FLOWS = ["n1", "n2", "n3", "n4", "n5"]

# a step that also writes a nullable integer obs column with one missing value
NULLABLE_STEP = (
    STEP + "\nadata.obs['n'] = pd.array([1, None] + list(range(3, adata.n_obs + 1)),"
    " dtype='Int64')"
)


def _nullable_values(n_obs):
    return [1, None] + list(range(3, n_obs + 1))


@needs_kernel
def test_a_nullable_integer_column_with_a_missing_value_reaches_a_match(
    tmp_path, monkeypatch
):
    workspace, notebook, raw = _flow_session(tmp_path, monkeypatch, "n1", nullable=True)
    out = json.loads(_verify_real(workspace, notebook, raw).stdout)
    assert out["verdict"] == "MATCH", out
    saved = ad.read_h5ad(workspace / "data" / "pbmc_normalized.h5ad")
    assert str(saved.obs["n"].dtype) == "Int64"
    assert saved.obs["n"].isna().tolist()[:2] == [False, True]


@needs_kernel
@pytest.mark.parametrize(
    "edit, replacement",
    [
        ("pd.array([1, None]", "pd.array([2, None]"),
        ("pd.array([1, None] + list(range(3,", "pd.array([None, 2] + list(range(3,"),
    ],
)
def test_a_changed_value_or_a_moved_missing_value_in_a_nullable_column_is_not_a_match(
    tmp_path, monkeypatch, edit, replacement
):
    workspace, notebook, raw = _flow_session(tmp_path, monkeypatch, "n1", nullable=True)
    text = notebook.read_text("utf-8")
    assert edit in text
    notebook.write_text(text.replace(edit, replacement), "utf-8")
    result = _verify_real(workspace, notebook, raw)
    out = json.loads(result.stdout)
    assert result.exit_code == 1 and out["verdict"] == "MISMATCH", out
    diffs = " ".join(d for s in out["report"]["slots"] for d in s["diffs"])
    assert "obs" in diffs and "/n" in diffs, diffs


@needs_kernel
@pytest.mark.parametrize("flow", FLOWS)
def test_the_common_flows_reach_a_match(tmp_path, monkeypatch, flow):
    workspace, notebook, raw = _flow_session(tmp_path, monkeypatch, flow)
    result = _verify_real(workspace, notebook, raw)
    out = json.loads(result.stdout)
    assert result.exit_code == 0 and out["verdict"] == "MATCH", out


@needs_kernel
@pytest.mark.parametrize("flow", FLOWS)
def test_the_common_flows_with_an_edited_step_are_not_a_match(
    tmp_path, monkeypatch, flow
):
    workspace, notebook, raw = _flow_session(tmp_path, monkeypatch, flow)
    edit_notebook(notebook, STEP + " + 1")
    result = _verify_real(workspace, notebook, raw)
    assert result.exit_code == 1, result.output


@needs_kernel
@pytest.mark.parametrize("flow", FLOWS)
def test_the_common_flows_with_an_edited_raw_file_are_not_a_match(
    tmp_path, monkeypatch, flow
):
    workspace, notebook, raw = _flow_session(tmp_path, monkeypatch, flow)
    edited = ad.read_h5ad(raw)
    edited.X = edited.X + 1
    other = tmp_path / "edited.h5ad"
    _save(edited, other)
    result = _verify_real(workspace, notebook, other)
    assert result.exit_code == 4, result.output
    assert (
        "not a file the session loaded"
        in json.loads(result.stdout)["report"]["refusal"]["reason"]
    )


@needs_kernel
def test_a_saved_copy_of_the_input_is_listed_not_hidden(tmp_path, monkeypatch):
    workspace, notebook, raw = _flow_session(tmp_path, monkeypatch, "n3")
    out = json.loads(_verify_real(workspace, notebook, raw).stdout)
    assert any("Saved copy of the input" in n for n in out["notes"]), out["notes"]
    _save(_adata(shift=9.0), workspace / "data" / "pbmc.h5ad")  # no longer a copy
    assert _verify_real(workspace, notebook, raw).exit_code in (3, 4)


@needs_kernel
def test_a_hashless_entry_of_a_loaded_modality_is_listed_as_the_loaded_input(
    tmp_path, monkeypatch
):
    workspace, notebook, raw = _flow_session(tmp_path, monkeypatch, "n1")
    out = json.loads(_verify_real(workspace, notebook, raw).stdout)
    assert any("Loaded input" in n and "pbmc.h5ad" in n for n in out["notes"]), out[
        "notes"
    ]


@needs_kernel
def test_a_match_needs_one_saved_output_compared(tmp_path, monkeypatch):
    # every saved file is the loaded input's copy and the notebook writes nothing
    workspace, notebook, raw = _flow_session(tmp_path, monkeypatch, "n3")
    edit_notebook(notebook, "pass")
    (workspace / "data" / "pbmc_normalized.h5ad").unlink()
    result = _verify_real(workspace, notebook, raw)
    assert result.exit_code == 3, result.output


@needs_kernel
def test_f1_a_saved_file_the_notebook_does_not_produce_is_not_compared(
    tmp_path, monkeypatch
):
    workspace, notebook, source = _real_session(tmp_path, monkeypatch, edited=("s1",))
    result = _verify_real(workspace, notebook, source)
    out = json.loads(result.stdout)
    assert result.exit_code == 3 and out["verdict"] == "NOT_COMPARED", out
    assert any(
        "not produced by the notebook" in n and "s1.h5ad" in n for n in out["notes"]
    ), out["notes"]


@needs_kernel
def test_f2_an_input_that_is_the_sessions_own_saved_result_is_refused(
    tmp_path, monkeypatch
):
    workspace, notebook, source = _real_session(
        tmp_path, monkeypatch, tool="clip", code="adata.X = np.clip(adata.X, 0, 0.5)"
    )
    ok = _verify_real(workspace, notebook, source)  # from the raw input: a match
    assert ok.exit_code == 0, ok.output
    result = _verify_real(workspace, notebook, workspace / "data" / "s1.h5ad")
    out = json.loads(result.stdout)
    assert result.exit_code == 4 and out["verdict"] == "REFUSED", out
    assert "a saved result of the session" in out["report"]["refusal"]["reason"]


@needs_kernel
@pytest.mark.parametrize("how", ["rewritten", "gzip"])
def test_r1_an_input_with_the_content_of_a_saved_result_is_refused(
    tmp_path, monkeypatch, how
):
    workspace, notebook, source = _real_session(tmp_path, monkeypatch)
    saved = ad.read_h5ad(workspace / "data" / "s1.h5ad")
    same = tmp_path / "same.h5ad"
    if how == "gzip":
        with ad.settings.override(allow_write_nullable_strings=True):
            saved.write_h5ad(same, compression="gzip")
    else:
        _save(saved, same)
    assert _sha(same) != _sha(workspace / "data" / "s1.h5ad")  # not a byte copy
    edit_notebook(notebook, "pass")  # nothing left to do: the input is the result
    result = _verify_real(workspace, notebook, same)
    out = json.loads(result.stdout)
    assert result.exit_code == 4 and out["verdict"] == "REFUSED", out
    assert "a saved result of the session" in out["report"]["refusal"]["reason"]
    assert _verify_real(workspace, notebook, source).exit_code == 1  # from the raw: no


@needs_kernel
def test_r2_an_intermediate_saved_modality_as_the_input_is_not_a_match(
    tmp_path, monkeypatch
):
    workspace, notebook, _source = _real_session(
        tmp_path,
        monkeypatch,
        tool="clip",
        code="adata.X = np.clip(adata.X, 0, 0.5)",
        copies=("s1",),  # an intermediate the idempotent step leaves as it is
    )
    result = _verify_real(workspace, notebook, workspace / "data" / "s1.h5ad")
    out = json.loads(result.stdout)
    assert result.exit_code == 4 and out["verdict"] == "REFUSED", out
    assert "saved result of the session" in out["report"]["refusal"]["reason"]


@needs_kernel
def test_r3_an_export_record_entry_without_a_hash_is_not_compared(
    tmp_path, monkeypatch
):
    # s1 and s2 saved, the notebook writes only s2's output: the record's other
    # entry is what stands between a hashless s1 and a MATCH
    workspace, notebook, source = _real_session(tmp_path, monkeypatch, copies=("s3",))
    sha = replay_commands._recorded_hashes(notebook)["s3.h5ad"]
    notebook.write_text(notebook.read_text("utf-8").replace(sha, "file_not_found"))
    result = _verify_real(workspace, notebook, source)
    out = json.loads(result.stdout)
    assert result.exit_code == 3 and out["verdict"] == "NOT_COMPARED", out
    assert any("no hash" in n and "s3.h5ad" in n for n in out["notes"]), out["notes"]


@needs_kernel
@pytest.mark.parametrize("flow", FLOWS)
def test_n_the_common_flows_with_an_extra_write_are_not_a_match(
    tmp_path, monkeypatch, flow
):
    workspace, notebook, raw = _flow_session(tmp_path, monkeypatch, flow)
    nb = json.loads(notebook.read_text("utf-8"))
    (cell,) = [
        c
        for c in nb["cells"]
        if c["cell_type"] == "code" and STEP in "".join(c["source"])
    ]
    cell["source"] = "".join(cell["source"]) + (
        f"\nopen({str(workspace / 'data' / 'stray.csv')!r}, 'w').write('x')\n"
    )
    notebook.write_text(json.dumps(nb), "utf-8")
    result = _verify_real(workspace, notebook, raw)
    assert result.exit_code == 4, result.output
    assert "stray.csv" in json.loads(result.stdout)["report"]["refusal"]["reason"]


@needs_kernel
@pytest.mark.parametrize("flow", FLOWS)
def test_n_the_common_flows_with_a_saved_file_edited_after_are_refused(
    tmp_path, monkeypatch, flow
):
    workspace, notebook, raw = _flow_session(tmp_path, monkeypatch, flow)
    name = {"n2": "geo_gse12345"}.get(flow, "pbmc")
    _save(_adata(shift=9.0), workspace / "data" / f"{name}_normalized.h5ad")
    result = _verify_real(workspace, notebook, raw)
    assert result.exit_code == 4, result.output


@needs_kernel
def test_b_annotation_on_a_loaded_copy_saved_under_its_own_name_is_not_excused(
    tmp_path, monkeypatch
):
    workspace, notebook, raw = _flow_session(
        tmp_path, monkeypatch, "n3", annotate_loaded=True
    )
    result = _verify_real(workspace, notebook, raw)
    out = json.loads(result.stdout)
    assert result.exit_code == 3 and out["verdict"] == "NOT_COMPARED", out
    assert any(
        "not produced by the notebook" in n and "pbmc.h5ad" in n for n in out["notes"]
    ), out["notes"]


# --- the same cases again, on real sessions ---------------------------------


def _in_the_step_cell(notebook: Path, extra: str) -> None:
    """Add code to the notebook's step cell, as a user editing the notebook would."""
    nb = json.loads(notebook.read_text("utf-8"))
    (cell,) = [
        c
        for c in nb["cells"]
        if c["cell_type"] == "code" and STEP in "".join(c["source"])
    ]
    cell["source"] = "".join(cell["source"]) + "\n" + extra + "\n"
    notebook.write_text(json.dumps(nb), "utf-8")


@needs_kernel
def test_f3_a_notebook_overwriting_a_saved_file_through_a_linked_data_folder_is_not_a_match(
    tmp_path, monkeypatch
):
    workspace, notebook, raw = _real_session(tmp_path, monkeypatch, copies=("s3",))
    elsewhere = tmp_path / "elsewhere"  # the saved files live outside the workspace
    (workspace / "data").rename(elsewhere)
    (workspace / "data").symlink_to(elsewhere)
    _in_the_step_cell(
        notebook, f"open({str(workspace / 'data' / 's3.h5ad')!r}, 'wb').write(b'x')"
    )
    result = _verify_real(workspace, notebook, raw)
    assert result.exit_code == 4, result.output
    assert "s3.h5ad" in json.loads(result.stdout)["report"]["refusal"]["reason"]


@needs_kernel
def test_s1_an_upper_case_data_output_is_not_ignored(tmp_path, monkeypatch):
    workspace, notebook, raw = _real_session(tmp_path, monkeypatch)
    _in_the_step_cell(notebook, "adata.write_h5ad(f'{output_prefix}_extra.H5AD')")
    result = _verify_real(workspace, notebook, raw)
    out = json.loads(result.stdout)
    assert result.exit_code == 3 and out["verdict"] == "NOT_COMPARED", out
    assert any("extra" in n for n in out["notes"]), out["notes"]


@needs_kernel
def test_r4_a_link_to_data_does_not_hide_a_write_to_it(tmp_path, monkeypatch):
    workspace, notebook, raw = _real_session(tmp_path, monkeypatch)
    (workspace / "data" / "x.csv").write_text("a")
    (workspace / "zzz").symlink_to(workspace / "data")
    _in_the_step_cell(
        notebook, f"open({str(workspace / 'zzz' / 'x.csv')!r}, 'a').write('b')"
    )
    result = _verify_real(workspace, notebook, raw)
    assert result.exit_code == 4, result.output
    assert "x.csv" in json.loads(result.stdout)["report"]["refusal"]["reason"]


@needs_kernel
def test_r5_a_narrower_policy_is_never_a_plain_match_and_says_what_it_skipped(
    tmp_path, monkeypatch
):
    workspace, notebook, raw = _real_session(tmp_path, monkeypatch)
    edit_notebook(notebook, STEP + " + 1")  # X differs; obs does not
    narrow = tmp_path / "narrow.json"
    narrow.write_text(
        json.dumps({"schema_version": 1, "required": ["files/s1.h5ad/obs"]})
    )
    result = _verify_real(workspace, notebook, raw, "--policy", str(narrow))
    out = json.loads(result.stdout)
    assert result.exit_code == 6 and out["verdict"] == "MATCH_CUSTOM_POLICY", out
    assert out["report"]["custom_policy"]["required"] == ["files/s1.h5ad/obs"]
    assert any("policy of your own" in n for n in out["notes"]), out["notes"]
    strict = tmp_path / "x.json"
    strict.write_text(
        json.dumps({"schema_version": 1, "required": ["files/s1.h5ad/X"]})
    )
    assert (
        _verify_real(workspace, notebook, raw, "--policy", str(strict)).exit_code == 1
    )


# --- a download hands over an object, not a file ----------------------------------


@pytest.fixture(autouse=True)
def _in_memory_inputs(request, monkeypatch):
    """The in-memory anchoring is off in 1.3.0; its tests turn it on (before the load)."""
    from lobster.core.provenance import content_digest

    off = request.node.name.startswith(("test_p4_", "test_g_in_memory_inputs_are_off"))
    monkeypatch.setattr(content_digest, "IN_MEMORY_INPUTS", not off)


def _geo_session(tmp_path, monkeypatch, *, saved_as="geo_gse12345_raw"):
    """A GEO-shaped session: an AnnData object is loaded, autosaved, then a step runs."""
    from lobster.core.provenance.analysis_ir import AnalysisStep
    from lobster.core.runtime.data_manager import DataManagerV2

    _clean_env(monkeypatch, tmp_path)
    workspace = tmp_path / "workspace"
    obj = _adata()
    obj.X = np.abs(obj.X)
    manager = DataManagerV2(workspace_path=workspace, auto_scan=False)
    manager.load_modality(
        "geo_gse12345", obj, adapter="transcriptomics_single_cell", validate=False
    )
    loaded = manager.get_modality("geo_gse12345")
    new = loaded.copy()
    manager.save_modality("geo_gse12345", workspace / "data" / f"{saved_as}.h5ad")
    new.X = np.log1p(new.X)
    step = AnalysisStep(
        operation="x", tool_name="normalize_log1p", description="d", library="scanpy",
        code_template=STEP, imports=["import numpy as np"], parameters={},
        parameter_schema={}, input_entities=["adata"], output_entities=["adata"],
        exportable=True,
    )  # fmt: skip
    manager.log_tool_usage("normalize_log1p", {}, "d", ir=step)
    manager.modalities["geo_gse12345_normalized"] = new
    manager.save_modality(
        "geo_gse12345_normalized", workspace / "data" / "geo_gse12345_normalized.h5ad"
    )
    notebook = Path(manager.export_notebook("nb", "S", filter_strategy="replay"))
    return workspace, notebook, workspace / "data" / f"{saved_as}.h5ad"


@needs_kernel
def test_g_a_loaded_object_is_recorded_by_content_not_by_path(tmp_path, monkeypatch):
    from lobster.core.provenance import content_digest

    workspace, notebook, saved = _geo_session(tmp_path, monkeypatch)
    (entry,) = _roles(notebook)["input_roles"]["loaded"]
    assert entry["modality"] == "geo_gse12345"
    assert entry["name"] == "geo_gse12345 (in memory)"
    assert len(entry["content_sha256"]) == 64 and entry["n_obs"] > 0
    assert (entry["digest"], entry["scope"]) == ("v2", content_digest.SCOPE)
    assert "sha256" not in entry and str(tmp_path) not in notebook.read_text("utf-8")


@needs_kernel
def test_g_the_autosaved_copy_of_a_downloaded_object_is_the_input_and_matches(
    tmp_path, monkeypatch
):
    workspace, notebook, saved = _geo_session(tmp_path, monkeypatch)
    result = _verify_real(workspace, notebook, saved)
    out = json.loads(result.stdout)
    assert result.exit_code == 0 and out["verdict"] == "MATCH", out


@needs_kernel
def test_g_an_edited_copy_of_a_downloaded_object_is_refused(tmp_path, monkeypatch):
    workspace, notebook, saved = _geo_session(tmp_path, monkeypatch)
    edited = ad.read_h5ad(saved)
    edited.X = edited.X + 1
    other = tmp_path / "edited.h5ad"
    _save(edited, other)
    result = _verify_real(workspace, notebook, other)
    assert result.exit_code == 4, result.output


@needs_kernel
def test_g_a_step_result_is_not_the_downloaded_object(tmp_path, monkeypatch):
    workspace, notebook, saved = _geo_session(tmp_path, monkeypatch)
    step_result = workspace / "data" / "geo_gse12345_normalized.h5ad"
    result = _verify_real(workspace, notebook, step_result)
    assert result.exit_code == 4, result.output


def test_g_the_content_digest_is_one_for_every_storage_of_the_same_matrix():
    import scipy.sparse as sp

    from lobster.core.provenance.content_digest import content_digest

    base = _adata()
    dense = np.asarray(base.X, dtype="float64")

    def make(x):
        other = base.copy()
        other.X = x
        return other

    same = [
        make(dense),
        make(np.asfortranarray(dense)),
        make(sp.csr_matrix(dense)),
        make(sp.csc_matrix(dense)),
        make(dense.astype("float32").astype("float64")),
    ]
    assert len({content_digest(a) for a in same}) == 1
    import hashlib

    from lobster.core.provenance.content_digest import _matrix

    def of(x):  # AnnData holds no COO: hash the matrix part alone
        h = hashlib.sha256()
        _matrix(h, x)
        return h.hexdigest()

    assert of(sp.coo_matrix(dense)) == of(sp.csr_matrix(dense)) == of(dense)
    # the same values held as float32: exact in float64, so the same digest
    exact = np.round(dense * 4) / 4
    assert content_digest(make(exact.astype("float32"))) == content_digest(
        make(exact.astype("float64"))
    )


def test_g_the_content_digest_changes_with_a_value_or_a_name():
    from lobster.core.provenance.content_digest import content_digest

    base = _adata()
    digests = {content_digest(base)}
    value = base.copy()
    value.X = np.asarray(value.X).copy()
    value.X[0, 0] += 1
    obs = base.copy()
    obs.obs_names = ["x"] + list(obs.obs_names[1:])
    var = base.copy()
    var.var_names = ["x"] + list(var.var_names[1:])
    digests |= {content_digest(a) for a in (value, obs, var)}
    assert len(digests) == 4


def test_g_the_content_digest_leaves_the_callers_matrix_alone():
    import scipy.sparse as sp

    from lobster.core.provenance.content_digest import content_digest

    adata = _adata()
    matrix = sp.csr_matrix(np.asarray(adata.X))
    matrix.data[0] = 0  # an explicit zero
    matrix.indices[:2] = matrix.indices[:2][::-1]  # unsorted
    adata.X = matrix
    before = (matrix.data.copy(), matrix.indices.copy(), matrix.indptr.copy())
    content_digest(adata)
    for kept, now in zip(before, (matrix.data, matrix.indices, matrix.indptr)):
        assert (kept == now).all()


# --- the saved-copy excuse and the in-memory digest ---------


def _qc_session(tmp_path, monkeypatch, axis, column):
    """A file load through the adapter; the user's own values in an adapter's column name."""
    from lobster.core.provenance.analysis_ir import AnalysisStep
    from lobster.core.runtime.data_manager import DataManagerV2

    _clean_env(monkeypatch, tmp_path)
    workspace = tmp_path / "ws"
    source = _adata()
    source.X = np.abs(source.X)
    raw = tmp_path / "raw.h5ad"
    _save(source, raw)
    manager = DataManagerV2(workspace_path=workspace, auto_scan=False)
    manager.load_modality("pbmc", raw, adapter="transcriptomics_single_cell")
    loaded = manager.get_modality("pbmc")
    new = loaded.copy()
    table = getattr(loaded, axis)
    if column in table.columns and table[column].dtype != bool:
        table[column] = table[column] + 1.0  # the user's values, not the adapter's
    else:
        table[column] = np.arange(len(table)) % 2 == 0
    manager.save_modality("pbmc", workspace / "data" / "pbmc.h5ad")
    new.X = np.log1p(new.X)
    step = AnalysisStep(
        operation="x", tool_name="normalize_log1p", description="d", library="scanpy",
        code_template=STEP, imports=["import numpy as np"], parameters={},
        parameter_schema={}, input_entities=["adata"], output_entities=["adata"],
        exportable=True,
    )  # fmt: skip
    manager.log_tool_usage("normalize_log1p", {}, "d", ir=step)
    manager.modalities["pbmc_normalized"] = new
    manager.save_modality(
        "pbmc_normalized", workspace / "data" / "pbmc_normalized.h5ad"
    )
    notebook = Path(manager.export_notebook("nb", "S", filter_strategy="replay"))
    return workspace, notebook, raw


QC_VARIANTS = [
    ("var", "mt"),
    ("var", "ribo"),
    ("var", "total_counts"),
    ("obs", "total_counts"),
    ("obs", "pct_counts_mt"),
    ("obs", "n_genes_by_counts"),
]


@needs_kernel
@pytest.mark.parametrize("axis, column", QC_VARIANTS)
def test_f_a_the_users_own_values_in_an_adapter_column_name_are_not_a_saved_copy(
    tmp_path, monkeypatch, axis, column
):
    workspace, notebook, raw = _qc_session(tmp_path, monkeypatch, axis, column)
    result = _verify_real(workspace, notebook, raw)
    out = json.loads(result.stdout)
    assert result.exit_code == 3 and out["verdict"] == "NOT_COMPARED", out
    assert any("not produced by the notebook" in n for n in out["notes"]), out["notes"]


@needs_kernel
def test_f_a_an_untouched_load_is_still_a_saved_copy_of_what_the_adapter_made(
    tmp_path, monkeypatch
):
    workspace, notebook, raw = _flow_session(tmp_path, monkeypatch, "n3")
    out = json.loads(_verify_real(workspace, notebook, raw).stdout)
    assert out["verdict"] == "MATCH", out
    assert any("Saved copy of the input" in n for n in out["notes"]), out["notes"]


def _object_session(
    tmp_path, monkeypatch, loaded, *, annotate=None, saved_as=None, carry=False
):
    """Load an AnnData object (no file), maybe annotate it in place, save, run a step."""
    from lobster.core.provenance.analysis_ir import AnalysisStep
    from lobster.core.runtime.data_manager import DataManagerV2

    _clean_env(monkeypatch, tmp_path)
    workspace = tmp_path / "ws"
    manager = DataManagerV2(workspace_path=workspace, auto_scan=False)
    manager.load_modality(
        "geo_x", loaded, adapter="transcriptomics_single_cell", validate=False
    )
    held = manager.get_modality("geo_x")
    new = held.copy()
    if annotate:
        annotate(held)
        if carry:  # the step starts from the edited object, as in a notebook replay
            new = held.copy()
    if saved_as:
        manager.modalities[saved_as] = held
        manager.save_modality(saved_as, workspace / "data" / f"{saved_as}.h5ad")
    new.X = np.log1p(new.X)
    step = AnalysisStep(
        operation="x", tool_name="normalize_log1p", description="d", library="scanpy",
        code_template=STEP, imports=["import numpy as np"], parameters={},
        parameter_schema={}, input_entities=["adata"], output_entities=["adata"],
        exportable=True,
    )  # fmt: skip
    manager.log_tool_usage("normalize_log1p", {}, "d", ir=step)
    manager.modalities["geo_x_norm"] = new
    manager.save_modality("geo_x_norm", workspace / "data" / "geo_x_norm.h5ad")
    return workspace, Path(manager.export_notebook("nb", "S", filter_strategy="replay"))


def _annotate(adata):
    adata.obs["cell_type"] = ["T" if i % 2 else "B" for i in range(adata.n_obs)]


def _pristine():
    obj = _adata()
    obj.X = np.abs(obj.X)
    return obj


@needs_kernel
@pytest.mark.parametrize("saved_as", ["geo_x_raw", "geo_x"])
def test_f_b1_b2_an_object_annotated_in_place_and_saved_is_not_the_loaded_input(
    tmp_path, monkeypatch, saved_as
):
    workspace, notebook = _object_session(
        tmp_path, monkeypatch, _pristine(), annotate=_annotate, saved_as=saved_as
    )
    result = _verify_real(workspace, notebook, workspace / "data" / f"{saved_as}.h5ad")
    assert result.exit_code == 4, result.output
    assert (
        "not a file the session loaded"
        in json.loads(result.stdout)["report"]["refusal"]["reason"]
    )


@needs_kernel
def test_f_b3_a_saved_unlogged_annotated_result_is_not_the_loaded_input(
    tmp_path, monkeypatch
):
    pristine = tmp_path / "pristine.h5ad"
    _save(_pristine(), pristine)
    workspace, notebook = _object_session(
        tmp_path, monkeypatch, _pristine(), annotate=_annotate, saved_as="geo_x_annot"
    )
    assert (
        _verify_real(
            workspace, notebook, workspace / "data" / "geo_x_annot.h5ad"
        ).exit_code
        == 4
    )
    # the real input is not refused (the unlogged annotated file keeps it from a MATCH)
    assert _verify_real(workspace, notebook, pristine).exit_code not in (0, 4)


def _exact(kind):
    return {
        "int64": (np.array([[2**53, 1], [0, 3]], dtype=np.int64),
                  np.array([[2**53 + 1, 1], [0, 3]], dtype=np.int64)),
        "uint64": (np.array([[2**64 - 1, 1], [0, 3]], dtype=np.uint64),
                   np.array([[2**64 - 2, 1], [0, 3]], dtype=np.uint64)),
        "complex": (np.array([[1 + 2j, 1], [0, 3]], dtype=np.complex128),
                    np.array([[1 + 3j, 1], [0, 3]], dtype=np.complex128)),
        "negative zero": (np.array([[-0.0, 1.0], [0.5, 3.0]]),
                          np.array([[0.0, 1.0], [0.5, 3.0]])),
    }[kind]  # fmt: skip


@needs_kernel
@pytest.mark.parametrize("kind", ["int64", "uint64", "complex", "negative zero"])
def test_f_c_values_the_old_digest_could_not_tell_apart_are_not_the_loaded_input(
    tmp_path, monkeypatch, kind
):
    loaded, twin = _exact(kind)
    workspace, notebook = _object_session(
        tmp_path, monkeypatch, ad.AnnData(loaded.copy())
    )
    same, other = tmp_path / "same.h5ad", tmp_path / "other.h5ad"
    ad.AnnData(loaded).write_h5ad(same)
    ad.AnnData(twin).write_h5ad(other)
    assert _verify_real(workspace, notebook, other).exit_code == 4
    assert _verify_real(workspace, notebook, same).exit_code in (0, 1)  # not refused


@needs_kernel
def test_g_in_memory_inputs_are_off_by_default_and_no_digest_is_made_at_load(
    tmp_path, monkeypatch
):
    from lobster.core.provenance import content_digest

    assert content_digest.IN_MEMORY_INPUTS is False  # the 1.3.0 shipping default
    monkeypatch.setattr(
        content_digest, "content_digest", lambda *a: pytest.fail("digest computed")
    )
    workspace, notebook, saved = _geo_session(tmp_path, monkeypatch)
    (entry,) = _roles(notebook)["input_roles"]["loaded"]
    assert "content_sha256" not in entry and entry["in_memory"] is True
    result = _verify_real(workspace, notebook, saved)
    out = json.loads(result.stdout)
    assert result.exit_code == 3 and out["verdict"] == "NOT_COMPARED", out
    assert any("not anchorable in this version" in n for n in out["notes"]), out[
        "notes"
    ]


@needs_kernel
def test_g_a_record_with_digest_v1_roles_is_not_compared_and_says_re_export(
    tmp_path, monkeypatch
):
    workspace, notebook, saved = _geo_session(tmp_path, monkeypatch)
    notebook.write_text(
        notebook.read_text("utf-8").replace('\\"version\\": 2', '\\"version\\": 1')
        if '\\"version\\": 2' in notebook.read_text("utf-8")
        else notebook.read_text("utf-8").replace('"version": 2', '"version": 1')
    )
    out = json.loads(_verify_real(workspace, notebook, saved).stdout)
    assert out["verdict"] == "NOT_COMPARED", out
    assert any("input roles v2" in n and "re-export" in n for n in out["notes"])


def test_g_the_content_digest_v2_covers_the_whole_object_but_uns():
    import pandas as pd
    import scipy.sparse as sp

    from lobster.core.provenance.content_digest import content_digest

    base = _adata()
    base.layers["counts"] = np.asarray(base.X).copy()
    base.obsm["pc"] = np.ones((base.n_obs, 2))
    base.obs["label"] = ["a", "b"] * (base.n_obs // 2) + ["a"] * (base.n_obs % 2)
    digest = content_digest(base)

    def changed(edit):
        other = base.copy()
        edit(other)
        return content_digest(other)

    assert changed(lambda a: a.layers["counts"].__setitem__((0, 0), 99.0)) != digest
    assert changed(lambda a: a.obsm["pc"].__setitem__((0, 0), 5.0)) != digest
    assert changed(lambda a: a.obs.__setitem__("label", "z")) != digest
    assert changed(lambda a: a.var.__setitem__("extra", 1)) != digest
    assert (
        changed(lambda a: a.uns.__setitem__("note", "x")) == digest
    )  # uns is left out
    # a column and its categorical copy (what an h5ad file holds) agree
    categorical = base.copy()
    categorical.obs["label"] = pd.Categorical(base.obs["label"])
    other_layout = base.copy()
    other_layout.layers["counts"] = sp.csc_matrix(base.layers["counts"])
    assert content_digest(categorical) == content_digest(other_layout) == digest


def test_g_the_content_digest_keeps_kinds_and_signs_apart():
    from lobster.core.provenance.content_digest import content_digest

    def of(array):
        return content_digest(ad.AnnData(array))

    assert of(np.array([[2**53, 1]], dtype=np.int64)) != of(
        np.array([[2**53 + 1, 1]], dtype=np.int64)
    )
    assert of(np.array([[1, 2]], dtype=np.int64)) != of(np.array([[1.0, 2.0]]))
    assert of(np.array([[-0.0, 1.0]])) != of(np.array([[0.0, 1.0]]))
    assert of(np.array([[1 + 2j]])) != of(np.array([[1 + 3j]]))
    assert of(np.array([[np.nan, 1.0]])) == of(np.array([[-np.nan, 1.0]]))


@needs_kernel
def test_f_a_the_users_own_values_in_an_adapter_column_name_are_not_a_saved_copy_catalog_on(
    tmp_path, monkeypatch
):
    from typer.testing import CliRunner

    from lobster.cli_internal.commands.light import lamin_commands
    from tests.unit.cli import test_replay_catalog as cat

    workspace, notebook, raw = _qc_session(tmp_path, monkeypatch, "var", "mt")
    fake = cat.FakeEngine(files={"pbmc": workspace / "data" / "pbmc.h5ad"})
    monkeypatch.setattr(lamin_commands, "_open", lambda *a, **k: fake)
    monkeypatch.setattr(
        replay_commands,
        "_catalog_record",
        lambda workspace: cat._ReplayRecord(workspace, None),
    )
    result = CliRunner().invoke(
        cat.app,
        ["--verify", str(notebook), "--input", str(raw), "-w", str(workspace), "--json"],
    )  # fmt: skip
    assert result.exit_code == 3, (result.output, result.exception)
    assert json.loads(result.stdout)["verdict"] == "NOT_COMPARED"


def _rewrite(notebook, old, new):
    """Edit a JSON fragment in the notebook text, quoted or escaped inside a cell."""
    text = notebook.read_text("utf-8")
    for form in (lambda s: s, lambda s: s.replace('"', '\\"')):
        if form(old) in text:
            notebook.write_text(text.replace(form(old), form(new)), "utf-8")
            return
    raise AssertionError(old)


@needs_kernel
@pytest.mark.parametrize(
    "new", ['"adapter": "no_such_adapter"', '"adapter": null'], ids=["unknown", "none"]
)
def test_f_a_a_load_adapter_that_cannot_be_run_leaves_the_copy_unexcused(
    tmp_path, monkeypatch, new
):
    workspace, notebook, raw = _flow_session(tmp_path, monkeypatch, "n3")
    _rewrite(notebook, '"adapter": "transcriptomics_single_cell"', new)
    out = json.loads(_verify_real(workspace, notebook, raw).stdout)
    assert out["verdict"] == "NOT_COMPARED", out


@needs_kernel
def test_g_an_object_entry_with_another_digest_version_is_not_compared(
    tmp_path, monkeypatch
):
    workspace, notebook, saved = _geo_session(tmp_path, monkeypatch)
    _rewrite(notebook, '"digest": "v2"', '"digest": "v1"')
    out = json.loads(_verify_real(workspace, notebook, saved).stdout)
    assert out["verdict"] == "NOT_COMPARED", out


# --- an in-place edit of the loaded object that no cell makes ---


def _p4_uns(a):
    a.uns["rank_genes_groups"] = {"names": np.array(["g1", "g2"])}


def _p4_float64(a):
    a.X = np.asarray(a.X).astype("float64")


def _p4_csr(a):
    import scipy.sparse as sp

    a.X = sp.csr_matrix(a.X)


def _p4_obs_reorder(a):
    a.obs["a"], a.obs["b"] = 1, 2
    a.obs = a.obs[["b", "a"] + [c for c in a.obs.columns if c not in "ab"]]


def _p4_category_reorder(a):
    import pandas as pd

    a.obs["c"] = pd.Categorical(["x", "y"] * (a.n_obs // 2) + ["x"] * (a.n_obs % 2))
    a.obs["c"] = a.obs["c"].cat.reorder_categories(["y", "x"])


def _p4_category_unused(a):
    import pandas as pd

    a.obs["c"] = pd.Categorical(["x"] * a.n_obs, categories=["x", "unused"])


def _p4_category_ordered(a):
    import pandas as pd

    a.obs["c"] = pd.Categorical(
        ["x", "y"] * (a.n_obs // 2) + ["x"] * (a.n_obs % 2), ordered=True
    )


def _p4_int_width(a):
    a.obs["n"] = np.arange(a.n_obs, dtype=np.int8)


P4 = [_p4_uns, _p4_float64, _p4_csr, _p4_obs_reorder, _p4_category_reorder,
      _p4_category_unused, _p4_category_ordered, _p4_int_width]  # fmt: skip


def _p4_verdict(tmp_path, monkeypatch, mutate):
    workspace, notebook = _object_session(
        tmp_path,
        monkeypatch,
        _pristine(),
        annotate=mutate,
        saved_as="geo_x",
        carry=True,
    )
    return _verify_real(workspace, notebook, workspace / "data" / "geo_x.h5ad")


@needs_kernel
@pytest.mark.parametrize("mutate", P4, ids=lambda f: f.__name__[4:])
def test_p4_an_in_place_edit_of_a_loaded_object_is_not_a_match_while_the_switch_is_off(
    tmp_path, monkeypatch, mutate
):
    result = _p4_verdict(tmp_path, monkeypatch, mutate)
    assert result.exit_code == 3, result.output
    assert any(
        "not anchorable in this version" in n
        for n in json.loads(result.stdout)["notes"]
    )


def _p4_on(tmp_path, monkeypatch, mutate):
    from lobster.core.provenance import content_digest

    monkeypatch.setattr(content_digest, "IN_MEMORY_INPUTS", True)
    return _p4_verdict(tmp_path, monkeypatch, mutate).exit_code


@needs_kernel
@pytest.mark.xfail(strict=True, reason="in-memory anchoring: 1.3.1")
@pytest.mark.parametrize(
    "mutate", [_p4_uns, _p4_float64, _p4_csr], ids=lambda f: f.__name__[4:]
)
def test_the_in_memory_digest_tells_an_in_place_edit_from_the_loaded_object(
    tmp_path, monkeypatch, mutate
):
    assert _p4_on(tmp_path, monkeypatch, mutate) != 0  # a false MATCH today


@needs_kernel
@pytest.mark.parametrize(
    "mutate",
    [
        _p4_obs_reorder,
        _p4_category_reorder,
        _p4_category_unused,
        _p4_category_ordered,
        _p4_int_width,
    ],
    ids=lambda f: f.__name__[4:],
)
def test_the_in_memory_digest_edits_that_the_round_trip_shows_are_not_a_match(
    tmp_path, monkeypatch, mutate
):
    assert _p4_on(tmp_path, monkeypatch, mutate) != 0


@needs_kernel
@pytest.mark.parametrize("which", ["edited copy", "other dataset"])
def test_p4_an_unanchorable_in_memory_input_is_not_compared_whatever_the_outputs_show(
    tmp_path, monkeypatch, which
):
    workspace, notebook, saved = _geo_session(tmp_path, monkeypatch)
    other = tmp_path / "other.h5ad"
    if which == "edited copy":
        edited = ad.read_h5ad(saved)
        edited.X = edited.X + 1.0
        edited.write_h5ad(other)
    else:
        _save(_adata(shift=3.0), other)
    result = _verify_real(workspace, notebook, other)
    out = json.loads(result.stdout)
    assert result.exit_code == 3 and out["verdict"] == "NOT_COMPARED", out
    assert any("not anchorable in this version" in n for n in out["notes"])


# --- a modality the session never saved is not a missing file ----------------


@needs_kernel
def test_g3_the_record_says_not_saved_for_a_modality_nobody_saved(
    tmp_path, monkeypatch
):
    workspace, notebook, raw = _flow_session(tmp_path, monkeypatch, "n1", unsaved=True)
    manifest = _roles(notebook)
    files = manifest["input_files"]
    assert files["pbmc.h5ad"] == "not_saved" and files["pbmc_qc.h5ad"] == "not_saved"
    assert len(files["pbmc_normalized.h5ad"]) == 64
    assert manifest["input_roles"]["saved"] == files


@needs_kernel
def test_g3_a_saved_file_that_is_gone_is_still_file_not_found(tmp_path, monkeypatch):
    workspace, notebook, raw = _flow_session(tmp_path, monkeypatch, "n1", lost=True)
    files = _roles(notebook)["input_files"]
    assert files["pbmc_normalized.h5ad"] == "file_not_found"
    assert files["pbmc.h5ad"] == "not_saved"


@needs_kernel
def test_g3_a_session_with_a_modality_nobody_saved_reaches_a_match(
    tmp_path, monkeypatch
):
    workspace, notebook, raw = _flow_session(tmp_path, monkeypatch, "n1", unsaved=True)
    result = _verify_real(workspace, notebook, raw)
    out = json.loads(result.stdout)
    assert result.exit_code == 0 and out["verdict"] == "MATCH", out
    assert not any("has no hash" in n for n in out["notes"]), out["notes"]


@needs_kernel
def test_g3_the_saved_file_edited_after_the_session_is_refused(tmp_path, monkeypatch):
    workspace, notebook, raw = _flow_session(tmp_path, monkeypatch, "n1", unsaved=True)
    saved = workspace / "data" / "pbmc_normalized.h5ad"
    edited = ad.read_h5ad(saved)
    edited.X = edited.X + 1.0
    edited.write_h5ad(saved)
    assert _verify_real(workspace, notebook, raw).exit_code == 4


@needs_kernel
def test_g3_an_old_record_without_the_marker_is_read_as_before(tmp_path, monkeypatch):
    workspace, notebook, raw = _flow_session(tmp_path, monkeypatch, "n1", unsaved=True)
    _rewrite(notebook, '"not_saved"', '"file_not_found"')  # what an old export wrote
    out = json.loads(_verify_real(workspace, notebook, raw).stdout)
    assert out["verdict"] == "NOT_COMPARED", out
    assert any("has no hash for: pbmc_qc.h5ad" in n for n in out["notes"]), out["notes"]


def _without_session_writer(monkeypatch, tmp_path):
    """Make the notebook kernel (only) fail to import the session writer."""
    site = tmp_path / "kernel_site"
    site.mkdir()
    (site / "sitecustomize.py").write_text(
        "import sys\nsys.modules['lobster.core.backends.h5ad_backend'] = None\n"
    )
    monkeypatch.setenv(
        "PYTHONPATH",
        os.pathsep.join(filter(None, [str(site), os.environ.get("PYTHONPATH")])),
    )


@needs_kernel
def test_w1_a_notebook_that_wrote_without_the_session_writer_says_so(
    tmp_path, monkeypatch
):
    workspace, notebook, raw = _flow_session(tmp_path, monkeypatch, "n1")
    _without_session_writer(monkeypatch, tmp_path)
    result = _verify_real(workspace, notebook, raw)
    out = json.loads(result.stdout)
    assert out["verdict"] == "MATCH" and result.exit_code == 0, out  # verdict unchanged
    assert "The notebook wrote without the session writer." in out["notes"], out[
        "notes"
    ]
    executed = json.dumps(
        json.loads(next(workspace.rglob("executed.ipynb")).read_text())
    )
    assert "Written without the session writer" in executed


@needs_kernel
def test_w2_a_notebook_with_the_session_writer_still_matches(tmp_path, monkeypatch):
    workspace, notebook, raw = _flow_session(tmp_path, monkeypatch, "n1")
    out = json.loads(_verify_real(workspace, notebook, raw).stdout)
    assert out["verdict"] == "MATCH" and not any(
        "session writer" in n for n in out["notes"]
    ), out


@needs_kernel
def test_w3_a_notebook_without_the_marker_and_without_the_fallback_is_unchanged(
    tmp_path, monkeypatch
):
    workspace, notebook, raw = _flow_session(tmp_path, monkeypatch, "n1")
    nb = json.loads(notebook.read_text())
    for cell in nb["cells"]:
        src = "".join(cell["source"])
        if "write_session_h5ad" in src:  # an old notebook: the plain write
            cell["source"] = 'adata.write_h5ad(output_path, compression="gzip")\n'
    notebook.write_text(json.dumps(nb))
    out = json.loads(_verify_real(workspace, notebook, raw).stdout)
    assert not any("session writer" in n for n in out["notes"]), out["notes"]
