"""``lobster replay --verify``: re-run an exported notebook, compare its output with
the session's saved file, exit by verdict. No LaminDB needed.

Most tests run a real notebook through papermill (about two seconds each); they
are skipped where the ``python3`` kernel is not installed.
"""

import ast
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import anndata as ad
import nbformat
import numpy as np
import pytest
import typer
from typer.testing import CliRunner

from lobster.cli_internal.commands.light import replay_commands
from lobster.cli_internal.commands.light.replay_commands import (
    EXIT_CODES,
    JsonUsageCommand,
    replay_command,
)
from lobster.i18n import set_lang
from lobster.testing.replay_contract import (
    ComparisonPolicy,
    Execution,
    Preflight,
    ReplayReport,
    Verdict,
    canonical_sha256,
)

pytestmark = pytest.mark.unit

app = typer.Typer()
app.command(cls=JsonUsageCommand)(replay_command)
runner = CliRunner()


def _built(path) -> str:
    """Python that builds ``path`` while the cell runs. The replay refuses a path
    written in the code before the run; these rows are about what the run itself
    sees, so their paths are made at run time."""
    parts = Path(path).parts
    if parts and parts[0] == os.sep:
        return (
            f"__import__('os').sep + __import__('os').path.join(*{list(parts[1:])!r})"
        )
    words = ("__import__('os').pardir" if p == ".." else repr(p) for p in parts)
    return f"__import__('os').path.join({', '.join(words)})"


def _kernel_installed() -> bool:
    try:
        from jupyter_client.kernelspec import KernelSpecManager

        KernelSpecManager().get_kernel_spec("python3")
        return True
    except Exception:
        return False


needs_kernel = pytest.mark.skipif(
    not _kernel_installed(), reason="the python3 Jupyter kernel is not installed"
)

COPY = """\
import anndata as ad
adata = ad.read_h5ad(input_data)
adata.X = adata.X + 0.25
adata.write_h5ad(f"{output_prefix}_pbmc.h5ad")
"""
SHIFTED = """\
import anndata as ad
adata = ad.read_h5ad(input_data)
adata.X = adata.X + 0.25
adata.X = adata.X + 1
adata.write_h5ad(f"{output_prefix}_pbmc.h5ad")
"""
NUDGED = """\
import anndata as ad
adata = ad.read_h5ad(input_data)
adata.X = adata.X + 0.25
adata.X = adata.X * (1 + 1e-5)
adata.write_h5ad(f"{output_prefix}_pbmc.h5ad")
"""
WITHOUT_RECORD = """\
import anndata as ad
adata = ad.read_h5ad(input_data)
adata.X = adata.X + 0.25
adata.uns.pop("provenance", None)
adata.write_h5ad(f"{output_prefix}_pbmc.h5ad")
"""
OTHER_NAME = """\
import anndata as ad
adata = ad.read_h5ad(input_data)
adata.X = adata.X + 0.25
adata.write_h5ad(f"{output_prefix}_unrelated.h5ad")
"""
BOOM = 'raise RuntimeError("boom")\n'
NOTHING = "pass\n"
ODD_NAME = """\
import anndata as ad
adata = ad.read_h5ad(input_data)
adata.X = adata.X + 0.25
adata.write_h5ad(f"{output_prefix}_a[1].h5ad")
"""


# What the notebooks of these tests add to their input, and so what the session saved.
SESSION_SHIFT = 0.25


def _adata(shift=0.0, record=False):
    rng = np.random.default_rng(0)
    adata = ad.AnnData(rng.random((120, 6), dtype=np.float32) + shift)
    adata.obs_names = [f"c{i}" for i in range(120)]
    adata.var_names = [f"g{i}" for i in range(6)]
    if record:  # what a session writes into its own file; the default policy skips it
        adata.uns["provenance"] = {"session": "s1"}
    return adata


def _save(adata, path: Path) -> None:
    # tests/conftest.py (like lobster/cli.py) turns pandas' string inference off,
    # which anndata only writes with this opt-in.
    with ad.settings.override(allow_write_nullable_strings=True):
        adata.write_h5ad(path)


def _save_with_the_data_manager(ws: Path, adata, name: str = "pbmc") -> None:
    """The session's own save: index names, string layout and provenance record."""
    from lobster.core.runtime.data_manager import DataManagerV2

    manager = DataManagerV2(workspace_path=ws, auto_scan=False)
    manager.modalities[name] = adata
    manager.save_modality(name, ws / "data" / f"{name}.h5ad")


SAVED = object()  # the export record holds the hash of the workspace's saved file


LOADED = object()  # the record says the session loaded the input file (``input.h5ad``)


def _loaded_entry(path: Path, modality="pbmc") -> dict:
    data = path.read_bytes()
    return {
        "modality": modality,
        "name": path.name,
        "sha256": hashlib.sha256(data).hexdigest(),
        "size": len(data),
    }


def _export_record(hashes: dict, loaded=None) -> str:
    """``loaded``: None = a record from before input roles, or the loaded entries."""
    manifest = {"input_files": hashes, "system": {}}
    if loaded is not None:
        manifest["input_roles"] = {"version": 2, "loaded": loaded, "saved": hashes}
    record = {"data_integrity_manifest": manifest}
    return (
        f"## Export record (hashes)\n\n```json\n{json.dumps(record, indent=2)}\n```\n"
    )


def _notebook(
    path: Path, body: str, parameters: bool = True, hashes=SAVED, loaded=LOADED
) -> Path:
    """A notebook as the exporter writes it; ``hashes`` is its export record."""
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python"}
    nb.metadata["lobster"] = {"source_session_id": "s1", "lobster_version": "1.3.0"}
    cells = []
    if hashes is SAVED:
        folders = (path.parent.parent / "data", path.parent.parent)
        saved = [f / "pbmc.h5ad" for f in folders if (f / "pbmc.h5ad").is_file()]
        hashes = (
            {"pbmc.h5ad": hashlib.sha256(saved[0].read_bytes()).hexdigest()}
            if saved
            else {}
        )
    if loaded is LOADED:
        raw = path.parent.parent.parent / "input.h5ad"
        loaded = [_loaded_entry(raw)] if raw.is_file() else []
    if hashes is not None:
        cells.append(nbformat.v4.new_markdown_cell(_export_record(hashes, loaded)))
    if parameters:
        cells.append(
            nbformat.v4.new_code_cell(
                (
                    parameters
                    if isinstance(parameters, str)
                    else "input_data = 'dataset.h5ad'\noutput_prefix = 'results'"
                ),
                metadata={"tags": ["parameters"]},
            )
        )
    cells.append(nbformat.v4.new_code_cell(body))
    nb.cells = cells
    path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, path)
    return path


@pytest.fixture
def ws(tmp_path, monkeypatch):
    for name in ("LOBSTER_LAMIN", "LOBSTER_LAMIN_INSTANCE", "LAMIN_CURRENT_INSTANCE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(
        "lobster.core.catalog.factory._global_config_path",
        lambda: tmp_path / "no-providers.json",
    )
    path = tmp_path / ".lobster_workspace"
    (path / "data").mkdir(parents=True)
    # the session's saved file: it carries the session's own record, so its bytes are
    # not the input's (a replay started from its own saved result is refused)
    _save(_adata(shift=SESSION_SHIFT, record=True), path / "data" / "pbmc.h5ad")
    _save(_adata(), tmp_path / "input.h5ad")
    return path


def _files(root: Path):
    return sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify(ws, body, *extra, notebook=None, parameters=True, as_json=True):
    nb = notebook or _notebook(ws / "notebooks" / "nb.ipynb", body, parameters)
    args = ["--verify", str(nb), "--input", str(ws.parent / "input.h5ad")]
    args += ["-w", str(ws), *extra]
    if as_json:
        args.append("--json")
    return runner.invoke(app, args)


def _replay_dirs(ws):
    root = ws / "replays"
    return sorted(root.iterdir()) if root.is_dir() else []


def _report(ws) -> ReplayReport:
    (folder,) = _replay_dirs(ws)
    return ReplayReport.from_json((folder / "replay_report.json").read_text())


def test_exit_codes_follow_the_verdicts():
    # 2 is the usage error of the whole CLI: no verdict uses it
    assert EXIT_CODES == {
        Verdict.MATCH: 0,
        Verdict.MISMATCH: 1,
        Verdict.NOT_COMPARED: 3,
        Verdict.REFUSED: 4,
    }
    assert replay_commands.EXIT_NOTEBOOK_FAILED == 5 and replay_commands.EXIT_USAGE == 2


@needs_kernel
def test_identical_output_is_a_match_and_leaves_the_workspace_data_alone(ws):
    nb = _notebook(ws / "notebooks" / "nb.ipynb", COPY)
    before = {p: _sha(ws / p) for p in _files(ws)}
    result = _verify(ws, COPY, notebook=nb)
    assert result.exit_code == 0, result.output
    out = json.loads(result.stdout)
    assert out["verdict"] == "MATCH" and out["exit_code"] == 0
    # the replay's files sit under replays/<id>/, nothing else changed
    (folder,) = _replay_dirs(ws)
    assert len(folder.name) == 16
    assert out["replay_dir"] == str(folder)
    assert {"out_pbmc.h5ad", "executed.ipynb", "replay_report.json"} <= {
        p.name for p in folder.iterdir()
    }
    after = {p: _sha(ws / p) for p in before}
    assert after == before
    assert [p for p in _files(ws) if not p.startswith("replays/")] == sorted(before)
    report = _report(ws)
    assert report.claim_achieved is None  # not isolated: certifies nothing
    assert report.scenario_id == "nb"
    assert report.verdict is Verdict.MATCH and report.preflight is Preflight.ADMITTED
    assert report.execution is Execution.OK and report.coverage_complete
    assert ReplayReport.from_dict(out["report"]).to_dict() == report.to_dict()


@needs_kernel
def test_changed_values_are_a_mismatch(ws):
    result = _verify(ws, SHIFTED)
    assert result.exit_code == 1, result.output
    out = json.loads(result.stdout)
    assert out["verdict"] == "MISMATCH" and out["exit_code"] == 1
    slot = out["report"]["slots"][0]
    assert slot["path"] == "files" and slot["status"] == "MISMATCH"
    assert any("pbmc.h5ad/X" in d for d in slot["diffs"])


@needs_kernel
def test_the_sessions_own_record_is_not_expected_back_by_default(ws):
    reference = _adata(shift=SESSION_SHIFT)
    reference.uns["provenance"] = {"activities": "session record"}
    reference.uns["lineage"] = {"pbmc": "from raw"}
    _save(reference, ws / "data" / "pbmc.h5ad")
    result = _verify(ws, WITHOUT_RECORD)
    assert result.exit_code == 0, result.output
    report = _report(ws)
    assert {e.path for e in report.excluded} == {
        "files/pbmc.h5ad/uns/provenance",
        "files/pbmc.h5ad/uns/lineage",
    }


@needs_kernel
def test_a_policy_file_replaces_the_default(ws, tmp_path):
    reference = _adata(shift=SESSION_SHIFT)
    reference.uns["provenance"] = {"activities": "session record"}
    _save(reference, ws / "data" / "pbmc.h5ad")
    strict = tmp_path / "strict.json"
    strict.write_text(json.dumps({"schema_version": 1, "required": ["files"]}))
    result = _verify(ws, WITHOUT_RECORD, "--policy", str(strict))
    assert result.exit_code == 1, result.output
    out = json.loads(result.stdout)
    assert out["report"]["excluded"] == []
    policy = ComparisonPolicy.from_json(strict.read_text())
    assert out["report"]["bindings"]["policy_sha256"] == policy.digest()


@needs_kernel
def test_a_tolerance_in_the_policy_applies(ws, tmp_path):
    # a policy of its own replaces the default, so the session's record is written too
    nudged = NUDGED.replace(
        "adata.write_h5ad",
        'adata.uns["provenance"] = {"session": "s1"}\nadata.write_h5ad',
    )
    exact = _verify(ws, nudged)
    assert exact.exit_code == 1
    loose = tmp_path / "loose.json"
    loose.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "required": ["files"],
                "tolerances": [{"path": "files/pbmc.h5ad/X", "rtol": 1e-4}],
            }
        )
    )
    result = _verify(ws, nudged, "--policy", str(loose))
    # a match under a policy of the user's own is never a plain MATCH
    assert result.exit_code == 6, result.output
    assert json.loads(result.stdout)["verdict"] == "MATCH_CUSTOM_POLICY"


@needs_kernel
def test_bindings_are_recomputed_from_the_files(ws):
    nb = _notebook(ws / "notebooks" / "nb.ipynb", COPY)
    result = _verify(ws, COPY, notebook=nb)
    bindings = json.loads(result.stdout)["report"]["bindings"]
    again = json.loads(_verify(ws, COPY, notebook=nb).stdout)["report"]["bindings"]
    assert again == bindings  # the same replay binds the same way
    assert bindings["params_sha256"] == canonical_sha256(
        {"papermill": {"input_data": "input.h5ad", "output_prefix": "out"}, "tools": {}}
    )  # machine paths stay out of the binding
    assert bindings["notebook_sha256"] == _sha(nb)
    assert bindings["recipe_sha256"] == canonical_sha256(
        json.loads(nb.read_text())["metadata"]["lobster"]
    )
    assert bindings["root_sha256"] == {"input.h5ad": _sha(ws.parent / "input.h5ad")}
    assert set(bindings) == {
        "notebook_sha256",
        "recipe_sha256",
        "root_sha256",
        "policy_sha256",
        "params_sha256",
        "source_revision",
        "source_dirty",
        "env_receipt_sha256",
    }
    assert isinstance(bindings["source_dirty"], bool)
    assert len(bindings["source_revision"]) == 40


@needs_kernel
def test_a_reference_saved_by_the_data_manager_matches(ws):
    (ws / "data" / "pbmc.h5ad").unlink()
    _save_with_the_data_manager(ws, _adata(shift=SESSION_SHIFT))
    saved = ad.read_h5ad(ws / "data" / "pbmc.h5ad")
    # what the file holds that a plain anndata write does not: in a session, the
    # provenance record (an unnamed index stays unnamed, as in a plain write)
    assert saved.obs.index.name is None and saved.var.index.name is None
    result = _verify(ws, COPY)
    assert result.exit_code == 0, result.output


@needs_kernel
def test_the_reference_may_be_a_file_in_the_workspace_root(ws):
    (ws / "data" / "pbmc.h5ad").rename(ws / "pbmc.h5ad")
    result = _verify(ws, COPY)
    assert result.exit_code == 0, result.output


@needs_kernel
def test_an_output_without_a_saved_file_is_not_compared(ws):
    result = _verify(ws, OTHER_NAME)
    assert result.exit_code == 3, result.output
    out = json.loads(result.stdout)
    assert out["verdict"] == "NOT_COMPARED"
    report = _report(ws)
    assert report.execution is Execution.OK and report.slots == ()
    assert report.verification_status.value == "not_run"
    assert any("unrelated" in note for note in out["notes"])


@needs_kernel
def test_a_notebook_that_writes_nothing_is_not_compared(ws):
    result = _verify(ws, NOTHING)
    assert result.exit_code == 3, result.output
    assert "The notebook wrote no .h5ad file." in json.loads(result.stdout)["notes"]
    assert _report(ws).slots == ()


@needs_kernel
def test_an_output_whose_name_cannot_be_a_policy_path_is_not_compared(ws):
    _save(_adata(), ws / "data" / "a[1].h5ad")  # it even has a saved file
    result = _verify(ws, ODD_NAME)
    assert result.exit_code == 3, result.output
    assert any("a[1].h5ad" in n for n in json.loads(result.stdout)["notes"])


@needs_kernel
def test_a_saved_file_that_cannot_be_read_is_not_compared(ws):
    (ws / "data" / "pbmc.h5ad").write_text("not an h5ad file")
    result = _verify(ws, COPY)
    assert result.exit_code == 3, result.output
    assert any("could not be read" in n for n in json.loads(result.stdout)["notes"])


@needs_kernel
def test_a_notebook_that_fails_is_its_own_verdict_not_a_mismatch(ws):
    result = _verify(ws, BOOM)
    assert result.exit_code == 5, result.output
    assert json.loads(result.stdout)["verdict"] == "NOTEBOOK_FAILED"
    report = _report(ws)
    assert report.execution is Execution.ERROR
    assert report.verdict is Verdict.NOT_COMPARED and report.slots == ()


@needs_kernel
def test_a_syntax_error_in_the_notebook_is_not_exit_1(ws):
    result = _verify(ws, "def broken(:\n")
    assert result.exit_code == 5, result.output
    assert "Verdict: NOTEBOOK_FAILED" in _verify(ws, "x = (", as_json=False).stdout


@needs_kernel
def test_a_notebook_without_the_parameters_is_refused_before_it_runs(ws):
    result = _verify(ws, COPY, parameters=False)
    assert result.exit_code == 4, result.output
    out = json.loads(result.stdout)
    assert out["verdict"] == "REFUSED"
    report = _report(ws)
    assert report.preflight is Preflight.REFUSED
    assert report.execution is Execution.NOT_RUN and report.slots == ()
    assert "output_prefix" in report.refusal.reason
    assert report.refusal.origin.value == "harness"
    assert not (_replay_dirs(ws)[0] / "executed.ipynb").exists()


def test_a_missing_input_is_refused_without_a_report(ws):
    nb = _notebook(ws / "notebooks" / "nb.ipynb", COPY)
    result = runner.invoke(
        app, ["--verify", str(nb), "--input", str(ws / "nope.h5ad"), "-w", str(ws)]
    )
    assert result.exit_code == 2
    assert "nope.h5ad" in result.output
    assert _replay_dirs(ws) == []


def test_no_input_is_refused_without_a_report(ws):
    nb = _notebook(ws / "notebooks" / "nb.ipynb", COPY)
    result = runner.invoke(app, ["--verify", str(nb), "-w", str(ws)])
    assert result.exit_code == 2
    assert "--input" in result.output
    assert _replay_dirs(ws) == []


def test_a_missing_notebook_is_refused(ws):
    result = runner.invoke(
        app,
        [
            "--verify",
            str(ws / "notebooks" / "gone.ipynb"),
            "--input",
            str(ws.parent / "input.h5ad"),
            "-w",
            str(ws),
        ],
    )
    assert result.exit_code == 2
    assert "gone.ipynb" in result.output
    assert _replay_dirs(ws) == []


def test_a_notebook_that_is_not_a_notebook_is_refused(ws):
    bad = ws / "notebooks" / "bad.ipynb"
    bad.parent.mkdir()
    bad.write_text("not json")
    result = _verify(ws, "", notebook=bad)
    assert result.exit_code == 2
    assert _replay_dirs(ws) == []


def test_a_policy_that_breaks_the_contract_is_refused(ws, tmp_path):
    nb = _notebook(ws / "notebooks" / "nb.ipynb", COPY)
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"required": []}))
    result = _verify(ws, COPY, "--policy", str(bad), notebook=nb)
    assert result.exit_code == 2
    assert "bad.json" in result.output
    assert _replay_dirs(ws) == []


def test_executor_results_map_to_the_report(ws, monkeypatch):
    """A stubbed executor: validation failure, timeout and error mapping."""
    from lobster.core.notebooks import executor

    answers = {
        "validation_failed": {"status": "validation_failed", "errors": ["bad input"]},
        "timeout": {
            "status": "failed",
            "error": "too slow",
            "error_type": "CellTimeoutError",
        },
    }
    state = {}
    monkeypatch.setattr(
        executor.NotebookExecutor, "execute", lambda self, *a, **k: state["answer"]
    )
    nb = _notebook(ws / "notebooks" / "nb.ipynb", COPY)
    strict = ws.parent / "strict.json"
    strict.write_text(
        json.dumps(
            {"schema_version": 1, "required": ["files"], "sparse": "representation"}
        )
    )
    digest = ComparisonPolicy.from_json(strict.read_text()).digest()
    state["answer"] = answers["validation_failed"]
    result = _verify(ws, COPY, "--policy", str(strict), notebook=nb)
    assert result.exit_code == 4
    assert "bad input" in _report(ws).refusal.reason
    state["answer"] = answers["timeout"]
    result = _verify(ws, COPY, "--policy", str(strict), notebook=nb)
    assert result.exit_code == 5
    reports = [
        ReplayReport.from_json((d / "replay_report.json").read_text())
        for d in _replay_dirs(ws)
    ]
    assert sorted(r.execution.value for r in reports) == ["NOT_RUN", "TIMEOUT"]
    # a report that compared nothing is still bound to the policy that was asked for
    assert {r.bindings["policy_sha256"] for r in reports} == {digest}


@needs_kernel
def test_the_text_output_names_the_verdict_and_where_the_files_are(ws):
    result = _verify(ws, SHIFTED, as_json=False)
    assert result.exit_code == 1
    assert "MISMATCH" in result.stdout
    assert str(_replay_dirs(ws)[0]) in result.stdout


@needs_kernel
def test_the_text_output_in_chinese(ws):
    set_lang("zh")
    try:
        result = _verify(ws, COPY, as_json=False)
    finally:
        set_lang("en")
    assert result.exit_code == 0
    assert "结论：MATCH" in result.stdout


def test_every_note_has_a_chinese_line():
    root = Path(__file__).parents[3] / "lobster"
    phrases = json.loads((root / "i18n" / "phrases.zh.json").read_text("utf-8"))
    tree = ast.parse(Path(replay_commands.__file__).read_text())
    # notes are translated when printed, so the scan of h() calls does not see them
    notes = [
        node.elts[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Tuple)
        and len(node.elts) == 2
        and isinstance(node.elts[0], ast.Constant)
        and isinstance(node.elts[0].value, str)
        and isinstance(node.elts[1], ast.Dict)
    ]
    assert len(notes) == 29, notes
    assert [n for n in notes if n not in phrases] == []


def test_without_a_checkout_the_revision_is_a_fingerprint_reported_dirty(
    tmp_path, monkeypatch
):
    from lobster.version import __version__

    monkeypatch.setattr(replay_commands, "_checkout_root", lambda: tmp_path)
    replay_commands._source_revision.cache_clear()
    try:
        revision, dirty = replay_commands._source_revision()
    finally:
        replay_commands._source_revision.cache_clear()
    assert revision == hashlib.sha1(f"lobster-ai=={__version__}".encode()).hexdigest()
    assert dirty is True


def test_in_a_checkout_the_revision_is_its_commit_and_dirt_is_tracked_changes(
    tmp_path, monkeypatch
):
    def git(*args):
        return subprocess.run(
            ["git", "-c", "user.name=t", "-c", "user.email=t@example.org", *args],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

    git("init", "-q")
    (tmp_path / "f.txt").write_text("one")
    git("add", "f.txt")
    git("commit", "-q", "-m", "one")
    monkeypatch.setattr(replay_commands, "_checkout_root", lambda: tmp_path)

    def read():
        replay_commands._source_revision.cache_clear()
        try:
            return replay_commands._source_revision()
        finally:
            replay_commands._source_revision.cache_clear()

    assert read() == (git("rev-parse", "HEAD"), False)
    (tmp_path / "untracked.txt").write_text("x")
    assert read()[1] is False
    (tmp_path / "f.txt").write_text("two")
    assert read() == (git("rev-parse", "HEAD"), True)


def test_the_command_is_registered_in_the_cli():
    from lobster.cli import app as lobster_app

    names = {c.name or c.callback.__name__ for c in lobster_app.registered_commands}
    assert "replay" in names
    assert (
        len((Path(__file__).parents[3] / "lobster" / "cli.py").read_text().splitlines())
        <= 1800
    )


def test_replay_never_loads_the_catalog_engine_or_the_notebook_stack():
    code = (
        "import sys\n"
        "import lobster.cli_internal.commands.light.replay_commands\n"
        "bad = [m for m in ('lamindb', 'lamindb_setup', 'papermill', 'anndata') "
        "if m in sys.modules]\n"
        "print(bad)\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "[]"


def test_module_names_its_marked_decisions():
    text = Path(replay_commands.__file__).read_text()
    assert (
        "# Decision (Nov 2026): no tolerance" in text
    )  # tolerances wait for the noise-floor runs


# --- a MATCH needs everything it stands on ----------------------------------------

TWO_OUTPUTS = """\
import anndata as ad
adata = ad.read_h5ad(input_data)
adata.X = adata.X + 0.25
adata.write_h5ad(f"{output_prefix}_pbmc.h5ad")
adata.write_h5ad(f"{output_prefix}_extra.h5ad")
"""


def _writing(body: str) -> str:
    return "import anndata as ad\nfrom pathlib import Path\n" + body


def _invariant_case(ws, case):
    """Set one condition of the MATCH invariant up; return the notebook body."""
    if case == "all hold":
        return COPY
    if case == "output without a reference":
        return TWO_OUTPUTS
    if case == "reference edited":
        return "EDIT"
    if case == "slot not compared":
        _save(_adata(), ws / "data" / "extra.h5ad")
        (ws / "data" / "extra.h5ad").write_bytes(b"not an h5ad file")
        return TWO_OUTPUTS
    if case == "wrote to data/":
        return _writing(
            f'Path({_built(ws / "data" / "new.txt")}).write_text("x")\n' + COPY
        )
    if case == "wrote to its input":
        return _writing(
            f'Path({_built(ws.parent / "input.h5ad")}).write_bytes(b"changed")\n'
        )
    raise AssertionError(case)


# (the four conditions, and what breaking each gives)
INVARIANT = [
    ("all hold", 0, "MATCH"),
    ("output without a reference", 3, "NOT_COMPARED"),
    ("reference edited", 4, "REFUSED"),
    ("slot not compared", 3, "NOT_COMPARED"),
    ("wrote to data/", 4, "REFUSED"),
    ("wrote to its input", 4, "REFUSED"),
]


@needs_kernel
@pytest.mark.parametrize("case,code,verdict", INVARIANT)
def test_match_only_if_every_output_was_compared_against_the_bytes_the_session_saved(
    ws, case, code, verdict
):
    body = _invariant_case(ws, case)
    nb = _notebook(ws / "notebooks" / "nb.ipynb", COPY if body == "EDIT" else body)
    if body == "EDIT":  # the saved file changes after the session (and its record)
        _save(_adata(shift=0.5), ws / "data" / "pbmc.h5ad")
    result = _verify(ws, body, notebook=nb)
    assert result.exit_code == code, result.output
    out = json.loads(result.stdout)
    assert out["verdict"] == verdict
    report = _report(ws)
    assert report.verdict.value == verdict
    # the invariant: exit 0 only when all four hold
    assert (result.exit_code == 0) == (case == "all hold")
    if verdict == "NOT_COMPARED":
        assert out["report"]["verification_status"] == "incomplete"
        assert report.slots and all(s.status.value != "MATCH" for s in report.slots)
    if verdict == "REFUSED":
        assert report.refusal is not None and report.slots == ()


@needs_kernel
def test_a_reference_changed_after_the_session_is_refused_with_its_name(ws):
    nb = _notebook(ws / "notebooks" / "nb.ipynb", COPY)
    _save(_adata(shift=0.5), ws / "data" / "pbmc.h5ad")
    result = _verify(ws, COPY, notebook=nb)
    assert result.exit_code == 4, result.output
    reason = _report(ws).refusal.reason
    assert reason == "The saved file changed after the session: pbmc.h5ad"
    assert "Refused: " + reason in _verify(ws, COPY, as_json=False, notebook=nb).stdout


@needs_kernel
def test_a_saved_file_the_export_record_has_no_hash_for_is_not_compared(ws):
    for hashes in (None, {}, {"pbmc.h5ad": "file_not_found"}, {"other.h5ad": "0" * 64}):
        for stale in _replay_dirs(ws):
            for item in stale.iterdir():
                item.unlink()
            stale.rmdir()
        nb = _notebook(ws / "notebooks" / "nb.ipynb", COPY, hashes=hashes)
        result = _verify(ws, COPY, notebook=nb)
        assert result.exit_code == 3, (hashes, result.output)
        assert any("has no hash" in n for n in json.loads(result.stdout)["notes"])


@needs_kernel
@pytest.mark.parametrize("where", ["data/new.txt", "data/pbmc.h5ad", "../input.h5ad"])
def test_a_notebook_that_changes_what_it_only_reads_is_refused_and_says_what(ws, where):
    target = (ws / where).resolve()
    if where.endswith("pbmc.h5ad"):
        body = _writing(f"Path({_built(target)}).write_bytes(b'x')\n")
    else:
        body = _writing(f"Path({_built(target)}).write_text('x')\n")
    result = _verify(ws, body)
    assert result.exit_code == 4, result.output
    reason = _report(ws).refusal.reason
    assert reason.startswith("The notebook changed files it should only read: ")
    expected = "input.h5ad" if where.startswith("..") else where
    assert expected in reason
    assert json.loads(result.stdout)["verdict"] == "REFUSED"


@needs_kernel
def test_a_file_removed_from_data_or_a_notebook_that_edits_itself_is_refused(ws):
    nb = ws / "notebooks" / "nb.ipynb"
    body = _writing(f"Path({_built(ws / 'data' / 'pbmc.h5ad')}).unlink()\n" + COPY)
    result = _verify(ws, body, notebook=_notebook(nb, body))
    assert result.exit_code == 4 and "data/pbmc.h5ad" in _report(ws).refusal.reason
    # the notebook's own file
    other = ws / "notebooks" / "me.ipynb"
    body = _writing(f"Path({_built(other)}).write_text('{{}}')\n")
    result = _verify(ws, body, notebook=_notebook(other, body))
    assert result.exit_code == 4
    reason = json.loads(result.stdout)["report"]["refusal"]["reason"]
    assert "notebooks/me.ipynb" in reason


@needs_kernel
def test_a_notebook_that_crashes_after_changing_data_is_refused_not_failed(ws):
    target = ws / "data" / "new.txt"
    result = _verify(
        ws, _writing(f"Path({_built(target)}).write_text('x')\nraise SystemExit(1)\n")
    )
    assert result.exit_code == 4, result.output


@needs_kernel
def test_the_kernel_runs_in_the_replay_folder(ws, tmp_path, monkeypatch):
    elsewhere = tmp_path / "cwd"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    body = _writing(
        'Path("relative.txt").write_text("x")\nPath("cwd.txt").write_text(str(Path.cwd()))\n'
        + COPY
    )
    result = _verify(ws, body)
    assert result.exit_code == 0, result.output
    (folder,) = _replay_dirs(ws)
    assert (folder / "relative.txt").is_file()
    assert Path((folder / "cwd.txt").read_text()).resolve() == folder.resolve()
    assert list(elsewhere.iterdir()) == []  # nothing in the caller's directory
    assert Path.cwd().resolve() == elsewhere.resolve()  # and it is back there


def test_the_workspace_reference_is_checked_against_the_export_record(ws):
    from lobster.cli_internal.commands.light.replay_commands import _workspace_reference

    saved = ws / "data" / "pbmc.h5ad"
    good = {"pbmc.h5ad": _sha(saved)}
    assert _workspace_reference(ws, "pbmc", good) == (saved, "ok")
    assert _workspace_reference(ws, "pbmc", {"pbmc.h5ad": "0" * 64}) == (
        None,
        "changed",
    )
    # the export record is keyed by file name: a stem is not a key
    assert _workspace_reference(ws, "pbmc", {"pbmc": _sha(saved)}) == (
        None,
        "unrecorded",
    )
    assert _workspace_reference(ws, "pbmc", {}) == (None, "unrecorded")
    assert _workspace_reference(ws, "nope", good) == (None, "none")
    (ws / "data" / "pbmc.h5ad").rename(ws / "pbmc.h5ad")  # a file in the root
    assert _workspace_reference(ws, "pbmc", good) == (ws / "pbmc.h5ad", "ok")


def test_the_export_record_is_read_from_the_notebook(ws):
    from lobster.cli_internal.commands.light.replay_commands import _recorded_hashes

    nb = _notebook(
        ws / "notebooks" / "a.ipynb",
        COPY,
        hashes={"p": "a" * 64, "q": "file_not_found"},
    )
    assert _recorded_hashes(nb) == {"p": "a" * 64}
    assert (
        _recorded_hashes(_notebook(ws / "notebooks" / "b.ipynb", COPY, hashes=None))
        == {}
    )
    assert _recorded_hashes(ws / "notebooks" / "missing.ipynb") == {}
    (ws / "notebooks" / "c.ipynb").write_text("not json")
    assert _recorded_hashes(ws / "notebooks" / "c.ipynb") == {}


def test_the_help_lists_the_exit_codes_in_both_languages():
    from lobster.cli import app as lobster_app

    registered = [c for c in lobster_app.registered_commands if c.name == "replay"]
    help_text = registered[0].help
    for line in (
        "0 MATCH",
        "1 MISMATCH",
        "2 usage error",
        "3 NOT_COMPARED",
        "4 REFUSED",
    ):
        assert line in help_text
    assert "5 the notebook failed to run" in help_text
    from lobster.i18n import h

    set_lang("zh")
    try:
        zh = h(help_text)
    finally:
        set_lang("en")
    assert zh != help_text and "5" in zh and "NOT_COMPARED" in zh


@needs_kernel
def test_a_difference_is_still_a_mismatch_when_another_output_was_not_compared(ws):
    body = """\
import anndata as ad
adata = ad.read_h5ad(input_data)
adata.X = adata.X + 0.25
adata.write_h5ad(f"{output_prefix}_extra.h5ad")
adata.X = adata.X + 1
adata.write_h5ad(f"{output_prefix}_pbmc.h5ad")
"""
    result = _verify(ws, body)
    assert result.exit_code == 1, result.output
    report = _report(ws)
    assert report.verdict is Verdict.MISMATCH
    assert any("extra.h5ad" in line for s in report.slots for line in s.not_compared)


def _clean_env(monkeypatch, tmp_path):
    for name in ("LOBSTER_LAMIN", "LOBSTER_LAMIN_INSTANCE", "LAMIN_CURRENT_INSTANCE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(
        "lobster.core.catalog.factory._global_config_path",
        lambda: tmp_path / "no-providers.json",
    )


@pytest.fixture
def real(tmp_path, monkeypatch):
    """A session and an export made by the product: ``(workspace, notebook, input)``.

    DataManagerV2 loads, changes and saves a modality; ``export_notebook`` writes
    the notebook and its export record. No file here is written by hand.
    """
    from lobster.core.provenance.analysis_ir import AnalysisStep
    from lobster.core.runtime.data_manager import DataManagerV2

    _clean_env(monkeypatch, tmp_path)
    workspace = tmp_path / "workspace"
    source = _adata()
    source.X = np.abs(source.X)
    _save(source, tmp_path / "input.h5ad")
    manager = DataManagerV2(workspace_path=workspace, auto_scan=False)
    manager.load_modality(
        "s1", tmp_path / "input.h5ad", adapter="transcriptomics_single_cell"
    )
    adata = manager.get_modality("s1")
    adata.X = np.log1p(adata.X)
    manager.modalities["s1"] = adata
    step = AnalysisStep(
        operation="scanpy.pp.normalize", tool_name="normalize_log1p",
        description="normalise", library="scanpy",
        code_template="adata.X = np.log1p(adata.X)", imports=["import numpy as np"],
        parameters={}, parameter_schema={}, input_entities=["adata"],
        output_entities=["adata"], exportable=True,
    )  # fmt: skip
    manager.log_tool_usage("normalize_log1p", {}, "normalise", ir=step)
    manager.save_modality("s1", workspace / "data" / "s1.h5ad")
    notebook = Path(
        manager.export_notebook("s1_replay", "S1", filter_strategy="replay")
    )
    return workspace, notebook, tmp_path / "input.h5ad"


STEP = "adata.X = np.log1p(adata.X)"


def edit_notebook(notebook: Path, replacement: str) -> None:
    """What a user does to an exported notebook: change the step."""
    text = notebook.read_text("utf-8")
    assert STEP in text
    notebook.write_text(text.replace(STEP, replacement), "utf-8")


# verdict -> (exit code, how the session's files or the notebook were disturbed)
REAL_CASES = {
    "MATCH": (0, "none"),
    "MISMATCH": (1, "edit"),
    "NOT_COMPARED": (3, "delete"),
    "REFUSED": (4, "tamper"),
    "NOTEBOOK_FAILED": (5, "crash"),
}


def disturb(how: str, workspace: Path, notebook: Path) -> None:
    if how == "edit":
        edit_notebook(notebook, STEP + " + 1")
    elif how == "crash":
        edit_notebook(notebook, "raise RuntimeError('boom')")
    elif how == "delete":
        (workspace / "data" / "s1.h5ad").unlink()
    elif how == "tamper":
        tampered = _adata(shift=0.5)
        _save(tampered, workspace / "data" / "s1.h5ad")


def _verify_real(workspace, notebook, source, *extra):
    return runner.invoke(
        app,
        ["--verify", str(notebook), "--input", str(source), "-w", str(workspace),
         "--json", *extra],
    )  # fmt: skip


@needs_kernel
@pytest.mark.parametrize("verdict", REAL_CASES)
def test_each_verdict_on_a_real_session_and_export_catalog_off(real, verdict):
    workspace, notebook, source = real
    code, how = REAL_CASES[verdict]
    disturb(how, workspace, notebook)
    result = _verify_real(workspace, notebook, source)
    assert result.exit_code == code, (result.output, result.exception)
    out = json.loads(result.stdout)
    assert out["verdict"] == verdict
    assert not any("has no hash" in n for n in out["notes"]), out["notes"]


@needs_kernel
def test_the_record_of_a_real_export_is_keyed_by_file_name(real):
    _workspace, notebook, _source = real
    record = replay_commands._recorded_hashes(notebook)
    assert set(record) == {"s1.h5ad"}


@needs_kernel
def test_a_syntax_error_in_the_parameters_cell_is_a_notebook_that_failed(ws):
    nb = _notebook(ws / "notebooks" / "nb.ipynb", COPY, parameters="input_data = (")
    result = _verify(ws, COPY, notebook=nb)
    assert result.exit_code == 5, (result.output, result.exception)
    assert json.loads(result.stdout)["verdict"] == "NOTEBOOK_FAILED"
    assert _report(ws).execution is Execution.ERROR


def _cli(ws, *args):
    """The real command in its own process: what a terminal sees on each stream."""
    # The update notice is printed by the whole CLI, not by this command.
    boot = (
        "import lobster.config.version_check as v;"
        "v.check_for_updates = lambda: (False, None);"
        "from lobster.cli import app; app()"
    )
    env = {**os.environ, "HOME": str(ws.parent), "LOBSTER_LANG": "en"}
    return subprocess.run(  # nosec B603 # fixed argv, no shell
        [sys.executable, "-c", boot, "replay", *args],
        capture_output=True, text=True, cwd=ws.parent, env=env, timeout=600,
    )  # fmt: skip


JSON_PATHS = {
    "match": (0, COPY, None),
    "mismatch": (1, SHIFTED, None),
    "usage": (2, COPY, "no-input"),
    "not compared": (3, COPY, "no-reference"),
    "refused": (4, COPY, "edited-reference"),
    "notebook failed": (5, BOOM, None),
}


@needs_kernel
@pytest.mark.parametrize("name", JSON_PATHS)
def test_json_stdout_is_exactly_one_document_on_every_exit_path(ws, name):
    code, body, how = JSON_PATHS[name]
    nb = _notebook(ws / "notebooks" / "nb.ipynb", body)
    args = ["--verify", str(nb), "-w", str(ws), "--json"]
    if how != "no-input":
        args += ["--input", str(ws.parent / "input.h5ad")]
    if how == "no-reference":
        (ws / "data" / "pbmc.h5ad").unlink()
    if how == "edited-reference":
        _save(_adata(shift=0.5), ws / "data" / "pbmc.h5ad")
    proc = _cli(ws, *args)
    assert proc.returncode == code, proc.stdout + proc.stderr
    document = json.loads(proc.stdout)  # one document, nothing before or after it
    assert document["exit_code"] == code


@needs_kernel
def test_a_refusal_reason_is_in_the_language_of_the_command(ws):
    nb = _notebook(ws / "notebooks" / "nb.ipynb", COPY)
    _save(_adata(shift=0.5), ws / "data" / "pbmc.h5ad")
    set_lang("zh")
    try:
        result = _verify(ws, COPY, notebook=nb)
    finally:
        set_lang("en")
    reason = json.loads(result.stdout)["report"]["refusal"]["reason"]
    assert "pbmc.h5ad" in reason and any("\u4e00" <= ch <= "\u9fff" for ch in reason)


@needs_kernel
def test_relative_paths_are_resolved_before_the_kernel_changes_directory(
    ws, monkeypatch
):
    _notebook(ws / "notebooks" / "nb.ipynb", COPY)
    monkeypatch.chdir(ws)
    result = runner.invoke(
        app,
        ["--verify", "notebooks/nb.ipynb", "--input", "../input.h5ad", "-w", ".",
         "--json"],
    )  # fmt: skip
    assert result.exit_code == 0, (result.output, result.exception)
    assert json.loads(result.stdout)["verdict"] == "MATCH"


@needs_kernel
def test_a_relative_policy_file_is_found_too(ws, monkeypatch):
    _notebook(ws / "notebooks" / "nb.ipynb", COPY)
    (ws.parent / "policy.json").write_text(json.dumps({"excluded": []}))
    monkeypatch.chdir(ws)
    result = runner.invoke(
        app,
        ["--verify", "notebooks/nb.ipynb", "--input", "../input.h5ad", "-w", ".",
         "--policy", "../policy.json", "--json"],
    )  # fmt: skip
    assert "not found" not in result.output, result.output


SLEEPS = "import time\ntime.sleep(60)\n"


@needs_kernel
def test_a_cell_that_runs_past_the_time_limit_is_a_notebook_that_failed(ws):
    result = _verify(ws, SLEEPS, "--timeout", "2")
    assert result.exit_code == 5, (result.output, result.exception)
    out = json.loads(result.stdout)
    assert out["verdict"] == "NOTEBOOK_FAILED"
    assert any("time limit" in n for n in out["notes"]), out["notes"]
    assert _report(ws).execution is Execution.TIMEOUT


def test_a_negative_time_limit_is_a_usage_error(ws):
    result = _verify(ws, COPY, "--timeout", "-1", as_json=False)
    assert result.exit_code == 2


@needs_kernel
def test_no_time_limit_with_zero(ws):
    assert _verify(ws, COPY, "--timeout", "0").exit_code == 0


@needs_kernel
def test_other_data_files_the_notebook_writes_keep_the_verdict_from_being_a_match(ws):
    body = COPY + (
        "adata.write_h5ad('extra.h5ad')\n"
        "adata.write_h5ad(f'{output_prefix}_pbmc.h5mu')\n"
        "open('scratch.txt', 'w').write('x')\n"
    )
    result = _verify(ws, body)
    out = json.loads(result.stdout)
    assert result.exit_code == 3 and out["verdict"] == "NOT_COMPARED", out
    notes = "\n".join(out["notes"])
    assert "extra.h5ad" in notes and "pbmc.h5mu" in notes and "scratch.txt" in notes
    assert out["report"]["slots"][0]["status"] == "NOT_COMPARED"


@needs_kernel
def test_other_files_alone_are_only_listed(ws):
    result = _verify(ws, COPY + "open('scratch.txt', 'w').write('x')\n")
    out = json.loads(result.stdout)
    assert result.exit_code == 0 and out["verdict"] == "MATCH"
    assert any("scratch.txt" in n for n in out["notes"]), out["notes"]


def test_an_interrupt_is_exit_130_and_no_verdict(ws, monkeypatch):
    from lobster.core.notebooks import executor

    def interrupted(self, *args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(executor.NotebookExecutor, "execute", interrupted)
    result = _verify(ws, COPY)
    assert result.exit_code == 130, (result.output, result.exception)
    out = json.loads(result.stdout)
    assert out["verdict"] is None and out["exit_code"] == 130
    assert not (ws / "replays").exists() or not list(
        (ws / "replays").rglob("replay_report.json")
    )


def test_a_catalog_that_fails_to_open_does_not_stop_the_check(ws, monkeypatch):
    def broken(workspace):
        raise RuntimeError("catalog config broke")

    monkeypatch.setattr(replay_commands, "_catalog_record", broken)
    result = _verify(ws, COPY)
    assert result.exit_code == 0, (result.output, result.exception)
    assert "catalog config broke" in result.stderr
    json.loads(result.stdout)


@pytest.mark.parametrize(
    "args",
    [
        ["--input", "x.h5ad"],  # no --verify
        ["--verify", "nb.ipynb", "--bogus"],
        ["--verify", "nb.ipynb", "--timeout", "soon"],
    ],
    ids=["missing --verify", "unknown option", "bad --timeout"],
)
def test_click_usage_errors_under_json_are_one_document(args):
    result = runner.invoke(app, [*args, "--json"])
    assert result.exit_code == 2
    out = json.loads(result.stdout)
    assert out["verdict"] is None and out["exit_code"] == 2 and out["error"]
    assert out["error"] in result.stderr


def test_click_usage_errors_without_json_stay_plain():
    result = runner.invoke(app, ["--input", "x.h5ad"])
    assert result.exit_code == 2 and result.stdout == ""


def test_the_registered_command_has_the_json_usage_behaviour():
    from lobster.cli import app as lobster_app

    result = runner.invoke(lobster_app, ["replay", "--json"])
    assert result.exit_code == 2
    # the whole CLI may print an update notice first (K16); the document is last
    assert json.loads(result.stdout.strip().splitlines()[-1])["exit_code"] == 2


@needs_kernel
@pytest.mark.parametrize("where", ["stray.h5ad", "deeper/stray.h5mu"])
def test_a_data_file_written_anywhere_in_the_workspace_is_refused(ws, where):
    target = ws / where
    body = _writing(
        f"Path({_built(target.parent)}).mkdir(parents=True, exist_ok=True)\n"
        f"Path({_built(target)}).write_bytes(b'x')\n" + COPY
    )
    result = _verify(ws, body)
    out = json.loads(result.stdout)
    assert result.exit_code == 4 and out["verdict"] == "REFUSED", out
    assert where in out["report"]["refusal"]["reason"]


@needs_kernel
def test_a_data_file_changed_in_the_workspace_root_is_refused(ws):
    root_file = ws / "other.h5ad"
    root_file.write_bytes(b"before")
    body = _writing(f"Path({_built(root_file)}).write_bytes(b'after')\n" + COPY)
    result = _verify(ws, body)
    assert result.exit_code == 4
    assert "other.h5ad" in json.loads(result.stdout)["report"]["refusal"]["reason"]


@pytest.mark.skipif(os.geteuid() == 0, reason="root can read any file")
def test_a_file_that_cannot_be_read_is_refused_with_its_name(ws):
    locked = ws / "data" / "locked.h5ad"
    locked.write_bytes(b"x")
    locked.chmod(0)
    try:
        result = _verify(ws, COPY)
    finally:
        locked.chmod(0o644)
    assert result.exit_code == 4, (result.output, result.exception)
    out = json.loads(result.stdout)
    assert out["verdict"] == "REFUSED"
    assert "locked.h5ad" in out["report"]["refusal"]["reason"]


def test_an_unexpected_error_is_refused_with_one_document_and_no_traceback(
    ws, monkeypatch
):
    def odd(*args, **kwargs):
        raise RuntimeError("odd")

    monkeypatch.setattr(replay_commands, "verify_notebook", odd)
    result = _verify(ws, COPY)
    assert result.exit_code == 4 and isinstance(result.exception, SystemExit)
    out = json.loads(result.stdout)
    assert out["verdict"] == "REFUSED" and out["exit_code"] == 4
    assert out["error"] == "The check could not run: RuntimeError: odd"
    assert "Traceback" not in result.stderr


def _real_session(
    tmp_path,
    monkeypatch,
    *,
    tool="normalize_log1p",
    code=STEP,
    edited=(),
    copies=(),
    raw="input.h5ad",
    in_memory=False,
):
    """A session and an export from the product, with its own saved modalities.

    ``s1`` is loaded from ``input.h5ad`` and changed by one step. Each name in
    ``edited`` is saved with other values; ``s2`` is saved as the step made it.
    """
    from lobster.core.provenance.analysis_ir import AnalysisStep
    from lobster.core.runtime.data_manager import DataManagerV2

    _clean_env(monkeypatch, tmp_path)
    workspace = tmp_path / "workspace"
    source = _adata()
    source.X = np.abs(source.X)
    raw_path = tmp_path / raw
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    _save(source, raw_path)
    manager = DataManagerV2(workspace_path=workspace, auto_scan=False)
    manager.load_modality(
        "s1",
        ad.read_h5ad(raw_path) if in_memory else raw_path,
        adapter="transcriptomics_single_cell",
    )
    adata = manager.get_modality("s1")
    adata.X = np.clip(adata.X, 0, 0.5) if tool == "clip" else np.log1p(adata.X)
    step = AnalysisStep(
        operation="x", tool_name=tool, description="d", library="scanpy",
        code_template=code, imports=["import numpy as np"], parameters={},
        parameter_schema={}, input_entities=["adata"], output_entities=["adata"],
        exportable=True,
    )  # fmt: skip
    manager.log_tool_usage(tool, {}, "d", ir=step)
    for name in (*edited, *copies):
        other = adata.copy()
        if name in edited:
            other.X = np.full_like(adata.X, 0.5)
        else:  # the same values, other bytes (the session's own record is not compared)
            other.uns["lineage"] = {"copied_as": name}
        manager.modalities[name] = other
        manager.save_modality(name, workspace / "data" / f"{name}.h5ad")
    last = "s2" if edited or copies else "s1"
    manager.modalities[last] = adata
    manager.save_modality(last, workspace / "data" / f"{last}.h5ad")
    notebook = Path(manager.export_notebook("nb", "S", filter_strategy="replay"))
    return workspace, notebook, raw_path


@needs_kernel
def test_a_data_folder_that_is_a_symlink_is_watched(tmp_path, monkeypatch):
    workspace, notebook, source = _real_session(tmp_path, monkeypatch)
    other = workspace / "data" / "other_saved.h5ad"
    _save(_adata(), other)
    moved = tmp_path / "bigdisk"
    (workspace / "data").rename(moved)
    (workspace / "data").symlink_to(moved)
    nb = nbformat.read(notebook, as_version=4)
    nb.cells.append(
        nbformat.v4.new_code_cell(
            "import anndata, numpy\n"
            "anndata.AnnData(numpy.zeros((2, 2), 'float32'))"
            f".write_h5ad({_built('../../data/other_saved.h5ad')})"
        )
    )
    nbformat.write(nb, notebook)
    result = _verify_real(workspace, notebook, source)
    out = json.loads(result.stdout)
    assert result.exit_code == 4 and out["verdict"] == "REFUSED", out
    assert "other_saved.h5ad" in out["report"]["refusal"]["reason"]


def test_the_help_says_what_the_check_covers_and_the_code_makes_no_bigger_claim():
    from lobster.cli import app as lobster_app

    (command,) = [c for c in lobster_app.registered_commands if c.name == "replay"]
    assert (
        "Compares what the notebook writes against what the session saved, and "
        "detects changes it can see; it does not prove how the outputs were computed."
        in command.help
    )
    tree = ast.parse(Path(replay_commands.__file__).read_text())
    words = re.compile(r"reproduc|verified|proves|guarantee", re.IGNORECASE)
    claims = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and words.search(node.value)
    ]
    assert claims == []


@needs_kernel
def test_a_saved_file_equal_to_the_input_is_not_taken_for_a_loaded_one(ws):
    """The export record cannot tell a loaded file from a step result: not compared."""
    raw = ws.parent / "input.h5ad"
    nb = _notebook(
        ws / "notebooks" / "nb.ipynb",
        COPY,
        hashes={
            "pbmc.h5ad": _sha(ws / "data" / "pbmc.h5ad"),
            "raw.h5ad": _sha(raw),
        },
    )
    result = _verify(ws, COPY, notebook=nb)
    out = json.loads(result.stdout)
    assert result.exit_code == 3 and out["verdict"] == "NOT_COMPARED", out
    assert any("raw.h5ad" in n for n in out["notes"]), out["notes"]


# --- the one place that may say MATCH ---------------------------------------------

from dataclasses import dataclass

ALL_TRUE = {
    "comparison": Verdict.MATCH,
    "record_complete": True,
    "input_disjoint": True,
    "loaded_is_input": True,
    "outputs_compared": True,
    "default_policy": True,
    "no_writes": True,
    "references_intact": True,
    "native_record": True,
}


def _facts(**change):
    return replay_commands.MatchFacts(**{**ALL_TRUE, **change})


def _grant_match(facts):
    return replay_commands._grant_match(facts)


def test_every_fact_established_is_a_match():
    assert _grant_match(_facts()) == ("MATCH", [])


@pytest.mark.parametrize(
    "change, word",
    [
        ({"comparison": Verdict.MISMATCH}, "MISMATCH"),
        ({"comparison": Verdict.NOT_COMPARED}, "NOT_COMPARED"),
        ({"comparison": Verdict.REFUSED}, "NOT_COMPARED"),
        ({"comparison": None}, "NOT_COMPARED"),
        ({"record_complete": False}, "NOT_COMPARED"),
        ({"record_complete": None}, "NOT_COMPARED"),
        ({"input_disjoint": False}, "REFUSED"),
        ({"input_disjoint": None}, "NOT_COMPARED"),
        ({"loaded_is_input": False}, "REFUSED"),
        ({"loaded_is_input": None}, "NOT_COMPARED"),
        ({"outputs_compared": False}, "NOT_COMPARED"),
        ({"outputs_compared": None}, "NOT_COMPARED"),
        ({"no_writes": False}, "REFUSED"),
        ({"no_writes": None}, "NOT_COMPARED"),
        ({"references_intact": False}, "REFUSED"),
        ({"references_intact": None}, "NOT_COMPARED"),
        ({"native_record": False}, "NOT_COMPARED"),
        ({"native_record": None}, "NOT_COMPARED"),
        ({"default_policy": None}, "NOT_COMPARED"),
        ({"default_policy": False}, "MATCH_CUSTOM_POLICY"),
    ],
)
def test_one_missing_or_violated_fact_alone_is_never_a_plain_match(change, word):
    assert _grant_match(_facts(**change))[0] == word


def test_a_custom_policy_with_another_failed_fact_is_not_a_custom_match():
    facts = _facts(default_policy=False, record_complete=False)
    assert _grant_match(facts)[0] == "NOT_COMPARED"


def test_facts_it_does_not_know_never_match():
    @dataclass(frozen=True)
    class Extra(replay_commands.MatchFacts):
        something_new: bool = True

    @dataclass(frozen=True)
    class Other:
        comparison: Verdict = Verdict.MATCH

    assert _grant_match(Extra(**ALL_TRUE))[0] == "NOT_COMPARED"
    assert _grant_match(Other())[0] == "NOT_COMPARED"
    assert _grant_match(None)[0] == "NOT_COMPARED"
    assert (
        _grant_match(replay_commands.MatchFacts())[0] == "NOT_COMPARED"
    )  # nothing established


def test_only_the_grant_function_says_match():
    assert callable(getattr(replay_commands, "_grant_match", None))
    tree = ast.parse(Path(replay_commands.__file__).read_text())
    for function in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
        if function.name == "_grant_match":
            continue
        for node in ast.walk(function):
            if isinstance(node, ast.Return) and node.value is not None:
                assert "Verdict.MATCH" not in ast.unparse(node.value), function.name


# --- what an adversarial pass found ------------------------------------------------


@needs_kernel
def test_a_match_says_what_it_covers(real):
    workspace, notebook, source = real
    text = runner.invoke(
        app, ["--verify", str(notebook), "--input", str(source), "-w", str(workspace)]
    )
    assert "Scope: the files the session saved and the export recorded." in text.stdout


@needs_kernel
def test_a_record_entry_without_a_hash_alone_keeps_everything_else_from_matching(ws):
    nb = _notebook(
        ws / "notebooks" / "nb.ipynb",
        COPY,
        hashes={
            "pbmc.h5ad": _sha(ws / "data" / "pbmc.h5ad"),
            "gone.h5ad": "hash_failed",
        },
    )
    result = _verify(ws, COPY, notebook=nb)
    out = json.loads(result.stdout)
    assert result.exit_code == 3 and out["verdict"] == "NOT_COMPARED", out
    statuses = {slot["status"] for slot in out["report"]["slots"]}
    assert (
        "MISMATCH" not in statuses
    )  # what was compared matched; the rest is not proven
    assert any("gone.h5ad" in n for n in out["notes"]), out["notes"]


@needs_kernel
def test_an_upper_case_data_file_the_notebook_writes_is_left_out_and_named(ws):
    body = COPY + "adata.write_h5ad('Extra.H5AD')\nopen('Other.H5MU', 'w').write('x')\n"
    result = _verify(ws, body)
    out = json.loads(result.stdout)
    assert result.exit_code == 3 and out["verdict"] == "NOT_COMPARED", out
    notes = "\n".join(out["notes"])
    assert "Extra.H5AD" in notes and "Other.H5MU" in notes
    assert "Other files the notebook wrote" not in notes  # data files are not "other"


def test_an_interrupt_leaves_one_document_even_with_an_orphaned_task(ws, monkeypatch):
    import gc
    import sys

    from lobster.core.notebooks import executor

    class Orphan:  # what a task nobody awaited prints when it is collected
        def __init__(self):
            self.me = self  # a cycle: only the collector frees it

        def __del__(self):
            print("Task exception was never retrieved: DeadKernelError")

    def interrupted(self, *args, **kwargs):
        Orphan()
        raise KeyboardInterrupt

    collected = []
    real_collect = gc.collect

    def spy(*args):
        collected.append(sys.stdout is sys.stderr)  # collected while stdout is stderr
        return real_collect(*args)

    monkeypatch.setattr(executor.NotebookExecutor, "execute", interrupted)
    monkeypatch.setattr(replay_commands.gc, "collect", spy)
    result = _verify(ws, COPY)
    assert result.exit_code == 130
    assert collected == [True]
    json.loads(result.stdout)


# --- input roles: which file the session loaded ------------------------------------


def _run_with_loaded(ws, loaded, *extra):
    nb = _notebook(ws / "notebooks" / "nb.ipynb", COPY, loaded=loaded)
    return json.loads(_verify(ws, COPY, *extra, notebook=nb).stdout)


@needs_kernel
def test_the_loaded_file_is_the_input_is_a_match(ws):
    out = _run_with_loaded(ws, LOADED)
    assert out["verdict"] == "MATCH", out


@needs_kernel
def test_a_record_from_before_input_roles_is_not_compared_and_says_re_export(ws):
    out = _run_with_loaded(ws, None)
    assert out["exit_code"] == 3 and out["verdict"] == "NOT_COMPARED", out
    assert any("predates input roles" in n and "re-export" in n for n in out["notes"])


@needs_kernel
def test_an_input_that_is_not_the_loaded_file_is_refused(ws, tmp_path):
    other = tmp_path / "other.h5ad"
    _save(_adata(shift=3.0), other)
    out = _run_with_loaded(ws, [_loaded_entry(other)])
    assert out["exit_code"] == 4 and out["verdict"] == "REFUSED", out
    assert "not a file the session loaded" in out["report"]["refusal"]["reason"]
    assert "other.h5ad" in out["report"]["refusal"]["reason"]


@needs_kernel
def test_the_same_file_loaded_twice_counts_once(ws):
    entry = _loaded_entry(ws.parent / "input.h5ad")
    out = _run_with_loaded(ws, [entry, dict(entry, modality="again")])
    assert out["verdict"] == "MATCH", out


@needs_kernel
def test_several_different_loaded_files_are_not_compared_with_one_input(ws, tmp_path):
    other = tmp_path / "other.h5ad"
    _save(_adata(shift=3.0), other)
    out = _run_with_loaded(
        ws, [_loaded_entry(ws.parent / "input.h5ad"), _loaded_entry(other, "b")]
    )
    assert out["exit_code"] == 3 and out["verdict"] == "NOT_COMPARED", out
    assert any("2 different files" in n for n in out["notes"]), out["notes"]


@needs_kernel
@pytest.mark.parametrize("loaded", [[], [{"modality": "x", "name": "a.h5ad"}]])
def test_a_record_with_no_usable_loaded_entry_is_not_compared(ws, loaded):
    out = _run_with_loaded(ws, loaded)
    assert out["exit_code"] == 3 and out["verdict"] == "NOT_COMPARED", out


def _roles(notebook: Path) -> dict:
    nb = json.loads(notebook.read_text("utf-8"))
    text = next(
        "".join(c["source"])
        for c in nb["cells"]
        if "data_integrity_manifest" in "".join(c["source"])
    )
    block = text.split("```json")[1].split("```")[0]
    return json.loads(block)["data_integrity_manifest"]


@needs_kernel
def test_the_export_records_the_loaded_file_by_name_hash_and_size_only(
    tmp_path, monkeypatch
):
    workspace, notebook, source = _real_session(
        tmp_path, monkeypatch, raw="elsewhere/GSE12345_counts.h5ad"
    )
    roles = _roles(notebook)["input_roles"]
    assert roles["version"] == 2
    assert roles["loaded"] == [
        {
            "modality": "s1",
            "name": "GSE12345_counts.h5ad",
            "sha256": _sha(source),
            "size": source.stat().st_size,
            "adapter": "transcriptomics_single_cell",
        }
    ]
    assert roles["saved"] == _roles(notebook)["input_files"]
    assert str(tmp_path) not in notebook.read_text("utf-8")  # notebooks get shared


@needs_kernel
@pytest.mark.parametrize(
    "raw", ["input.h5ad", "workspace/raw/pbmc3k.h5ad", "geo/GSE12345_counts.h5ad"]
)
def test_a_real_session_matches_whatever_the_loaded_file_is_called_or_kept(
    tmp_path, monkeypatch, raw
):
    workspace, notebook, source = _real_session(tmp_path, monkeypatch, raw=raw)
    result = _verify_real(workspace, notebook, source)
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["verdict"] == "MATCH"


@needs_kernel
def test_a_real_session_with_an_edited_raw_file_is_refused(tmp_path, monkeypatch):
    workspace, notebook, source = _real_session(tmp_path, monkeypatch)
    edited = tmp_path / "edited.h5ad"
    _save(ad.read_h5ad(source), edited)
    shifted = ad.read_h5ad(edited)
    shifted.X = shifted.X + 1
    _save(shifted, edited)
    result = _verify_real(workspace, notebook, edited)
    out = json.loads(result.stdout)
    assert result.exit_code == 4 and out["verdict"] == "REFUSED", out
    assert "not a file the session loaded" in out["report"]["refusal"]["reason"]


@needs_kernel
def test_a_real_session_loaded_from_memory_is_matched_by_content(tmp_path, monkeypatch):
    from lobster.core.provenance import content_digest

    monkeypatch.setattr(content_digest, "IN_MEMORY_INPUTS", True)
    workspace, notebook, source = _real_session(tmp_path, monkeypatch, in_memory=True)
    (entry,) = _roles(notebook)["input_roles"]["loaded"]
    assert "sha256" not in entry and entry["content_sha256"]
    assert _verify_real(workspace, notebook, source).exit_code == 0
    other = tmp_path / "other.h5ad"
    _save(_adata(shift=3.0), other)
    assert _verify_real(workspace, notebook, other).exit_code == 4


@needs_kernel
def test_a_real_export_without_roles_is_not_compared_and_says_re_export(
    tmp_path, monkeypatch
):
    workspace, notebook, source = _real_session(tmp_path, monkeypatch)
    nb = json.loads(notebook.read_text("utf-8"))
    for cell in nb["cells"]:
        text = "".join(cell["source"])
        if "data_integrity_manifest" in text:
            head, rest = text.split("```json", 1)
            block, tail = rest.split("```", 1)
            record = json.loads(block)
            del record["data_integrity_manifest"]["input_roles"]
            cell["source"] = head + "```json" + json.dumps(record) + "```" + tail
    notebook.write_text(json.dumps(nb))
    result = _verify_real(workspace, notebook, source)
    out = json.loads(result.stdout)
    assert result.exit_code == 3 and out["verdict"] == "NOT_COMPARED", out
    assert any("re-export" in n for n in out["notes"])


@needs_kernel
def test_a_custom_policy_match_is_named_in_the_report_file_and_in_json(ws, tmp_path):
    narrow = tmp_path / "narrow.json"
    narrow.write_text(
        json.dumps({"schema_version": 1, "required": ["files/pbmc.h5ad/obs"]})
    )
    result = _verify(ws, SHIFTED, "--policy", str(narrow))
    out = json.loads(result.stdout)
    assert result.exit_code == 6, out
    assert out["report"]["verdict"] == "MATCH_CUSTOM_POLICY"
    assert out["report"]["custom_policy"]["required"] == ["files/pbmc.h5ad/obs"]
    (folder,) = _replay_dirs(ws)
    on_file = json.loads((folder / "replay_report.json").read_text())
    assert on_file["verdict"] == "MATCH_CUSTOM_POLICY"
    assert on_file["custom_policy"] == out["report"]["custom_policy"]


# --- a record an earlier version of Lobster wrote ("migrated") ----------------------


def _migrate(notebook: Path) -> Path:
    """The export record of a session an earlier version wrote: it has a ``migrated`` block."""
    nb = json.loads(notebook.read_text())
    for cell in nb["cells"]:
        source = "".join(cell["source"])
        if "data_integrity_manifest" in source:
            record = json.loads(source.split("```json\n", 1)[1].split("\n```", 1)[0])
            record["data_integrity_manifest"]["migrated"] = {
                "source": "metadata.json",
                "lobster": "1.2.2",
                "software_versions": {"lobster": "1.2.2"},
                "input": "not recorded by the source version",
            }
            cell["source"] = f"## Export record\n\n```json\n{json.dumps(record)}\n```\n"
    notebook.write_text(json.dumps(nb))
    return notebook


MIGRATED_LABEL = "migrated: input not recorded"


@needs_kernel
def test_a_migrated_record_never_matches_even_when_every_fact_holds(ws):
    plain = _notebook(ws / "notebooks" / "plain.ipynb", COPY)
    assert _verify(ws, COPY, notebook=plain).exit_code == 0  # the same facts, native
    nb = _migrate(_notebook(ws / "notebooks" / "nb.ipynb", COPY))
    result = _verify(ws, COPY, notebook=nb)
    assert result.exit_code == 3, result.output
    out = json.loads(result.stdout)
    assert out["verdict"] == "NOT_COMPARED"
    assert out["report"]["verdict"] == "NOT_COMPARED"


@needs_kernel
def test_a_migrated_replay_shows_each_file_under_its_label(ws):
    nb = _migrate(_notebook(ws / "notebooks" / "nb.ipynb", COPY))
    out = json.loads(_verify(ws, COPY, notebook=nb).stdout)
    assert out["report"]["migrated"] == {
        "label": MIGRATED_LABEL,
        "files": [{"file": "pbmc.h5ad", "status": "MATCH", "diffs": []}],
    }
    assert any(MIGRATED_LABEL in n for n in out["notes"])
    assert any(n == "pbmc.h5ad: MATCH" for n in out["notes"])
    assert all(s["status"] != "MATCH" for s in out["report"]["slots"])


@needs_kernel
def test_a_migrated_replay_names_the_keys_that_differ(ws):
    nb = _migrate(_notebook(ws / "notebooks" / "nb.ipynb", SHIFTED))
    result = _verify(ws, SHIFTED, notebook=nb)
    assert result.exit_code == 3, result.output
    out = json.loads(result.stdout)
    (entry,) = out["report"]["migrated"]["files"]
    assert entry["file"] == "pbmc.h5ad" and entry["status"] == "MISMATCH"
    assert any("pbmc.h5ad/X" in d for d in entry["diffs"])
    assert any(n.startswith("pbmc.h5ad: MISMATCH") and "X" in n for n in out["notes"])


@needs_kernel
def test_the_console_shows_the_per_file_result_of_a_migrated_replay(ws):
    nb = _migrate(_notebook(ws / "notebooks" / "nb.ipynb", COPY))
    result = _verify(ws, COPY, notebook=nb, as_json=False)
    assert "Verdict: NOT_COMPARED" in result.stdout
    assert MIGRATED_LABEL in result.stdout and "pbmc.h5ad: MATCH" in result.stdout


@needs_kernel
def test_a_record_without_the_migrated_block_still_withholds_the_per_file_result(ws):
    nb = _notebook(ws / "notebooks" / "nb.ipynb", COPY, loaded=[])
    out = json.loads(_verify(ws, COPY, notebook=nb).stdout)
    assert out["verdict"] == "NOT_COMPARED"
    assert "migrated" not in out["report"]
    assert not any(MIGRATED_LABEL in n for n in out["notes"])
    (slot,) = out["report"]["slots"]
    assert slot["status"] == "NOT_COMPARED"
    assert any("not established: loaded_is_input" in x for x in slot["not_compared"])


def test_only_a_native_record_is_granted_a_match():
    facts = dict(
        comparison=Verdict.MATCH,
        record_complete=True,
        input_disjoint=True,
        loaded_is_input=True,
        outputs_compared=True,
        default_policy=True,
        no_writes=True,
        references_intact=True,
    )
    assert replay_commands._grant_match(
        replay_commands.MatchFacts(**facts, native_record=True)
    ) == ("MATCH", [])
    word, why = replay_commands._grant_match(
        replay_commands.MatchFacts(**facts, native_record=False)
    )
    assert (word, why) == ("NOT_COMPARED", ["native_record"])
    mismatch = {**facts, "comparison": Verdict.MISMATCH}
    word, _ = replay_commands._grant_match(
        replay_commands.MatchFacts(**mismatch, native_record=False)
    )
    assert word == "NOT_COMPARED"  # a migrated record has no verdict either way
