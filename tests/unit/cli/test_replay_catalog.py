"""``lobster replay --verify`` with the LaminDB catalog on.

The command's side of it (what it asks the engine, what a catalog problem does)
runs against a stand-in engine in this process. A few tests run the real engine in
their own processes, as ``tests/unit/core/catalog/test_lamin_engine.py`` does; the
engine's own behaviour is tested there.
"""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import typer
from typer.testing import CliRunner

from lobster.cli_internal.commands.light import lamin_commands, replay_commands
from lobster.cli_internal.commands.light.replay_commands import (
    _catalog_record,
    _notebook_key,
    _ReplayRecord,
    JsonUsageCommand,
    replay_command,
)
from tests.unit.cli.test_replay_commands import (  # noqa: F401  (ws is a fixture)
    BOOM,
    COPY,
    REAL_CASES,
    SHIFTED,
    SESSION_SHIFT,
    _adata,
    disturb,
    _notebook,
    _replay_dirs,
    _save,
    needs_kernel,
    real,
    ws,
)

pytestmark = pytest.mark.unit

app = typer.Typer()
app.command(cls=JsonUsageCommand)(replay_command)
runner = CliRunner()
UID = "ABCDEFGHIJKLMNOP"


class FakeEngine:
    """What ``lobster replay`` asks of the catalog engine, and what it can answer."""

    def __init__(self, *, session=True, files=None, fail=(), refuse=False, states=None):
        self.session = session
        self.files = files or {}
        self.states = states or {}
        self.fail = fail
        self.refuse = refuse
        self.calls = []

    def _maybe_fail(self, name):
        if name in self.fail:
            raise RuntimeError(f"{name} broke")

    def begin_replay(self, **fields):
        self.calls.append(("begin", fields))
        self._maybe_fail("begin")
        if self.refuse:
            return None
        return SimpleNamespace(uid=UID, session_uid="SESSION" if self.session else None)

    def replay_reference_state(self, stem):
        self.calls.append(("reference", stem))
        self._maybe_fail("reference")
        found = self.files.get(stem)
        return (found, "ok") if found else (None, self.states.get(stem, "none"))

    def end_replay(self, status, **fields):
        self.calls.append(("end", {"status": status, **fields}))
        self._maybe_fail("end")


@pytest.fixture
def engine(ws, monkeypatch):
    fake = FakeEngine(files={"pbmc": ws / "data" / "pbmc.h5ad"})
    monkeypatch.setattr(lamin_commands, "_open", lambda *a, **k: fake)
    monkeypatch.setattr(
        replay_commands,
        "_catalog_record",
        lambda workspace: _ReplayRecord(workspace, None),
    )
    return fake


def _args(ws, nb, *extra):
    return [
        "--verify",
        str(nb),
        "--input",
        str(ws.parent / "input.h5ad"),
        "-w",
        str(ws),
        *extra,
    ]


def _replay(ws, body=COPY):
    nb = _notebook(ws / "notebooks" / "nb.ipynb", body)
    return runner.invoke(app, _args(ws, nb, "--json"))


def _calls(engine, name):
    return [fields for call, fields in engine.calls if call == name]


@needs_kernel
def test_a_recorded_replay_runs_in_the_folder_named_by_the_run_and_ends_with_the_outcome(
    ws, engine
):
    engine.files = {"pbmc": ws / "data" / "pbmc.h5ad"}
    result = _replay(ws)
    assert result.exit_code == 0, result.output
    out = json.loads(result.stdout)
    (folder,) = _replay_dirs(ws)
    assert folder.name == UID and out["replay_dir"] == str(folder)
    (begun,) = _calls(engine, "begin")
    assert begun["key"] == "notebooks/nb.ipynb"
    assert begun["source"] == (ws / "notebooks" / "nb.ipynb").read_text("utf-8")
    assert begun["params"] == {"input_data": "input.h5ad", "output_prefix": "out"}
    assert begun["lobster_version"]
    (ended,) = _calls(engine, "end")
    assert ended["status"] == "completed" and ended["verdict"] == "MATCH"
    assert ended["report_path"] == folder / "replay_report.json"
    assert ended["report_path"].read_text().endswith("}\n")  # whole lines, for the log
    assert ended["outputs"] == [folder / "out_pbmc.h5ad"]
    assert engine.calls[0][0] == "begin" and engine.calls[-1][0] == "end"


@needs_kernel
def test_the_reference_is_what_the_catalog_recorded_not_the_workspace_file(
    ws, engine, tmp_path
):
    _save(_adata(shift=1.0), ws / "data" / "pbmc.h5ad")  # changed since the session
    kept = tmp_path / "kept.h5ad"
    _save(_adata(shift=SESSION_SHIFT, record=True), kept)  # what the session saved
    engine.files = {"pbmc": kept}
    result = _replay(ws)
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["verdict"] == "MATCH"
    assert ("reference", "pbmc") in engine.calls
    # the catalog answers no file: there is nothing to compare, whatever the workspace holds
    engine.files = {}
    (ws / "replays").rename(ws.parent / "earlier")
    result = _replay(ws)
    assert result.exit_code == 3, result.output
    assert json.loads(result.stdout)["verdict"] == "NOT_COMPARED"


@needs_kernel
def test_without_a_session_in_the_catalog_the_workspace_file_is_the_reference(
    ws, engine
):
    engine.session = False
    result = _replay(ws)
    assert result.exit_code == 0, result.output
    assert not [c for c in engine.calls if c[0] == "reference"]
    (folder,) = _replay_dirs(ws)
    assert folder.name == UID and _calls(engine, "end")  # still recorded


@needs_kernel
@pytest.mark.parametrize(
    "body,status,verdict,code",
    [
        (SHIFTED, "completed", "MISMATCH", 1),
        (BOOM, "errored", "NOTEBOOK_FAILED", 5),
    ],
)
def test_the_run_ends_with_the_verdict_of_the_replay(
    ws, engine, body, status, verdict, code
):
    result = _replay(ws, body)
    assert result.exit_code == code, result.output
    (ended,) = _calls(engine, "end")
    assert (ended["status"], ended["verdict"]) == (status, verdict)


@needs_kernel
def test_a_notebook_the_executor_refuses_ends_its_run_as_refused(ws, engine):
    nb = _notebook(ws / "notebooks" / "nb.ipynb", COPY, parameters=False)
    result = runner.invoke(app, _args(ws, nb, "--json"))
    assert result.exit_code == 4, result.output
    (ended,) = _calls(engine, "end")
    assert (ended["status"], ended["verdict"]) == ("errored", "REFUSED")


@needs_kernel
def test_a_session_file_that_changed_since_the_session_is_refused_not_compared(
    ws, engine
):
    engine.files = {}
    engine.states = {"pbmc": "changed"}
    result = _replay(ws)
    assert result.exit_code == 4, result.output
    out = json.loads(result.stdout)
    assert out["verdict"] == "REFUSED"
    assert out["report"]["refusal"]["reason"] == (
        "The saved file changed after the session: pbmc.h5ad"
    )
    (ended,) = _calls(engine, "end")
    assert (ended["status"], ended["verdict"]) == ("errored", "REFUSED")


@needs_kernel
def test_a_notebook_that_changes_data_ends_its_run_as_refused(ws, engine):
    target = ws / "data" / "new.txt"
    body = f"from pathlib import Path\nPath({str(target)!r}).write_text('x')\n" + COPY
    result = _replay(ws, body)
    assert result.exit_code == 4, result.output
    (ended,) = _calls(engine, "end")
    assert ended["verdict"] == "REFUSED"


def test_nothing_is_recorded_for_a_replay_that_is_refused_before_it_starts(ws, engine):
    nb = _notebook(ws / "notebooks" / "nb.ipynb", COPY)
    result = runner.invoke(app, ["--verify", str(nb), "-w", str(ws)])  # no --input
    assert result.exit_code == 2  # a usage error, not a verdict
    assert engine.calls == [] and _replay_dirs(ws) == []


@needs_kernel
@pytest.mark.parametrize("failing", ["begin", "reference", "end"])
def test_a_catalog_problem_is_a_warning_and_the_verdict_stays(ws, engine, failing):
    engine.fail = (failing,)
    result = _replay(ws)
    assert result.exit_code == 0, result.output  # MATCH, as without a catalog
    assert json.loads(result.stdout)["verdict"] == "MATCH"  # stdout is only the JSON
    assert "The catalog could not record the replay" in result.stderr
    assert f"{failing} broke" in result.stderr
    (folder,) = _replay_dirs(ws)
    assert (folder.name == UID) == (failing != "begin")


@needs_kernel
def test_a_notebook_the_catalog_refuses_is_checked_without_a_run(ws, engine):
    engine.refuse = True
    result = _replay(ws)
    assert result.exit_code == 0, result.output
    assert "The notebook was not recorded in the catalog" in result.stderr
    (folder,) = _replay_dirs(ws)
    assert len(folder.name) == 16 and folder.name != UID
    assert [c[0] for c in engine.calls] == ["begin"]


@needs_kernel
def test_a_replay_that_stops_on_an_error_closes_its_run(ws, engine, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("comparison broke")

    # after the notebook ran: not an error of the notebook
    monkeypatch.setattr(replay_commands, "_changes", broken)
    result = _replay(ws)
    assert result.exit_code == 4, (result.output, result.exception)
    (ended,) = _calls(engine, "end")
    assert ended["status"] == "errored" and ended["verdict"] == "REFUSED"
    assert ended["report_path"] is None and ended["outputs"] == []


@needs_kernel
def test_a_close_that_is_interrupted_is_not_repeated(ws, engine):
    def interrupted(status, **fields):
        engine.calls.append(("end", {"status": status, **fields}))
        raise KeyboardInterrupt

    engine.end_replay = interrupted
    result = _replay(ws)
    assert isinstance(result.exception, (KeyboardInterrupt, SystemExit))
    assert len(_calls(engine, "end")) == 1


def test_the_catalog_is_used_only_when_it_is_on_and_installed_and_has_something(
    tmp_path, monkeypatch
):
    workspace = tmp_path / "w"
    workspace.mkdir()
    for name in ("LOBSTER_LAMIN", "LOBSTER_LAMIN_INSTANCE", "LAMIN_CURRENT_INSTANCE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(
        "lobster.core.catalog.factory._global_config_path",
        lambda: tmp_path / "no-providers.json",
    )
    installed, slug = [True], [None]
    monkeypatch.setattr(lamin_commands, "_engine_installed", lambda: installed[0])
    monkeypatch.setattr(lamin_commands, "_slug", lambda path: slug[0])
    assert _catalog_record(workspace) is None  # off
    monkeypatch.setenv("LOBSTER_LAMIN", "1")
    assert _catalog_record(workspace) is None  # on, nothing recorded here yet
    slug[0] = "anonymous/w"
    record = _catalog_record(workspace)
    assert record.workspace == workspace and record.instance is None
    installed[0] = False
    assert _catalog_record(workspace) is None  # on, LaminDB not installed
    installed[0], slug[0] = True, None
    monkeypatch.setenv("LOBSTER_LAMIN_INSTANCE", "anonymous/shared")
    assert _catalog_record(workspace).instance == "anonymous/shared"  # an instance


def test_the_notebook_key_is_its_path_in_the_workspace(tmp_path):
    workspace = tmp_path / "ws"
    assert _notebook_key(workspace / "notebooks" / "a.ipynb", workspace) == (
        "notebooks/a.ipynb"
    )
    assert _notebook_key(workspace / "x" / "a.ipynb", workspace) == "x/a.ipynb"
    assert _notebook_key(tmp_path / "elsewhere" / "b.ipynb", workspace) == (
        "notebooks/b.ipynb"
    )


# --- the real engine, in its own processes ---------------------------------------

pytest.importorskip("lamindb_setup")
from tests.unit.core.catalog.test_lamin_engine import (  # noqa: E402
    _isolated_env,
    _run,
)

SETUP = """
    import shutil
    from pathlib import Path
    from lobster.core.catalog.lamin import LaminCatalog
    from lobster.core.catalog.port import SessionInfo
    ws = Path(os.environ["WS"]).resolve()
    eng = LaminCatalog(ws)
    kw = lambda key: dict(key=key, kind="dataset", otype=None, description="d", meta={})
    saved = ws / "data" / "pbmc.h5ad"
    eng.begin_run(SessionInfo("s1", ws, "0.0.0"))
    saved.write_bytes(Path(os.environ["A"]).read_bytes())
    eng.register_artifact(saved, **kw("data/pbmc.h5ad"))
    nb = Path(os.environ["NB"])
    if not os.environ.get("NOT_EXPORTED"):
        eng.register_transform(key=nb.relative_to(ws).as_posix(), kind="notebook",
                               source=nb.read_text(encoding="utf-8"), description="export")
    eng.end_run("completed")
    if os.environ.get("LATER"):  # a later session saved other bytes, keeping the first
        eng.begin_run(SessionInfo("s2", ws, "0.0.0"))
        target = eng.keep_path(saved)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(saved, target)
        saved.write_bytes(Path(os.environ["B"]).read_bytes())
        eng.register_artifact(saved, **kw("data/pbmc.h5ad"))
        eng.end_run("completed")
    emit()
"""
INSPECT = """
    from pathlib import Path
    from lobster.core.catalog.lamin import LaminCatalog
    ws = Path(os.environ["WS"]).resolve()
    eng = LaminCatalog(ws, create=False)
    ln = eng.ln
    rows = []
    for run in ln.Run.filter(name__startswith="replay:"):
        files = ln.Artifact.filter(run=run).exclude(kind="__lamindb_run__")
        initiated = run.initiated_by_run
        rows.append(dict(
            uid=run.uid, name=run.name, params=run.params, extra=run.extra_data,
            initiated=None if initiated is None else initiated.name,
            transform=run.transform.key, status=run._status_code,
            finished=run.finished_at is not None,
            report=None if run.report is None else run.report.path.read_text(),
            outputs=sorted(f.key for f in files),
        ))
    emit(rows=rows, sessions=eng.overview()["runs"],
         versions=ln.Artifact.filter(key="data/pbmc.h5ad").count(),
         notebooks=ln.Transform.filter(kind="notebook").count(),
         listed=[r["name"] for r in eng.list_runs()])
"""


def _recorded_workspace(tmp_path, *, later=False, body=COPY, exported=True):
    workspace = tmp_path / "ws"
    (workspace / "data").mkdir(parents=True)
    a, b = tmp_path / "a.h5ad", tmp_path / "b.h5ad"
    _save(_adata(shift=SESSION_SHIFT, record=True), a)
    _save(
        _adata(shift=SESSION_SHIFT, record=True), workspace / "data" / "pbmc.h5ad"
    )  # so the export record has its hash
    _save(_adata(shift=1.0), b)
    _save(_adata(), tmp_path / "input.h5ad")
    nb = _notebook(workspace / "notebooks" / "nb.ipynb", body)
    env = {"WS": str(workspace), "A": str(a), "B": str(b), "NB": str(nb)}
    if later:
        env["LATER"] = "1"
    if not exported:
        env["NOT_EXPORTED"] = "1"
    _run(SETUP, tmp_path, env_extra=env)
    return workspace, nb, env


def _cli(tmp_path, workspace, nb, drop_enable=False):
    env = _isolated_env(tmp_path, {}, drop_enable)
    cwd = tmp_path / "cwd"
    cwd.mkdir(exist_ok=True)
    args = ["--verify", str(nb), "--input", str(tmp_path / "input.h5ad")]
    args += ["-w", str(workspace), "--json"]
    return subprocess.run(
        [sys.executable, "-c", "from lobster.cli import app; app()", "replay", *args],
        capture_output=True,
        text=True,
        env=env,
        cwd=cwd,
        timeout=900,
    )


@needs_kernel
def test_a_replay_with_the_catalog_on_is_a_run_of_the_notebook_checked_against_the_session(
    tmp_path,
):
    workspace, nb, env = _recorded_workspace(tmp_path, later=True)
    now = (workspace / "data" / "pbmc.h5ad").read_bytes()
    proc = _cli(tmp_path, workspace, nb)
    # the workspace file was saved again by a later session; the catalog still holds
    # the notebook's session's bytes, which the replay reproduces
    assert proc.returncode == 0, proc.stdout + proc.stderr
    out = json.loads(proc.stdout)  # stdout is only the result
    assert out["verdict"] == "MATCH"
    assert (workspace / "data" / "pbmc.h5ad").read_bytes() == now  # data untouched
    r = _run(INSPECT, tmp_path, env_extra=env)
    (row,) = r["rows"]
    assert Path(out["replay_dir"]).name == row["uid"]
    assert row["name"] == "replay:nb" and row["transform"] == "notebooks/nb.ipynb"
    assert row["initiated"] == "s1"
    assert row["params"] == {"input_data": "input.h5ad", "output_prefix": "out"}
    assert row["extra"]["verdict"] == "MATCH" and row["extra"]["kind"] == "replay"
    assert row["finished"] and row["status"] == 0
    assert json.loads(row["report"])["verdict"] == "MATCH"
    assert all(k.startswith("replays/") for k in row["outputs"])
    # no new version of the data, no second Transform, and a replay is not a session
    assert r["versions"] == 2 and r["notebooks"] == 1
    assert r["sessions"] == 2 and "replay:nb" not in r["listed"]


@needs_kernel
def test_a_mismatch_is_recorded_with_its_output_under_the_replay_folder(tmp_path):
    workspace, nb, env = _recorded_workspace(tmp_path, body=SHIFTED)
    proc = _cli(tmp_path, workspace, nb)
    assert proc.returncode == 1, proc.stdout + proc.stderr
    out = json.loads(proc.stdout)
    r = _run(INSPECT, tmp_path, env_extra=env)
    (row,) = r["rows"]
    assert row["extra"]["verdict"] == "MISMATCH" and row["status"] == 0
    assert row["outputs"] == [f"replays/{row['uid']}/out_pbmc.h5ad"]
    assert (Path(out["replay_dir"]) / "out_pbmc.h5ad").is_file()
    assert r["versions"] == 1  # never a new version of the saved file


@needs_kernel
def test_a_notebook_the_session_did_not_export_is_replayed_against_the_workspace(
    tmp_path,
):
    workspace, nb, env = _recorded_workspace(tmp_path, exported=False)
    proc = _cli(tmp_path, workspace, nb)
    assert proc.returncode == 0, proc.stdout + proc.stderr  # the workspace file matches
    r = _run(INSPECT, tmp_path, env_extra=env)
    (row,) = r["rows"]
    assert row["initiated"] is None and row["extra"]["verdict"] == "MATCH"
    assert r["notebooks"] == 1  # recorded now, so the replay has a Transform


@needs_kernel
def test_with_the_catalog_off_a_replay_records_nothing(tmp_path):
    workspace, nb, env = _recorded_workspace(tmp_path)
    proc = _cli(tmp_path, workspace, nb, drop_enable=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    (folder,) = _replay_dirs(workspace)
    assert len(folder.name) == 16
    r = _run(INSPECT, tmp_path, env_extra=env)
    assert r["rows"] == []


@needs_kernel
def test_a_session_file_edited_after_the_session_is_refused_with_the_catalog_on(
    tmp_path,
):
    workspace, nb, env = _recorded_workspace(tmp_path)
    saved = workspace / "data" / "pbmc.h5ad"
    saved.write_bytes(saved.read_bytes() + b"x")  # the same data, other bytes
    proc = _cli(tmp_path, workspace, nb)
    assert proc.returncode == 4, proc.stdout + proc.stderr
    out = json.loads(proc.stdout)
    assert out["verdict"] == "REFUSED"
    assert "changed after the session: pbmc.h5ad" in out["report"]["refusal"]["reason"]
    (row,) = _run(INSPECT, tmp_path, env_extra=env)["rows"]
    assert row["extra"]["verdict"] == "REFUSED" and row["status"] == 1


@needs_kernel
@pytest.mark.parametrize("verdict", REAL_CASES)
def test_each_verdict_on_a_real_session_and_export_catalog_on(
    real, verdict, monkeypatch
):
    workspace, notebook, source = real
    code, how = REAL_CASES[verdict]
    disturb(how, workspace, notebook)
    saved = workspace / "data" / "s1.h5ad"
    # the catalog's side: the session's file, or what it says about it
    if how == "tamper":
        fake = FakeEngine(states={"s1": "changed"})
    else:
        fake = FakeEngine(files={"s1": saved} if saved.exists() else {})
    monkeypatch.setattr(lamin_commands, "_open", lambda *a, **k: fake)
    monkeypatch.setattr(
        replay_commands,
        "_catalog_record",
        lambda workspace: _ReplayRecord(workspace, None),
    )
    result = runner.invoke(
        app,
        ["--verify", str(notebook), "--input", str(source), "-w", str(workspace),
         "--json"],
    )  # fmt: skip
    assert result.exit_code == code, (result.output, result.exception)
    assert json.loads(result.stdout)["verdict"] == verdict
    (ended,) = _calls(fake, "end")
    assert ended["verdict"] == verdict


@needs_kernel
def test_json_stdout_is_one_document_when_the_catalog_fails(ws, engine):
    engine.fail = ("begin", "end")
    result = _replay(ws)
    assert result.exit_code == 0, result.output
    json.loads(result.stdout)


def test_an_interrupt_closes_the_run_as_aborted(ws, engine, monkeypatch):
    from lobster.core.notebooks import executor

    def interrupted(self, *args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(executor.NotebookExecutor, "execute", interrupted)
    result = _replay(ws)
    assert result.exit_code == 130, (result.output, result.exception)
    (ended,) = _calls(engine, "end")
    assert ended["status"] == "aborted" and ended["report_path"] is None
