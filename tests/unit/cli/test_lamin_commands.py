"""``lobster lamin``: workspace settings and messages (no LaminDB needed).

Commands that open a catalog run in their own process; see
``tests/unit/core/catalog/test_lamin_engine.py``.
"""

import json
import logging
import os
import stat

import pytest
from dotenv import dotenv_values
from typer.testing import CliRunner

from lobster.cli_internal.commands.light import lamin_commands
from lobster.cli_internal.commands.light.lamin_commands import instance_slug, lamin_app
from lobster.i18n import set_lang

pytestmark = pytest.mark.unit

runner = CliRunner()


@pytest.fixture
def ws(tmp_path, monkeypatch):
    for name in ("LOBSTER_LAMIN", "LOBSTER_LAMIN_INSTANCE", "LAMIN_CURRENT_INSTANCE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(
        "lobster.core.catalog.factory._global_config_path",
        lambda: tmp_path / "no-providers.json",
    )
    path = tmp_path / ".lobster_workspace"
    path.mkdir()
    return path


def _invoke(*args):
    return runner.invoke(lamin_app, list(args))


def test_status_of_a_workspace_without_catalog(ws):
    result = _invoke("status", "-w", str(ws), "--json")
    assert result.exit_code == 0, result.output
    info = json.loads(result.output)
    assert info["enabled"] is False and info["enabled_source"] == "default"
    assert info["mode"] == "workspace" and info["slug"] is None
    assert info["contents"] is None
    assert not (ws / ".lamindb").exists()


def test_status_text_names_the_catalog_folder(ws):
    result = _invoke("status", "-w", str(ws))
    assert result.exit_code == 0
    assert "Catalog: off (setting from default)" in result.output
    assert str(ws / ".lamindb") in result.output
    assert "This workspace has no catalog yet." in result.output


def test_status_in_chinese(ws):
    set_lang("zh")
    try:
        result = _invoke("status", "-w", str(ws))
    finally:
        set_lang("en")
    assert "目录：关闭（设置来自 默认）" in result.output
    assert "这个工作区还没有目录。" in result.output


@pytest.mark.parametrize(
    "target,slug",
    [
        ("lab/shared", "lab/shared"),
        ("https://lamin.ai/lab/shared", "lab/shared"),
        ("https://www.lamin.ai/lab/shared/artifacts?tab=x#y", "lab/shared"),
        ("http://lamin.ai/lab/shared", "lab/shared"),
        (" lab/shared/ ", "lab/shared"),
        ("lab-1/shared_2.v3", "lab-1/shared_2.v3"),
    ],
)
def test_use_writes_the_instance_to_the_workspace(ws, target, slug):
    result = _invoke("use", target, "-w", str(ws))
    assert result.exit_code == 0, result.output
    assert dotenv_values(ws / ".env")["LOBSTER_LAMIN_INSTANCE"] == slug
    assert "copied into that instance's storage" in result.output
    assert (
        "Lamin's servers (lamin.ai) only if the instance is registered" in result.output
    )
    assert "to look up an instance that is not set up on this machine" in result.output


def test_use_a_local_instance_folder(ws, tmp_path):
    store = tmp_path / "store"
    (store / ".lamindb").mkdir(parents=True)
    (store / ".lamindb" / "storage_uid.txt").write_text(
        "abc\ncreation info:\ninstance_slug=anonymous/shared\ninstance_id=00\n"
    )
    result = _invoke("use", str(store), "-w", str(ws))
    assert result.exit_code == 0, result.output
    assert dotenv_values(ws / ".env")["LOBSTER_LAMIN_INSTANCE"] == "anonymous/shared"


def test_use_refuses_folders_that_are_not_instances(ws, tmp_path):
    plain = tmp_path / "plain"
    plain.mkdir()
    other_ws = tmp_path / "other"
    (other_ws / ".lamindb").mkdir(parents=True)
    (other_ws / ".lamindb" / "lobster.json").write_text('{"slug": "anonymous/x"}')
    with pytest.raises(ValueError, match="No LaminDB instance"):
        instance_slug(str(plain), ws)
    with pytest.raises(ValueError, match="another workspace's catalog"):
        instance_slug(str(other_ws), ws)
    with pytest.raises(ValueError, match="use --clear"):
        instance_slug(str(ws), ws)
    for bad in (
        "shared",
        "a/b/c",
        "/",
        "a/b\nINJECTED=1",
        "a/b #comment",
        "a/${HOME}",
        "a/'b'",
        "https://evil.example/lab/shared",
        "ftp://lamin.ai/lab/shared",
        "https://lamin.ai/lab",
    ):
        with pytest.raises(ValueError):
            instance_slug(bad, ws)
    result = _invoke("use", "a/b/c", "-w", str(ws))
    assert result.exit_code == 2 and not (ws / ".env").exists()


def test_use_refuses_the_slug_of_this_workspace_catalog(ws):
    (ws / ".lamindb").mkdir()
    (ws / ".lamindb" / "lobster.json").write_text('{"slug": "anonymous/lobster-ws-1"}')
    with pytest.raises(ValueError, match="use --clear"):
        instance_slug("anonymous/lobster-ws-1", ws)


def test_a_new_env_file_is_private(ws):
    _invoke("use", "lab/shared", "-w", str(ws))
    assert stat.S_IMODE(os.stat(ws / ".env").st_mode) == 0o600


def test_an_existing_env_file_keeps_its_mode_and_other_keys(ws):
    env = ws / ".env"
    env.write_text("ANTHROPIC_API_KEY=k\n")
    env.chmod(0o640)
    _invoke("use", "lab/shared", "-w", str(ws))
    _invoke("use", "lab/other", "-w", str(ws))
    assert stat.S_IMODE(os.stat(env).st_mode) == 0o640
    assert dotenv_values(env) == {
        "ANTHROPIC_API_KEY": "k",
        "LOBSTER_LAMIN_INSTANCE": "lab/other",
    }


def test_use_clear_goes_back_to_the_workspace_catalog(ws):
    _invoke("use", "lab/shared", "-w", str(ws))
    (ws / ".env").write_text((ws / ".env").read_text() + "OTHER=kept\n")
    result = _invoke("use", "--clear", "-w", str(ws))
    assert result.exit_code == 0
    values = dotenv_values(ws / ".env")
    assert "LOBSTER_LAMIN_INSTANCE" not in values and values["OTHER"] == "kept"
    assert _invoke("use", "-w", str(ws)).exit_code == 2


def test_use_clear_without_a_setting_is_quiet(ws, caplog):
    (ws / ".env").write_text("OTHER=kept\n")
    with caplog.at_level(logging.WARNING):
        result = _invoke("use", "--clear", "-w", str(ws))
    assert result.exit_code == 0 and caplog.records == []
    assert (ws / ".env").read_text() == "OTHER=kept\n"


def test_init_without_the_engine_switches_on_and_says_what_is_missing(ws, monkeypatch):
    monkeypatch.setattr(lamin_commands, "_engine_installed", lambda: False)
    result = _invoke("init", "-w", str(ws))
    assert result.exit_code == 0, result.output
    assert dotenv_values(ws / ".env")["LOBSTER_LAMIN"] == "1"
    assert "Install the 'lamin' extra" in result.output
    assert not (ws / ".lamindb").exists()
    info = json.loads(_invoke("status", "-w", str(ws), "--json").output)
    assert info["enabled"] is True and info["enabled_source"] == "workspace"


def test_init_says_when_the_environment_switches_it_off(ws, monkeypatch):
    monkeypatch.setattr(lamin_commands, "_engine_installed", lambda: True)
    opened = []
    monkeypatch.setattr(lamin_commands, "_open", lambda *a, **k: opened.append(a))
    monkeypatch.setenv("LOBSTER_LAMIN", "0")
    result = _invoke("init", "-w", str(ws))
    assert result.exit_code == 0
    assert "LOBSTER_LAMIN=0 keeps the catalog off until you unset it" in result.output
    assert "The catalog is on" not in result.output
    assert dotenv_values(ws / ".env")["LOBSTER_LAMIN"] == "1" and opened == []


def test_init_with_a_named_instance_creates_no_workspace_catalog(ws, monkeypatch):
    monkeypatch.setattr(lamin_commands, "_engine_installed", lambda: True)
    _invoke("use", "lab/shared", "-w", str(ws))
    result = _invoke("init", "-w", str(ws))
    assert result.exit_code == 0
    assert "Records go to the LaminDB instance lab/shared." in result.output
    assert not (ws / ".lamindb").exists()


def test_ls_without_a_catalog(ws):
    result = _invoke("ls", "-w", str(ws))
    assert result.exit_code == 1
    assert _invoke("ls", "--runs", "--artifacts", "-w", str(ws)).exit_code == 2
    assert _invoke("ls", "--limit", "0", "-w", str(ws)).exit_code == 2


def _engine_with_runs(ws, monkeypatch, rows):
    (ws / ".lamindb").mkdir()
    (ws / ".lamindb" / "lobster.json").write_text('{"slug": "anonymous/lobster-ws-1"}')
    monkeypatch.setattr(lamin_commands, "_engine_installed", lambda: True)

    class Engine:
        def list_runs(self, limit):
            return rows[:limit]

    monkeypatch.setattr(lamin_commands, "_open", lambda workspace: Engine())


RUN_ROWS = [
    {
        "name": "both",
        "status": "completed",
        "started_at": "2026-11-11T08:06:39",
        "outputs": 2,
        "inputs": 0,
        "report": True,
        "environment": True,
    },
    {
        "name": "env_only",
        "status": "completed",
        "started_at": "2026-11-11T08:05:00",
        "outputs": 0,
        "inputs": 1,
        "report": False,
        "environment": True,
    },
    {
        "name": "neither",
        "status": "errored",
        "started_at": "2026-11-11T08:04:00",
        "outputs": 0,
        "inputs": 0,
        "report": False,
        "environment": False,
    },
]


def test_ls_runs_names_the_records_a_run_has(ws, monkeypatch):
    _engine_with_runs(ws, monkeypatch, RUN_ROWS)
    lines = _invoke("ls", "--runs", "-w", str(ws)).output.splitlines()
    assert lines[0].endswith("both  (2 out, 0 in)  [report, environment]")
    assert lines[1].endswith("env_only  (0 out, 1 in)  [environment]")
    assert lines[2].endswith("neither  (0 out, 0 in)")  # no brackets: nothing recorded
    as_json = json.loads(_invoke("ls", "--runs", "--json", "-w", str(ws)).output)
    assert [(r["name"], r["report"], r["environment"]) for r in as_json] == [
        ("both", True, True),
        ("env_only", False, True),
        ("neither", False, False),
    ]


def test_ls_runs_marks_a_replay_with_its_verdict(ws, monkeypatch):
    replay = {
        **RUN_ROWS[0],
        "name": "replay:nb",
        "kind": "replay",
        "verdict": "MATCH",
        "report": False,
        "environment": False,
    }
    _engine_with_runs(ws, monkeypatch, [replay, RUN_ROWS[2]])
    lines = _invoke("ls", "--runs", "-w", str(ws)).output.splitlines()
    assert lines[0].endswith("replay:nb  (2 out, 0 in)  [replay MATCH]")
    assert lines[1].endswith("neither  (0 out, 0 in)")


def test_ls_runs_in_chinese(ws, monkeypatch):
    _engine_with_runs(ws, monkeypatch, RUN_ROWS)
    set_lang("zh")
    try:
        output = _invoke("ls", "--runs", "-w", str(ws)).output
    finally:
        set_lang("en")
    assert "[报告, 环境]" in output.splitlines()[0]


def test_status_json_gives_the_size_of_the_kept_versions(ws, monkeypatch):
    assert json.loads(_invoke("status", "-w", str(ws), "--json").output)[
        "versions"
    ] == {"files": 0, "bytes": 0}
    _prune_engine(ws, monkeypatch)
    info = json.loads(_invoke("status", "-w", str(ws), "--json").output)
    assert info["versions"] == {"files": 2, "bytes": 2058}  # what prune would remove
    assert info["enabled"] is False and info["mode"] == "workspace"  # keys kept


class PruneEngine:
    ROWS = [
        {"key": "data/x.csv", "version": "0000", "uid": "u0", "bytes": 10,
         "file": ".lamindb/versions/data/x.csv/v0000.csv"},
        {"key": "data/x.csv", "version": "0001", "uid": "u1", "bytes": 2048,
         "file": ".lamindb/versions/data/x.csv/v0001.csv"},
    ]  # fmt: skip

    def __init__(
        self,
        rows=None,
        failed=None,
        skipped=None,
        raises=None,
        chatty=False,
        raises_on_kept=None,
    ):
        self.rows = self.ROWS if rows is None else rows
        self.failed = failed or []
        self.skipped = skipped or []
        self.raises = raises
        self.chatty = chatty
        self.raises_on_kept = raises_on_kept
        self.pruned = 0
        self.shown = None

    def kept_versions(self):
        if self.raises_on_kept:
            raise self.raises_on_kept
        if self.chatty:
            print("a message LaminDB prints")
        return list(self.rows)

    def prune_versions(self, shown):
        self.pruned += 1
        self.shown = shown
        if self.chatty:
            print("a message LaminDB prints")
        if self.raises:
            raise self.raises
        return {"removed": self.rows[: len(self.rows) - len(self.failed)],
                "failed": self.failed, "skipped": self.skipped}  # fmt: skip

    def overview(self):
        return {"artifacts": 3, "runs": 1, "notebooks": 0, "last_run": None}

    def status(self):
        return type("S", (), {"location": "x"})()


def _prune_engine(ws, monkeypatch, **kwargs):
    (ws / ".lamindb").mkdir(exist_ok=True)
    (ws / ".lamindb" / "lobster.json").write_text('{"slug": "anonymous/lobster-ws-1"}')
    monkeypatch.setattr(lamin_commands, "_engine_installed", lambda: True)
    engine = PruneEngine(**kwargs)
    monkeypatch.setattr(lamin_commands, "_open", lambda workspace: engine)
    return engine


def test_status_text_names_the_size_of_the_kept_versions(ws, monkeypatch):
    _prune_engine(ws, monkeypatch)
    output = _invoke("status", "-w", str(ws)).output
    assert "Earlier versions kept: 2.0 KB (files: 2)" in output
    assert "lobster lamin prune" in output


def test_status_text_is_quiet_without_kept_versions(ws, monkeypatch):
    _prune_engine(ws, monkeypatch, rows=[])
    assert "Earlier versions" not in _invoke("status", "-w", str(ws)).output


def test_prune_dry_run_lists_and_removes_nothing(ws, monkeypatch):
    engine = _prune_engine(ws, monkeypatch)
    result = _invoke("prune", "--dry-run", "-w", str(ws))
    assert result.exit_code == 0, result.output
    assert engine.pruned == 0
    assert "data/x.csv v0000" in result.output and "data/x.csv v0001" in result.output
    assert "Would remove: 2.0 KB (files: 2)." in result.output
    as_json = json.loads(_invoke("prune", "--dry-run", "--json", "-w", str(ws)).output)
    assert as_json == {
        "dry_run": True,
        "files": 2,
        "bytes": 2058,
        "versions": PruneEngine.ROWS,
        "removed": [],
        "failed": [],
        "skipped": [],
    }
    assert engine.pruned == 0


def test_prune_asks_first(ws, monkeypatch):
    engine = _prune_engine(ws, monkeypatch)
    declined = runner.invoke(lamin_app, ["prune", "-w", str(ws)], input="n\n")
    assert declined.exit_code == 1 and engine.pruned == 0
    assert "Nothing removed." in declined.output
    no_tty = runner.invoke(lamin_app, ["prune", "-w", str(ws)], input="")
    assert no_tty.exit_code == 1 and engine.pruned == 0
    accepted = runner.invoke(lamin_app, ["prune", "-w", str(ws)], input="y\n")
    assert accepted.exit_code == 0, accepted.output
    assert engine.pruned == 1
    assert "Removed: 2.0 KB (files: 2)." in accepted.output
    assert "records stay" in accepted.output.lower()


def test_prune_yes_json(ws, monkeypatch):
    engine = _prune_engine(ws, monkeypatch)
    result = _invoke("prune", "--yes", "--json", "-w", str(ws))
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert engine.pruned == 1
    assert (data["dry_run"], data["removed"], data["failed"]) == (
        False,
        PruneEngine.ROWS,
        [],
    )
    assert (data["files"], data["bytes"]) == (2, 2058)
    assert engine.shown == PruneEngine.ROWS  # what was shown is what is removed


def test_prune_json_keeps_the_question_off_stdout(ws, monkeypatch):
    _prune_engine(ws, monkeypatch)
    result = runner.invoke(lamin_app, ["prune", "--json", "-w", str(ws)], input="y\n")
    assert result.exit_code == 0, result.output
    # (the runner echoes the typed answer to stdout; a terminal shows it on its own)
    assert json.loads(result.stdout[result.stdout.index("{") :])["removed"] == (
        PruneEngine.ROWS
    )
    assert "Remove 2.0 KB" in result.stderr and "Remove" not in result.stdout


def test_prune_reports_files_it_could_not_remove(ws, monkeypatch):
    failed = [{"file": PruneEngine.ROWS[1]["file"], "error": "PermissionError"}]
    _prune_engine(ws, monkeypatch, failed=failed)
    result = _invoke("prune", "--yes", "-w", str(ws))
    assert result.exit_code == 1
    assert "Removed: 10.0 B (files: 1)." in result.output
    assert "PermissionError" in result.output and "v0001.csv" in result.output
    as_json = json.loads(_invoke("prune", "--yes", "--json", "-w", str(ws)).output)
    assert as_json["failed"] == failed and len(as_json["removed"]) == 1


def _stdout_is_one_document(result):
    assert (
        result.stdout.strip().startswith("{") and "LaminDB prints" not in result.stdout
    )
    return json.loads(result.stdout)


def test_prune_json_has_only_the_document_on_stdout(ws, monkeypatch):
    _prune_engine(ws, monkeypatch, chatty=True)
    result = _invoke("prune", "--yes", "--json", "-w", str(ws))
    assert result.exit_code == 0, result.output
    assert _stdout_is_one_document(result)["files"] == 2
    assert "LaminDB prints" in result.stderr


def test_status_json_has_only_the_document_on_stdout(ws, monkeypatch):
    _prune_engine(ws, monkeypatch, chatty=True)
    result = _invoke("status", "--json", "-w", str(ws))
    assert _stdout_is_one_document(result)["versions"]["files"] == 2


def test_prune_refuses_when_the_catalog_is_busy(ws, monkeypatch):
    from lobster.core.catalog.lamin import CatalogBusy

    engine = _prune_engine(ws, monkeypatch, raises=CatalogBusy("held"))
    result = _invoke("prune", "--yes", "-w", str(ws))
    assert result.exit_code == 1 and engine.pruned == 1
    assert "Another process is using the catalog" in result.stderr
    assert "Removed:" not in result.output


def test_prune_refuses_a_catalog_that_cannot_be_written(ws, monkeypatch):
    from lobster.core.catalog.lamin import CatalogReadOnly

    _prune_engine(ws, monkeypatch, raises=CatalogReadOnly("readonly"))
    result = _invoke("prune", "--yes", "--json", "-w", str(ws))
    assert result.exit_code == 1
    assert "The catalog cannot be written to; nothing was removed." in result.stderr
    assert result.stdout.strip() == ""


def test_a_failure_while_pruning_is_not_called_a_failure_to_open(ws, monkeypatch):
    _prune_engine(ws, monkeypatch, raises=RuntimeError("disk gone"))
    result = _invoke("prune", "--yes", "-w", str(ws))
    assert result.exit_code == 1
    assert "could not be pruned: RuntimeError: disk gone" in result.stderr
    assert "could not be opened" not in result.stderr


def test_prune_lists_what_it_skipped_and_fails(ws, monkeypatch):
    skipped = [{"file": PruneEngine.ROWS[1]["file"]}]
    _prune_engine(ws, monkeypatch, skipped=skipped)
    result = _invoke("prune", "--yes", "-w", str(ws))
    assert result.exit_code == 1
    assert "Not removed, it changed after it was listed: " in result.stderr
    assert "v0001.csv" in result.stderr
    as_json = json.loads(_invoke("prune", "--yes", "--json", "-w", str(ws)).stdout)
    assert as_json["skipped"] == skipped


def test_prune_with_nothing_kept(ws, monkeypatch):
    engine = _prune_engine(ws, monkeypatch, rows=[])
    result = _invoke("prune", "-w", str(ws))
    assert result.exit_code == 0 and "No earlier versions to remove." in result.output
    assert engine.pruned == 0
    as_json = json.loads(_invoke("prune", "--json", "-w", str(ws)).output)
    assert as_json["files"] == 0 and as_json["removed"] == []


def test_prune_without_a_catalog_or_engine(ws, monkeypatch):
    assert _invoke("prune", "-w", str(ws)).exit_code == 1
    _prune_engine(ws, monkeypatch)
    monkeypatch.setattr(lamin_commands, "_engine_installed", lambda: False)
    assert _invoke("prune", "-w", str(ws)).exit_code == 1


def test_a_catalog_that_cannot_be_opened_fails_prune(ws, monkeypatch):
    _damaged(ws, monkeypatch)
    result = _invoke("prune", "-w", str(ws))
    assert result.exit_code == 1 and "settings missing" in result.output


def test_prune_in_chinese(ws, monkeypatch):
    _prune_engine(ws, monkeypatch)
    set_lang("zh")
    try:
        status = _invoke("status", "-w", str(ws)).output
        dry = _invoke("prune", "--dry-run", "-w", str(ws)).output
        done = _invoke("prune", "--yes", "-w", str(ws)).output
    finally:
        set_lang("en")
    for output in (status, dry, done):
        assert any("\u4e00" <= ch <= "\u9fff" for ch in output), output
    assert "Earlier versions" not in status and "Removed:" not in done


def _damaged(ws, monkeypatch):
    (ws / ".lamindb").mkdir()
    (ws / ".lamindb" / "lobster.json").write_text('{"slug": "anonymous/lobster-ws-1"}')
    monkeypatch.setattr(lamin_commands, "_engine_installed", lambda: True)

    def broken(*args, **kwargs):
        raise RuntimeError("settings missing")

    monkeypatch.setattr(lamin_commands, "_open", broken)


def test_a_damaged_catalog_fails_status_in_text_and_json(ws, monkeypatch):
    _damaged(ws, monkeypatch)
    text = _invoke("status", "-w", str(ws))
    as_json = _invoke("status", "-w", str(ws), "--json")
    assert text.exit_code == as_json.exit_code == 1
    assert "settings missing" in text.output
    assert "settings missing" in json.loads(as_json.output)["problem"]


def test_status_in_instance_mode_labels_the_workspace_catalog(ws, monkeypatch):
    (ws / ".lamindb").mkdir()
    (ws / ".lamindb" / "lobster.json").write_text('{"slug": "anonymous/lobster-ws-1"}')
    monkeypatch.setattr(lamin_commands, "_engine_installed", lambda: False)
    _invoke("use", "lab/shared", "-w", str(ws))
    result = _invoke("status", "-w", str(ws))
    assert "Records go to the LaminDB instance lab/shared." in result.output
    assert "Workspace catalog (not in use): anonymous/lobster-ws-1" in result.output


def test_lamin_is_a_top_level_command():
    from lobster.cli import app

    result = runner.invoke(app, ["lamin", "--help"])
    assert result.exit_code == 0
    for name in ("status", "ls", "use", "init"):
        assert name in result.output


@pytest.mark.parametrize("error", [RuntimeError("db locked"), OSError("gone")])
def test_status_keeps_its_lines_when_the_kept_versions_cannot_be_counted(
    ws, monkeypatch, error
):
    _prune_engine(ws, monkeypatch, raises_on_kept=error)
    text = _invoke("status", "-w", str(ws))
    assert text.exit_code == 0, text.output
    assert "Instance: anonymous/lobster-ws-1" in text.output
    assert "Earlier versions: size unknown." in text.output
    assert "The catalog could not be opened" not in text.output
    info = json.loads(_invoke("status", "-w", str(ws), "--json").output)
    assert info["versions"] is None and info["contents"]["artifacts"] == 3
    assert info["location"] == "x" and info["problem"] is None


def test_status_says_the_size_is_unknown_when_the_engine_is_missing_or_will_not_open(
    ws, monkeypatch
):
    _prune_engine(ws, monkeypatch)
    monkeypatch.setattr(lamin_commands, "_engine_installed", lambda: False)
    text = _invoke("status", "-w", str(ws))
    assert text.exit_code == 0 and "Earlier versions: size unknown." in text.output
    assert (
        json.loads(_invoke("status", "-w", str(ws), "--json").output)["versions"]
        is None
    )


def test_a_catalog_that_will_not_open_keeps_its_exit_code_and_says_size_unknown(
    ws, monkeypatch
):
    ws.joinpath(".lamindb").mkdir(exist_ok=True)
    _damaged_again = ws / ".lamindb" / "lobster.json"
    _damaged_again.write_text('{"slug": "anonymous/lobster-ws-1"}')
    monkeypatch.setattr(lamin_commands, "_engine_installed", lambda: True)

    def broken(*args, **kwargs):
        raise RuntimeError("settings missing")

    monkeypatch.setattr(lamin_commands, "_open", broken)
    text = _invoke("status", "-w", str(ws))
    assert text.exit_code == 1 and "settings missing" in text.output
    assert "Earlier versions: size unknown." in text.output
    assert (
        json.loads(_invoke("status", "-w", str(ws), "--json").output)["versions"]
        is None
    )


def test_the_size_is_not_unknown_without_a_catalog(ws):
    text = _invoke("status", "-w", str(ws))
    assert "size unknown" not in text.output


def test_size_unknown_in_chinese(ws, monkeypatch):
    _prune_engine(ws, monkeypatch, raises_on_kept=RuntimeError("x"))
    set_lang("zh")
    try:
        output = _invoke("status", "-w", str(ws)).output
    finally:
        set_lang("en")
    assert "size unknown" not in output
    assert any("\u4e00" <= ch <= "\u9fff" for ch in output)
