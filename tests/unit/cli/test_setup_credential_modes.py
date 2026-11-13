"""Credential file creation and rewrite permissions under a normal umask."""

import importlib
import inspect
import io
import os
import stat
from types import SimpleNamespace

import pytest
import typer
from rich.console import Console
from typer.testing import CliRunner

pytestmark = pytest.mark.no_auto_config
KEY = "synthetic-mode-test-key"


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    # Resolve at call time: the CLI decomposition tests reload these modules.
    global_config = importlib.import_module("lobster.config.global_config")
    credentials = importlib.import_module("lobster.config.credentials")
    oauth_store = importlib.import_module("lobster.config.auth.oauth_store")
    init_commands = importlib.import_module(
        "lobster.cli_internal.commands.heavy.init_commands"
    )
    uv_tool_env = importlib.import_module("lobster.core.uv_tool_env")
    i18n = importlib.import_module("lobster.i18n")
    monkeypatch.chdir(tmp_path)
    # Start each test as a plain English run with no language choice yet.
    monkeypatch.setattr(i18n, "_LANG", "en")
    monkeypatch.setattr(i18n, "_SOURCE", "default")
    monkeypatch.setattr(global_config, "CONFIG_DIR", tmp_path / "global")
    monkeypatch.setattr(credentials, "CREDENTIALS_DIR", tmp_path / "profiles")
    monkeypatch.setattr(
        credentials, "CREDENTIALS_FILE", tmp_path / "profiles/credentials.json"
    )
    monkeypatch.setattr(oauth_store, "AUTH_DIR", tmp_path / "oauth")
    monkeypatch.setattr(init_commands, "console", Console(width=2000))
    monkeypatch.setattr(uv_tool_env, "is_uv_tool_env", lambda: False)
    for name in (
        "OMICS_OS_API_KEY",
        "OMICS_OS_ENDPOINT",
        "LOBSTER_LLM_PROVIDER",
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "LOBSTER_WORKSPACE",
    ):
        monkeypatch.delenv(name, raising=False)
    original_umask = os.umask(0o022)
    try:
        yield
    finally:
        os.umask(original_umask)


def invoke(*options):
    from lobster.cli import app

    return CliRunner().invoke(
        app, ["init", "--non-interactive", "--skip-extras", "--skip-ssl-test", *options]
    )


class ObservedStream:
    """Record permissions immediately before data reaches the real writer."""

    def __init__(self, stream, modes):
        self.stream = stream
        self.modes = modes

    def write(self, data):
        self.modes.append(stat.S_IMODE(os.fstat(self.stream.fileno()).st_mode))
        return self.stream.write(data)

    def __getattr__(self, name):
        return getattr(self.stream, name)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return self.stream.__exit__(*args)


@pytest.mark.parametrize("existing", [False, True], ids=["create", "rewrite-0644"])
@pytest.mark.parametrize(
    "writer",
    [
        "workspace-bridge",
        "global-env",
        "oauth-json",
        "v2-json",
    ],
)
def test_each_writer_is_private_before_first_write_and_after_save(
    writer, existing, monkeypatch, tmp_path
):
    from lobster.config import credentials, global_config
    from lobster.config.auth import oauth_store
    from lobster.ui.bridge import init_adapter

    paths = {
        "workspace-bridge": tmp_path / ".env",
        "global-env": global_config.get_global_credentials_path(),
        "global-json": global_config.CONFIG_DIR / global_config.CONFIG_FILE_NAME,
        "workspace-json": tmp_path / ".lobster_workspace/provider_config.json",
        "oauth-json": oauth_store.AUTH_DIR / "synthetic.json",
        "v2-json": credentials.CREDENTIALS_FILE,
    }
    path = paths[writer]
    path.parent.mkdir(parents=True, exist_ok=True)
    if existing:
        path.write_text("legacy synthetic contents\n")
        path.chmod(0o644)
    targets = {path}
    if writer == "v2-json":
        targets.add(path.with_suffix(f".tmp.{os.getpid()}"))
    modes = []
    original_open, original_fdopen = io.open, os.fdopen

    def observed_open(file, *args, **kwargs):
        stream = original_open(file, *args, **kwargs)
        if not isinstance(file, int) and any(
            os.fspath(file) == str(p) for p in targets
        ):
            return ObservedStream(stream, modes)
        return stream

    def observed_fdopen(fd, *args, **kwargs):
        stream = original_fdopen(fd, *args, **kwargs)
        inode = os.fstat(fd).st_ino
        if any(p.exists() and p.stat().st_ino == inode for p in targets):
            return ObservedStream(stream, modes)
        return stream

    monkeypatch.setattr(io, "open", observed_open)
    monkeypatch.setattr(os, "fdopen", observed_fdopen)
    # builtins.open is used by the bridge rather than Path.open/io.open.
    monkeypatch.setattr("builtins.open", observed_open)
    if writer == "workspace-bridge":
        init_adapter._write_workspace_env(path, [f"OMICS_OS_API_KEY={KEY}"])
    elif writer == "global-env":
        global_config.save_global_credentials({"OMICS_OS_API_KEY": KEY})
    elif writer == "oauth-json":
        oauth_store.save(
            oauth_store.OAuthCredentials(KEY, "synthetic-refresh", 1, "synthetic")
        )
    else:
        credentials.save_credentials({"api_key": KEY})
    assert modes, "the real data-writing boundary was not observed"
    assert set(modes) == {0o600}, f"{writer}: write modes {modes}"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert path.read_bytes()


@pytest.mark.parametrize(
    "existing", [False, True], ids=["create", "force-rewrite-0644"]
)
def test_noninteractive_workspace_key_is_private_and_nonsecret_json_unchanged(
    existing, tmp_path
):
    env_path = tmp_path / ".env"
    config_path = tmp_path / ".lobster_workspace/provider_config.json"
    if existing:
        assert invoke("--omics-os-key", "synthetic-old-key").exit_code == 0
        env_path.chmod(0o644)
        config_path.chmod(0o644)
    result = invoke("--omics-os-key", KEY, *(["--force"] if existing else []))
    assert result.exit_code == 0, result.output
    assert stat.S_IMODE(env_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(config_path.stat().st_mode) == 0o644
    assert KEY not in config_path.read_text()
    assert f"OMICS_OS_API_KEY={KEY}" in env_path.read_text()
    assert KEY not in result.output


@pytest.mark.parametrize("global_mode", [False, True], ids=["workspace", "global"])
def test_force_backups_of_legacy_modes_are_private_and_byte_exact(
    global_mode, tmp_path
):
    from lobster.config import global_config

    options = ["--global"] if global_mode else []
    assert invoke(*options, "--omics-os-key", "synthetic-old-key").exit_code == 0
    if global_mode:
        paths = [global_config.CONFIG_DIR / global_config.CONFIG_FILE_NAME]
    else:
        paths = [
            tmp_path / ".env",
            tmp_path / ".lobster_workspace/provider_config.json",
        ]
    before = {path: path.read_bytes() for path in paths}
    for path in paths:
        path.chmod(0o644)
    result = invoke(*options, "--omics-os-key", KEY, "--force")
    assert result.exit_code == 0, result.output
    for path, content in before.items():
        backups = list(path.parent.glob(path.name + ".backup.*"))
        assert len(backups) == 1
        assert backups[0].read_bytes() == content
        expected_mode = 0o600 if path.name == ".env" else 0o644
        assert stat.S_IMODE(backups[0].stat().st_mode) == expected_mode
        assert stat.S_IMODE(path.stat().st_mode) == expected_mode
    assert KEY not in result.output


def test_nonsecret_provider_jsons_retain_their_modes(monkeypatch, tmp_path):
    from lobster.config import global_config, workspace_config

    workspace = tmp_path / ".lobster_workspace"
    workspace.mkdir()
    global_config.CONFIG_DIR.mkdir()
    cases = [
        (
            global_config.GlobalProviderConfig(default_provider="omics-os"),
            global_config.CONFIG_DIR / "providers.json",
            None,
        ),
        (
            workspace_config.WorkspaceProviderConfig(global_provider="omics-os"),
            workspace / "provider_config.json",
            workspace,
        ),
    ]
    for config, path, argument in cases:
        assert not any(
            word in name.lower()
            for name in type(config).model_fields
            for word in ("key", "token", "secret", "password", "credential")
        )
        for original_mode in (0o644, 0o640):
            path.write_text("legacy nonsecret metadata")
            path.chmod(original_mode)
            config.save() if argument is None else config.save(argument)
            assert stat.S_IMODE(path.stat().st_mode) == original_mode
            assert KEY not in path.read_text()


@pytest.mark.parametrize("existing", [False, True])
def test_classic_workspace_writer_is_private(existing, monkeypatch, tmp_path):
    from lobster.cli_internal.commands.heavy import init_commands

    if existing:
        (tmp_path / ".env").write_text("ANTHROPIC_API_KEY=synthetic-old-key\n")
        (tmp_path / ".env").chmod(0o644)
    monkeypatch.setattr("sys.stdin", SimpleNamespace(isatty=lambda: True))
    answers = iter(["1", "1", KEY])  # language (English), provider, key
    monkeypatch.setattr(init_commands.Prompt, "ask", lambda *a, **kw: next(answers))
    monkeypatch.setattr(init_commands.Confirm, "ask", lambda *a, **kw: False)
    monkeypatch.setattr(
        init_commands,
        "_perform_agent_selection_interactive",
        lambda *a, **kw: ([], None),
    )
    options = {
        name: (
            param.default.default
            if isinstance(param.default, typer.models.OptionInfo)
            else param.default
        )
        for name, param in inspect.signature(init_commands.init_impl).parameters.items()
    }
    options.update(
        ui_mode="classic", skip_extras=True, skip_ssl_test=True, force=existing
    )
    try:
        init_commands.init_impl(**options)
    except typer.Exit as error:
        assert error.exit_code == 0
    path = tmp_path / ".env"
    assert f"OMICS_OS_API_KEY={KEY}" in path.read_text()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_secret_backup_is_private_before_the_first_byte(monkeypatch, tmp_path):
    assert invoke("--omics-os-key", "synthetic-old-key").exit_code == 0
    (tmp_path / ".env").chmod(0o644)
    modes = []
    original_open, original_fdopen = io.open, os.fdopen

    def observed_open(file, *args, **kwargs):
        stream = original_open(file, *args, **kwargs)
        if not isinstance(file, int) and ".env.backup." in os.fspath(file):
            return ObservedStream(stream, modes)
        return stream

    def observed_fdopen(fd, *args, **kwargs):
        stream = original_fdopen(fd, *args, **kwargs)
        inode = os.fstat(fd).st_ino
        if any(p.stat().st_ino == inode for p in tmp_path.glob(".env.backup.*")):
            return ObservedStream(stream, modes)
        return stream

    monkeypatch.setattr(io, "open", observed_open)
    monkeypatch.setattr("builtins.open", observed_open)
    monkeypatch.setattr(os, "fdopen", observed_fdopen)
    # Use copyfileobj's observable writer, not the OS sendfile fast path.
    monkeypatch.setattr("shutil._USE_CP_SENDFILE", False)
    monkeypatch.setattr("shutil._HAS_FCOPYFILE", False)
    result = invoke("--omics-os-key", KEY, "--force")
    assert result.exit_code == 0, result.output
    assert modes
    assert set(modes) == {0o600}


def test_noncloud_bridge_global_secret_failure_propagates(monkeypatch):
    from lobster.config import global_config
    from lobster.ui.bridge import init_adapter

    def refused(*args, **kwargs):
        raise PermissionError("synthetic mode denial")

    monkeypatch.setattr(global_config, "save_global_credentials", refused)
    with pytest.raises(PermissionError, match="synthetic mode denial"):
        init_adapter._write_global(
            {"provider": "anthropic"}, [f"ANTHROPIC_API_KEY={KEY}"]
        )


@pytest.mark.parametrize("operation", ["creation", "chmod"])
@pytest.mark.parametrize("global_mode", [False, True])
def test_secret_permission_failure_is_terminal_without_saved_key(
    operation, global_mode, monkeypatch, tmp_path
):
    from lobster.config import global_config

    target = (
        global_config.get_global_credentials_path()
        if global_mode
        else tmp_path / ".env"
    )
    original_open, original_chmod = os.open, os.fchmod

    def guarded_open(path, *args, **kwargs):
        if operation == "creation" and os.fspath(path) == str(target):
            raise PermissionError("synthetic secret creation denial")
        return original_open(path, *args, **kwargs)

    def guarded_chmod(fd, *args, **kwargs):
        if target.exists() and os.fstat(fd).st_ino == target.stat().st_ino:
            raise PermissionError("synthetic secret mode denial")
        return original_chmod(fd, *args, **kwargs)

    monkeypatch.setattr(os, "open", guarded_open)
    if operation == "chmod":
        monkeypatch.setattr(os, "fchmod", guarded_chmod)
    result = invoke("--omics-os-key", KEY, *(["--global"] if global_mode else []))
    assert result.exit_code == 1
    assert "partial" in result.output.lower()
    assert "setup is not complete" in result.output.lower()
    assert "configuration saved" not in result.output.lower()
    assert "saved, not checked" not in result.output.lower()
    assert not target.exists() or KEY not in target.read_text()


@pytest.mark.parametrize(
    "noninteractive", [False, True], ids=["human", "noninteractive"]
)
@pytest.mark.parametrize("global_mode", [False, True], ids=["workspace", "global"])
def test_init_skip_tightens_legacy_secret_without_changing_bytes(
    noninteractive, global_mode, tmp_path
):
    from lobster.cli import app
    from lobster.config import global_config

    options = ["--global"] if global_mode else []
    assert invoke(*options, "--omics-os-key", KEY).exit_code == 0
    path = (
        global_config.get_global_credentials_path()
        if global_mode
        else tmp_path / ".env"
    )
    original = path.read_bytes()
    path.chmod(0o644)
    result = (
        invoke(*options, "--use-omics-os")
        if noninteractive
        else CliRunner().invoke(app, ["init", *options])
    )
    assert result.exit_code == 0, result.output
    assert path.read_bytes() == original
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert f"tightened permissions on {path} to 0600" in result.output
    assert not list(path.parent.glob(path.name + ".backup.*"))
    assert KEY not in result.output


def test_init_skip_mode_failure_warns_with_manual_command_but_preserves_exit(
    monkeypatch, tmp_path
):
    assert invoke("--omics-os-key", KEY).exit_code == 0
    path = tmp_path / ".env"
    original = path.read_bytes()
    path.chmod(0o644)

    def refused(*args, **kwargs):
        raise PermissionError("synthetic repair denial")

    monkeypatch.setattr(os, "chmod", refused)
    monkeypatch.setattr(os, "fchmod", refused)
    result = invoke("--use-omics-os")
    assert result.exit_code == 0, result.output
    assert path.read_bytes() == original
    assert stat.S_IMODE(path.stat().st_mode) == 0o644
    assert "warning" in result.output.lower()
    assert f"chmod 600 {path}" in result.output
    assert "tightened permissions" not in result.output
    assert KEY not in result.output


@pytest.mark.parametrize("kind", ["missing", "directory", "foreign-owned"])
def test_skip_repair_does_not_claim_unowned_or_missing_secret(
    kind, monkeypatch, tmp_path, capsys
):
    from pathlib import Path

    from lobster.cli_internal.commands.heavy import init_commands

    path = tmp_path / "legacy-secret"
    if kind == "directory":
        path.mkdir()
    elif kind == "foreign-owned":
        path.write_text(KEY)
        path.chmod(0o644)
        original_lstat = Path.lstat

        def foreign_lstat(self, *args, **kwargs):
            info = original_lstat(self, *args, **kwargs)
            if self == path:
                return SimpleNamespace(st_mode=info.st_mode, st_uid=os.getuid() + 1)
            return info

        monkeypatch.setattr(Path, "lstat", foreign_lstat)
    init_commands._repair_skipped_secret_permissions(path)
    assert capsys.readouterr().out == ""
    if kind == "foreign-owned":
        assert path.read_text() == KEY
        assert stat.S_IMODE(path.stat().st_mode) == 0o644


def test_global_permission_failure_is_not_silently_accepted(monkeypatch):
    from lobster.config import global_config

    def refused(*args, **kwargs):
        raise PermissionError("synthetic mode denial")

    monkeypatch.setattr(os, "fchmod", refused)
    monkeypatch.setattr(os, "chmod", refused)
    with pytest.raises(PermissionError, match="synthetic mode denial"):
        global_config.save_global_credentials({"OMICS_OS_API_KEY": KEY})
    path = global_config.get_global_credentials_path()
    assert not path.exists() or KEY not in path.read_text()
