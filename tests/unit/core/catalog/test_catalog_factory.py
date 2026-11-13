"""Catalog selection: off by default, resolution order, one catalog per process."""

import json
import logging
import os

import pytest

from lobster.core.catalog import factory
from lobster.core.catalog.factory import get_catalog, parse_flag, resolve_catalog_config
from lobster.core.catalog.null import NullCatalog
from lobster.core.catalog.thread import ThreadedCatalog

pytestmark = pytest.mark.unit


@pytest.fixture
def ws(tmp_path):
    path = tmp_path / "ws"
    path.mkdir()
    return path


@pytest.fixture
def global_json(tmp_path):
    def write(data):
        path = tmp_path / "providers.json"
        path.write_text(json.dumps(data))
        return path

    return write


@pytest.mark.parametrize(
    "value,expected",
    [
        ("1", True),
        ("true", True),
        (" YES ", True),
        ("on", True),
        ("0", False),
        ("False", False),
        ("off", False),
        ("", None),
        ("maybe", None),
        (None, None),
        (True, True),
        (False, False),
        (1, None),
    ],
)
def test_parse_flag(value, expected):
    assert parse_flag(value) is expected


def test_off_by_default(ws, tmp_path):
    config = resolve_catalog_config(
        ws, environ={}, global_config_path=tmp_path / "missing.json"
    )
    assert (config.enabled, config.source, config.instance, config.mode) == (
        False,
        "default",
        None,
        "workspace",
    )


def test_env_beats_workspace_env_beats_global(ws, global_json):
    (ws / ".env").write_text("LOBSTER_LAMIN=0\n")
    gpath = global_json({"lamin_enabled": True})

    assert (
        resolve_catalog_config(
            ws, environ={"LOBSTER_LAMIN": "1"}, global_config_path=gpath
        ).source
        == "env"
    )
    config = resolve_catalog_config(ws, environ={}, global_config_path=gpath)
    assert (config.enabled, config.source) == (False, "workspace")
    (ws / ".env").unlink()
    config = resolve_catalog_config(ws, environ={}, global_config_path=gpath)
    assert (config.enabled, config.source) == (True, "global")


def test_unparseable_values_fall_through(ws, global_json):
    (ws / ".env").write_text("LOBSTER_LAMIN=perhaps\n")
    gpath = global_json({"lamin_enabled": "yes"})
    config = resolve_catalog_config(
        ws, environ={"LOBSTER_LAMIN": "?"}, global_config_path=gpath
    )
    assert (config.enabled, config.source) == (True, "global")


def test_broken_global_config_means_off(ws, tmp_path):
    bad = tmp_path / "providers.json"
    bad.write_text("{not json")
    assert (
        resolve_catalog_config(ws, environ={}, global_config_path=bad).enabled is False
    )


def test_instance_resolution(ws):
    (ws / ".env").write_text("LOBSTER_LAMIN_INSTANCE=lab/ws-instance\n")
    env = {
        "LOBSTER_LAMIN_INSTANCE": "lab/env-instance",
        "LAMIN_CURRENT_INSTANCE": "lab/ambient",
    }
    assert resolve_catalog_config(ws, environ=env).instance == "lab/env-instance"
    env.pop("LOBSTER_LAMIN_INSTANCE")
    config = resolve_catalog_config(ws, environ=env)
    assert (config.instance, config.instance_source, config.mode) == (
        "lab/ws-instance",
        "workspace",
        "instance",
    )
    (ws / ".env").unlink()
    assert resolve_catalog_config(ws, environ=env).instance == "lab/ambient"
    assert (
        resolve_catalog_config(ws, environ={"LAMIN_CURRENT_INSTANCE": "  "}).instance
        is None
    )


def test_disabled_gives_null_catalog(ws, tmp_path):
    catalog = get_catalog(ws, environ={}, global_config_path=tmp_path / "none.json")
    assert isinstance(catalog, NullCatalog) and catalog.status().mode == "off"


def test_enabled_without_engine_gives_null_with_reason_and_one_hint(
    ws, tmp_path, monkeypatch, caplog
):
    monkeypatch.setattr(factory, "_load_engine", lambda: None)
    with caplog.at_level(logging.WARNING, logger="lobster.core.catalog.factory"):
        first = get_catalog(ws, environ={"LOBSTER_LAMIN": "1"})
        second = get_catalog(ws, environ={"LOBSTER_LAMIN": "1"})
    assert isinstance(first, NullCatalog) and isinstance(second, NullCatalog)
    assert "not installed" in first.status().reason
    assert sum("not installed" in r.getMessage() for r in caplog.records) == 1


def test_engine_needs_the_extra(monkeypatch):
    """Without lamindb_setup installed there is no engine, whatever else is present."""
    import importlib.util

    real = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *a: None if name == "lamindb_setup" else real(name, *a),
    )
    assert factory._load_engine() is None


def test_engine_module_imports_no_lamin():
    """The engine module itself stays importable without the extra."""
    import subprocess
    import sys

    code = (
        "import sys; import lobster.core.catalog.lamin; "
        "print(sorted(m for m in sys.modules if m.startswith(('lamindb', 'django'))))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0 and out.stdout.strip() == "[]", out.stderr


def test_enabled_with_engine_is_threaded_and_process_wide(
    ws, tmp_path, monkeypatch, recording_catalog_cls
):
    built = []

    def engine_class(workspace, instance=None):
        engine = recording_catalog_cls(workspace, instance)
        built.append(engine)
        return engine

    monkeypatch.setattr(factory, "_load_engine", lambda: engine_class)
    env = {"LOBSTER_LAMIN": "1", "LOBSTER_LAMIN_INSTANCE": "lab/x"}
    catalog = get_catalog(ws, environ=env)
    assert isinstance(catalog, ThreadedCatalog)
    assert get_catalog(ws, environ=env) is catalog
    catalog.status()  # waits for the engine build on the worker
    assert built[0].workspace_path == ws.resolve() and built[0].instance == "lab/x"
    assert catalog.status().mode == "workspace"  # the engine's own answer

    other = tmp_path / "other"
    other.mkdir()
    second = get_catalog(other, environ=env)
    assert isinstance(second, NullCatalog)
    assert "already records" in second.status().reason
    assert len(built) == 1


def test_named_instance_discloses_copy_and_server_contact(
    ws, monkeypatch, caplog, recording_catalog_cls
):
    monkeypatch.setattr(factory, "_load_engine", lambda: recording_catalog_cls)
    env = {"LOBSTER_LAMIN": "1", "LOBSTER_LAMIN_INSTANCE": "lab/shared"}
    with caplog.at_level(logging.WARNING, logger=factory.__name__):
        catalog = get_catalog(ws, environ=env)
    assert isinstance(catalog, ThreadedCatalog)
    warning = " ".join(
        r.getMessage() for r in caplog.records if r.levelno == logging.WARNING
    )
    assert "'lab/shared' (setting from env)" in warning
    assert "copied into that instance's storage" in warning
    assert (
        "contacts Lamin's servers (lamin.ai) only if the instance is registered"
        in warning
    )


def test_workspace_catalog_does_not_print_the_instance_warning(
    ws, monkeypatch, caplog, recording_catalog_cls
):
    monkeypatch.setattr(factory, "_load_engine", lambda: recording_catalog_cls)
    with caplog.at_level(logging.WARNING, logger=factory.__name__):
        get_catalog(ws, environ={"LOBSTER_LAMIN": "1"})
    assert not [r for r in caplog.records if "servers" in r.getMessage()]


def test_selection_never_raises(ws, monkeypatch):
    def explode(*a, **k):
        raise RuntimeError("bad settings")

    monkeypatch.setattr(factory, "resolve_catalog_config", explode)
    catalog = get_catalog(ws)
    assert isinstance(catalog, NullCatalog) and catalog.status().reason


@pytest.mark.parametrize(
    "shell,workspace,enabled,source",
    [
        ("0", "1", False, "env"),  # LOBSTER_LAMIN=0 lobster chat after lamin init
        ("1", "0", True, "env"),
        (None, "1", True, "workspace"),
        (None, "0", False, "workspace"),
    ],
)
def test_loading_credentials_keeps_the_shell_above_the_workspace(
    ws, tmp_path, monkeypatch, shell, workspace, enabled, source
):
    """Settings load the workspace .env into os.environ; the catalog keys stay out."""
    from lobster.config import global_config
    from lobster.config.settings import get_settings

    monkeypatch.setattr(global_config, "global_credentials_exist", lambda: False)
    for key in ("LOBSTER_LAMIN", "LOBSTER_LAMIN_INSTANCE"):
        monkeypatch.delenv(key, raising=False)
    if shell is not None:
        monkeypatch.setenv("LOBSTER_LAMIN", shell)
    (ws / ".env").write_text(
        f"LOBSTER_LAMIN={workspace}\nLOBSTER_LAMIN_INSTANCE=lab/ws\n"
    )
    get_settings().reload_credentials(ws)
    assert os.environ.get("LOBSTER_LAMIN") == shell
    assert "LOBSTER_LAMIN_INSTANCE" not in os.environ
    config = resolve_catalog_config(ws, global_config_path=tmp_path / "none.json")
    assert (config.enabled, config.source) == (enabled, source)
    assert (config.instance, config.instance_source) == ("lab/ws", "workspace")
