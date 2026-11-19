"""An empty XDG_CONFIG_HOME or APPDATA means unset: it never names ./lobster."""

import importlib
import platform
import types
from pathlib import Path

import pytest

from lobster import i18n
from lobster.config import global_config


@pytest.fixture
def reload_config(monkeypatch):
    """Reload global_config under the test's environment, then reload it back."""
    yield lambda: importlib.reload(global_config)
    monkeypatch.undo()
    importlib.reload(global_config)


@pytest.mark.parametrize("system", ["Linux", "Windows"])
def test_the_config_folder_ignores_an_empty_variable(
    tmp_path, monkeypatch, reload_config, system
):
    (tmp_path / "home").mkdir()
    (tmp_path / "here").mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_CONFIG_HOME", "")
    monkeypatch.setenv("APPDATA", "")
    monkeypatch.setattr(platform, "system", lambda: system)
    monkeypatch.chdir(tmp_path / "here")

    folder = reload_config().CONFIG_DIR

    assert folder.is_absolute() and folder != Path.cwd() / "lobster"


@pytest.mark.parametrize("name", ["posix", "nt"])
def test_the_language_lookup_ignores_an_empty_variable(tmp_path, monkeypatch, name):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(i18n, "os", types.SimpleNamespace(name=name, environ={}))

    folder = i18n._global_config_dir({"XDG_CONFIG_HOME": "", "APPDATA": ""})

    assert folder.is_absolute() and folder != Path.cwd() / "lobster"
