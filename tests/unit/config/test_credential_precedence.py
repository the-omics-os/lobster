"""Which value wins when the same variable sits in several places.

Order, highest first: workspace ``.env`` (``./.env`` when no workspace is given) and
cwd ``.env``, a real environment variable, a ``.env`` found upward from the installed
package, the global ``credentials.env``. The workspace/cwd ``.env`` overrides the
environment; the upward ``.env`` and the global file only fill what is unset.

The real start-up does two loads: ``Settings()`` at import with no workspace, then
``reload_credentials(workspace)``. ``_load`` runs both.
"""

import os
from pathlib import Path

import pytest

from lobster.config.settings import Settings

KEY = "PREC_TEST_KEY"
LAMIN = ("LOBSTER_LAMIN", "LOBSTER_LAMIN_INSTANCE")


@pytest.fixture
def world(tmp_path, monkeypatch):
    saved = dict(os.environ)
    for name in (KEY, *LAMIN):
        monkeypatch.delenv(name, raising=False)
    config_dir = tmp_path / "home" / ".config" / "lobster"
    config_dir.mkdir(parents=True)
    monkeypatch.setattr("lobster.config.global_config.CONFIG_DIR", config_dir)
    cwd = tmp_path / "cwd"
    ws = tmp_path / "ws"
    upward = tmp_path / "venv_folder"
    for folder in (cwd, ws, upward):
        folder.mkdir()
    monkeypatch.chdir(cwd)

    # What python-dotenv's upward search would find. Patched in both places so the old
    # path-less ``load_dotenv`` and a direct ``find_dotenv`` call see the same file.
    def find(*args, **kwargs):
        found = upward / ".env"
        return str(found) if found.exists() else ""

    monkeypatch.setattr("dotenv.main.find_dotenv", find)
    monkeypatch.setattr("lobster.config.settings.find_dotenv", find, raising=False)

    class World:
        files = {
            "W": ws / ".env",
            "C": cwd / ".env",
            "U": upward / ".env",
            "G": config_dir / "credentials.env",
        }
        workspace = ws

        def put(self, where, body=None):
            self.files[where].write_text(
                body if body is not None else f"{KEY}={where}\n"
            )

    yield World()
    os.environ.clear()
    os.environ.update(saved)


def _load(world, sources, shell=None, workspace=True):
    for where in sources:
        world.put(where)
    if shell is not None:
        os.environ[KEY] = shell
    settings = Settings.__new__(Settings)
    settings._load_credentials()  # what ``Settings()`` does at import
    if workspace:
        settings._load_credentials(world.workspace)  # session start-up reload
    return os.environ.get(KEY)


# (sources, shell value, workspace given, winner)
ROWS = [
    ("", "S", True, "S"),
    ("G", None, True, "G"),
    ("G", "S", True, "S"),
    ("U", None, True, "U"),
    ("UG", None, True, "U"),  # upward beats global (as before)
    ("U", "S", True, "S"),  # a real export is not overridden by an upward .env
    ("U", "S", False, "S"),
    ("UG", "S", True, "S"),
    ("C", None, False, "C"),
    ("C", None, True, "C"),  # cwd .env honoured when the workspace has none
    ("C", "S", False, "C"),  # cwd .env still wins over the environment
    ("C", "S", True, "C"),
    ("W", "S", True, "W"),  # workspace .env still wins over the environment
    ("WC", None, True, "W"),
    ("CU", None, False, "C"),
    ("CU", None, True, "C"),  # upward no longer beats cwd
    ("WU", None, True, "W"),
    ("WU", "S", True, "W"),
    ("GC", None, True, "C"),
    ("GW", None, True, "W"),
    ("WCUG", "S", True, "W"),
]


@pytest.mark.parametrize("sources,shell,workspace,winner", ROWS)
def test_precedence(world, sources, shell, workspace, winner):
    assert _load(world, sources, shell, workspace) == winner


def test_workspace_reload_is_idempotent(world):
    first = _load(world, "WUG", shell="S")
    settings = Settings.__new__(Settings)
    settings._load_credentials(world.workspace)
    assert first == os.environ[KEY] == "W"


@pytest.mark.parametrize("where", ["W", "C", "U", "G"])
@pytest.mark.parametrize("shell", [None, "0"])
def test_shell_only_keys_never_come_from_a_file(world, where, shell):
    body = "LOBSTER_LAMIN=1\nLOBSTER_LAMIN_INSTANCE=from-file\n"
    for name in LAMIN:
        if shell is not None:
            os.environ[name] = shell
    world.put(where, body)
    settings = Settings.__new__(Settings)
    settings._load_credentials()
    settings._load_credentials(world.workspace)
    for name in LAMIN:
        assert os.environ.get(name) == shell


def test_cwd_env_is_read_by_the_reload_alone(world):
    """No import-time load before the reload (the process changed folder after it)."""
    world.put("C")
    os.environ[KEY] = "S"
    Settings.__new__(Settings)._load_credentials(world.workspace)
    assert os.environ[KEY] == "C"
