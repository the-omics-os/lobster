"""``lobster purge --scope global`` removes Lobster's own files, never the user's."""

import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from lobster.cli_internal.commands.light.purge_commands import (
    LobsterPaths,
    PurgeTarget,
    execute_purge,
)


@pytest.fixture
def home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home / ".config"))
    monkeypatch.setenv("APPDATA", str(home / ".config"))
    monkeypatch.chdir(tmp_path)
    return home


@pytest.fixture
def lobster_home(home):
    folder = home / ".lobster"
    folder.mkdir()
    (folder / "lobster_history").write_text("ls\n")
    (folder / ".version_check_cache.json").write_text("{}")
    (folder / "vector_store").mkdir()
    (folder / "vector_store" / "index.bin").write_text("x")
    return folder


@pytest.fixture
def config_dir(home):
    folder = home / ".config" / "lobster"
    folder.mkdir(parents=True)
    (folder / "providers.json").write_text("{}")
    (folder / "credentials.env").write_text("K=v\n")
    (folder / "projects.json").write_text("{}")
    return folder


def purge(*args):
    from lobster.cli import app

    return CliRunner().invoke(app, ["purge", "--scope", "global", *args])


def flat(result) -> str:
    return " ".join(result.output.split())


def test_the_users_file_in_the_lobster_home_folder_is_kept(lobster_home):
    (lobster_home / "notes.txt").write_text("mine")

    result = purge("--force")

    assert result.exit_code == 0, result.output
    assert (lobster_home / "notes.txt").read_text() == "mine"
    assert not (lobster_home / "lobster_history").exists()
    assert not (lobster_home / "vector_store").exists()
    assert "kept: notes.txt" in flat(result)


def test_the_users_file_in_the_config_folder_is_kept(config_dir):
    (config_dir / "my_script.sh").write_text("echo")

    result = purge("--force")

    assert result.exit_code == 0, result.output
    assert (config_dir / "my_script.sh").exists()
    assert not (config_dir / "providers.json").exists()
    assert not (config_dir / "credentials.env").exists()


def test_a_dry_run_lists_what_is_kept(lobster_home):
    (lobster_home / "notes.txt").write_text("mine")

    result = purge("--dry-run")

    assert result.exit_code == 0, result.output
    assert (lobster_home / "lobster_history").exists()
    assert "kept: notes.txt" in flat(result)


def test_folders_with_only_lobster_files_are_removed(lobster_home, config_dir):
    (lobster_home / ".DS_Store").write_text("")

    result = purge("--force")

    assert result.exit_code == 0, result.output
    assert not lobster_home.exists() and not config_dir.exists()


def test_a_linked_lobster_home_folder_is_refused(home, tmp_path):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "lobster_history").write_text("ls\n")
    (elsewhere / "notes.txt").write_text("mine")
    try:
        (home / ".lobster").symlink_to(elsewhere, target_is_directory=True)
    except OSError:
        pytest.skip("this system cannot make links")

    result = purge("--force")

    assert (elsewhere / "lobster_history").exists(), result.output
    assert (elsewhere / "notes.txt").exists()
    assert "is reached through a link" in flat(result)


def test_removal_keeps_unknown_files_even_when_told_it_is_verified(lobster_home):
    (lobster_home / "notes.txt").write_text("mine")
    target = PurgeTarget(path=lobster_home, description="", is_verified=True)

    result = execute_purge([target])

    assert result.success, result.errors
    assert (lobster_home / "notes.txt").exists()
    assert not (lobster_home / "lobster_history").exists()


def test_an_empty_config_variable_does_not_name_a_folder_here(home, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", "")
    monkeypatch.setenv("APPDATA", "")
    here = Path.cwd() / "lobster"
    here.mkdir()
    (here / "providers.json").write_text("{}")
    (here / "notes.txt").write_text("mine")

    result = purge("--force")

    assert LobsterPaths.get_global_config_dir().is_absolute()
    assert (here / "providers.json").exists() and (here / "notes.txt").exists()
    assert result.exit_code == 0, result.output


def test_the_language_lookup_ignores_an_empty_config_variable(home):
    from lobster.i18n import _global_config_dir

    folder = _global_config_dir({"XDG_CONFIG_HOME": "", "APPDATA": ""})

    assert folder.is_absolute()


@pytest.mark.parametrize(
    "folder, name",
    [(".lobster", "vector_store"), (".config/lobster", "projects.json")],
)
def test_a_linked_lobster_entry_is_unlinked_and_its_target_kept(
    home, tmp_path, folder, name
):
    data = tmp_path / "data"
    data.mkdir()
    (data / "results.csv").write_text("a,b\n")
    parent = home / folder
    parent.mkdir(parents=True)
    marker = "lobster_history" if folder == ".lobster" else "providers.json"
    (parent / marker).write_text("x")
    try:
        (parent / name).symlink_to(data, target_is_directory=True)
    except OSError:
        pytest.skip("this system cannot make links")

    result = purge("--force")

    assert result.exit_code == 0, result.output
    assert not os.path.lexists(parent / name)
    assert (data / "results.csv").read_text() == "a,b\n"


def test_a_config_folder_with_only_the_project_list_is_removed(home):
    # lobster init <folder> without a provider of the user's writes only this
    folder = home / ".config" / "lobster"
    folder.mkdir(parents=True)
    (folder / "projects.json").write_text('{"schema_version": 1, "projects": []}')

    result = purge("--force")

    assert result.exit_code == 0, result.output
    assert not folder.exists()


def test_the_project_list_goes_and_the_users_file_stays(home):
    folder = home / ".config" / "lobster"
    folder.mkdir(parents=True)
    (folder / "projects.json").write_text("{}")
    (folder / "notes.txt").write_text("mine")

    result = purge("--force")

    assert result.exit_code == 0, result.output
    assert not (folder / "projects.json").exists()
    assert (folder / "notes.txt").read_text() == "mine"
    assert "notes.txt" in flat(result)
