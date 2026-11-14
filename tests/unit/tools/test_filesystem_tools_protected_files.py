"""The file tools never hand credentials or workspace settings to the model.

The tool root is the folder that holds ``.lobster_workspace``, so keys written there by
``lobster init`` sit inside it. Anything the tools return goes to the model provider.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from lobster.tools.filesystem_tools import create_filesystem_tools

SECRET = "sk-ant-PROTECTED-0001"


@pytest.fixture
def root(tmp_path):
    ws = tmp_path / ".lobster_workspace"
    ws.mkdir()
    (ws / ".env").write_text(f"ANTHROPIC_API_KEY={SECRET}\n")
    (ws / "provider_config.json").write_text(f'{{"key": "{SECRET}"}}')
    (ws / "config.toml").write_text(f'token = "{SECRET}"\n')
    (tmp_path / ".env").write_text(f"OPENAI_API_KEY={SECRET}\n")
    (tmp_path / ".env.local").write_text(f"X={SECRET}\n")
    (tmp_path / "credentials.env").write_text(f"X={SECRET}\n")
    (tmp_path / "server.PEM").write_text(f"-----BEGIN {SECRET}-----\n")
    (tmp_path / "id.key").write_text(f"{SECRET}\n")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "expression.csv").write_text("gene,count\nTP53,7\n")
    (tmp_path / "notes.env.txt").write_text("ordinary notes about the environment\n")
    return tmp_path


@pytest.fixture
def tools(root):
    return {t.name: t for t in create_filesystem_tools(root)}


PROTECTED = [
    ".env",
    ".lobster_workspace/.env",
    ".lobster_workspace/provider_config.json",
    ".lobster_workspace/config.toml",
    ".env.local",
    "credentials.env",
    "server.PEM",
    "id.key",
]


@pytest.mark.parametrize("path", PROTECTED)
def test_read_refuses_protected_files_without_echoing_content(tools, path):
    reply = tools["read_file"].invoke({"path": path})
    assert reply.startswith("Error")
    assert SECRET not in reply


@pytest.mark.parametrize(
    "path",
    [
        ".lobster_workspace/.ENV",
        ".lobster_workspace/Provider_Config.JSON",
        ".Env.Local",
        "CREDENTIALS.env",
        "ID.KEY",
    ],
)
def test_the_check_ignores_case(root, tools, path):
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(f"X={SECRET}\n")
    reply = tools["read_file"].invoke({"path": path})
    assert reply.startswith("Error") and SECRET not in reply


@pytest.mark.parametrize(
    "spelling",
    ["data/../.env", "./.lobster_workspace/./.env", "data/../.lobster_workspace/.env"],
)
def test_a_roundabout_spelling_is_refused(tools, spelling):
    reply = tools["read_file"].invoke({"path": spelling})
    assert reply.startswith("Error") and SECRET not in reply


def test_an_absolute_path_is_refused(root, tools):
    reply = tools["read_file"].invoke(
        {"path": str(root / ".lobster_workspace" / ".env")}
    )
    assert reply.startswith("Error") and SECRET not in reply


def test_a_symlink_to_a_protected_file_is_refused(root, tools):
    (root / "data.txt").symlink_to(root / ".lobster_workspace" / ".env")
    (root / "viewdir").symlink_to(root / ".lobster_workspace")
    for path in ("data.txt", "viewdir/.env", "viewdir/provider_config.json"):
        reply = tools["read_file"].invoke({"path": path})
        assert reply.startswith("Error") and SECRET not in reply


def test_a_refusal_is_the_same_whether_or_not_the_file_exists(root, tools):
    missing = tools["read_file"].invoke({"path": "sub/.env"})
    present = tools["read_file"].invoke({"path": ".env"})
    assert missing.split(":")[0] == present.split(":")[0] == "Error"
    assert "not found" not in missing.lower()


@pytest.mark.parametrize("path", PROTECTED)
def test_write_refuses_protected_files(root, tools, path):
    target = root / path
    before = target.read_text() if target.exists() else None
    reply = tools["write_file"].invoke({"path": path, "content": "overwritten"})
    assert reply.startswith("Error")
    assert (target.read_text() if target.exists() else None) == before


def test_write_refuses_to_create_a_new_protected_file(root, tools):
    reply = tools["write_file"].invoke({"path": "data/.env", "content": "X=1"})
    assert reply.startswith("Error") and not (root / "data" / ".env").exists()


def test_list_omits_protected_files_and_says_so(tools):
    reply = tools["list_files"].invoke({"path": ".lobster_workspace"})
    for name in (".env", "provider_config.json", "config.toml"):
        assert name not in reply
    assert re.search(r"\b3 protected files not shown", reply)

    top = tools["list_files"].invoke({"path": "."})
    assert "data/" in top and "notes.env.txt" in top
    assert ".env" not in top.replace("notes.env.txt", "")
    assert re.search(r"\b\d+ protected files? not shown", top)


def test_glob_omits_protected_files_and_says_so(tools):
    reply = tools["glob_files"].invoke({"pattern": "**/.env"})
    assert SECRET not in reply and ".env  (" not in reply
    assert re.search(r"\b\d+ protected files? not shown", reply)

    everything = tools["glob_files"].invoke({"pattern": "**/*"})
    assert "expression.csv" in everything and "notes.env.txt" in everything
    for name in ("provider_config.json", "credentials.env", "server.PEM", "id.key"):
        assert name not in everything
    assert re.search(r"\b\d+ protected files? not shown", everything)


def test_grep_never_matches_inside_protected_files(tools):
    reply = tools["grep_files"].invoke({"pattern": "sk-ant"})
    assert SECRET not in reply
    assert re.search(r"\b\d+ protected files? not shown", reply)


def test_ordinary_data_files_still_work(tools):
    assert "TP53" in tools["read_file"].invoke({"path": "data/expression.csv"})
    assert "expression.csv" in tools["list_files"].invoke({"path": "data"})
    assert "protected" not in tools["list_files"].invoke({"path": "data"})
    assert "expression.csv" in tools["glob_files"].invoke({"pattern": "data/*.csv"})
    assert "TP53" in tools["grep_files"].invoke({"pattern": "tp53", "path": "data"})
    assert "ordinary notes" in tools["read_file"].invoke({"path": "notes.env.txt"})
    wrote = tools["write_file"].invoke({"path": "data/new.csv", "content": "a,b\n"})
    assert wrote.startswith("Wrote")


def test_folding_spellings_of_a_protected_name_are_refused(root, tools):
    # Case-insensitive disks fold 'ſ' (long s) to 's', so these reach the real files.
    (root / ".lobster_workspace" / "providers.json").write_text(f"{SECRET}")
    (root / "credentials.json").write_text(f"{SECRET}")
    for path in (
        ".lobster_workspace/providerſ.json",
        "credentialſ.json",
        ".lobſter_workspace/providers.json",
    ):
        reply = tools["read_file"].invoke({"path": path})
        assert SECRET not in reply
        assert reply.startswith("Error")


@pytest.mark.parametrize("tool", ["list_files", "glob_files", "grep_files"])
def test_pointing_search_tools_at_a_protected_file_tells_nothing(root, tools, tool):
    def call(path):
        args = {"path": path, "pattern": "sk-ant"}
        return tools[tool].invoke(
            {k: v for k, v in args.items() if k in tools[tool].args}
        )

    present, absent = call(".env"), call("sub/.env")
    assert present.startswith("Error") and "not found" not in present
    assert present.replace(".env", "X") == absent.replace("sub/.env", "X")


def test_folders_named_like_secrets_stay_visible_and_their_files_follow_the_rules(
    root, tools
):
    (root / "credentials").mkdir()
    (root / "credentials" / "notes.txt").write_text("plain notes")
    listing = tools["list_files"].invoke({"path": "."})
    assert "credentials/" in listing
    assert "plain notes" in tools["read_file"].invoke({"path": "credentials/notes.txt"})
    (root / "credentials" / ".env").write_text(SECRET)
    assert SECRET not in tools["read_file"].invoke({"path": "credentials/.env"})


@pytest.mark.parametrize("name", [".envrc", ".netrc", ".npmrc", ".pypirc", ".NetRC"])
def test_tool_login_files_are_protected(root, tools, name):
    (root / name).write_text(f"password {SECRET}\n")
    assert SECRET not in tools["read_file"].invoke({"path": name})
    assert tools["read_file"].invoke({"path": name}).startswith("Error")
    assert (
        tools["write_file"].invoke({"path": name, "content": "x"}).startswith("Error")
    )
    assert (root / name).read_text() == f"password {SECRET}\n"
    assert name.lower() not in tools["glob_files"].invoke({"pattern": "*"}).lower()
