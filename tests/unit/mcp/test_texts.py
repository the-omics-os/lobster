"""The tool texts teach one way to wait, the setup lines a host may run itself,
and bounds a host keeps by default."""

from __future__ import annotations

import pytest

from lobster.cli_internal import mcp_contract, mcp_overview
from lobster.cli_internal.mcp_stdio import McpServer, ThreadRunner
from lobster.core import projects
from tests.unit.mcp.test_run import call


def texts():
    return {t["name"]: t["description"] for t in mcp_contract.list_tools()}


def every_text():
    parts = [mcp_contract.INSTRUCTIONS, mcp_contract.CONNECTOR_DESCRIPTION]
    for tool in mcp_contract.list_tools():
        parts.append(tool["description"])
        for prop in tool["inputSchema"]["properties"].values():
            parts.append(prop.get("description", ""))
    return " ".join(parts)


@pytest.mark.parametrize(
    "name",
    [
        "lobster_run",
        "lobster_status",
        "lobster_answer",
        "lobster_cancel",
        "lobster_continue",
    ],
)
def test_every_run_text_waits_with_lobster_wait(name):
    text = texts()[name]
    assert "lobster_wait" in text
    assert "poll lobster_status" not in text
    assert "polled" not in text


def test_no_text_teaches_polling_the_status():
    text = every_text()
    assert f"every {mcp_contract.POLL_AFTER_S} seconds" not in text
    assert "Without lobster_wait" not in text
    assert "one-off look" in texts()["lobster_status"]


def test_paths_are_absolute_without_the_project_directory_remark():
    text = every_text()
    assert "does not run in your project directory" not in text
    for name in ("lobster_run", "lobster_continue"):
        assert "absolute path" in texts()[name]


def test_the_wait_default_is_50_and_the_text_asks_for_300():
    assert mcp_contract.WAIT_DEFAULT_S == 50
    assert "Pass timeout_s 300 unless your host ends tool calls sooner" in (
        texts()["lobster_wait"]
    )


def test_a_follow_up_is_as_long_as_a_question():
    assert mcp_contract.INSTRUCTION_MAX_CHARS == mcp_contract.QUESTION_MAX_CHARS
    tool = {t["name"]: t for t in mcp_contract.list_tools()}["lobster_continue"]
    instruction = tool["inputSchema"]["properties"]["instruction"]
    assert instruction["maxLength"] == 4000
    assert "What to do next" in instruction["description"]
    assert "one sentence" not in every_text()


def test_a_moved_session_folder_is_kept_until_removed_or_the_run_deleted():
    assert "kept until the user removes it or deletes the run" in (
        texts()["lobster_run"]
    )
    assert "kept until the user deletes them" not in every_text()


@pytest.fixture
def lines(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr(mcp_overview, "installed_experts", lambda: set())
    monkeypatch.setattr(mcp_overview, "loaded_experts", lambda: None)
    monkeypatch.setattr(mcp_overview, "_lobster_path", lambda: "/opt/l/lobster")


def overview(start):
    return call(McpServer(ThreadRunner(), start=start), "lobster_overview", {})


def test_the_setup_lines_are_for_the_host_to_run(lines, tmp_path):
    made, _ = projects.init_project(tmp_path / "p", name="P")
    empty = tmp_path / "empty"
    empty.mkdir()
    for start in (made.root, empty):
        nxt = overview(start)["next"].lower()
        assert "in a shell" in nxt
        assert "tell the user what you ran" in nxt
        assert "the user runs" not in nxt and "Ask the user to run" not in nxt
    text = texts()["lobster_overview"]
    assert "the user types" not in text
    assert "in a shell" in text and "tell the user what you ran" in text
