"""A path the operating system cannot resolve is a one-line tool error, not a
protocol error."""

from __future__ import annotations

import pytest

from lobster.cli_internal.mcp_stdio import McpServer, ThreadRunner
from tests.unit.mcp.test_run import call, data, run_args  # noqa: F401  (fixture)


@pytest.fixture
def server():
    srv = McpServer(ThreadRunner(lambda request, cancel_event: {}))
    yield srv
    srv.runner.stop(5)


def odd_paths(tmp_path):
    loop = tmp_path / "loop"
    loop.symlink_to(loop)
    return {"nul": str(tmp_path / "a") + "\x00", "loop": str(loop)}


@pytest.mark.parametrize("kind", ["nul", "loop"])
def test_lobster_inspect_refuses_an_odd_path(server, tmp_path, kind):
    reply = call(server, "lobster_inspect", {"path": odd_paths(tmp_path)[kind]})
    assert reply["error_text"].startswith("path_not_allowed:")
    assert "\n" not in reply["error_text"]


@pytest.mark.parametrize("kind", ["nul", "loop"])
@pytest.mark.parametrize("field", ["input_path", "output_dir"])
def test_lobster_run_refuses_an_odd_path(server, tmp_path, data, kind, field):
    args = {**run_args(data, tmp_path / "out"), field: odd_paths(tmp_path)[kind]}
    reply = call(server, "lobster_run", args)
    assert "error_text" in reply and "\n" not in reply["error_text"]
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("kind", ["nul", "loop"])
def test_lobster_continue_refuses_an_odd_output_dir(server, tmp_path, kind):
    reply = call(
        server,
        "lobster_continue",
        {"output_dir": odd_paths(tmp_path)[kind], "instruction": "go on"},
    )
    assert reply["error_text"].startswith("path_not_allowed:")
