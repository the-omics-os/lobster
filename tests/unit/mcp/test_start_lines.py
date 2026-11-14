"""With no project, the overview and a lobster_run without output_dir give the
init lines of the provider hint (absolute lobster path), for the host to run in
a shell; the bare line, run without a terminal on a global setup, makes the
project the next overview shows."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys

import pytest

from lobster.cli_internal import mcp_overview, provider_hint
from lobster.cli_internal.mcp_stdio import McpServer, ThreadRunner
from lobster.config import global_config
from tests.unit.mcp.test_run import call

# conftest's autouse provider fixture would set LOBSTER_LLM_PROVIDER=anthropic
pytestmark = pytest.mark.no_auto_config

EXE = "/opt/lobster/bin/lobster"
KEY = "OLLAMA_API_KEY"


@pytest.fixture
def isolated(monkeypatch, tmp_path):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(global_config, "CONFIG_DIR", home / ".config" / "lobster")
    for name in [KEY, "LOBSTER_LLM_PROVIDER", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(mcp_overview, "installed_experts", lambda: set())
    monkeypatch.setattr(mcp_overview, "loaded_experts", lambda: None)
    monkeypatch.setattr(mcp_overview, "_lobster_path", lambda: EXE)
    empty = tmp_path / "empty"
    empty.mkdir()
    return empty


def overview(start):
    return call(McpServer(ThreadRunner(), start=start), "lobster_overview", {})


def test_a_key_in_the_environment_gives_its_provider_line(monkeypatch, isolated):
    monkeypatch.setenv(KEY, "sk-not-shown-0123")
    seen = overview(isolated)
    line = f"{EXE} init {isolated} --provider ollama-cloud"
    assert seen["start_lines"] == [line]
    assert (
        line in seen["next"] and "run one of the start_lines in a shell" in seen["next"]
    )
    for flag in ("--model", "--agents", "--preset"):
        assert flag in seen["next"]
    assert "sk-not-shown" not in str(seen)


def test_without_a_key_the_bare_line_is_given(isolated):
    seen = overview(isolated)
    assert seen["start_lines"][0] == f"In a terminal: {EXE} init {isolated}"
    assert f"run this line in a shell: {EXE} init {isolated}" in seen["next"]
    assert provider_hint.NO_SETUP_CODE in seen["next"]
    assert "tell the user what you ran" in seen["next"].lower()


def test_the_run_refusal_without_a_project_gives_the_same_lines(
    monkeypatch, isolated, tmp_path
):
    monkeypatch.setenv(KEY, "sk-not-shown-0123")
    refused = call(
        McpServer(ThreadRunner(), start=isolated), "lobster_run", {"question": "q"}
    )["error_text"]
    assert refused.startswith("bad_argument: output_dir is needed.")
    assert f"{EXE} init {isolated} --provider ollama-cloud" in refused


def test_the_printed_line_run_without_a_terminal_makes_the_project(
    monkeypatch, isolated, tmp_path
):
    global_config.GlobalProviderConfig(
        default_provider="ollama", ollama_default_model="llama3"
    ).save()
    line = overview(isolated)["start_lines"][0].removeprefix("In a terminal: ")
    argv = shlex.split(line)
    assert argv[0] == EXE
    env = {**os.environ, "HOME": str(tmp_path / "home")}
    done = subprocess.run(
        [sys.executable, "-m", "lobster", *argv[1:]],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    assert done.returncode == 0, done.stdout[-1500:] + done.stderr[-1500:]
    seen = overview(isolated)
    assert seen["project"]["folder"] == str(isolated)
    assert (seen["provider"], seen["provider_source"]) == ("ollama", "global file")
    assert seen["experts"]["source"] == "every installed expert"


def test_with_keys_next_points_at_the_start_lines_and_names_the_variables(
    monkeypatch, isolated
):
    monkeypatch.setenv(KEY, "sk-not-shown-0123")
    seen = overview(isolated)
    assert "run one of the start_lines in a shell" in seen["next"]
    assert f"{KEY} is set" in seen["next"]
    assert "terminal" not in seen["next"]
    assert "sk-not-shown" not in str(seen)


def test_a_global_setup_without_keys_gives_the_bare_line_and_no_terminal(isolated):
    global_config.GlobalProviderConfig(default_provider="ollama").save()
    seen = overview(isolated)
    assert f"run this line in a shell: {EXE} init {isolated}" in seen["next"]
    assert "terminal" not in seen["next"]
    assert provider_hint.NO_SETUP_CODE not in seen["next"]


def test_only_without_any_setup_the_user_needs_a_terminal(isolated):
    seen = overview(isolated)
    assert "terminal" in seen["next"]
    refused = call(
        McpServer(ThreadRunner(), start=isolated), "lobster_run", {"question": "q"}
    )["error_text"]
    assert "terminal" in refused
