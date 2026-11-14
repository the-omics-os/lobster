"""The lines shown to a user or host that has no provider setup yet."""

import subprocess
import sys

from lobster.cli_internal import provider_hint


def test_a_provider_is_listed_when_all_its_key_variables_are_set():
    env = {
        "OPENAI_API_KEY": "sk-x",
        "AWS_BEDROCK_ACCESS_KEY": "a",
        "NEBIUS_API_KEY": "",
    }
    assert provider_hint.providers_with_keys(env) == ["openai"]
    assert provider_hint.suggested_lines("/w/study", env, command="/x/lobster") == [
        "/x/lobster init /w/study --provider openai"
    ]
    assert provider_hint.suggested_lines(
        "/w/study", env, command="/x/lobster", extra=["--model", "m 1"]
    ) == ["/x/lobster init /w/study --provider openai --model 'm 1'"]


def test_without_any_key_the_terminal_form_is_given():
    lines = provider_hint.suggested_lines("/w/study", {}, command="/x/lobster")
    assert lines[0] == "In a terminal: /x/lobster init /w/study"
    assert lines[1] == "Options: /x/lobster init --help"


def test_the_default_command_is_the_running_executable(monkeypatch):
    monkeypatch.setattr(provider_hint, "lobster_path", lambda: "/opt/a b/lobster")
    line = provider_hint.command_line("init", provider_hint.Placeholder("<folder>"))
    assert line == "'/opt/a b/lobster' init <folder>"


def test_every_caller_value_is_quoted_even_when_it_looks_like_a_placeholder():
    line = provider_hint.command_line(
        "init", "d", "--model", "<a;touch X", command="/x/lobster"
    )
    assert line == "/x/lobster init d --model '<a;touch X'"
    assert provider_hint.quote("<folder>") != "<folder>"


def test_the_message_starts_with_the_code_and_never_holds_a_value():
    env = {"OLLAMA_API_KEY": "secret-value-77"}
    text = provider_hint.no_setup_message("/w/study", env)
    assert text.startswith(f"{provider_hint.NO_SETUP_CODE}:")
    assert "OLLAMA_API_KEY is set in the environment for ollama-cloud" in text
    assert "secret-value-77" not in text


def test_the_helper_pulls_in_no_engine():
    code = (
        "import sys; from lobster.cli_internal import provider_hint as p; "
        "p.no_setup_message('/w', {}); "
        "bad=[m for m in ('langchain','langchain_core','langgraph','anndata','scanpy') if m in sys.modules]; "
        "print(bad)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout
    assert out.strip() == "[]"
