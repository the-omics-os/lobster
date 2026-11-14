"""A client without a provider/model override gets the configured model's context window,
and the supervisor's trim never drops the request of the current turn."""

import logging

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from lobster.agents.context_management import (
    create_supervisor_pre_model_hook,
    resolve_context_window,
)
from lobster.config import global_config
from lobster.config.providers import get_provider
from lobster.config.workspace_config import WorkspaceProviderConfig
from lobster.core.config_resolver import ConfigResolver

# The suite-wide autouse fixture replaces ConfigResolver with a mock; these tests need the real one.
pytestmark = pytest.mark.no_auto_config


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """An empty workspace, no user-level config, and no cached resolver."""
    monkeypatch.setattr(global_config, "CONFIG_DIR", tmp_path / "xdg" / "lobster")
    monkeypatch.delenv("LOBSTER_LLM_PROVIDER", raising=False)
    ConfigResolver.reset_instance()
    ws = tmp_path / "workspace"
    ws.mkdir()
    yield ws
    ConfigResolver.reset_instance()


def default_window(provider_name):
    provider = get_provider(provider_name)
    return provider.get_model_info(provider.get_default_model()).context_window


@pytest.mark.parametrize(
    "provider_name", ["ollama-cloud", "bedrock", "anthropic", "openai"]
)
def test_no_override_returns_the_window_of_the_provider_default_model(
    workspace, monkeypatch, provider_name
):
    monkeypatch.setenv("LOBSTER_LLM_PROVIDER", provider_name)
    assert resolve_context_window(None, None, workspace) == default_window(
        provider_name
    )


def test_ollama_cloud_default_model_is_not_the_catalog_fallback(workspace, monkeypatch):
    monkeypatch.setenv("LOBSTER_LLM_PROVIDER", "ollama-cloud")
    assert resolve_context_window(None, None, workspace) >= 1_000_000


def test_a_per_agent_model_in_the_workspace_decides_the_window(workspace, monkeypatch):
    monkeypatch.setenv("LOBSTER_LLM_PROVIDER", "ollama-cloud")
    WorkspaceProviderConfig(
        global_provider="ollama-cloud", per_agent_models={"supervisor": "gpt-oss:20b"}
    ).save(workspace)
    assert resolve_context_window(None, None, workspace) == 131_072


def test_an_unknown_model_gets_the_provider_default_window_as_before(
    workspace, monkeypatch
):
    monkeypatch.setenv("LOBSTER_LLM_PROVIDER", "ollama-cloud")
    assert resolve_context_window(None, "no-such-model", workspace) == 8192


def test_explicit_overrides_are_unchanged(workspace):
    assert resolve_context_window("ollama-cloud", "gpt-oss:120b", workspace) == 131_072


def test_a_string_workspace_path_resolves_like_a_path(workspace, monkeypatch, caplog):
    monkeypatch.setenv("LOBSTER_LLM_PROVIDER", "ollama-cloud")
    with caplog.at_level(logging.WARNING):
        window = resolve_context_window(None, None, str(workspace))
    assert window == resolve_context_window(None, None, workspace) >= 1_000_000
    assert "Could not resolve provider" not in caplog.text


# --- the supervisor's trim keeps the request of the current turn -----------------------


def history(n_steps, step_chars=2000, request="RUN THE QC RECIPE"):
    messages = [SystemMessage(content="system"), HumanMessage(content=request)]
    for i in range(n_steps):
        messages.append(
            AIMessage(
                content="",
                tool_calls=[{"id": f"call_{i}", "name": "t", "args": {}}],
            )
        )
        messages.append(
            ToolMessage(
                content=f"result {i} " + "x" * step_chars, tool_call_id=f"call_{i}"
            )
        )
    return messages


def run_hook(messages, budget):
    hook = create_supervisor_pre_model_hook(max_tokens=budget)
    return hook({"messages": messages})["llm_input_messages"]


def test_a_long_tool_history_in_a_small_window_keeps_the_request():
    messages = history(12)
    kept = run_hook(messages, budget=4096)
    assert len(kept) < len(messages)  # something was cut
    humans = [m for m in kept if isinstance(m, HumanMessage)]
    assert [m.content for m in humans] == ["RUN THE QC RECIPE"]
    assert isinstance(kept[0], SystemMessage) and kept[1] is humans[0]


def test_the_newest_messages_survive_and_the_cut_is_from_the_middle():
    messages = history(12)
    kept = run_hook(messages, budget=4096)
    assert "result 11" in kept[-1].content
    body = kept[2:]  # between the request and the end: a contiguous newest run
    assert not isinstance(body[0], ToolMessage)  # no orphaned tool result
    indices = [int(m.content.split()[1]) for m in body if isinstance(m, ToolMessage)]
    assert indices == list(range(indices[0], 12))
    assert indices[0] > 0  # the oldest steps are what was cut


def test_an_earlier_turn_may_go_but_the_current_request_stays():
    messages = history(1, request="first request")
    messages += [AIMessage(content="answer " + "y" * 6000), HumanMessage("second")]
    messages += history(12)[2:]
    kept = run_hook(messages, budget=4096)
    humans = [m.content for m in kept if isinstance(m, HumanMessage)]
    assert humans[-1] == "second" and "first request" not in humans


def test_a_request_larger_than_the_budget_is_kept_whole_and_a_warning_is_logged(
    caplog,
):
    big = "word " * 5000
    messages = history(3, request=big)
    with caplog.at_level(logging.WARNING):
        kept = run_hook(messages, budget=1024)
    assert [m.content for m in kept if isinstance(m, HumanMessage)] == [big]
    assert "exceeds the context budget" in caplog.text


def test_nothing_changes_when_the_history_fits():
    messages = history(2, step_chars=50)
    kept = run_hook(messages, budget=100_000)
    assert len(kept) == len(messages)
    assert [m.content for m in kept[:2]] == ["system", "RUN THE QC RECIPE"]
