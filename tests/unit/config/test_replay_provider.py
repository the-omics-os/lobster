"""Replay model: opt-in only, and loud on any unscripted request."""

import json

import pytest

from lobster.config.providers.replay_provider import (
    REPLAY_ENV,
    ReplayChatModel,
    ReplayMismatchError,
    replay_model_from_env,
    reset_replay_scripts,
)


@pytest.fixture
def script(tmp_path, monkeypatch):
    def write(responses):
        path = tmp_path / "script.json"
        path.write_text(
            json.dumps({"schema": "lobster.replay/1", "responses": responses})
        )
        monkeypatch.setenv(REPLAY_ENV, str(path))
        reset_replay_scripts()
        return path

    yield write
    reset_replay_scripts()


def test_default_configuration_never_resolves_replay(monkeypatch):
    from lobster.config.constants import VALID_PROVIDERS
    from lobster.config.providers import ProviderRegistry

    monkeypatch.delenv(REPLAY_ENV, raising=False)
    assert replay_model_from_env("supervisor") is None
    assert not any("replay" in name for name in VALID_PROVIDERS)
    assert not any("replay" in name for name in ProviderRegistry.get_provider_names())

    from lobster.config.llm_factory import LLMFactory

    try:
        model = LLMFactory.create_llm(
            {}, agent_name="supervisor", provider_override="ollama-cloud"
        )
    except Exception:
        return  # no provider configured in the test environment: not a replay model
    assert not isinstance(model, ReplayChatModel)


def test_serves_scripted_responses_in_order(script):
    script(
        [
            {"agent": "supervisor", "content": "first"},
            {
                "agent": "supervisor",
                "tool_calls": [{"name": "ask_user", "args": {"question": "q"}}],
            },
        ]
    )
    model = replay_model_from_env("supervisor")
    assert model.invoke("hi").content == "first"
    second = model.invoke("hi")
    assert second.tool_calls[0]["name"] == "ask_user"
    assert second.tool_calls[0]["args"] == {"question": "q"}


def test_request_beyond_script_raises(script):
    script([{"content": "only one"}])
    model = replay_model_from_env("supervisor")
    model.invoke("hi")
    with pytest.raises(ReplayMismatchError, match="beyond the 1 scripted"):
        model.invoke("again")


def test_wrong_agent_raises(script):
    script([{"agent": "supervisor", "content": "x"}])
    with pytest.raises(ReplayMismatchError, match="expects 'supervisor'"):
        replay_model_from_env("transcriptomics_expert").invoke("hi")


def test_scripted_error_raises(script):
    script([{"error": "boom"}])
    with pytest.raises(RuntimeError, match="boom"):
        replay_model_from_env().invoke("hi")


def test_replay_announces_itself_once_on_stderr(script, capsys):
    path = script([{"content": "hi"}])
    replay_model_from_env(agent_name="supervisor")
    replay_model_from_env(agent_name="research_agent")
    out, err = capsys.readouterr()
    assert out == ""
    assert err == (
        f"Replay fixture active: answers come from {path}, not from a model.\n"
    )


def test_scripted_usage_is_set_as_usage_metadata(script):
    script(
        [{"content": "priced", "usage": {"input_tokens": 1000, "output_tokens": 100}}]
    )
    message = replay_model_from_env("supervisor").invoke("hi")
    assert message.usage_metadata == {
        "input_tokens": 1000,
        "output_tokens": 100,
        "total_tokens": 1100,
    }


def test_scripted_usage_makes_the_tracked_run_complete(script):
    from lobster.utils.callbacks import TokenTrackingCallback

    prices = {"model": {"input_per_million": 3, "output_per_million": 15}}
    script(
        [{"content": "priced", "usage": {"input_tokens": 1000, "output_tokens": 100}}]
    )
    tracker = TokenTrackingCallback("usage", prices)
    replay_model_from_env("supervisor").invoke("hi", config={"callbacks": [tracker]})
    latest = tracker.get_latest_cost()
    assert latest["usage_complete"] is True
    assert latest["missing_usage_invocations"] == 0
    assert latest["total_tokens"] == 1100


def test_a_script_without_usage_reports_none(script):
    from lobster.utils.callbacks import TokenTrackingCallback

    script([{"content": "plain"}])
    tracker = TokenTrackingCallback("usage", {})
    message = replay_model_from_env("supervisor").invoke(
        "hi", config={"callbacks": [tracker]}
    )
    assert message.usage_metadata is None
    latest = tracker.get_latest_cost()
    assert latest["usage_complete"] is False
    assert latest["cost_complete"] is False
    assert latest["missing_usage_invocations"] == 1
