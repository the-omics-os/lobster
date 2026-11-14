"""``response_style`` shapes the supervisor's reply rules; "chat" is unchanged."""

import hashlib

import pytest

from lobster.agents import supervisor as sup
from lobster.config.supervisor_config import SupervisorConfig

# sha256 of the text built before response_style existed (default config).
CHAT_RESPONSE_BEHAVIOR = {
    True: "2a7ae24bafba16338db944c93c2885a128a626ec6edbde80fbdc02374dcffa8c",
    False: "0b5ed462b57e1349295088e48b23bb08d9004515e3a39d7b39f2366232badfb9",
}
CHAT_ORCHESTRATION = "1decc202c92b04dbfc99d810b338102b2886da676466d333fa6ab557fc7c730f"


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


@pytest.fixture(autouse=True)
def no_env(monkeypatch):
    monkeypatch.delenv("LOBSTER_RESPONSE_STYLE", raising=False)


@pytest.mark.parametrize("interactive", [True, False])
def test_chat_text_is_byte_identical_to_before(interactive):
    config = SupervisorConfig()
    assert config.response_style == "chat"
    assert (
        digest(sup._build_response_behavior(config, interactive=interactive))
        == CHAT_RESPONSE_BEHAVIOR[interactive]
    )
    assert digest(sup._build_orchestration_principles("chat")) == CHAT_ORCHESTRATION
    assert digest(sup._build_orchestration_principles()) == CHAT_ORCHESTRATION


@pytest.mark.parametrize("interactive", [True, False])
def test_run_has_no_opener_no_offer_and_next_steps_as_statements(interactive):
    config = SupervisorConfig(response_style="run")
    behavior = sup._build_response_behavior(config, interactive=interactive)
    assert "no greeting, no opener" in behavior
    assert "Do not end with an offer or a question" in behavior
    assert "Possible next steps:" in behavior
    assert "only through the ask_user tool" in behavior
    # the chat rule that invites a closing offer is gone, from both places
    assert "Suggest logical next steps" not in behavior
    assert "Suggest logical next steps" not in sup._build_orchestration_principles(
        "run"
    )
    assert "Possible next steps" in sup._build_orchestration_principles("run")


def test_run_does_not_depend_on_interactive():
    config = SupervisorConfig(response_style="run")
    a = sup._build_response_behavior(config, interactive=True)
    b = sup._build_response_behavior(config, interactive=False)
    # the three style rules are the same lines in both modes
    for marker in (
        "RESPONSE STYLE (run)",
        "Do not end with",
        "only through the ask_user",
    ):
        line_a = [x for x in a.splitlines() if marker in x]
        line_b = [x for x in b.splitlines() if marker in x]
        assert line_a and line_a == line_b, marker


def test_run_rules_survive_auto_suggest_being_off_or_on():
    for auto in (True, False):
        config = SupervisorConfig(response_style="run", auto_suggest_next_steps=auto)
        assert "Suggest logical next steps" not in sup._build_response_behavior(config)


def test_the_environment_is_only_the_default_source(monkeypatch):
    monkeypatch.setenv("LOBSTER_RESPONSE_STYLE", "run")
    assert SupervisorConfig.from_env().response_style == "run"
    assert SupervisorConfig().response_style == "chat"  # a built config ignores env
    assert SupervisorConfig(response_style="chat").response_style == "chat"
    monkeypatch.setenv("LOBSTER_RESPONSE_STYLE", " RUN ")
    assert SupervisorConfig.from_env().response_style == "run"
    monkeypatch.setenv("LOBSTER_RESPONSE_STYLE", "")
    assert SupervisorConfig.from_env().response_style == "chat"


def test_an_unknown_value_is_a_clear_error_not_a_fallback(monkeypatch):
    with pytest.raises(ValueError, match="response_style.*chat, run"):
        SupervisorConfig(response_style="terse")
    monkeypatch.setenv("LOBSTER_RESPONSE_STYLE", "terse")
    with pytest.raises(ValueError, match="LOBSTER_RESPONSE_STYLE.*chat, run"):
        SupervisorConfig.from_env()


def test_the_full_prompt_follows_the_config():
    class _Dm:
        def list_modalities(self):
            return []

    prompt_run = sup.create_supervisor_prompt(
        _Dm(), SupervisorConfig(response_style="run"), active_agents=[]
    )
    prompt_chat = sup.create_supervisor_prompt(
        _Dm(), SupervisorConfig(), active_agents=[]
    )
    assert "RESPONSE STYLE (run)" in prompt_run
    assert "RESPONSE STYLE" not in prompt_chat
    assert "Suggest logical next steps" in prompt_chat
    assert "Suggest logical next steps" not in prompt_run


def test_to_dict_reports_the_style():
    assert SupervisorConfig(response_style="run").to_dict()["response_style"] == "run"
