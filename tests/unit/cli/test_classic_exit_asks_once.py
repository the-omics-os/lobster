"""``/exit`` in a classic ``lobster chat`` asks "exit?" once and says goodbye once."""

import io
from pathlib import Path

import pytest
from rich.console import Console

from lobster.cli_internal.commands.heavy import chat_commands


class _Client:
    session_id = "s-1"
    workspace_path = Path(".")

    def __init__(self):
        self.saved = self.reads = 0

    def get_token_usage(self):
        return None

    def _save_session_json(self):
        self.saved += 1


def _chat(monkeypatch, typed, answers, slash_for=None):
    """Run a classic chat on ``typed`` lines, answering each question from
    ``answers``; input that runs out is Ctrl+C. Returns the questions, the
    goodbyes and the client."""
    typed, answers = iter(typed), iter(answers)
    questions, goodbyes = [], []
    client = _Client()

    def read(prompt, _client=None):
        client.reads += 1
        line = next(typed, None)
        if line is None:
            raise KeyboardInterrupt
        return line

    def ask(question, *args, **kwargs):
        questions.append(question)
        return next(answers)

    for name, value in {
        "setup_logging": lambda *a, **k: None,
        "validate_startup_or_raise_startup_diagnostic": lambda *a, **k: None,
        "display_welcome": lambda *a, **k: None,
        "display_goodbye": lambda *a, **k: goodbyes.append(True),
        "init_client_with_animation": lambda *a, **k: client,
        "_show_workspace_prompt": lambda c: None,
        "get_user_input_with_editing": read,
        "note_run_status": lambda *a, **k: None,
        "_maybe_print_timings": lambda *a, **k: None,
        "check_for_missing_slash_command": lambda text: slash_for,
        "console": Console(file=io.StringIO()),
    }.items():
        monkeypatch.setattr(chat_commands, name, value)
    monkeypatch.setattr(chat_commands.Confirm, "ask", ask)
    chat_commands.chat_impl(
        workspace=Path("."),
        session_id=None,
        reasoning=False,
        verbose=False,
        debug=False,
        profile_timings=None,
        provider=None,
        model=None,
        stream=False,
    )
    return questions, goodbyes, client


@pytest.mark.parametrize("command", ["/exit", "/EXIT"])
def test_exit_asks_once_and_says_goodbye_once(monkeypatch, command):
    questions, goodbyes, client = _chat(monkeypatch, [command], [True])
    assert len(questions) == 1 and "exit?" in questions[0]
    assert goodbyes == [True]
    assert client.saved == 1
    assert client.reads == 1  # the chat ended on /exit


def test_no_to_exit_keeps_the_chat_without_a_goodbye(monkeypatch):
    # "n" to /exit, then Ctrl+C and "y".
    questions, goodbyes, _ = _chat(monkeypatch, ["/exit"], [False, True])
    assert len(questions) == 2
    assert goodbyes == [True]


def test_exit_without_the_slash_asks_did_you_mean_then_exit(monkeypatch):
    questions, goodbyes, _ = _chat(monkeypatch, ["exit"], [True, True], "/exit")
    assert len(questions) == 2
    assert "Did you mean" in questions[0] and "exit?" in questions[1]
    assert goodbyes == [True]
