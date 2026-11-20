"""Each way a command leaves ends its catalog run as what happened.

completed only on success; errored on an error exit (with the exit code or the
error class); aborted on Ctrl-C, a cancel or SIGTERM. A run nobody ended stays
open (see the ThreadedCatalog tests): it never reads as completed.
"""

import signal
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import typer
from rich.console import Console

from lobster.core.catalog import factory

pytestmark = pytest.mark.unit


class _Catalog:
    """The catalog's ``set_end_status`` rules, recording what each run would end as."""

    def __init__(self):
        self.status = self.reason = None
        self.turn_open = False

    def set_end_status(
        self,
        status,
        session_id=None,
        reason=None,
        *,
        if_unset=False,
        if_unfinished=False,
    ):
        if status is None:  # a turn started
            self.status = self.reason = None
            self.turn_open = True
            return
        if if_unset and (self.status is not None or self.turn_open):
            return
        if if_unfinished and self.status is not None:
            return
        self.status, self.reason, self.turn_open = status, reason, False

    @property
    def end(self):
        return (self.status, self.reason)


class _Client:
    """A client as the commands see it (weak-referenceable, as the run API needs)."""

    def __init__(self, session_id):
        self.session_id = session_id
        self.catalog = _Catalog()
        self.data_manager = SimpleNamespace(catalog=self.catalog)
        self.token_tracker = None
        self.callbacks = []


def _client(session_id="s1"):
    return _Client(session_id)


# -- the shared helpers ---------------------------------------------------------


@pytest.mark.parametrize(
    "raised, end",
    [
        (None, ("completed", None)),
        (typer.Exit(0), ("completed", None)),
        (SystemExit(0), ("completed", None)),
        (typer.Exit(1), ("errored", "exit code 1")),
        (SystemExit(2), ("errored", "exit code 2")),
        (ValueError("bad"), ("errored", "ValueError")),
        (KeyboardInterrupt(), ("aborted", "Ctrl-C")),
        (SystemExit(143), ("aborted", "SIGTERM")),
        (typer.Exit(130), ("aborted", "Ctrl-C")),
        (__import__("click").exceptions.Abort(), ("aborted", "Ctrl-C")),
    ],
)
def test_a_block_ends_the_run_as_it_left(raised, end):
    client = _client()
    with pytest.raises(BaseException) if raised is not None else _nothing():
        with factory.recorded_run_status(client):
            if raised is not None:
                raise raised
    assert client.catalog.end == end


class _nothing:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_a_status_noted_in_the_block_wins_over_the_exit_code():
    client = _client()
    with pytest.raises(typer.Exit):
        with factory.recorded_run_status(client):
            factory.note_run_status(client, "aborted", reason="Ctrl-C")
            raise typer.Exit(1)
    assert client.catalog.end == ("aborted", "Ctrl-C")


def test_a_chat_that_ends_normally_keeps_its_last_turn():
    client = _client()
    with factory.recorded_run_status(client, session=True):
        factory.note_run_status(client, "errored", reason="error")
    assert client.catalog.end == ("errored", "error")
    fresh = _client()
    with factory.recorded_run_status(fresh, session=True):
        pass
    assert fresh.catalog.end == ("completed", None)


def test_noting_never_raises():
    class Broken:
        def set_end_status(self, *args, **kwargs):
            raise RuntimeError("boom")

    factory.note_run_status(SimpleNamespace(catalog=Broken()), "completed")
    factory.note_run_status(object(), "completed")  # no catalog at all


@pytest.fixture
def exit_later(monkeypatch):
    """The handler's hard-exit timer, recorded instead of armed (it would end pytest)."""
    armed = []
    monkeypatch.setattr(factory, "_exit_later", armed.append)
    return armed


def test_sigterm_aborts_the_open_run_and_exits(monkeypatch, exit_later):
    catalog = _Catalog()
    monkeypatch.setattr(factory, "_process_catalog", (Path("."), catalog))
    with pytest.raises(SystemExit) as stopped:
        factory._on_sigterm(signal.SIGTERM, None)
    assert stopped.value.code == 143
    assert catalog.end == ("aborted", "SIGTERM")
    assert exit_later == [143]


def test_sigterm_aborts_a_running_turn(monkeypatch, exit_later):
    catalog = _Catalog()
    catalog.set_end_status("completed")
    catalog.set_end_status(None)  # the next turn is running
    monkeypatch.setattr(factory, "_process_catalog", (Path("."), catalog))
    with pytest.raises(SystemExit):
        factory._on_sigterm(signal.SIGTERM, None)
    assert catalog.end == ("aborted", "SIGTERM")


def test_sigterm_between_a_sessions_turns_keeps_the_last_outcome(
    monkeypatch, exit_later
):
    catalog = _Catalog()
    catalog.set_end_status("completed")
    monkeypatch.setattr(factory, "_process_catalog", (Path("."), catalog))
    with pytest.raises(SystemExit):
        factory._on_sigterm(signal.SIGTERM, None)
    assert catalog.end == ("completed", None)


SIGTERM_CHILD = textwrap.dedent("""
    import os, signal, sys, time
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path
    from lobster.core.catalog import factory, thread
    thread.AT_EXIT_TIMEOUT_S = 1.0
    class Catalog:
        def set_end_status(self, *args, **kwargs):
            pass
    if sys.argv[1] == "catalog":
        factory._process_catalog = (Path("."), Catalog())
    factory.abort_runs_on_sigterm()
    pool = ThreadPoolExecutor(1)
    pool.submit(time.sleep, 30)  # a tool still running: a non-daemon thread
    print("ready", flush=True)
    time.sleep(60)
""")


SIGTERM_QUERY_CHILD = textwrap.dedent("""
    import atexit, time
    from pathlib import Path
    from types import SimpleNamespace
    from langchain_core.language_models.fake_chat_models import FakeListChatModel
    from lobster.cli_internal.commands.heavy import query_commands
    from lobster.core.catalog import factory, thread
    from lobster.utils.callbacks import TerminalCallbackHandler
    thread.AT_EXIT_TIMEOUT_S = 1.0
    class Catalog:
        status = reason = None
        def set_end_status(self, status, session_id=None, reason=None, **only):
            if self.status is None:
                self.status, self.reason = status, reason
    catalog = Catalog()
    atexit.register(lambda: print("RUN", catalog.status, catalog.reason, flush=True))
    class Hung(FakeListChatModel):
        def _generate(self, *args, **kwargs):
            print("ready", flush=True)
            time.sleep(30)  # a model call that never answers
    class Client:
        callbacks = [TerminalCallbackHandler()]
        data_manager = SimpleNamespace(catalog=catalog)
        def query(self, question, stream=False):
            config = {"callbacks": self.callbacks}
            answer = Hung(responses=["x"]).invoke(question, config=config)
            return {"success": True, "response": answer.content}
    def init_client(*args, **kwargs):
        factory._process_catalog = (Path("."), catalog)
        factory.abort_runs_on_sigterm()  # as a client with a catalog run does
        return Client()
    query_commands.init_client = init_client
    query_commands.validate_startup_or_raise_startup_diagnostic = lambda **kw: None
    from lobster.cli import app
    app(["query", "--no-stream", "hello"])
""")


def test_sigterm_during_a_query_prints_no_error_and_aborts_the_run(tmp_path):
    """A SIGTERM is not an agent error: no error panel, exit 143, run aborted."""
    pytest.importorskip("langchain_core")
    child = subprocess.Popen(
        [sys.executable, "-c", SIGTERM_QUERY_CHILD],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=tmp_path,
        env={"PATH": "", "PYTHONPATH": ":".join(sys.path), "HOME": str(tmp_path)},
    )
    try:
        assert child.stdout.readline().strip() == "ready"
        child.send_signal(signal.SIGTERM)
        out, _ = child.communicate(timeout=20)
    finally:
        child.kill()
    assert child.returncode == 143, out
    assert "RUN aborted SIGTERM" in out
    assert "Agent Error" not in out and "Report this issue" not in out, out
    assert "143" not in out.replace("RUN aborted SIGTERM", "")


@pytest.mark.parametrize("bound", ["catalog", "none"])
def test_sigterm_ends_the_process_within_the_exit_budget(tmp_path, bound):
    """A tool still running must not hold the process for its whole length."""
    child = subprocess.Popen(
        [sys.executable, "-c", SIGTERM_CHILD, bound],
        stdout=subprocess.PIPE,
        text=True,
        env={"PATH": "", "PYTHONPATH": ":".join(sys.path), "HOME": str(tmp_path)},
    )
    try:
        assert child.stdout.readline().strip() == "ready"
        sent = time.monotonic()
        child.send_signal(signal.SIGTERM)
        code = child.wait(timeout=20)
        took = time.monotonic() - sent
    finally:
        child.kill()
    if bound == "catalog":
        assert code == 143 and took < 1.0 + 2.5
    else:  # no catalog: SIGTERM does what it did before, at once
        assert code == -signal.SIGTERM and took < 2.0


def test_a_run_installs_the_sigterm_handler_only_where_none_is_set(monkeypatch):
    installed = []
    monkeypatch.setattr(signal, "getsignal", lambda sig: signal.SIG_DFL)
    monkeypatch.setattr(signal, "signal", lambda sig, fn: installed.append((sig, fn)))
    factory.abort_runs_on_sigterm()
    assert installed == [(signal.SIGTERM, factory._on_sigterm)]

    installed.clear()
    monkeypatch.setattr(signal, "getsignal", lambda sig: print)  # someone else's
    factory.abort_runs_on_sigterm()
    assert installed == []

    monkeypatch.setattr(signal, "getsignal", lambda sig: signal.SIG_DFL)
    worker = threading.Thread(target=factory.abort_runs_on_sigterm)
    worker.start()
    worker.join()
    assert installed == []  # only the main thread may set a handler


def test_beginning_a_run_installs_the_sigterm_handler(monkeypatch, tmp_path):
    calls = []
    catalog = SimpleNamespace(begin_run=lambda session: None)
    monkeypatch.setattr(factory, "get_catalog", lambda path: catalog)
    monkeypatch.setattr(factory, "abort_runs_on_sigterm", lambda: calls.append(1))
    factory.begin_session_run(SimpleNamespace(), tmp_path, "s1")
    assert calls == [1]


def test_the_mcp_server_sets_the_sigterm_handler_on_its_main_thread(monkeypatch):
    import io

    from lobster.cli_internal import mcp_stdio

    calls = []
    monkeypatch.setattr(factory, "abort_runs_on_sigterm", lambda: calls.append(1))
    mcp_stdio.run_mcp_stdio(stdin=io.StringIO(""), stdout=io.StringIO())
    assert calls == [1]


# -- lobster query ------------------------------------------------------------


def _query(monkeypatch, tmp_path, client, *, json_output=False, stream=False):
    from lobster.cli_internal.commands.heavy import query_commands

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        query_commands,
        "validate_startup_or_raise_startup_diagnostic",
        lambda *args, **kwargs: tmp_path,
    )
    monkeypatch.setattr(query_commands, "console", Console(record=True, width=120))
    monkeypatch.setattr(query_commands, "_maybe_print_timings", lambda *a, **k: None)
    monkeypatch.setattr(query_commands, "should_show_progress", lambda client: False)
    monkeypatch.setattr(
        query_commands, "_render_human_query_result", lambda *a, **k: None
    )
    monkeypatch.setattr(query_commands, "init_client", lambda *a, **k: client)
    query_commands.query_impl(
        "hello",
        workspace=None,
        session_id=None,
        reasoning=False,
        verbose=False,
        debug=False,
        output=None,
        profile_timings=None,
        provider=None,
        model=None,
        stream=stream,
        json_output=json_output,
    )


def _answering(result):
    client = _client()

    def query(question, **kwargs):
        if isinstance(result, BaseException):
            raise result
        return result

    client.query = query
    return client


def test_a_query_that_answers_is_completed(monkeypatch, tmp_path):
    client = _answering({"success": True, "response": "hi"})
    _query(monkeypatch, tmp_path, client)
    assert client.catalog.end == ("completed", None)


@pytest.mark.parametrize("json_output", [False, True])
def test_a_failed_query_is_errored_with_its_exit_code(
    monkeypatch, tmp_path, json_output
):
    client = _answering({"success": False, "response": "", "error": "no"})
    with pytest.raises(typer.Exit):
        _query(monkeypatch, tmp_path, client, json_output=json_output)
    assert client.catalog.end == ("errored", "exit code 1")


def test_a_query_that_raises_is_errored_with_the_error_class(monkeypatch, tmp_path):
    client = _answering(ConnectionError("down"))
    with pytest.raises(ConnectionError):
        _query(monkeypatch, tmp_path, client)
    assert client.catalog.end == ("errored", "ConnectionError")


def test_a_query_stopped_with_ctrl_c_is_aborted(monkeypatch, tmp_path):
    client = _answering(KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        _query(monkeypatch, tmp_path, client)
    assert client.catalog.end == ("aborted", "Ctrl-C")


def test_a_streamed_query_stopped_with_ctrl_c_is_aborted(monkeypatch, tmp_path):
    from lobster.cli_internal.commands.heavy import query_commands

    client = _client()
    monkeypatch.setattr(
        query_commands,
        "_display_streaming_response",
        lambda *a: {"success": False, "error": "Interrupted", "interrupted": True},
    )
    with pytest.raises(typer.Exit):
        _query(monkeypatch, tmp_path, client, stream=True)
    assert client.catalog.end == ("aborted", "Ctrl-C")


# -- lobster command ------------------------------------------------------------


def _command(monkeypatch, tmp_path, dispatch):
    from lobster.cli_internal.commands.heavy import slash_commands

    client = _client()
    monkeypatch.setattr(slash_commands.console, "_file", slash_commands.console._file)
    monkeypatch.setattr(
        "lobster.core.workspace.resolve_workspace",
        lambda explicit_path=None, create=True: tmp_path,
    )
    monkeypatch.setattr(
        "lobster.cli_internal.commands.heavy.session_infra.CommandClient",
        lambda workspace_path, session_id=None: client,
    )
    monkeypatch.setattr(slash_commands, "_dispatch_command", dispatch)
    try:
        slash_commands.command_cmd_impl(
            cmd="status", workspace=tmp_path, session_id=None, json_output=True
        )
    except typer.Exit:
        pass
    return client.catalog.end


def test_a_command_that_runs_is_completed(monkeypatch, tmp_path):
    assert _command(monkeypatch, tmp_path, lambda *a: None) == ("completed", None)


def test_an_unknown_command_is_errored(monkeypatch, tmp_path):
    from lobster.cli_internal.commands.heavy import slash_commands

    end = _command(monkeypatch, tmp_path, lambda *a: slash_commands._UNKNOWN_COMMAND)
    assert end == ("errored", "exit code 1")


def test_a_command_that_fails_is_errored_with_the_error_class(monkeypatch, tmp_path):
    def fail(*args):
        raise OSError("disk")

    assert _command(monkeypatch, tmp_path, fail) == ("errored", "OSError")


# -- MCP turns ------------------------------------------------------------------


def _mcp_turn(monkeypatch, tmp_path, outcome, earlier=None):
    from lobster.cli_internal import mcp_stdio

    client = _client()
    client.messages = None
    client.catalog.set_end_status(earlier)

    def follow(*args):
        if isinstance(outcome, BaseException):
            raise outcome
        return dict(outcome)

    monkeypatch.setattr(mcp_stdio, "_follow_turn", follow)
    monkeypatch.setattr(mcp_stdio, "_record_turn_end", lambda *a: None)
    monkeypatch.setattr(mcp_stdio, "_forget_cancelled_turn", lambda *a: None)
    request = mcp_stdio.RunRequest(run_id="r1", input_path=None, output_dir=".")
    try:
        mcp_stdio._run_turn(client, None, "q", request, threading.Event(), tmp_path)
    except BaseException:
        pass
    return client.catalog.end


@pytest.mark.parametrize(
    "outcome, end",
    [
        ({"status": "completed", "summary": "ok"}, ("completed", None)),
        (
            {"status": "failed", "summary": "x", "error": {"code": "model_error"}},
            ("errored", "model_error"),
        ),
        (
            {"status": "cancelled", "summary": "x", "cancel_reason": "cancel"},
            ("aborted", "cancel"),
        ),
        (RuntimeError("boom"), ("errored", "RuntimeError")),
        (KeyboardInterrupt(), ("aborted", "KeyboardInterrupt")),
    ],
)
def test_an_mcp_turn_ends_the_session_run_as_it_ended(
    monkeypatch, tmp_path, outcome, end
):
    assert _mcp_turn(monkeypatch, tmp_path, outcome) == end


def test_a_turn_waiting_for_a_decision_leaves_the_run_without_an_end(
    monkeypatch, tmp_path
):
    """The earlier turn's outcome is cleared as the turn starts: a session killed
    while a turn waits stays open, not completed."""
    end = _mcp_turn(
        monkeypatch,
        tmp_path,
        {"status": "needs_decision", "summary": ""},
        earlier="completed",
    )
    assert end == (None, None)


# -- chat in the Go TUI (and the stdio bridge) --------------------------------------


class _Bridge:
    cancel_event = None

    def send(self, *args, **kwargs):
        pass

    def recv_event(self, timeout=None):
        return None


def _go_client(events):
    client = _client()
    client.token_tracker = SimpleNamespace(total_tokens=0, total_cost=0.0)
    client.query = lambda text, stream=True, cancel_event=None: events(cancel_event)
    return client


def _go_turn(events):
    from lobster.cli_internal import go_tui_launcher

    client = _go_client(events)
    go_tui_launcher._handle_user_query(
        _Bridge(), client, "q", cancel_event=threading.Event()
    )
    return client.catalog.end


def _complete(cancel):
    yield {"type": "complete"}


def _error(cancel):
    yield {"type": "error", "error": "backend"}


def _raises(cancel):
    yield {"type": "content_delta", "delta": "x"}
    raise RuntimeError("broke")


def _cancelled(cancel):
    yield {"type": "content_delta", "delta": "x"}
    cancel.set()


def _ctrl_c(cancel):
    yield {"type": "content_delta", "delta": "x"}
    raise KeyboardInterrupt


@pytest.mark.parametrize(
    "events, end",
    [
        (_complete, ("completed", None)),
        (_error, ("errored", "error")),
        (_raises, ("errored", "RuntimeError")),
        (_cancelled, ("aborted", "cancelled")),
        (_ctrl_c, ("aborted", "cancelled")),
    ],
)
def test_a_go_tui_turn_ends_the_session_run_as_it_ended(events, end):
    assert _go_turn(events) == end


def test_a_running_go_tui_turn_has_no_end_status():
    from lobster.cli_internal import go_tui_launcher

    seen = []

    def events(cancel):
        seen.append(client.catalog.end)
        yield {"type": "complete"}

    client = _go_client(events)
    client.catalog.set_end_status("errored", reason="error")  # the turn before
    go_tui_launcher._handle_user_query(
        _Bridge(), client, "q", cancel_event=threading.Event()
    )
    assert seen == [(None, None)]


def test_quitting_a_go_tui_chat_while_a_turn_runs_leaves_the_run_open(monkeypatch):
    """A turn that never settled is not a completed chat."""
    from lobster.cli_internal import go_tui_launcher

    def events(cancel):
        yield {"type": "content_delta", "delta": "x"}
        raise SystemExit(0)  # the process leaves mid-turn

    client = _go_client(events)
    client.catalog.set_end_status("completed")  # an earlier turn
    session = _Session([{"type": "input", "payload": {"content": "q"}}])
    with pytest.raises(SystemExit):
        go_tui_launcher._go_tui_event_loop(session, client)
    assert client.catalog.end == (None, None) and client.catalog.turn_open


class _Session(_Bridge):
    def __init__(self, events):
        self.events = list(events)

    def recv_event(self, timeout=None):
        event = self.events.pop(0) if self.events else None
        if isinstance(event, BaseException):
            raise event
        return event


@pytest.mark.parametrize(
    "events, end",
    [
        ([{"type": "quit"}], ("completed", None)),
        ([], ("completed", None)),  # the TUI went away
        (
            [{"type": "input", "payload": {"content": "q"}}, {"type": "quit"}],
            ("errored", "error"),  # the turn's outcome stands
        ),
        ([OSError("pipe")], ("errored", "OSError")),
    ],
)
def test_a_go_tui_chat_ends_as_its_last_turn(monkeypatch, events, end):
    from lobster.cli_internal import go_tui_launcher

    monkeypatch.setattr(go_tui_launcher, "_save_session_json_if_available", print)
    client = _go_client(_error)
    try:
        go_tui_launcher._go_tui_event_loop(_Session(events), client)
    except OSError:
        pass
    assert client.catalog.end == end


# -- classic chat ---------------------------------------------------------------------


def _classic_chat(monkeypatch, tmp_path, inputs, answer):
    from lobster.cli_internal.commands.heavy import chat_commands

    client = _client()
    client.workspace_path = tmp_path
    client.get_token_usage = lambda: None
    inputs = iter(inputs)

    def read(prompt, client=None):
        value = next(inputs, KeyboardInterrupt())
        if isinstance(value, BaseException):
            raise value
        return value

    def query(question, **kwargs):
        if isinstance(answer, BaseException):
            raise answer
        return answer

    for name, value in {
        "validate_startup_or_raise_startup_diagnostic": lambda *a, **k: tmp_path,
        "display_welcome": lambda *a, **k: None,
        "display_goodbye": lambda *a, **k: None,
        "init_client_with_animation": lambda *a, **k: client,
        "_show_workspace_prompt": lambda c: None,
        "get_user_input_with_editing": read,
        "should_show_progress": lambda c: False,
        "_maybe_print_timings": lambda *a, **k: None,
        "_query_classic": lambda c, text: query(text),
        "check_for_missing_slash_command": lambda text: None,
        "execute_shell_command": lambda text: False,
        "console": Console(record=True, width=120),
    }.items():
        monkeypatch.setattr(chat_commands, name, value)
    # Ctrl-C at the prompt asks "exit?": yes.
    monkeypatch.setattr(chat_commands.Confirm, "ask", lambda *a, **k: True)
    chat_commands.chat_impl(
        workspace=tmp_path,
        session_id=None,
        reasoning=False,
        verbose=False,
        debug=False,
        profile_timings=None,
        provider=None,
        model=None,
        stream=False,
    )
    return client.catalog.end


@pytest.mark.parametrize(
    "answer, end",
    [
        ({"success": True, "response": "hi"}, ("completed", None)),
        ({"success": False, "error": "no"}, ("errored", "error")),
        (ValueError("bad"), ("errored", "ValueError")),
        (KeyboardInterrupt(), ("aborted", "Ctrl-C")),
    ],
)
def test_a_classic_chat_ends_as_its_last_turn(monkeypatch, tmp_path, answer, end):
    assert _classic_chat(monkeypatch, tmp_path, ["hello"], answer) == end


def test_a_running_classic_turn_has_no_end_status(monkeypatch, tmp_path):
    from lobster.cli_internal.commands.heavy import chat_commands

    seen = []
    monkeypatch.setattr(
        chat_commands,
        "note_run_status",
        lambda client, status, *a, **k: seen.append(status),
    )
    _classic_chat(monkeypatch, tmp_path, ["hello"], {"success": True})
    assert seen[:2] == [None, "completed"]


def test_a_classic_chat_ended_without_a_turn_is_completed(monkeypatch, tmp_path):
    assert _classic_chat(monkeypatch, tmp_path, [], None) == ("completed", None)


def test_a_streamed_classic_turn_stopped_with_ctrl_c_is_aborted():
    from lobster.cli_internal.commands.heavy import chat_commands

    client = _client()
    chat_commands._note_turn_status(
        client, {"success": False, "error": "Interrupted by user", "interrupted": True}
    )
    assert client.catalog.end == ("aborted", "Ctrl-C")


# -- lobster serve ---------------------------------------------------------------


@pytest.mark.parametrize(
    "answer, end",
    [
        ({"success": True, "response": "hi"}, ("completed", None)),
        ({"success": False, "error": "no"}, ("errored", "error")),
        (RuntimeError("boom"), ("errored", "RuntimeError")),
    ],
)
def test_a_served_query_ends_its_run_as_it_ended(answer, end):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from lobster.cli_internal.commands.heavy.serve_app import build_serve_app

    client = _answering(answer)
    api = TestClient(
        build_serve_app("127.0.0.1", None, lambda: client),
        base_url="http://localhost:8000",
    )
    api.post("/query", json={"question": "q"})
    assert client.catalog.end == end


def test_a_served_status_request_is_completed():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from lobster.cli_internal.commands.heavy.serve_app import build_serve_app

    client = _client()
    client.get_status = lambda: {"ok": True}
    api = TestClient(
        build_serve_app("127.0.0.1", None, lambda: client),
        base_url="http://localhost:8000",
    )
    api.get("/status")
    assert client.catalog.end == ("completed", None)


# -- a Ctrl-C or SIGTERM ends the run before the exit waits for a running tool --


@pytest.fixture
def exit_hooks(monkeypatch):
    """Hooks registered to run as the exit begins, recorded instead of registered."""
    hooks = []
    monkeypatch.setattr(threading, "_register_atexit", hooks.append)
    monkeypatch.setattr(factory, "_ending_early", False, raising=False)
    return hooks


def test_the_early_end_is_registered_once_and_only_for_a_bound_run(
    monkeypatch, exit_hooks
):
    factory.end_run_when_exit_begins()
    assert exit_hooks == []  # no run: nothing to end

    monkeypatch.setattr(factory, "_process_catalog", (Path("."), _Catalog()))
    factory.end_run_when_exit_begins()
    factory.end_run_when_exit_begins()
    assert len(exit_hooks) == 1


def test_a_ctrl_c_registers_the_early_end(monkeypatch, exit_hooks):
    monkeypatch.setattr(factory, "_process_catalog", (Path("."), _Catalog()))
    with pytest.raises(KeyboardInterrupt):
        with factory.recorded_run_status(_client("s")):
            raise KeyboardInterrupt
    assert len(exit_hooks) == 1


def test_a_ctrl_c_at_a_prompt_registers_the_early_end(monkeypatch, exit_hooks):
    from click.exceptions import Abort

    monkeypatch.setattr(factory, "_process_catalog", (Path("."), _Catalog()))
    with pytest.raises(Abort):
        with factory.recorded_run_status(_client("s")):
            raise Abort
    assert len(exit_hooks) == 1


@pytest.mark.parametrize("code, registered", [(130, 1), (143, 1), (1, 0), (0, 0)])
def test_an_exit_code_of_an_abort_registers_the_early_end(
    monkeypatch, exit_hooks, code, registered
):
    monkeypatch.setattr(factory, "_process_catalog", (Path("."), _Catalog()))
    with pytest.raises(typer.Exit):
        with factory.recorded_run_status(_client("s")):
            raise typer.Exit(code)
    assert len(exit_hooks) == registered


def test_an_exit_already_begun_registers_nothing_and_raises_nothing(monkeypatch):
    def too_late(fn):
        raise RuntimeError("can't register atexit after shutdown")

    monkeypatch.setattr(threading, "_register_atexit", too_late)
    monkeypatch.setattr(factory, "_ending_early", False, raising=False)
    monkeypatch.setattr(factory, "_process_catalog", (Path("."), _Catalog()))
    factory.end_run_when_exit_begins()
    assert factory._ending_early is False


def test_a_sigterm_registers_the_early_end(monkeypatch, exit_hooks, exit_later):
    monkeypatch.setattr(factory, "_process_catalog", (Path("."), _Catalog()))
    with pytest.raises(SystemExit):
        factory._on_sigterm(signal.SIGTERM, None)
    assert len(exit_hooks) == 1


def test_a_streamed_query_stopped_with_ctrl_c_registers_the_early_end(
    monkeypatch, tmp_path, exit_hooks
):
    from lobster.cli_internal.commands.heavy import query_commands

    monkeypatch.setattr(factory, "_process_catalog", (Path("."), _Catalog()))
    monkeypatch.setattr(
        query_commands,
        "_display_streaming_response",
        lambda *a: {"success": False, "error": "Interrupted", "interrupted": True},
    )
    with pytest.raises(typer.Exit):
        _query(monkeypatch, tmp_path, _client(), stream=True)
    assert len(exit_hooks) == 1


def test_the_hard_exit_ends_the_run_first(monkeypatch):
    done = []
    catalog = SimpleNamespace(end_at_exit=lambda: done.append("ended"))
    monkeypatch.setattr(factory, "_process_catalog", (Path("."), catalog))
    timers = []
    monkeypatch.setattr(
        threading, "Timer", lambda s, fn, args=(): timers.append((fn, args)) or Mock()
    )
    monkeypatch.setattr(factory.os, "_exit", lambda code: done.append(code))

    factory._exit_later(143)
    fn, args = timers[0]
    fn(*args)

    assert done == ["ended", 143]


@pytest.fixture
def sigint(monkeypatch):
    """SIGINT handlers set, recorded instead of set; starts as Python's own."""
    handlers = [signal.default_int_handler]
    monkeypatch.setattr(signal, "getsignal", lambda sig: handlers[-1])
    monkeypatch.setattr(signal, "signal", lambda sig, handler: handlers.append(handler))
    return handlers


@pytest.fixture
def started(monkeypatch):
    """Threads started, recorded instead of run."""
    threads = []

    class _Thread:
        def __init__(self, target, args=(), **kw):
            self.run = lambda: target(*args)
            self.daemon = kw.get("daemon")

        def start(self):
            threads.append(self)

    monkeypatch.setattr(factory.threading, "Thread", _Thread)
    return threads


def test_the_first_ctrl_c_ends_the_run_from_a_thread_and_arms_the_second(
    monkeypatch, sigint, started, exit_hooks, capsys
):
    client = _client()
    catalog = client.catalog
    ended = []
    catalog.end_at_exit = lambda: ended.append(catalog.status)
    monkeypatch.setattr(factory, "_process_catalog", (Path("."), catalog))
    with pytest.raises(KeyboardInterrupt):
        with factory.recorded_run_status(client):
            first_press = sigint[-1]
            try:
                first_press(signal.SIGINT, None)  # the press, where Python calls it
            finally:
                stop_now = sigint[-1]  # while the command still runs

    assert WAITING in " ".join(capsys.readouterr().err.split())
    assert (
        ended == []
    )  # not in the handler: it may run where the catalog's lock is held
    (thread,) = started
    assert thread.daemon
    thread.run()
    assert ended == ["aborted"]
    assert stop_now not in (first_press, signal.default_int_handler, signal.SIG_DFL)
    monkeypatch.setattr(factory, "_stop_waiting", lambda text: ended.append(text))
    stop_now(signal.SIGINT, None)
    assert ended[-1] == STOPPING


def test_a_command_that_goes_on_after_the_first_ctrl_c_is_still_aborted(
    monkeypatch, sigint, started
):
    client = _client()
    monkeypatch.setattr(factory, "_process_catalog", (Path("."), client.catalog))
    with factory.recorded_run_status(client):
        try:
            sigint[-1](signal.SIGINT, None)
        except KeyboardInterrupt:
            pass  # a step that swallows it; the command ends normally
    assert client.catalog.end == ("aborted", "Ctrl-C")
    # a caller that goes on gets Python's own Ctrl-C back, not a hard exit
    assert sigint[-1] is signal.default_int_handler


class _ReenteredStderr:
    """stderr written to from a signal handler while the thread was writing it."""

    def write(self, text):
        raise RuntimeError("reentrant call inside <_io.BufferedWriter name='<stderr>'>")

    def flush(self):
        pass


def test_the_first_ctrl_c_goes_on_when_its_line_cannot_be_written(
    monkeypatch, sigint, started
):
    client = _client()
    monkeypatch.setattr(factory, "_process_catalog", (Path("."), client.catalog))
    with pytest.raises(KeyboardInterrupt):
        with factory.recorded_run_status(client):
            monkeypatch.setattr(sys, "stderr", _ReenteredStderr())
            sigint[-1](signal.SIGINT, None)
    assert len(started) == 1


def test_the_stop_now_exits_when_its_line_cannot_be_written(monkeypatch, sigint):
    import atexit

    exits = []
    monkeypatch.setattr(atexit, "_run_exitfuncs", lambda: None)
    monkeypatch.setattr(factory.os, "_exit", exits.append)
    monkeypatch.setattr(sys, "stderr", _ReenteredStderr())
    factory._stop_waiting("bye")
    assert exits == [130]


def test_without_a_press_the_ctrl_c_handler_is_put_back(monkeypatch, sigint):
    monkeypatch.setattr(factory, "_process_catalog", (Path("."), _Catalog()))
    with factory.recorded_run_status(_client("s")):
        assert sigint[-1] is not signal.default_int_handler
    assert sigint[-1] is signal.default_int_handler


@pytest.mark.parametrize("case", ["session", "no catalog", "own handler", "thread"])
def test_the_first_ctrl_c_is_left_alone_where_it_is_not_ours(monkeypatch, sigint, case):
    if case != "no catalog":
        monkeypatch.setattr(factory, "_process_catalog", (Path("."), _Catalog()))
    if case == "own handler":
        sigint.append(signal.SIG_IGN)
    if case == "thread":
        monkeypatch.setattr(factory.threading, "main_thread", lambda: None)
    before = list(sigint)
    with factory.recorded_run_status(_client("s"), session=case == "session"):
        pass
    assert sigint == before


def test_a_ctrl_c_during_the_exit_hooks_is_not_handled_twice(monkeypatch, sigint):
    import atexit

    seen = []
    monkeypatch.setattr(atexit, "_run_exitfuncs", lambda: seen.append(sigint[-1]))
    monkeypatch.setattr(factory.os, "_exit", lambda code: seen.append(code))
    factory._stop_waiting("bye")
    assert seen == [signal.SIG_IGN, 130]


def test_the_exit_hooks_get_python_s_own_ctrl_c_back(monkeypatch, sigint):
    import atexit

    registered = []
    monkeypatch.setattr(atexit, "register", lambda *a: registered.append(a))
    monkeypatch.setattr(factory, "_end_run_now", lambda: None)
    factory._exit_begins()
    assert registered == [(signal.signal, signal.SIGINT, signal.default_int_handler)]


def test_the_hard_exit_exits_even_when_ending_the_run_fails(monkeypatch):
    def broken():
        raise RuntimeError("catalog gone")

    catalog = SimpleNamespace(end_at_exit=broken)
    monkeypatch.setattr(factory, "_process_catalog", (Path("."), catalog))
    exits = []
    monkeypatch.setattr(factory.os, "_exit", exits.append)
    with pytest.raises(RuntimeError):
        factory._end_then_exit(143)
    assert exits == [143]


CTRL_C_CHILD = textwrap.dedent("""
    import atexit, sys, time
    from pathlib import Path
    from types import SimpleNamespace
    from lobster.cli_internal.commands.heavy import query_commands
    from lobster.core.catalog import factory, thread
    thread.AT_EXIT_TIMEOUT_S = 1.0
    mode, bound = sys.argv[1], sys.argv[2]
    class Catalog:  # as ThreadedCatalog: the last status wins until the run ends
        status = reason = None
        ended = 0
        def set_end_status(self, status, session_id=None, reason=None, **only):
            if not Catalog.ended:
                self.status, self.reason = status, reason
        def end_at_exit(self):
            if not Catalog.ended:
                Catalog.ended += 1
                print("RUN", self.status, self.reason, flush=True)
    catalog = Catalog()
    # a stdout no logging handler holds, not flushed: an exit that skips the flush
    # loses the line
    sys.stdout = open(sys.stdout.fileno(), "w", closefd=False)
    atexit.register(
        lambda: print("EXIT HOOKS RAN; ended", Catalog.ended, "as", catalog.status)
    )
    # the tool runs where a real one does: in a langgraph ToolNode, whose executor
    # waits for it before a Ctrl-C leaves the graph
    from langchain_core.messages import AIMessage
    from langchain_core.tools import tool
    from langgraph.graph import START, MessagesState, StateGraph
    from langgraph.prebuilt import ToolNode
    @tool
    def slow(seconds: float) -> str:
        'A call that answers late.'
        print("ready", flush=True)
        time.sleep(seconds)
        Path("out.bin").write_bytes(b"x" * 1_000_000)
        print("TOOL done", flush=True)
        return "done"
    graph = StateGraph(MessagesState)
    graph.add_node("tools", ToolNode([slow]))
    graph.add_edge(START, "tools")
    graph = graph.compile()
    class Client:
        callbacks = []
        data_manager = SimpleNamespace(catalog=catalog)
        def query(self, question, stream=False):
            if mode == "free":
                print("ready", flush=True)
                time.sleep(30)  # waits in the main thread: nothing holds the exit
            call = {"name": "slow", "args": {"seconds": float(mode)}, "id": "1"}
            graph.invoke({"messages": [AIMessage(content="", tool_calls=[call])]})
            return {"success": True, "response": "x"}
    def init_client(*args, **kwargs):
        if bound == "catalog":
            factory._process_catalog = (Path("."), catalog)
            factory.abort_runs_on_sigterm()
        return Client()
    query_commands.init_client = init_client
    query_commands.validate_startup_or_raise_startup_diagnostic = lambda **kw: None
    from lobster.cli import app
    app(["query", "--no-stream", "hello"])
""")

STOPPING = (
    "Stopping without waiting for the running tool; "
    "a file it was writing may be incomplete."
)
WAITING = (
    "Stopping: waiting for the running step to finish; "
    "press Ctrl+C again to stop now."
)


def _signalled(tmp_path, mode, bound, presses):
    """Run the child; ``presses``: (seconds after ready, signal). Returns the exit
    code and each output line with the seconds since the first signal."""
    pytest.importorskip("langgraph.prebuilt")
    child = subprocess.Popen(
        [sys.executable, "-c", CTRL_C_CHILD, mode, bound],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=tmp_path,
        env={"PATH": "", "PYTHONPATH": ":".join(sys.path), "HOME": str(tmp_path)},
    )
    lines = []
    try:
        assert child.stdout.readline().strip() == "ready"
        began = time.monotonic()

        def press():
            for at, sig in presses:
                time.sleep(max(0.0, began + at - time.monotonic()))
                child.send_signal(sig)

        threading.Thread(target=press, daemon=True).start()
        for line in child.stdout:
            lines.append((time.monotonic() - began, line.rstrip()))
        code = child.wait(timeout=40)
        lines.append((time.monotonic() - began, "<exit>"))
    finally:
        child.kill()
    return code, lines


def _at(lines, text):
    return next(at for at, line in lines if line.startswith(text))


def _said(lines):
    return "\n".join(line for _, line in lines)


def test_a_ctrl_c_ends_the_run_while_a_tool_still_runs_and_cuts_nothing(tmp_path):
    code, lines = _signalled(tmp_path, "4", "catalog", [(0, signal.SIGINT)])

    said = _said(lines)
    assert code == 130, said
    assert _at(lines, WAITING) < 1.0
    assert _at(lines, "RUN aborted Ctrl-C") < 1.0 + 1.0  # within the exit budget
    assert _at(lines, "RUN") < _at(lines, "TOOL done")  # the tool still ran
    assert (tmp_path / "out.bin").read_bytes() == b"x" * 1_000_000
    # the tool finishing later never turns the run completed
    assert "EXIT HOOKS RAN; ended 1 as aborted" in said
    assert "Traceback" not in said


def test_a_ctrl_c_with_nothing_running_exits_at_once(tmp_path):
    code, lines = _signalled(tmp_path, "free", "catalog", [(0, signal.SIGINT)])

    assert code == 130, _said(lines)
    assert _at(lines, "<exit>") < 1.0
    assert "EXIT HOOKS RAN; ended 1 as aborted" in _said(lines)


def test_a_second_ctrl_c_stops_waiting_for_the_tool_without_a_traceback(tmp_path):
    code, lines = _signalled(
        tmp_path, "30", "catalog", [(0, signal.SIGINT), (2.0, signal.SIGINT)]
    )

    said = _said(lines)
    assert code == 130, said
    assert _at(lines, "RUN aborted Ctrl-C") < 1.0 + 1.0
    assert _at(lines, STOPPING) >= 2.0
    assert "Traceback" not in said and "Exception ignored" not in said
    assert "EXIT HOOKS RAN; ended 1 as aborted" in said
    assert _at(lines, "<exit>") < 2.0 + 1.5


def test_a_ctrl_c_without_a_run_is_unchanged(tmp_path):
    code, lines = _signalled(tmp_path, "3", "none", [(0, signal.SIGINT)])

    said = _said(lines)
    assert code == 130, said
    assert "RUN" not in said and STOPPING not in said and WAITING not in said
    assert _at(lines, "TOOL done") < _at(lines, "<exit>")  # it waits, as before


def test_a_sigterm_ends_the_run_before_the_hard_exit(tmp_path):
    code, lines = _signalled(tmp_path, "30", "catalog", [(0, signal.SIGTERM)])

    assert code == 143, _said(lines)
    assert _at(lines, "RUN aborted SIGTERM") < _at(lines, "<exit>") < 1.0 + 2.5
