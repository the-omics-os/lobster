"""Ctrl-C in a classic ``lobster chat`` turn while a tool runs.

The first press ends the turn once its running step is done, and says so; the
session goes on. The second press stops at once. The tool runs in a real
langgraph ToolNode, whose executor waits for it before a Ctrl-C leaves the graph.
"""

import signal
import subprocess
import sys
import textwrap
import threading
import time

import pytest

CHAT_CHILD = textwrap.dedent("""
    import atexit, io, sys, time
    from pathlib import Path
    from types import SimpleNamespace
    from rich.console import Console
    from lobster.cli_internal.commands.heavy import chat_commands
    from lobster.core.catalog import factory, thread
    thread.AT_EXIT_TIMEOUT_S = 1.0
    mode, turns, answers = sys.argv[1], iter(sys.argv[2].split(",")), iter(sys.argv[3].split(","))
    class Catalog:  # as ThreadedCatalog: a turn clears the status; the run ends once
        status = reason = None
        turn_open = False
        ended = 0
        def set_end_status(self, status, session_id=None, reason=None, *, if_unset=False, if_unfinished=False):
            if Catalog.ended:
                return
            if status is None:
                self.status, self.reason, self.turn_open = None, None, True
                return
            if if_unset and (self.status is not None or self.turn_open):
                return
            self.status, self.reason, self.turn_open = status, reason, False
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
    from langchain_core.messages import AIMessage
    from langchain_core.tools import tool
    from langgraph.graph import START, MessagesState, StateGraph
    from langgraph.prebuilt import ToolNode
    calls = []
    @tool
    def slow(seconds: float) -> str:
        'A call that answers late.'
        calls.append(seconds)
        n = len(calls)
        print("ready", flush=True)
        time.sleep(seconds)
        Path(f"out{n}.bin").write_bytes(b"x" * 1_000_000)
        print("TOOL done", n, flush=True)
        return "done"
    graph = StateGraph(MessagesState)
    graph.add_node("tools", ToolNode([slow]))
    graph.add_edge(START, "tools")
    graph = graph.compile()
    def run(seconds):
        call = {"name": "slow", "args": {"seconds": float(seconds)}, "id": "1"}
        graph.invoke({"messages": [AIMessage(content="", tool_calls=[call])]})
    def events(seconds):
        run(seconds)
        yield {"type": "complete", "response": "x"}
    class Client:
        session_id = "s"
        workspace_path = Path(".")
        callbacks = []
        data_manager = SimpleNamespace(catalog=catalog)
        def get_token_usage(self):
            return None
        def query(self, question, stream=False):
            if stream:
                return events(question)
            run(question)
            return {"success": True, "response": "x"}
    def init_client(*args, **kwargs):
        factory._process_catalog = (Path("."), catalog)
        factory.abort_runs_on_sigterm()
        atexit.register(catalog.end_at_exit)  # as the catalog does when it binds
        return Client()
    def read(prompt, client=None):
        question = next(turns, None)
        if question is None:
            raise KeyboardInterrupt
        return question
    def ask(*args, **kwargs):
        print("ASK exit?", catalog.status, "ended", Catalog.ended, flush=True)
        return next(answers) == "y"
    for name, value in {
        "validate_startup_or_raise_startup_diagnostic": lambda *a, **k: None,
        "display_welcome": lambda *a, **k: None,
        "display_goodbye": lambda *a, **k: None,
        "init_client_with_animation": init_client,
        "_show_workspace_prompt": lambda c: None,
        "get_user_input_with_editing": read,
        "should_show_progress": lambda c: mode == "stream",
        "_maybe_print_timings": lambda *a, **k: None,
        "check_for_missing_slash_command": lambda text: None,
        "execute_shell_command": lambda text: False,
        "console": Console(file=io.StringIO()),
    }.items():
        setattr(chat_commands, name, value)
    chat_commands.Confirm.ask = ask
    chat_commands.chat_impl(
        workspace=Path("."), session_id=None, reasoning=False, verbose=False,
        debug=False, profile_timings=None, provider=None, model=None,
        stream=mode == "stream",
    )
""")

STOPPING = (
    "Stopping without waiting for the running tool; "
    "a file it was writing may be incomplete."
)
WAITING = (
    "Stopping: waiting for the running step to finish; "
    "press Ctrl+C again to stop now."
)
MODES = pytest.mark.parametrize("mode", ["classic", "stream"])


def _chat(tmp_path, mode, turns, answers, presses):
    """Run a chat of ``turns`` (each a tool's seconds), answering "exit?" with
    ``answers``; ``presses``: turn -> [(seconds after its tool starts, signal)].
    Returns the exit code and each output line with the seconds since the first
    tool started."""
    pytest.importorskip("langgraph.prebuilt")
    child = subprocess.Popen(
        [sys.executable, "-c", CHAT_CHILD, mode, turns, answers],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=tmp_path,
        env={"PATH": "", "PYTHONPATH": ":".join(sys.path), "HOME": str(tmp_path)},
    )

    def press(at, sigs):
        for after, sig in sigs:
            time.sleep(max(0.0, at + after - time.monotonic()))
            child.send_signal(sig)

    lines, began, turn = [], None, 0
    try:
        for line in child.stdout:
            now = time.monotonic()
            began = began or now
            lines.append((now - began, line.rstrip()))
            if line.strip() == "ready":
                turn += 1
                threading.Thread(
                    target=press, args=(now, presses.get(turn, [])), daemon=True
                ).start()
        code = child.wait(timeout=40)
        lines.append((time.monotonic() - began, "<exit>"))
    finally:
        child.kill()
    return code, lines


def _at(lines, text):
    return next(at for at, line in lines if line.startswith(text))


def _said(lines):
    return "\n".join(line for _, line in lines)


@MODES
def test_a_first_ctrl_c_ends_the_turn_after_its_tool_and_the_session_goes_on(
    tmp_path, mode
):
    code, lines = _chat(tmp_path, mode, "3", "n,y", {1: [(0, signal.SIGINT)]})

    said = _said(lines)
    assert code == 0, said
    assert _at(lines, WAITING) < 1.0
    # the turn ends once its tool is done; the session's run is not ended
    assert _at(lines, "TOOL done 1") < _at(lines, "ASK exit? aborted ended 0")
    assert (tmp_path / "out1.bin").read_bytes() == b"x" * 1_000_000
    assert "RUN aborted Ctrl-C" in said and "Traceback" not in said


@MODES
def test_a_second_ctrl_c_in_a_turn_stops_at_once(tmp_path, mode):
    code, lines = _chat(
        tmp_path, mode, "30", "y", {1: [(0, signal.SIGINT), (2.0, signal.SIGINT)]}
    )

    said = _said(lines)
    assert code == 130, said
    assert _at(lines, STOPPING) >= 2.0
    assert "RUN aborted Ctrl-C" in said
    assert "EXIT HOOKS RAN; ended 1 as aborted" in said
    assert "Traceback" not in said and "Exception ignored" not in said
    assert "ASK" not in said
    assert _at(lines, "<exit>") < 2.0 + 1.5


@MODES
def test_a_turn_without_a_ctrl_c_is_unchanged(tmp_path, mode):
    code, lines = _chat(tmp_path, mode, "1", "y", {})

    said = _said(lines)
    assert code == 0, said
    assert WAITING not in said and STOPPING not in said
    assert "ASK exit? completed ended 0" in said
    assert "RUN completed None" in said


@MODES
def test_the_next_turn_after_a_ctrl_c_works_and_hears_its_own(tmp_path, mode):
    code, lines = _chat(
        tmp_path,
        mode,
        "3,3",
        "n,n,y",
        {1: [(0, signal.SIGINT)], 2: [(0, signal.SIGINT)]},
    )

    said = _said(lines)
    assert code == 0, said
    assert said.count(WAITING) == 2
    assert _at(lines, "TOOL done 1") < _at(lines, "TOOL done 2")
    for n in (1, 2):
        assert (tmp_path / f"out{n}.bin").read_bytes() == b"x" * 1_000_000
    assert "RUN aborted Ctrl-C" in said and STOPPING not in said


@MODES
def test_exiting_after_a_first_ctrl_c_ends_the_run_aborted_at_once(tmp_path, mode):
    code, lines = _chat(tmp_path, mode, "3", "y", {1: [(0, signal.SIGINT)]})

    said = _said(lines)
    assert code == 0, said
    assert "EXIT HOOKS RAN; ended 1 as aborted" in said
    assert "RUN aborted Ctrl-C" in said
    assert _at(lines, "<exit>") < _at(lines, "TOOL done 1") + 1.5


def test_a_turn_ends_as_its_caller_says_after_a_press(monkeypatch):
    from pathlib import Path
    from types import SimpleNamespace

    from lobster.core.catalog import factory

    seen = []

    class Catalog:
        def set_end_status(self, status, session_id=None, reason=None, **only):
            if threading.current_thread().name == "lobster-ctrl-c":
                time.sleep(0.3)  # the press's note is late
            seen.append(status)

    holder = SimpleNamespace(catalog=Catalog())
    handlers = [signal.default_int_handler]
    monkeypatch.setattr(signal, "getsignal", lambda sig: handlers[-1])
    monkeypatch.setattr(signal, "signal", lambda sig, handler: handlers.append(handler))
    monkeypatch.setattr(factory, "_process_catalog", (Path("."), holder.catalog))
    with factory.ctrl_c_ends_the_turn(holder):
        with pytest.raises(KeyboardInterrupt):
            handlers[-1](signal.SIGINT, None)
    factory.note_run_status(holder, "completed")  # the turn went on to the end

    assert seen == ["aborted", "completed"]
