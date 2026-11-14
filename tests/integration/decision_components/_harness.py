"""End-to-end harness: the real lobster-tui binary on a pseudo-terminal, the real
launcher bridge and event loop, and the real AgentClient with a scripted model.

What is real
------------
* ``lobster-tui`` (the Go binary), started with the same command line, fds,
  ``pass_fds``, ``preexec_fn=os.setsid`` and child environment
  (``_prepare_go_tui_chat_env``) as ``launch_go_tui_chat``; its stdin/stdout
  are the slave side of a stdlib ``pty.openpty()`` pair with a fixed size.
* ``_LightBridge``, ``_go_tui_event_loop`` → ``_dispatch_user_query`` →
  ``_handle_user_query`` → ``_handle_interrupts`` → ``_answer_interrupt`` /
  ``_handle_interrupt`` from ``go_tui_launcher`` (not re-implemented).
* ``AgentClient`` (``client.query`` / ``resume_from_interrupt`` /
  ``_stream_query``), the real supervisor graph from
  ``create_bioinformatics_graph`` (``create_react_agent``, ToolNode, the
  InMemorySaver checkpointer) and the real ``ask_user`` tool.

What is scripted
----------------
* The supervisor's chat model: ``create_llm`` in ``lobster.agents.graph`` is
  patched to return :class:`ScriptedSupervisor`, which emits fixed ask_user
  tool calls. No LLM, no network, no provider key.
* No worker agents: the workspace ``config.toml`` enables one agent name that
  does not exist, so only the supervisor is built.
* Two relay threads sit on the protocol pipes between the bridge and the
  binary. They forward every line unchanged and record it; a test can also
  write a crafted line towards the bridge (a "crafted reply").
"""

from __future__ import annotations

import fcntl
import importlib.util
import json
import os
import pty
import re
import select
import signal
import struct
import subprocess  # nosec B404 - starts the built lobster-tui binary with a fixed argument list
import termios
import threading
import time
import uuid
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union
from unittest.mock import patch

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from tests.integration.decision_components._screen import Screen

REPO = Path(__file__).resolve().parents[3]
GENERATOR = REPO / "docs" / "dynamic-ui" / "demo" / "make_synthetic_rnaseq.py"
DEFAULT_BINARY = REPO / "lobster-tui" / "lobster-tui"

# Waits are bounded everywhere. These are generous for a loaded machine; a
# passing run is far below them.
WAIT = 20.0
QUIET = 0.15  # seconds without screen output that count as "drawn"

# --------------------------------------------------------------------------
# Keys (written to the pty master, as a terminal would send them)
# --------------------------------------------------------------------------

KEYS = {
    "enter": b"\r",
    "tab": b"\t",
    "shift+tab": b"\x1b[Z",
    "esc": b"\x1b",
    "up": b"\x1b[A",
    "down": b"\x1b[B",
    "right": b"\x1b[C",
    "left": b"\x1b[D",
    "pgup": b"\x1b[5~",
    "pgdown": b"\x1b[6~",
    "backspace": b"\x7f",
    "space": b" ",
    "ctrl+s": b"\x13",
    "ctrl+d": b"\x04",
    "ctrl+c": b"\x03",
}


def binary_path() -> Optional[Path]:
    """The lobster-tui binary to test, or None when it is not built."""
    env = os.environ.get("LOBSTER_TUI_BINARY")
    path = Path(env) if env else DEFAULT_BINARY
    return path if path.is_file() and os.access(path, os.X_OK) else None


# --------------------------------------------------------------------------
# Synthetic data (docs/dynamic-ui/demo/make_synthetic_rnaseq.py)
# --------------------------------------------------------------------------

_generator = None


def _load_generator():
    global _generator
    if _generator is None:
        import sys

        name = "_e2e_make_synthetic_rnaseq"
        spec = importlib.util.spec_from_file_location(name, GENERATOR)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module  # dataclasses look their module up here
        spec.loader.exec_module(module)
        _generator = module
    return _generator


def _read_csv(path: Path) -> List[Dict[str, str]]:
    import csv

    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


@dataclass
class Dataset:
    """One generated case in a workspace, read back from its own files."""

    case: str
    prefix: str
    workspace: Path
    samples: List[Dict[str, str]]
    sources: List[Dict[str, str]]
    suggestions: Dict[str, str]
    qc: Dict[str, Dict[str, str]]

    def rel(self, name: str) -> str:
        """Workspace-relative path of one of the case's files."""
        return f"data/{self.prefix}_{name}"

    def path(self, name: str) -> Path:
        return self.workspace / self.rel(name)

    @property
    def ids(self) -> List[str]:
        return [s["sample_id"] for s in self.samples]

    def label(self, sid: str) -> str:
        return next(s["display_label"] for s in self.samples if s["sample_id"] == sid)

    def condition(self, sid: str) -> str:
        return next(s["condition"] for s in self.samples if s["sample_id"] == sid)

    @property
    def unknown(self) -> List[str]:
        return [s["sample_id"] for s in self.samples if s["condition"] == "unknown"]

    @property
    def warned(self) -> List[str]:
        """Samples with a QC warning (not the unresolved-condition ones)."""
        mod = _load_generator()
        return [
            s["sample_id"]
            for s in self.samples
            if s["supplied_warning"] in (mod.LOW_LIBRARY, mod.HIGH_MITO)
        ]

    def of(self, condition: str) -> List[str]:
        return [s["sample_id"] for s in self.samples if s["condition"] == condition]


def make_dataset(case: str, workspace: Path) -> Dataset:
    """Generate ``case`` into ``workspace`` with the repo's generator."""
    mod = _load_generator()
    out = mod.CASES[case](workspace)
    prefix = mod.PREFIXES[case]
    samples = _read_csv(out / f"{prefix}_sample_metadata.csv")
    sources = _read_csv(out / f"{prefix}_sources_manifest.csv")
    aux = _read_csv(out / f"{prefix}_auxiliary_manifest.csv")
    qc = {r["sample_id"]: r for r in _read_csv(out / f"{prefix}_qc_metrics.csv")}
    return Dataset(
        case=case,
        prefix=prefix,
        workspace=workspace,
        samples=samples,
        sources=sources,
        suggestions={r["sample_id"]: r["proposed_condition"] for r in aux},
        qc=qc,
    )


# --------------------------------------------------------------------------
# Questions built from the data files (what a supervisor would send)
# --------------------------------------------------------------------------

LEAVE_OUT = "leave_out"
# The reason press_in_plate types for a sample left out (a removal always
# asks one: check_request refuses a removal option without a reason field).
LEAVE_OUT_REASON = "e2e: left out"


def source_field(ds: Dataset) -> Dict[str, Any]:
    options = []
    for src in ds.sources:
        option: Dict[str, Any] = {
            "id": src["source_id"],
            "label": src["description"],
            "details": f"{src['representation']}, reference {src['reference_build']}",
        }
        if (
            src["representation"] != "raw gene counts"
            or src["reference_build"] != "Build-A"
        ):
            option["unavailable"] = (
                f"{src['representation']}; reference build {src['reference_build']}"
            )
        options.append(option)
    return {
        "key": "source",
        "type": "choice",
        "question": "Which count source should the analysis use?",
        "options": options,
    }


def groups_field(
    ds: Dataset,
    *,
    view: Optional[str] = "plate_map",
    leave_out: bool = True,
    prefill: bool = True,
    depends_on: Sequence[str] = (),
    order: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    items = []
    for sid in order or ds.ids:
        row = next(s for s in ds.samples if s["sample_id"] == sid)
        item: Dict[str, Any] = {"id": sid, "label": row["display_label"]}
        notes = [row["supplied_warning"]] if row["supplied_warning"] else []
        if sid in ds.suggestions:
            notes.append(
                f"a second manifest suggests {ds.suggestions[sid]} (suggestion only)"
            )
        if notes:
            item["details"] = "; ".join(notes)
        items.append(item)
    options: List[Any] = ["control", "treated"]
    if leave_out:
        options.append(
            {
                "id": LEAVE_OUT,
                "label": "leave out",
                "fields": [{"key": "reason", "type": "text"}],
            }
        )
    field: Dict[str, Any] = {
        "key": "groups",
        "type": "choice",
        "question": "Which group is each sample in?",
        "for_each": items,
        "options": options,
    }
    if view:
        field["view"] = view
    if prefill:
        field["default"] = {
            s["sample_id"]: {"id": s["condition"]}
            for s in ds.samples
            if s["condition"] in ("control", "treated")
        }
    if depends_on:
        field["depends_on"] = list(depends_on)
    return field


def direction_field(depends_on: Sequence[str] = ()) -> Dict[str, Any]:
    field: Dict[str, Any] = {
        "key": "direction",
        "type": "choice",
        "question": "Compare which group with which?",
        "options": [
            {"id": "treated_vs_control", "label": "treated compared with control"},
            {"id": "control_vs_treated", "label": "control compared with treated"},
        ],
    }
    if depends_on:
        field["depends_on"] = list(depends_on)
    return field


def qc_field(
    ds: Dataset,
    *,
    items: Optional[Sequence[str]] = None,
    depends_on: Sequence[str] = (),
    default: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    for_each = []
    for sid in items or ds.warned:
        row = next(s for s in ds.samples if s["sample_id"] == sid)
        qc = ds.qc[sid]
        for_each.append(
            {
                "id": sid,
                "label": row["display_label"],
                "details": f"{row['supplied_warning'] or 'no warning'}; library size "
                f"{qc['library_size']}, mito {qc['mito_fraction']}",
            }
        )
    field: Dict[str, Any] = {
        "key": "qc",
        "type": "choice",
        "question": "Keep or exclude each flagged sample?",
        "for_each": for_each,
        "options": [
            "retain",
            "defer",
            {
                "id": "exclude",
                "fields": [
                    {"key": "reason", "type": "text", "question": "Why exclude it?"}
                ],
            },
        ],
    }
    if depends_on:
        field["depends_on"] = list(depends_on)
    if default is not None:
        field["default"] = default
    return field


def groups_kept(minus: bool = False) -> Dict[str, Any]:
    check: Dict[str, Any] = {
        "check": "each_option_kept",
        "field": "groups",
        "options": ["control", "treated"],
    }
    if minus:
        check["minus"] = {"field": "qc", "options": ["exclude", "defer"]}
    return check


QC_CHECKS = [
    {"check": "reason_required", "field": "qc", "options": ["exclude"]},
    {"check": "no_default_for", "field": "qc", "options": ["exclude"]},
]


def no_default_for_unknown(ds: Dataset) -> Dict[str, Any]:
    return {"check": "no_default_for", "field": "groups", "items": ds.unknown}


def dashboard_context(ds: Dataset, *, plate: bool = True) -> Dict[str, Any]:
    """The full source → groups → direction → QC dashboard."""
    return {
        "view": "dashboard",
        "data_files": [
            ds.rel("SRC-A_author_counts.csv"),
            ds.rel("sample_metadata.csv"),
        ],
        "fields": [
            source_field(ds),
            groups_field(
                ds, view="plate_map" if plate else None, depends_on=["source"]
            ),
            direction_field(depends_on=["groups"]),
            qc_field(ds, depends_on=["groups"]),
        ],
        "checks": [groups_kept(minus=True), no_default_for_unknown(ds), *QC_CHECKS],
    }


# --------------------------------------------------------------------------
# Scripted supervisor (the chat model behind the real supervisor agent)
# --------------------------------------------------------------------------

Step = Union[AIMessage, Callable[[List[BaseMessage]], AIMessage]]


def ask(*calls: tuple) -> AIMessage:
    """One supervisor message making ask_user calls.

    Each call is ``(call_id, question, context)``; several calls in one
    message run in parallel (two pending questions).
    """
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "ask_user",
                "id": call_id,
                "args": {"question": question, "context": context},
            }
            for call_id, question, context in calls
        ],
    )


def call(name: str, call_id: str, **args: Any) -> AIMessage:
    """One supervisor message calling any tool."""
    return AIMessage(
        content="", tool_calls=[{"name": name, "id": call_id, "args": args}]
    )


def new_id(tag: str = "call") -> str:
    return f"{tag}_{uuid.uuid4().hex[:10]}"


class ScriptedSupervisor(BaseChatModel):
    """A chat model that replays fixed steps; after the last step it says DONE.

    ``seen`` records every tool result the model was shown, in order, so a
    test can check what the supervisor finally received ("raw" as shown,
    "content" parsed from inside the <tool_data> wrapper the supervisor's
    message path puts around tool output).
    """

    steps: List[Any] = []
    calls: int = 0
    seen: List[Dict[str, Any]] = []
    final_text: str = "Scripted supervisor: DONE"

    @property
    def _llm_type(self) -> str:
        return "scripted-supervisor"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedSupervisor":
        return self

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        known = {entry["tool_call_id"] for entry in self.seen}
        for message in messages:
            if message.type == "tool" and message.tool_call_id not in known:
                raw = (
                    message.content
                    if isinstance(message.content, str)
                    else str(message.content)
                )
                inner = re.sub(r"^<tool_data>(.*)</tool_data>$", r"\1", raw, flags=re.S)
                try:
                    content = json.loads(inner)
                except (TypeError, ValueError):
                    content = inner
                self.seen.append(
                    {
                        "tool_call_id": message.tool_call_id,
                        "content": content,
                        "raw": raw,
                    }
                )
        if self.calls < len(self.steps):
            step = self.steps[self.calls]
            reply = step(messages) if callable(step) else step
        else:
            reply = AIMessage(content=self.final_text)
        self.calls += 1
        return ChatResult(generations=[ChatGeneration(message=reply)])


def make_client(
    workspace: Path,
    steps: Sequence[Step],
    extra_tools: Sequence[Any] = (),
):
    """A real AgentClient whose supervisor model is scripted; no worker agents."""
    (workspace / "config.toml").write_text('enabled = ["__e2e_no_worker_agents__"]\n')
    model = ScriptedSupervisor(steps=list(steps), seen=[])
    with ExitStack() as stack:
        stack.enter_context(
            patch("lobster.agents.graph.create_llm", return_value=model)
        )
        if extra_tools:
            import lobster.agents.graph as graph_module

            original = graph_module._build_supervisor_tools

            def with_extra(*args: Any, **kwargs: Any):
                tools, names = original(*args, **kwargs)
                return [*tools, *extra_tools], names

            stack.enter_context(
                patch.object(graph_module, "_build_supervisor_tools", with_extra)
            )
        from lobster.core.client import AgentClient

        client = AgentClient(
            workspace_path=workspace,
            session_id=f"e2e_{uuid.uuid4().hex[:8]}",
            enable_reasoning=False,
        )
    client.e2e_model = model  # type: ignore[attr-defined]
    return client


def tool_results(client: Any) -> Dict[str, Any]:
    """Every tool result in the graph state, keyed by tool_call_id."""
    state = client.graph.get_state({"configurable": {"thread_id": client.session_id}})
    out = {}
    for message in state.values.get("messages", []):
        if message.type == "tool":
            try:
                out[message.tool_call_id] = json.loads(message.content)
            except (TypeError, ValueError):
                out[message.tool_call_id] = message.content
    return out


def pending_interrupts(client: Any) -> List[Any]:
    state = client.graph.get_state({"configurable": {"thread_id": client.session_id}})
    return [i for task in state.tasks for i in task.interrupts]


# --------------------------------------------------------------------------
# The TUI session
# --------------------------------------------------------------------------


@dataclass
class Message:
    direction: str  # "to_go" (bridge → binary) or "from_go"
    msg: Dict[str, Any]
    at: float
    pty_bytes: int = 0  # how much screen output there was when it passed


@dataclass
class Render:
    """One component_render the bridge sent to the binary."""

    index: int  # position in the to_go log
    msg_id: str
    component: str
    data: Dict[str, Any]
    fallback: str


class _Relay(threading.Thread):
    """Forwards JSON lines from one fd to another, recording each message."""

    def __init__(self, session: "TuiSession", src: int, dst: int, direction: str):
        super().__init__(daemon=True, name=f"e2e-relay-{direction}")
        self.session, self.src, self.dst, self.direction = session, src, dst, direction

    def run(self) -> None:
        buf = b""
        while True:
            try:
                chunk = os.read(self.src, 65536)
            except OSError:
                break
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                self.session._record(self.direction, line)
                try:
                    self.session._write(self.dst, line + b"\n", self.direction)
                except OSError:
                    return
        try:
            os.close(self.dst)
        except OSError:
            pass


class TuiSession:
    """The real binary on a pty, wired to the real bridge and event loop."""

    def __init__(
        self,
        binary: Path,
        client: Any,
        workspace: Path,
        rows: int = 40,
        cols: int = 120,
    ) -> None:
        self.binary, self.client, self.workspace = binary, client, workspace
        self.rows, self.cols = rows, cols
        self.screen = Screen(rows, cols)
        self.log: List[Message] = []
        self._cond = threading.Condition()
        self._write_locks = {"to_go": threading.Lock(), "from_go": threading.Lock()}
        self.proc: Optional[subprocess.Popen] = None
        self.bridge: Any = None
        self.loop_thread: Optional[threading.Thread] = None
        self.loop_error: Optional[BaseException] = None
        self.resumes: List[Any] = []  # every value client.resume_from_interrupt got
        self._inject_fd: Optional[int] = None
        self.out_bytes = 0  # screen output so far
        self.last_out = 0.0  # when the last screen output arrived

    # -- start / stop ------------------------------------------------------

    def start(self) -> "TuiSession":
        from lobster.cli_internal import go_tui_launcher as launcher

        master, slave = pty.openpty()
        self._winsize(slave, self.rows, self.cols)
        self.master = master

        # Exactly the launcher's pipes and command line (launch_go_tui_chat
        # §1), with a relay pipe on each side.
        p2g_r, p2g_w = os.pipe()  # relay → Go
        g2p_r, g2p_w = os.pipe()  # Go → relay
        b2r_r, b2r_w = os.pipe()  # bridge → relay
        r2b_r, r2b_w = os.pipe()  # relay → bridge
        cmd = [
            str(self.binary),
            "chat",
            "--proto-fd-in",
            str(p2g_r),
            "--proto-fd-out",
            str(g2p_w),
            "--inline",
        ]
        base_env = {
            k: v
            for k, v in os.environ.items()
            if not k.startswith("LOBSTER_TUI_") and not k.endswith("_API_KEY")
        }
        base_env.update(
            TERM="xterm-256color", COLUMNS=str(self.cols), LINES=str(self.rows)
        )
        child_env = launcher._prepare_go_tui_chat_env(
            base_env, workspace=str(self.workspace), no_intro=True
        )
        self.proc = subprocess.Popen(  # nosec B603 - test binary, argv list
            cmd,
            pass_fds=(p2g_r, g2p_w),
            stdin=slave,
            stdout=slave,
            stderr=subprocess.DEVNULL,
            text=False,
            preexec_fn=os.setsid,
            env=child_env,
        )
        os.close(p2g_r)
        os.close(g2p_w)
        os.close(slave)

        self._screen_thread = threading.Thread(
            target=self._pump_screen, daemon=True, name="e2e-pty"
        )
        self._screen_thread.start()
        _Relay(self, b2r_r, p2g_w, "to_go").start()
        _Relay(self, g2p_r, r2b_w, "from_go").start()
        self._inject_fd = r2b_w

        # The real bridge, on the relay's ends of the pipes.
        self.bridge = launcher._LightBridge(self.proc, b2r_w, r2b_r)
        handshake = self.bridge.recv_event(timeout=10.0)
        assert handshake and handshake.get("type") == "handshake", handshake
        payload = handshake.get("payload", {})
        assert payload.get("protocol_version") in launcher._SUPPORTED_PROTOCOL_VERSIONS
        # The launcher's startup messages (§3 and §7), then its event loop (§8).
        self.bridge.send("spinner", {"active": True, "label": "Initializing agents"})
        self.bridge.send("spinner", {"active": False})
        self.bridge.send("ready", {})

        self._wrap_resume()

        def loop() -> None:
            try:
                launcher._go_tui_event_loop(self.bridge, self.client)
            except BaseException as error:  # pragma: no cover - reported by tests
                self.loop_error = error

        self.loop_thread = threading.Thread(target=loop, daemon=True, name="e2e-loop")
        self.loop_thread.start()
        self.wait_screen("❯", timeout=WAIT)
        return self

    def _wrap_resume(self) -> None:
        """Record every resume value; a test may set ``rewrite_resume``."""
        original = self.client.resume_from_interrupt
        self.rewrite_resume: Optional[Callable[[Any], Any]] = None

        def resume(response: Any, *args: Any, **kwargs: Any):
            if self.rewrite_resume is not None:
                response = self.rewrite_resume(response)
            self.resumes.append(response)
            return original(response, *args, **kwargs)

        self.client.resume_from_interrupt = resume

    def close(self) -> None:
        """End the event loop, close the bridge (it kills the binary), reap."""
        try:
            if self.loop_thread is not None:
                deadline = time.time() + 10
                while self.loop_thread.is_alive() and time.time() < deadline:
                    # The protocol quit ends a pending question, then the loop.
                    self.bridge._events.put({"type": "quit", "payload": {}})
                    self.loop_thread.join(timeout=0.5)
        finally:
            if self.bridge is not None:
                self.bridge.close()
            proc = self.proc
            if proc is not None and proc.poll() is None:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
                proc.wait(timeout=5)
            for fd in (self.master, self._inject_fd):
                try:
                    if fd is not None:
                        os.close(fd)
                except OSError:
                    pass

    # -- pty ---------------------------------------------------------------

    @staticmethod
    def _winsize(fd: int, rows: int, cols: int) -> None:
        fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))

    def resize(self, rows: int, cols: int) -> None:
        """Change the pty size as a terminal window resize would."""
        self.rows, self.cols = rows, cols
        self._winsize(self.master, rows, cols)
        self.screen.resize(rows, cols)
        # The binary runs in its own session (os.setsid, as the launcher
        # starts it), so the kernel's SIGWINCH for the pty does not reach it;
        # deliver it the way a terminal would to the foreground process.
        assert self.proc is not None
        os.kill(self.proc.pid, signal.SIGWINCH)

    def _pump_screen(self) -> None:
        while True:
            try:
                ready, _, _ = select.select([self.master], [], [], 0.2)
            except (OSError, ValueError):
                return
            if not ready:
                if self.proc is not None and self.proc.poll() is not None:
                    return
                continue
            try:
                data = os.read(self.master, 65536)
            except OSError:
                return
            if not data:
                return
            self.screen.feed(data)
            with self._cond:
                self.out_bytes += len(data)
                self.last_out = time.time()
                self._cond.notify_all()

    def key(self, *names: str, pause: float = 0.0) -> None:
        """Press keys by name (see KEYS) or literal text."""
        for name in names:
            data = KEYS.get(name)
            if data is None:
                data = name.encode()
            os.write(self.master, data)
            if name == "esc":
                # A lone ESC is only an ESC once nothing follows it quickly.
                time.sleep(0.15)
            elif pause:
                time.sleep(pause)

    def type(self, text: str) -> None:
        os.write(self.master, text.encode())

    # -- protocol log ------------------------------------------------------

    def _record(self, direction: str, line: bytes) -> None:
        try:
            msg = json.loads(line)
        except ValueError:
            return
        with self._cond:
            self.log.append(Message(direction, msg, time.time(), self.out_bytes))
            self._cond.notify_all()

    def _write(self, fd: int, data: bytes, direction: str) -> None:
        with self._write_locks[direction]:
            os.write(fd, data)

    def inject(self, msg_type: str, payload: Dict[str, Any], msg_id: str = "") -> None:
        """Write a crafted message towards the bridge, as if the binary sent it."""
        msg: Dict[str, Any] = {"version": 1, "type": msg_type, "payload": payload}
        if msg_id:
            msg["id"] = msg_id
        assert self._inject_fd is not None
        self._write(self._inject_fd, (json.dumps(msg) + "\n").encode(), "from_go")

    def messages(self, direction: str, msg_type: Optional[str] = None) -> List[Dict]:
        with self._cond:
            return [
                m.msg
                for m in self.log
                if m.direction == direction
                and (msg_type is None or m.msg.get("type") == msg_type)
            ]

    def _wait(
        self,
        predicate: Callable[[], Any],
        timeout: float,
        what: str,
        poll: float = 0.2,
    ) -> Any:
        deadline = time.time() + timeout
        with self._cond:
            while True:
                value = predicate()
                if value:
                    return value
                left = deadline - time.time()
                if left <= 0:
                    raise AssertionError(
                        f"timed out after {timeout:.0f}s waiting for {what}\n"
                        f"--- screen ---\n{self.screen.text()}\n"
                        f"--- last protocol messages ---\n"
                        + "\n".join(
                            f"{m.direction}: {json.dumps(m.msg)[:300]}"
                            for m in self.log[-8:]
                        )
                    )
                self._cond.wait(min(left, poll))

    def renders(self) -> List[Render]:
        with self._cond:
            out = []
            for index, m in enumerate(self.log):
                if m.direction == "to_go" and m.msg.get("type") == "component_render":
                    payload = m.msg.get("payload", {})
                    out.append(
                        Render(
                            index=index,
                            msg_id=m.msg.get("id", ""),
                            component=payload.get("component", ""),
                            data=payload.get("data", {}),
                            fallback=payload.get("fallback_prompt", ""),
                        )
                    )
            return out

    def wait_render(self, number: int, timeout: float = WAIT) -> Render:
        """Wait for the ``number``-th component_render (1-based) to be drawn.

        The binary doesn't acknowledge a render, and a re-asked screen can
        look like the one before it, so: wait for screen output after the
        render was passed to the binary, then for the screen to be quiet
        (nothing else animates while a question is shown). Keys pressed after
        this reach the new component.
        """
        render = self._wait(
            lambda: len(self.renders()) >= number and self.renders()[number - 1],
            timeout,
            f"component_render #{number}",
        )
        offset, sent = self.log[render.index].pty_bytes, self.log[render.index].at
        # A frame taller than the terminal is repainted without pause
        # (E2E_REPORT #4), so continuous output for 1.5 s also counts.
        self._wait(
            lambda: self.out_bytes > offset
            and (time.time() - self.last_out > QUIET or time.time() - sent > 1.5),
            timeout,
            f"the binary to draw component_render #{number}",
            poll=0.03,
        )
        return render

    def responses(self) -> List[Dict[str, Any]]:
        return self.messages("from_go", "component_response")

    def wait_response(self, msg_id: str, timeout: float = WAIT) -> Dict[str, Any]:
        def found():
            for msg in self.responses():
                payload = msg.get("payload") or {}
                if (payload.get("id") or msg.get("id")) == msg_id:
                    return msg
            return None

        return self._wait(found, timeout, f"component_response for {msg_id}")

    def turn_ends(self) -> List[str]:
        return [
            m["payload"].get("summary", "")
            for m in self.messages("to_go", "done")
            if m["payload"].get("summary") != "interrupt"
        ]

    def wait_turn_end(self, number: int = 1, timeout: float = WAIT) -> str:
        return self._wait(
            lambda: len(self.turn_ends()) >= number and [self.turn_ends()[number - 1]],
            timeout,
            f"end of turn #{number}",
        )[0]

    def wait_screen(
        self, needle: Union[str, Callable[[str], bool]], timeout: float = WAIT
    ) -> str:
        def check():
            text = self.flat()
            ok = needle(text) if callable(needle) else _norm(needle) in text
            return text if ok else None

        return self._wait(check, timeout, f"screen text {needle!r}")

    def output_rate(self, seconds: float = 1.0) -> float:
        """Screen output in bytes per second over the next ``seconds``."""
        before = self.out_bytes
        time.sleep(seconds)
        return (self.out_bytes - before) / seconds

    def flat(self) -> str:
        """The screen as one line of words (wrapping joined, spacing normalised)."""
        return _norm(self.screen.text())

    # -- a chat turn -------------------------------------------------------

    def ask_supervisor(self, text: str = "run the scripted review") -> None:
        """Type a chat message and press Enter, as the user would."""
        before = len(self.messages("from_go", "input"))
        self.type(text)
        self.key("enter")
        self._wait(
            lambda: len(self.messages("from_go", "input")) > before,
            WAIT,
            "the chat input to reach the bridge",
        )


def _norm(text: str) -> str:
    """One line of words. A word wrapped at its hyphen ("Repository-" then
    "generated" on the next line) is joined back without a space."""
    text = re.sub(r"(?<=\w)-\n\s*(?=\w)", "-", text)
    return " ".join(text.split())


GROUP_CHOICES = ("control", "treated", LEAVE_OUT)
QC_OPTIONS = ("retain", "defer", "exclude")


def press_in_plate(
    tui: TuiSession,
    ids: Sequence[str],
    assignments: Dict[str, str],
    choices: Sequence[str] = GROUP_CHOICES,
    columns: int = 12,
    submit: bool = True,
    start: Optional[str] = None,
) -> None:
    """Move the plate cursor to each sample and press its choice's number.

    ``ids`` is the well order (row by row from A1, as the adapter lays them
    out); the cursor starts on A1, or on ``start`` (a refused enter moves it
    to the first sample it names).
    ``assignments`` maps sample id → choice id, or "0" for unresolved.
    Keys are processed in order by the binary, so no wait is needed between
    them; the last assignment's message is awaited before Enter.
    """
    row, col = divmod(list(ids).index(start), columns) if start else (0, 0)
    last = None
    for sid, choice in assignments.items():
        index = list(ids).index(sid)
        target_row, target_col = divmod(index, columns)
        keys: List[str] = []
        keys += (
            ["down"] * (target_row - row)
            if target_row > row
            else ["up"] * (row - target_row)
        )
        keys += (
            ["right"] * (target_col - col)
            if target_col > col
            else ["left"] * (col - target_col)
        )
        digit = "0" if choice == "0" else str(list(choices).index(choice) + 1)
        tui.key(*keys, digit)
        if choice == LEAVE_OUT:  # the reason box opens on the well
            tui.type(LEAVE_OUT_REASON)
            tui.key("enter")
        row, col = target_row, target_col
        last = (sid, choice)
    if last:
        if last[1] == LEAVE_OUT:
            tui.wait_screen(f"{last[0]}: leave out, reason saved.")
        else:
            tui.wait_screen(f"{last[0]} set to ")  # the option's label follows
    if submit:
        tui.key("enter")


def answer_rows(
    tui: TuiSession,
    rows: Sequence[tuple],
    options: Sequence[str] = QC_OPTIONS,
    submit: bool = True,
) -> None:
    """Answer a choice_list from its first row: each row is (option, reason).

    ``reason`` is typed into the reason box the option opens (None: no box).
    """
    for number, (option, reason) in enumerate(rows):
        if number:
            tui.key("down")
        tui.key(str(list(options).index(option) + 1))
        if reason is not None:
            tui.type(reason)
            tui.key("enter")
    if submit:
        tui.key("enter")


def last_tool_result(
    client: Any, call_id: str, timeout: float = WAIT
) -> Dict[str, Any]:
    """Poll the graph state for one tool call's result."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        results = tool_results(client)
        if call_id in results:
            return results[call_id]
        time.sleep(0.05)
    raise AssertionError(f"no tool result for {call_id}; have {tool_results(client)}")
