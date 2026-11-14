"""One worker tool runs at a time in the process.

Tool calls of one model message run on separate threads (the ToolNode's
thread pool, or one executor thread each under ainvoke). Native code that is
not thread-safe, such as Numba's parallel kernels behind neighbors and UMAP,
aborts the whole process when two calls reach it at once. Hand-off tools stay
outside the lock: the expert they start needs it for its own tools.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from lobster.agents.graph import _create_agent_tool, _one_tool_at_a_time
from lobster.tools._async_compat import enable_sync_fallback

REPO = Path(__file__).resolve().parents[3]


class Fake(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def calls(*names, prefix=""):
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": name,
                "args": (
                    {"task_description": "Your task is to work."}
                    if name.startswith("handoff")
                    else {}
                ),
                "id": f"{prefix}{name}",
            }
            for name in names
        ],
    )


class Recorder:
    """Tools that record when they run, each taking a little time."""

    def __init__(self):
        self.spans = {}
        self.waited = []

    def tool(self, name, seconds=0.3):
        @tool(name)
        def work() -> str:
            """One analysis step."""
            start = time.monotonic()
            time.sleep(seconds)
            self.spans[name] = (start, time.monotonic())
            return f"{name} done"

        return work

    def async_tool(self, name):
        """A tool that waits on I/O, as lobster's coroutine tools do."""

        @tool(name)
        async def wait() -> str:
            """Wait on I/O."""
            await asyncio.sleep(0.2)
            self.waited.append(name)
            return f"{name} done"

        return enable_sync_fallback(wait)

    def overlapping(self):
        spans = sorted(self.spans.values())
        return [(a, b) for a, b in zip(spans, spans[1:]) if b[0] < a[1]]


def build(rec, lock=True):
    """A supervisor that, in one message, hands off and runs a sync and an
    async tool of its own; the expert then does the same with its tools."""
    expert = create_react_agent(
        Fake(
            responses=[
                calls("expert_a", "expert_b", "expert_io", prefix="e"),
                AIMessage("ok"),
            ]
        ),
        [rec.tool("expert_a"), rec.tool("expert_b"), rec.async_tool("expert_io")],
    )
    handoff = _create_agent_tool("expert", expert, "handoff_to_expert", "Delegate.")
    supervisor = create_react_agent(
        Fake(
            responses=[
                calls(
                    "handoff_to_expert", "supervisor_own", "supervisor_io", prefix="s"
                ),
                AIMessage("done"),
            ]
        ),
        [handoff, rec.tool("supervisor_own"), rec.async_tool("supervisor_io")],
    )
    if lock:
        _one_tool_at_a_time(expert)
        _one_tool_at_a_time(supervisor)
    return supervisor


def run_bounded(fn, seconds=30):
    """Run fn on a thread; a deadlock shows as a timeout, not a hung suite."""
    out = {}
    worker = threading.Thread(target=lambda: out.update(result=fn()), daemon=True)
    worker.start()
    worker.join(seconds)
    assert not worker.is_alive(), "deadlock: the run did not finish"
    return out["result"]


@pytest.mark.parametrize("path", ["invoke", "ainvoke"])
def test_tools_of_one_message_never_overlap_and_a_handoff_completes(path):
    rec = Recorder()
    supervisor = build(rec)
    state = {"messages": [("user", "go")]}
    if path == "invoke":
        result = run_bounded(lambda: supervisor.invoke(state))
    else:
        result = run_bounded(lambda: asyncio.run(supervisor.ainvoke(state)))

    assert result["messages"][-1].content == "done"
    assert set(rec.spans) == {"expert_a", "expert_b", "supervisor_own"}
    assert sorted(rec.waited) == ["expert_io", "supervisor_io"]
    assert rec.overlapping() == []


def test_without_the_lock_the_same_tools_overlap():
    # Guards the test above: the overlap it rules out does happen.
    rec = Recorder()
    run_bounded(lambda: build(rec, lock=False).invoke({"messages": [("user", "go")]}))
    assert rec.overlapping() != []


def test_handoffs_and_tools_with_a_coroutine_stay_outside():
    rec = Recorder()
    ask = rec.async_tool("ask")
    ask_func = ask.func
    expert = create_react_agent(Fake(responses=[AIMessage("ok")]), [rec.tool("x")])
    handoff = _create_agent_tool("expert", expert, "handoff_to_expert", "Delegate.")
    work = rec.tool("own")
    plain = work.func
    agent = create_react_agent(Fake(responses=[]), [handoff, ask, work])
    handoff_func = handoff.func

    _one_tool_at_a_time(agent)
    _one_tool_at_a_time(agent)  # a second pass does not lock twice

    assert handoff.func is handoff_func and ask.func is ask_func
    assert work.func is not plain and work.func.__wrapped__ is plain


SCANPY_PAIR = """
import sys, numpy as np, anndata as ad, scanpy as sc
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from lobster.agents.graph import _one_tool_at_a_time

class Fake(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self

def embed(name, seed):
    @tool(name)
    def run() -> str:
        \"\"\"Neighbors and UMAP on one dataset.\"\"\"
        rng = np.random.default_rng(seed)
        a = ad.AnnData(np.zeros((3000, 10), dtype=np.float32))
        a.obsm["X_pca"] = rng.normal(size=(3000, 30)).astype(np.float32)
        sc.pp.neighbors(a, use_rep="X_pca", n_neighbors=15)
        sc.tl.umap(a)
        return "ok"
    return run

two = AIMessage(content="", tool_calls=[
    {"name": "embed_a", "args": {}, "id": "a"},
    {"name": "embed_b", "args": {}, "id": "b"},
])
agent = create_react_agent(Fake(responses=[two, AIMessage("done")]),
                           [embed("embed_a", 0), embed("embed_b", 1)])
if sys.argv[1] == "lock":
    _one_tool_at_a_time(agent)
agent.invoke({"messages": [("user", "go")]})
print("survived")
"""


def run_scanpy_pair(tmp: Path, mode: str) -> subprocess.CompletedProcess:
    script = tmp / "scanpy_pair.py"
    script.write_text(SCANPY_PAIR)
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": str(tmp),
        "PYTHONPATH": str(REPO),
        "NUMBA_CACHE_DIR": str(tmp / "numba"),
    }
    return subprocess.run(
        [sys.executable, str(script), mode],
        env=env,
        cwd=tmp,
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_two_numba_tools_in_one_message_do_not_abort_the_process(tmp_path):
    pytest.importorskip("umap")
    pytest.importorskip("scanpy")
    done = run_scanpy_pair(tmp_path, "lock")
    assert done.returncode == 0, done.stderr[-2000:]
    assert "survived" in done.stdout
