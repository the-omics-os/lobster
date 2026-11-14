"""``lobster mcp --stdio`` contract, spoken over a pipe the way an MCP host does.

Host rules checked here were read from AIPOCH Open Science @8a50f6a
(``packages/connector-mcp-client/src/client-manager.ts`` ``unwrapToolResult``)
and from ``@modelcontextprotocol/sdk`` 1.31.0 (``types.js``
``LATEST_PROTOCOL_VERSION`` / ``SUPPORTED_PROTOCOL_VERSIONS``, read on unpkg).
"""

from __future__ import annotations

import ast
import json
import queue
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from lobster.cli_internal import mcp_contract, mcp_inspect
from lobster.cli_internal.commands.light.mcp_commands import mcp_command

SDK_LATEST = "2025-11-25"

_SERVER = (
    "import sys\n"
    "from lobster.cli_internal.mcp_stdio import FakeRunner, run_mcp_stdio\n"
    "code = run_mcp_stdio(runner=FakeRunner())\n"
    "engine = sorted(m for m in sys.modules\n"
    "    if m.startswith(('lobster.core', 'lobster.agents')))\n"
    "print('ENGINE_MODULES=' + ','.join(engine), file=sys.stderr)\n"
    "sys.exit(code)\n"
)


class Host:
    """A minimal MCP client: newline-delimited JSON-RPC over the child's pipes."""

    def __init__(self, argv=(sys.executable, "-c", _SERVER)):
        self.proc = subprocess.Popen(
            list(argv),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self._lines: "queue.Queue[str]" = queue.Queue()
        threading.Thread(target=self._pump, daemon=True).start()
        self._next_id = 0

    def _pump(self):
        for line in self.proc.stdout:
            self._lines.put(line)

    def send(self, message: dict) -> None:
        self.proc.stdin.write(json.dumps(message) + "\n")
        self.proc.stdin.flush()

    def read(self) -> dict:
        return json.loads(self._lines.get(timeout=30))

    def request(self, method: str, params: dict | None = None) -> dict:
        self._next_id += 1
        message = {"jsonrpc": "2.0", "id": self._next_id, "method": method}
        if params is not None:
            message["params"] = params
        self.send(message)
        reply = self.read()
        assert reply["id"] == self._next_id
        return reply

    def initialize(self, version: str = SDK_LATEST) -> dict:
        reply = self.request(
            "initialize",
            {
                "protocolVersion": version,
                "capabilities": {},
                "clientInfo": {"name": "test-host", "version": "0"},
            },
        )
        self.send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        return reply

    def call(self, name: str, arguments: dict) -> dict:
        return self.request("tools/call", {"name": name, "arguments": arguments})[
            "result"
        ]

    def close(self) -> tuple[int, str]:
        self.proc.stdin.close()
        code = self.proc.wait(timeout=30)
        return code, self.proc.stderr.read()


@pytest.fixture
def host():
    h = Host()
    yield h
    if h.proc.poll() is None:
        h.proc.kill()
        h.proc.wait()


def unwrap(result: dict):
    """Open Science's ``unwrapToolResult`` rule (client-manager.ts @8a50f6a)."""
    content = result.get("content")
    first = content[0] if isinstance(content, list) and content else None
    text = (
        first.get("text")
        if isinstance(first, dict) and first.get("type") == "text"
        else None
    )
    if result.get("isError"):
        raise RuntimeError(text or "MCP tool call failed")
    if (
        isinstance(text, str)
        and len(content) == 1
        and set(result) <= {"content", "isError"}
        and set(first) <= {"type", "text"}
    ):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"text": text}
    return result


def test_initialize_list_and_fake_run_over_the_pipe(host, tmp_path):
    init = host.initialize()["result"]
    assert init["protocolVersion"] == SDK_LATEST
    assert init["serverInfo"]["name"] == "lobster"
    assert "tools" in init["capabilities"]
    assert init["instructions"] == mcp_contract.INSTRUCTIONS

    assert host.request("ping")["result"] == {}
    tools = host.request("tools/list")["result"]["tools"]
    assert [t["name"] for t in tools] == [
        "lobster_run",
        "lobster_status",
        "lobster_answer",
        "lobster_cancel",
        "lobster_continue",
        "lobster_inspect",
    ]
    for tool in tools:
        assert set(tool) == {"name", "description", "inputSchema", "annotations"}
        assert tool["inputSchema"]["type"] == "object"
    # By design: run is not claimed non-destructive.
    assert tools[0]["annotations"] == {"readOnlyHint": False, "openWorldHint": True}
    assert tools[1]["annotations"] == {
        "readOnlyHint": True,
        "idempotentHint": True,
        "openWorldHint": False,
    }
    # An answer starts the next part of the run: not read-only, not repeatable.
    assert tools[2]["annotations"] == {
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False,
    }
    answer_schema = tools[2]["inputSchema"]
    assert answer_schema["required"] == [
        "run_id",
        "decision_id",
        "request_version",
        "answer",
        "source",
    ]
    properties = answer_schema["properties"]
    assert properties["source"]["enum"] == ["human", "model"]
    assert "default" not in properties["source"]  # the host says who answered
    assert properties["request_version"]["type"] == "integer"
    assert set(properties["answer"]["type"]) == {
        "string",
        "number",
        "boolean",
        "object",
    }
    assert properties["comment"]["type"] == "string"
    assert tools[3]["annotations"] == {
        "readOnlyHint": False,
        "destructiveHint": True,
        "idempotentHint": True,
        "openWorldHint": False,
    }
    assert tools[3]["inputSchema"]["required"] == ["run_id"]
    run_schema = tools[0]["inputSchema"]
    assert run_schema["required"] == ["input_path", "output_dir", "analysis"]
    assert run_schema["properties"]["analysis"]["enum"][0] == "scrna_qc"

    out = tmp_path / "out"
    source = tmp_path / "pbmc.h5ad"
    source.write_bytes(b"")
    started = host.call(
        "lobster_run",
        {
            "input_path": str(source),
            "output_dir": str(out),
            "analysis": "scrna_qc",
        },
    )
    assert set(started) == {"content"}  # no structuredContent, no isError
    run = unwrap(started)
    assert run["status"] == "running" and run["run_id"]

    first = unwrap(host.call("lobster_status", {"run_id": run["run_id"]}))
    assert first["status"] == "running"
    assert first["poll_after_s"] == mcp_contract.POLL_AFTER_S
    assert isinstance(first["elapsed_s"], int)
    done = unwrap(host.call("lobster_status", {"run_id": run["run_id"]}))
    assert done["status"] == "completed" and isinstance(done["elapsed_s"], int)
    nothing_to_answer = host.call(
        "lobster_answer",
        {
            "run_id": run["run_id"],
            "decision_id": "dec_x",
            "request_version": 1,
            "answer": "x",
            "source": "model",
        },
    )
    assert nothing_to_answer["isError"] is True
    assert nothing_to_answer["content"][0]["text"] == (
        f"not_waiting: run {run['run_id']} is completed; there is no question "
        "to answer."
    )
    for key in ("notebook_path", "result_path"):
        path = Path(done[key])
        assert path.is_absolute() and path.parent == out and path.exists()

    # A folder that holds a run is not reused; a new one is fine.
    reused = host.call(
        "lobster_run",
        {"input_path": str(source), "output_dir": str(out), "analysis": "scrna_qc"},
    )
    assert reused["isError"] is True
    assert reused["content"][0]["text"].startswith("output_dir_in_use: ")
    other = unwrap(
        host.call(
            "lobster_run",
            {
                "input_path": str(source),
                "output_dir": str(tmp_path / "out2"),
                "analysis": "scrna_qc",
            },
        )
    )
    assert other["run_id"] != run["run_id"]

    assert host.close()[0] == 0


@pytest.mark.parametrize(
    "offered, answered",
    [
        (SDK_LATEST, SDK_LATEST),
        ("2025-06-18", "2025-06-18"),
        ("1999-01-01", SDK_LATEST),
    ],
)
def test_protocol_version_negotiation(host, offered, answered):
    assert host.initialize(offered)["result"]["protocolVersion"] == answered
    assert host.request("ping")["result"] == {}
    assert host.request("tools/list")["result"]["tools"]
    code, stderr = host.close()
    assert code == 0
    # The server module answers initialize, ping and tools/list without the
    # engine (60 s host cap). The `lobster mcp` entry itself imports lobster.cli,
    # which loads lobster.core first; its latency is checked separately below.
    assert "ENGINE_MODULES=\n" in stderr


def test_discover_probe_before_initialize_falls_back(host):
    """MCP 2026-07-28 clients (Claude Code v2 runtime, Codex behind a flag) probe
    ``server/discover`` before ``initialize``; any JSON-RPC error sends them to
    the ``initialize`` handshake. This server never speaks 2026-07-28."""
    host.send(
        {
            "jsonrpc": "2.0",
            "id": "server-discover-probe-1",
            "method": "server/discover",
            "params": {},
        }
    )
    probe = host.read()
    assert probe["id"] == "server-discover-probe-1"
    assert probe["error"]["code"] == -32601 and "result" not in probe

    init = host.initialize("2026-07-28")["result"]
    assert init["protocolVersion"] == SDK_LATEST
    assert host.request("tools/list")["result"]["tools"]
    assert host.close()[0] == 0


def test_errors(host):
    host.initialize()
    unknown = host.request("tools/call", {"name": "lobster_nope", "arguments": {}})
    assert unknown["error"]["code"] == -32602

    assert host.request("no/such/method")["error"]["code"] == -32601

    bad = host.call(
        "lobster_run", {"input_path": "/a", "output_dir": "/b", "analysis": "x"}
    )
    assert bad["isError"] is True and "scrna_qc" in bad["content"][0]["text"]
    with pytest.raises(RuntimeError, match="Unknown run_id"):
        unwrap(host.call("lobster_status", {"run_id": "nope"}))

    host.proc.stdin.write("{not json\n")
    host.proc.stdin.flush()
    assert host.read()["error"]["code"] == -32700


def test_descriptions_are_the_skill_contract():
    by_name = {t["name"]: t for t in mcp_contract.list_tools()}
    for name in ("lobster_run", "lobster_status"):
        assert mcp_contract.NO_REWRITE in by_name[name]["description"]
    cancel = by_name["lobster_cancel"]["description"]
    assert "Use only when the user asks to stop." in cancel
    assert "the run ends after the step in progress returns" in cancel
    assert mcp_contract.ABSOLUTE_PATHS in by_name["lobster_run"]["description"]
    answer = by_name["lobster_answer"]["description"]
    for needle in (
        "lobster_status",
        "decision_id",
        "request_version",
        "source human if the user gave the answer in this conversation",
        "source model if you decided",
        "ask them",
        "result.json",
    ):
        assert needle in answer
    for name in ("lobster_run", "lobster_status"):
        assert mcp_contract.DECISIONS in by_name[name]["description"]
    assert "needs_decision" in by_name["lobster_status"]["description"]
    assert "decision_abandoned" in by_name["lobster_cancel"]["description"]

    # A host waits between status calls instead of watching the output folder.
    wait = f"{mcp_contract.POLL_AFTER_S} seconds"
    for name in ("lobster_run", "lobster_status"):
        assert wait in by_name[name]["description"]
        assert "poll_after_s" in by_name[name]["description"]
    status = by_name["lobster_status"]["description"]
    for field in ("elapsed_s", "phase", "step", "steps_done"):
        assert field in status
    assert "output folder" in status

    text = " ".join(
        [mcp_contract.CONNECTOR_DESCRIPTION, mcp_contract.INSTRUCTIONS]
        + [t["description"] for t in by_name.values()]
    ).lower()
    for word in ("resum", "reproducib", "hardened", "sandbox", "state-of-the-art"):
        assert word not in text
    # No claim about who answered: the host tells Lobster, Lobster only repeats it.
    ours = " ".join(
        [mcp_contract.DECISIONS, mcp_contract.INSTRUCTIONS, answer]
        + [by_name["lobster_cancel"]["description"]]
    ).lower()
    for word in ("verif", "prove", "proof", "certif", "reproducib", "sandbox"):
        assert word not in ours

    assert mcp_contract.CONNECTOR_DESCRIPTION.startswith("Use when")
    assert len(mcp_contract.CONNECTOR_DESCRIPTION) <= 2000
    assert re.fullmatch(r"[a-z0-9-]+", mcp_contract.SERVER_NAME)


def test_status_text_describes_elapsed_files_and_the_open_question():
    status = {t["name"]: t for t in mcp_contract.list_tools()}["lobster_status"][
        "description"
    ]
    # elapsed_s is monotonic time since the run was accepted, not a wall clock.
    assert "wall" not in status.lower()
    assert "A finished status carries elapsed_s, the seconds since the run was" in (
        status.replace("\n", " ")
    )
    assert (
        "If open_question is true, the summary ends with a question nobody was "
        "asked during the run; treat it as a suggestion, and if the user wants "
        "it, follow up with lobster_continue."
    ) in status
    for field in ("files", "files_truncated", "files_written_error", "path"):
        assert field in status
    assert "relative" not in status  # the paths are the published ones
    assert "leaves out response and files_written" in status


def test_status_text_describes_metrics_modalities_and_warnings():
    status = {t["name"]: t for t in mcp_contract.list_tools()}["lobster_status"][
        "description"
    ]
    for field in (
        "metrics",
        "cells_before",
        "cells_after",
        "genes_before",
        "genes_after",
        "qc",
        "tokens",
        "modality_details",
        "stage",
        "parent",
        "n_obs",
        "n_vars",
        "warnings",
        "code",
        "message",
        "result, notebook, data or plots",
        "plots_truncated",
        "plots_total",
    ):
        assert field in status, field
    assert "Tell the user about every warning." in status
    assert "result.json also holds the steps and the request" in status


def test_run_text_describes_the_output_folder_and_the_instruction():
    run = {t["name"]: t for t in mcp_contract.list_tools()}["lobster_run"]
    text = run["description"]
    assert mcp_contract.OUTPUT in text and mcp_contract.INSTRUCTION in text
    for needle in (
        "output_dir_in_use",
        "result.json, the notebook and the saved .h5ad files",
        "output_dir/.lobster_workspace",
        "500 characters at most",
        "as you reported it",
    ):
        assert needle in text, needle
    assert "overwrite" not in text
    prop = run["inputSchema"]["properties"]["instruction"]
    assert prop["type"] == "string" and prop["maxLength"] == 500
    assert "instruction" not in run["inputSchema"]["required"]


def test_the_new_text_makes_no_claim():
    text = " ".join(
        [
            mcp_contract.FINISHED,
            mcp_contract.OUTPUT,
            mcp_contract.INSTRUCTION,
            mcp_contract.ANALYSIS_PROMPTS["scrna_qc"],
            {t["name"]: t for t in mcp_contract.list_tools()}["lobster_run"][
                "inputSchema"
            ]["properties"]["instruction"]["description"],
        ]
    ).lower()
    for word in ("verif", "prove", "proof", "guarantee", "reproducib", "sandbox"):
        assert word not in text, word


CLAIMS = re.compile(
    r"\b(verif\w*|reproducib\w*|sandbox\w*|proves?|proof|guarantee\w*|certif\w*)\b",
    re.IGNORECASE,
)
# Notebook cells are written for the reader of the notebook, who is told what
# Lobster cannot check; every other string is a text the host or the model gets.
NOTEBOOK_CELLS = {"instruction_ir"}


def _strings(module):
    """Every string literal of *module* that is not a docstring or notebook cell."""
    tree = ast.parse(Path(module.__file__).read_text())
    skip = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module)):
            body = node.body
            if body and isinstance(body[0], ast.Expr):
                skip.add(id(body[0].value))
        if isinstance(node, ast.FunctionDef) and node.name in NOTEBOOK_CELLS:
            skip.update(id(n) for n in ast.walk(node))
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in skip
        ):
            yield node.value


def test_no_text_the_host_or_the_model_gets_makes_a_claim():
    from lobster.cli_internal import (
        mcp_decision,
        mcp_inspect,
        mcp_record,
        mcp_stdio,
    )

    found = [
        (module.__name__, text)
        for module in (mcp_contract, mcp_record, mcp_stdio, mcp_decision, mcp_inspect)
        for text in _strings(module)
        if CLAIMS.search(text)
    ]
    assert found == []
    connector = json.loads(
        (
            Path(mcp_contract.__file__).parents[1]
            / "integrations"
            / "open-science.connector.json"
        ).read_text()
    )
    whole = [
        connector["description"],
        mcp_contract.INSTRUCTIONS,
        mcp_record._INSTRUCTION_HEADER,
        mcp_record._CONTINUE_LEAD,
        mcp_record._CONTINUE_RULES,
        mcp_record._INSTRUCTION_RULES,
        json.dumps(mcp_contract.list_tools()),
    ]
    assert [t for t in whole if CLAIMS.search(t)] == []


def test_the_texts_say_another_process_can_hold_the_folder():
    by_name = {t["name"]: t for t in mcp_contract.list_tools()}
    assert (
        "another Lobster process is using it" in by_name["lobster_run"]["description"]
    )
    assert (
        "A folder that another Lobster process is using is refused (output_dir_in_use)"
        in by_name["lobster_continue"]["description"]
    )


def test_the_inspect_text_names_the_row_limit():
    text = {t["name"]: t["description"] for t in mcp_contract.list_tools()}[
        "lobster_inspect"
    ]
    assert f"{mcp_inspect.MAX_COLUMN_ROWS:,} rows" in text


def test_the_status_text_says_tokens_are_the_turns_own():
    text = {t["name"]: t["description"] for t in mcp_contract.list_tools()}[
        "lobster_status"
    ]
    assert "tokens has scope turn" in text and "result.token_usage" in text
    assert "cost_complete false means a call was unpriced" in text
    assert "latest_cost_usd is the last model call, not the turn" in text


def test_the_texts_say_where_answers_are_listed():
    by_name = {t["name"]: t["description"] for t in mcp_contract.list_tools()}
    assert "decisions lists each answer (question, answer, source" in (
        by_name["lobster_status"]
    )
    assert "decisions_note" in by_name["lobster_status"]
    assert "model_source is replay when the model was a scripted" in (
        by_name["lobster_status"]
    )
    assert "the completed status and result.json note source" in (
        by_name["lobster_answer"]
    )


def test_the_continue_text_explains_unfinished_instructions():
    text = {t["name"]: t for t in mcp_contract.list_tools()}["lobster_continue"][
        "description"
    ]
    assert "lists it in unfinished_instructions (instruction, outcome and" in text
    assert "not carried out as asked" in text
    assert "A stop takes effect after the step that is running" in text


def test_the_continue_text_names_only_its_own_parameters():
    tool = {t["name"]: t for t in mcp_contract.list_tools()}["lobster_continue"]
    assert "input_path" not in tool["description"]
    assert "output_dir" in tool["description"]


RECIPE = mcp_contract.ANALYSIS_PROMPTS["scrna_qc"]


def test_the_recipe_has_one_placeholder_and_the_five_steps_in_order():
    assert re.findall(r"\{[^}]*\}", RECIPE) == ["{input_path}"]
    assert "{" not in RECIPE.replace("{input_path}", "")
    assert "}" not in RECIPE.replace("{input_path}", "")
    steps = [m.start() for m in re.finditer(r"(?m)^[1-5]\. ", RECIPE)]
    assert len(steps) == 5 and steps == sorted(steps)
    positions = [
        RECIPE.index(name)
        for name in (
            "assess_data_quality",
            "create_qc_plots",
            "filter_and_normalize",
            "Reply in this order",
        )
    ]
    assert positions == sorted(positions)
    assert RECIPE.startswith("Run standard quality control on the single-cell")
    assert RECIPE.format(input_path="/a/{b}.h5ad").count("/a/{b}.h5ad") == 1


def test_the_recipe_never_picks_a_ribosomal_number():
    # Digits are only step numbers; the ribosomal value comes from the tool
    # default or from the host's instruction, never from this text.
    without_steps = re.sub(r"(?m)^\d\. |\bstep \d", "", RECIPE)
    assert not re.search(r"\d", without_steps)
    assert "Never choose a ribosomal threshold yourself." in RECIPE
    for name in ("max_ribo_percent", "min_genes_per_cell", "max_mito_percent"):
        assert name in RECIPE


def test_the_recipe_quotes_the_paragraph_the_filter_prints():
    assert (
        'begins "**QC flags carried into this result:**", quoted unchanged'
    ) in RECIPE


def test_instructions_are_sentences_of_the_connector_description():
    instructions = mcp_contract.INSTRUCTIONS
    assert len(instructions) <= 512
    for sentence in re.split(r"(?<=\.) ", instructions):
        assert sentence in mcp_contract.CONNECTOR_DESCRIPTION
    assert "needs_decision" in instructions
    assert "lobster_answer" in instructions
    assert "source human if the user gave the answer in this conversation" in (
        instructions
    )
    assert "ask" in instructions and "Runs without asking questions" not in instructions
    assert "One run at a time" in instructions
    assert "lobster_cancel" in instructions
    assert f"every {mcp_contract.POLL_AFTER_S} s" in instructions
    assert "new output_dir" in instructions and "read the warnings" in instructions
    assert "lobster_continue" in instructions
    connector = mcp_contract.CONNECTOR_DESCRIPTION
    assert "lobster_continue" in connector and "lobster_inspect" in connector


def test_lobster_mcp_stdio_command_answers_initialize():
    """The real CLI entry: nothing but JSON-RPC on stdout, well inside 60 s."""
    host = Host((sys.executable, "-m", "lobster", "mcp", "--stdio"))
    try:
        started = time.monotonic()
        init = host.initialize()["result"]
        elapsed = time.monotonic() - started
        assert init["serverInfo"]["name"] == "lobster"
        tools = host.request("tools/list")["result"]["tools"]
        assert [t["name"] for t in tools] == [
            "lobster_run",
            "lobster_status",
            "lobster_answer",
            "lobster_cancel",
            "lobster_continue",
            "lobster_inspect",
        ]
        assert elapsed < 45, f"initialize took {elapsed:.1f}s"
        assert host.close()[0] == 0
    finally:
        if host.proc.poll() is None:
            host.proc.kill()
            host.proc.wait()


def test_lobster_mcp_without_stdio_exits_non_zero():
    result = subprocess.run(
        [sys.executable, "-m", "lobster", "mcp"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode != 0 and result.stdout == ""
    assert "--stdio" in result.stderr


def test_mcp_without_stdio_exits_non_zero():
    app = typer.Typer()
    app.command(name="mcp")(mcp_command)
    app.command(name="noop")(lambda: None)  # keep "mcp" a named subcommand
    result = CliRunner().invoke(app, ["mcp"])
    assert result.exit_code != 0
