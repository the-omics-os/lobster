"""``lobster_docs``: the public Omics-OS docs, searched or read through the MCP
server. The fetching, caching and framing live in ``mcp_docs`` (its own tests);
these check the tool around it: one of query or slug, bounds, the notice passed
through, one-line refusals, and a reply that never holds the other tools."""

from __future__ import annotations

import json
import threading

import pytest

from lobster.cli_internal import mcp_contract, mcp_docs
from lobster.cli_internal.mcp_stdio import McpServer, ThreadRunner
from tests.unit.mcp.test_run import call

HIT = {
    "slug": "agents/genomics",
    "title": "Genomics",
    "description": "VCF and GWAS",
    "url": "https://docs.omics-os.com/docs/agents/genomics",
    "kind": "reference",
    "matched": ["title"],
    "cached_at": None,
}
PAGE = {
    "slug": "agents/genomics",
    "title": "Genomics",
    "kind": "reference",
    "source_url": "https://docs.omics-os.com/raw/docs/agents/genomics.md",
    "page": 2,
    "pages": 3,
    "text": "## Variant calling\n...",
    "cached_at": None,
    "notice": mcp_docs.NOTICE,
}


@pytest.fixture
def docs(monkeypatch):
    calls = []

    def search(query, limit=5):
        calls.append(("search", query, limit))
        return {"results": [HIT], "notice": mcp_docs.NOTICE}

    def read(slug, page=1):
        calls.append(("read", slug, page))
        return PAGE

    monkeypatch.setattr(mcp_docs, "search", search)
    monkeypatch.setattr(mcp_docs, "read", read)
    return calls


def tool():
    return {t["name"]: t for t in mcp_contract.list_tools()}["lobster_docs"]


def test_the_tool_is_listed_read_only_and_open_world():
    docs_tool = tool()
    schema = docs_tool["inputSchema"]
    assert set(schema["properties"]) == {"query", "limit", "slug", "page"}
    assert "required" not in schema or schema["required"] == []
    assert schema["properties"]["limit"]["maximum"] == mcp_docs.MAX_RESULTS
    assert docs_tool["annotations"]["readOnlyHint"] is True
    assert docs_tool["annotations"]["openWorldHint"] is True
    for words in ("docs.omics-os.com", "not instructions", "slug", "page"):
        assert words in docs_tool["description"], words
    assert [t["name"] for t in mcp_contract.list_tools()][-1] == "lobster_docs"


def test_search_passes_the_query_and_the_notice(docs):
    found = call(
        McpServer(ThreadRunner()), "lobster_docs", {"query": "gwas", "limit": 3}
    )
    assert found == {"results": [HIT], "notice": mcp_docs.NOTICE}
    assert docs == [("search", "gwas", 3)]


def test_read_passes_the_slug_and_page(docs):
    page = call(
        McpServer(ThreadRunner()),
        "lobster_docs",
        {"slug": "agents/genomics", "page": 2},
    )
    assert page == PAGE
    assert docs == [("read", "agents/genomics", 2)]


@pytest.mark.parametrize(
    "args",
    [{}, {"query": "a", "slug": "b"}, {"query": ""}, {"slug": " "}],
)
def test_exactly_one_of_query_or_slug(docs, args):
    refused = call(McpServer(ThreadRunner()), "lobster_docs", args)["error_text"]
    assert refused.startswith("exactly_one_of: ")
    assert docs == []


@pytest.mark.parametrize(
    "args",
    [
        {"query": "a", "limit": 0},
        {"query": "a", "limit": True},
        {"query": "a", "limit": mcp_docs.MAX_RESULTS + 1},
        {"query": "x" * 201},
        {"slug": "a", "page": 0},
        {"slug": "a", "page": "2"},
        {"query": "a", "page": 2},
        {"slug": "a", "limit": 2},
    ],
)
def test_bad_bounds_are_refused(docs, args):
    refused = call(McpServer(ThreadRunner()), "lobster_docs", args)["error_text"]
    assert refused.startswith("bad_argument: ")
    assert docs == []


def test_a_docs_error_is_one_line(monkeypatch):
    def fail(query, limit=5):
        raise mcp_docs.DocsError("docs.omics-os.com could not be reached; no copy.")

    monkeypatch.setattr(mcp_docs, "search", fail)
    refused = call(McpServer(ThreadRunner()), "lobster_docs", {"query": "x"})
    assert refused["error_text"] == (
        "docs_unavailable: docs.omics-os.com could not be reached; no copy."
    )


def test_a_page_past_the_end_is_a_bad_argument(monkeypatch):
    def past(slug, page=1):
        raise ValueError("page must be between 1 and 3")

    monkeypatch.setattr(mcp_docs, "read", past)
    refused = call(McpServer(ThreadRunner()), "lobster_docs", {"slug": "a", "page": 9})
    assert refused["error_text"] == "bad_argument: page must be between 1 and 3"


def test_a_slow_docs_call_holds_no_other_reply(monkeypatch):
    entered, release = threading.Event(), threading.Event()

    def slow(query, limit=5):
        entered.set()
        release.wait(10)
        return {"results": [], "notice": mcp_docs.NOTICE}

    monkeypatch.setattr(mcp_docs, "search", slow)
    sent = []
    server = McpServer(ThreadRunner(), send=sent.append)
    line = {
        "jsonrpc": "2.0",
        "id": 7,
        "method": "tools/call",
        "params": {"name": "lobster_docs", "arguments": {"query": "x"}},
    }
    assert server.handle_line(json.dumps(line)) is None
    assert entered.wait(5)
    ping = server.handle_line(json.dumps({"jsonrpc": "2.0", "id": 8, "method": "ping"}))
    assert ping["result"] == {}
    release.set()
    for _ in range(200):
        if sent:
            break
        threading.Event().wait(0.025)
    assert sent[0]["id"] == 7
