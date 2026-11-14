"""MCP host config snippets in ``lobster/integrations/`` parse and launch the server.

Shapes read on 2026-11-11 from each host's documentation:

- Claude Code ``.mcp.json``: ``mcpServers.<name>`` with ``type``, ``command``,
  ``args`` (code.claude.com/docs/en/mcp.md).
- Claude Desktop ``claude_desktop_config.json``: ``mcpServers.<name>`` with
  ``command``, ``args`` (modelcontextprotocol.io connect-local-servers).
- Codex ``config.toml``: openai/codex @ rust-v0.167.1
  ``codex-rs/config/src/mcp_types.rs:391-458`` (``RawMcpServerConfig``, unknown
  keys rejected), ``:84-92`` (per-tool ``approval_mode``), ``:29-35``
  (``AppToolApproval``: auto, prompt, writes, approve).
- Cursor ``mcp.json``: ``mcpServers.<name>``; install deeplink carries the
  base64 of the bare server object (cursor.com/docs/mcp/install-links).
"""

from __future__ import annotations

import base64
import json
import re
import tomllib
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

from lobster.cli_internal import mcp_contract
from lobster.ui.bridge.init_adapter import _PROVIDER_ENV_KEYS

INTEGRATIONS = Path(__file__).resolve().parents[3] / "lobster" / "integrations"
LAUNCH = {"command": "lobster", "args": ["mcp", "--stdio"]}
JSON_SNIPPETS = {
    "claude-code.mcp.json": {"type": "stdio", **LAUNCH},
    "claude-desktop.config.json": LAUNCH,
    "cursor.mcp.json": {"type": "stdio", **LAUNCH},
}
CODEX_KEYS = {
    "command",
    "args",
    "env",
    "env_vars",
    "cwd",
    "startup_timeout_sec",
    "tool_timeout_sec",
    "enabled",
    "required",
    "enabled_tools",
    "disabled_tools",
    "default_tools_approval_mode",
    "tools",
}


def _no_secrets_or_user_paths(raw: str) -> None:
    for marker in ("sk-", "AKIA", "Bearer ", "/Users/", "/home/", "C:\\"):
        assert marker not in raw


@pytest.mark.parametrize("name, server", JSON_SNIPPETS.items())
def test_json_snippet(name, server):
    raw = (INTEGRATIONS / name).read_text(encoding="utf-8")
    doc = json.loads(raw)
    assert doc == {"mcpServers": {mcp_contract.SERVER_NAME: server}}
    _no_secrets_or_user_paths(raw)


def test_codex_snippet():
    raw = (INTEGRATIONS / "codex.config.toml").read_text(encoding="utf-8")
    servers = tomllib.loads(raw)["mcp_servers"]
    assert set(servers) == {mcp_contract.SERVER_NAME}
    server = servers[mcp_contract.SERVER_NAME]
    assert set(server) <= CODEX_KEYS
    assert {k: server[k] for k in LAUNCH} == LAUNCH
    assert "env" not in server  # keys come from the user's environment
    # By design: only the read-only tool is pre-approved.
    assert server["tools"] == {
        "lobster_overview": {"approval_mode": "approve"},
        "lobster_status": {"approval_mode": "approve"},
        "lobster_wait": {"approval_mode": "approve"},
    }
    _no_secrets_or_user_paths(raw)


def test_codex_env_vars_match_the_engine_provider_keys():
    """Codex forwards only allowlisted variables; the list must follow the keys
    ``lobster init`` writes (``lobster/ui/bridge/init_adapter.py:40``)."""
    server = tomllib.loads(
        (INTEGRATIONS / "codex.config.toml").read_text(encoding="utf-8")
    )["mcp_servers"][mcp_contract.SERVER_NAME]
    engine = [key for keys in _PROVIDER_ENV_KEYS.values() for key in keys]
    assert server["env_vars"] == engine
    assert all(re.fullmatch(r"[A-Z][A-Z0-9_]*", key) for key in engine)


def test_cursor_deeplink_round_trips():
    server = json.loads((INTEGRATIONS / "cursor.mcp.json").read_text())["mcpServers"][
        mcp_contract.SERVER_NAME
    ]
    config = base64.b64encode(json.dumps(server).encode()).decode()
    link = (
        "cursor://anysphere.cursor-deeplink/mcp/install"
        f"?name={mcp_contract.SERVER_NAME}&config={config}"
    )
    query = parse_qs(urlparse(link).query)
    assert query["name"] == [mcp_contract.SERVER_NAME]
    assert json.loads(base64.b64decode(query["config"][0])) == server


def test_the_readme_names_every_tool_the_server_lists():
    from lobster.cli_internal import mcp_contract

    readme = (INTEGRATIONS / "README.md").read_text()
    tools = [t["name"] for t in mcp_contract.list_tools()]
    assert (
        f"lists {['zero','one','two','three','four','five','six','seven','eight','nine','ten'][len(tools)]} tools"
        in readme
    )
    for name in tools:
        assert f"`{name}`" in readme
