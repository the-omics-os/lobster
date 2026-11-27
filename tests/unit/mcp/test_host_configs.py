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

Read on 2026-11-27:

- pi ``mcp.json`` (pi-mcp-adapter 2.33.0 README, "Server Options"):
  ``mcpServers.<name>`` with ``command``, ``args``, ``directTools``,
  ``toolPrefix`` (``"none"``), ``requestTimeoutMs``. The os native MCP preview
  reads the same entry, takes ``timeout`` in seconds and ignores the adapter's
  keys (os ``docs/extensions.md`` "Native MCP runtime reference";
  ``src/core/mcp-servers.ts`` ``validateMcpServerConfig``).
- opencode ``opencode.json`` (opencode.ai/docs/mcp-servers): V2
  ``mcp.servers.<name>`` with ``type: "local"``, ``command`` as one array,
  ``codemode: false`` for the provider's native tool list; tool calls time out
  after 12 hours by default.
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


def test_codex_allows_a_whole_run_per_tool_call():
    server = tomllib.loads(
        (INTEGRATIONS / "codex.config.toml").read_text(encoding="utf-8")
    )["mcp_servers"][mcp_contract.SERVER_NAME]
    assert server["tool_timeout_sec"] == 3600


def test_pi_snippet():
    raw = (INTEGRATIONS / "pi.mcp.json").read_text(encoding="utf-8")
    assert json.loads(raw) == {
        "mcpServers": {
            mcp_contract.SERVER_NAME: {
                **LAUNCH,
                "directTools": True,  # the tools by name, not the mcp proxy
                "toolPrefix": "none",  # lobster_run, not lobster_lobster_run
                "requestTimeoutMs": 3_600_000,
                "timeout": 3600,  # os native preview, in seconds
            }
        }
    }
    _no_secrets_or_user_paths(raw)


def test_opencode_snippet():
    raw = (INTEGRATIONS / "opencode.json").read_text(encoding="utf-8")
    assert json.loads(raw) == {
        "$schema": "https://opencode.ai/config.json",
        "mcp": {
            "servers": {
                mcp_contract.SERVER_NAME: {
                    "type": "local",
                    "command": [LAUNCH["command"], *LAUNCH["args"]],
                    "codemode": False,
                }
            }
        },
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
