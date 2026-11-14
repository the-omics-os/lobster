"""The Open Science connector file passes the import rules of their parser.

Rules read from the Open Science clone at commit 8a50f6a (2026-11-11):

- src/main/settings/connector-template.ts:44-56 ROOT_FIELDS (allowed keys),
  enforced by rejectUnknownFields at :219 and :781; :57 SECRET_FIELDS.
- connector-template.ts:782-799 schema_version must be 1, kind must be
  "open-science.connector".
- connector-template.ts:802-821 name and display_name required, trimmed,
  limits from src/main/settings/connector-resource-limits.ts:9-14 (name 64,
  display_name 128, description 2000); name pattern ^[a-z0-9-]+$ from
  src/shared/custom-connector.ts:1-2,11-12.
- connector-template.ts:822-833, 841-849 transport stdio needs command;
  :860-866 stdio cannot carry url or oauth.
- connector-template.ts:941-949 name must not be a built-in connector id.
- The os-/mcp- prefixes are reserved for resource ids
  (src/shared/resource-id.ts:4); the connector skill is named mcp-<name>
  (src/shared/custom-connector.ts:14).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from lobster.cli_internal import mcp_contract

CONNECTOR_PATH = (
    Path(__file__).resolve().parents[3]
    / "lobster/integrations/open-science.connector.json"
)
ROOT_FIELDS = {
    "schema_version",
    "kind",
    "name",
    "display_name",
    "description",
    "transport",
    "command",
    "args",
    "url",
    "required_secrets",
    "oauth",
}


def _raw() -> str:
    return CONNECTOR_PATH.read_text(encoding="utf-8")


def _doc() -> dict:
    return json.loads(_raw())


def test_header():
    doc = _doc()
    assert doc["schema_version"] == 1
    assert doc["kind"] == "open-science.connector"


def test_name_is_importable():
    name = _doc()["name"]
    assert re.fullmatch(r"[a-z0-9-]+", name)
    assert len(name) <= 64
    assert not name.startswith(("os-", "mcp-"))
    # Checked against packages/connector-builtins/src/catalog.ts at 8a50f6a on
    # 2026-11-11: "lobster" is not a built-in id.
    assert name == mcp_contract.SERVER_NAME == "lobster"


def test_display_name():
    display_name = _doc()["display_name"]
    assert display_name.strip()
    assert len(display_name) <= 128


def test_description_is_the_contract_constant():
    description = _doc()["description"]
    assert description == mcp_contract.CONNECTOR_DESCRIPTION
    assert len(description) <= 2000


def test_stdio_launch():
    doc = _doc()
    assert doc["transport"] == "stdio"
    assert doc["command"] == "lobster"
    assert doc["args"] == ["mcp", "--stdio"]


def test_only_parser_keys_and_no_secrets():
    doc = _doc()
    assert set(doc) <= ROOT_FIELDS
    for key in ("env", "environment", "headers", "url", "required_secrets", "oauth"):
        assert key not in doc
    raw = _raw()
    for marker in ("sk-", "AKIA", "Bearer "):
        assert marker not in raw
