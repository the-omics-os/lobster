"""Both ways a session starts write the format marker, and a continued one keeps its origin.

A session of the command line (``AgentClient``) and one of the MCP run path
(``mcp_stdio._build_client`` and the restore ``_restore_session`` does) reach the same
provenance tracker, so the marker comes from one place.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from lobster.cli_internal import mcp_stdio
from lobster.cli_internal.commands.heavy import session_infra
from lobster.core.client import AgentClient

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _no_graph():
    with patch(
        "lobster.core.client.create_bioinformatics_graph",
        return_value=(MagicMock(), {}),
    ):
        yield


def _folder(workspace, client):
    return workspace / ".lobster" / "sessions" / client.session_id


def _step(client):
    client.data_manager.log_tool_usage("probe", {}, "a step")


def _marker(workspace, client):
    path = _folder(workspace, client) / "metadata.json"
    return json.loads(path.read_text()).get("record_format")


def _mcp_client(workspace, **kwargs):
    """The client the MCP run path builds, with the engine's startup stubbed out."""

    def build(workspace=None, interactive=True, **more):
        return AgentClient(workspace_path=workspace, **more)

    with patch.object(session_infra, "init_client_or_raise_startup_diagnostic", build):
        return mcp_stdio._build_client(workspace, None, None, **kwargs)


def _continue_like_mcp(workspace):
    """What ``_restore_session`` does: the latest session, opened by its id."""
    file, session_id, found = session_infra.resolve_session_continuation(
        workspace, "latest"
    )
    assert found
    client = _mcp_client(workspace, session_id=session_id)
    if file is not None:
        assert client.load_session(file).get("success")
    return client


def test_a_command_line_session_writes_the_marker(tmp_path):
    client = AgentClient(workspace_path=tmp_path)
    _step(client)
    assert _marker(tmp_path, client)
    assert client.data_manager.provenance.migrated_from() is None


def test_an_mcp_session_writes_the_marker(tmp_path):
    client = _mcp_client(tmp_path)
    _step(client)
    assert _marker(tmp_path, client)
    assert client.data_manager.provenance.migrated_from() is None


def test_an_mcp_continue_of_a_native_session_stays_native(tmp_path):
    first = _mcp_client(tmp_path)
    _step(first)
    again = _continue_like_mcp(tmp_path)
    _step(again)
    assert again.session_id == first.session_id
    assert again.data_manager.provenance.migrated_from() is None
    assert _marker(tmp_path, again)


def test_a_command_line_resume_of_a_native_session_stays_native(tmp_path):
    first = AgentClient(workspace_path=tmp_path)
    _step(first)
    again = AgentClient(workspace_path=tmp_path, session_id=first.session_id)
    _step(again)
    assert again.data_manager.provenance.migrated_from() is None
    assert _marker(tmp_path, again)


def test_an_mcp_continue_of_an_earlier_session_stays_migrated(tmp_path):
    first = _mcp_client(tmp_path)
    _step(first)
    meta = _folder(tmp_path, first) / "metadata.json"
    written = json.loads(meta.read_text())
    del written["record_format"]
    meta.write_text(json.dumps(written))
    again = _continue_like_mcp(tmp_path)
    _step(again)
    assert again.data_manager.provenance.migrated_from() is not None
    assert _marker(tmp_path, again) is None
