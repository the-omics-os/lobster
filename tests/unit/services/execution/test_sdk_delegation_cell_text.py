"""The delegated task and its result stay comment lines in the step's cell.

Both texts went into the template with an f-string: a line break ended the
comment, so the rest of the line ran with the notebook, and ``{{ ... }}`` in them
was template source that Lobster evaluated when it wrote the cell.
"""

import ast
from types import SimpleNamespace

import pytest

from lobster.services.execution import sdk_delegation_service as sdk

pytestmark = pytest.mark.unit

BREAKS = ["\n", "\r", "\r\n", "\x0b", "\x0c", "\x1c", "\x1d", "\x1e", "\x85"]
BREAKS += ["\u2028", "\u2029"]


@pytest.fixture
def service(tmp_path, monkeypatch):
    monkeypatch.setattr(sdk, "SDK_AVAILABLE", True)
    manager = SimpleNamespace(workspace_path=tmp_path, list_modalities=lambda: [])
    return sdk.SDKDelegationService(manager)


@pytest.mark.parametrize("brk", BREAKS, ids=[repr(b) for b in BREAKS])
@pytest.mark.parametrize("where", ["task", "result"])
def test_task_and_result_text_never_becomes_code(service, where, brk):
    text = f"look{brk}import os; os.system('x'){brk}{{{{ 7*7 }}}}{brk}end"
    texts = {"task": "a task", "result": "a result", where: text}

    ir = service._create_ir(texts["task"], "d", texts["result"], True, {})
    cell = ir.render()

    assert ast.parse(cell).body == [], cell
    assert "49" not in cell
    assert "{{ 7*7 }}" in cell
    assert all(line.startswith("#") for line in cell.splitlines()), cell
